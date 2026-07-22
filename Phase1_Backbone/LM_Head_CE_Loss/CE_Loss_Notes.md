# LM Head 与交叉熵损失 -- 理论与工程笔记

> 关联文档：[README.md](README.md) -- 本目录的入门索引（文件清单 + 运行指令）<br>
> 关联学术笔记：[CE_Loss_Academic_Notes.md](CE_Loss_Academic_Notes.md) -- 从 Shannon 公理到 Proper Scoring Rule 的严谨数学推导<br>
> 关联代码：[lm_head.py](lm_head.py) -- LM Head + LogSumExp CE Loss 的完整 PyTorch 实现

---

## 目录

- [Part I -- CE 理论基础](#part-i--ce-理论基础)
  - [1.1 为什么不能用 MSE？](#11-为什么不能用-mse)
  - [1.2 LogSumExp 推导：从朴素实现到数值稳定](#12-logsumexp-推导从朴素实现到数值稳定)
  - [1.3 连乘概率 vs 对数似然：为什么取 log 不是可选的](#13-连乘概率-vs-对数似然为什么取-log-不是可选的)
  - [1.4 LogSumExp vs 标准 CE：形式对比](#14-logsumexp-vs-标准-ce形式对比)
- [Part II -- 浮点精度与显存](#part-ii--浮点精度与显存)
  - [2.1 BF16 的吞噬效应与大数吃小数](#21-bf16-的吞噬效应与大数吃小数)
  - [2.2 torch.gather() 的完整使用指南](#22-torchgather-的完整使用指南)
  - [2.3 Logits 张量的工程意义：[B, S, V] 的显存冲击](#23-logits-张量的工程意义b-s-v-的显存冲击)
  - [2.4 为什么 Softmax 前必须转 FP32](#24-为什么-softmax-前必须转-fp32)
- [Part III -- LM Head 架构与工程边界](#part-iii--lm-head-架构与工程边界)
  - [3.1 训练模式 vs 推理模式：双栖 API 设计](#31-训练模式-vs-推理模式双栖-api-设计)
  - [3.2 Weight Tying：共享 Embedding 与 LM Head 权重](#32-weight-tying共享-embedding-与-lm-head-权重)
  - [3.3 LogSumExp vs Safe Softmax：为什么有了 Safe Softmax 还需要 LogSumExp](#33-logsumexp-vs-safe-softmax为什么有了-safe-softmax-还需要-logsumexp)
  - [3.4 $e^{-100}$ 下溢问题：为什么它不是灾难](#34-e-100-下溢问题为什么它不是灾难)
  - [3.5 `ignore_index=-100` 的角色：Padding 掩码与 SFT 领域适配](#35-ignore_index-100-的角色padding-掩码与-sft-领域适配)
  - [3.6 DDP 死锁：当所有标签都是 -100](#36-ddp-死锁当所有标签都是--100)
- [Part IV -- 面试 Q&A](#part-iv--面试-qa)
- [本目录文件](#本目录文件)

---

## Part I -- CE 理论基础

### 1.1 为什么不能用 MSE？

> 注意：本小节聚焦于工程直觉和面试导向的解释。严格的数学推导（包括 MSE 在总体水平上也是 Proper Scoring Rule 的完整证明，以及 MSE + Softmax 梯度消失的结构性分析）请参见 [CE_Loss_Academic_Notes.md §5.4](CE_Loss_Academic_Notes.md#54-对比mse--softmax-为什么失败)。

**核心直觉：交叉熵把"概率差距"放大了。**

假设模型给正确 token 的概率是 $0.01$（即模型几乎确信它是错的），正确 token 的 one-hot 标签是 $1.0$：

$$L_{\text{CE}} = -\ln(0.01) = 4.605$$
$$L_{\text{MSE}} = (0.01 - 1.0)^2 = 0.9801$$

CE 给的损失是 MSE 的 **4.7 倍**。在反向传播中，这个差异会被进一步放大：

**CE + Softmax 的梯度**（完整推导见 [CE_Loss_Academic_Notes.md §5.2](CE_Loss_Academic_Notes.md#52-cross-entropy--softmax-合体求导)）：

$$\frac{\partial L_{\text{CE}}}{\partial z_i} = a_i - y_i$$

- 如果模型错得离谱（$a_c \approx 0,\ y_c = 1$），梯度 $\approx -1$（最大拉力，强力修正）
- 如果模型已经正确（$a_c \approx 1,\ y_c = 1$），梯度 $\approx 0$（不再调整）

这个公式的惊人之处在于**梯度是线性的** -- 不会因为 Softmax 的饱和而消失。

**MSE + Softmax 的梯度**（详细推导见 [CE_Loss_Academic_Notes.md §5.4](CE_Loss_Academic_Notes.md#54-对比mse--softmax-为什么失败)）：

$$\frac{\partial L_{\text{MSE}}}{\partial z_i} = (a_i - y_i)a_i(1-a_i) - a_i \sum_{j \neq i}(a_j - y_j)a_j$$

- 当模型错得离谱时（$a_c \approx 0$），梯度 $\approx 0$（梯度消失！）
- 当模型过于自信时（$a_c \approx 1$），梯度 $\approx 0$（同样消失！）

**一句话总结：MSE 在模型最需要学习的时候反而停止了学习。CE 在最需要学习的时候给出最强梯度。**

深层原因是 CE 的对数求导项 $1/a_j$ 恰好与 Softmax Jacobian 中的 $a_j$ 因子消去，产生线性梯度。MSE 没有这种"反补偿"机制，其梯度被 Softmax 输出饱和区的收缩因子"污染"了。

**对比表格：**

| 维度 | 交叉熵 (CE) | 均方误差 (MSE) |
|:-----|:-----------|:-------------|
| CE 梯度公式 | $a_i - y_i$（线性，永不消失） | 含 $a_i(1-a_i)$（饱和区消失） |
| 模型错误时的梯度 | $\approx -1$（强力修正） | $\approx 0$（停止学习） |
| 模型正确时的梯度 | $\approx 0$（保持稳定） | $\approx 0$（保持稳定） |
| 隐含假设 | 输出空间具有竞争性（词表互斥） | 输出空间独立（各维度独立） |
| 语言建模适用性 | **匹配**：词表维度互相竞争 | **不匹配**：词表中的词彼此竞争 |
| 数值范围 | 需要 LogSumExp 防溢出 | 无需特殊处理 |

### 1.2 LogSumExp 推导：从朴素实现到数值稳定

**问题**：直接计算 $L = -\ln\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right)$ 会发生什么？

如果 $z_c = 100$，则 $e^{100} \approx 2.688 \times 10^{43}$，远超 FP32 上限（$\approx 3.4 \times 10^{38}$），直接 **上溢为 inf**。

**LogSumExp 技巧的核心思路**：利用对数的性质将除法变成减法，再利用指数平移不变性消除溢出。

**Step 1 -- 对数恒等式**：

$$L = -\ln\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right) = -z_c + \ln\sum_j e^{z_j}$$

这步消除了分母中的指数项可能下溢造成的 $\ln(0) = -\infty$ 风险。

**Step 2 -- 指数平移不变性**：

设 $M = \max_j z_j$，则：

$$\ln\sum_j e^{z_j} = \ln\left(e^M \sum_j e^{z_j - M}\right) = M + \ln\sum_j e^{z_j - M}$$

**关键**：$z_j - M \leq 0$（对所有 $j$），因此 $e^{z_j - M} \in (0, 1]$。最大的指数项恰好为 $e^0 = 1$。这保证了：
- **不会上溢**：所有指数项 $\leq 1$
- **不会 $\ln(0)$**：求和至少包含 $e^0 = 1$，故 $\ln(\text{求和}) \geq 0$

**Step 3 -- 合并**：

$$\boxed{L = -(z_c - M) + \ln\sum_j e^{z_j - M}}$$

**代码中的对应**（来自 `lm_head.py`）：

```python
# 手写 LogSumExp CE Loss 的核心五步
max_logits, _ = torch.max(valid_logits_fp32, dim=-1, keepdim=True)   # [M, 1]
safe_logits = valid_logits_fp32 - max_logits                          # [M, V]
true_safe_logits = safe_logits.gather(dim=-1, index=...).squeeze(-1)  # [M]
exp_logits = torch.exp(safe_logits)                                    # [M, V]
log_sum_exp = torch.log(torch.sum(exp_logits, dim=-1))                # [M]
loss = (-true_safe_logits + log_sum_exp).mean()
```

这段代码等价于 PyTorch 的 `F.cross_entropy(logits, labels)`（内部实现同样是 LogSumExp）。

**为什么 `-(z_c - M)` 等价于 `-z_c + M`？**

因为 LogSumExp 的公式展开后 $M$ 在两项中符号相反，恰好抵消手加的一次 $M$。直接在 `safe_logits`（已减 $M$）上做 `gather` 取出 $z_c - M$，然后置负号，自然包含了 $M$ 的贡献，无需额外处理。

### 1.3 连乘概率 vs 对数似然：为什么取 log 不是可选的

在极大似然估计中，似然函数是每个样本的模型概率的连乘：

$$L(\theta; D) = \prod_{i=1}^N P_\theta(x^{(i)})$$

对于语言模型，$N$ 通常是数十亿量级。每个 $P_\theta(x^{(i)})$ 都在 $(0, 1)$ 之间。连乘 10 亿个 $(0, 1)$ 之间的数会**极快下溢为 0**。例如 $0.5^{10^9}$ 在浮点系统中等于绝对 0。

取对数是唯一的数值稳定解：

$$\ln L(\theta; D) = \sum_{i=1}^N \ln P_\theta(x^{(i)})$$

三个好处：

1. **数值稳定性**：把乘法变成加法，消除连乘下溢
2. **凸性保持**：对数似然常为凸函数，利于优化
3. **信息论桥梁**：$\frac{1}{N} \sum \ln P_\theta(x^{(i)})$ 收敛到 $\mathbb{E}_{x \sim P_{\text{data}}}[\ln P_\theta(x)]$（大数定律），恰好是交叉熵的负值

**一句话**：$-\ln$ 不是"顺便取了个对数" -- 它既拯救了数值稳定性，又连接了统计估计和信息论。

### 1.4 LogSumExp vs 标准 CE：形式对比

| 维度 | 标准交叉熵公式 | LogSumExp 公式 |
|:-----|:-------------|:-------------|
| 表达式 | $-\ln\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right)$ | $-(z_c - M) + \ln\sum_j e^{z_j - M}$ |
| 上溢防护 | 无 -- 大 logit 直接 inf | 有 -- $M$ 移位保证指数 $\leq 1$ |
| 下溢防护 | 无 -- 小 logit 可能造成 $\ln(0)$ | 有 -- 求和中至少含 $e^0 = 1$ |
| 数学等价性 | 精确相等 | 精确相等（恒等变换） |
| 计算步骤 | 2 步（exp + log+除法） | 5 步（max + subtract + exp + sum + log） |
| 实际使用 | 只在教科书和低风险场景 | 所有工业级 LLM 训练 |

---

## Part II -- 浮点精度与显存

### 2.1 BF16 的吞噬效应与大数吃小数

BF16（Brain Float 16）是 LLM 训练的默认精度格式。它的位分配是：1 位符号 + 8 位指数 + **7 位尾数**。

与 FP16（1 + 5 + 10 = 16 bits）相比，BF16 牺牲了尾数精度（7 vs 10），换取了与 FP32 相同的指数范围（8 位指数）。这意味着：

- **BF16 不会轻易溢出**（指数范围与 FP32 相同，max $\approx 3.4 \times 10^{38}$）
- **但 BF16 的精度很低**（7 位尾数 $\approx 2$ 位十进制有效数字）

**大数吃小数（Swamping Effect）**：BF16 的关键问题。

BF16 在数值 $1.0$ 附近的机器精度（machine epsilon）为：

$$\epsilon_{\text{BF16}} = 2^{-7} = 0.0078125$$

这意味着在 BF16 下：
- $1.0 + 0.007 = 1.0$（$0.007$ 被吞噬）
- $1.0 + 0.008 = 1.0078125$（勉强存活）

对于一个 128,000 词的词表，LogSumExp 的求和项是：

$$S = e^0 + \sum_{j: z_j < M} e^{z_j - M}$$

最大项 $e^0 = 1$ 与大量小项累加。当小项的指数小于 $e^{-4.85}$（即 $z_j - M < -4.85$）时，在 BF16 下累加到 $1.0$ 上会被完全截断为 0。

**量化估算**：假设有 99% 的词（约 126,720 个）低于此阈值，它们的平均 $e^{z_j - M} \approx e^{-8} \approx 0.000335$。数学上它们合计贡献约 $126720 \times 0.000335 \approx 42.5$，但在 BF16 下这 42.5 **全部消失** -- LogSumExp 的求和值从正确的 $\ln(1 + 42.5 + \dots) \approx 3.77$ 偏小到 $\ln(1) = 0$。损失函数完全错误。

**工业级解决方案**：LogSumExp 的求和阶段**必须在 FP32 下进行**。这也是为什么 `lm_head.py` 中有：

```python
valid_logits_fp32 = valid_logits.float()  # 强制提升到 FP32
```

**FP32 安全吗？** FP32 的 $\epsilon_{\text{FP32}} \approx 1.19 \times 10^{-7}$，吞噬阈值约为 $\ln(10^{-7}) \approx -16.1$。这意味着只有 logit 低于最大值 16 nats 以上的词才会被吞噬 -- 在实际的深层 Transformer 中，logits 的范围通常不会达到如此极端，FP32 是安全的。

**BF16 vs FP16 对比表：**

| 格式 | 符号位 | 指数位 | 尾数位 | 指数范围 | $\epsilon$ | 溢出风险 | 精度风险 |
|:-----|:------|:------|:------|:--------|:----------|:--------|:--------|
| FP32 | 1 | 8 | 23 | $\approx 10^{\pm 38}$ | $1.19 \times 10^{-7}$ | 低 | 低 |
| BF16 | 1 | 8 | 7 | $\approx 10^{\pm 38}$ | $7.81 \times 10^{-3}$ | 低 | **高** |
| FP16 | 1 | 5 | 10 | $\approx 10^{\pm 4.8}$ | $9.77 \times 10^{-4}$ | **高** | 中 |

**一句话：BF16 不会爆炸，但会"睁眼瞎"（看不到小量）；FP16 能看到小量，但容易爆炸。LLM 训练选择 BF16，然后在跨量级求和时手动提升到 FP32。**

### 2.2 torch.gather() 的完整使用指南

`torch.gather()` 是处理 CE Loss 时绕不开的核心操作 -- 用于从 logits 张量中提取目标 token 对应的 logit 值。

**函数签名**：

```python
torch.gather(input, dim, index)
```

**核心功能**：沿着 `dim` 维度，根据 `index` 中指定的索引值，从 `input` 中"收集"对应的元素。

**关键规则**：`index` 的形状必须与 `input` 的形状**除 `dim` 维度外完全一致**。

**在 CE Loss 中的典型用法**：

```python
# 场景：从 [M, V] 的 logits 矩阵中，提取 M 个目标 token 对应的 logit 值
# valid_logits: [M, V]  -- M 个 token 的 logits，每个是 V 维向量
# valid_labels: [M]     -- M 个目标 token 的 ID（范围 [0, V-1]）

# Step 1：将 labels 从 [M] 扩张为 [M, 1]
index = valid_labels.unsqueeze(-1)  # [M] -> [M, 1]

# Step 2：gather -- 沿 dim=-1（词汇维度）收集
gathered = valid_logits.gather(dim=-1, index=index)  # [M, 1]

# Step 3：压平回 [M]
result = gathered.squeeze(-1)  # [M]
```

**为什么必须 unsqueeze？**

`torch.gather` 要求 `index.shape` 与 `input.shape` 沿非 gather 维度匹配。如果 `valid_logits` 的形状是 `[M, V]` 而 `index` 是 `[M]`，PyTorch 无法确定如何在 `[V]` 维度上广播索引。`unsqueeze(-1)` 将 `[M]` 变成 `[M, 1]`，此时：
- 第 0 维（`M`）：与 input 的第 0 维匹配
- 第 1 维（`1`，gather 维度）：在 gather 后会被消除

**gather 的数学含义**：

```python
# gathered[i, 0] == valid_logits[i, index[i, 0]]
# 即：沿 dim=-1，第 i 行取第 index[i,0] 个元素
```

**常见的三处 `gather` 误区**：

1. **忘了 unsqueeze**：直接传 `[M]` 给 `[M, V]` 的 gather → 形状不匹配报错
2. **gather 后忘了 squeeze**：得到 `[M, 1]` 而不是期望的 `[M]` → 广播错误
3. **维度参数填错**：`dim=0` 是沿行收集，`dim=1` 是沿列收集。从 logits 取 token 应沿词表维度 `dim=-1`

**为什么在 `safe_logits` 上 gather 而非在原始 logits 上 gather？**

因为在 LogSumExp 中我们需要的是 $-(z_c - M)$ 而非 $-z_c$。在 `safe_logits = logits - M` 上 gather 自然得到 $z_c - M$，然后加负号，正确且省去手动处理 $M$ 的麻烦。

### 2.3 Logits 张量的工程意义：[B, S, V] 的显存冲击

在 Transformer 训练中，logits 张量是**最大的中间激活**。它的形状是 `[B, S, V]`。

**显存估算**：

假设一个典型的训练配置：
- Batch size $B = 8$
- 序列长度 $S = 4096$
- 词表大小 $V = 128{,}000$（如 LLaMA-3）
- 数据类型 FP32（4 bytes）

单个 logits 张量的显存：

$$8 \times 4096 \times 128{,}000 \times 4\text{ bytes} \approx 16.8\text{ GB}$$

**这是全精度 FP32 下的数字。** 在实际 BF16 训练中（2 bytes）：$\approx 8.4\text{ GB}$。

**但这仍然是单个张量的占用，而且它必须存在于显存中** -- 因为 Softmax / CE 操作需要完整的 $[V]$ 维度的概率分布。

**为什么 logits 这么大？** 因为 $V = 128{,}000$ 这个维度是"必须完整"的 -- 不能像 Batch/Seq 维度那样分片。Softmax 的归一化需要对**所有** $V$ 个 logits 求和，$\log$ 操作也需要完整的 $V$ 维向量。

**工程权衡**：

- **训练时**：logits 被写入 HBM（显存），因为它需要用于 CE Loss 计算和可能的 Top-K / beam search 辅助操作
- **推理时**：logits 只用最后一个位置（$S = 1$），尺寸为 $[B, 1, V] \approx 1\text{ MB}$（BF16），不是瓶颈

**为什么 LogSumExp 能减少显存？**

标准 Safe Softmax + Log 的两步方法：

```python
probs = safe_softmax(logits)       # Step 1：产生 [M, V] 的 FP32 概率矩阵写入 HBM
loss = -torch.log(probs[target])   # Step 2：从 HBM 读取取 log
```

中间概率张量 `probs` 的显存 = $M \times V \times 4$ bytes。对于 100 万个 token（$M \approx 10^6$），$V = 128{,}000$，这就是 $10^6 \times 128000 \times 4 \approx 512\text{ GB}$——**完全不现实**。

LogSumExp 融合了这两步：

```python
loss = -(z_c - M) + log(sum(exp(z - M)))
```

不需要任何中间 `[M, V]` 张量写入 HBM。exp、sum、log 在寄存器或 shared memory 中完成（取决于是否使用 Triton 融合 Kernel），显存占用仅 $O(M)$ 即约 4 MB。

**本质上是 Online Softmax / FlashAttention 的同构思路：数学等价，但内存访问模式根本不同。**

### 2.4 为什么 Softmax 前必须转 FP32

FP16/BF16 在 0 附近的最小正数（最小正规数）约为 $6 \times 10^{-8}$（FP16）或 $1.2 \times 10^{-38}$（BF16，可以与 FP32 相同因为共享 8 位指数）。但是在跨量级求和（如 Softmax 的分母 $\sum_j e^{z_j}$）中，BF16 的精度问题不在"下溢为 0"，而在**累加时的舍入误差**。

具体来说：

- Softmax 分母涉及对 $V$ 个 $e^{z_j - M} \in (0, 1]$ 的求和
- 这些值中有一个 $1.0$（最大项），其余几千到几十万个值在 $(0, 1]$ 之间
- BF16 只有 7 位尾数，$1.0$ + 小数的有效累加精度约为 $2^{-7} \approx 0.0078$
- 任何小于 0.0078 的 $e^{z_j - M}$ 在 BF16 累加中**不起作用** -- 分母偏小，Softmax 输出偏大（对所有 token 一致偏大，不是均匀膨胀）

这种系统性偏小导致 LogSoftmax 值偏大（因为分母偏小），最终 CE Loss 偏小。对单个 token，误差可能只有 $10^{-3}$ 量级，但在几十亿 token 平均后，系统误差会放大。更重要的是，它引入了无法预测的噪声，削弱了训练的稳定性。

**强制转 FP32 的代价很小**：logits 本身是 FP32（Python float 的默认类型）或 BF16（GPU 计算的结果），`tensor.float()` 只是改变了解释方式（对于 BF16 -> FP32 需要额外的显存，但 $M \times V$ 的 logits 本身已占用大量显存，FP32 版本的额外开销在此语境下是可接受的）。实际操作中更多使用**混合精度**的 `autocast` 配合 `F.cross_entropy`，让 PyTorch 自动在内部提升关键操作为 FP32。

---

## Part III -- LM Head 架构与工程边界

### 3.1 训练模式 vs 推理模式：双栖 API 设计

`LMHead` 被设计为**训练/推理双栖**：

```python
class LMHead(nn.Module):
    def forward(self, hidden_states, targets=None):
        logits = self.head(hidden_states)  # [B, L, V]

        loss = None
        if targets is not None:
            # 训练模式：计算 CE Loss
            ...
            loss = (-true_safe_logits + log_sum_exp).mean()

        return logits, loss
```

**设计理念**：

- **推理模式**（`targets=None`）：只做前向投影 `hidden → logits`，不计算损失。logits 交给下游的采样器（Top-P/Top-K/Beam Search）。loss 返回 `None`。
- **训练模式**（`targets` 传入）：在前向投影的同时计算 CE Loss，一次性完成"logits + loss"两条通路。

**为什么这么设计？**

1. **避免重复**：如果训练时分开调用 `lm_head(hidden)` 得到 logits 再调用 `F.cross_entropy(logits, targets)`，LogSumExp 会被计算两次（一次在 lm_head 内部，一次在外部）
2. **反向传播效率**：当 loss 从 lm_head 内部返回时，PyTorch 可以直接构建计算图，不需要额外的 Python 层面的 `F.cross_entropy` 调用。这在 DDP/FSDP 场景下减少了通信-计算的间隙
3. **API 一致性**：与 HuggingFace 的 `model(inputs, labels=labels)` 风格保持一致

**自回归 Shift 逻辑**：

语言模型训练的核心操作："预测下一个 token"。位置 $t$ 产生的隐藏状态 $h_t$ 应该预测位置 $t+1$ 的 token：

```python
shifted_logits = logits[..., :-1, :]   # 丢弃最后一个位置的 logits（无目标）
shifted_labels = targets[:, 1:]         # 丢弃第一个位置的标签（无上下文）
```

**为什么 logits 的第 $i$ 个位置预测 targets 的第 $i+1$ 个位置？** 因为输入序列 `[token_0, token_1, ..., token_{L-1}]` 经过 Transformer 后，位置 $t$ 的隐藏状态编码了 $token_0$ 到 $token_t$ 的信息，它"看到"的信息恰好足以预测 $token_{t+1}$。

### 3.2 Weight Tying：共享 Embedding 与 LM Head 权重

**Weight Tying** 是指：将词嵌入矩阵（Embedding，维度 $[V, d]$）与 LM Head 的投影矩阵（维度 $[V, d]$，或将 $[d, V]$ 转置）共享为同一个权重矩阵。

**动机**：

- 词嵌入矩阵将 token ID 映射为**输入表示**（离散 → 连续）
- LM Head 将最后一层的隐藏状态映射为**输出概率**（连续 → 离散）

两者在语义上是"互逆"的操作：一个从词表到向量空间，一个从向量空间回到词表。共享它们的参数在直觉上是合理的——如果两个 token 在嵌入空间中很接近（具有相似的输入表示），那么模型在输出它们时也应该赋予相近的概率。

**好处**：

1. **参数减少**：省掉一个 $V \times d$ 的矩阵（约 $128000 \times 4096 \times 2\text{ bytes} \approx 1.05\text{ GB}$ 对于 BF16 LLaMA-7B 规模）。注意在实际的大模型（LLaMA-7B 及以上）中，$V \times d$ 的量级与 Attention 的 $4 \times d^2$ 相当甚至更大，Weight Tying 节省的参数不可忽视
2. **正则化效果**：共享权重作为一种归纳偏置，强制模型使用一致的语义空间——如果一个 token 在输入侧被理解为某种语义，在输出侧也必须用同一套语义来解释。这减少了过拟合的风险
3. **理论优雅**：符合"编码器-解码器对称"的美感（虽然 Decoder-only 模型没有独立的编码器）

**代价与争议**：

1. **嵌入空间与输出空间的不对称**：输入嵌入将 token 映射为 Transformer 可以处理的向量，它需要捕捉的是词义的"丰富性"；LM Head 将隐藏状态映射为概率分布，它需要捕捉的是上下文的"区分度"。两者的优化方向不完全一致
2. **梯度冲突**：共享权重意味着嵌入矩阵同时从两个不同的目标接收梯度——来自 Embedding 层的"如何表示这个 token"和来自 LM Head 的"如何预测这个 token"。这两个目标在某些 token 上可能冲突（例如，function words 如 "the"/"and" 的输入表示可以是通用的，但输出预测需要极其精细的区分）
3. **大模型中是否有效？** LLaMA-1 使用了 Weight Tying，LLaMA-2/3 也沿用了。基于 Transformer 的模型中，Weight Tying 几乎已成为标准配置（Press & Wolf, 2017 首次提出，Inan et al., 2017 独立提出）

**代码实现（示意）**：

```python
class LMHeadWithTying(nn.Module):
    def __init__(self, embedding_weight):
        super().__init__()
        # 不创建独立的 head 权重，直接复用 embedding_weight

    def forward(self, hidden_states, targets=None):
        # 直接使用 self.embedding.weight 作为 LM Head 的权重
        logits = F.linear(hidden_states, self.embedding.weight)
        ...
```

**面试要点**：Weight Tying 是"用一致性假设换参数效率"——它假设嵌入空间和输出空间共享语义结构，牺牲了一定的表达能力，换来了约 15-20% 的参数减少和隐式的正则化效果。

### 3.3 LogSumExp vs Safe Softmax：为什么有了 Safe Softmax 还需要 LogSumExp

**Safe Softmax** 解决的是"直接算 Softmax 会溢出"的问题：

$$a_i = \frac{e^{z_i - M}}{\sum_j e^{z_j - M}}$$

这保证了指数计算安全（$\leq 1$）。**但 Safe Softmax 产出一个完整的 $[M, V]$ 概率矩阵**，显存需求巨大。

**LogSumExp** 更进一步——它把 Softmax + $\log$ + CE 整个计算**融合成一个表达式**：

$$L = -(z_c - M) + \ln\sum_j e^{z_j - M}$$

**关键区别**：

| | Safe Softmax + Log + NLL | LogSumExp (融合) |
|:--|:------------------------|:----------------|
| 中间张量 | $[M, V]$ 概率矩阵（$\sim 10\text{ GB}$） | 无（exp 后可立即 reduce） |
| 显存占用 | $O(MV)$ | $O(M)$ |
| $\log(0)$ 风险 | **有** -- 当 $P(c) = 0.0$（BF16 下溢） | **无** -- $-(z_c - M)$ 不会为 $-\infty$ |
| 数学等价性 | 完全等价 | 完全等价 |
| 计算图节点 | 3 个 op（softmax + log + nll） | 5 个 op（但无需分配大张量） |

**LogSumExp 如何消除 $\log(0)$ 风险：**

Safe Softmax：当 $z_c - M = -110$ 时，$P(c) = e^{-110} / \sum_j e^{z_j - M}$ 在 BF16/FP16 下可能下溢为绝对的 $0.0$，后续 $\ln(0.0) = -\infty$。

LogSumExp：同一场景下，$-(z_c - M) = 110$，而 $\ln\sum_j e^{z_j - M} \geq 0$（因为 $e^0 = 1 \in$ 求和项），所以 $L = 110 + \text{一个小正数}$——一个极差的预测被转化为一个**巨大的线性惩罚**（而非致命的 $-\infty$），梯度仍然有效。

### 3.4 $e^{-100}$ 下溢问题：为什么它不是灾难

承接上节——"$e^{-100}$ 下溢不是问题，$\ln(e^{-100})$ 才是问题"。

**核心洞察**：

在 LogSumExp 中，指数项的下溢是**安全的下溢**——被下溢为 $0.0$ 的项对求和的贡献本来就是**可以忽略不计**的。

- $e^{-100} \approx 3.72 \times 10^{-44}$，在 FP32 中可表示（FP32 最小正规数 $\approx 1.18 \times 10^{-38}$，次正规数 $\approx 1.4 \times 10^{-45}$），但在 BF16 中确实下溢为 $0.0$
- 但这个值对求和的贡献是 $3.72 \times 10^{-44}$，相对于求和中的其他项（最小是 $e^0 = 1$），它影响的是小数点后第 43 位——**完全不可观测**。
- 所以 $e^{-100}$ 在 BF16 下被截断为 $0.0$ 对 CE Loss 的数值影响是 $< 10^{-10}$，完全可以忽略。

**真正危险的是 $\ln(0)$**——但 LogSumExp 避免了它：

- 在 LogSumExp 中，我们计算的是 $\ln\sum_j e^{z_j - M}$，其中求和至少包含 $e^0 = 1$，$\ln(\text{求和})$ 至少是 $0$
- 正确 token 的贡献 $-(z_c - M)$ **不是**通过对数取 $e^{z_c - M}$ 再 $\ln$ 得到的（那才会遭遇 $\ln(0)$），而是直接从 logits 中减出来的——绕过了整个"下溢后取 log"的危险路径

**一句话总结**：$e^{-100}$ 的下溢让一项极小的贡献变为绝对 $0$——它对总和的影响不可观测，是"可容忍的误差"。但 $\ln(0.0) = -\infty$ 是让损失变为负无穷大——这是"不可容忍的错误"。LogSumExp 将前者保留为无害的下溢，完全消除了后者的可能性。

### 3.5 `ignore_index=-100` 的角色：Padding 掩码与 SFT 领域适配

**Padding 问题**：在批处理中，序列长度不同，需要填充（padding）到统一长度。填充位置的 token 不应参与损失计算。

**PyTorch 约定**：`F.cross_entropy` 使用 `ignore_index=-100`。所有标签值为 `-100` 的 token 被自动忽略——不贡献损失，不反向传播梯度。

**为什么是 -100？** 这是 PyTorch 的历史约定（继承自 Torch/Lua Torch 时代），因为词表 ID 通常从 0 开始，负数自然落在有效范围之外。也有其他框架使用 `-1`、`0` 或 `pad_token_id`。

**在 lm_head.py 中的实现**：

```python
valid_mask = (flat_labels != self.ignore_index)  # [M]
valid_logits = flat_logits[valid_mask]              # [M, V] -> [M_valid, V]
valid_labels = flat_labels[valid_mask]              # [M]    -> [M_valid]
```

使用布尔掩码直接**切片出有效 token**，而不是通过 "乘以零后求和"的方式。这样做的好处是：

1. **explicit**：清楚表明哪些 token 被排除
2. **不引入噪声梯度**：如果使用 `loss = masked_loss.sum() / mask.sum()`，虽然数学正确，但 loss 的均值计算中分母可能为 0（全 mask 场景）。切片方式天然避免了这个问题
3. **SFT 领域适配**：在 SFT（Supervised Fine-Tuning）中，通常使用 `ignore_index=-100` 来屏蔽非助手回复的部分（只对助手的回复计算损失）

**典型 SFT 的数据构造**：

```
[User] 什么是交叉熵？ [/User] [Assistant] 交叉熵是信息论中的... [/Assistant]
```

标签中，User 部分的 token 全部设为 `-100`，只有 Assistant 部分的 token 保留真实 ID。模型只学习如何生成高质量的回复，而非学习复述用户的问题。

### 3.6 DDP 死锁：当所有标签都是 -100

**问题场景**：在分布式训练（DDP，Distributed Data Parallel）中，如果某个 GPU 上的一个 micro-batch 中**所有标签都是 `-100`**（例如该 micro-batch 恰好全是填充 token 或用户 token），则该 GPU 上的 loss 为 `0.0`。

**DDP 死锁的机理**：

1. DDP 在反向传播时会自动对**所有参数的梯度**进行 AllReduce 同步
2. 如果某个 rank 上的 loss 为 $0.0$，`loss.backward()` 会产生**全零梯度**
3. 但是，loss 的计算图涉及 `valid_mask` 的分支逻辑——如果所有 token 都被 mask 掉，则 `valid_labels.numel() == 0`，返回的 loss 是 `torch.tensor(0.0, requires_grad=True)`
4. 这个张量是一个**叶子节点**（leaf tensor），它的计算图不包含任何模型参数
5. 当这个 leaf tensor 做 `backward()` 时，DDP 的 AllReduce hook 找不到对应的参数梯度，**等待其他 rank 的同步信号**——但其他 rank 也在等待这个 rank——形成**死锁**

**解决方案：`logits.sum() * 0.0` 技巧**：

```python
if valid_labels.numel() == 0:
    # 不要直接返回 torch.tensor(0.0, requires_grad=True)
    # 而是构造一个"确实涉及模型参数"的零梯度
    return logits, (logits.sum() * 0.0)
```

`logits.sum() * 0.0` 的值是 $0.0$，但它**确实在计算图中连接了模型参数**（`logits = self.head(hidden_states)`）。反向传播时：

1. `0.0` 乘以 `logits.sum()` 的梯度 → 对所有参数的梯度都是 $0.0$
2. DDP 的 AllReduce hook **可以看到所有参数的零梯度**
3. 所有 rank 成功同步——没有死锁

**为什么不是 `logits.mean() * 0.0` 或 `sum()` 而是 `sum()`？** `sum()` 产生一个标量（`mean()` 也一样），乘以 `0.0` 后数值为 $0$。关键是它**连接了整个 `logits` 的计算图**，而 `logits` 的计算涉及 `self.head(hidden_states)`。

**更优雅的替代方案**：在 PyTorch 2.0+ 中，可以使用 `torch._dynamo` 或自定义 `autograd.Function` 来处理，但 `logits.sum() * 0.0` 是**最简洁、最广泛兼容**的工业级做法。

**完整防御代码**：

```python
if valid_labels.numel() == 0:
    # DDP 安全：通过 logits 的计算图产生一个零 loss
    # 确保所有参数的梯度存在（即使是零梯度），避免 AllReduce 死锁
    return logits, logits.sum() * 0.0
```

---

## Part IV -- 面试 Q&A

### Q1：为什么 LLM 用交叉熵而不用 MSE？

**短答案**：CE + Softmax 的梯度是 $a_i - y_i$（线性，永不消失）；MSE + Softmax 的梯度包含 $a_i(1-a_i)$ 因子，在模型最需要学习的时候（$a_c \approx 0$）反而梯度消失。更深层的原因是对数评分规则是唯一局部严格恰当的评分规则。

详见 [Part I §1.1](#11-为什么不能用-mse) 和 [CE_Loss_Academic_Notes.md §5.4](CE_Loss_Academic_Notes.md#54-对比mse--softmax-为什么失败)。

### Q2：LogSumExp 和 Safe Softmax 有什么区别？

**短答案**：Safe Softmax 先算完整的 Softmax 概率矩阵再取 log，产生 $O(MV)$ 的中间张量。LogSumExp 将 Softmax + log + CE 融合成一步，无需中间张量，且完全消除了 $\ln(0)$ 的风险。数学等价，工程完全不同。

详见 [Part III §3.3](#33-logsumexp-vs-safe-softmax为什么有了-safe-softmax-还需要-logsumexp)。

### Q3：为什么要 `valid_logits.float()` 转为 FP32？

**短答案**：BF16 只有 7 位尾数，在跨量级求和时 $1.0 + \text{小量} \approx 1.0$（大数吃小数）。10 万词表的 LogSumExp 求和需要累加数千到数万个小值到 $1.0$ 上，BF16 会吞噬掉几乎所有的贡献。FP32 的机器精度 $\approx 10^{-7}$，足以覆盖。

详见 [Part II §2.1](#21-bf16-的吞噬效应与大数吃小数) 和 [Part II §2.4](#24-为什么-softmax-前必须转-fp32)。

### Q4：`gather` 函数是干什么的？为什么需要 `unsqueeze`？

**短答案**：`torch.gather(input, dim, index)` 沿指定维度按索引收集元素。`unsqueeze` 是因为 gather 要求 `index` 的形状与 `input` 的形状除 gather 维度外完全一致——从 `[M]` 变成 `[M, 1]` 才能与 `[M, V]` 的 input 对齐。

详见 [Part II §2.2](#22-torchgather-的完整使用指南)。

### Q5：`ignore_index=-100` 是做什么的？

**短答案**：标记为 `-100` 的 token 不参与损失计算（不贡献梯度）。用于屏蔽 padding token 和 SFT 中的非目标 token（如用户输入部分）。`-100` 是 PyTorch 的历史约定。

详见 [Part III §3.5](#35-ignore_index-100-的角色padding-掩码与-sft-领域适配)。

### Q6：DDP 训练中全 mask 的 micro-batch 会有什么问题？怎么解决？

**短答案**：如果所有标签都是 `-100`，loss 为标量 `0.0`（一个 leaf tensor），它不连接模型参数的计算图。DDP 的 AllReduce 在等待梯度同步时会**死锁**。解决方案是用 `logits.sum() * 0.0` 替代纯标量 `0.0`——它确实连接了计算图，让 DDP 能看到零梯度并正常通信。

详见 [Part III §3.6](#36-ddp-死锁当所有标签都是--100)。

### Q7：Weight Tying 是什么？LLaMA 用了吗？

**短答案**：Weight Tying 是将 Embedding 矩阵和 LM Head 矩阵共享为同一组权重。LLaMA-1/2/3 都使用了。省掉约 1 GB 参数（对于 7B 模型），提供隐式正则化。代价是输入嵌入和输出投影的优化方向不完全一致。

详见 [Part III §3.2](#32-weight-tying共享-embedding-与-lm-head-权重)。

### Q8：为什么 $e^{-100}$ 下溢不是问题？

**短答案**：$e^{-100} \approx 3.72 \times 10^{-44}$，在 BF16 下确实会下溢为 $0.0$。但它对 LogSumExp 求和的贡献是小数点后第 43 位，完全不可观测。真正危险的是 $\ln(0) = -\infty$，但 LogSumExp 避免了这一点——因为 $-(z_c - M)$ 直接从减过 $M$ 的 logits 中提取，不经过 $\log$ of 极小值。

详见 [Part III §3.4](#34-e-100-下溢问题为什么它不是灾难)。

### Q9：Logits 张量 [B, S, V] 为什么是显存杀手？

**短答案**：因为 $V$ 很大（$\sim 128\text{K}$）且不能分片（Softmax 需要全词表归一化）。$B=8, S=4096, V=128000$ 时，单个 logits 张量 FP32 下占用约 16.8 GB 显存。这是训练时最大的中间激活。LogSumExp 通过算子融合避免了产出额外的 $[B, S, V]$ 中间概率矩阵。

详见 [Part II §2.3](#23-logits-张量的工程意义b-s-v-的显存冲击)。

### Q10：交叉熵、KL 散度、NLL 三者是什么关系？

**短答案**：核心恒等式 $H(P, Q) = H(P) + D_{KL}(P \parallel Q)$。交叉熵 = 数据固有熵 + KL 散度。当 $P$ 固定时，最小化交叉熵 $\equiv$ 最小化 KL 散度 $\equiv$ 最小化负对数似然（NLL）。三个目标在优化意义下等价。

详见 [CE_Loss_Academic_Notes.md §1.4](CE_Loss_Academic_Notes.md#14-交叉熵的严格定义与核心恒等式)。

---

## 本目录文件

| 文件 | 说明 |
| :--- | :--- |
| [README.md](README.md) | **入门索引** -- 文件清单、学习路径、运行指令 |
| [lm_head.py](lm_head.py) | **手撕实现** -- LM Head + LogSumExp CE Loss，含五重测试（前向对齐、反向梯度、极端数值、全 mask、推理模式） |
| [CE_Loss_Notes.md](CE_Loss_Notes.md)（本文件） | **理论与工程笔记** -- 四大部分：CE 理论基础 / 浮点精度与显存 / LM Head 架构与工程边界 / 面试 Q&A |
| [CE_Loss_Academic_Notes.md](CE_Loss_Academic_Notes.md) | **学术深度推导** -- 从 Shannon 公理出发，经 Proper Scoring Rule 统一框架，到困惑度与信息密度，含三份附录 |

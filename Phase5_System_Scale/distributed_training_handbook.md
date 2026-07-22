# 分布式训练系统手册

> 一份简化版的分布式训练教科书。覆盖从「为什么单卡不够」到「2026 年 DeepSeek 前沿实践」的完整知识体系。
> 可与 [PLAN.md](PLAN.md) 配合使用：PLAN.md 告诉每天干什么，本手册提供每章要掌握的核心理论。

---

## 目录

- [第一部分：地基 — 为什么要分布式](#第一部分地基--为什么要分布式)
  - [第 1 章：单卡之墙](#第-1-章单卡之墙)
  - [第 2 章：混合精度训练](#第-2-章混合精度训练)
  - [第 3 章：GPU 硬件基础](#第-3-章gpu-硬件基础)
- [第二部分：通信 — 多卡如何对话](#第二部分通信--多卡如何对话)
  - [第 4 章：集合通信原语](#第-4-章集合通信原语)
  - [第 5 章：分布式训练基础设施](#第-5-章分布式训练基础设施)
- [第三部分：并行策略 — 五种切分维度](#第三部分并行策略--五种切分维度)
  - [第 6 章：并行的分类学](#第-6-章并行的分类学)
  - [第 7 章：数据并行 — 最朴素、最重要的基线](#第-7-章数据并行--最朴素最重要的基线)
  - [第 8 章：张量并行 — 把一层矩阵切成多份](#第-8-章张量并行--把一层矩阵切成多份)
  - [第 9 章：流水线并行 — 把层序列切成多段](#第-9-章流水线并行--把层序列切成多段)
  - [第 10 章：ZeRO & FSDP — 不是切模型，是切状态](#第-10-章zero--fsdp--不是切模型是切状态)
  - [第 11 章：序列并行 — 当上下文比模型还大](#第-11-章序列并行--当上下文比模型还大)
  - [第 12 章：专家并行 — MoE 的规模化](#第-12-章专家并行--moe-的规模化)
- [第四部分：工业实战 — 拼起来与跑起来](#第四部分工业实战--拼起来与跑起来)
  - [第 13 章：混合并行策略选择](#第-13-章混合并行策略选择)
  - [第 14 章：主流框架概览](#第-14-章主流框架概览)
- [第五部分：前沿与展望](#第五部分前沿与展望)
  - [第 15 章：2024-2026 五大突破](#第-15-章2024-2026-五大突破)
  - [第 16 章：分布式推理](#第-16-章分布式推理)
  - [第 17 章：未竟之地](#第-17-章未竟之地)
- [附录 A：关键公式与常数速查](#附录-a关键公式与常数速查)
- [附录 B：推荐阅读清单](#附录-b推荐阅读清单)
- [附录 C：术语中英对照表](#附录-c术语中英对照表)

---

# 第一部分：地基 — 为什么要分布式

> **本手册的参考模型**：除非特别说明，全手册的 FLOPs 和显存计算以 **Qwen3-8B**（2026.4，Alibaba）为基准。配置：$d_{model}=4096$, $d_{ff}=12288$, $n_h=32$, $n_{kv}=8$ (GQA 4:1), $d_{head}=128$, $n_{layers}=36$, $\text{vocab}=152064$, 总参数量 ~8.2B。其他模型（LLaMA 3 8B/70B, DeepSeek V3/V4）的配置见[附录 B](#附录-b推荐阅读清单)。

# 第 1 章：单卡之墙

> **核心问题**：训练一个现代大语言模型到底需要多少计算和显存？为什么一张 GPU 不够？

## 1.1 算力账：训练一个模型要多少计算

### 矩阵乘法的 FLOPs 计算

分布式训练的第一笔账，是搞清楚一次前向传播到底花了多少计算量。一切从矩阵乘法开始。

一个 `nn.Linear(d_in, d_out)` 的前向计算等价于矩阵乘法 $Y = X \cdot W$，其中 $X \in \mathbb{R}^{M \times d_{in}}$，$W \in \mathbb{R}^{d_{in} \times d_{out}}$。输出 $Y$ 的每个元素由内积得到：

$$Y_{i,j} = \sum_{k=1}^{d_{in}} X_{i,k} \cdot W_{k,j}$$

这个内积涉及 $d_{in}$ 次乘法 + $d_{in}$ 次加法 = $2 d_{in}$ 次浮点运算。整个矩阵有 $M \times d_{out}$ 个元素，因此总 FLOPs 为：

$$\text{FLOPs} = 2 \times M \times d_{in} \times d_{out}$$

这个「2」的来源至关重要——每个乘加对（multiply-add）算 2 次浮点运算。在 Transformer 的语境下，$M = B \times S$（batch size × 序列长度），所以一层 Linear 的 FLOPs = $2 \times B \times S \times d_{in} \times d_{out}$。

### Transformer 前向 FLOPs 完整分解

以 Qwen3-8B（2026 年 4 月发布，截至本手册撰写时是最新的 open-source dense LLM）为例，先列出各层的矩阵形状：

| 参数 | 值 |
|------|-----|
| $d_{model}$ (hidden_dim) | 4096 |
| $d_{ff}$ (FFN intermediate) | 12288（SwiGLU 用 gate+up 两个投影各 4096→12288） |
| $n_{heads}$ (Q heads) | 32 |
| $n_{kv\_heads}$ (KV heads, GQA) | 8（GQA 4:1，Q heads 32 / KV heads 8） |
| $d_{head}$ | 128 |
| $n_{layers}$ | 36 |
| vocab_size | 152064 |

**Qwen3-8B 的 GQA 对 FLOPs 的影响**：KV 投影的 FLOPs 比 Q 投影少。具体来说：
- Q 投影：$2 \times B \times S \times 4096 \times (32 \times 128) = 2BS \times 4096^2$（全部 32 个 heads）
- K 投影：$2 \times B \times S \times 4096 \times (8 \times 128) = 2BS \times 4096 \times 1024$
- V 投影：同 K

所以 QKV 投影总计为 $2BS \times 4096 \times (4096 + 1024 + 1024) = 2BS \times 4096 \times 6144$，而非之前的 $6BS \times 4096^2$。$\frac{6144}{4096 \times 3} = \frac{6144}{12288} = 0.5$——GQA 4:1 下，QKV 投影 FLOPs 恰好是 dense MHA 的 50%。这是一个有实际意义的节省。

**Attention 层的四部分 FLOPs**：

1. **QKV 投影**：Q、K、V 各一个 `nn.Linear`，但 GQA 4:1 下 K 和 V 的输出维度是 Q 的 1/4。

   $$\begin{aligned}
   \text{FLOPs}_{Q}    &= 2 \times B \times S \times 4096 \times (32 \times 128) = 2BS \times 4096^2 \\
   \text{FLOPs}_{K}    &= 2 \times B \times S \times 4096 \times (8 \times 128)  = 2BS \times 4096 \times 1024 \\
   \text{FLOPs}_{V}    &= 2 \times B \times S \times 4096 \times (8 \times 128)  = 2BS \times 4096 \times 1024 \\
   \text{FLOPs}_{QKV}  &= 2BS \times 4096 \times (4096 + 1024 + 1024) = 2BS \times 4096 \times 6144
   \end{aligned}$$

   相比 dense MHA（$6BS \times 4096^2 = 6BS \times 4096 \times 4096$），GQA 4:1 的 QKV 投影恰为 dense 的 $\frac{6144}{3 \times 4096} = \frac{6144}{12288} = 50\%$。

2. **Attention Scores（Q×K^T）**：$Q \in \mathbb{R}^{B \times 32 \times S \times 128}$，$K^T \in \mathbb{R}^{B \times 8 \times S \times 128}$。但 GQA 下 K 的 heads 数 ≠ Q 的，PyTorch 实际实现是对 K expand 到 32 heads（通过 `repeat_interleave`），所以有效 dimension 仍为 $32 \times S \times 128$。

   $$\text{FLOPs}_{scores} = 2 \times B \times 32 \times S \times S \times 128 = 2BS^2 \times 4096$$

   注意：这里的 $S^2$ 是注意机制的计算瓶颈——随序列长度**二次**增长。

3. **Weighted Sum（softmax(A)×V）**：

   $$\text{FLOPs}_{wsum} = 2 \times B \times 32 \times S \times S \times 128 = 2BS^2 \times 4096$$

   和 Scores 的 FLOPs 相同——都是 $2BS^2 \times (n_h \cdot d_h)$。

4. **Output 投影**：$O \in \mathbb{R}^{B \times S \times 4096}$。

   $$\text{FLOPs}_{out} = 2 \times B \times S \times 4096 \times 4096 = 2BS \times 4096^2$$

**FFN 层（SwiGLU）的三部分 FLOPs**：

SwiGLU 的公式为 $\text{SwiGLU}(x) = (\text{SiLU}(xW_{gate}) \odot xW_{up}) \cdot W_{down}$。三个矩阵乘法：

1. **gate_proj**：$W_{gate} \in \mathbb{R}^{4096 \times 12288}$
   $$\text{FLOPs}_{gate} = 2 \times B \times S \times 4096 \times 12288$$

2. **up_proj**：同上
   $$\text{FLOPs}_{up} = 2 \times B \times S \times 4096 \times 12288$$

3. **down_proj**：$W_{down} \in \mathbb{R}^{12288 \times 4096}$
   $$\text{FLOPs}_{down} = 2 \times B \times S \times 12288 \times 4096$$

注：SiLU 激活函数（$\text{SiLU}(x) = x \cdot \sigma(x)$，约 6 FLOPs/element）和逐元素乘法（$\odot$，1 FLOP/element）总计约 $7 B S d_{ff} \approx 7 \times 1 \times 2048 \times 12288 \approx 1.76 \times 10^8$ FLOPs。相对于 gate_proj matmul 的 $2.06 \times 10^{11}$ FLOPs，**占比 0.085%——完全可以忽略**。同理 Softmax 的 FLOPs（exp + sum + div，约 $8 B n_h S^2 \approx 1.07 \times 10^9$ FLOPs at S=2048）相对 Attention Scores 的 $3.44 \times 10^{10}$ FLOPs 约占 3.1%，也忽略。（更详细的论证见 [System_Math 笔记](System_Math/system_math_notes.md) 第 3 章。）

**关键对比**：代入 $B=1, S=2048$，用 Python 精确计算：

```python
B, S = 1, 2048
d_model, d_ff, n_heads, n_kv, d_head = 4096, 12288, 32, 8, 128

# Attention 四部分
# QKV: Q 投影(全 heads) + K 投影(仅 kv_heads) + V 投影(仅 kv_heads)
q_proj = 2 * B * S * d_model * (n_heads * d_head)           # 68,719,476,736
k_proj = 2 * B * S * d_model * (n_kv * d_head)             # 17,179,869,184
v_proj = 2 * B * S * d_model * (n_kv * d_head)             # 17,179,869,184
qkv    = q_proj + k_proj + v_proj                           # 103,079,215,104
# GQA 节省：dense MHA 下 QKV = 206,158,430,208。GQA 4:1 节省 50%

scores = 2 * B * n_heads * S * S * d_head                  #  34,359,738,368
wsum   = 2 * B * n_heads * S * S * d_head                  #  34,359,738,368
out    = 2 * B * S * d_model * d_model                     #  68,719,476,736
attn   = qkv + scores + wsum + out                          # 240,518,168,576

# FFN 三部分
gate   = 2 * B * S * d_model * d_ff                        # 206,158,430,208
up     = 2 * B * S * d_model * d_ff                        # 206,158,430,208
down   = 2 * B * S * d_ff * d_model                        # 206,158,430,208
ffn    = gate + up + down                                   # 618,475,290,624
```

- Attention（一层，含 GQA QKV+Scores+WSum+Output）≈ $2.41 \times 10^{11}$ FLOPs
- FFN（一层，gate+up+down）≈ $6.18 \times 10^{11}$ FLOPs
- **FFN 占比 ≈ 72%，Attention 占比 ≈ 28%**（在 S=2048 时）
- **FFN 占比 ≈ 72%，Attention 占比 ≈ 28%**（在 S=2048 时）

但注意：Attention 的四部分中，平方项（Scores + WSum ≈ $6.87 \times 10^{10}$）占比约为 $6.87 \times 10^{10} / 2.41 \times 10^{11} \approx 29\%$。当 S 增长时，平方项从 29% 迅速爬到主导地位：

但当 S 从 2048 增长到 8192（4×）：

| S | Attention FLOPs | FFN FLOPs | Attention 占比 | FFN 占比 | Attn/FFN 比 |
|---|----------------|-----------|---------------|---------|------------|
| 512 | $2.49 \times 10^{10}$ | $1.55 \times 10^{11}$ | ~14% | ~86% | 0.16 |
| 2048 | $2.41 \times 10^{11}$ | $6.18 \times 10^{11}$ | ~28% | ~72% | 0.39 |
| 8192 | $3.25 \times 10^{12}$ | $2.47 \times 10^{12}$ | ~57% | ~43% | 1.32 |
| 32768 | $4.93 \times 10^{13}$ | $9.89 \times 10^{12}$ | ~83% | ~17% | 4.98 |

**核心洞察**：短序列瓶颈在 FFN（矩阵乘法的参数量大），长序列瓶颈在 Attention（$S^2$ 项压倒一切）。这个翻转点是理解序列并行（Ch11）为什么存在的关键。

### 训练总 FLOPs：6ND 公式

一次前向传播的总 FLOPs 可以近似为 $C_{fwd} \approx 2ND$，其中 $N$ 是模型参数量，$D$ 是训练的 token 数。这个近似的推导逻辑是：

- 模型有 $N$ 个参数，每个 token 的一次前向，每个参数约参与 2 次浮点运算（一次乘一次加，在 matmul 中）
- $D$ 个训练 token，总前向 FLOPs ≈ $2ND$

反向传播的 FLOPs 约为前向的 2 倍，因为需要计算两类梯度：
- **Weight gradients**（参数梯度）：计算量 ≈ 前向 FLOPs（再来一次 matmul 但方向不同）= $2ND$
- **Input gradients**（激活值梯度，用于链式法则继续往前传）：计算量 ≈ 前向 FLOPs = $2ND$

因此反向传播合计 ≈ $4ND$，训练总 FLOPs：

$$C_{total} = C_{fwd} + C_{bwd} \approx 2ND + 4ND = 6ND$$

代入 Qwen3-8B ($N = 8.2 \times 10^9$)，假设训练 $D = 2 \times 10^{12}$ tokens（2T）：

$$C_{total} \approx 6 \times 8.2 \times 10^9 \times 2 \times 10^{12} = 9.84 \times 10^{22} \text{ FLOPs}$$

H100 的 bf16 理论峰值约为 989 TFLOPS（$9.89 \times 10^{14}$ FLOPs/s）。如果单卡训练：

$$\text{时间} = \frac{9.84 \times 10^{22}}{9.89 \times 10^{14}} \approx 9.95 \times 10^7 \text{ 秒} \approx 3.2 \text{ 年}$$

这还没算通信开销、MFU 折扣（实际利用率只有 50-60%）、数据加载等。**单卡训练 Qwen3-8B 根本不可行**——而 8B 已经是最小的实用 LLM 之一了。

### Scaling Laws 与 Chinchilla 修正

Kaplan et al. (2020) 的 Scaling Laws 发现：对于给定的计算预算 $C$，最优的模型大小 $N$ 和数据量 $D$ 与 $C$ 呈幂律关系。他们得出应该优先增大模型大小而非数据量。

但 Hoffmann et al. (2022) 的 Chinchilla 论文修正了这一结论：**参数和数据应该等比增长**。具体来说，对于给定的计算预算 $C$，最优配置为 $N_{opt} \propto C^{0.5}$，$D_{opt} \propto C^{0.5}$。Chinchilla-70B 仅用 1.4T tokens 训练就超越了用了 300B tokens 训练的 Gopher-280B。

这个修正的工程含义是：很多大模型其实是「欠训练」（undertrained）的——参数量太大，数据量不够。要训练一个 70B 模型到饱和，至少需要 ~1.4T tokens。

## 1.2 显存账：16 bytes/param 从哪来

算力账告诉我们训练要多久。但更紧迫的问题是：**模型能不能放进 GPU**。

### 模型状态的五层分解

训练时，GPU 显存中驻留的数据分为以下部分（以 bf16 混合精度 + Adam 优化器为例）：

| 组件 | 精度 | 字节/参数 | 说明 |
|------|------|----------|------|
| 模型参数 | bf16 | 2 | 前向和反向都需要 |
| 梯度 | bf16 | 2 | 反向传播产生 |
| Adam m（一阶动量） | fp32 | 4 | $\beta_1=0.9$ 的指数移动平均 |
| Adam v（二阶动量） | fp32 | 4 | $\beta_2=0.999$ 的指数移动平均 |
| Master weights | fp32 | 4 | fp32 精度的参数副本，用于精确更新 |
| **合计（Model States）** | | **16** | |

加上激活值和临时缓冲区（Residual States），Qwen3-8B 的总显存需求：

**Model States = 8.2 × 10⁹ × 16 = 131.2 GB**

这个数字已经超过了 A100-80G 和 H100-80G 的显存容量。而且还没算激活值。

### 为什么 Adam 是头号杀手

Adam 优化器占 12 bytes/param = 75% 的模型状态显存。根本原因是它为每个参数维护了两个 fp32 的动量估计，再加上 fp32 master weights。**如果没有优化器状态，单卡就能装下 Qwen3-8B——这正是推理（inference）为什么可以在单卡上跑的原因。推理只需要 2 bytes/param（约 16.4 GB for 8.2B），而训练需要 16 bytes/param（131.2 GB）。**

### 激活值的估算

激活值（activations）是前向传播中需要保存下来供反向使用的中间结果。在 Transformer 中主要包括：

- 每层的 hidden_states（输入到 Attention/FFN 之前的）
- Attention 的中间结果：Q, K, V, softmax 前的 attention scores
- FFN 的中间结果：gate_proj 输出, up_proj 输出, SiLU 后的 gated 输出
- RMSNorm 的输入（用于反向计算梯度）

以 Qwen3-8B 的一层 SwiGLU FFN 为例：
- hidden_states 输入：$B \times S \times 4096 \times 2$ bytes = $1 \times 2048 \times 4096 \times 2 = 16$ MB
- gate_proj 输出：$B \times S \times 12288 \times 2 = 48$ MB
- up_proj 输出：$B \times S \times 12288 \times 2 = 48$ MB
- down_proj 输入（SiLU 后）：$B \times S \times 12288 \times 2 = 48$ MB

仅 FFN 一层的激活值 ≈ 约 180 MB。36 层总计 ≈ 6.5 GB（仅 FFN 部分）。加上 Attention 的 Q/K/V/Scores + Residual states，总激活值 ≈ 8-12 GB（取决于具体的中间张量保留策略和是否使用 FlashAttention）。

**总计 = Model States(131.2GB) + Activations(~10GB) ≈ 141.2 GB > A100-80G**

### Activation Checkpointing：用计算换显存

Activation Checkpointing（也称 gradient checkpointing 或 recomputation）的核心思想：**不存储全部中间激活值，而是每 N 层设一个检查点；反向传播时从最近的检查点重新前向计算需要的激活值**。

- 如果每层都设检查点 → 激活值接近 0，但反向多了一次完整前向 → 训练 FLOPs +33%
- 如果每 2 层设检查点 → 激活值减半，额外计算 +16%
- 常见配置：`checkpoint_every = 1`（极致省显存）或只在 Attention 层 checkpoint（省最多且只重算 Attention，因为 Attention 的激活值远大于 FFN）

**关键理解**：AC 省的是激活值（Residual States），不碰模型状态（Model States）。Qwen3-8B 的 Model States = 131.2GB，AC 对此无能为力。所以 AC 从来不是单卡训练的独立解药——它是 ZeRO/TP 的补充。

### Gradient Accumulation：用时间换 batch size

Gradient Accumulation（梯度累积）的机制：将一个大 batch 拆成多个 micro-batch，每个 micro-batch 独立做前向+反向，梯度在本地累积（相加），累积够 N 步之后再 AllReduce + optimizer.step。

- peak memory：只看一个 micro-batch 的前向+反向所需内存
- effective batch_size = micro_batch_size × accumulation_steps × DP_size
- 额外代价：无（只是多跑几步），但总吞吐不变——你算的还是那么多 FLOPs

GA 和 AC 是不同的维度：AC 省激活值显存（用计算换），GA 不省任何显存但让大 effective batch 在显存受限时变得可行。

## 1.3 小结：三条出路与解法地图

面对「单卡装不下」的问题，现代分布式训练有三个互补的解法维度：

| 解法 | 省什么 | 代价 | 代表技术 |
|------|--------|------|---------|
| 省计算 | 激活值显存 | +33% FLOPs（重算） | Activation Checkpointing |
| 省显存 | 优化器状态 75% | +通信量 | ZeRO-1/2/3, FSDP |
| 分而治之 | 所有状态（参数+梯度+Adam） | +通信量 +复杂性 | TP, PP, EP |

第一条路让你省了 ~10GB，但 112GB 的 Model States 纹丝不动。第二条和第三条路才是真正的分布式解决方案——它们把 Model States 分布到多张 GPU 上，让每张卡只承担 1/N 的显存压力。

但要理解这些策略如何工作，首先需要理解多张 GPU 之间如何通信——这是第 2 章的主题。

---

# 第 2 章：混合精度训练

> **核心问题**：训练时必须用 fp32 吗？能不能用更低精度来省显存、加速计算？bf16 和 fp16 有什么不同？fp8 可行吗？

## 2.1 浮点数的表示 — 从 IEEE 754 讲起

要理解混合精度训练，必须先理解浮点数在计算机中如何表示。

### IEEE 754 二进制浮点数

任何浮点数都可以写成：

$$x = (-1)^s \times 1.m \times 2^{e - bias}$$

其中 $s$ 是符号位（0 正 1 负），$m$ 是尾数（mantissa，隐含前导 1），$e$ 是指数（exponent），$bias$ 是指数偏移值。不同精度的格式用不同长度：

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | bias |
|------|--------|--------|--------|--------|------|
| FP32 | 32 | 1 | 8 | 23 | 127 |
| FP16 | 16 | 1 | 5 | 10 | 15 |
| BF16 | 16 | 1 | 8 | 7 | 127 |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | 7 |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | 15 |

### 数值范围与精度

FP32 的数值范围：
- 最大正数：$(2 - 2^{-23}) \times 2^{127} \approx 3.4 \times 10^{38}$
- 最小正数（normal）：$1.0 \times 2^{-126} \approx 1.2 \times 10^{-38}$
- 最小正数（subnormal）：$2^{-23} \times 2^{-126} \approx 1.4 \times 10^{-45}$
- 精度：约 7 位有效十进制数字（$2^{-23} \approx 1.2 \times 10^{-7}$）

FP16 的问题一目了然——5 位指数只有 $2^5 = 32$ 个值，范围从 $2^{-14}$ 到 $2^{15}$。最小正数仅 $6.0 \times 10^{-8}$。训练中很多梯度远小于这个值（特别是在大模型的深层、小 batch 的训练后期），这些梯度会被截断为 0——称为「梯度下溢」。

**具体算一下**：假设一个 70B 模型的第 60 层有一个参数，其梯度约为 $10^{-7}$。这个值在 FP16 的表示范围内（$> 6.0 \times 10^{-8}$），勉强可以存。但如果这个梯度再小一点——比如 $10^{-8}$——FP16 就无法表示了。而 Adam 的更新量通常比梯度本身更小（因为被学习率缩放），$lr \times m_t / (\sqrt{v_t} + \epsilon)$ 这个量经常在 $10^{-9}$ 量级甚至更小——远远超出 FP16 的范围。

BF16 的解决方案极其优雅：只把指数位从 5 扩展到 8——和 FP32 的指数相同。这保证了 BF16 的数值范围和 FP32 完全一致（±3.4×10³⁸），只是精度从 7 位降到约 2 位有效十进制数字（$2^{-7} \approx 7.8 \times 10^{-3}$）。对深度学习来说这完全够用——梯度本身的随机性远大于 1% 量级，2 位精度足够捕获梯度的方向和相对大小。

**为什么 BF16 叫「Brain Float」？** BF16 由 Google Brain 团队在 TPU 上首次提出和部署。Google 发现：FP16 的数值范围问题是一个持续的工程负担（需要 loss scaling），而把 FP32 的尾数从 23 位砍到 7 位对深度学习训练几乎没有影响。因此 BF16 本质上是「FP32 指数 + 极短的尾数」。这个简单到近乎粗暴的想法反而成了最优解——现在所有 LLM 训练都默认使用 BF16。

## 2.2 混合精度训练的机制

### 为什么 Model States 没减反而「增了」

首先澄清一个常见的混淆：

| 训练精度 | 参数存储 | 梯度存储 | Adam m | Adam v | master weights | 总计/param | 备注 |
|----------|---------|---------|--------|--------|---------------|-----------|------|
| 全 FP32 | 4 bytes | 4 bytes | 4 bytes | 4 bytes | 0（不需要） | **16 bytes** | baseline |
| FP16 混合精度 | 2 bytes | 2 bytes | 4 bytes | 4 bytes | 4 bytes | **16 bytes** | FP16 用于计算，FP32 master 用于更新 |
| BF16 混合精度 | 2 bytes | 2 bytes | 4 bytes | 4 bytes | 4 bytes | **16 bytes** | BF16 用于计算，FP32 master 用于更新 |

**全 FP32 训练不需要 master weights**：因为参数和梯度本身就是 FP32 的，Adam 更新也是 FP32 的——所有运算都在同一个精度下，没有信息损失。

**混合精度需要 master weights**：前向和反向在 BF16/FP16 下进行。但 optimizer.step() 的更新量通常非常小（$lr \times m_t / (\sqrt{v_t} + \epsilon)$），如果直接在低精度参数上做加法，大量低位会被截断——参数的累积更新会「漂移」。解决方案：维护一份 FP32 的 master weights（额外的 4 bytes/param），Adam 在 FP32 下更新 master weights，然后将 master weights 截断/舍入到 BF16/FP16 供下一次前向使用。

**所以混合精度没有节省 Model States 显存**——但它确实节省了**激活值**显存（从 FP32 的 4 bytes/元素 降到 BF16 的 2 bytes/元素），因为前向的中间结果用 BF16 存储即可。

### 激活值显存的节省

以 LLaMA-7B 的 (bs=1, S=2048) 配置为例：

| 组件 | FP32 训练 | BF16 训练 | 节省 |
|------|----------|----------|------|
| Model States | 16 × 7B = 112 GB | 16 × 7B = 112 GB | 0 GB |
| 激活值（估计） | ~20 GB | ~10 GB | ~10 GB |
| 总显存 | ~132 GB | ~122 GB | ~10 GB (7.6%) |

激活值的节省虽然不如 Model States 那么大（因为 Model States 是 112 GB 的大头），但 10 GB 的节省对于在单卡上训练 7B 模型来说是把 122 GB 压到 112 GB——仍然超 80 GB，但总算往 80 GB 靠近了一步。对于 70B 模型，激活值节省约 100 GB——这个数字就更可观了。

### 计算吞吐的翻倍

混合精度训练的**真正最大收益**是计算速度。H100 上各精度的 Tensor Core 理论峰值：

| 精度 | TFLOPS | 相对 FP32 | 适用架构 |
|------|--------|----------|---------|
| FP32 | 67 | 1× | All |
| TF32（A100+） | 156 | 2.3× | Ampere+ |
| FP16 | 989 | 14.8× | Volta+ |
| BF16 | 989 | 14.8× | Ampere+ |
| FP8 (E4M3) | 1979 | 29.5× | Hopper+ |

**TF32 的说明**：TF32（TensorFloat-32）是 NVIDIA Ampere（A100）引入的格式，用于替代默认的 FP32 累加。它在 Tensor Core 内部用 19 位（1+8+10，和 FP16 相同的尾数 + BF16 相同的指数），但在寄存器中表示为 FP32。这是「免费午餐」——只需要在 PyTorch 中设置 `torch.backends.cuda.matmul.allow_tf32 = True`（PyTorch 1.7+ 默认开启），不需要修改任何代码，矩阵乘法的吞吐就能翻倍。

FP16/BF16 相比 FP32 快了约 15×——这不是小数字。对于一次 7B 模型的完整训练（数万亿 FLOPs），这意味着计算时间从「数年」变为「数周到数月」。

## 2.3 FP16 的数值陷阱与 Loss Scaling

### 为什么 FP16 需要 Loss Scaling

FP16 的最小正数（normal）约为 $6.0 \times 10^{-5}$（指数 -14 + 尾数 10 位）。这看起来还行——梯度的典型量级在 $10^{-3}$ 到 $10^{-6}$。但有两类值在这个范围之下：

**第一类：小梯度的累积更新**。Adam 的更新量 = $lr \times m_t / (\sqrt{v_t} + \epsilon)$。当 $lr = 10^{-4}$，momentum $m_t \approx 10^{-5}$，second moment $\sqrt{v_t} \approx 10^{-3}$ 时，更新量 ≈ $10^{-4} \times 10^{-5} / 10^{-3} = 10^{-6}$。如果梯度再小一些（$10^{-6}$），更新量就是 $10^{-7}$——在 FP16 的边缘。训练后期梯度尺度下降 + 学习率衰减时，更新量可能到 $10^{-8}$ 以下——直接 flush to zero。

**第二类：偏置参数的梯度**。Layernorm 的 bias、embedding 的 token frequency（罕见 token 的梯度在 FP16 下可能为 0）。

### Loss Scaling 的完整机制

Loss scaling 的核心思路在前向端做放大，反向端自动继承，更新端缩回：

1. **确定 scale factor S**：运行若干步 warm-up，监测梯度的最大值。如果 `max_grad × S` 在 FP16 可表示范围的上半部分（如 $10^2-10^4$），scale 就选对了。通常初始取 $S = 2^{16} = 65536$
2. **scaled 前向**：`loss = criterion(output, labels) * S`
3. **scaled 反向**：`scaled_loss.backward()`。根据链式法则，所有梯度自动乘以 S——不需要显示操作
4. **unscale + 更新**：优化器更新前，`param.grad /= S` 将梯度恢复到真实尺度，然后 `optimizer.step()`
5. **动态调整 S**：
   - 如果梯度没有 inf/NaN 且梯度 max 值远小于 FP16 可表示范围的下半部分 → 增大 S（给下溢留更多空间）
   - 如果有 inf/NaN → 减小 S 并跳过该步（不更新参数）

PyTorch 提供了 `torch.cuda.amp.GradScaler` 来自动处理这个流程：

```python
scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()  # 动态调整 scale
```

`autocast` 上下文内，PyTorch 自动将兼容的操作（matmul, convolution）转为 FP16，保留不兼容操作（softmax, layer_norm, loss computation）在 FP32。

### 为什么 BF16 省掉了这一整套机制

BF16 和 FP32 共享 8 位指数 → 共享 $2^{-126} \approx 1.2 \times 10^{-38}$ 的最小正数。这个范围对于任何实际训练中出现的梯度都完全足够——不需要 loss scaling，不需要动态调整 S，不需要 `GradScaler`。

BF16 训练的代码和 FP32 训练几乎一样：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, labels)

loss.backward()
optimizer.step()
```

没有 scaler，没有 scale/unscale 逻辑。这就是为什么社区从 FP16 转向 BF16 的原因——不是精度问题，而是**工程复杂度**。FP16 的 loss scaling 是一个持续的调参负担（scale 太小→下溢，太大→溢出→跳过更新→收敛变慢）。

## 2.4 FP8 训练 — 从理论到工业生产

### FP8 的两种格式

FP8 是 Nvidia Hopper（H100，2023）引入的新精度格式。和 BF16 不同，FP8 有两种变体，服务于不同的目的：

**E4M3（前向/激活值）**：
- 4 位指数 + 3 位尾数 → bias=7
- 最大正数：$(2 - 2^{-3}) \times 2^{15-7} = 1.75 \times 2^8 = 448$
- 最小正数（normal）：$1.0 \times 2^{-6} = 0.0156$
- 精度：约 1 位有效十进制数字（$2^{-3} = 0.125$ 的相对精度）
- 适用：前向的权重和激活值。通常数值在 $[-1, 1]$ 或 $[-10, 10]$ 范围，E4M3 能很好地表示

**E5M2（反向/梯度）**：
- 5 位指数 + 2 位尾数 → bias=15
- 最大正数：$(2 - 2^{-2}) \times 2^{30-15} = 1.75 \times 2^{15} = 57344$
- 最小正数：$1.0 \times 2^{-14} \approx 6.1 \times 10^{-5}$
- 精度：约 0.6 位有效十进制数字（$2^{-2} = 0.25$ 的相对精度）
- 适用：反向的梯度。梯度可能跨越多个数量级（从模型前层的 $10^{-2}$ 到后层的 $10^{-6}$），需要更大的动态范围

**为什么需要两种格式？** 前向传播中的激活值通常在一个可控的数值范围内（weight normalization 的副作用），只需要 4 位指数。但反向传播中的梯度可以非常小（深层后小到 $10^{-9}$），需要额外的指数位来防止下溢。这恰好呼应了 FP16 vs BF16 的同一问题——在 8 位上，这个问题需要用两种专门的格式来解决。

### 为什么不能直接把整张量量化到 FP8

FP8 的精度——E4M3 的 3 位尾数——意味着两个相邻可表示值的间距约 12.5% 的相对误差。对于矩阵乘法中的大部分元素来说，这足够了：神经网络的权重和激活值本身就是有噪声的近似解。**但有一个致命的例外：outlier。**

考虑一个 $4096 \times 14336$ 的权重矩阵。绝大多数元素在 $[-0.5, 0.5]$ 之间，但有一个元素的值是 100。如果对这个矩阵做统一量化：

1. 计算 scale factor：$s = \max(|W|) / \text{fp8\_max\_e4m3} = 100 / 448 \approx 0.223$
2. 量化：$W_{q} = \text{clamp}(\text{round}(W_{original} / s))$ — 所有值除以 0.223
3. 在正常的 $[-0.5, 0.5]$ 范围内的值，除以 0.223 后变成 $[-2.24, 2.24]$ — 这只能取 4 个整数位置（-2, -1, 0, 1, 2），加上 3 位尾数→约 8×4=32 个不同值。但原来有几百个不同的值——**所有的 fine-grained 信息全部被毁掉了**。

这就是「outlier 污染」——一个 outlier 的存在迫使 scale factor 变得很大，导致正常值被压缩到几个离散的量化色阶中。

### Blockwise/Tilewise Scaling

DeepSeek-V3 的解决方案：**不按整个张量做 scale，而是把张量切成 128×128 的小块（tile），每个 tile 独立选择 scale factor**。

思路非常直观：
- tile A（包含 outlier 100）：$s_A = 100/448 \approx 0.223$。This tile's 正常值被 bad scale factor 污染了——但只影响这一个 tile（共 128×128=16384 个元素中的一小部分）
- tile B（正常范围 [-0.5, 0.5]）：$s_B = 0.5/448 \approx 0.00112$。量化后覆盖 $[-448, 448]$，正常值占据 $[-447, 447]$ 中的约 800 个量化阶——**信息几乎没有损失**

这和 FlashAttention 的 tiling 原理不是同一件事（FlashAttention 的 tiling 是为了减少 HBM 访存量），但在设计哲学上是相通的——**用「局部优化替代全局优化」来避免全局中的单个瓶颈元素拖累整个系统**。

### FP8 在 DeepSeek-V3 的实战

DeepSeek-V3 的 FP8 训练策略：

**哪些操作用 FP8？**
- 所有 `nn.Linear` 的矩阵乘法（GEMM）：权重和激活值都量化到 FP8 → Tensor Core FP8 计算，吞吐倍
- 通信：梯度在 AllReduce 前量化到 FP8 → 通信量减半

**哪些操作保持 BF16/FP32？**
- RMSNorm：逐元素乘加，计算量小，不量化（量化的开销比收益大）
- Softmax：涉及 exp 函数，对数值精度敏感，保持 BF16
- MoE gating：路由决策对概率分布敏感，保持 BF16
- Embedding lookup：查表操作，不涉及 GEMM，保持 BF16
- Residual connections（残差连接）：$x = x + f(x)$——这涉及不同尺度的加法，对精度敏感

**Scale factor 的存储**：每个 128×128 tile 需要一个 FP8 的 scale（1 byte）。对于 $4096 \times 14336$ 的矩阵，需要 $(4096/128) \times (14336/128) = 32 \times 112 = 3584$ 个 scale factors，约占原始数据量（$4096 \times 14336 \times 1$ byte = 56 MB）的 $3584 / (56 \times 10^6) \approx 0.006\%$——可以忽略的 overhead。

**结果**：
- 训练 loss 曲线和 BF16 基线几乎完全重合
- 下游 benchmark（MMLU, HumanEval, GSM8K）精度差异 < 0.5%
- 2024 年在 671B 参数、2048 H800 集群上完成首个生产级 FP8 训练验证
- 训练吞吐相比 BF16 提升约 1.8×（并非 2×，因为部分操作保持 BF16）

### FP8 的硬件依赖性

| GPU 架构 | 代次 | FP8 支持? | 特点 |
|---------|------|----------|------|
| Volta (V100) | 2017 | ❌ | FP16 Tensor Core only |
| Ampere (A100) | 2020 | ❌ | BF16 added, TF32 introduced |
| Hopper (H100/H800) | 2023 | ✅ | 首次支持 FP8 (1979 TFLOPS) |
| Blackwell (B200) | 2024 | ✅ | FP4 added, 4500 TFLOPS FP8 |
| Ada Lovelace (RTX 4090) | 2022 | ❌ | 消费级不支持 FP8 Tensor Core（有 FP8 的 CUDA Core 吗？没有） |

FP8 训练只能在 Hopper 或更新的数据中心 GPU 上进行。A100 不支持 FP8——这是许多团队仍在用 BF16 的原因之一（A100 的装机量远大于 H100）。但到 2025-2026 年，H100/H200/B200 逐渐成为主流后，FP8 训练会从「前沿突破」变为「默认配置」。

## 2.5 精度选择的决策框架

| 场景 | 推荐精度 | 原因 |
|------|---------|------|
| <7B 模型 fine-tune (A100) | BF16 | 不需要 FP8 的额外复杂度 |
| 7-70B 预训练 (H100) | FP8 (GEMM) + BF16 (Rest) | 吞吐收益巨大 (1.8×) |
| 训练中梯度异常 | BF16 | 排除 FP8 量化误差的可能性 |
| 推理 | FP8/INT8 量化 | 不需要训练中的 master weights，极致省显存 |
| 调试/验证计算正确性 | FP32 | 排除精度问题的干扰 |
| 消费级 GPU（RTX 4090） | BF16 | 不支持 FP8 Tensor Core |

### 格式速查速记

| 格式 | 字节/元素 | Tensor Core 速度 (H100) | 数值范围 | 需要 Loss Scaling? | 需要 Master Weights? |
|------|----------|------------------------|---------|------------------|---------------------|
| FP32 | 4 | 67 TFLOPS | ±3.4×10³⁸ | ❌ | ❌ |
| TF32 | 4 | 156 TFLOPS | ±3.4×10³⁸ | ❌ | ❌（免费） |
| FP16 | 2 | 989 TFLOPS | ±6.5×10⁴ | ✅ 需要 | ✅ 需要 |
| BF16 | 2 | 989 TFLOPS | ±3.4×10³⁸ | ❌ 不需要 | ✅ 需要 |
| FP8 (E4M3) | 1 | 1979 TFLOPS | ±448 | ❌ 不需要 | ✅ 需要 |
| FP8 (E5M2) | 1 | 1979 TFLOPS | ±57344 | ❌ 不需要 | ✅ 需要 |

### 本章核心 takeaways

1. **混合精度不省 Model States 显存**——只省激活值显存 + 翻倍计算吞吐
2. **BF16 = FP32 的动态范围 + FP16 的速度**——不需要 loss scaling，是 LLM 训练的默认选择
3. **FP8 = BF16 的吞吐翻倍**——但需要 blockwise scaling + 部分操作保持高精度
4. **Loss Scaling 是 FP16 的「拐杖」**——BF16 不需要它，社区因此从 FP16 转向 BF16
5. **FP8 训练在 2024-2026 从「前沿突破」走向「默认配置」**——DeepSeek-V3 是第一个在千亿级规模验证 FP8 可行性的模型

---

# 第 3 章：GPU 硬件基础

> **核心问题**：GPU 为什么能加速深度学习？Tensor Core 是什么？HBM 和 SRAM 有什么区别？NVLink 和 InfiniBand 的带宽差异为什么决定了并行策略的选择？

## 3.1 GPU 的并行计算模型

### SIMT：单指令多线程

GPU 的核心设计哲学是 **SIMT**（Single Instruction, Multiple Threads）：一个指令被广播给成千上万个线程，每个线程处理不同的数据。这和 CPU 的 MIMD（多指令多数据）有本质区别——CPU 擅长分支、跳转、复杂逻辑，GPU 擅长「对一组数据做同样的运算」。

NVIDIA GPU 的线程层级：
- **Thread**：最基本的执行单元，处理一个标量
- **Warp**：32 个 thread 为一组，同时执行同一条指令（这是 GPU 调度的最小单位）
- **Thread Block**：多个 warp 组成，可共享 Shared Memory
- **Grid**：多个 thread block 组成，对应一个 kernel launch

当一个 warp 中的线程遇到分支（if-else），如果有些线程走 if 分支、有些走 else，GPU 必须串行执行两个分支——这称为 **warp divergence**，是 GPU 编程中最大的性能杀手之一。

### Tensor Core：矩阵乘法的「作弊器」

Tensor Core 是 Volta 架构（V100, 2017）引入的专用硬件单元，专为 4×4×4 的小矩阵乘法设计。它的核心操作是：

$$D = A \times B + C$$

其中 A、B、C、D 都是小矩阵。一个 Tensor Core 在一个时钟周期内能完成 4×4×4 = 64 次乘加 = 128 FLOPs。

H100 有 456 个 Tensor Core（144 个 SM × 4 个 Tensor Core/SM 的一部分？实际是 528 个），配合 FP8 精度，能达到 1979 TFLOPS。作为对比：同样数量 SM 上的 FP32 CUDA Core 只有约 60 TFLOPS。Tensor Core 是 GPU 能被称为「AI 加速器」的根本原因。

### A100 vs H100 关键指标对比

| 指标 | A100 (80GB) | H100 (80GB) |
|------|------------|------------|
| 架构 | Ampere | Hopper |
| FP16/BF16 TFLOPS | 312 | 989 |
| FP8 TFLOPS | 不支持 | 1979 |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s |
| NVLink 带宽 | 600 GB/s | 900 GB/s |
| SM 数量 | 108 | 132 |
| L2 Cache | 40 MB | 50 MB |
| 上市年份 | 2020 | 2023 |

## 3.2 GPU 内存层级

GPU 有多个内存层级，每一层都有自己的容量/带宽/延迟三角均衡：

| 内存类型 | 容量（H100） | 带宽 | 作用域 | 特点 |
|---------|------------|------|--------|------|
| HBM3e | 80 GB | 3.35 TB/s | 全局（所有 SM 共享） | 大但「远」 |
| L2 Cache | 50 MB | ~7 TB/s | 全局 | 中等 |
| Shared Memory | 228 KB/SM | ~20 TB/s | 同一 Thread Block | 快但小，需要手动管理 |
| Register | 65536/SM | ~80 TB/s | 同一 Thread | 最快但极少 |

**关键洞察**：HBM 虽然带宽高达 3.35 TB/s，但相比于 Tensor Core 的计算速度（1979 TFLOPS FP8），它仍然是一个瓶颈。对矩阵乘法来说：

- 计算量：$2 \times M \times K \times N$ FLOPs
- 访存量：$M \times K + K \times N + M \times N$ 个元素（各 1-2 bytes）

计算/访存比 = FLOPs / Bytes。这个比值越高，说明计算密度越大，越不受显存带宽限制。Transformer 的 matmul 通常有较高的计算/访存比，而 element-wise 操作（如 ReLU、Dropout、LayerNorm）的计算/访存比极低——它们几乎纯粹是访存密集型。

### FlashAttention 的硬件启发

FlashAttention 的成功源于一个硬件观察：标准 attention 实现中，$QK^T$ 的结果（$[B, n_h, S, S]$）被写到 HBM，然后马上被读回来做 softmax，再写回 HBM，再读回来乘 V。这三次 HBM 往返是瓶颈。

FlashAttention 的核心创新：**在 SRAM 中分块计算 attention，避免将完整的 attention matrix 写入 HBM**。利用 online softmax 的分块修正公式，只需要在 SRAM 中维护 (m, l, O) 三个小的统计量，最终只把结果写回 HBM。这减少了约 5-10× 的 HBM 访存量。

这个原理在分布式场景中同样适用：Ring Attention（Ch11）做的就是同一件事——把 online softmax 的分块修正推广到「多卡之间的 KV 数据传输」，本质上是用通信（跨 GPU 传 KV）替代访存（跨 HBM 读写）。

### MFU：模型浮点运算利用率

MFU（Model FLOPs Utilization）定义：

$$\text{MFU} = \frac{\text{实际达到的 FLOPs/s}}{\text{GPU 理论峰值 FLOPs/s}}$$

为什么 MFU 通常只有 40-60% 而不是 100%？

1. **非 matmul 操作**：LayerNorm、softmax、attention mask 等操作不用 Tensor Core，效率低
2. **通信开销**：AllReduce 等通信操作期间 GPU 在等待（不做计算）
3. **kernel launch 开销**：每个 CUDA kernel 有启动延迟
4. **内存带宽限制**：小 batch 时数据不够 Tensor Core「吃饱」
5. **流水线泡**：PP 场景下 stage 间的等待

业界参考：Megatron-LM 训练 GPT-3（175B）的 MFU 约 30-35%。DeepSeek-V3 通过 DualPipe + FP8 + 精细的 kernel fusion 达到了 51%——这是目前（2024-2026）的最高水平之一。

## 3.3 节点内与节点间的互联

### 带宽层级

GPU 之间的通信带宽有巨大的数量级差异，这是分布式并行策略选择的**底层物理约束**：

| 互联方式 | 单向带宽 | 典型范围 | 适用场景 |
|---------|---------|---------|---------|
| NVLink 4.0 | 900 GB/s | 单节点 8 卡（通过 NVSwitch 全互联） | Tensor Parallelism |
| InfiniBand NDR400 | 400 GB/s | 跨节点（集群规模） | Pipeline Parallelism, ZeRO |
| PCIe 5.0 ×16 | 64 GB/s | 扩展到 CPU/NIC | 数据加载、Checkpointing，不用于训练通信 |

### NVSwitch：节点内的全互联

H100 节点（DGX H100）通过 NVSwitch 实现节点内 8 张 GPU 的全互联（all-to-all）。每张 GPU 的 NVLink bridge 被 NVSwitch 聚合，任意两张 GPU 之间的带宽都是 900 GB/s（单向）。这意味着节点内的 8 卡可以像一个「大 GPU」一样工作——TP 在这个域内几乎没有通信瓶颈。

**为什么 TP 不能出节点**：跨节点时，数据必须经过 InfiniBand（400 GB/s），带宽只有 NVLink 的 44%。TP 每层都需要 2 次 AllReduce，通信量 ∝ B×S×d。当 batch size 或序列长度增大时，通信时间足以让 GPU 停下来等待——计算/通信比恶化到不可接受的程度。

### 计算/通信比：并行策略的底层约束

以一个 LLaMA-7B FFN 层为例（$d_{model}=4096$，$d_{ff}=11008$，B=1, S=2048）：

- 计算时间（gate_proj matmul）：$\frac{2 \times 1 \times 2048 \times 4096 \times 11008}{989 \times 10^{12}} \approx 0.19 \text{ ms}$

- 激活值大小（输出，bf16）：$1 \times 2048 \times 11008 \times 2 \approx 43 \text{ MB}$
- AllReduce 通信时间：
  - NVLink 内：43MB / 900GB/s ≈ 48 μs → 计算/通信比 ≈ 3.9:1 ✅
  - InfiniBand 跨节点（+overhead）：43MB / (400GB/s/2) ≈ 215 μs → 计算/通信比 ≈ 0.88:1 ❌

跨 InfiniBand 后通信时间接近甚至超过计算时间——GPU 大部分时间在等数据传输。**这就是 TP 为什么只在节点内、PP 为什么能跨节点、ZeRO 为什么偏重跨节点的根本物理原因。**

---

# 第二部分：通信 — 多卡如何对话

# 第 4 章：集合通信原语

> **核心问题**：N 张 GPU 之间如何高效地共享数据？AllReduce、AllGather、ReduceScatter 各解决什么问题？为什么 Ring-AllReduce 能让每张卡的通信量不随 GPU 数量增长？

## 4.1 六种基本集合通信操作

集合通信（Collective Communication）是指一组进程（GPU）共同参与的通信操作。NCCL（NVIDIA Collective Communications Library）实现了以下六种最核心的操作：

### AllReduce

**输入**：每张 GPU 各有一个张量 $x_i$
**输出**：每张 GPU 得到 $\sum_{i=0}^{N_p-1} x_i$（或乘积、最大值、最小值，但 sum 是最常见的）

```
GPU 0:  [1, 2]     →    [1+3+5, 2+4+6] = [9, 12]
GPU 1:  [3, 4]     →    [9, 12]
GPU 2:  [5, 6]     →    [9, 12]
```

**用途**：DDP 中的梯度同步、TP 中 RowParallelLinear 的输出加和。

### AllGather

**输入**：每张 GPU 各有一个张量 $x_i$
**输出**：每张 GPU 得到 $[x_0, x_1, ..., x_{N_p-1}]$ 的拼接

```
GPU 0:  [1]        →    [1, 3, 5]
GPU 1:  [3]        →    [1, 3, 5]
GPU 2:  [5]        →    [1, 3, 5]
```

**用途**：ZeRO-3 前向时拼回完整参数、TP 中 ColumnParallelLinear 的 gather_output。

### ReduceScatter

**输入**：每张 GPU 各有一个张量 $x_i$
**输出**：每张 GPU 得到 sum 结果的第 i 片（大小为 $|x_i|/N_p$）

```
GPU 0:  [1, 2, 3]     →    [1+4+7] = [12]      （第 0 片）
GPU 1:  [4, 5, 6]     →    [2+5+8] = [15]      （第 1 片）
GPU 2:  [7, 8, 9]     →    [3+6+9] = [18]      （第 2 片）
```

**用途**：ZeRO-2 中梯度同步——不需要每张卡都有完整梯度，只需各自的 1/Np 分片。

### Broadcast

**输入**：1 张 GPU 持有数据，其余 GPU 无
**输出**：所有 GPU 持有相同数据

```
GPU 0:  [1, 2, 3]     →    [1, 2, 3]（发）
GPU 1:  [ ]           →    [1, 2, 3]（收）
GPU 2:  [ ]           →    [1, 2, 3]（收）
```

**用途**：模型初始化时将权重从 rank 0 广播给所有 GPU。

### Scatter

**输入**：1 张 GPU 持有完整数据
**输出**：数据被切成 Np 片，每张 GPU 收到 1 片

```
GPU 0:  [1, 2, 3]     →    [1]（自己留第 0 片）
GPU 1:  [ ]           →    [2]（收第 1 片）
GPU 2:  [ ]           →    [3]（收第 2 片）
```

**用途**：数据并行中将 dataset 的不同 shard 分发给各 GPU。

### All-to-All

**输入**：每张 GPU 有 Np 个 chunk，每个 chunk 对应一个目标 GPU
**输出**：每张 GPU 收到来自所有 GPU 的、发给自己的那 Np 个 chunk

```
GPU 0:  [a0, b0, c0]     →    [a0, a1, a2]（发 a0 给 GPU0，收 a1 from GPU1，收 a2 from GPU2）
GPU 1:  [a1, b1, c1]     →    [b0, b1, b2]
GPU 2:  [a2, b2, c2]     →    [c0, c1, c2]
```

**用途**：EP（Expert Parallelism）中的 token dispatch 和 combine、Ulysses 序列并行中的 head 切分。

## 4.2 Ring-AllReduce 算法

Ring-AllReduce 是分布式训练中最核心的通信算法。理解了它，就理解了为什么 DDP 能扩展到上千张 GPU。

### 算法的直觉

把 N 张 GPU 排成一个环。每张 GPU 的数据分成 N 份。整个 AllReduce 分两个阶段：

**阶段 1：Scatter-Reduce（N-1 轮）**。每轮，每张 GPU 把自己当前累积的部分和发给右邻居，同时从左邻居接收对方的累积和。收到后累加到自己的对应分片上。经过 N-1 轮，每张 GPU 拥有完整 sum 的 1/N 分片。

**阶段 2：AllGather（N-1 轮）**。每轮，每张 GPU 把已经得到的最终结果（完整 sum 的分片之一）发给右邻居，同时从左邻居接收对方的分片。经过 N-1 轮，每张 GPU 拥有完整 sum 的所有分片。

### 通信量推导

设每张 GPU 有 D 个元素的数据需要 AllReduce。

- 每轮每张 GPU 发送 D/N 个元素
- 阶段 1：N-1 轮 → 总发送 = (N-1) × D/N
- 阶段 2：N-1 轮 → 总发送 = (N-1) × D/N
- 每张 GPU 总发送 = $2\frac{N-1}{N} D$

当 N → ∞ 时，$(N-1)/N → 1$，因此每张 GPU 发送 ≈ 2D。

**关键结论**：每张 GPU 发送的数据量几乎不随 N 增长——始终约等于自己数据的 2 倍。总通信量 = 2(N-1)D ≈ 2ND 分布到 N 张卡上。（注意：总通信量确实随 N 增长，但每张卡的发送量 ≈ 2D 是恒定的。在 N=4 时是 1.5D，N=64 时是 1.97D。这就是 Ring-AllReduce 的 **bandwidth-optimal** 性质——每张卡带宽压力恒定，不像参数服务器那样 O(N) 增长。）

### 与参数服务器（Parameter Server）的对比

参数服务器是一个中心化的架构：所有 worker 把梯度发给 parameter server（PS），PS 更新后把新参数发回。PS 的入站带宽 = N × D。当 N 大时，PS 成为瓶颈。

Ring-AllReduce 是去中心化的：没有单一节点承载所有通信压力。这就是为什么现代深度学习训练几乎不用参数服务器——除非在特殊场景（如联邦学习、异步 SGD）。

### Ring vs Tree

NCCL 内部不只用 Ring。对于大消息（> 256KB），Tree 算法（基于树形拓扑的 reduce+broadcast）有更低的延迟——log(N) 轮 vs N 轮。NCCL 会根据消息大小和 GPU 拓扑自动选择最优算法。用户通常不需要手动调，但了解这个差异有助于解读 NCCL 的 profiling 输出。

## 4.3 通信与计算的重叠

通信最大的代价不是带宽，而是 GPU 在等通信完成期间不能做计算——这是纯 idle 时间。

**异步通信**：`isend/irecv`（i 代表 immediate，即非阻塞）允许 GPU 在数据传输进行的同时做计算。DDP 的 `backward()` 中，一个 bucket 的梯度算好后立即发起异步 AllReduce，同时继续反向传播计算下一个 bucket 的梯度——通信和计算被重叠（overlap）。

**梯度 bucketing**：DDP 不是每个参数的梯度单独 AllReduce（那样有大量的小消息 + 大量通信发起开销），而是把参数按反向传播的倒序分成若干个 bucket（通常按层分组），每个 bucket 算完才发起一次 AllReduce。

**DualPipe 的极致重叠**：DeepSeek 的 DualPipe（Ch9）把通信重叠做到极致——不仅重叠计算和通信，还让前向和反向的通信在相反方向同时进行，像双向车道一样永不空闲。这使得 GPU 利用率从 ~61% 提升到 ~92%。

---

# 第 5 章：分布式训练基础设施

> **核心问题**：怎么启动一个分布式的 PyTorch 训练任务？进程怎么互相发现？训练中途 GPU 挂了怎么办？怎么知道瓶颈在哪？

## 5.1 进程启动与初始化

### torchrun

`torchrun` 是 PyTorch 提供的分布式训练启动工具。最基本的用法：

```bash
torchrun --nproc-per-node=8 --nnodes=4 train.py
```

这会在当前节点启动 8 个进程，并期望总共 4 个节点（共计 32 个进程）。torchrun 内部处理了每个进程的 RANK/WORLD_SIZE 分配。

### 关键环境变量

每个进程通过环境变量获知自己的身份：

| 变量 | 含义 | 示例 |
|------|------|------|
| WORLD_SIZE | 总进程数（= GPU 总数） | 32 |
| RANK | 全局进程编号（0 到 WORLD_SIZE-1） | 5 |
| LOCAL_RANK | 本节点内进程编号（0 到 NPROC_PER_NODE-1） | 5 |
| MASTER_ADDR | rank 0 所在节点的 IP | 10.0.0.1 |
| MASTER_PORT | rank 0 用于 rendezvous 的端口 | 29500 |

### init_process_group

在训练代码中，需要初始化进程组：

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",           # 通信后端：nccl(GPU) / gloo(CPU) / mpi
    init_method="env://",     # 从环境变量读取配置
    world_size=int(os.environ["WORLD_SIZE"]),
    rank=int(os.environ["RANK"])
)
```

`init_method="env://"` 是最常用的方式——torchrun 把必要信息写到环境变量，`init_process_group` 从那里读。

## 5.2 容错与检查点

### 分布式检查点保存什么

单卡训练的检查点只需要 model.state_dict() + optimizer.state_dict()。分布式训练还需要：

- 每个 rank 各自的 optimizer state shard
- RNG states（保证恢复后数据 shuffle 顺序一致）
- Learning rate scheduler 的状态
- Data iterator 的位置

PyTorch 提供了 `torch.distributed.checkpoint`（DCP）来处理分布式保存和加载。相比于直接 `torch.save`，DCP 知道如何在 ZeRO/FSDP 分片场景下正确地分片存储和重组。

### 常见故障模式

| 故障 | 症状 | 恢复方式 |
|------|------|---------|
| GPU OOM | RuntimeError: CUDA out of memory | 减小 batch size 或 micro-batch 数，启用 AC |
| NaN loss | loss 突然变成 NaN | 从最近检查点恢复，降低 learning rate |
| NCCL 超时 | NCCL WATCHDOG timeout | 增大超时阈值或排查网络问题 |
| 硬件故障 | GPU ECC error / NVLink 降级 | 跳过故障 GPU，从检查点恢复（elastic training） |

现代分布式训练的典型恢复流程：检测到故障 → 所有进程终止 → 启动新 job（可能跳过坏节点）→ 从最新的分布式检查点恢复 → 继续训练。

## 5.3 性能剖析

### 怎么做 profiling

`torch.profiler` 是 PyTorch 自带的性能分析工具：

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
) as prof:
    for step in range(10):
        train_one_step()
        prof.step()
```

产生的 trace 可以用 TensorBoard 或 Chrome://tracing 打开。在 timeline 视图中可以清晰地看到：
- GPU kernel 的执行时间（哪些 kernel 占时最多）
- NCCL AllReduce 的通信时间（GPU 空闲 + 数据传输）
- CPU 和 GPU 之间的同步点（stream synchronization）

### MFU 的计算

$$\text{MFU} = \frac{\text{实际 FLOPs/s}}{\text{理论峰值 FLOPs/s}}$$

实际 FLOPs 可以通过 `torch.profiler` 的 `_op_exact_compute_flops` 获取（近似），或自己用 6ND 公式估算训练一定步数需要的总 FLOPs 除以耗时。理论峰值查 GPU 规格表。

例如：H100 bf16 理论峰值 = 989 TFLOPS。如果 8 张 H100 训练 7B 模型，每秒处理 10000 tokens：
- 每 token 2 forward FLOPs = 2 × 7B = 14G FLOPs（近似）
- 总 FLOPs/s = 10000 × 14G × 3（fwd+bwd 约 3×）= 420 TFLOPS
- 理论峰值 = 8 × 989 = 7912 TFLOPS
- MFU = 420/7912 = 5.3% ——这说明训练严重 I/O 瓶颈（比如小 batch + 大通信开销 + 数据加载慢）

好的分布式训练 MFU 通常在 30-55% 之间。低于 20% 说明有优化空间，高于 50% 是行业顶尖（DeepSeek-V3 的 51% MFU 至今是标杆）。

---

# 第三部分：并行策略 — 五种切分维度

# 第 6 章：并行的分类学

> **核心问题**：到底有多少种方法把一个模型分布到多张 GPU 上？每种方法在切什么？它们之间是什么关系？

## 6.1 五种切分维度的统一框架

分布式训练的核心思想非常简单：**把一个放不进单卡的模型/数据，切成多份，每张 GPU 处理其中的一部分，然后通过通信来协调**。不同的并行策略，区别只在于「沿着哪个维度切」。

要理解这五种并行，一个直观的类比是团队写一本书：

- **数据并行（DP）**：每人拿到所有书页的完整副本，每人写不同的章节（不同的数据 batch），写完一章后大家碰头同步（AllReduce 梯度）。每个 writer 都拿着整本书，太重了——但方式最简单。
- **张量并行（TP）**：把一页纸上的矩阵乘法拆成两半——一个人算上半页，一个人算下半页，算完拼在一起。适用于一页太大放不进一个人的桌子。
- **流水线并行（PP）**：把书的章节分给不同的人——Alice 写第 1-8 章，Bob 写第 9-16 章。Bob 必须等 Alice 写完第 8 章才能开始。空闲等待时间就是「泡」。
- **序列并行（SP/CP）**：把一句话切成两段给两个 writer——一个处理「猫坐在垫子上」，另一个处理「狗坐在沙发上」。但注意力机制要求看到整句话，所以需要传信息。
- **专家并行（EP）**：如果一本书有 8 个不同的副主题（experts），每篇文章只需要其中 2 个 expert 来处理。让 4 个 writer 各专精 2 个 expert，文章通过路由送到正确的 writer。

| 并行类型 | 缩写 | 切分对象 | 切分维度 | 典型通信操作 | 通信量 |
|---------|------|---------|---------|------------|--------|
| 数据并行 | DP | 数据批次 | batch 维 | AllReduce | 2N bytes/GPU（参数和梯度不切时） |
| 张量并行 | TP | 单层矩阵 | 行/列 | AllReduce（2次/层） | B×S×d（每层 AllReduce 两次） |
| 流水线并行 | PP | 层序列 | 层深度 | P2P send/recv | (P-1)×B×S×d（总共传 P-1 次） |
| 序列并行 | SP/CP | 注意力矩阵 | 序列长度维 | ReduceScatter+AllGather 或 All-to-All 或 环形 send/recv | 4×S×d_head（Ring Attention 每卡） |
| 专家并行 | EP | MoE experts | expert 集合 | All-to-All | 取决于 expert 分布和 token 路由 |

**关键理解**：TP、PP、ZeRO（FSDP）是同一种问题（模型太大放不进单卡）的三种不同解法。TP 和 PP 是把模型「物理上切成多份」，ZeRO 是把模型状态「逻辑上分布开」——使用时不切模型，只在通信时切 optimizer states/梯度/参数。它们可以组合使用——事实上，现代大规模训练几乎都用 3D 甚至 4D 并行。

### ZeRO 属于哪种并行？

ZeRO/FSDP 本质上是对 Data Parallelism 的增强：它仍然是每卡用自己的数据 batch，每卡做完整的前向/反向计算，但在通信和存储上做了 sharding。所以 ZeRO 不被列为独立的「第六种并行」，而是 **DP 的 sharding 优化**。但在实践中，ZeRO 和 TP/PP 被平级对待，因为它解决的是同一根本问题（显存不足），只是从不同的角度（优化器状态 vs 矩阵 vs 层序列）。

## 6.2 五种并行的适用域速览表

| 并行类型 | 适用域 | 典型规模 | 主要限制 |
|---------|--------|---------|---------|
| DP | 通用 | 7B→任意 | 模型状态不能被切（ZeRO=DP+sharding） |
| TP | 节点内（NVLink） | ≤ 8 GPUs | NVLink 域、heads 数、通信量 ∝ B×S×d |
| PP | 跨节点 | 层数 ≥ stages 数 | 泡、激活值存储 ∝ M、负载不均 |
| ZeRO/FSDP | 通用（跨或节点内） | 7B→175B | ZeRO-3 通信 ×1.5，需求 NVMe offload 对 175B+ |
| SP/CP | 长上下文（128K+） | 支持序列长度超 1M | 通信量大、causal 负载不均 |
| EP | MoE 模型 | N_experts ≥ 8 | expert 负载不均、All-to-All overhead |

## 6.3 3D/4D 并行的概念

实际大规模训练中，几乎不存在只使用一种并行的场景。常见的组合方式：

- **2D (TP + DP)**：节点内 TP=4-8，节点间 DP。适用于 13-70B 模型
- **3D (TP + PP + DP)**：TP 节点内切矩阵，PP 跨节点切层，DP 在最外层。适用于 175B+ dense 模型
- **4D (TP + PP + EP + DP)**：在 3D 基础上，MoE 的 expert 层用 EP。典型的 DeepSeek-V3 配置：TP=4, PP=16, EP=64, DP=2 → 2048 H800
- **5D (+CP)**：在以上基础上，长序列用 Context Parallelism 切序列维

DeviceMesh（PyTorch DTensor 中的核心概念）正是用来表示这种多维拓扑的：一个 $N$ 维 mesh，每维对应一种并行策略的 process group。例如，一个 2×4 的 2D mesh 可以表示 TP(2) + DP(4)：2 张卡做 TP（同一个 tp_group），4 个 tp_group 之间做 DP。

---

# 第 7 章：数据并行 — 最朴素、最重要的基线

> **核心问题**：每张 GPU 独立算自己的数据，然后同步梯度——这个最简单的并行方式为什么有效？什么时候失效？

## 7.1 DDP 的工作原理

### DataParallel vs DistributedDataParallel

PyTorch 有两个「数据并行」实现：

- **`nn.DataParallel` (DP)**：单进程多线程。主 GPU 负责汇总梯度、广播参数。有 Python GIL 限制、主 GPU 显存压力大、通信效率低。**不要用**。
- **`DistributedDataParallel` (DDP)**：多进程（每个 GPU 一个进程）。梯度同步通过 NCCL AllReduce 在所有 GPU 之间直接进行，无主从之分。**唯一正确的选择**。

### DDP 的一步训练流程

```
1. 每个 GPU 加载不同的 micro-batch（DistributedSampler 保证不重叠）
2. 各自做 forward：loss = model(batch_i)
3. 各自做 backward：loss.backward() → 本地计算梯度
4. AllReduce(梯度)：所有 GPU 的梯度被平均（不是求和！默认 DDP 做 average）
5. 各自做 optimizer.step()：因为梯度已同步，所有 GPU 的模型更新一致
```

第 4 步的「平均」很关键：如果 DDP 做的是 sum 而非 average，那么 global batch size 会影响 loss scale。平均保证了 loss 不随 GPU 数变化。

### Gradient Bucketing 与通信计算重叠

DDP 不是等 backward 全部完成再一次性 AllReduce。Backward 是逆序计算各层梯度的。DDP 按反向传播的顺序，把参数分成若干个 bucket。每当一个 bucket 内的所有参数的梯度都算完了，立即发起这个 bucket 的异步 AllReduce——同时继续算下一个 bucket 的梯度。

这实现了 **通信与计算的重叠**：GPU 在算 bucket N+1 的梯度时，bucket N 的 AllReduce 在网络中进行。这是 DDP 相比 DP 快的关键原因之一。

### DDP 的 autograd hook 机制

DDP 之所以能「自动」在 backward 过程中触发梯度同步，依赖的是 PyTorch autograd 的 hook 系统。具体流程：

1. **初始化阶段**：DDP 构建 `Reducer` 对象，遍历模型的所有参数，按反向传播的逆序将它们分组为 buckets
2. **注册 hook**：对每个参数的 `.grad` 属性注册 autograd hook。当 backward 计算到该参数的梯度时，hook 被触发
3. **Bucket 就绪检测**：hook 中检查当前 bucket 是否所有参数的梯度都已计算完。如果就绪，触发异步 AllReduce
4. **异步执行**：AllReduce 在独立的 CUDA stream 上执行，主计算 stream 继续反向传播而不等待

这里的巧妙之处在于 bucket 的顺序必须和 autograd 的计算顺序一致——如果 bucket 按 forward 顺序排列，那么 backward 时最后一个 bucket 的梯度先算完（因为反向从最后一层开始），导致前几个 bucket 一直在等它们的梯度。DDP 的 `Reducer` 在构建时故意 reversed 了 bucket 的顺序，使得最早算完的梯度所在的 bucket 最先被触发 AllReduce。

一个具体的例子：假设模型有 4 层，参数分组为 4 个 bucket：

```
Layer 1 (embedding)  → bucket 3（最后被触发——它的梯度最后算完）
Layer 2 (attention)  → bucket 2
Layer 3 (ffn)        → bucket 1
Layer 4 (lm_head)    → bucket 0（最先被触发——backward 从最后一层开始）
Backward 开始 ↓
  → lm_head 梯度算完 → hook 触发 → bucket 0 就绪 → 异步 AllReduce(bucket 0)
    （同时继续算 FFN 的梯度）
  → FFN 梯度算完 → bucket 1 就绪 → 异步 AllReduce(bucket 1)
    ...
```

理想情况下，AllReduce(bucket 0) 的通信时间被 bucket 1 的梯度计算时间完全遮盖（overlap），GPU 从不空闲。

### DDP 的全局 batch 计算

DDP + Gradient Accumulation 组合时，有效全局 batch 的公式为：

$$B_{global} = micro\\_batch\\_size \\times accumulation\\_steps \\times N_{dp}$$

其中 $N_{dp}$ 是 DP group 中的 GPU 数。例如：micro_batch=2, ga=4, Ndp=32 → global_batch = 2×4×32 = 256 samples/step。

## 7.2 梯度累积：用时间换 batch size

当单卡甚至装不下一个大 batch 的激活值时，梯度累积把一个大的 effective batch 切成 M 个 micro-batch：

```
for micro_step in range(accumulation_steps):
    batch_i = get_micro_batch()
    loss = model(batch_i) / accumulation_steps  # 注意除以 M！
    loss.backward()  # 梯度在本地 buffer 中累积

# M 步累积完后，一次性 AllReduce + optimizer.step
```

- peak memory：一个 micro-batch 的激活值大小
- effective batch_size = micro_batch_size × accumulation_steps × DP_size
- 代价：需要串行跑 M 步（时间增加 M×，但这是线性增加的，因为每步前向计算量相同）

## 7.3 大 batch 训练的挑战

当 DP 数很大时（例如 256 GPU），即使 micro_batch_size=1，effective batch_size 也可能达到 256 甚至更高（加 GA）。这带来了几个问题：

### 学习率线性缩放法则

经验法则：当 batch_size 翻倍时，学习率也翻倍。即：

$$lr_{new} = lr_{base} \times \frac{B_{new}}{B_{base}}$$

物理直觉：大 batch 的梯度噪声更小（平均了更多样本的梯度），可以走更大的步长。GPT-3（175B）用了 3.2M tokens 的 global batch_size，学习率 = 6×10⁻⁴。

### Warmup

训练初期，模型权重是随机的，梯度的方差很大。直接用大学习率会导致训练发散。Warmup 的做法是：前几千步逐步将 LR 从 0 升到目标值（线性或 cosine warmup），让优化器有机会「适应」梯度的尺度。

### 泛化差距

OpenAI 和 DeepMind 的研究（Keskar et al., 2017; McCandlish et al., 2018）发现：当 batch_size 超过某个「临界值」后，继续增大 batch_size 虽然在训练 loss 上可能持平甚至更优，但验证 loss（泛化性能）会变差。这就是「大 batch 泛化差距」。

临界 batch size 的值因任务而异：语言模型约为 2-8M tokens，图像分类约为数千到数万 sample。LAMB 和 LARS 优化器专门为大 batch 设计，通过逐层自适应学习率来缓解这个问题。

## 7.4 梯度压缩

当 DP 数很大时（256+ GPU），AllReduce 的通信量虽然仍然是 2N，但 N（模型参数量）本身很大（175B 的梯度 = 350GB）。梯度压缩以减少通信量为目标。

### 1-bit SGD

Seide et al. (2014) 提出：梯度只需要保留符号（正或负，1 bit），加上 error feedback（把量化残差加到下一次梯度中）：

```
g_compressed = sign(g + error)
error = g + error - g_compressed  # 补偿量化残差
AllReduce(g_compressed)  # 通信量降低 32×（从 32 bits → 1 bit）
```

DeepSpeed 1-bit Adam 把这个思路用于 Adam 优化器：在 AllReduce 阶段用 1-bit 压缩，在 optimizer step 阶段使用全精度。通信量减少约 5×，对收敛几乎没有影响。

---

# 第 8 章：张量并行 — 把一层矩阵切成多份

> **核心问题**：如果一层 Linear(4096, 14336) 的权重矩阵太大，能不能把它切成两块放在两张 GPU 上？怎么切才能让通信最少？

## 8.1 ColumnParallel 与 RowParallel

Megatron-LM（Shoeybi et al., 2019）提出了两种基础的 TP 切分模式。本节的推导以 TP=2 为例。

### ColumnParallelLinear

把权重矩阵 $W \in \mathbb{R}^{d_{in} \times d_{out}}$ **沿列切成 2 份**：

$$W = [W_{col}^{(0)} \;|\; W_{col}^{(1)}], \quad W_{col}^{(i)} \in \mathbb{R}^{d_{in} \times d_{out}/2}$$

每张 GPU 拿到相同的输入 $X$，各自做局部矩阵乘法：

$$Y_{partial}^{(0)} = X \cdot W_{col}^{(0)}, \quad Y_{partial}^{(1)} = X \cdot W_{col}^{(1)}$$

$Y_{partial}^{(i)}$ 的形状是 $[B, S, d_{out}/2]$。要得到完整输出 $Y = [Y_{partial}^{(0)} | Y_{partial}^{(1)}]$，需要将两份 partial 沿最后一维拼起来。

- 如果 `gather_output=True`：在 partial 输出后做 AllGather，拼成完整 $Y$。这对应 Attention 的 QKV 投影（后面需要完整的 hidden_states 做 attention scores）。
- 如果 `gather_output=False`：不通信，直接返回 partial $Y_{partial}^{(i)}$。这对应 MLP 中的 gate_proj 和 up_proj——它们的输出会直接送给下一层 RowParallel，不需要拼成全量。

### RowParallelLinear

把权重矩阵 $W \in \mathbb{R}^{d_{in} \times d_{out}}$ **沿行切成 2 份**：

$$W = \begin{bmatrix} W_{row}^{(0)} \\ W_{row}^{(1)} \end{bmatrix}, \quad W_{row}^{(i)} \in \mathbb{R}^{d_{in}/2 \times d_{out}}$$

输入 $X$ 也沿最后一维切成 $X^{(0)}, X^{(1)}$。每张 GPU 做局部矩阵乘：

$$Y_{partial}^{(0)} = X^{(0)} \cdot W_{row}^{(0)}, \quad Y_{partial}^{(1)} = X^{(1)} \cdot W_{row}^{(1)}$$

$Y_{partial}^{(i)}$ 的形状是 $[B, S, d_{out}]$。但 $Y_{partial}^{(0)}$ 和 $Y_{partial}^{(1)}$ 各自只包含了一半列维的贡献，真实输出 $Y = X \cdot W = X^{(0)}W_{row}^{(0)} + X^{(1)}W_{row}^{(1)} = Y_{partial}^{(0)} + Y_{partial}^{(1)}$。所以做一次 **AllReduce(SUM)** 把两张 GPU 的 partial 输出加起来。

## 8.2 Megatron 的 MLP 配对设计

SwiGLU MLP 的完整前向：

$$Y = \text{SiLU}(X \cdot W_{gate}) \odot (X \cdot W_{up}) \cdot W_{down}$$

在 TP=2 下：

1. **Gate & Up**：ColumnParallelLinear，`gather_output=False`
   - 每卡输出 $Y_{gate}^{(i)}$ 和 $Y_{up}^{(i)}$，形状 $[B,S,d_{ff}/2]$
2. **SiLU & element-wise 乘法**：本地操作，形状不变
   - $Y_{gated}^{(i)} = \text{SiLU}(Y_{gate}^{(i)}) \odot Y_{up}^{(i)}$，形状 $[B,S,d_{ff}/2]$
3. **Down**：RowParallelLinear
   - 输入恰好是 $Y_{gated}^{(i)}$，形状 $[B,S,d_{ff}/2]$——这正是 RowParallel 需要的列切分输入！
   - 每卡算 $Z_{partial}^{(i)} = Y_{gated}^{(i)} \cdot W_{down}^{(i)}$，形状 $[B,S,d_{model}]$
   - AllReduce(SUM) → 完成

**关键洞察**：ColumnParallel(gather_output=False) → RowParallel 这个配对，中间**不需要任何通信**。ColumnParallel 的 partial 输出沿 d_ff 维切分，正好是 RowParallel 的列切分输入。如果 ColumnParallel 做了 AllGather（拼成完整的 d_ff 维），再送给 RowParallel 前又需要切分——白白浪费一次通信，且语义上完全没有必要。

这个设计是 Megatron-LM 最精妙的地方：它不是简单地把每一层都切一遍然后所有层间都 AllReduce，而是通过「配对」来省掉一半的通信。

## 8.3 完整 Transformer Block 的 TP 布局

以 LLaMA 的 decoder block 为例（RMSNorm + Attention + FFN + 两次 Residual）：

```
Input [B, S, d_model]  ← 已经是完整副本（上一层 AllReduce 后）
    │
    ▼
RMSNorm  ← 不切！计算量极小且输入已是完整副本
    │
    ▼
┌─ Attention ──────────────────────────────────────────┐
│ Q: ColumnParallel(gather_output=True) → [B,S,d_model]   ← 每卡只有部分 heads
│ K: ColumnParallel(gather_output=True) → [B,S,d_model]   ← GQA 下 KV heads ≠ Q heads 时注意！
│ V: ColumnParallel(gather_output=True) → [B,S,d_model]
│      │
│   各自算 local heads 的 Scaled Dot-Product Attention
│      │
│ O: RowParallel → AllReduce → [B,S,d_model]           ← 第 1 次 AllReduce
└──────────────────────────────────────────────────────┘
    │
    + Residual (各自加，因为 O 已经 AllReduce 同步过)
    │
    ▼
RMSNorm  ← 不切
    │
    ▼
┌─ FFN (SwiGLU) ──────────────────────────────────────┐
│ Gate: ColumnParallel(gather_output=False) → [B,S,d_ff/2]
│ Up:   ColumnParallel(gather_output=False) → [B,S,d_ff/2]
│      │
│   SiLU(Gate) ⊙ Up  (element-wise, 本地)
│      │
│ Down: RowParallel → AllReduce → [B,S,d_model]        ← 第 2 次 AllReduce
└──────────────────────────────────────────────────────┘
    │
    + Residual
```

整个 Block 仅 **2 次 AllReduce**（Attention Output + FFN Down），每 次传输 $B \times S \times d_{model} \times 2$ bytes（bf16）。

## 8.4 TP 的边界

### 为什么 TP 不超过 8

TP 有三个硬上限：

1. **Attention heads 数**：每个 GPU 必须至少拥有 1 个完整的 attention head（head 不能切！否则 attention scores 的计算会涉及跨 GPU 通信，极其复杂）。TP ≤ N_q_heads。LLaMA-7B 有 32 个 Q heads，TP 最多 32——这个限制不紧。但 GQA 模型（如 LLaMA 3 70B，Q=64 heads, KV=8 heads）下，TP 超过 8 时某些卡就没有 KV heads 了。

2. **GEMM 尺寸不能太小**：TP=8 时，每卡只分到 d_model/8 = 512 维的输出。一个 $[1, 2048, 4096] \times [4096, 512]$ 的 matmul 只有约 8.6M FLOPs，不到 H100 理论峰值 1 微秒的量——GPU 的 tensor core 根本「吃不满」。每卡矩阵太小 → GPU 利用率低。

3. **通信量 ∝ B×S×d_model**：大 batch / 长序列时，AllReduce 传输的激活值变大。虽然 NVLink 带宽足够（900 GB/s），但 AllReduce 的延迟（启动开销）也是 O(log₂N) 的增大。通常 TP=8 是 H100 单节点（8 卡 NVSwitch）的甜点。

### 推理场景的退化

推理时 batch_size=1, seq_len≈1（decode 阶段逐 token 生成）：

- 每卡 GEMM：$1 \times 1 \times 4096 \times 2048 \approx 16M$ FLOPs → < 1 μs
- 2 次 AllReduce：各 $1 \times 1 \times 4096 \times 2 \text{ bytes} \approx 8 \text{ KB}$ → < 10 ns on NVLink

不是通信慢——是**计算太小**。GPU 的 SM 根本没装满，且通信的 kernel launch 开销比计算本身还大。这就是为什么推理生成阶段（decode）不用 TP，而用 ZeRO-3（让每卡全量做 GEMM）或 PP（跨层切）。Prefill 阶段（长序列、一次处理所有 prompt tokens）TP 依然有意义——因为 S 很大，GEMM 足够饱满。

---

# 第 9 章：流水线并行 — 把层序列切成多段

> **核心问题**：TP 在跨节点时通信吃不消，怎么办？把模型的 32 层分成 4 段，每段 8 层给一张 GPU。只有层边界才需要传数据——这就是流水线并行。

## 9.1 动机

回看 Ch8 的结论：TP 每次 AllReduce 通信量 ∝ B×S×d_model，跨 InfiniBand（400 GB/s vs NVLink 900 GB/s）时计算/通信比恶化到不可接受。

PP 的解决方案非常直观：不切单层矩阵，而是切整层。每张 GPU 持有连续的若干层。前向时，GPU_i 算完自己的层后，把最后一层的激活值发给 GPU_{i+1}，作为后者的输入。反向时反过来传梯度。通信只在层边界进行，总通信量 = (P-1) × B×S×d_model——不随层内计算量增长。

注意：PP 和 TP 的通信量公式基本相同（都传 B×S×d_model 的激活值），但 TP **每层都传**（32 层 = 32 × 2 × B×S×d_model），而 PP 只传 **P-1 次**。P=4 时 TP 通信量是 PP 的 32×2/3 ≈ 21 倍。

## 9.2 GPipe 与泡率

### Micro-batch 机制

PP 的一个关键设计是将一个 batch 切成 M 个 micro-batch。第 i 个 micro-batch 不需要等第 i-1 个 micro-batch 走完整个 pipeline，而是可以「追随」前一个 micro-batch 进入流水线——就像工厂流水线，第 1 个产品还在第 3 站装配时，第 2 个产品已经进入第 2 站了。

### GPipe 调度

GPipe（Huang et al., 2019）是最简单的调度方式：

1. **全部前向**：每个 GPU 依次处理所有 M 个 micro-batch，前向结束后传给下一 stage
2. **全部反向**：全部前向完成后，从最后一个 stage 开始反向，依次往前
3. **统一更新**：所有反向完成后，各卡 AllReduce 梯度，optimizer.step

**泡的来源**：全部前向阶段，前面的 GPU 处理完 M 个 micro-batch 后，必须**等待**后面的 GPU 完成。等反向开始后，前面的 GPU 又在等反向从后往前传。每个 GPU 的工作区间因此存在大量「空闲」（bubble），像一个时段只有少数 GPU 在真正干活。

**泡率公式**：

$$\text{Bubble} = \frac{P-1}{P-1+M}$$

- 当 M=1 时，泡率 = (P-1)/P → 接近 100%（几乎完全串行）
- 当 M >> P 时，泡率趋近于 0（流水线被充分填满）
- 典型配置：P=4, M=8 → 泡率 = 3/11 ≈ 27%

增大 M 减少了泡率，但代价是显存：每个 GPU 必须同时保存 M 个 micro-batch 的前向激活值。**M 不能无限大——显存墙又出现了。**

### 泡率公式的完整推导

设每个 micro-batch 的前向时间为 $f$，反向时间为 $b$（通常 $b \approx 2f$），总 micro-batch 数为 $M$，pipeline stage 数为 $P$。

GPipe 调度下，时间线如下：
- **前向阶段**：GPU 0 先处理所有 M 个 micro-batch 的前向（耗时 Mf），输出传给 GPU 1。GPU 1 在 GPU 0 完成第一个 micro-batch 后开始（延迟 f 开始），也需要 Mf。依此类推。第 P 个 GPU 的前向开始时间为 (P-1)f，总前向耗时 = (P-1+M)f。
- **反向阶段**：前向完毕后，GPU P-1 开始反向（耗时 Mb）。GPU P-2 在 GPU P-1 完成第一个 micro-batch 的反向后开始。总反向耗时 ≈ (P-1+M)b。
- **总时间** ≈ $(P-1+M)(f+b)$。有效计算时间（无泡）应为 $M(f+b)$（如果不考虑 GPU 间等待）。

因此泡 = 总时间 - 有效时间 = $(P-1+M)(f+b) - M(f+b) = (P-1)(f+b)$。泡率 = $(P-1)/(P-1+M)$。

以具体数字验证：P=4, M=8 → 泡率 = 3/11 ≈ 27.3%。这意味着整个训练过程中，约 27% 的 GPU 时间花在等待上。如果 M=32，泡率 = 3/35 ≈ 8.6%。如果 M=128，泡率 = 3/131 ≈ 2.3%。

但 M=128 意味着每张 GPU 需要保存 128 个 micro-batch 的前向激活值。以 LLaMA-7B 为例，每 micro-batch 的激活值约为 8-12GB / M。当 M=8 时每步激活值约 1-1.5GB，M=128 则约 16-150MB per micro-batch。总激活值存储 = M × 每步激活值 ≈ M × (total_activations / M) = total_activations。所以实际上，M 增大时每步的激活值变小（因为 micro-batch 更小），但所有 micro-batch 的激活值总和大致不变——因为总计算量相同。

**泡率公式的误导**：公式 (P-1)/(P-1+M) 看起来增加 M 就能无限降低泡率。但在实践中存在隐式上限——M 的上限来自 (1) 总 batch size 不能无限大（泛化性能会下降）；(2) 显存必须能 hold 住所有 micro-batch 的激活值（如果 micro-batch 太小，每步的 overhead 占比升高）。大多数实践中的 M 在 8-64 之间。

## 9.3 填缝优化史

### 1F1B（One-Forward-One-Backward，PipeDream, 2018）

不需要等所有前向完成。前向跑够 P 步（建立流水线的最小步数）后，每完成一次前向立即插入一次反向（交替进行）。

对比 GPipe（P=4, M=8）：
- GPipe：总时间 = P+M-1=11 单位，泡 = P-1=3 单位 → 泡率 27%
- 1F1B：总时间 = P+M-1=11 单位，泡更少 → 泡率 ~15%

核心改进：GPipe 像「先所有前向，再所有反向」的单向公路（白天进城、晚上出城），1F1B 像「交错通行」——任何时候都有前向和反向往来。

### Interleaved 1F1B（Megatron-LM 2, 2021）

每层再切成 chunks。例如，每层切 2 个 chunk → P=4 stages → 总阶段数 = 4×2 = 8。更细粒度的阶段 → 每个 GPU 的等待间隔更短 → 泡率进一步降到 ~5%。

代价：更细的 chunks = 更频繁的层边界通信 + 更复杂的状态管理（需要维护每个 chunk 的激活值和中间梯度）。

### Zero-Bubble（Qi et al., ICLR 2024）

反向传播分为两个阶段：
- **B（参数梯度计算，Backward w.r.t. Parameters）**：需要激活值，且需要通信（AllReduce 梯度）
- **W（输入梯度计算，Backward w.r.t. Inputs）**：需要激活值，但**不需要通信**（只计算链式法则传给上一层的梯度，本地操作）

ZB 的调度让 W 阶段填满原本是泡的时间 slot。结果：泡率 < 1%。

### DualPipe（DeepSeek-V3, 2024）

让两条数据流从 pipeline 两端**同时对向注入**。流 A 从左端注入 micro-batches，做前向（左→右）；流 B 从右端注入 micro-batches，做反向（右→左）。两条流在中间相遇时交叉进行——GPU_i 在做流 A 的前向的同时，流 B 的反向数据从 GPU_{i+1} 传过来。

这就像双向车道：前向车从左往右开，反向车从右往左开，每辆车在同一个时刻可能在不同的 GPU 上被处理。通信（右→左的梯度传输）和计算（左→右的前向）被完全重叠——GPU 几乎没有任何空闲时间。

结果：DualPipe 使得 DeepSeek-V3 在 2048 H800 上的 **MFU 达到 51%**，远超同期 GPT-4 训练的 ~35% 和 Llama 3 405B 的 ~38%。

## 9.4 PP 的工程代价

1. **激活值存储**：每个 GPU 需要保存 M 个 micro-batch 的所有前向激活值（用于反向）。M 不能太小（否则泡大）、不能太大（否则显存炸）。通常在 M=8-64 之间微调。Activation Checkpointing 是标配——每 N 层存一个 checkpoint，反向时重算中间激活。
2. **Stage 间负载不均**：LLaMA-7B 的 32 层并不完全相同：Attention 和 FFN 的 FLOPs 不同，且第一层和最后一层通常有额外操作（embedding、lm_head）。简单的「等分层数」会导致有的 stage 比其他慢 → 全体等最慢的 stage。
3. **与 TP 的配合**：在实际的 3D 并行中，PP 通常跨节点使用，TP 在节点内使用。典型的 Megatron 训练配置：TP=8（单节点 NVLink）+ PP=16（跨节点）+ DP=若干。

---

# 第 10 章：ZeRO & FSDP — 不是切模型，是切状态

> **核心问题**：DP 训练中，每张 GPU 都有一份完整的模型状态（参数 + 梯度 + Adam 状态）。能不能只切这些状态——参数只在需要时才拼起来？这就是 ZeRO。

## 10.1 数据并行的浪费

回顾 DDP 的训练过程：

1. **前向**：每张 GPU 用自己的数据做前向，需要完整的模型参数
2. **反向**：每张 GPU 算自己的梯度，需要完整参数和前向激活值
3. **AllReduce 梯度**：所有卡同步梯度
4. **Optimizer step**：每张 GPU 更新自己的参数副本

问题出在第 3、4 步之后：每张 GPU 的参数副本仍然是完整的——即便它们完全相同。同时，Adam 的 m 和 v 也是每卡一份完整的。这就是 ZeRO 论文 (Rajbhandari et al., 2020) 所谓的「数据并行的浪费」：模型状态被每张 GPU 全量复制。

ZeRO 的核心洞察：**不是所有训练状态都需要全程全量驻留在每张 GPU 上**。Optimizer states、gradients、甚至 parameters，都可以分布到多张 GPU，只在**需要的时刻**通过通信临时拼起来。

## 10.2 三级递进

### ZeRO-1：只切 Optimizer States（P_os）

Adam 的 m、v、master weights（fp32）总计 12N bytes/param，是最大的显存消耗源。ZeRO-1 把这些优化器状态均匀切到 Np 张 GPU 上：每张 GPU 只持有 12N/Np bytes 的 optimizer states，**只负责更新自己那部分参数**。

通信模式：AllReduce 梯度（1× 通信量）→ 每卡各自对自己那 1/Np 的参数做 Adam 更新 → 无需通信（因为 optimizer step 是 local 的）。最后 AllGather 参数（所有卡拼回完整参数，但这步可以延迟到下一次前向）。

结果：显存 = 2N(param+grad) + 12N/Np(os)。Np=4 → 32+21 = 53 GB（从 112GB 降至 53GB，仍超 80G 但接近了）。

### ZeRO-2：加切 Gradients（P_os+g）

梯度也是每卡全量的浪费。ZeRO-2 将梯度也切分：反向时，每张 GPU 只保留自己那 1/Np 的梯度分片，用 **ReduceScatter** 替换 AllReduce。

ReduceScatter 和 AllReduce 的通信量相同（都是 (N-1)/N × D），但结果每张 GPU 只拿 1/Np 的梯度而非全量。这省了 (1 - 1/Np) × 2N bytes 的梯度存储。

结果：显存 = 2N(params) + 2N/Np(grad) + 12N/Np(os)。Np=4 → 28 + 3.5 + 21 = 52.5 GB。

### ZeRO-3：连 Parameters 也切（P_os+g+p）

ZeRO-3 最终把参数本身也切分了。这需要改变前向/反向的执行方式：

- **前向开始**：AllGather 参数（把自己缺失的 1-1/Np 参数从其他卡拼回来）→ 得到完整参数 → 做前向计算 → **立即释放**拼回来的参数
- **反向开始**：再次 AllGather 参数 → 得到完整参数 → 做反向计算（产生梯度，ReduceScatter 分片）→ 释放参数

通信量：AllGather(前向) + AllGather(反向) + ReduceScatter(梯度) = 2×(N-1)Np - 是分均的参数通信，加上梯度的正常通信。实际上约为 DP 的 1.5×。显存 = (2N + 2N + 12N)/Np = 16N/Np → Np=8 → 14GB！（终于比 A100-80G 小了）This is the key unlock.

**关键**：ZeRO-3 让训练 7B 模型需要的单卡显存从 112GB → 14GB（at Np=8）。虽然通信多了 0.5×，但这个 trade-off 几乎总是值得的——显存才是瓶颈，通信可以靠带宽堆。

## 10.3 FSDP2 = PyTorch 的 ZeRO-3

FSDP2（Fully Sharded Data Parallelism v2，PyTorch 2.0+）是 ZeRO-3 的 PyTorch 原生实现：

- `reshard_after_forward=True`：前向 AllGather 参数 → 前向完后立即释放（reshard）→ 反向时重新 AllGather → ZeRO-3 语义
- `reshard_after_forward=False`：前向 AllGather 后保留完整参数直到反向结束 → 通信少了一次（反向不需要重新 AllGather）→ ZeRO-2 语义

FSDP2 相比 FSDP1 的核心改进：
- **Per-parameter sharding**：用 DTensor 对每个参数独立分片，不再需要 FlatParameter（FSDP1 把所有参数 flatten 成一个大张量，带来很多约束）
- **Composable API**：`fully_shard(module)` 可以对每个 submodule 独立调用，比 FSDP1 的整模型策略灵活得多
- **DeviceMesh 集成**：天然支持多维并行（TP×FSDP）

### FSDP2 vs DeepSpeed ZeRO-3

| 特性 | FSDP2 | DeepSpeed ZeRO-3 |
|------|-------|-----------------|
| CPU Offload | 有限（`offload_to_cpu`） | 成熟（ZeRO-Offload，支持 Adam 运算在 CPU 执行） |
| NVMe Offload | 不支持 | ZeRO-Infinity（万亿参数模型的唯一选择） |
| 与 TP 共存 | DTensor DeviceMesh 天然支持 | DeepSpeed 3D Parallelism 配置 |
| torch.compile 兼容 | 更好（PyTorch 原生） | 有限 |
| 社区 & 文档 | PyTorch 官方生态 | Microsoft 维护，文档偏向 DeepSpeed 全家桶 |

**简单决策**：单节点 8 卡内 → FSDP2；需要 CPU/NVMe offload → DeepSpeed；MoE 训练 → DeepSpeed（有专门 MoE + EP 支持）。

## 10.4 显存节省的互补关系

AC、ZeRO、TP、PP 这四种技术节省的显存部分**不完全重叠**，因此可以组合：

| 技术 | 省什么 | 不省什么 |
|------|--------|---------|
| AC | 激活值（中间层 inputs/outputs） | 模型状态（参数/梯度/Adam） |
| ZeRO | 模型状态（参数/梯度/Adam） | 激活值 |
| TP | 激活值 + 参数（都在层内切分） | 梯度（AllReduce 后在每卡恢复完整） |
| PP | 激活值分散到不同 stage | 每 stage 仍需保存 M 个微批次激活值 |

**互补组合示例**：ZeRO-3 + AC
- ZeRO-3：参数切 1/Np → 显存省 Np×
- AC：每隔 2 层 checkpoint → 激活值省 50%
- 组合：模型状态 = 16N/Np，激活值 = 50% × baseline。7B@Np=4：14GB + 5GB = 19GB → 可以放 80GB 中轻松训练

## 10.5 CPU/NVMe Offloading

当 Np 不够大（你没有 64 张 H100）或模型太大（例如 405B）时，ZeRO-3 的显存仍然可能超标。这时需要把状态搬到更便宜的存储介质：

- **ZeRO-Offload（CPU）**：把 optimizer states（Adam m, v）搬到 CPU 内存。前向/反向仍在 GPU 做，但 optimizer step 时把梯度传到 CPU，在 CPU 上更新 Adam，再把新参数传回 GPU。带宽 ≈ 50 GB/s（PCIe），约 HBM 的 1/67。
- **ZeRO-Infinity（NVMe）**：进一步把 optimizer states + 参数分片搬到 NVMe SSD。带宽 ≈ 2-7 GB/s。仅用于 500B+ 参数的极端 case。

Offloading 的 trade-off 仍然是时间 vs 空间：慢但便宜。

---

# 第 11 章：序列并行 — 当上下文比模型还大

> **核心问题**：前三类并行切的是参数/层/数据。但 128K token 的注意力矩阵（128K² × 2 bytes = 32GB 单头）本身单卡就放不下——无论你怎么切参数都没用。需要第四维并行：沿序列长度切分。

## 11.1 128K 序列的 O(S²) 困境

注意力计算的核心是 $QK^T$，产生一个 $[B, n_h, S, S]$ 的 attention score 矩阵。在标准的 scaled dot-product attention 实现中，这个矩阵需要被完整地物化（materialize）在 HBM 中：

$$\text{Memory} = B \times n_h \times S \times S \times 2 \text{ bytes (bf16)}$$

以 LLaMA-7B（$n_h = 32$）为例：

| 序列长度 S | Attention Matrix 大小 | 说明 |
|-----------|---------------------|------|
| 2,048 | 32 × 2048² × 2 = 256 MB | 完全可承受 |
| 8,192 | 32 × 8192² × 2 = 4 GB | 变得吃力 |
| 32,768 | 32 × 32768² × 2 = 64 GB | 接近单卡 HBM 极限 |
| 131,072 | 32 × 131072² × 2 = 1 TB | 单卡完全不可能 |

前三类并行（TP/PP/ZeRO）切的是参数、层、或优化器状态，对 attention matrix 本身的显存爆炸无能为力。Sequence Parallelism（也称 Context Parallelism, CP）沿序列长度维切分，是解决长上下文训练的唯一手段。

注意：FlashAttention 通过分块计算避免了 attention matrix 的完整物化（只在 SRAM 中维护小 tile），因此单卡可以支持到约 8K-16K 的序列。但超过这个范围，即使 FA 的 tile 机制也需要额外的跨卡切分——这就是 Ring Attention 的动机。

## 11.2 Ring Attention

Ring Attention（Liu et al., 2023）的核心思想：**Q 驻留本地不动，KV 块在 GPU 之间环形传递**。每张 GPU 持有序列的 1/Np 段，算自己的 Q 对当前 KV 块的部分 attention，然后用 online softmax 修正合并所有轮次的结果。

### 核心算法（以 4 张 GPU 为例）

```
初始化：每张 GPU 持有 Q_i, K_i, V_i（i=0,1,2,3，各含 S/4 个 token）

Round 0: 用 K_0, V_0 对 Q_i 做 local attention（online softmax，初始化 m, l, O）
         发 K_0, V_0 给 GPU_1
Round 1: 收到 K_3, V_3（从 GPU_0 收到），用它们更新 online softmax
         发刚收到的 K_3, V_3 给 GPU_1
Round 2: 收到 K_2, V_2，更新 online softmax
Round 3: 收到 K_1, V_1，最后一次更新 → 最终 attention 输出
```

### Online Softmax 修正公式

Ring Attention 的数学核心是 online softmax 的分块修正（和 FlashAttention 在单卡上做的事完全相同，只是推广到了多卡之间）。

**标准 softmax 回顾**：对于 attention score 向量 $s = [s_1, ..., s_S]$，softmax 定义为 $\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$。但直接计算 $\exp$ 在 fp16/bf16 下容易溢出（$e^{100} \approx 2.7 \times 10^{43}$ 远超 fp16 范围）。数值稳定版本使用 $m = \max(s)$ 做偏移：

$$\text{softmax}(s)_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}$$

其中 $e^{s_i - m} \in (0, 1]$，不再溢出。

**分块 softmax（Online Softmax）**：当 $s$ 被分成 $K$ 个块 $s^{(1)}, s^{(2)}, ..., s^{(K)}$ 依次到达时，如何计算等效的 softmax？

关键观察：给定当前全局状态 $(m_{old}, l_{old}, O_{old})$（分别表示全局最大值、softmax 分母、加权输出），当新块 $s_{new}$ 到达时：

$$
\begin{aligned}
m_{local} &:= \max(s_{new}) \\
m_{new}   &:= \max(m_{old}, m_{local}) \\
l_{local} &:= \sum_j e^{s_{new}[j] - m_{local}} \\
l_{new}   &:= e^{m_{old} - m_{new}} \cdot l_{old} + e^{m_{local} - m_{new}} \cdot l_{local}
\end{aligned}
$$

这里 $e^{m_{old} - m_{new}}$ 是「缩放因子」——它把以前的结果 rescale 到新的 max 下。如果新块的 max 大于旧块的 max（$m_{local} > m_{old}$），则 $m_{new} = m_{local}$，旧结果被缩小；反之旧结果被放大或保持不变。

注意 $e^{m_{old} - m_{new}}$ 和 $e^{m_{local} - m_{new}}$ 至少有一个是 $e^0 = 1$（因为 $m_{new}$ 必然等于 $m_{old}$ 或 $m_{local}$ 之一），这保证了数值稳定性——不会出现两个极小的指数函数值。

加权输出的更新：

$$O_{new} = \frac{l_{old}}{l_{new}} e^{m_{old} - m_{new}} \cdot O_{old} + \frac{l_{local}}{l_{new}} \cdot \left(\text{softmax}(s_{new}) \cdot V_{new}\right)$$

这两项权重之和为 1：$\frac{l_{old}}{l_{new}} e^{m_{old} - m_{new}} + \frac{l_{local}}{l_{new}} = 1$。物理意义就是把旧的 attention output 和新的 attention output 做加权平均，权重由各块的 softmax 分母占比决定。

**扩展注意力维度（Ring Attention 的做法）**：上面的 online softmax 处理的是向量 $s \in \mathbb{R}^{S}$。在 Attention 中，Q 和 K 的维度是 $[B, n_h, S_q, d_h]$ 和 $[B, n_h, S_{kv}, d_h]$。Online softmax 对 Q 的每一行（每个 query token）独立维护 $(m, l)$ 状态。

在 Ring Attention 中，每张 GPU 拿到的 Q 块是 $[B, n_h, S/Np, d_h]$。在每一轮环形通信中，GPU 收到新的 KV 对（$[B, n_h, S/Np, d_h]$），对这一批 KV 算 $s_{local} = Q_{local} \cdot K_{new}^T / \sqrt{d_h}$，然后用上面的公式更新全局 online softmax 状态。

**伪代码**（以 GPU rank=i 为例）：

```python
# 初始化 online softmax 状态
m = -inf * torch.ones(B, n_h, S_i, 1)    # S_i = S/Np, 每 query token 一个 max
l = torch.zeros(B, n_h, S_i, 1)          # softmax 分母
O = torch.zeros(B, n_h, S_i, d_h)        # 加权输出

# Q_i 驻留 GPU i
Q_i = Q_chunks[i]   # [B, n_h, S_i, d_h]

for round in range(world_size):
    # 本轮 KV 来自哪张卡
    src = (i - round) % world_size
    K_src = K_chunks[src]  # [B, n_h, S_i, d_h]
    V_src = V_chunks[src]
    
    # 计算本轮的 local attention scores
    scores = Q_i @ K_src.transpose(-2, -1) / sqrt(d_h)  # [B, n_h, S_i, S_i]
    m_local = scores.max(dim=-1, keepdim=True).values   # 本轮的 max
    l_local = (scores - m_local).exp().sum(dim=-1, keepdim=True)  # 本轮的 softmax 分母
    
    # 更新 online softmax 状态
    m_new = max(m, m_local)
    old_scale = (m - m_new).exp()      # rescale 旧结果
    local_scale = (m_local - m_new).exp()
    
    l_new = old_scale * l + local_scale * l_local
    O = (l / l_new) * old_scale * O + (l_local / l_new) * local_scale * (F.softmax(scores, dim=-1) @ V_src)
    
    m, l = m_new, l_new
    
    # 如果还没到最后 round，把刚收到的 KV 发给下一张卡
    if round < world_size - 1:
        K_chunks[src], V_chunks[src] = send_to_next(K_src, V_src)

# 完成 world_size 轮后，O 即为完整序列的 attention 输出
```

**为什么 Online Softmax 在这里是必须的？** 如果不用 online softmax，每轮每个 GPU 独立算 softmax 再平均，结果不等于对完整 KV 算的 attention。因为不同的 K 块产生不同的 softmax 分母，不能简单地各算各的再平均。Online softmax 的 $(m, l, O)$ 三元组保证了渐进修正等价于一次性计算。这是数学保证，而非近似。

### 通信量分析

每张 GPU 每轮发送 K_i + V_i（各 S/Np × d_head 个 bf16），共 Np 轮。总通信量：

$$\text{Comm} = N_p \times \frac{S}{N_p} \times d_{head} \times 2 \times 2 \text{ bytes} = 4 \times S \times d_{head} \text{ bytes}$$

注意：这是每张 GPU **发送**的量（接收量相同）。Ring Attention 的通信量 ∝ S（线性），而计算量 ∝ S × S/Np（每卡算自己的 Q 对所有 KV）。**计算/通信比 = S/Np**——序列越长，这个比值越大，Ring Attention 越划算。

## 11.3 Causal 场景的负载不均

Causal self-attention 中，token 只能 attend 前面的 token。在 Ring Attention 的 causal 模式下：

- GPU 0（序列前 1/4）：Q_0 只需要 attend K_0（自己的部分），后面的 K 全被 causal mask 掉
- GPU 3（序列后 1/4）：Q_3 需要 attend K_0 + K_1 + K_2 + K_3（所有前面的 K）
- **GPU 3 的计算量是 GPU 0 的 4 倍**

这意味着整个 ring 在等待最慢的 GPU 完成计算——和 PP 的泡是同类问题。

**Striped/Zigzag Attention** 通过交错排列 token 来修复这个问题：不按连续段切分，而是 GPU 0 拿 token {0, 4, 8, ...}, GPU 1 拿 {1, 5, 9, ...}, 等等。这样每张 GPU 的 Q 分布在整个序列中而非集中在某一段，负载更均匀。

## 11.4 DeepSpeed Ulysses 及其局限

Ulysses（Jacobs et al., 2023）提供了另一种思路：

1. 输入：每卡持有 $[B, N_{heads}, S/N_p, d_{head}]$（序列维切分）
2. **第一次 All-to-All**：把序列切分变成 head 切分 → 每卡持有 $[B, N_{heads}/N_p, S, d_{head}]$（完整序列，但只有部分 heads）
3. 各卡各自做完整的 attention（S 不切，不需要环形传递）
4. **第二次 All-to-All**：从 head 切分变回序列切分

优势：不需要环形传递的多轮通信——总共只有 2 次 All-to-All。通信量更少（相比于 Ring 的 Np 轮 KV 传递）。

**关键限制**：$N_{heads}$ 必须 ≥ $N_p$（否则某些卡分不到 head），且 $N_{heads}$ 必须能被 $N_p$ 整除。这在 GQA 模型（Q heads > KV heads）中尤其苛刻：如果 N_kv_heads = 8，Np > 8 时无法使用 Ulysses。

### Ring vs Ulysses 选择矩阵

| 场景 | 推荐 | 原因 |
|------|------|------|
| N_heads ≥ Np 且非 GQA | Ulysses | 2 次 All-to-All，通信少 |
| GQA（KV heads 少） | Ring Attention | Ulysses 退化（head 数约束） |
| 极长序列（512K+） | Zigzag Ring | 因果负载更均匀 |
| 异构集群 | Ring | 不要求 head 数约束，更 flexible |

---

# 第 12 章：专家并行 — MoE 的规模化

> **核心问题**：Mixture-of-Experts 模型中，一层有 8-256 个 expert 子网络。如果所有 expert 都挤在一张 GPU 上，显存就是 8-256 倍。但每个 token 只激活 1-2 个 expert——能不能让 expert 分布到多卡，token 通过网络路由到正确的 GPU？

## 12.1 从单卡 MoE 到分布式 Expert

单卡 MoE（如 Mixtral 8×7B）：8 个 expert FFN 全部放在同一 GPU，每个 token 通过 gate network 选择 top-2 expert 进行前向。显存 = 8 × 每个 expert 的参数。虽然每个 token 只激活 2 个 expert，但**所有 8 个 expert 的参数都必须驻留在显存中**——否则未被选中的 expert 无法被其他 token 使用。

EP 的本质：把 N 个 experts 均匀分布到 E GPUs 上，每个 GPU 持有 N/E 个 experts。token 通过网络路由到「拥有它所需 expert 的 GPU」，在该 GPU 上执行 expert compute，然后路由回原来的 GPU。

## 12.2 EP 的四步循环

```
Step 1: Token Routing（本地）
  - 每张 GPU 对本地 tokens 做 Top-K gating
  - 确定每个 token 需要去哪些 experts（路由决策）

Step 2: All-to-All Dispatch（通信）
  - 每张 GPU 把 tokens 按目标 expert 分组
  - All-to-All 发送：每个 chunk → 对应 expert 所在 GPU

Step 3: Expert Compute（计算）
  - 每张 GPU 对收到的 tokens 执行本地 experts 的前向
  - 注意：tokens 的数量可能不均衡（hot expert）

Step 4: All-to-All Combine（通信）
  - 每张 GPU 把计算结果按原 token 序号发回
  - All-to-All 和 dispatch 对称但方向相反
```

### 与 TP 的通信对比

| 特性 | TP | EP |
|------|-----|-----|
| 通信操作 | AllReduce（每层 2 次） | All-to-All（每层 2 次） |
| 参与 GPU | 所有 GPU 都参与每层计算 | 每 token 只去 1-2 expert 所在 GPU |
| 计算模式 | 密集（每卡都算自己的那份矩阵） | 稀疏（token 只算被路由到的 expert） |
| 通信量 | ∝ B×S×d_model | ∝ B×S×d_model（token 本身的传输） + expert 参数分布 |

EP 的通信量和 TP 相同量级，但因为 EP 只在 MoE 层用（而非每层都用），实际通信量远少于全 TP。这也是为什么 MoE 层用 EP、Attention 层用 TP——各取所需。

### All-to-All 通信的细节

EP 的两次 All-to-All 和 TP 的两次 AllReduce 在数据流向上完全不同。以 dispatch 为例（4 GPU, 4 experts）：

```
初始状态（每 GPU 有若干 tokens）:
GPU 0: [t0 → expert 3, t1 → expert 0, t2 → expert 2]
GPU 1: [t3 → expert 1, t4 → expert 2, t5 → expert 0]
GPU 2: [t6 → expert 0, t7 → expert 3, t8 → expert 1]
GPU 3: [t9 → expert 2, t10→ expert 1, t11→ expert 3]

All-to-All Dispatch 后:
GPU 0 (hosts expert 0): [t1, t5, t6]           ← 三个 token 都要 expert 0
GPU 1 (hosts expert 1): [t3, t8, t10]         
GPU 2 (hosts expert 2): [t2, t4, t9]          
GPU 3 (hosts expert 3): [t0, t7, t11]         
```

每个 GPU 在 All-to-All 中同时做发送和接收。GPU 0 把 t0 发给 GPU 3（t0→exp3），t2 发给 GPU 2（t2→exp2），同时从 GPU 1 接收 t5、从 GPU 2 接收 t6。

**通信量的计算**：在最理想（负载均衡）的情况下，每张 GPU 平均分配 tokens 到 Np 张卡上。每张卡发送 $(Np-1)/Np \times (B \times S \times d_{model})$ 的数据（给自己的 1/Np 留在本地）。总通信量 $\approx (Np-1)/Np \times B \times S \times d_{model} \times 2$ bytes (bf16)。和 AllReduce 完全相同的量级。

**稀疏性带来的节省**：但在实际中，每个 token 只被路由到 Top-K experts（通常 K=1 或 2），而非所有的 Np experts。这意味着：
- 每个 token 只被发给 K 个 GPU，而非 Np 个
- 通信量 = $K/Np \times (Np-1)/Np \times B \times S \times d_{model}$
- 当 K=2, Np=64 时，通信量是 full All-to-All 的约 2/64 = 3.1%

这就是 EP 的关键优势——All-to-All 在 MoE 场景下的稀疏性使得通信量远小于 TP 的密集 AllReduce。

## 12.3 Hot Expert 与 Capacity Factor

EP 的一个尖锐工程问题：如果大部分 token 的路由决策指向少数几个「热门 expert」，那么这些 expert 所在的 GPU 会收到远超预期的 token 数。这导致：
- 计算负载严重不均（某些卡忙于处理大量 token，其他卡空闲）
- 显存溢出（收到的 tokens 太多，前向中间结果放不下）

### Capacity Factor

Capacity Factor 限制了每张 GPU 的每个 expert 每步最多处理多少 token：

$$\text{Capacity} = \frac{\text{total\_tokens\_per\_batch}}{N_{experts}} \times \text{capacity\_factor}$$

- CF = 1.0：理想均衡（每个 expert 处理相等数量的 token）。一旦不均衡就溢出
- CF = 1.25：允许 25% 超额。溢出的 token 被「drop」——不经过 expert，经 residual connection 直通
- CF < 1.0：允许部分 token「不被任何 expert 处理」。常用于极大 batch，牺牲质量换吞吐

### 负载均衡的三种策略

1. **Auxiliary Loss（传统方案，Switch Transformer, 2021）**：在训练 loss 中加一项惩罚路由不均衡的项。问题：loss scale 是超参——太小无效果，太大干扰主任务收敛。

2. **Capacity Factor + Token Drop（GShard, 2021）**：超过容量的 token 被丢弃（经 residual 直通）。简单粗暴但有效。

3. **Bias-based Routing（DeepSeek-V2/V3, 2024）**：为每个 expert 维护一个可学习的 bias，加在 gate logits 上。每 N 步根据历史负载统计更新 bias——负载过重的 expert 降低 bias，过轻的提升 bias。优势：不需要 auxiliary loss，不对主任务 loss 增加任何项；bias 的更新完全基于统计（而非梯度），稳定且可控。

## 12.4 DeepSeek-V3 的 EP 实战

解读 DeepSeek-V3 训练配置中的每个数字：

| 配置项 | 值 | 含义 |
|--------|-----|------|
| 总参数 | 671B | MoE 模型，激活参数 ~37B/token |
| Experts | 256 per layer | 巨大的 expert 集合 |
| Top-K | 8 | 每个 token 激活 8 个 experts + 1 个 shared expert |
| TP | 4 | 节点内 Attention 层张量并行 |
| EP | 64 | Expert Parallelism — 256/64 = 4 experts per GPU |
| PP | 16 | 跨节点 Pipeline Parallelism |
| DP | 2 | 数据并行（相对少——因为 EP 已经提供了高 parallelism） |
| 总 GPU | 2048 | TP(4) × EP(64) × PP(16) × DP(2) = 8192？不对 —— Attention 和 MoE 的并行拓扑不同。Parallel Folding 允许 TP 只在 Attention 层用 |

EP=64 为什么是最大的头？因为 256 experts 是显存重灾区——每个 expert 的参数量与一个中等 FFN 相当。如果不用 EP，显存会爆炸。EP 的切分收益 = 256/64 = 4 experts per GPU，relieving the memory pressure from 256 full experts to 4.

### DeepEP

传统的 All-to-All 由 CPU 发起调度：CPU 准备 send/recv buffer → 通知 GPU → GPU 执行通信。DeepEP 让 GPU kernel 直接发起 All-to-All，bypass CPU。结果：通信延迟降低 41%。

### FP8 Blockwise Training

DeepSeek-V3 在 671B 规模验证了 FP8 训练的可行性。关键技术：按 128×128 子矩阵独立做 scale（而非整个张量用一个 scale），确保 outlier 只污染局部子块、不影响全局精度。

### Multi-Token Prediction (MTP)

训练时不仅预测下一个 token，还预测未来 n 个 token。MTP 使得每个训练 step 提供的监督信号密度翻 n 倍 → 训练效率 ↑。推理时，MTP 的头可以自然地用于 speculative decoding（用小 heads 预测未来 token 给主模型验证），加速约 1.8×。

---

# 第四部分：工业实战 — 拼起来与跑起来

# 第 13 章：混合并行策略选择

> **核心问题**：给定一个模型（比如 70B Dense）和一个集群（比如 32×H100），你应该选什么并行策略？TP=？PP=？ZeRO=？它们怎么组合？

## 13.1 策略选择的决策框架

从硬约束出发，逐步加入软优化：

**硬约束（不满足就不行）**：

1. **单卡显存**：参数 + Adam + 激活值 < HBM（80GB）。这是第一道硬墙。如果 ZeRO-3 后单卡仍然超 → 必须加 TP 或 PP
2. **TP ≤ N_heads**：每卡至少 1 个 attention head。GQA 模型下 TP ≤ N_kv_heads
3. **N_experts % EP = 0**：All-to-All 要求 expert 均匀分布
4. **TP 在 NVLink 域内**：TP 跨 InfiniBand → 通信/计算比恶化 → MFU 骤降

**软约束（优先级从高到低）**：

1. **优先 ZeRO/FSDP**：通信开销最小、代码侵入性最低。够用就不加其他
2. **TP 在节点内**：仅在 ZeRO-3 不够时叠加（7B 模型不需要 TP）
3. **PP 跨节点**：模型层数多（≥32）、需跨节点时使用（泡是代价）
4. **EP 仅在 MoE**：Dense 模型不需要 EP

## 13.2 三个案例完整推演

### 案例 1：7B Dense @ 8×A100-80G

- 参数 bf16 = 14 GB → 单卡 fit。但 Adam(fp32) = 84 GB → model states = 14 + 14 + 84 = 112 GB > 80 GB
- 不需要 TP：模型足够小，参数 14 GB 能放进单卡
- 不需要 PP：层数 32，PP 有泡，且单节点 8 卡不走跨节点
- ZeRO-3：每卡 112/8 + 激活值(~5) = 19 GB
- **推荐：FSDP/ZeRO-3, DP=8（8 GPUs, 纯 FSDP）。不需要 TP 或 PP**

### 案例 2：70B Dense @ 32×H100-80G

- 参数 bf16 = 140 GB → 单卡绝对装不下 → 必须 TP
- TP=4（节点内 NVLink）：每卡 140/4 = 35 GB（参数）
- Adam + 梯度：ZeRO-3 across the rest of parallelism dims
- 32 GPUs / TP=4 = 8 DP groups → FSDP/ZeRO-3 across 8 groups
- 单卡显存 ≈ 35(params) + 5(activations) + 12×70B/8(os, ZeRO-1) ≈ 35 + 5 + 105/8 ≈ 53 GB ✓
- **推荐：TP=4 + ZeRO-3(DP across 8) + PP=1, 32 GPUs**

### 案例 4：13B Dense @ 4×A100-80G（单节点小集群）

- 参数 bf16 = 26 GB → 单卡 fit（但显存非常紧张——加上激活值后接近 80GB 上限）
- Adam(fp32) = 156 GB → model states = 26 + 26 + 156 = 208 GB >> 80 GB
- 只有 4 张 A100，TP=4 可行（节点内 NVLink）。每卡参数 = 26/4 = 6.5 GB
- Adam：ZeRO-3 across 4 GPUs → 12×13B/4 ≈ 39 GB/卡
- 总显存 = 6.5(params) + 5(activations) + 39(Adam) = 50.5 GB/卡 ✓
- **推荐：TP=4 + ZeRO-3(DP=1), 4 GPUs**

这个案例展示了小集群上的优化：TP 不仅用于解决参数太大装不下的问题（7B 不需要 TP），也用于在显存容量紧张时和 ZeRO-3 组合以进一步压榨显存空间。

### 案例 3：671B MoE (DeepSeek-V3) @ 2048×H800-80G（续）

深入解读每个并行数字的推导：

- **为什么 TP=4 不是 8？** H800 节点是 8 卡 NVLink。TP≤8 是可行的。但 TP=4 留给 EP 更大的空间——如果 TP=8，每卡只能处理 1 个 attention head，且留给其他并行的 GPU 数更少。trade-off：TP=4 的每卡 activation 通信压力小于 TP=8。
- **为什么 EP=64？** 256 experts / 64 = 4 experts per GPU。这是基于「每个 micro-batch 中每 expert 约收到多少个 token」的 trade-off。如果 EP 太小（比如 16），每卡就要存 16 个 experts → 显存压力大。如果 EP 太大（比如 128），每卡只有 2 个 experts → All-to-All 通信 overhead 占比升高（每卡处理的 token 数变少 → 通信/计算比恶化）。
- **为什么 PP=16？** 2048 GPUs / 8 GPUs per node = 256 nodes。256 nodes 切成 16 个 PP stages → 每个 stage 有 16 个节点。PP=16 既保证流水线泡率可控（M 足够大时泡率可接受），又不至于 stage 太多导致层分得太碎（671B 约 60 层, 60/16 ≈ 3.75 层/stage——偏少但可行，因为 MoE 层比 Dense 层深得多）。

## 13.3 并行策略速查表

| 模型规模 | 8×A100 | 32×H100 | 64×H100 | 128×H100 | 256+×H100 |
|---------|--------|---------|---------|----------|----------|
| 7B Dense | ZeRO-3 | ZeRO-3 | ZeRO-3 | ZeRO-3 | ZeRO-3 |
| 13B Dense | ZeRO-3 | ZeRO-3 | ZeRO-3 | ZeRO-3 | ZeRO-3 |
| 70B Dense | — | TP=4+ZeRO-3 | TP=4+ZeRO-3 | TP=8+ZeRO-3 | TP=8+PP=4+ZeRO-3 |
| 175B Dense | — | TP=8+PP=4+ZeRO-3 | TP=8+PP=4+ZeRO-3 | TP=8+PP=8+ZeRO-3 | TP=8+PP=8+ZeRO-3 |
| 405B Dense | — | — | TP=8+PP=8+ZeRO-3 | TP=8+PP=16+ZeRO-3 | TP=8+PP=16+ZeRO-3 |
| 671B MoE | — | — | — | — | TP=4+EP=64+PP=16 |

---

# 第 14 章：主流框架概览

> **核心问题**：PyTorch DDP、FSDP2、DeepSpeed、Megatron-Core 各自解决什么问题？我该用哪个？

## 14.1 PyTorch 分布式生态

PyTorch 的分布式能力按复杂度递增：

```
DDP → FSDP2 → DTensor/DeviceMesh → TorchTitan
 ↓       ↓            ↓                  ↓
最简单   自动切分      手动多维并行      全栈 3D 并行
```

- **DDP**：入门级。每卡全量模型 + AllReduce 梯度。适合 7B 或更小的模型（单卡能放下参数时）
- **FSDP2**：中等。自动切分参数/梯度/Adam（ZeRO-3 语义）。适合 13-70B 模型的 fine-tune 和中小规模预训练
- **DTensor/DeviceMesh**：进阶。允许手动定义多维并行拓扑（如 2D mesh: TP×FSDP）。适合需要精细控制的场景
- **TorchTitan**：PyTorch 官方的 3D 并行参考实现（2024+）。集成 TP + PP + FSDP + checkpointing + float8。适合了解「工业级训练代码长什么样」

**FSDP2 代码示例**：

```python
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DeviceMesh, Shard

# 创建 2D DeviceMesh: (4 TP groups) × (8 FSDP groups) = 32 GPUs
mesh = DeviceMesh("cuda", torch.arange(32).reshape(4, 8))

# 包装模型——每个 Transformer Block 一个 FSDP unit
for block in model.layers:
    fully_shard(block, mesh=mesh["fsdp"], reshard_after_forward=True)
fully_shard(model, mesh=mesh["fsdp"], reshard_after_forward=True)

# 前向：FSDP 自动管理参数 AllGather 和 reshard
output = model(input_data)
loss = criterion(output, labels)
loss.backward()  # FSDP 自动管理梯度 ReduceScatter
optimizer.step()
optimizer.zero_grad()
```

**关键参数解析**：
- `reshard_after_forward=True`：前向 AllGather 参数 → 前向完后立即释放 → 反向再次 AllGather（ZeRO-3 语义，最省显存）
- `reshard_after_forward=False`：前向 AllGather 后保留直到反向结束（ZeRO-2 语义，通信少一次但显存多一倍）
- `fully_shard` 可以分 module 调用——这和 FSDP1 的全局 `wrap_policy` 不同，允许对不同层用不同策略

**DDP 代码示例**（作为对比）：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])
# 前向/反向完全和单卡一样
# DDP 在 backward() 中自动触发 AllReduce
```

## 14.2 DeepSpeed

DeepSpeed 的独特优势：

- **ZeRO-Offload**：成熟的 CPU offload（Adam 更新在 CPU 做）
- **ZeRO-Infinity**：NVMe offload，支持万亿参数模型
- **MoE 原生支持**：EP + ZeRO + TP 组合配置相对简单
- **DeepSpeed Chat/RLHF**：专门为 RLHF 设计的 pipeline

典型配置示例（DeepSpeed JSON）：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {"enabled": false},
  "bf16": {"enabled": true}
}
```

## 14.3 Megatron-LM / Megatron-Core

NVIDIA 的 Megatron 系列是业内训练最大 dense 模型（GPT-3 175B、Llama 3 405B）的实际框架。核心特点：

- **TP 的奠基者**：ColumnParallel + RowParallel 的配对被 Megatron-LM 首创
- **Parallel Folding**：允许 Attention 和 MoE 用不同并行拓扑（Attention 用 TP，MoE 用 EP）——两者独立，不互相锁定
- **Interleaved PP**：比标准 1F1B 更高效的流水线
- **Distributed Optimizer**：和 ZeRO-1 类似的优化器状态切分

Megatron-Core（2024+）是 Megatron-LM 的重构版本，将并行策略抽象为可组合的模块。

## 14.4 框架选型指南

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 7B fine-tune @ 单机 8 卡 | FSDP2 | 简单够用、PyTorch 原生 |
| 70B pre-train @ 多机 | Megatron-Core | TP+PP 最佳实现 |
| MoE 训练 | DeepSpeed | EP + ZeRO MoE 原生支持 |
| 快速实验（<7B） | DDP + GA | 不需要分布式优化器，DDP 够用 |
| RLHF/DPO | DeepSpeed Chat | 专门 pipeline |
| 学习分布式原理 | FSDP2 / 从头写 | 理解后再看 Megatron |

---

# 第五部分：前沿与展望

# 第 15 章：2024-2026 五大突破

## 15.1 DualPipe — 泡的消失（DeepSeek-V3, 2024）

DualPipe 的核心创新在第 9 章已详述。总结其意义：这是第一次有人证明了 PP 的泡可以通过「双向注入 + 通信计算重叠」被消除到接近零。DualPipe 的 MFU 达到 51%（相比同期业界平均 ~35%），证明了高效分布式训练的 50% 利用率天花板是可以被打破的。

工程对应：DeepSeek 开源了 [DualPipe GitHub](https://github.com/deepseek-ai/DualPipe)，提供了调度算法的 Python 实现。调度逻辑的核心是一个双向的时间表（schedule），用 Python 列表定义每个 GPU 在每个时间步应该执行什么操作（forward/backward/send/recv）。

DualPipe 的实现细节：
- 每个 transformer block 被分解为 4 个 chunks：attention forward, attention backward, FFN forward, FFN backward
- 两个数据流（左→右 和 右→左）在时间上交错——GPU_i 在处理流 A 的 FFN forward 时，同时从 GPU_{i-1} 接收流 B 的 attention backward 梯度
- 通信库使用 NVLink 做节点内 P2P，InfiniBand 做跨节点通信
- 调度表是预计算的（offline），不需要运行时动态调度

## 15.2 FP8 大规模验证（DeepSeek-V3, 2024）

FP8 训练从「可能可行」→「671B 规模生产验证」。关键技术贡献：

**Blockwise/tilewise scaling（128×128 tiles）**：
传统量化做法是对整个张量计算一个 scale factor：$s = \max(|X|) / \text{fp8\_max}$。但这有一个根本缺陷——如果张量中有一个元素的值是 1000，其他都是 0.01-0.1，为了表示 1000，$s$ 必须很大 → 其他正常值除以 $s$ 后变得极小 → FP8 的 3 位尾数根本无法区分它们。

DeepSeek 的方案是把 4096×14336 这种大矩阵切成 128×128 的 tile，每个 tile 独立 scale。包含 outlier 的 tile 用大的 scale——只影响本 tile；正常 tile 用正常的 scale——精度不受 outlier 污染。这和 FlashAttention 的 tiling 思路一致：都是通过局部化处理来避免全局瓶颈。

**在 H800 上验证**：H800 是 H100 的中国特供版（受限于出口管制，NVLink 带宽从 900GB/s 降至约 400GB/s，且 inter-node 通信受限）。尽管如此，FP8 训练依然成功，证明 blockwise scaling 在受限硬件上是可行的。

**精度损失与消融实验**：
- 训练 loss 和 BF16 基线几乎重合
- 下游任务（MMLU, HumanEval 等）的精度差异在 ±0.5% 内
- 唯一需用 BF16 保留精度的操作：RMSNorm、softmax、gate_logits（MoE 路由）

**训练成本减半**：H100/H800 的 FP8 Tensor Core 峰值 = 1979 TFLOPS，是 BF16（989 TFLOPS）的 2×。实测速度提升约 1.8×（非 2×，因为部分操作保持 BF16）。对一次 $5.57M 的训练来说，这意味着省了约 $2.5M——不是小数。

## 15.3 Aux-loss-free Load Balancing（DeepSeek-V2, 2024）

传统 MoE 的 auxiliary load balancing loss 有一个根本问题：loss scale 是超参。辅助损失的形式通常为：

$$\mathcal{L}_{aux} = \alpha \cdot N_{experts} \cdot \sum_{i} f_i \cdot P_i$$

其中 $f_i$ 是路由到 expert i 的 token 比例，$P_i$ 是 gate network 对 expert i 的平均 softmax 概率，$\alpha$ 是 multiplier。$\alpha$ 太大 → 主任务 loss 被干扰，收敛困难；$\alpha$ 太小 → 负载不均衡，部分 GPU 过载、部分 GPU 空闲。

DeepSeek 的方案：每个 expert 加一个 bias $b_i$（初始 0），gate logits 变为 $\text{logits}_i + b_i$（只在 Top-K 选择时生效，不影响 softmax 概率值）。每 $\tau$ 步（如 100 步），根据这段期间的负载统计更新 bias：
- 如果 expert i 的 token 数 > 平均值：$b_i \leftarrow b_i - \gamma$（降低 bias，减少路由倾向）
- 如果 expert i 的 token 数 < 平均值：$b_i \leftarrow b_i + \gamma$（升高 bias，增加路由倾向）

关键优势：
- **不需要梯度**：bias 更新基于统计（计数），不通过 autograd
- **不影响主任务 loss**：bias 在 softmax 之前而非之后，只影响选择，不影响被选中的概率值（因为 softmax 前加常数字不改变 softmax 的相对分布？——不，加 bias 确实改变 softmax 输出的值，但 DeepSeek 只在 Top-K 排序阶段用 bias，然后用原始 logits 计算 softmax 概率）
- **消除超参 $\alpha$**：不需要手动调整 loss scale

## 15.4 Multi-Token Prediction（MTP）

MTP（Gloeckle et al., 2024；被 DeepSeek-V3 采用）在训练时让模型同时预测未来 n 个 token。DeepSeek-V3 使用 n=1（深度=1），即预测 $x_{t+1}$ 和 $x_{t+2}$。

**架构**：
- 主 transformer 输出最后一层 hidden states $h_t$
- $h_t$ 经过一个小的「MTP 模块」（一个 transformer block + 一个输出头）→ 预测 $x_{t+2}$
- 标准 next-token prediction 预测 $x_{t+1}$ 不变
- 两个 loss 加权求和：$\mathcal{L} = \mathcal{L}_{next\_token} + \lambda \cdot \mathcal{L}_{mtp}$

**为什么有效**：
- 每个训练步的监督信号密度翻倍——模型从每个 token 位置学到了「现在该输出什么」和「下一步该输出什么」
- MTP 迫使模型学习更长程的依赖关系——预测 $x_{t+2}$ 需要理解 $x_{t+1}$ 应该是什么，即使模型还没有显式输出它
- 推理时，MTP 的输出头可直接用于 speculative decoding——主模型预测 $x_{t+1}$ 的同时，MTP 模块用 $h_t$ 预测 $x_{t+2}$ → 1.8× 加速

**局限性**：MTP 在某种程度上是 redundant 的——如果你有一个足够好的主模型，它本来就能预测 $x_{t+2}$（通过先预测 $x_{t+1}$ 再预测 $x_{t+2}$）。MTP 的价值在于训练效率（更多监督信号）+ 推理加速（speculative decoding），而非模型能力的根本提升。

## 15.5 DeepEP — GPU 发起的通信

传统的 All-to-All 调度由 CPU 控制：
1. CPU 分配 pinned memory buffer
2. CPU 通知 GPU 启动 kernel（拷贝数据到 buffer）
3. CPU 通知 NCCL 发起 All-to-All
4. GPU 等待 NCCL 完成
5. CPU 通知 GPU 消费结果

每一步 1-5 都有 CPU-GPU 同步延迟（kernel launch + cudaStreamSynchronize）。在 MoE 的 All-to-All 场景中（每层每步都要两次 All-to-All），这个开销累积起来很可观。

DeepEP 让 GPU kernel 直接发起 All-to-All：
- GPU kernel 1：计算 expert routing + 准备 dispatch buffer → 直接调用 NCCL Device API 发起 All-to-All
- 在通信进行中：GPU 不需要空等——可以在另一个 stream 上处理 residual connection 或 attention
- GPU kernel 2：通信完成后，直接从 receive buffer 读取数据，开始 expert compute

优势：消除 CPU 在 critical path 中的所有参与——没有 CPU-GPU 同步、没有 CPU side buffer copy、没有 kernel launch 开销。

结果：MoE 层的 All-to-All 延迟降低 41%。对 256 experts、每步 2 次 All-to-All 的 60 层模型来说，这 41% 的放大效应是巨大的。

---

# 第 16 章：分布式推理

> **核心问题**：训练时的并行策略和推理时有什么不同？KV Cache 怎么在分布式场景下管理？

## 16.1 推理与训练的并行差异

| 维度 | 训练 | 推理 |
|------|------|------|
| 优化目标 | 吞吐（tokens/sec） | 延迟（TTFT）+ 吞吐 |
| 模型状态 | 参数 + 梯度 + Adam + 激活值 | 参数 + KV Cache |
| 并行重点 | 显存切分（ZeRO/TP/PP） | KV Cache 切分（CP为主）+ 参数切分（TP） |
| Batch | 大 batch（几千 tokens） | 动态 batch（continuous batching） |
| 时序 | 离线的 | 在线服务（SLA 约束） |

推理时不需要存梯度、Adam、激活值（前向结束后释放），唯一的大头是 **KV Cache**。每层每个 token 需要存储 K 和 V（各 d_head 维），GQA 模型下 KV heads 少 → KV cache 压力相对小。

**连续批处理（Continuous Batching）**：传统静态 batching 下，batch 中所有请求必须同时开始、同时结束。如果 batch 中有一个请求生成了 100 个 token 而另一个只生成了 10 个，前者必须等后者完成后才能一起返回。Continuous batching 在迭代级动态调整 batch 组成——有新请求到达时动态加入 batch，有请求完成时动态退出——大幅提升 GPU 利用率。

**TTFT（Time To First Token）**：从用户发送 prompt 到第一个 token 生成的时间。包含 prefill（处理整个 prompt 的一次前向）+ 第一个 token 的 decode。TTFT 是交互式场景下最重要的用户体验指标。

**TPOT（Time Per Output Token）**：生成每个后续 token 的平均延迟。TPOT 和 TTFT 一起决定用户感知到的「流畅度」。

## 16.2 KV Cache 的分布式管理

### PagedAttention（vLLM, Kwon et al., SOSP 2023）

PagedAttention 的核心思想：**将 KV Cache 的存储从「每序列一大块连续内存」改为「分页——每页存固定数量 token 的 KV」**。这和操作系统虚拟内存中的分页机制完全同构。

好处：
- **零内存碎片**：每页大小固定（如 16 tokens），不会出现一个长序列卡住一大块内存的内部碎片，也不会有内存池中的外部碎片
- **共享**：多个请求共享相同 prompt 前缀时，对应的 KV 页可以共享（引用计数）。例如 100 个请求都有相同的 system prompt，只需要存 1 份 KV，引用计数 100
- **灵活调度**：batch 中某个序列生成完毕后，其 KV 页立即被回收供其他请求使用

vLLM 是当前（2025-2026）最广泛采用的开源 LLM 推理引擎，其 PagedAttention 的实现已被 TensorRT-LLM、TGI 等框架参考或集成。

### RadixAttention（SGLang, Zheng et al., 2024）

SGLang 的 RadixAttention 更进一步：将 KV Cache 组织为**前缀树（Radix Tree）**。当多个请求共享前缀时（如 agentic LLM 场景中，大量请求有相同的 system prompt + tool definitions），树节点被自动共享。LRU 驱逐策略管理树节点的生命周期。

PagedAttention 和 RadixAttention 的关键差异：
- PagedAttention 的共享是「完全相同的前缀」→ 引用计数。前缀不同就没法共享
- RadixAttention 的共享是「重叠的前缀」→ 树节点共享。即使两个请求的 prompt 不完全相同，重叠部分（如共同的 system prompt）仍然能共享

在 agent 场景中（大量请求共享 system prompt + 工具定义前缀），RadixAttention 的命中率远高于简单的 hash-based prefix caching。这在 2026 年的 multi-agent、tool-use、RAG 场景中尤为重要。

## 16.3 Disaggregated Serving

Disaggregated serving 将推理的两个阶段分离到不同 GPU 组：

- **Prefill 集群**：处理全部 prompt token 一次前向（compute-bound，高计算量，需 TP）。用高 TFLOPS 的 GPU（如 H100）做 prefill
- **Decode 集群**：逐 token 生成（memory-bound，KV Cache 是瓶颈，需大显存 + 高带宽 HBM）。用大 HBM 的 GPU（如 H200 141GB）做 decode

好处：
- **独立扩缩容**：如果请求的 prompt 很长（prefill bottleneck），就扩展 prefill 集群；如果生成很长（decode bottleneck），就扩展 decode 集群
- **异构硬件匹配**：prefill 和 decode 对 GPU 的需求不同——prefill 需要高 compute、decode 需要高 memory bandwidth。可以为它们选配不同型号的 GPU
- **故障隔离**：prefill 集群挂了不影响已在 decode 的请求

挑战：
- **KV Cache 迁移**：prefill 完成后，KV Cache 需要从 prefill GPU 传到 decode GPU。这可能成为瓶颈——尤其是对于 128K+ 长序列，KV Cache 可能达数 GB
- **调度复杂度**：两类集群的规模需要动态匹配负载，增加了调度器的复杂性

**NVIDIA Dynamo**（2026）是一个生产级的 disaggregated inference orchestrator：
- KV-aware routing：将请求路由到已有相关 KV Cache 的 GPU（避免重复 prefill）
- Planner：基于 SLA 做自动扩缩决策
- NIXL（Network In-Xfer Library）：高效的跨节点 KV Cache 传输协议
- 发布 1.0 后，在 DeepSeek R1 上实测 7× 吞吐提升

---

# 附录 A：关键公式与常数速查

## FLOPs

| 公式 | 说明 |
|------|------|
| $C_{matmul} = 2 \times M \times K \times N$ | 矩阵乘法 FLOPs（一次乘一次加） |
| $C_{fwd} \approx 2ND$ | 一次前向（N 参数, D tokens） |
| $C_{bwd} \approx 4ND$ | 一次反向（weight grad 2ND + input grad 2ND） |
| $C_{total} \approx 6ND$ | 完整训练 pass |

## 显存

| 公式 | 说明 |
|------|------|
| $M_{params} = 2N$ | bf16 参数（bytes） |
| $M_{grads} = 2N$ | bf16 梯度 |
| $M_{adam} = 12N$ | fp32 Adam (m 4N + v 4N + master 4N) |
| $M_{model\_states} = 16N$ | 总模型状态 |
| $M_{ZeRO-3} = 16N/N_p$ | ZeRO-3 后的模型状态/卡 |

## 通信量

| 操作 | 发/卡 | 总通信 |
|------|--------|--------|
| Ring AllReduce | $2\frac{N-1}{N}D \approx 2D$ | $2(N-1)D$ |
| AllGather | $\frac{N-1}{N}D$ | $(N-1)D$ |
| ReduceScatter | $\frac{N-1}{N}D$ | $(N-1)D$ |
| Ring Attention | $4 \times S \times d_{head}$ | $4 \times S \times d_{head} \times N_p$ |

## 泡率

| 调度 | 泡率公式 | N=4 典型值 |
|------|---------|-----------|
| GPipe | $\frac{P-1}{P-1+M}$ | 27% (M=8) |
| 1F1B | 复杂（与 M,P 相关） | ~15% |
| Interleaved 1F1B | — | ~5% |
| DualPipe | — | ~8%（其中大部分被通信重叠） |

## 硬件常数（H100, 2023）

| 常数 | 值 |
|------|-----|
| BF16 TFLOPS | 989 |
| FP8 TFLOPS | 1979 |
| HBM 带宽 | 3.35 TB/s |
| NVLink 4.0 | 900 GB/s（单向） |
| InfiniBand NDR400 | 400 GB/s |
| PCIe 5.0 ×16 | 64 GB/s |
| HBM 容量 | 80 GB |
| SM 数 | 132 |
| NVSwitch 域 | 8 GPUs |

---

# 附录 B：推荐阅读清单

### 奠基论文（按阅读顺序）

| # | 论文 | 核心贡献 | 必读章节 |
|---|------|---------|---------|
| 1 | Kaplan et al., 2020. *Scaling Laws for Neural Language Models* | FLOPs-参数-数据量的幂律关系 | §1-2 |
| 2 | Hoffmann et al., 2022. *Training Compute-Optimal Large Language Models* | Chinchilla 修正（参数和 token 等比） | Abstract, §3-4 |
| 3 | Rajbhandari et al., 2020. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | 三级递进、显存分析 | §2（图1+表1）, §4-6 |
| 4 | Shoeybi et al., 2019. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* | Column/Row Parallel Linear | §3-4 |
| 5 | Huang et al., 2019. *GPipe: Efficient Training of Large Neural Networks using Pipeline Parallelism* | Micro-batch pipeline | §3.1 |
| 6 | Narayanan et al., 2021. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* | Interleaved 1F1B | §2.2 |
| 7 | Liu et al., 2023. *Ring Attention with Blockwise Transformers for Near-Infinite Context* | Ring Attention 核心算法 | §3 |
| 8 | DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report* | DualPipe + FP8 + MTP + EP 实战 | §2.1-2.3 |

### 博客与教程

| 资源 | 主题 |
|------|------|
| [Transformer Math (EleutherAI)](https://blog.eleuther.ai/transformer-math/) | FLOPs 分解 |
| [BaPipe AllReduce 图解](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) | Ring-AllReduce 可视化 |
| [FSDP2 Blog (PyTorch)](https://pytorch.org/blog/fsdp2/) | FSDP2 原理与用法 |
| [Ring Attention 图解 (Spheron, 2026)](https://www.spheron.network/blog/ring-attention-tree-attention-sequence-parallelism-gpu-cloud/) | Ring/Ulysses 可视化对比 |
| [NVIDIA Dynamo Blog (2026)](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) | 生产级推理基础设施 |
| [DeepSeek-V4 Analysis (LMSYS, 2026)](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) | V4 最新改进方向 |

### 源码参考

| 仓库 | 用途 |
|------|------|
| [NCCL](https://github.com/NVIDIA/nccl) | Ring/Tree 算法的工业实现 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | TP/PP 的参考实现 |
| [DualPipe](https://github.com/deepseek-ai/DualPipe) | 双向调度示例 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | GPU-initiated EP 通信 |

---

# 附录 C：术语中英对照表

| 英文 | 中文 | 一句话定义 |
|------|------|-----------|
| Data Parallelism (DP) | 数据并行 | 每卡拿到不同 data batch，梯度 AllReduce 同步 |
| Tensor Parallelism (TP) | 张量并行 | 切单层矩阵的行/列，每卡算部分输出 |
| Pipeline Parallelism (PP) | 流水线并行 | 切层序列，层边界传 activation |
| Sequence Parallelism (SP/CP) | 序列并行 | 沿序列长度维切分，注意力矩阵分布到多卡 |
| Expert Parallelism (EP) | 专家并行 | 切 MoE experts，tokens 通过 All-to-All 路由 |
| Fully Sharded Data Parallel (FSDP) | 全分片数据并行 | PyTorch 的 ZeRO-3 等价实现 |
| Collective Communication | 集合通信 | 多 GPU 共同参与的通信操作（如 AllReduce） |
| AllReduce | 全局归约 | 每卡不同数据 in → 每卡相同 sum out |
| AllGather | 全局收集 | 每卡不同数据 in → 每卡所有数据的拼接 out |
| ReduceScatter | 归约分散 | 每卡不同数据 in → 每卡 sum 的 1/N 分片 out |
| All-to-All | 全交换 | 每卡不同 chunks → 稀疏路由到各卡 |
| Activation Checkpointing (AC) | 激活检查点 | 不存全部中间激活，反向时重算 |
| Gradient Accumulation (GA) | 梯度累积 | 多步累积再同步，increase effective batch |
| Micro-batch | 微批次 | PP 中将 batch 切成的更小单位 |
| Pipeline Bubble | 流水线泡 | PP 中某 GPU 等待其他 GPU 时的空闲时间 |
| MFU (Model FLOPs Utilization) | 模型算力利用率 | 实际 FLOPs / 理论峰值 FLOPs |
| Mixed Precision Training | 混合精度训练 | FP32 master + BF16/FP16 compute |
| FP8 | 8 位浮点 | Hopper 架构（H100+）支持的训练精度，比 BF16 减半 |
| Blockwise Scaling | 块级缩放 | 按子矩阵而非整张量做 FP8 量化 |
| Capacity Factor | 容量因子 | EP 中限定每 expert 最大 token 数的超参 |
| Auxiliary-loss-free Balancing | 无辅助损失负载均衡 | DeepSeek 的 bias-based MoE 路由 |
| Multi-Token Prediction (MTP) | 多 token 预测 | 训练时预测未来 n 个 token |
| DualPipe | 双向流水线 | DeepSeek 的 bidirectional pipeline |
| Disaggregated Serving | 分离式推理 | Prefill 和 Decode 分属不同 GPU 组 |
| PagedAttention | 分页注意力 | vLLM 的 KV Cache 分页管理 |
| RadixAttention | 基数树注意力 | SGLang 的前缀树 KV Cache 共享 |
| DeviceMesh | 设备网格 | PyTorch DTensor 的多维并行拓扑描述 |
| NVSwitch | NVSwitch | 节点内 8 卡全互联的硬件交换机 |
| NCCL | NVIDIA 集合通信库 | GPU 间通信的标准库 |

---

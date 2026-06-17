# Transformer 显存建模手册

> **定位**：从 16 bytes/param 出发，推导训练时的完整显存占用，以及 Activation Checkpointing 和 Gradient Accumulation 两种省显存技术。不涉及 FLOPs 计算。
> **参考模型**：Qwen3-8B（2026.4, Alibaba）— $d_{model}=4096$, $d_{ff}=12288$, $n_{heads}=32$ (Q), $n_{kv\_heads}=8$ (GQA 4:1), $d_{head}=128$, $n_{layers}=36$, $V=152064$, 总参数量 ~8.2B。
> **关联笔记**：[compute_flops_notes.md](compute_flops_notes.md) — 算力建模 · [distributed_training_handbook.md](../distributed_training_handbook.md) — 分布式全局手册

---

## 目录

- [第 1 章：Model States — 16 bytes/param 从哪来](#第-1-章model-states--16-bytesparam-从哪来)
- [第 2 章：激活值显存逐层分解](#第-2-章激活值显存逐层分解)
- [第 3 章：Activation Checkpointing 的数学](#第-3-章activation-checkpointing-的数学)
- [第 4 章：Gradient Accumulation 的数学](#第-4-章gradient-accumulation-的数学)
- [第 5 章：完整案例 — Qwen3-8B 显存账单](#第-5-章完整案例--qwen3-8b-显存账单)
- [第 6 章：推理 vs 训练显存](#第-6-章推理-vs-训练显存)
- [附录 A：模型参数速查表](#附录-a模型参数速查表)
- [附录 B：推导假定与适用范围](#附录-b推导假定与适用范围)

---

# 第 1 章：Model States — 16 bytes/param 从哪来

> **知识定位**：FLOPs 告诉你训练要多久，显存告诉你「能不能训」。从 16 bytes/param 这个 magic number 出发。

## 1.1 概念区分：Model States vs Residual States

| 类型                  | 生命周期       | 示例                                 | 省它的技术                    |
| ------------------- | ---------- | ---------------------------------- | ------------------------ |
| **Model States**    | 跨 step 持久  | 参数、梯度、Adam m、Adam v、master weights | ZeRO、FSDP                |
| **Residual States** | 单 step 内临时 | 激活值、通信 buffer、显存碎片                 | Activation Checkpointing |

两个完全不同的优化方向，互不重叠。

## 1.2 Model States：16 bytes/param 的逐项分解

以 bf16 混合精度 + Adam 训练为例。训练时需要 GPU 显存中驻留以下五类数据：

| 组件                       | 精度   | Bytes/param | Qwen3-8B (8.2B params) |
| ------------------------ | ---- | ----------- | ---------------------- |
| 模型参数                     | bf16 | 2           | 16.4 GB                |
| 梯度                       | bf16 | 2           | 16.4 GB                |
| Adam m ($\beta_1=0.9$)   | fp32 | 4           | 32.8 GB                |
| Adam v ($\beta_2=0.999$) | fp32 | 4           | 32.8 GB                |
| Master weights           | fp32 | 4           | 32.8 GB                |
| **Model States 合计**      |      | **16**      | **131.2 GB**           |

#### 5.2.1 模型参数和梯度

这两项最直观。参数是权重矩阵本身，每个元素以 bf16 存储（2 bytes）。梯度是反向传播计算出的 $\frac{\partial L}{\partial W}$，同样以 bf16 存储。两项合计 4 bytes/param，占 Model States 的 25%。

#### 5.2.2 Adam 优化器的「记忆」—— m 和 v

Adam 不是简单的 SGD。SGD 的更新规则只有一行：
$$W_{t+1} = W_t - \eta \cdot g_t$$

其中 $W_t$ 是第 $t$ 步的参数值，$\eta$ 是学习率，$g_t$ 是当前 batch 上的梯度。这有两个根本问题：第一，$g_t$ 来自单个 mini-batch，噪声很大，更新时间方向剧烈摆动；第二，所有参数共享同一个 $\eta$，但不同参数的梯度尺度可能差几个数量级——embedding 层低频 token 的梯度接近 0，attention 层高频 token 的梯度可能很大。

Adam 用三个机制逐一解决这些问题。

**（一）一阶动量 $m$ —— 用惯性平滑更新方向**

Momentum 的思想来自物理学：一个在曲面上滚动的球不会因每个局部的凹凸而瞬间改变方向，它有惯性。对应到优化中，更新方向不只看当前梯度，还要看历史梯度的加权累积。
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t, \quad \beta_1 = 0.9$$将这个递推展开，可以看到每一项的贡献随时间指数衰减：$$m_t = (1-\beta_1)\big[g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \beta_1^3 g_{t-3} + \cdots\big]$$

$\beta_1 = 0.9$ 意味着 10 步前的梯度权重衰减到 $(0.9)^{10} \approx 0.35$。有效记忆窗口约为 $\frac{1}{1-0.9} = 10$ 步。效果：

- 梯度持续朝同一方向（连续 1000 步 $g_t$ 为正）→ $m_t$ 逼近 $\mathbb{E}[g]$，更新幅度不会因单步波动变小——「保持速度」
- 梯度方向反复震荡（+1, -1, +1, -1）→ $m_t$ 趋近于 0，参数几乎不更新——「过滤噪声」

**（二）二阶动量 $v$ —— 为每个参数定制步长**

Momentum 解决了方向问题，但没解决步长问题。考虑两个参数：参数 A 在 embedding 层，对应一个低频 token，梯度 $g \approx 10^{-7}$；参数 B 在 attention 层高频 token，梯度 $g \approx 10^{-2}$。同一个 $\eta$ 下，要么 B 被这么大的步长震飞（overshoot），要么 A 永远不动。

Adam 用梯度**平方**的指数移动平均来估计每个参数的「陡峭程度」：
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2, \quad \beta_2 = 0.999$$展开后：$$v_t = (1-\beta_2)\big[g_t^2 + \beta_2 g_{t-1}^2 + \beta_2^2 g_{t-2}^2 + \cdots\big]$$

$\beta_2 = 0.999$ 的有效记忆窗口约为 $\frac{1}{1-0.999} = 1000$ 步——比 $\beta_1$ 的 10 步长 100 倍。这是有意设计的：梯度平方的方差比梯度本身的方差更大，需要更长的时间窗口才能获得稳定估计。

在最终的更新公式中，$v_t$ 出现在分母：
$$W_{t+1} = W_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

每个参数的有效学习率为 $\eta / \sqrt{\hat{v}_t}$。$v$ 大的参数（历史梯度大，该方向「陡峭」）→ 自动缩步长，避免 overshoot；$v$ 小的参数（历史梯度小，该方向「平坦」）→ 自动放大步长。这就是**自适应学习率**——不需要手动为每层调 $\eta$。

$\epsilon = 10^{-8}$ 防止除零（在 $v$ 趋近 0 的罕见情况下）。

**（三）偏差校正 —— 为什么用 $\hat{m}_t, \hat{v}_t$ 而非 $m_t, v_t$**

$m_0 = 0$, $v_0 = 0$。在训练刚开始时（$t=1,2,3,\ldots$），初始的几百步中递推的信大部分来自 0 而非真实梯度。

例如 $t=1$：$m_1 = \beta_1 \cdot 0 + (1-\beta_1) g_1 = (1-\beta_1) g_1$。如果不加修正直接用 $m_1$ 去更新，它的期望是 $(1-\beta_1)\mathbb{E}[g]$，比真实梯度小了 10 倍。训练前几步的步长会被严重低估。

Adam 的修正很简单：除以一个随时间衰减的因子，抵消初始化的偏差。
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

验证：$t=1$ 时 $1-\beta_1^1 = 0.1$，于是 $\hat{m}_1 = \frac{(1-\beta_1)g_1}{0.1} = g_1$ ✓。$t$ 很大时 $\beta_1^t \to 0$，$1-\beta_1^t \to 1$，修正自动消失——此时 $m_t$ 已经有足够的真实梯度累积，不需要修正了。

**（四）完整 Adam 算法（伪代码）**

```
m, v = 0, 0          # 每个参数初始化

for t = 1, 2, 3, ...:
    g = compute_gradient(batch)      

    m = β₁·m + (1-β₁)·g              # 一阶动量 —— 平滑方向
    v = β₂·v + (1-β₂)·g²             # 二阶动量 —— 估计陡峭程度

    m̂ = m / (1 - β₁ᵗ)                # 偏差校正 —— 修正早期低估
    v̂ = v / (1 - β₂ᵗ)

    W = W - η · m̂ / (√v̂ + ε)          # ε = 10⁻⁸
```

**（五）为什么 m 和 v 必须用 fp32——不能再省了**

$m_t$ 累积了成千上万步梯度。每一步的增量 $\Delta m = (1-\beta_1)(g_t - m_{t-1})$ 约在 $10^{-6} \sim 10^{-8}$。bf16 的相对精度约 $2^{-7} \approx 0.8\%$——这么小的增量加到 $m_{t-1}$ 上会被截断。

数值例子：$m_{t-1} = 0.01$（fp32），$\Delta m = 3\times 10^{-7}$。$m_t$ 应为 $0.0100003$。bf16 下 $0.01$ 的可表示精度约 $0.01 \times 2^{-7} \approx 8\times 10^{-5}$——$0.01$ 和 $0.0100003$ 在 bf16 下是无法区分的同一个数。$\Delta m$ 被吞。动量停止更新，Adam 退化回 SGD。

$v_t$ 同理。**因此 m 和 v 必须在 fp32 下累积。** 每个参数：m(4B) + v(4B) = 8 bytes。Qwen3-8B 共 82 亿参数 → m+v = **65.6 GB**，占 Model States 的 50%。

#### 5.2.3 Master Weights — 混合精度的精度锚点

前向/反向以 bf16 进行（利用 Tensor Core 加速），但 optimizer.step() 的更新量通常在 $10^{-6} \sim 10^{-8}$ 量级。

**如果直接在 bf16 参数上做加法会怎样？** 例如参数当前值是 $0.5$（bf16）。Adam 算出更新量是 $-3 \times 10^{-7}$，新值应为 $0.4999997$。但 bf16 只有 ~2 位有效十进制精度——$0.5 - 3\times 10^{-7}$ 在 bf16 下被截断，结果仍然是 $0.5$。更新被「吞」了。一万步这样的吞并累积起来，模型参数会严重漂移，收敛失败。

**解决方案**：维护一份 fp32 的 master weights（4 bytes/param）。所有 Adam 更新在 fp32 精度下进行，每次前向传播前将 master weights 截断到 bf16 供计算使用。同时获得 bf16 的计算加速和 fp32 的更新精度。

**所以混合精度训练的 Model States 和全 fp32 训练一样大**——都是 16 bytes/param（见 §1.3 对比）。混合精度省的不是 model states，是计算吞吐（Tensor Core bf16 比 fp32 快 ~15×）+ 激活值显存（bf16 存储比 fp32 省一半）。

#### 5.2.4 显存占比总结

| 组件                    | Bytes/param | 占 Model States | Qwen3-8B |
| --------------------- | ----------- | -------------- | -------- |
| 参数 + 梯度 (bf16)        | 4           | 25%            | 32.8 GB  |
| Adam m + v (fp32)     | 8           | **50%**        | 65.6 GB  |
| Master weights (fp32) | 4           | 25%            | 32.8 GB  |

**Adam 相关的三份 fp32 状态（m + v + master）合计 12 bytes/param = 75% 的 Model States。** 这就是 ZeRO 的第一刀砍向 optimizer states 的根本原因——只此一项省掉 75%，显存从 131 GB 骤降至 ~33 GB（ZeRO-1, Np=4），5.3 节的表给出了完整对比。

## 1.3 为什么全 fp32 训练也是 16 bytes/param

全 fp32 下，参数和梯度本身就是 fp32，优化器更新直接在 fp32 参数上做——不需要单独的 master weights。

| 组件 | 全 fp32 | bf16 混合精度 |
|------|--------|-------------|
| 参数 | 4 | 2 |
| 梯度 | 4 | 2 |
| Adam m | 4 | 4 |
| Adam v | 4 | 4 |
| Master weights | 0（不需要） | 4 |
| **合计** | **16** | **16** |

**混合精度不省 Model States。** 省的是计算吞吐（Tensor Core bf16 比 fp32 快 15×）和激活值显存（bf16 存储比 fp32 省一半）。

## 1.4 Residual States 的分类

Residual States 是单步训练内临时分配、用完即弃的显存。除了激活值（大头，第 2 章详述），还有两类常被忽略但实际存在的开销。

### 1.4.2 通信缓冲区（Communication Buffers）

分布式训练中，NCCL 的每次集合通信需要临时缓冲区来暂存待传输的数据。这些 buffer 不是模型状态的一部分——它们只在通信进行时存在，通信完成后立即回收。但它们的峰值占用是真实存在于显存中的。

**Gradient AllReduce（DDP / ZeRO-1/2）**：DDP 的 `Reducer` 将参数按 bucket 分组（典型 bucket size 为 25 MB）。每个 bucket 的梯度算完后，NCCL 需要一块与 bucket 等大的 send buffer 和一块 recv buffer。同时，在异步通信进行中，主计算流正在 backward 产生下一个 bucket 的梯度——这个过程中**至多 2 个 bucket 的 buffer 同时存在**（一个正在通信、一个正在被填充）。对 25 MB 的 bucket，通信 buffer 峰值约 **50 MB**。

**ZeRO-3 的 AllGather**：每次前向开始时，FSDP/ZeRO-3 需要 AllGather 拼回完整参数。这需要一块临时 buffer 放下全量参数分片的重组结果。对 Qwen3-8B 的完整参数（~16.4 GB bf16），$N_p=4$ 时每卡持有的分片为 $16.4/4 \approx 4.1$ GB，AllGather 结果需要完整的 16.4 GB buffer——但这块 buffer **复用**了参数存储空间（先释放分片、再 AllGather 填入完整参数），所以不会额外新增 16.4 GB 开销。实际额外开销来自 NCCL 内部的通信缓冲——约 **100-300 MB**。

**Pipeline Parallelism 的 send/recv**：PP 在层边界通过 P2P send/recv 传递 activation（大小 = $B \times S \times d_{model} \times 2$ bytes）。Qwen3-8B, B=1, S=2048 下约 16 MB。通常 PP 的通信 buffer 开销极小（< 100 MB），因为只需缓存少数几个 micro-batch 的激活值。

**总结**：通信 buffer 的峰值通常在 **200 MB ~ 2 GB**，取决于并行策略和 bucket 配置。ZeRO-3 + DDP 组合时各家框架的实现差异较大（FSDP2 的 DTensor 通信与 DDP 的 bucket 通信可能复用同一块 buffer），实际值需通过 `torch.cuda.memory_stats()` 实测。

### 1.4.3 显存碎片与框架开销

PyTorch 使用 **caching allocator** 管理 GPU 显存。当一个张量被释放时，allocator 并不立即调用 `cudaFree` 将显存归还给 CUDA driver——而是将这块内存标记为「空闲」，保留在自己的池中。后续的 `torch.empty()` 或算子输出分配新张量时，allocator 优先从池中复用已缓存的块，避免频繁的 GPU 驱动调用开销。

这个机制对性能有益（避免 cudaMalloc/cudaFree 的 driver 往返延迟），但也引入了一个隐蔽的显存开销：

**内部碎片**：allocator 按固定粒度（通常 512 bytes 或 2 MB 的块）分配显存。一个 100 KB 的张量会占据一个完整的 2 MB 块，其中 1.9 MB 被浪费。大量小张量（如每个参数独立的梯度、RMSNorm 的中间结果）累积的内部碎片可能达数 GB。

**外部碎片**：反复分配/释放不同大小的张量后，池中可能存在大量空闲空隙，但没有一块**连续**的空闲区域足够容纳一个大张量（如 attention score 矩阵 $[32, S, S]$）。这导致即使 `torch.cuda.memory_allocated()` 报告的值远小于 HBM 容量，一个大张量的分配仍然抛出 OOM。

**cuBLAS workspace**：每次 GEMM 调用，cuBLAS 需要在 GPU 上分配临时 workspace 用于存储中间 tile。典型大小在 **几十 MB 到数百 MB**。这些 workspace 由 cuBLAS 内部管理，不经过 PyTorch allocator，因此不会出现在 `torch.cuda.memory_allocated()` 的统计中——但它们实实在在占用了 HBM。

**NCCL internal buffers**：NCCL 内部维护的环形通信缓冲区（用于 Ring AllReduce 的分段传递）、NVLink peer memory registration 等。大小取决于 world size 和 GPU 拓扑，通常 **100-500 MB**。



---

## 1.5 推理 vs 训练的显存

---

# 第 2 章：激活值显存逐层分解

> **知识定位**：Model States 讲了「参数本身就占的显存」，这章讲「为了训练需要额外保存的中间结果」。激活值的量级由 B、S、d、L 决定——先推导通用公式，再用 Qwen3-8B 验证。

## 2.1 激活值是什么、为什么必须保存

考虑一个简单的两层计算：
$$z = W_2 \cdot \text{ReLU}(W_1 \cdot x)

$$

反向传播计算$\frac{\partial L}{\partial W_2}$时需要$\text{ReLU}(W_1 \cdot x)$的值。计算$\frac{\partial L}{\partial W_1}$时需要$x$、$\frac{\partial L}{\partial z}$、以及$W_1 \cdot x$的 pre-activation（ReLU 在$x<0$处梯度为 0，需要知道哪些元素是激活的）。

**激活值 = 前向传播过程中产生的、反向传播需要消费的中间张量。** 每经过一个算子，都可能产生需要保留的中间结果。默认框架保存所有激活值（避免重算开销）——这就是本章分析的对象。重算的反面（Activation Checkpointing）见第 3 章。

## 2.2 通用推导：哪些操作产生激活值

一个 Transformer Block 中，需要保存中间结果的操作：

| 操作类型 | 反向为什么需要它 |
|---------|----------------|
| `nn.Linear`（参数化 matmul） | 输入$X$—— 因为$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$|
| 无参数 matmul（$QK^T$、$P \cdot V$） | 两个输入$A$和$B$|
| SiLU | pre-activation$x$——$\frac{\partial \text{SiLU}}{\partial x}$依赖$x$|
| RMSNorm | 原始输入$\mathbf{x}$|
| 逐元素乘法$\odot$| 两个输入之一（反向需要另一个输入的值） |
| Softmax | scores$s$（或 softmax 后的$p$） |

**关键观察**：激活值来源分为两类——(a) **matmul 的输入张量**（形状即参与 matmul 的张量的维度），(b) **激活函数的输入**（形状通常是 matmul 的输出）。前者是显存的主体。

本章假定激活值以 **bf16（2 bytes）** 存储。

## 2.3 Attention 层 — 通用激活值公式

### 2.3.1 前向数据流

```
X → RMSNorm → Q_proj → Q  ─┐
             → K_proj → K  ─┤→ Scores = QK^T/√d → Softmax → P → P·V → O_proj → Output
             → V_proj → V  ─┘
```

需要保存的张量：

| # | 张量 | 形状 | 为什么需要 |
|---|------|------|-----------|
| ① | RMSNorm 输入 |$[B, S, d]$| RMSNorm 反向需要原始$\mathbf{x}$|
| ② | Q |$[B, n_h, S, d_h]$|$QK^T$matmul 的输入 |
| ③ | K |$[B, n_{kv}, S, d_h]$|$QK^T$matmul 的输入 |
| ④ | V |$[B, n_{kv}, S, d_h]$|$PV$matmul 的输入 |
| ⑤ | Scores（$QK^T$结果） |$[B, n_h, S, S]$| Softmax 反向需要 |
| ⑥ |$P = \text{softmax}(\text{Scores})$|$[B, n_h, S, S]$|$PV$matmul 的输入 |
| ⑦ |$PV$结果 |$[B, n_h, S, d_h]$| Output 投影的输入（concat 前） |
| ⑧ | Output 投影输入（concat 后） |$[B, S, d]$| O 投影反向需要 |

**⑤ 和 ⑥ 是两个$S^2$张量**——整个 Transformer 中最大的单层激活值。FlashAttention（§6.3.3）通过分块计算回避了它们的 HBM 物化。

### 2.3.2 逐项显存的通用公式

①②③④⑦⑧ 各张量的记忆 = 元素数量$\times 2$bytes。

①$M_{norm\_in} = 2 B S d$

②$M_Q = 2 B n_h S d_h = 2 B S d$（因$n_h d_h = d$）

③$M_K = 2 B n_{kv} S d_h$

④$M_V = 2 B n_{kv} S d_h = M_K$

⑤$M_{scores} = 2 B n_h S^2$**（$S^2$项——最大单张激活值）**

⑥$M_P = 2 B n_h S^2$同 Scores

⑦$M_{PV} = 2 B S d$（$n_h d_h = d$，concat 前元素总数不变）

⑧$M_{o\_in} = 2 B S d$

**一层 Attention 激活值（不使用 FA）**：
$$

\begin{aligned}
M_{attn}^{no\_FA} &= ①+②+③+④+⑤+⑥+⑦+⑧ \\[4pt]
&= 2BSd + 2BSd + 2BS(n_{kv}d_h) + 2BS(n_{kv}d_h) + 2Bn_hS^2 + 2Bn_hS^2 + 2BSd + 2BSd \\[4pt]
&= 2BS(3d + 2 n_{kv} d_h) + 4 B n_h S^2
\end{aligned}

$$GQA 4:1（$n_{kv} = n_h/4, n_{kv} d_h = d/4$）：$$

M_{attn}^{no\_FA,GQA\;4:1} = 2BS \cdot 3.5d + 4 B n_h S^2

$$Dense MHA（$n_{kv} = n_h, n_{kv} d_h = d$）：$$

M_{attn}^{no\_FA,MHA} = 2BS \cdot 5d + 4 B n_h S^2

$$

### 2.3.3 FlashAttention 的激活值节省

FlashAttention 不在 HBM 中物化$[B, n_h, S, S]$的 Scores 和 Softmax 矩阵。Q、K、V 切成 tile 在 SRAM 中处理，仅维护$(m, l)$两个 online softmax 统计量（形状$[B, n_h, S]$，远小于$S^2$）。

使用 FA 后：
- ⑤$M_{scores}$**移除**（$2 B n_h S^2 \to 0$）
- ⑥$M_P$**移除**（$2 B n_h S^2 \to 0$）
- 额外存储$(m, l)$≈$2 \cdot 2 B n_h S$bytes（对 S=2048 约 0.5 MB，忽略）

**一层 Attention 激活值（使用 FA）**：
$$

M_{attn}^{FA} = 2BS(3d + 2 n_{kv} d_h)

$$GQA 4:1：$$

 M_{attn}^{FA,GQA\;4:1} = 2BS \cdot 3.5d

$$

> **例：Qwen3-8B（FA, GQA 4:1, B=1, S=2048, d=4096, d_h=128, n_{kv}=8）**。
>
> 通用公式：$2 \cdot 1 \cdot 2048 \cdot 3.5 \cdot 4096 = 58.7$MB。
> 逐项验证：① norm 输入 =$1 \cdot 2048 \cdot 4096 \cdot 2 = 16$MB。② Q =$1 \cdot 32 \cdot 2048 \cdot 128 \cdot 2 = 16$MB。③ K =$1 \cdot 8 \cdot 2048 \cdot 128 \cdot 2 = 4$MB。④ V = 4 MB。⑦ PV 结果 =$1 \cdot 32 \cdot 2048 \cdot 128 \cdot 2 = 16$MB。⑧ O 投影输入 =$1 \cdot 2048 \cdot 4096 \cdot 2 = 16$MB。合计 =$16+16+4+4+16+16 = 72$MB。⑦⑧ 可能共享内存（同一数据的不同视图），框架实际开销在 59−72 MB 之间。

## 2.4 FFN 层（SwiGLU）— 通用激活值公式

### 2.4.1 前向数据流

```
X → RMSNorm → gate_proj → g → SiLU → gated ─┐
            → up_proj   → u ───────────────⊙──→ down_proj → Output
```

需要保存的张量：

| # | 张量 | 形状 | 为什么需要 |
|---|------|------|-----------|
| ① | RMSNorm 输入 |$[B, S, d]$| RMSNorm 反向需要 |
| ② | gate_proj 输出$g$|$[B, S, d_{ff}]$| SiLU 反向需要 pre-activation |
| ③ | up_proj 输出$u$|$[B, S, d_{ff}]$|$\odot$反向需要另一个输入 |
| ④ | gated 结果 |$[B, S, d_{ff}]$| down_proj matmul 的输入 |

（④ 和 down_proj 输入是同一块内存，框架会复用。）

### 2.4.2 逐项显存的通用公式

①$M_{norm\_in} = 2 B S d$

②$M_{gate\_out} = 2 B S d_{ff}$

③$M_{up\_out} = 2 B S d_{ff}$

④$M_{gated} = 2 B S d_{ff}$

**一层 FFN 激活值**：
$$

M_{ffn} = 2BSd + 2BSd_{ff} + 2BSd_{ff} + 2BSd_{ff} = 2BS(d + 3d_{ff})

$$

> **例：Qwen3-8B（d=4096, d_ff=12288, B=1, S=2048）**。
>$M_{ffn} = 2 \cdot 1 \cdot 2048 \cdot (4096 + 3 \cdot 12288) = 2 \cdot 2048 \cdot 40960 = 160$MB。
> 逐项验证：① 16 MB + ② 48 MB + ③ 48 MB + ④ 48 MB = 160 MB ✓。

## 2.5 全模型激活值汇总

### 2.5.1 通用公式（Pre-LN, FA, GQA 4:1）

每层 Block：$M_{block} = M_{attn}^{FA} + M_{ffn}$
$$

M_{block} = 2BS(3d + 2 n_{kv} d_h) + 2BS(d + 3d_{ff})

$$

LM Head 输入：$M_{lm\_in} = 2BSd$

额外 RMSNorm 输入（与 Attention/FFN 的 norm 输入有重叠，保守额外计 L 份）：$\approx 2BSd \cdot L$

Residual states：$\approx 4BSd \cdot L$

**全模型激活值**：
$$

\begin{aligned}
M_{activations} &\approx L \cdot 2BS(3d + 2 n_{kv} d_h + d + 3d_{ff}) \;+\; 2BSd \;+\; 2BSd \cdot L \;+\; 4BSd \cdot L \\[4pt]
&= L \cdot 2BS(4d + 2 n_{kv} d_h + 3d_{ff}) \;+\; 2BSd \;+\; 4BSd \cdot L
\end{aligned}

$$GQA 4:1 下$2 n_{kv} d_h = d/2$：$$

M_{activations}^{GQA\;4:1} \approx 2BS \cdot L \cdot (4.5d + 3d_{ff}) \;+\; 2BSd \;(+\; 4BSd \cdot L)

$$

### 2.5.2 Qwen3-8B 数值（B=1, S=2048, bf16, FA）

| 组件 | 通用公式 | 单层 | ×36 |
|------|---------|------|-----|
| Attention（FA, GQA） |$2BS(3d + 2 n_{kv} d_h)$| ~59−72 MB | ~2.1−2.6 GB |
| FFN |$2BS(d + 3d_{ff})$| 160 MB | 5.76 GB |
| LM Head 输入 |$2BSd$| 16 MB | 16 MB |
| 额外 norm 输入 |$2BSd$per | ~16 MB | ~0.6 GB |
| Residual states | 估算 | | ~0.5 GB |
| **总计** | | | **≈ 9.0 GB** |

（注：Attention 行取 59 MB 时全模型 ≈ 8.6 GB；取 72 MB 时 ≈ 9.0 GB。具体差异取决于框架内存复用策略。）

**不使用 FlashAttention**：⑤+⑥ 恢复$4 B n_h S^2 = 4 \cdot 1 \cdot 32 \cdot 2048^2 = 512$MB/层。36 层 = +18.4 GB → **总计 ≈ 27.4 GB**。FA 节省 67%。

### 2.5.3 激活值与 S、B 的标度关系

从通用公式直接读出各项对 S 和 B 的依赖：

| 组件 | S 依赖 | B 依赖 | S 翻倍时 |
|------|--------|--------|---------|
| Q, K, V 等线性项 |$\propto S$|$\propto B$| ×2 |
| Scores + Softmax（无 FA） |$\propto S^2$|$\propto B$| **×4** |
| FFN 激活值 |$\propto S$|$\propto B$| ×2 |
| Norm 输入 |$\propto S$|$\propto B$| ×2 |

**关键洞察**：$S^2$项是长序列的头号敌人。S=2048 时 FA 已将其消除；S 继续增大 FA 依然有效（tile 大小恒定）。如果不用 FA，S=32768 时 Scores 矩阵 =$2 \cdot 32 \cdot 32768^2 \approx 64$GB **per layer**——一层的 Scores 就已超单卡 HBM 容量。

---

# 第 3 章：Activation Checkpointing 的数学

> **知识定位**：第 2 章算出了激活值约 9 GB（FA 下）。这章讲怎么把这个数字再压下去——用额外的计算换显存。AC 是 ZeRO/TP 的天然互补：ZeRO 省 Model States（第 1 章），AC 省 Residual States（本章）。

## 3.1 基本思想：不存，需要时重算

### 3.1.1 什么是 Checkpointing

标准训练在每一步前向传播时，把$L$层中所有算子产生的中间激活值全部存下来（第 2 章分析的那份清单），供反向传播使用。总激活值 ≈ 9 GB。

Activation Checkpointing 修改了前向阶段的存储策略：

- 把$L$层分成若干**段（segment）**，每段包含连续的若干层
- 每段的**输入** hidden states 被保存下来（称为 checkpoint）
- 段**内部**的所有激活值（Q, K, V, gate 输出, up 输出, gated 结果等）**全部丢弃**

反向传播到达某段时：
1. 从该段的 checkpoint（保存好的输入）开始，**重新做一遍前向**——这次把中间激活值产出来
2. 用刚产出的激活值做反向传播
3. 该段的反向完成后，这些临时激活值也被丢弃

**代价**：每段被额外计算了一次前向——用 FLOPs 换显存。

### 3.1.2 一个具体例子

假设模型有 4 层，分 2 段（每段 2 层）：

```
Segment 0（层 0−1）:
  Forward: 层0→层1（保存 input_0 作为 checkpoint，丢弃内部激活值）→ output_1
  Backward 时: 从 input_0 重新 forward 层0→层1（产出内部激活值）→ backward 层1→层0

Segment 1（层 2−3）:
  Forward: 层2→层3（保存 input_2 作为 checkpoint，丢弃内部激活值）→ output_3
  Backward 时: 从 input_2 重新 forward 层2→层3（产出内部激活值）→ backward 层3→层2
```

每层被额外多算了一次前向。4 层共多算了 4 次额外前向 = 1 次完整前向 → +100% 前向 FLOPs。

**如果每层都设 checkpoint（segment size = 1）**：每层的 checkpoint 就是该层的输入 hidden state。全模型保存的是$L+1$个边界 hidden states（$[B, S, d]$），所有层内部的激活值全部丢弃。反向时每层从自己的 checkpoint 重算一次前向。额外 FLOPs = 同样 1 次完整前向 = +100% 前向 FLOPs。

**关键观察**：无论 segment 多大，额外 FLOPs 都是恰好 1 次完整前向——因为每层恰好被多算 1 次（一次在初始前向，一次在重算时）。所以相对于总训练 FLOPs：
$$C_{total}^{AC} = C_{fwd}^{normal} + C_{bwd} + C_{fwd}^{recompute}$$

$$= 2ND + 4ND + 2ND = 8ND$$

也就是$6ND \to 8ND$，**增加 33%**。**AC 的额外 FLOPs 比例不随 segment size 变化——始终是 +33%。**

### 3.1.3 省了多少显存？

AC 的显存节省来源非常明确：段内部的激活值被丢弃。保留下来的只有每段的输入 checkpoint——形状为$[B, S, d]$的 hidden states，每段一份。

对于 Qwen3-8B（L=36, B=1, S=2048, FA），取 segment size = 1（每层 checkpoint）：

**保留**：$L+1 = 37$个边界 hidden states。每个 =$2 B S d = 2 \cdot 1 \cdot 2048 \cdot 4096 = 16$MB。总计 ≈$37 \times 16 = 0.59$GB。

**丢弃**：所有层内部的激活值。根据 §6.5，总激活值 ≈ 9.0 GB，去掉约 0.6 GB 的边界 hidden states 后，约 8.4 GB 内部激活值被丢弃。

**结论**：segment size = 1 时，激活值从 9.0 GB → ~0.6 GB，节省 **93%**。

取 segment size = 2（每 2 层 checkpoint）：保留$L/2 + 1 = 19$个边界 hidden states ≈ 0.30 GB。丢弃约 8.1 GB 内部激活值（段边界从每层变为每 2 层，段内部的第 2 层不再丢弃其输入——仍被保留为 checkpoint 的一部分，但这不是内部激活值。实际上段大小只是改变了 checkpoint 的密度）。节省约 90%。

**无论 segment size 多少，FLOPs 代价始终是 +33%。** 显存节省随 segment size 减小而增大（checkpoint 越密 → 丢弃得越彻底 → 保留得越少）。segment size = 1 是最极端的省显存配置。

## 3.2 选择性 Checkpointing：不是所有层都需要 checkpoint

并不是模型中的每层都产生同样多的激活值。回顾 §6.3 和 §6.4：

| 模块 | 激活值（FA, GQA, S=2048） | 占全层激活值 | FLOPs 占全层 |
|------|-------------------------|------------|------------|
| Attention（含 norm 输入） | ~72 MB | 31% | 28% |
| FFN（含 norm 输入） | ~160 MB | 69% | 72% |

**Attention-only AC**：只 checkpoint attention 层，FFN 层的激活值正常保留。

- FLOPs 代价：只有 attention 被重算 → 额外 FLOPs ≈ 28% 的前向 FLOPs → 总训练 FLOPs 增加约 9%
- 显存节省：丢弃 attention 内部激活值（~56 MB/层，去掉 norm 输入 16 MB 后）× 36 ≈ 2.0 GB

**适合场景**：不使用 FlashAttention 时（Scores 256MB/层），attention-only AC 能省掉 512MB/层（Scores + Softmax 各 256MB）→ 36 层 = 18.4 GB。但在使用 FA 后，attention 激活值已经被压缩到 ~72 MB/层，attention-only AC 的性价比下降——省 ~2 GB 换 +9% FLOPs，看场景权衡。

## 3.3 AC 不能省什么

AC 只动激活值（Residual States），**不碰 Model States**。

这一点需要非常清楚：就算你把 segment size 设为 1（每层 checkpoint），激活值几乎降到 0，Model States 依然是 131.2 GB 纹丝不动（参数 16.4 + 梯度 16.4 + Adam m 32.8 + Adam v 32.8 + master 32.8）。**AC 省的是 ~9 GB 的激活值，不是 131 GB 的 Model States。**

这就是为什么 AC 永远不是独立解药。它在分布式训练中的正确位置是：

- **ZeRO-3** 把 Model States 从 131 GB → 131/Np GB（Np=4 → 33 GB）
- **AC** 把激活值从 9 GB → ~1 GB
- **组合**：每卡 33 + 1 ≈ 34 GB，在 A100-80G 中轻松训练

---

# 第 4 章：Gradient Accumulation 的数学

> **知识定位**：AC 解决了激活值过高，GA 解决了一个不同的问题——需要大 effective batch size 但显存放不下。

## 4.1 基本机制

传统训练：batch_size = 64，前向+反向，显存 peak 在批量激活值。

GA 训练：把 64 = 8 × 8（accululation_steps = 8, micro_batch_size = 8）：

```
for step in range(accumulation_steps):
    loss = model(micro_batch_8) / 8  # 除以 accumulation_steps
    loss.backward()  # 梯度在 param.grad 中累加

# M 步完成后
optimizer.step()
optimizer.zero_grad()
```

**Peak memory** = 一个 micro_batch_8 的激活值（而非 batch_64 的大激活值）。Model States 不变。

**Effective batch size** = micro_batch_size × accumulation_steps × DP_size（当使用 DDP 时）。

## 4.2 为什么 loss 需要除以 accumulation_steps

如果不除：`loss.backward()` 的梯度 = micro_batch 的平均梯度。累加 8 次后 = 8 × micro_batch 平均梯度的和。而一次大 batch 的 backward 产生的是大 batch 的平均梯度。

所以要除以 accumulation_steps，使得 8 个 micro_batch 的累加梯度 ≈ 大 batch 的梯度。

## 4.3 GA + AC + ZeRO 的组合

这是现代 LLM 训练的标配组合：

| 技术 | 省什么 | 省多少（8B model） | 代价 |
|------|--------|-------------------|------|
| GA | 激活值 peak（用 micro_batch 代替大 batch） | 激活值 ∝ 1/accumulation_steps | 训练时间 ∝ accumulation_steps |
| AC | 激活值存储（跨步间的 checkpoint） | 激活值 ∝ 1/checkpoint_interval | FLOPs +17-33% |
| ZeRO-3 | Model States（参数、梯度、Adam） | 131 GB → 131/Np GB | 通信 +50% |

三者互不重叠：
- GA 省的是**一次 micro-step 内的**激活值（因为 micro_batch 小）
- AC 省的是**跨层间的**激活值（不存 checkpoint 间）
- ZeRO-3 省的是**跨 step 间持久**的 model states

---



# 第 5 章：完整案例 — Qwen3-8B 显存账单

> **本章回答什么问题**：把前 4 章的显存公式套到 Qwen3-8B 上，算出一张完整的显存账单，并给出多组 (bs, seq) 配置下的对比。

## 5.1 模型参数速查

（同算力笔记附录 A，此处重复以保持独立可读。）

| 参数 | 值 |
|------|-----|
| $d_{model}$ | 4096 |
| $d_{ff}$ | 12288 |
| $n_{heads}$ (Q) | 32 |
| $n_{kv\_heads}$ | 8 (GQA 4:1) |
| $d_{head}$ | 128 |
| $n_{layers}$ | 36 |
| $V$ | 152064 |
| 总参数量 | ~8.2B |
| 训练总 Model States | 131.2 GB (16 bytes/param) |
| 激活值 (FA) | ~9 GB ($B=1,S=2048$) |

## 5.2 显存完整账单（单卡 bf16 训练）

### 不使用 AC / GA / ZeRO（naive DP 单卡）

| 组件                    | 大小             |
| --------------------- | -------------- |
| 参数 (bf16)             | 16.4 GB        |
| 梯度 (bf16)             | 16.4 GB        |
| Adam m (fp32)         | 32.8 GB        |
| Adam v (fp32)         | 32.8 GB        |
| Master weights (fp32) | 32.8 GB        |
| **Model States 小计**   | **131.2 GB**   |
| 激活值 (FA)              | ~10 GB         |
| 激活值 (no FA)           | ~19 GB         |
| **总计 (FA)**           | **141.2 GB** ⛔ |
| **总计 (no FA)**        | **150.2 GB** ⛔ |

### 加入 AC (每 2 层 checkpoint, FA)

| 组件 | 大小 |
|------|------|
| Model States | 131.2 GB |
| 激活值 (AC C=2) | ~5 GB |
| **总计** | **136.2 GB** ⛔ |

### 加入 AC + ZeRO-3 (Np=4, FA)

| 组件                      | 大小             |
| ----------------------- | -------------- |
| 参数 (bf16, 1/4)          | 4.1 GB         |
| 梯度 (bf16, 1/4)          | 4.1 GB         |
| Adam m (fp32, 1/4)      | 8.2 GB         |
| Adam v (fp32, 1/4)      | 8.2 GB         |
| Master (fp32, 1/4)      | 8.2 GB         |
| **Model States 小计**     | **32.8 GB**    |
| 激活值 (AC+FA)             | ~5 GB          |
| ZeRO-3 AllGather buffer | ~1 GB          |
| **总计 (per GPU)**        | **~38.8 GB** ✅ |

Np=4 时，每卡 38.8 GB——正好在 A100-80G / H100-80G 的舒适区内。

## 5.3 多组 (bs, seq) 配置对比

| B | S | Model States | 激活值 (FA) | 创建值 (no FA) | Total (FA+AC+Z3 Np=4) | 结论 |
|---|----|-------------|-----------|-------------|----------------------|------|
| 1 | 512 | 32.8 GB | ~2.5 GB | ~5 GB | ~35.3 GB | ✅ |
| 1 | 2048 | 32.8 GB | ~5 GB | ~19 GB | ~37.8 GB | ✅ |
| 1 | 8192 | 32.8 GB | ~12 GB | ~76 GB | ~44.8 GB | ✅ |
| 1 | 32768 | 32.8 GB | ~40 GB | ~304 GB | ⛔ (even with Z3) | ❌ |
| 8 | 2048 | 32.8 GB | ~8 GB | ~32 GB | ~40.8 GB | ✅ |
| 64 | 2048 | 32.8 GB | ~20 GB | ~80 GB | ⛔ | ❌ |

## 5.4 扩展到 70B 和 175B

| | Qwen3-8B | LLaMA 3 70B | LLaMA 3 175B (hypothetical) |
|---|---------|------------|---------------------------|
| 参数量 | 8.2B | 70B | 175B |
| Model States (单卡) | 131 GB | 1120 GB | 2800 GB |
| 单卡推理 | 16.4 GB ✅ | 140 GB ⛔ | 350 GB ⛔ |
| Np 需要 (ZeRO-3 only, 80GB) | 2 | 14 | 35 |
| 推荐策略 | ZeRO-3, DP=8 | TP=8+ZeRO-3 | TP=8+PP+ZeRO-3 |

---



# 第 6 章：推理 vs 训练显存

## 6.1 推理 vs 训练的显存

|                | 推理                                                        | 训练                             |
| 

# 附录 A：模型参数速查表

| 参数 | Qwen3-8B | LLaMA 3 8B | LLaMA 3 70B | DeepSeek V4 Flash |
|------|---------|-----------|------------|------------------|
| $d_{model}$ | 4096 | 4096 | 8192 | 4096 |
| $d_{ff}$ | 12288 | 14336 | 28672 | 2048 (expert) |
| $n_{heads}$ (Q) | 32 | 32 | 64 | 64 |
| $n_{kv}$ | 8 | 8 | 8 | 1 (MQA) |
| $d_{head}$ | 128 | 128 | 128 | 512 (MLA) |
| $n_{layers}$ | 36 | 32 | 80 | 43 |
| $V$ | 152064 | 128256 | 128256 | 129280 |
| 总参数量 | ~8.2B | ~8.0B | ~70B | 284B (13B active) |
| 架构类型 | Dense, GQA | Dense, GQA | Dense, GQA | MoE, MLA |
| 发布时间 | 2026.4 | 2024.4 | 2024.4 | 2026.1 |
| 来源 | Alibaba | Meta | Meta | DeepSeek |



# 附录 B：推导假定与适用范围

## B.1 假定总结

本手册的所有推导基于以下假定。偏离这些假定时，需要相应调整：

1. **bf16 混合精度 + Adam**：如果使用 fp16（需要 loss scaling）、SGD（不需要 Adam states）、Adafactor（无 momentum）或 LAMB（逐层自适应 LR），Model States 的 16 bytes/param 需要重新计算
2. **FlashAttention**：激活值计算假定使用了 FlashAttention（Scores/Softmax 不物化）。如不使用，激活值约翻倍
3. **SwiGLU FFN**：如果使用 ReLU FFN（$W_1(W_2x) + W_3x$ 形式，如原始的 T5 FFN），FFN 的 FLOPs 应为 $2BS \times (d_{model} \cdot d_{ff}) \times 2$（只有 2 个 matmul ∨ 3 个），而非 3×
4. **Pre-LN 架构**：Qwen3-8B 使用 Pre-LayerNorm（RMSNorm 在每个 sublayer 之前，而非之后）。Post-LN 架构下激活值的 shape 可能稍有不同
5. **无 MoE**：dense 模型假定。MoE 模型的 FLOPs = dense FFN FLOPs × (active_experts/total_experts) × routing_overhead
6. **S ≥ 512**：本文的「softmax 可忽略」论证在 S ≥ 512 的前提下成立。S < 256 时，softmax 的相对占比可能超过 5%，需要更精确的建模

## B.2 常见替代场景的快速修正

**场景：全 fp32 训练**
- Model States = 16 bytes/param（不变——参数 4 + 梯度 4 + Adam m 4 + Adam v 4 = 16；无 master weights）
- 激活值 = ×2（所有中间结果用 fp32 存储）
- FLOPs = 不变（matmul 的 FLOPs 只取决于矩阵维度，不取决于元素精度）

**场景：fp16 混合精度**
- Model States = 16 bytes/param（不变——和 bf16 相同）
- 额外负担：GradScaler 的动态 scale adjustment（对 FLOPs 和显存增量 < 0.1%）
- 可能出现的问题：scale 太小时梯度下溢 → 部分参数得不到更新 → 收敛变慢

**场景：推理（inference）**
- Model States = 参数(2N) + KV Cache
- KV Cache = $2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times S$ bytes
- Qwen3-8B, S=2048: KV Cache = $2 \times 36 \times 8 \times 128 \times 2048 = 144$ MB（总）
- 总推理显存 = 16.4 GB + 0.144 GB ≈ 16.5 GB（远小于训练的 141 GB）

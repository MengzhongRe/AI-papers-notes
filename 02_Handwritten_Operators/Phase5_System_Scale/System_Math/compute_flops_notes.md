# Transformer 算力建模手册

> **定位**：从矩阵乘法 FLOPs 公理出发，推导 Transformer 每一步前向+反向的精确计算量。不涉及显存。
> **参考模型**：Qwen3-8B（2026.4, Alibaba）— $d_{model}=4096$, $d_{ff}=12288$, $n_{heads}=32$ (Q), $n_{kv\_heads}=8$ (GQA 4:1), $d_{head}=128$, $n_{layers}=36$, $V=152064$, 总参数量 ~8.2B。
> **关联笔记**：[memory_modeling_notes.md](memory_modeling_notes.md) — 显存建模 · [distributed_training_handbook.md](../distributed_training_handbook.md) — 分布式全局手册

---

## 目录

- [第 1 章：矩阵乘法的 FLOPs 计算](#第-1-章矩阵乘法的-flops-计算)
- [第 2 章：Transformer 前向 FLOPs 逐层分解](#第-2-章transformer-前向-flops-逐层分解)
- [第 3 章：为什么这些操作可以忽略](#第-3-章为什么这些操作可以忽略)
- [第 4 章：反向传播与 6ND 公式](#第-4-章反向传播与-6nd-公式)
- [第 5 章：完整案例 — Qwen3-8B FLOPs 汇总](#第-5-章完整案例--qwen3-8b-flops-汇总)
- [第 6 章：训练时间估算](#第-6-章训练时间估算)
- [附录 A：模型参数速查表](#附录-a模型参数速查表)
- [附录 B：常用 GPU 规格速查](#附录-b常用-gpu-规格速查)
- [附录 C：关键公式速查卡](#附录-c关键公式速查卡)

---

# 第 1 章：矩阵乘法的 FLOPs 计算

> **知识定位**：这是全书的公理。理解了矩阵乘法的 FLOPs 怎么算，Transformer 每一层的 FLOPs 都是它的直接推论。

## 1.1 内积的 FLOPs

两个长度为 $d$ 的向量 $\mathbf{x}, \mathbf{w} \in \mathbb{R}^{d}$ 的内积定义为：

$$y = \sum_{k=1}^{d} x_k \cdot w_k$$

这涉及 $d$ 次乘法 + $d$ 次加法（严格说是 $d-1$ 次加法，但在大 $d$ 近似下 $d-1 \approx d$）= **$2d$ 次浮点运算（FLOPs）**。

### MAC 与 FLOPs 的区别

一个常见混淆点：现代 GPU 的 Tensor Core 将「乘加」作为一条硬件指令（MAC: Multiply-Accumulate），1 MAC = 1 次乘法 + 1 次加法。如果按 MAC 计数，内积的操作数是 $d$ MAC。如果按 FLOPs 计数，是 $2d$ FLOPs。

**本手册统一使用 FLOPs = 2 × MAC**。这意味着：
- 如果你看到某篇论文说「前向 FLOPs ≈ ND」，它可能用的是 MAC 计数（即 $1ND$ MAC = $2ND$ FLOPs）
- 如果你看到「6ND」，用的是 FLOPs（即 $3ND$ MAC = $6ND$ FLOPs）

**验证**：一个 $2 \times 3$ 矩阵乘 $3 \times 4$ 矩阵：
- 输出有 $2 \times 4 = 8$ 个元素
- 每个元素由长度为 3 的内积得到：3 次乘法 + 3 次加法 = 6 FLOPs
- 总 FLOPs = 8 × 6 = $2 \times 2 \times 3 \times 4 = 48$ FLOPs ✓

## 1.2 矩阵乘法的 FLOPs

推广到一般形式：$Y = X \cdot W$，其中 $X \in \mathbb{R}^{M \times K}$，$W \in \mathbb{R}^{K \times N}$，$Y \in \mathbb{R}^{M \times N}$。

输出元素个数：$M \times N$。每个元素的计算量：内积长度 $K$ → $2K$ FLOPs。
$$\text{FLOPs}_{matmul}(M \times K, K \times N) = 2 \times M \times K \times N$$

### 验证维度正确性

$2 \times M \times K \times N$ 的单位是 FLOPs（无量纲的浮点运算次数）。$M, K, N$ 是矩阵的各维度。如果把 $M = B \times S$（batch × seq_len）代入，得 $2 \times B \times S \times K \times N$。

在 Transformer 中，对于 $\text{nn.Linear}(4096, 12288)$：
- $K = 4096$, $N = 12288$
- $M = B \times S$
- $\text{FLOPs} = 2 \times B \times S \times 4096 \times 12288$

### 一次前向中所有参数化矩阵乘法的 FLOPs 之和

观察：对于第 $\ell$ 层有 `nn.Linear` 权重的矩阵乘法 $W^{(\ell)} \in \mathbb{R}^{d_{in}^{(\ell)} \times d_{out}^{(\ell)}}$，FLOPs = $2 \times B \times S \times d_{in}^{(\ell)} \times d_{out}^{(\ell)}$。

而该权重的参数量 = $d_{in}^{(\ell)} \times d_{out}^{(\ell)}$。因此对于**参数化的矩阵乘法**：
$$\text{FLOPs}_{matmul}^{(\ell)} = 2 \times B \times S \times \text{params}^{(\ell)}_{matmul}$$

将全模型所有 `nn.Linear` 的参数求和得 $N_{matmul}$，参数化 matmul 的前向总 FLOPs = $2 \times B \times S \times N_{matmul}$。由于 $N_{matmul} \approx N$，可写为 $C_{fwd} \approx 2ND$。

### 一个关键问题：Attention 里无参数的矩阵乘法被漏掉了

Attention 层里有两次巨大的矩阵乘法，**它们没有对应的 `nn.Linear` 权重**：

1. **$QK^T$（Attention Scores）**：纯激活值之间的乘法，不涉及任何权重矩阵。FLOPs = $2 B n_h S^2 d_h$。
2. **$\text{softmax}(A) \cdot V$（Weighted Sum）**：同样是激活值之间。FLOPs = $2 B n_h S^2 d_h$。

这两次乘法的 FLOPs **不被** $2BS \times \text{params}$ 捕获，因为 params = 0 而 FLOPs ≠ 0。

**以 Qwen3-8B S=2048 验证**，一层 Attention 的逐项：

| 子操作 | 有 nn.Linear 权重？ | params | 实际 FLOPs | $2BS \times \text{params}$ |
|--------|-------------------|--------|-----------|--------------------------|
| Q 投影 | 是 | $4096^2$ | $6.87 \times 10^{10}$ | $6.87 \times 10^{10}$ ✓ |
| K 投影 | 是 | $4096 \times 1024$ | $1.72 \times 10^{10}$ | $1.72 \times 10^{10}$ ✓ |
| V 投影 | 是 | 同 K | $1.72 \times 10^{10}$ | $1.72 \times 10^{10}$ ✓ |
| **$QK^T$（无权重！）** | **否** | **0** | $3.44 \times 10^{10}$ | **0** ✗ |
| **$AV$（无权重！）** | **否** | **0** | $3.44 \times 10^{10}$ | **0** ✗ |
| O 投影 | 是 | $4096^2$ | $6.87 \times 10^{10}$ | $6.87 \times 10^{10}$ ✓ |

参数化 matmul 小计（$2BS \times \text{params}$ 覆盖的）：$1.72 \times 10^{11}$ FLOPs
无参数 matmul（$2BS \times \text{params}$ 漏掉的）：$6.87 \times 10^{10}$ FLOPs

**对于 Attention 层，$2BS \times \text{params}$ 低估了 $6.87 \times 10^{10} / 1.72 \times 10^{11} \approx 40\%$ 的计算量。** 这根本不是可忽略的小误差——单层漏掉 40%，如果累积到 36 层就是大数。

### 那为什么 $C_{fwd} \approx 2ND$ 整体上还是对的？

因为有两个方向相反的系统误差**恰好抵消**：

**$2ND$ 高估的方向**（多算了）：Embedding 层。它有 $V \times d_{model}$ 个参数，$2BS \times \text{params}$ 给它们算了 $2 \times 2048 \times 152064 \times 4096 \approx 2.55 \times 10^{12}$ FLOPs。但 Embedding 是查表操作——**FLOPs ≈ 0**。多算了 $2.55 \times 10^{12}$ FLOPs。

**$2ND$ 低估的方向**（漏算了）：Attention 中 36 层的无参数 matmul。$36 \times 6.87 \times 10^{10} = 2.47 \times 10^{12}$ FLOPs。漏算了 $2.47 \times 10^{12}$ FLOPs。

**两者的抵消**：
$$\underbrace{2.55 \times 10^{12}}_{\text{Embedding 高估}} \;\; vs \;\; \underbrace{2.47 \times 10^{12}}_{\text{无参数 matmul 漏算}}$$

差距仅 $0.08 \times 10^{12}$，占前向总 FLOPs（$3.35 \times 10^{13}$）的 ~0.2%。几乎完美抵消。

**$2ND$ 在 Qwen3-8B S=2048 下误差 0.6%，不是因为公式本身精确，而是两个方向相反的系统误差恰好量级相当、基本互消。** 这是特定配置（S、V、模型架构）下的巧合，不是普遍性质。

### 这种抵消什么时候失效

1. **长序列（S ≫ 4096）**：无参数 matmul FLOPs 随 $S^2$ 增长，而 Embedding 高估是线性的。S 增大时漏算迅速超过高估 → $2ND$ **低估**。S=32768 时，漏算的 $QK^T+AV$ ≈ $6.3 \times 10^{13}$ FLOPs，远超 Embedding 高估 $4.1 \times 10^{13}$ → $2ND$ 低估约 15%。

2. **小词表模型（V 小）**：Embedding 参数少 → 高估方向变小 → 抵消不充分。LLaMA 2 7B（V=32000）在 S=2048 下，Embedding 高估仅 $1.05 \times 10^{12}$，而漏算仍是 $2.47 \times 10^{12}$ → $2ND$ 低估约 4%。

3. **推理场景（S=1, decode）**：无参数 matmul 几乎消失（$S^2=1$），而 Embedding 高估依然在 → $2ND$ **高估**。好在推理场景不用 $6ND$。

### 「可忽略」操作完整修正表

| 操作 | 对 FLOPs 的贡献 | $2BS \times \text{params}$ 的处理 | 净误差方向 | 误差量级 |
|------|----------------|--------------------------------|-----------|---------|
| 所有 `nn.Linear` | matmul，贡献主体 | 精确匹配 | — | 0 |
| **$QK^T$ + $AV$** | matmul $O(S^2)$，显著 | 漏算（params=0, FLOPs≠0） | **低估** | S=2048 时 ~7% |
| **Embedding** | 查表，≈0 | **高估**（当 matmul 算了） | **高估** | S=2048 时 ~7% |
| Softmax | exp+sum+div | 漏算 | 低估 | < 2% |
| SiLU + $\odot$ | element-wise | 漏算 | 低估 | < 0.1% |
| RMSNorm | element-wise | 关系不成立 | 不定 | < 0.01% |
| Residual | element-wise | 漏算 | 低估 | < 0.001% |

**核心结论：$2BS \times \text{params}$ 不能逐层使用**——它在 Attention 层漏掉了高达 40% 的无参数 matmul。$2ND$ 的全局精度来自两个大误差的巧合抵消，而非公式本身的普适性。当你需要逐层的精确 FLOPs（如做 kernel fusion 分析、计算某层的计算/通信比）时，必须用 §2.1-2.2 的完整逐项公式，不能用 $2BS \times \text{params}$ 的捷径。

## 1.3 反向传播的 FLOPs

对于 $Y = X \cdot W$，反向传播需要计算两个梯度：

**Weight gradient**（$\frac{\partial L}{\partial W}$）：$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$

- $X^T \in \mathbb{R}^{K \times M}$，$\frac{\partial L}{\partial Y} \in \mathbb{R}^{M \times N}$
- FLOPs = $2 \times K \times M \times N = 2 \times M \times K \times N$ —— **和前向相同**

**Input gradient**（$\frac{\partial L}{\partial X}$）：$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$

- $\frac{\partial L}{\partial Y} \in \mathbb{R}^{M \times N}$，$W^T \in \mathbb{R}^{N \times K}$
- FLOPs = $2 \times M \times N \times K = 2 \times M \times K \times N$ —— **和前向相同**

**结论**：对矩阵乘法而言，反向传播的 FLOPs **精确等于前向的 2 倍**。这不是近似。非 matmul 操作的反向 FLOPs 同理可忽略（第 3 章论证）。

### 6ND 公式的完整推导

- 前向所有 matmul 的 FLOPs ≈ $2ND$
- 反向 weight grad（需要把每层的 $\frac{\partial L}{\partial W}$ 都算一遍）：≈ $2ND$
- 反向 input grad（需要把每层的 $\frac{\partial L}{\partial X}$ 都算一遍）：≈ $2ND$
- **总和 = $6ND$**

前向的系数是 2（每个参数每个 token 参与 2 FLOPs），反向两个梯度各需要 2，合计 4。所以 $2+4=6$。

这就是 $6ND$ 公式的完整来源。详细论证见[第 4 章](#第-4-章反向传播-flops-与-6nd-公式)。

---

# 第 2 章：Transformer 前向 FLOPs 逐层分解

> **知识定位**：第 1 章给了 matmul 的公理，这章展开为 Transformer 每个组件的通用 FLOPs 公式。每个组件先做符号推导（不代入具体数字），得到一般性结论，再用 Qwen3-8B 验证。

本章涉及的符号约定：

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $S$ | 序列长度 (sequence length) |
| $d$ | $d_{model}$，hidden dimension |
| $d_{ff}$ | FFN intermediate dimension |
| $n_h, n_{kv}$ | Q heads 数, KV heads 数（GQA 下 $n_{kv} \le n_h$） |
| $d_h$ | head dimension，$d_h = d / n_h$ |
| $L$ | 层数 (number of layers) |
| $V$ | 词表大小 (vocab size) |

## 2.1 Attention 模块 — 通用推导

一个 Attention 模块包含六次矩阵乘法（Q/K/V/O 各有权重参数，Scores 和 Weighted Sum 是无参数的激活值间乘法），以及一次 Softmax。

### 2.1.1 QKV 投影

Q 投影：输入 $[B, S, d]$，权重 $W_Q \in \mathbb{R}^{d \times (n_h \cdot d_h)}$。
$$\text{FLOPs}_Q = 2 \cdot B \cdot S \cdot d \cdot (n_h \cdot d_h) = 2 B S d (n_h d_h)$$

由于 $d = n_h \cdot d_h$（by definition），所以 $\text{FLOPs}_Q = 2 B S d^2$。

K 投影：输入 $[B, S, d]$，权重 $W_K \in \mathbb{R}^{d \times (n_{kv} \cdot d_h)}$。
$$\text{FLOPs}_K = 2 B S d (n_{kv} d_h)$$

V 投影同理。

**QKV 投影合计**：
$$\text{FLOPs}_{QKV} = 2 B S d \, [n_h d_h + n_{kv} d_h + n_{kv} d_h] = 2 B S d \, d_h \, (n_h + 2 n_{kv})$$

**一般性结论**：
- **Dense MHA**（$n_{kv} = n_h$）：$\text{FLOPs}_{QKV} = 6 B S d^2$（因为 $n_h d_h = d$，$d_h (n_h+2n_h) = 3d_h n_h = 3d$，所以 $2BSd \cdot 3d = 6BSd^2$）
- **GQA (Grouped Query Attention)**：每减少 $n_{kv}$，QKV FLOPs 线性下降。GQA 4:1（$n_{kv} = n_h/4$）下，$\text{FLOPs}_{QKV} = 2BSd \cdot d_h (n_h + 2 n_h/4) = 2BSd^2 (1 + 0.5) = 3BSd^2$，恰为 dense MHA 的 **50%**

> **例：Qwen3-8B**。$d=4096$, $n_h=32$, $n_{kv}=8$, $d_h=128$, B=1, S=2048:
> - Q: $2 \cdot 1 \cdot 2048 \cdot 4096 \cdot 4096 = 6.87 \times 10^{10}$
> - K: $2 \cdot 1 \cdot 2048 \cdot 4096 \cdot 1024 = 1.72 \times 10^{10}$
> - V: 同 K = $1.72 \times 10^{10}$
> - **合计 $1.03 \times 10^{11}$**。若为 dense MHA 则为 $2.06 \times 10^{11}$ —— GQA 4:1 精确省 50%。

### 2.1.2 Attention Scores（$QK^T$）— 无参数 matmul

Q reshape 为 $[B, n_h, S, d_h]$，K 为 $[B, n_{kv}, S, d_h]$。GQA 下 K 通过 `repeat_interleave` 扩展到 $n_h$ heads（零 FLOP 开销——只是改变了 stride/view）。有效 K 维度为 $n_h \times S \times d_h$。
$$\text{FLOPs}_{scores} = 2 \cdot B \cdot n_h \cdot S \cdot S \cdot d_h = 2 B n_h d_h S^2$$

**一般性结论**：
- 这是 Attention 中唯一随 $S^2$ 增长的项——是长序列瓶颈的根源
- **无参数**：$QK^T$ 不涉及任何 `nn.Linear` 权重，因此 $2BS \times \text{params}$ 公式在此失效（params=0, FLOPs≠0）
- $n_h d_h = d$，所以 $\text{FLOPs}_{scores} = 2 B d S^2$——**不依赖 $n_{kv}$（GQA 对 Scores 的 FLOPs 无影响）**

> **例：Qwen3-8B**。$n_h=32$, $d_h=128$, B=1, S=2048:
> $2 \cdot 1 \cdot 32 \cdot 2048^2 \cdot 128 = 3.44 \times 10^{10}$

### 2.1.3 Weighted Sum（$\text{softmax}(A) \cdot V$）— 无参数 matmul

V 同样扩展到 $n_h$ heads。
$$\text{FLOPs}_{wsum} = 2 \cdot B \cdot n_h \cdot S \cdot S \cdot d_h = 2 B n_h d_h S^2$$

**一般性结论**：FLOPs 与 Scores 完全相同。同样是无参数 matmul，同样不依赖 $n_{kv}$。

> **例：Qwen3-8B**。$3.44 \times 10^{10}$，与 Scores 相同。

### 2.1.4 Output 投影

`nn.Linear(d, d)`，输入为拼接后的 multi-head 输出 $[B, S, d]$。
$$\text{FLOPs}_{out} = 2 B S d^2$$

**一般性结论**：与 Q 投影的 FLOPs 相同（两个 matmul 的维度完全对称）。不依赖 $n_{kv}$。

> **例：Qwen3-8B**。$6.87 \times 10^{10}$

### 2.1.5 Attention 模块总 FLOPs — 通用公式
$$\boxed{\text{FLOPs}_{attn} = \underbrace{2 B S d \cdot d_h (n_h + 2 n_{kv})}_{\text{QKV 投影}} \;+\; \underbrace{4 B n_h d_h S^2}_{\text{Scores + WSum}} \;+\; \underbrace{2 B S d^2}_{\text{Output 投影}}}$$**GQA 4:1 ($n_{kv} = n_h/4$) 简化**：记 $d = n_h d_h$，则 QKV = $3 B S d^2$，Scores+WSum = $4 B d S^2$，Output = $2 B S d^2$。合计：$$\text{FLOPs}_{attn}^{GQA\;4:1} = 5 B S d^2 + 4 B d S^2$$

**Dense MHA ($n_{kv} = n_h$) 简化**：QKV = $6 B S d^2$，合计 = $8 B S d^2 + 4 B d S^2$。

**关键洞察**：参数化 matmul（QKV+Output）贡献 $5BS d^2$（GQA）或 $8BS d^2$（MHA），而无参数 matmul（Scores+WSum）贡献 $4B d S^2$。两者量级之比 = $(5d)/(4S)$ 或 $(8d)/(4S)$。**当 $S \approx d$ (=4096) 时两者量级相当；$S \ll d$ 时投影主导；$S \gg d$ 时 Scores/WSum 爆炸。**

> **例：Qwen3-8B (S=2048)**。投影项 = $5 \cdot 1 \cdot 2048 \cdot 4096^2 = 1.72 \times 10^{11}$；Scores+WSum = $4 \cdot 1 \cdot 4096 \cdot 2048^2 = 6.87 \times 10^{10}$。合计 = $2.41 \times 10^{11}$。投影占 71%，Scores+WSum 占 29%。

## 2.2 FFN 模块 — 通用推导

SwiGLU FFN：$Y = \text{SiLU}(XW_{gate}) \odot (XW_{up}) \cdot W_{down}$。

三个线性层，输入 $[B, S, d]$：
- $W_{gate} \in \mathbb{R}^{d \times d_{ff}}$，FLOPs = $2 B S d d_{ff}$
- $W_{up} \in \mathbb{R}^{d \times d_{ff}}$，FLOPs 同上
- $W_{down} \in \mathbb{R}^{d_{ff} \times d}$，FLOPs 同上
$$\boxed{\text{FLOPs}_{ffn} = 6 B S d d_{ff}}$$

**一般性结论**：
- FFN 全部是**参数化 matmul**——$2BS \times \text{params}$ 精确匹配
- FFN 的 FLOPs $\propto S$（线性），不随 $S^2$ 增长；但常系数 $6 d_{ff}$ 通常很大（$d_{ff} \approx 3d$）
- FFN 与 Attention 投影的 FLOPs 对比：$\frac{\text{FFN}}{\text{Attn\_projections}} = \frac{6 d d_{ff}}{5 d^2} = \frac{6}{5} \cdot \frac{d_{ff}}{d} \approx 3.6$（取 $d_{ff}/d \approx 3$）。**FFN 的参数量和计算量都远大于 Attention 的参数化部分。**

> **例：Qwen3-8B (d=4096, d_ff=12288, B=1, S=2048)**：
> 每个投影 = $2 \cdot 1 \cdot 2048 \cdot 4096 \cdot 12288 = 2.06 \times 10^{11}$。三投影合计 = $6.19 \times 10^{11}$。
> FFN / Attn = $6.19 / 2.41 = 2.57$。

## 2.3 RMSNorm — 通用推导
$$\text{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i$$

对每个 token 的 $d$ 维向量：
- 平方：$d$ 次乘法
- 求和：$d-1 \approx d$ 次加法
- 除以 $d$：1 次除法
- 加 $\epsilon$：1 次加法
- 平方根：1 次（GPU `__fsqrt_rn`，≈1 FLOP）
- 逐元素除法（归一化）：$d$ 次
- 逐元素乘法（$\gamma$）：$d$ 次
$$\boxed{\text{FLOPs}_{rmsnorm\_per\_token} \approx 4d + O(1)}$$

Pre-LN 架构下每层有 2 个 RMSNorm（Attention 前、FFN 前），加 1 个 final RMSNorm。总 RMSNorm FLOPs = $(2L + 1) \cdot B \cdot S \cdot 4d$。

**一般性结论**：$\text{FLOPs}_{rmsnorm} / \text{FLOPs}_{ffn\_per\_layer} \approx \frac{2 \cdot 4d}{6 d d_{ff}} = \frac{4}{3 d_{ff}}$。因为 $d_{ff} \approx 12000$，RMSNorm 占比约 $10^{-4}$ 量级——可以忽略。

> **例：Qwen3-8B (L=36, d=4096)**。$(2 \cdot 36 + 1) \cdot 1 \cdot 2048 \cdot 4 \cdot 4096 \approx 2.45 \times 10^9$。对比一层 FFN 的 $6.19 \times 10^{11}$ → 占比 0.004%。

## 2.4 Embedding & LM Head — 通用推导

**Embedding**：查表操作。从 $V \times d$ 的权重表中取出 $B \times S$ 行。FLOPs = **0**（这是 GPU 的加载指令，不经过 Tensor Core）。参数量 = $V d$。

**LM Head**：`nn.Linear(d, V)`。$$\boxed{\text{FLOPs}_{lm\_head} = 2 B S d V}$$

**一般性结论**：
- LM Head 是正宗的 matmul，FLOPs 按期计算
- 但它只出现 1 次（而 Attention+FFN 各出现 $L$ 次），在 $L$ 足够大时贡献趋近于 0
- Embedding 是 $2ND$ 公式的一个系统高估来源（给它算了 FLOPs 但它并不消耗浮点运算）

> **例：Qwen3-8B (d=4096, V=152064, L=36)**。LM Head = $2 \cdot 1 \cdot 2048 \cdot 4096 \cdot 152064 = 2.55 \times 10^{12}$。占前向总 FLOPs 约 7.6%。

## 2.5 Residual Connections — 通用推导

每层 2 次：$x = x + f(x)$，逐元素加法。
$$\boxed{\text{FLOPs}_{residual} = 2 L B S d}$$

占比 < 0.001%。忽略。

> **例：Qwen3-8B**。$2 \cdot 36 \cdot 1 \cdot 2048 \cdot 4096 = 6.04 \times 10^{8}$。

## 2.6 单层合计与全模型汇总

**单层 Transformer Block**（不含 Embedding/LM Head）：
$$\boxed{\text{FLOPs}_{block} = \underbrace{2BSd \cdot d_h(n_h+2n_{kv}) + 4Bn_hd_hS^2 + 2BSd^2}_{\text{Attention}} + \underbrace{6BSdd_{ff}}_{\text{FFN}}}$$**全模型前向 FLOPs**：$$C_{fwd} = L \cdot \text{FLOPs}_{block} + \text{FLOPs}_{lm\_head} + \text{FLOPs}_{rmsnorm} + \text{FLOPs}_{residual}$$

对 Qwen3-8B (B=1, S=2048) 的汇总：

| 组件 | 通用公式 | Qwen3-8B 数值 | 占 Block 的比例 |
|------|---------|-------------|---------------|
| Attention × 36 | $L(5BSd^2 + 4BdS^2)$ | $8.66 \times 10^{12}$ | — |
| FFN × 36 | $6LBSdd_{ff}$ | $2.23 \times 10^{13}$ | — |
| **Block 小计** | | **$3.10 \times 10^{13}$** | **Attention 28% / FFN 72%** |
| LM Head | $2BSdV$ | $2.55 \times 10^{12}$ | (一次性，非逐层) |
| RMSNorm | $(2L+1)BS \cdot 4d$ | $2.45 \times 10^9$ | |
| Residual | $2LBSd$ | $6.04 \times 10^8$ | |
| Embedding | 0 | 0 | |
| **总计** | | **$3.36 \times 10^{13}$** | |

### 参数量计算

在验证 $2ND$ 之前，先推 Qwen3-8B 的精确参数量 $N$。每层 Attention 的 `nn.Linear` 权重：

- Q 投影：$d \times (n_h d_h) = d \times d = 4096 \times 4096$
- K 投影：$d \times (n_{kv} d_h) = 4096 \times 1024$
- V 投影：同上，$4096 \times 1024$
- Output 投影：$d \times d = 4096 \times 4096$

每层 FFN 的 `nn.Linear` 权重：
- gate_proj：$d \times d_{ff} = 4096 \times 12288$
- up_proj：同上
- down_proj：$d_{ff} \times d = 12288 \times 4096$

每层 Attention 参数量：
$$P_{attn} = d^2 + 2 \cdot d \cdot (n_{kv} d_h) + d^2 = 2d^2 + 2d \cdot n_{kv} d_h$$代入 $d=4096, n_{kv}=8, d_h=128$：$$P_{attn} = 2 \cdot 4096^2 + 2 \cdot 4096 \cdot 1024 = 33{,}554{,}432 + 8{,}388{,}608 = 41{,}943{,}040$$每层 FFN 参数量：$$P_{ffn} = 3 \cdot d \cdot d_{ff} = 3 \cdot 4096 \cdot 12288 = 150{,}994{,}944$$单层 Block 参数量（不含 RMSNorm $\gamma$ 和 bias——Qwen3-8B 无 bias）：$$P_{block} = P_{attn} + P_{ffn} = 41{,}943{,}040 + 150{,}994{,}944 = 192{,}937{,}984 \approx 1.93 \times 10^8$$全模型参数量：$$N = L \cdot P_{block} + \underbrace{V \cdot d}_{\text{Embedding}} + \underbrace{d \cdot V}_{\text{LM Head}} + \underbrace{(2L+1) \cdot d}_{\text{RMSNorm }\gamma}$$

网络部分：$36 \cdot 1.93 \times 10^8 = 6.95 \times 10^9$

Embedding + LM Head（weight tying 下共享）：$152064 \times 4096 = 6.23 \times 10^8$

RMSNorm $\gamma$：$(2 \cdot 36 + 1) \cdot 4096 = 73 \cdot 4096 \approx 2.99 \times 10^5$（可忽略）

**总参数量**：$N = 6.95 \times 10^9 + 6.23 \times 10^8 + 2.99 \times 10^5 \approx 7.57 \times 10^9$（weight tying）或 $8.20 \times 10^9$（无 tying）。

Qwen3-8B 官方报告的参数量为 ~8.2B（Embedding 和 LM Head 不共享权重）。本手册统一用 **$N = 8.2 \times 10^9$**。

> **为什么 $N_{matmul} \neq N$？** Embedding 的 $6.23 \times 10^8$ 个参数参与了参数计数，但它们不是 matmul 参数（查表操作）。所以 $N_{matmul} = 8.2 \times 10^9 - 6.23 \times 10^8 \approx 7.58 \times 10^9$。又因为 $N_{matmul}$ 和 $N$ 相差不到 8%，在 $2ND$ 近似中混用两者误差可控。

### $2ND$ 验证：$N=8.2 \times 10^9$, $D=2048$, $2ND = 2 \times 8.2 \times 10^9 \times 2048 = 3.36 \times 10^{13}$。逐项合计 $3.35 \times 10^{13}$。**误差 = (3.36 - 3.35) / 3.35 ≈ 0.3%**。

## 2.7 S 变化时的瓶颈转移

用通用公式分析 S 的影响。GQA 4:1 下，直接比较 Attention 和 FFN 的 FLOPs 来看谁主导：
$$\frac{\text{FLOPs}_{attn}}{\text{FLOPs}_{ffn}} = \frac{5BSd^2 + 4BdS^2}{6BSdd_{ff}} = \frac{5d + 4S}{6d_{ff}}$$

这个比值随 S 线性增长。翻转点出现在比值 = 1 时：$5d + 4S = 6d_{ff}$，即 $S = \frac{6d_{ff} - 5d}{4}$。代入 $d=4096, d_{ff}=12288$ → $S \approx 13312$。也就是说，序列长度超过 ~13K 后，Attention 的计算量开始超过 FFN。

用通用公式验证各 S 下的 Attn/FFN 比值。公式和逐项精确值高度一致（以 S=2048, 4096 为例验算）：
$$\text{FLOPs}_{attn}^{(S=4096)} = 5 \cdot 1 \cdot 4096 \cdot 4096^2 + 4 \cdot 1 \cdot 4096 \cdot 4096^2 = 5BSd^2 + 4BdS^2\text{ with }S=d \to 9Bd^3 = 9 \cdot 1 \cdot 4096^3 = 6.18 \times 10^{11}$$$$\text{FLOPs}_{ffn}^{(S=4096)} = 6 \cdot 1 \cdot 4096 \cdot 4096 \cdot 12288 = 1.24 \times 10^{12}$$$$\text{Attn/FFN} = 6.18 \times 10^{11} / 1.24 \times 10^{12} = 0.50 \checkmark$$

验证几个关键点（代入通用公式算单层）：

- **S=8192**：Attn = $5 \cdot 1 \cdot 8192 \cdot 4096^2 + 4 \cdot 1 \cdot 4096 \cdot 8192^2 = 6.87 \times 10^{11} + 1.10 \times 10^{12} = 1.79 \times 10^{12}$；FFN = $6 \cdot 1 \cdot 8192 \cdot 4096 \cdot 12288 = 2.47 \times 10^{12}$。**Attn/FFN = 1.79 / 2.47 = 0.72**
- **S=13312（翻转点）**：Attn = $5 \cdot 13312 \cdot 4096^2 + 4 \cdot 4096 \cdot 13312^2 = 1.117 \times 10^{12} + 2.903 \times 10^{12} = 4.020 \times 10^{12}$；FFN = $6 \cdot 13312 \cdot 4096 \cdot 12288 = 4.020 \times 10^{12}$。**Attn/FFN = 1.00** ✓

| S | Attn/FFN | 解读 |
|---|---------|------|
| 512 | 0.31 | FFN 是 Attention 的 3.3× |
| 2048 | 0.39 | FFN 是 Attention 的 2.6× |
| 4096 | 0.50 | 恰好平分——$S = d$ 时 Scores+WSum 与 QKV+Output 投影各贡献一半的 Attention FLOPs |
| 8192 | 0.72 | FFN 仍 > Attention，但差距在缩小 |
| 13312 | **1.00** | **翻转点**——Attention 和 FFN 的 FLOPs 相等 |
| 16384 | 1.17 | Attention 反超 |
| 32768 | 2.06 | Attention 是 FFN 的 2×，长序列瓶颈已明显转移 |

---

# 第 3 章：为什么这些操作可以忽略

> **知识定位**：第 2 章中多个操作的 FLOPs 被宣称「可以忽略」。这一章给每个判断提供**一般性推导**：先写出通用 FLOPs 公式，再与 matmul 的 FLOPs 对比，得出占比的量级估计，最后用 Qwen3-8B 数值验证。

## 3.1 衡量标准

**FLOPs 占比 < 1%** → 在 $6ND$ 级别近似中可以忽略。

这不意味着这些操作不存在——在极致优化（kernel fusion、Triton 手写算子）时，它们可能是**访存带宽**瓶颈（而非计算瓶颈）。但就纯 FLOPs 预算而言，占比 < 1% = 可以忽略。

判定方法：对于每个操作，推导其 FLOPs 通用公式 $F_{op}$，找到与之最近的 matmul 操作的 FLOPs $F_{matmul}$，比较 $F_{op} / F_{matmul}$ 的数量级。

## 3.2 Softmax

### 3.2.1 通用推导
$$\text{softmax}(s)_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}, \quad m = \max(s)$$

对 $[B, n_h, S, S]$ 的 scores：
1. 减 max：$B n_h S^2$ FLOPs
2. exp：$B n_h S^2$ 次（GPU `__expf` ≈ 1 FLOP）
3. 求和：$B n_h S^2$ FLOPs
4. 除法：$B n_h S^2$ FLOPs
$$\boxed{\text{FLOPs}_{softmax} \approx 4 B n_h S^2}$$最近的 matmul 是 Scores（$QK^T$）：$2 B n_h S^2 d_h$。$$\frac{\text{Softmax}}{\text{Scores}} = \frac{4 B n_h S^2}{2 B n_h S^2 d_h} = \frac{2}{d_h}$$

**一般性结论：Softmax / Scores = $2/d_h$。** 在标准 $d_h=128$ 下为 1.6%。这个比值**不随 S 增长**（分子分母都 ∝ S²），仅取决于 head_dim。当 $d_h$ 很大时（如 DeepSeek V4 的 MLA 用 $d_h=512$），softmax 占比降至 0.4%。

> **例：Qwen3-8B (S=2048, d_h=128)**。Softmax ≈ $4 \cdot 1 \cdot 32 \cdot 2048^2 = 5.37 \times 10^8$。Scores = $3.44 \times 10^{10}$。占比 = 1.6%。

**结论**：$d_h \geq 64$ 的模型中 attention softmax 占比始终 < 3%，可以忽略。只在极短 head dim（$d_h \leq 32$）或追求极致精度时需计入。

### 3.2.2 Transformer 中其他位置的 Softmax

Attention 之外，还有两处 Softmax：

**LM Head Softmax**（训练时）：

训练时 Compute Loss 步骤需要对 LM Head 输出做 softmax + cross-entropy。但在 PyTorch 中，`F.cross_entropy` 内部使用 logits + log-softmax（等价于 `log_softmax + nll_loss`），而 log-softmax 本身就是 softmax + log 的融合版，FLOPs 相同——约 $4 B S V + O(B S V)$。代入 $V=152064$：
$$\text{FLOPs}_{lm\_softmax} \approx 4 \cdot 1 \cdot 2048 \cdot 152064 = 1.25 \times 10^9$$对比 LM Head matmul 自身的 $2.55 \times 10^{12}$：$$\frac{\text{LM Softmax}}{\text{LM Head matmul}} = \frac{4 B S V}{2 B S d V} = \frac{2}{d} = \frac{2}{4096} \approx 0.05\%$$

**同样因为 $d$ 因子——LM Head 的 matmul 包含了内积长度 $d=4096$ 的维度，而 softmax 只有 $O(V)$ 的逐元素运算。占比 $2/d$，和 Attention softmax 的 $2/d_h$ 是完全同构的比值：元素级 vs 内积级。**

> **注**：推理时（generate 阶段）decode 每步只预测一个 token（$S=1$），LM Head matmul 骤降至 $2 B d V$（$B \approx 1$），此时 softmax 占比升至 $2/d = 0.05\%$ 仍可忽略。

**MoE Router Softmax**（仅 MoE 模型有）：gate network 输出 logits 后做 softmax 得到各 expert 的概率分布。输入形状 $[B, S, N_{experts}]$。$N_{experts}$ 通常在 8-256 之间，远小于 $S$ 和 $d$。FLOPs $\approx 4 B S N_{experts}$，较 Attention softmax 的 $4 B n_h S^2$ 小 $n_h S / N_{experts}$ 倍——在 $S \geq 512$ 时即可忽略。

> **例**：Mixtral 8×7B（$N_{experts}=8$），S=2048 时 MoE softmax ≈ $4 \cdot 1 \cdot 2048 \cdot 8 = 6.55 \times 10^4$ FLOPs。占比 < 0.0001%。数量级完全不值得计入。

## 3.3 SiLU + 逐元素乘法

### 3.3.1 问题：element-wise 操作在什么条件下可以忽略？

SwiGLU FFN 有两个计算步骤——三个矩阵乘法（gate/up/down），加上两个 element-wise 操作（SiLU 激活 + 逐元素乘法 $\odot$）。矩阵乘法的 FLOPs 我们在 §2.2 已经算过了（$6BSdd_{ff}$）。Element-wise 操作的计算量是多少？重要吗？

### 3.3.2 Element-wise 与 Matmul 的本质区别

**Element-wise 操作**：每个输出元素只依赖**一个**输入元素。计算规则是标量函数 $y_i = f(x_i)$。

例如 $\text{SiLU}(x) = x \cdot \frac{1}{1+e^{-x}}$，计算一个输出元素需要：取负 + exp + 加 1 + 取倒数 + 乘法 ≈ **5 次浮点运算**。逐元素乘法 $y_i = a_i \cdot b_i$ 每个元素只需 **1 次浮点运算**。

对于一个 $[B, S, d_{ff}]$ 的张量，总元素数量 = $B \times S \times d_{ff}$。SiLU + $\odot$ 合计约 6 FLOPs/element，所以：
$$\text{FLOPs}_{swiglu\_elem} \approx 6 \cdot B \cdot S \cdot d_{ff}$$

**矩阵乘法（Matmul）**：每个输出元素依赖**一整行/列**的输入元素。$\text{gate\_proj}$：输入 $X \in \mathbb{R}^{BS \times d}$，权重 $W \in \mathbb{R}^{d \times d_{ff}}$，输出 $Y \in \mathbb{R}^{BS \times d_{ff}}$。

输出 $Y$ 有 $BS \times d_{ff}$ 个元素，每个元素 $Y_{i,j} = \sum_{k=1}^{d} X_{i,k} \cdot W_{k,j}$——做 $d$ 次乘法 + $d$ 次加法 = $2d$ FLOPs。所以 gate_proj 总 FLOPs = $2 \cdot BS \cdot d \cdot d_{ff}$。

### 3.3.3 量级对比：为什么 matmul 大了一千多倍

两式相比：
$$\frac{\text{FLOPs}_{element}}{\text{FLOPs}_{matmul}} = \frac{6 \cdot B \cdot S \cdot d_{ff}}{2 \cdot B \cdot S \cdot d \cdot d_{ff}} = \frac{3}{d}$$

**分子和分母的 $B$、$S$、$d_{ff}$ 全部消掉了。** 这和直觉一致——两种操作都需要处理同样数量的元素，唯一的差异来自 matmul 中每个元素花 $2d$ FLOPs（内积），而 element-wise 只花常数次标量运算。

代入 $d = 4096$：$3/4096 \approx 0.073\%$。也就是说 **gate_proj 的矩阵乘法是 element-wise 操作计算量的约 $d/3 \approx 1365$ 倍。**

### 3.3.4 一般性结论

任何 $O(BSd_{ff})$ 或 $O(BSd)$ 的 element-wise 操作（SiLU、GELU、ReLU、逐元素加法/乘法、Dropout），与相邻的 matmul（$O(BSdd_{ff})$ 或 $O(BSd^2)$）相比，比值都落在 $O(1/d) \approx 0.02\%\text{-}0.07\%$。

**这个 $d$ 因子（内积长度）就是所有非 matmul 操作在 FLOPs 上被忽略的物理根源。** 它不依赖于 batch size、序列长度、模型层数、FFN 维度——只取决于 hidden dimension。只要 $d$ 在千量级（现有一切 LLM 都满足），element-wise 的 FLOPs 就永远在 matmul 的千分之一以下。

> **例：Qwen3-8B (d=4096, d_ff=12288, B=1, S=2048)**。$6 \cdot 1 \cdot 2048 \cdot 12288 = 1.51 \times 10^8$。gate_proj = $2.06 \times 10^{11}$。占比 = 0.073%。

## 3.4 RMSNorm

### 3.4.1 通用推导

从 §2.3：$\text{FLOPs}_{rmsnorm\_per\_token} \approx 4d$。全模型 = $(2L+1) \cdot B \cdot S \cdot 4d$。

最近的 matmul 是 Attention 中任意一个投影（$2 B S d^2$）。
$$\frac{\text{RMSNorm (全模型)}}{\text{一层 Q 投影}} = \frac{(2L+1) \cdot 4d}{2d^2} = \frac{2(2L+1)}{d}$$

**一般性结论**：全模型 RMSNorm / 一层投影 ≈ $4L/d$。在 $L=36$, $d=4096$ 下 ≈ 3.5%，但这只是一层投影——对比全模型 Attention+FFN，分母扩大约 $L \times 2.5 \approx 90$ 倍 → 占比 < 0.04%。

> **例：Qwen3-8B**。$2.45 \times 10^9 / 2.87 \times 10^{13} = 0.009\%$。**可以忽略。**

## 3.5 Residual Connections

### 3.5.1 通用推导
$$\boxed{\text{FLOPs}_{residual} = 2 L B S d}$$

对比全 Attention+FFN（$L \cdot (5BSd^2 + 4BdS^2 + 6BSdd_{ff})$），residual 约小 $O(d)$ 倍。

> **例：Qwen3-8B**。$6.04 \times 10^8 / 2.87 \times 10^{13} \approx 0.002\%$。**可以忽略。**

## 3.6 Embedding & LM Head

**Embedding**：FLOPs = 0（查表）。**不需要忽略，也无需计入 FLOPs。**

**LM Head**：$\text{FLOPs} = 2 B S d V$。这是正规 matmul，**不可忽略**——但它只出现 1 次，全模型占比 = $2dV / (L \times \text{FLOPs}_{block} + 2dV)$。这个比例随 $L$ 增大而减小。

> **例：Qwen3-8B**。$2.55 \times 10^{12} / 3.35 \times 10^{13} = 7.6\%$。对 8B 模型不可忽略；对 70B 模型（L=80）降至约 3%。

## 3.7 可忽略总表

| 操作 | 通用 FLOPs 公式 | 对比基准（通用） | 占比（通用） | Qwen3-8B 数值验证 |
|------|----------------|----------------|------------|-----------------|
| Softmax | $4 B n_h S^2$ | Scores: $2 B n_h S^2 d_h$ | $2/d_h$ ≈ 1.6% | 1.6% ✓ |
| SiLU + ⊙ | $6 B S d_{ff}$ | gate_proj: $2 B S d d_{ff}$ | $3/d$ ≈ 0.07% | 0.07% ✓ |
| RMSNorm | $(2L+1)BS \cdot 4d$ | Attn+FFN: $L \cdot 6BSdd_{ff}$ | $\sim 4L/(3L d_{ff})$ ≈ 0.004% | 0.009% ✓ |
| Residual | $2 L B S d$ | 同上 | $\sim 1/(3d_{ff})$ ≈ 0.003% | 0.002% ✓ |
| Embedding | 0 | — | 0 | — |
| LM Head | $2 B S d V$ | 全模型前向 | $\sim 2V/(L d_{ff})$ | 7.6% — **不可忽略** |

---

# 第 4 章：反向传播 FLOPs 与 6ND 公式

> **知识定位**：第 1 章给了 matmul 反向 = 前向 × 2 的结论。这章给出完整推导，并论证 6ND 公式的适用范围和精度。

## 4.1 为什么反向 ≈ 前向 × 2（对 matmul 精确）

第 1 章 §1.3 给出了完整推导。这里以一个具体算例验证。

对 Qwen3-8B 的 gate_proj（$4096 \times 12288$），前向 FLOPs = $2 \times B \times S \times 4096 \times 12288$。

反向：
- $\frac{\partial L}{\partial W}$：$X^T \cdot \frac{\partial L}{\partial Y}$，$X^T \in \mathbb{R}^{4096 \times BS}$，$\frac{\partial L}{\partial Y} \in \mathbb{R}^{BS \times 12288}$
  FLOPs = $2 \times 4096 \times BS \times 12288$ ← **和前向完全相同**
- $\frac{\partial L}{\partial X}$：$\frac{\partial L}{\partial Y} \cdot W^T$，$\frac{\partial L}{\partial Y} \in \mathbb{R}^{BS \times 12288}$，$W^T \in \mathbb{R}^{12288 \times 4096}$
  FLOPs = $2 \times BS \times 12288 \times 4096$ ← **和前向完全相同**

反向合计 = 2 × 前向 FLOPs。**精确成立**，不是近似。

## 4.2 非 matmul 操作的贡献

Softmax、SiLU、RMSNorm 的反向传播 FLOPs 如何？

**Softmax 反向**：$\frac{\partial L}{\partial s} = \text{diag}(p) - pp^T$，涉及和外积。对 $S$ 维向量，反向约为 $O(S^2)$。和我们已论证过的一样（§3.2），前向 softmax 本身就 < 2%，反向亦然。在 $S \geq 2048$ 时可忽略。

**SiLU 反向**：$\frac{\partial \text{SiLU}}{\partial x} = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))$。涉及 sigmoid + 三次乘法，约 5 FLOPs/element。同理可忽略（前向占 0.07%，反向同等量级）。

**RMSNorm 反向**：$O(B S d)$。前向忽略，反向也忽略。

**结论**：6ND 公式忽略这些非 matmul 操作引入的误差 < 0.2%。

## 4.3 6ND 公式的完整表述与适用范围
$$C_{total} \approx 6ND$$

其中:
- $N$ = 模型总参数量
- $D$ = 训练 tokens 总数
- 系数 6 的来源：前向 matmul FLOPs ≈ $2ND$（§1.2），反向 weight grad ≈ $2ND$（§4.1），反向 input grad ≈ $2ND$（§4.1）。$2 + 2 + 2 = 6$

这个公式是一个近似。第 1 章已经论证了 $2ND$ 近似忽略了两类方向相反的误差（Embedding 参数量被高估、Attention 无参数 matmul 被低估），在典型配置下两者的抵消使净误差 < 1%。$6ND$ 继承了相同的误差边界。



---


# 第 5 章：完整案例 — Qwen3-8B FLOPs 汇总

> **本章回答什么问题**：把前 4 章的通用公式套到一个真实模型上，算出一张完整的 FLOPs 账单。

## 5.1 模型参数速查

| 参数 | 值 |
|------|-----|
| $d_{model}$ | 4096 |
| $d_{ff}$ | 12288 |
| $n_{heads}$ (Q) | 32 |
| $n_{kv\_heads}$ | 8 (GQA 4:1) |
| $d_{head}$ | 128 |
| $n_{layers}$ | 36 |
| $V$ (vocab_size) | 152064 |
| 总参数量 | ~8.2B |
| 每层 Attention FLOPs | $2.41 \times 10^{11}$ (S=2048) |
| 每层 FFN FLOPs | $6.19 \times 10^{11}$ (S=2048) |
| 训练总 Model States | 131.2 GB (16 bytes/param) |
| 激活值 (FA) | ~10 GB (S=2048) |
| 激活值 (no FA) | ~19 GB (S=2048) |

## 5.2 前向 FLOPs 完整账单

| 组件 | 一层 FLOPs | 全模型 FLOPs | 占比 |
|------|-----------|-------------|------|
| Attention | $2.41 \times 10^{11}$ | $8.66 \times 10^{12}$ | 25.8% |
| FFN | $6.19 \times 10^{11}$ | $2.23 \times 10^{13}$ | 66.5% |
| LM Head | — | $2.55 \times 10^{12}$ | 7.6% |
| RMSNorm | — | $2.45 \times 10^9$ | 0.007% |
| Residuals | — | $6.04 \times 10^8$ | 0.002% |
| **总计 (S=2048)** | | **$3.35 \times 10^{13}$** | |
| $2ND$ 近似 | | $3.36 \times 10^{13}$ | 误差 0.3% |



# 第 6 章：训练时间估算 — 从 6ND 到 GPU 小时

> **本章回答什么问题**：$6ND$ 告诉你总 FLOPs。结合 GPU 的理论 TFLOPS、MFU 和工程损耗，能给出训练所需 GPU 小时数的可信估算。

## 6.1 核心公式

$$\text{GPU Hours} = \frac{C_{total}}{\text{单卡每小时 FLOPs} \times \text{MFU}} = \frac{6ND}{E_{hour} \times \text{MFU}}$$

其中：
- $N$：模型参数量
- $D$：训练 tokens 总数
- $E_{hour}$：单卡每小时理论 FLOPs（H100 SXM5 BF16 下 $E_{hour} = 3.56 \times 10^{18}$）

> **注意**：估算训练 FLOPs 时使用 dense BF16/FP16 Tensor Core 指标（H100 = 989 TFLOPS），不使用稀疏算力（Sparsity）指标（1979 TFLOPS）。大模型预训练通常无法启用结构化稀疏。

## 6.2 H100 单卡理论算力

以 **NVIDIA H100 SXM5 (80GB)** 为例：

- BF16 Tensor Core 理论峰值：**989 TFLOPS** = $9.89 \times 10^{14}$ FLOPs/s
- 单卡每小时最大理论 FLOPs：

$$E_{hour} = 989 \times 10^{12} \text{ FLOPs/s} \times 3600 \text{ s} \approx 3.56 \times 10^{18} \text{ FLOPs}$$

## 6.3 MFU — 为什么 100% 是不可能的

在多卡分布式训练中，以下因素使 GPU 无法达到理论峰值：

- **通信开销**：AllReduce 梯度同步、All-to-All MoE 路由、Pipeline bubble 等使 GPU 空闲等待
- **算子开销**：非 matmul 操作（Softmax、SiLU、RMSNorm）不用 Tensor Core，效率低
- **内存带宽限制**：小 batch 时数据不够 Tensor Core「吃饱」
- **Kernel launch 延迟**：每个 CUDA kernel 有启动开销

**MFU 通常为 35-55%。** 在计算 MFU 时，分子通常固定采用**不含重计算（Activation Checkpointing）**的理论计算量 $6N$。因此 MFU 测值中已经隐含了 AC 带来的额外计算开销。

根据业界工程水平，不同规模集群上的 MFU 参考值：

| 集群规模 | MFU 范围 | 说明 |
|---------|---------|------|
| 128-256 GPUs | 40-50% | 通信压力小、故障少 |
| 512-1024 GPUs | 35-42% | 标准企业级集群 |
| 2048+ GPUs | 30-38% | 万卡级，通信和故障开销显著 |
| DeepSeek-V3 (2048×H800) | **51%** | 目前标杆（DualPipe + FP8 + DeepEP） |

## 6.4 完整案例：1T tokens 训练 70B 模型

### 6.4.1 总理论计算量

$$C_{total} = 6 \times N \times D = 6 \times (70 \times 10^9) \times (1 \times 10^{12}) = 4.2 \times 10^{23} \text{ FLOPs}$$

### 6.4.2 不同 MFU 下的 GPU 小时数

| 场景 | MFU | GPU Hours | 公式 |
|------|-----|-----------|------|
| 高度优化（小集群） | 45% | **~26.2 万** | $4.2 \times 10^{23} / (3.56 \times 10^{18} \times 0.45)$ |
| 中等优化（常规集群） | 40% | **~29.5 万** | $4.2 \times 10^{23} / (3.56 \times 10^{18} \times 0.40)$ |
| 保守估算（万卡集群） | 35% | **~33.7 万** | $4.2 \times 10^{23} / (3.56 \times 10^{18} \times 0.35)$ |

用不同 GPU 数量折算为实际训练天数（取 MFU=40%，总 29.5 万 GPU Hours）：

| GPUs | 训练天数 | 说明 |
|------|---------|------|
| 256 | ~48 天 | 单集群满负荷 |
| 512 | ~24 天 | 标准企业集群 |
| 1024 | ~12 天 | 中型预训练集群 |

## 6.5 现实工程中的额外损耗

以上是「纯运行时间」。实际商业预算中还需要考虑：

1. **硬件故障与重启（5-15%）**：千卡集群的 MTTF（平均无故障时间）显著降低。节点挂掉、网络中断或慢节点（stragglers）导致回滚到最近 checkpoint 重跑，带来额外 5-15% 的 GPU 时间浪费。
2. **Checkpoint 写入阻塞**：70B 模型的完整 checkpoint（权重 + 优化器状态）达数百 GB 甚至 TB 级，定期写入全局存储会造成流水线短暂停顿。
3. **中间评估（Evaluation）**：训练过程中定期在验证集上跑评测，这些 token 不计入 $D$，但计入 GPU 账单。
4. **长上下文预训练**：如果后期需要外推到 8K+ 长上下文，Attention 算力占比二次方上升，整体 MFU 显著下降。

将这些损耗折算后，实际有效利用率约为标称 MFU 的 85-90%。例如 MFU 40% 的集群，综合考虑故障和评估后实际效率约 **34-36%**。

## 6.6 业界真实案例：Llama 3 70B

| 指标 | 数值 |
|------|------|
| 训练规模 | 15T Tokens |
| 硬件平台 | H100-80GB (SXM5) |
| 集群规模 | ~2.4 万张 H100（混合 3D 并行） |
| 官方披露总 GPU 小时 | **640 万 H100 GPU Hours** |
| 折算到 1T Tokens | $640 / 15 \approx \textbf{42.7 万}$ GPU Hours/1T |

为什么 Llama 3 的 42.7 万明显高于理论估算的 30 万？主要因为 Meta 采用了极大规模的分布式架构（2.4 万张 H100），跨节点通信开销（AllReduce、PP bubble）、高频硬件故障重排队、以及复杂的混合 3D 并行策略，使整体系统利用率显著下降。

## 6.7 总结与预算建议

根据团队工程水平和集群规模，定位以下区间：

| 梯队 | 集群规模 | 工程水平 | 等效 MFU | 1T tokens 训练 70B 预估 |
|------|---------|---------|---------|----------------------|
| 第一梯队 | 128-256 卡 | 高水平优化，极少故障 | ~42% | **29 万 ~ 32 万** GPU Hours |
| 第二梯队 | 512-1024 卡 | 标准企业级，正常维护 | ~35-38% | **33 万 ~ 37 万** GPU Hours |
| 第三梯队 | 2000+ 卡 | 万卡级超大规模 | ~30-33% | **40 万 ~ 43 万** GPU Hours |

> **关键教训**：集群规模越大，单位 GPU 效率越低。256 卡上 1 GPU 能顶 0.45 张理论卡，2.4 万卡只能顶 ~0.28 张。这并非「大规模集群做得不好」——通信和故障的物理代价天然随 $N_{GPU}$ 增长。DeepSeek-V3 的 51% MFU 在 2048 卡上实现，正是依赖 DualPipe 和 DeepEP 把这些天然代价压到了极致。

## 6.8 Qwen3-8B 速算

代入 Qwen3-8B：$N = 8.2 \times 10^9$, $D = 2 \times 10^{12}$：

$$C_{total} \approx 6 \times 8.2 \times 10^9 \times 2 \times 10^{12} = 9.84 \times 10^{22} \text{ FLOPs}$$

| 配置 | MFU | GPU Hours | 256 卡天数 |
|------|-----|-----------|-----------|
| 理想（100% MFU） | 100% | ~2.8 万 | ~4.5 天 |
| 实际（40% MFU） | 40% | ~6.9 万 | ~11 天 |

## 6.9 扩展到其他模型规模

用 $D=1T$, MFU=40%, H100 单卡：

| 模型 | N | $C_{total}$ | GPU Hours | 256 卡天数 |
|------|---|------------|-----------|-----------|
| Qwen3-8B | 8.2B | $4.92 \times 10^{22}$ | ~3.8 万 | ~6 天 |
| Qwen3-32B | 32.8B | $1.97 \times 10^{23}$ | ~15.4 万 | ~25 天 |
| LLaMA 3 70B | 70B | $4.2 \times 10^{23}$ | ~29.5 万 | ~48 天 |
| LLaMA 3 405B | 405B | $2.43 \times 10^{24}$ | ~170 万 | ~277 天 |

> **注意**：405B 行仅说明「如果按 256 卡单集群训需要近 10 个月」。实际上 405B 必须用更大的集群（千卡级）+ 更复杂的 3D 并行策略来缩短 wall-clock 时间。Llama 3 405B 用了 ~1.6 万 H100。

---



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



# 附录 B：常用 GPU 规格速查

| 规格               | A100 80GB | H100 80GB | H200 141GB | B200      |
| ---------------- | --------- | --------- | ---------- | --------- |
| 架构               | Ampere    | Hopper    | Hopper     | Blackwell |
| 上市年份             | 2020      | 2023      | 2024       | 2024      |
| HBM 容量           | 80 GB     | 80 GB     | 141 GB     | 192 GB    |
| HBM 带宽           | 2.0 TB/s  | 3.35 TB/s | 4.8 TB/s   | 8.0 TB/s  |
| FP32 TFLOPS      | 19.5      | 67        | 67         | 90        |
| TF32 TFLOPS      | 156       | 494       | 494        | 720       |
| BF16/FP16 TFLOPS | 312       | 989       | 989        | 2250      |
| FP8 TFLOPS       | —         | 1979      | 1979       | 4500      |
| NVLink 带宽        | 600 GB/s  | 900 GB/s  | 900 GB/s   | 1800 GB/s |



# 附录 C：关键公式速查卡

| 公式 | 说明 | 用途 |
|------|------|------|
| $2MKN$ | matmul FLOPs ($M \times K \times N$) | 所有线性层 FLOPs 的公理 |
| $C_{fwd} \approx 2ND$ | 前向总 FLOPs | 算力预算 |
| $C_{total} \approx 6ND$ | 训练总 FLOPs（前向 + 反向 weight grad + 反向 input grad） | 训练时间估计 |
| $\text{Attn/FFN} = (5d + 4S) / (6d_{ff})$ | GQA 4:1 下 Attention vs FFN 的 FLOPs 比 | 判断 S 变化时瓶颈转移 |
| $\text{Softmax}/\text{Scores} = 2/d_h$ | Softmax 相对于 Attention Scores matmul 的 FLOPs 占比 | 论证 softmax 可忽略 |
| $\text{SiLU}/\text{gate\\_proj} = 3/d$ | Element-wise 相对于 matmul 的 FLOPs 占比 | 论证 element-wise 可忽略 |
| $\text{GPU Hours} = 6ND / (E_{hour} \times \text{MFU})$ | 训练所需 GPU 小时数 | 训练预算 |


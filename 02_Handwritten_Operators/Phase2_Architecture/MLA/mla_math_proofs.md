# MLA 数学证明：低秩压缩与权重吸收

> 本文涵盖 MLA 涉及的严格数学推导，包括低秩压缩定义、SVD 最优近似定理、MLA 与 MHA 等价性证明、权重吸收推导，以及关键设计问题的数学分析。
>
> 相关文件：
> - [MLA 论文精读笔记](mla_paper_notes.md) -- DeepSeek-V2 论文核心内容总结
> - [MLA 分块矩阵乘法](block_matmul.md) -- 分块矩阵乘法基础
> - [MLA 代码走读](mla_code_walkthrough.md) -- 手撕代码详解

---

## 核心结论速览

> MLA 并不是在数学上与任意 MHA 完全等价。它与一种"受低秩约束的 MHA"是等价的；同时，MLA 的 latent cache 推理形式与 MLA 的显式恢复 key/value 形式是严格等价的。论文中说的"$W^{UK}$ 可吸收到 $W^Q$、$W^{UV}$ 可吸收到 $W^O$"，指的是 MLA 内部推理计算的等价变形，不是说 MLA 等价于任意标准 MHA。

---

# 第一部分：低秩压缩的数学基础

## 1. 什么是低秩压缩？线性代数角度的严格定义

### 1.1 矩阵的秩

设矩阵 $A \in \mathbb{R}^{m \times n}$，矩阵 $A$ 的秩定义为其列空间维度，等价地也是行空间维度：

$$
\operatorname{rank}(A) = \dim(\operatorname{Col}(A)) = \dim(\operatorname{Row}(A))
$$

也等价于 $A$ 中非零奇异值的个数。

若 $\operatorname{rank}(A)=r$，则说明 $A$ 所实现的线性变换 $x \mapsto Ax$ 虽然输入空间是 $\mathbb{R}^n$，输出空间是 $\mathbb{R}^m$，但它的输出实际上落在一个至多 $r$ 维的子空间中。

### 1.2 低秩矩阵

如果矩阵 $A \in \mathbb{R}^{m \times n}$ 满足 $\operatorname{rank}(A) = r \ll \min(m,n)$，则称 $A$ 是低秩矩阵。

### 1.3 低秩分解

一个秩不超过 $r$ 的矩阵 $A \in \mathbb{R}^{m \times n}$ 可以写成两个小矩阵的乘积：

$$
A = UV,\quad U \in \mathbb{R}^{m \times r},\; V \in \mathbb{R}^{r \times n}
$$

因此线性映射 $x \mapsto Ax$ 可以拆成两步：$x \mapsto Vx \mapsto U(Vx)$，其中 $Vx \in \mathbb{R}^r$ 就是一个低维 latent 表示。这就是所谓的**低秩压缩**。

### 1.4 低秩压缩的严格定义

给定一个高维线性映射 $A \in \mathbb{R}^{m \times n}$，如果我们用两个矩阵 $D \in \mathbb{R}^{r \times n}$ 和 $U \in \mathbb{R}^{m \times r}$ 来替代它：$A \approx UD$，并且 $r \ll \min(m,n)$，那么这个过程可以称为对 $A$ 的低秩压缩。

其中：
- $D$ 是 down-projection，把输入压缩到 $r$ 维；
- $U$ 是 up-projection，把 $r$ 维 latent 映射回 $m$ 维；
- 中间向量 $c = Dx \in \mathbb{R}^r$ 就是压缩表示。

在 MLA 中，对 key 的低秩形式是：

$$
k_t^C = W^{UK} W^{DKV} h_t
$$

其中 $W^{DKV} \in \mathbb{R}^{d_c \times d}$，$W^{UK} \in \mathbb{R}^{n_h d_h \times d_c}$。

于是完整 key projection 等价于：

$$
W^K_{\text{MLA}} = W^{UK} W^{DKV}
$$

因此 $\operatorname{rank}(W^K_{\text{MLA}}) \leq d_c$。这就是低秩约束。

value 同理：$W^V_{\text{MLA}} = W^{UV} W^{DKV}$，且 $\operatorname{rank}(W^V_{\text{MLA}}) \leq d_c$。

更特殊的是，MLA 中 key 和 value 共享同一个 down-projection：$c_t^{KV} = W^{DKV}h_t$，所以它不只是低秩分解，而是 **key/value joint low-rank compression**。

---

## 2. 低秩分解的数学定理

### 2.1 定理一：秩不超过 $r$ 当且仅当可以分解为 $UV$

**定理**：设 $A \in \mathbb{R}^{m \times n}$，则 $\operatorname{rank}(A) \leq r$ 当且仅当存在矩阵 $U \in \mathbb{R}^{m \times r}$ 和 $V \in \mathbb{R}^{r \times n}$ 使得 $A = UV$。

**充分性证明**：假设 $A = UV$，则有矩阵秩不等式 $\operatorname{rank}(UV) \leq \min(\operatorname{rank}(U), \operatorname{rank}(V))$。而 $\operatorname{rank}(U) \leq r$，$\operatorname{rank}(V) \leq r$，所以 $\operatorname{rank}(A) = \operatorname{rank}(UV) \leq r$。充分性得证。

**必要性证明**：假设 $\operatorname{rank}(A)=s \leq r$。由秩分解定理，存在 $B \in \mathbb{R}^{m \times s}$ 和 $C \in \mathbb{R}^{s \times n}$ 使得 $A = BC$。

现在构造：

$$
U = \begin{bmatrix} B & 0_{m \times (r-s)} \end{bmatrix} \in \mathbb{R}^{m \times r},\quad
V = \begin{bmatrix} C \\ 0_{(r-s) \times n} \end{bmatrix} \in \mathbb{R}^{r \times n}
$$

则 $UV = BC = A$。必要性得证。

因此：$\operatorname{rank}(A) \leq r \iff A = UV$。

### 2.2 定理二：SVD 保证存在最优低秩近似 (Eckart-Young-Mirsky)

如果矩阵 $A$ 不是低秩，也可以做近似。核心定理是 **Eckart-Young-Mirsky 定理**。

**奇异值分解**：任意矩阵 $A \in \mathbb{R}^{m \times n}$ 都存在奇异值分解 $A = P \Sigma Q^T$，其中 $P \in \mathbb{R}^{m \times m}$ 和 $Q \in \mathbb{R}^{n \times n}$ 是正交矩阵，$\Sigma \in \mathbb{R}^{m \times n}$ 是对角型矩阵，奇异值满足 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p \geq 0$，$p = \min(m,n)$。

**截断 SVD**：定义 rank-$r$ 截断近似 $A_r = P_r \Sigma_r Q_r^T$，则 $\operatorname{rank}(A_r) \leq r$。

**Eckart-Young-Mirsky 定理**：对于任意 $A \in \mathbb{R}^{m \times n}$，在所有秩不超过 $r$ 的矩阵中，$A_r$ 是 $A$ 在 Frobenius 范数和谱范数意义下的最优近似：

$$
A_r = \arg\min_{\operatorname{rank}(B)\leq r} \|A-B\|_F
$$

并且：

$$
\|A-A_r\|_F^2 = \sum_{i=r+1}^{p} \sigma_i^2
$$

在谱范数下：

$$
A_r = \arg\min_{\operatorname{rank}(B)\leq r} \|A-B\|_2,\quad \|A-A_r\|_2 = \sigma_{r+1}
$$

**证明思路**：由于 Frobenius 范数在正交变换下不变，$\|A-B\|_F = \|P^T(A-B)Q\|_F$。令 $C = P^T B Q$，则 $\operatorname{rank}(C)=\operatorname{rank}(B)\leq r$。问题等价于 $\min_{\operatorname{rank}(C)\leq r} \|\Sigma-C\|_F$。因为 $\Sigma$ 是对角矩阵，若 $C$ 秩不超过 $r$，它最多只能保留 $r$ 个独立方向。为使误差最小，最优策略是保留最大的 $r$ 个奇异值，丢弃其余。因此 $C^\star = \operatorname{diag}(\sigma_1,\dots,\sigma_r,0,\dots,0)$，$B^\star = P C^\star Q^T = A_r$。

这说明低秩近似在数学上是严格成立的。

### 2.3 低秩压缩为什么在神经网络中合理？

从纯数学上看，低秩分解总是可以做，但效果好不好取决于矩阵的奇异值谱。如果矩阵 $A$ 的奇异值衰减很快，$\sigma_{r+1},\sigma_{r+2},\dots$ 很小，那么近似误差 $\|A-A_r\|_F^2 = \sum_{i=r+1}^{p}\sigma_i^2$ 就会很小。

但注意：**MLA 不是先训练一个 MHA 再对其 $W^K,W^V$ 做 SVD 压缩。MLA 是从一开始就把 key/value projection 参数化为低秩形式，让模型在训练过程中直接学习这种低秩结构。** 所以 MLA 的低秩性不是事后近似，而是一种结构性参数化。

---

# 第二部分：MLA 与 MHA 的等价性分析

## 3. MLA 中的 key/value 压缩为什么"是对的"？

严格地说，需要分三层回答。

### 3.1 第一层：MLA 不等价于任意 MHA

标准 MHA 中 $k_t = W^K h_t$，$v_t = W^V h_t$，$W^K,W^V \in \mathbb{R}^{n_hd_h \times d}$ 是两个任意矩阵。

而 MLA 中 $k_t^C = W^{UK}W^{DKV}h_t$，$v_t^C = W^{UV}W^{DKV}h_t$。也就是说，MLA 等价于使用 $W^K_{\text{MLA}} = W^{UK}W^{DKV}$ 和 $W^V_{\text{MLA}} = W^{UV}W^{DKV}$ 的 MHA。但是这两个矩阵受到约束：

$$
\operatorname{rank}(W^K_{\text{MLA}}) \leq d_c,\quad \operatorname{rank}(W^V_{\text{MLA}}) \leq d_c
$$

而标准 MHA 中的 $W^K,W^V$ 未必满足这个秩约束。所以：**MLA 不是任意 MHA 的等价改写，而是一个低秩约束版本的 MHA。**

### 3.2 第二层：MLA 可以精确等价于某些 MHA

**命题 1**：设标准 MHA 的 key/value projection 为 $W^K, W^V \in \mathbb{R}^{m \times d}$（$m = n_h d_h$）。如果存在矩阵 $D \in \mathbb{R}^{d_c \times d}$，$U_K, U_V \in \mathbb{R}^{m \times d_c}$，使得 $W^K = U_KD$ 且 $W^V = U_VD$，那么 MLA 取 $W^{DKV}=D, W^{UK}=U_K, W^{UV}=U_V$ 时，对任意输入 $h_t$ 有 $k_t^C = W^K h_t$ 和 $v_t^C = W^V h_t$。因此 MLA 的 key/value 与该 MHA 完全相同。

**证明**：MLA 中 $c_t^{KV}=W^{DKV}h_t=Dh_t$，于是 $k_t^C=W^{UK}c_t^{KV}=U_KDh_t=W^Kh_t$。同理 $v_t^C=W^{UV}c_t^{KV}=U_VDh_t=W^Vh_t$。因为对于每个 token，每个 head 的 query、key、value 都相同，所以 attention score 相同，softmax 权重相同，输出相同。证毕。

**存在共享 $D$ 的充要条件**：上面的条件等价于说 $W^K$ 和 $W^V$ 的行空间都包含在 $D$ 的行空间中，即 $\operatorname{Row}(W^K) \subseteq \operatorname{Row}(D)$ 且 $\operatorname{Row}(W^V) \subseteq \operatorname{Row}(D)$。所以存在这样的 $D \in \mathbb{R}^{d_c \times d}$ 的充要条件是：

$$
\dim\left(\operatorname{Row}(W^K) + \operatorname{Row}(W^V)\right) \leq d_c
$$

也就是：

$$
\operatorname{rank}\begin{bmatrix} W^K \\ W^V \end{bmatrix} \leq d_c
$$

注意这里是把 $W^K,W^V$ 纵向拼接，$\begin{bmatrix}W^K \\ W^V\end{bmatrix} \in \mathbb{R}^{2m \times d}$。

**命题 2：MLA 精确表示标准 MHA 的充要条件**

给定标准 MHA 的 $W^K,W^V$，存在 MLA 参数 $W^{DKV},W^{UK},W^{UV}$ 使得 $W^K = W^{UK}W^{DKV}$ 且 $W^V = W^{UV}W^{DKV}$，当且仅当：

$$
\operatorname{rank}\begin{bmatrix} W^K \\ W^V \end{bmatrix} \leq d_c
$$

**必要性证明**：若存在，则 $\begin{bmatrix}W^K \\ W^V\end{bmatrix} = \begin{bmatrix}W^{UK} \\ W^{UV}\end{bmatrix} W^{DKV}$，因此 $\operatorname{rank}\begin{bmatrix}W^K \\ W^V\end{bmatrix} \leq \operatorname{rank}(W^{DKV}) \leq d_c$。

**充分性证明**：若 $\operatorname{rank}\begin{bmatrix}W^K \\ W^V\end{bmatrix}=s \leq d_c$，根据秩分解定理，存在 $A \in \mathbb{R}^{2m \times s}$ 和 $D_s \in \mathbb{R}^{s \times d}$ 使得 $\begin{bmatrix}W^K \\ W^V\end{bmatrix} = A D_s$。把 $A$ 按行分块为 $A = \begin{bmatrix}A_K \\ A_V\end{bmatrix}$，如果 $s<d_c$ 补零扩展到 $d_c$ 维即可构造出 $W^{DKV}, W^{UK}, W^{UV}$。证毕。

### 3.3 MLA 与 MHA 的关系总结

| 关系 | 是否成立 |
|------|---------|
| MLA 等价于任意 MHA | 不成立 |
| MLA 等价于低秩约束的 MHA | 成立 |
| MLA 可以精确表示某些 MHA | 成立，条件是 $\operatorname{rank}\begin{bmatrix}W^K\\ W^V\end{bmatrix}\leq d_c$ |
| MLA 可以近似任意 MHA | 在低秩近似意义下成立，但误差取决于奇异值谱 |
| MLA 的 latent-cache 推理形式与显式 key/value 形式 | 严格等价 |

---

# 第三部分：权重吸收的严格推导

## 4. MLA 的 latent-cache 推理为什么与显式 key/value 计算等价？

先不考虑 RoPE 分支，只看 content 部分。

MLA 显式形式为：

$$
c_j^{KV}=W^{DKV}h_j,\quad k_j^C = W^{UK}c_j^{KV},\quad v_j^C = W^{UV}c_j^{KV}
$$

### 4.1 Key 侧吸收的严格推导

把 $W^{UK}$ 按 head 分块：

$$
W^{UK} = \begin{bmatrix} W^{UK}_1 \\ W^{UK}_2 \\ \vdots \\ W^{UK}_{n_h} \end{bmatrix},\quad W^{UK}_i \in \mathbb{R}^{d_h \times d_c}
$$

于是 $k_{j,i}^C = W^{UK}_i c_j^{KV}$。

attention score 的 content 部分为：

$$
(q_{t,i}^C)^T k_{j,i}^C = (q_{t,i}^C)^T W^{UK}_i c_j^{KV} = \left((W^{UK}_i)^T q_{t,i}^C\right)^T c_j^{KV}
$$

定义 latent query：

$$
\tilde q_{t,i} = (W^{UK}_i)^T q_{t,i}^C \in \mathbb{R}^{d_c}
$$

则：

$$
(q_{t,i}^C)^T k_{j,i}^C = \tilde q_{t,i}^T c_j^{KV}
$$

这说明：**计算 attention score 时，不必显式恢复 $k_{j,i}^C$。只需要缓存 $c_j^{KV}$，并把 query 变换到 latent 空间即可。** 这是严格相等，不是近似。

### 4.2 Value 侧吸收的严格推导

显式 MLA 的第 $i$ 个 head 输出为：

$$
o_{t,i} = \sum_{j=1}^t a_{t,j,i} v_{j,i}^C = \sum_{j=1}^t a_{t,j,i} W^{UV}_i c_j^{KV}
$$

由于 $W^{UV}_i$ 与 $j$ 无关，可以提出求和号：

$$
o_{t,i} = W^{UV}_i \left(\sum_{j=1}^t a_{t,j,i} c_j^{KV}\right)
$$

定义 latent value aggregation：

$$
z_{t,i} = \sum_{j=1}^t a_{t,j,i} c_j^{KV} \in \mathbb{R}^{d_c}
$$

则 $o_{t,i} = W^{UV}_i z_{t,i}$。

最后 MHA 输出投影为 $u_t = W^O [o_{t,1};\dots;o_{t,n_h}]$。将每个 $o_{t,i}$ 代入，这等价于把 $W^{UV}$ 吸收到 $W^O$ 里。

因此：**不需要显式恢复所有 $v_{j,i}^C$。可以先对 latent $c_j^{KV}$ 做 attention 加权求和，再经过等价的线性映射得到输出。** 这也是严格相等。

### 4.3 加上 Decoupled RoPE 后的等价性

MLA 最终形式下，score 是：

$$
q_{t,i}^T k_{j,i} = (q_{t,i}^C)^T k_{j,i}^C + (q_{t,i}^R)^T k_j^R
$$

其中 $k_{j,i}^C = W_i^{UK}c_j^{KV}$，所以：

$$
(q_{t,i}^C)^T k_{j,i}^C = \left((W_i^{UK})^Tq_{t,i}^C\right)^T c_j^{KV}
$$

而 RoPE 部分 $(q_{t,i}^R)^T k_j^R$ 直接通过缓存 $k_j^R$ 计算。

所以完整 score 可以写成：

$$
q_{t,i}^T k_{j,i} = \tilde q_{t,i}^T c_j^{KV} + (q_{t,i}^R)^T k_j^R
$$

其中 $\tilde q_{t,i}=(W_i^{UK})^Tq_{t,i}^C$。

这说明，推理时只缓存 $c_j^{KV}$ 和 $k_j^R$，即可精确计算与显式恢复 key 后完全相同的 attention score。这正是论文中说的 $(d_c+d_h^R)l$ 的 KV cache。

### 4.4 关于 "$W^{UK}$ 吸收到 $W^Q$、$W^{UV}$ 吸收到 $W^O$" 的正确理解

严格来说，应当这样表述：

- **正确说法 1**：MLA 的显式 key/value 版本，与 MLA 的 latent-cache 优化版本，在数学上完全等价。这是论文中推理优化的核心。
- **正确说法 2**：MLA 等价于一种 key/value projection 被低秩约束、且 key/value 共享低维 latent 的 MHA 变体。
- **错误说法**：MLA 与任意标准 MHA 完全等价。

核心等价式是：

$$
(q_{t,i}^C)^T W_i^{UK}c_j^{KV} = \left((W_i^{UK})^Tq_{t,i}^C\right)^T c_j^{KV}
$$

以及：

$$
\sum_j a_j W_i^{UV}c_j^{KV} = W_i^{UV}\left(\sum_j a_j c_j^{KV}\right)
$$

这两个等式说明：推理时只缓存 latent $c_j^{KV}$，仍然可以得到与显式恢复 $k_j^C,v_j^C$ 完全相同的 MLA attention 输出。

---

## 5. 如果用近似角度看 MLA，可以给出什么误差结论？

假设有一个标准 MHA 的 $W^K,W^V$，我们用 MLA 的低秩形式近似它。令：

$$
\bar W = \begin{bmatrix} W^K \\ W^V \end{bmatrix}
$$

对它做 rank-$d_c$ 截断 SVD：

$$
\bar W_{d_c} = \arg\min_{\operatorname{rank}(B)\leq d_c} \|\bar W-B\|_F
$$

则存在 MLA 参数使得 $\begin{bmatrix} W^K_{\text{MLA}} \\ W^V_{\text{MLA}} \end{bmatrix} = \bar W_{d_c}$，且误差最小：

$$
\left\|\begin{bmatrix} W^K \\ W^V \end{bmatrix} - \begin{bmatrix} W^K_{\text{MLA}} \\ W^V_{\text{MLA}} \end{bmatrix}\right\|_F^2 = \sum_{i=d_c+1}^{p} \sigma_i^2
$$

这给出了"MLA 可以作为 MHA 的最优低秩近似"的数学依据。但需要注意：attention 中有 softmax，因此 $W^K,W^V$ 的小误差不必然线性地转化为输出小误差。不过在输入范数有界、score 不极端的情况下，可以用 softmax 的 Lipschitz 性给出连续性意义上的误差界。

---

# 第四部分：MLA 设计选择的数学分析

## 6. 为什么 key 的 RoPE 向量所有 heads 共享，而 query 每个 head 独立？

MLA 中 RoPE 分支的设计目标是：**用尽可能小的 cache 成本，为 attention score 提供足够的位置相关信息。**

在 MLA 里，真正需要缓存的是历史 token 的信息。对于自回归推理：
- 当前 token 的 query 是现算的，不需要缓存；
- 历史 token 的 key/value 需要缓存。

所以，**key 侧的维度设计直接决定 KV cache 大小**，而 query 侧的维度只影响当前步计算量，不影响历史缓存大小。

### 6.1 如果 key 的 RoPE 分支也每个 head 独立

假设每个 head 都有独立的 RoPE key $k_{t,i}^R \in \mathbb{R}^{d_h^R}$，那么每个 token 需要缓存 $n_h d_h^R$ 个 RoPE key 元素。DeepSeek-V2 中 $n_h=128, d_h^R=64$，RoPE key cache 是 $128 \times 64 = 8192$ 个元素。而论文采用 shared RoPE key 只缓存 64 个元素，相差 128 倍。

### 6.2 与 MQA 的关系

MLA 的 RoPE 分支中：多个 query heads 共享同一个 positional key $k_t^R$，但每个 head 仍然有自己的 positional query $q_{t,i}^R$。所以它在位置 key 这一部分有点像 MQA：key 侧共享减少 cache，query 侧多头独立保持 head-specific 的打分能力。

但是 MLA 的 content key/value 并不是简单共享的。MLA 中每个 head 的 content key/value 仍然可以不同：$k_{t,i}^C = W_i^{UK} c_t^{KV}$，$v_{t,i}^C = W_i^{UV} c_t^{KV}$。所以 MLA 不是简单的 MQA，而是：**content 部分通过 latent KV 生成多头 key/value；RoPE 位置 key 部分采用共享 key 以降低 cache。**

### 6.3 为什么共享一个 $k_t^R$ 还够用？

因为位置分数是 $(q_{t,i}^R)^T k_j^R$。虽然 $k_j^R$ 对所有 heads 共享，但 $q_{t,i}^R$ 是每个 head 独立的。所以不同 head 的位置打分仍然可以不同。只要 $q_{t,i}^R \neq q_{t,r}^R$，那么两个 head 对同一个位置 key $k_j^R$ 的响应就不同。

这类似于：共享一个位置特征库，每个 head 用自己的 query 去读取这个位置特征库。

### 6.4 为什么 query 端不共享？

因为 query 端不需要缓存。当前步生成时，query 只为当前 token 计算一次，用完即可。因此 MLA 可以让 query 保持多头独立。

| 分支 | 是否缓存 | 设计倾向 |
|------|---------|---------|
| query RoPE | 不缓存 | 可以多头独立，增强表达 |
| key RoPE | 需要缓存 | 尽量共享，降低 KV cache |

---

## 7. 为什么 value 不需要位置信息？

位置信息主要用于 attention score 的计算，也就是决定"该看哪里"。一旦注意力权重算出来，value 只需要提供被读取的内容即可。

标准 attention 是 $\operatorname{Attention}(Q,K,V) = \operatorname{Softmax}(\frac{QK^T}{\sqrt{d}})V$，其中 $Q,K$ 决定注意力权重，$V$ 是被加权汇聚的内容。

位置关系进入 $QK^T$，影响 $a_{t,j} = \operatorname{Softmax}_j(\frac{q_t^T k_j}{\sqrt{d}})$。一旦 $a_{t,j}$ 得到，输出就是 $o_t = \sum_j a_{t,j} v_j$。这里的 $v_j$ 已经通过索引 $j$ 与序列位置绑定。

value 的位置身份由两部分保证：
1. cache 中的索引 $j$
2. attention 权重 $a_{t,j}$

因此没有必要再对 value 加 RoPE。

在经典 Transformer 设计中，位置编码通常参与 attention score 的计算，而不是直接作用于 value。RoPE 尤其如此。RoPE 的一个重要性质是，旋转后的点积可以编码相对位置：$\operatorname{RoPE}(q,t)^T \operatorname{RoPE}(k,j)$ 包含与 $t-j$ 相关的信息。因此它天然应该作用在 Q/K 上，而不是 V 上。

---

## 8. $W^{UK}$ 被吸收到 $W^Q$、$W^{UV}$ 被吸收到 $W^O$，是否意味着推理时只需要缓存两个矩阵？

**不是。** 这里有一个常见误解。

论文说的"吸收"不是指"推理时只缓存两个矩阵"，而是指：**在推理计算图中，可以通过矩阵结合律改写计算，避免显式恢复完整的 historical key/value。**

模型权重本来就一直存在显存或权重存储中，不叫 KV cache。KV cache 指的是每个历史 token 的中间激活（key cache、value cache，在 MLA 中是 latent KV cache 和 RoPE key cache）。

### 8.1 MHA 中缓存的是每个历史 token 的 key/value 激活

$k_j = W^K h_j$，$v_j = W^V h_j$，不是缓存 $W^K,W^V$。$W^K,W^V$ 是模型参数，所有请求共享，不随序列长度增长。

### 8.2 MLA 中缓存的是 $c_j^{KV}$ 和 $k_j^R$

即 $\text{cache}_j = \{c_j^{KV}, k_j^R\}$，而不是缓存完整的 $k_j^C, v_j^C$。

### 8.3 吸收的实际含义

- **$W^{UK}$ 吸收到 query 侧**：推理时不显式算历史 token 的 $k_{j,i}^C$，而是当前 token 算一个 latent-space query $\tilde q_{t,i}$，直接和缓存的 $c_j^{KV}$ 做点积。
- **$W^{UV}$ 吸收到 output 侧**：可以先对 latent $c_j^{KV}$ 做加权求和，再通过合并后的输出投影得到最终结果，而不是先恢复所有 $v_{j,i}^C$。

推理时仍然需要模型权重和每个请求的 KV cache。不是只缓存 $W^Q,W^O$。

---

## 9. 训练和推理时 MLA 的参数、行为、数据流有什么不同？

### 9.1 参数：训练和推理本质上是同一套参数

MLA 的参数包括 $W^{DQ}, W^{UQ}, W^{QR}, W^{DKV}, W^{UK}, W^{UV}, W^{KR}, W^O$。训练和推理使用的是同一组学到的参数。

如果推理时做了权重吸收，可能会预先构造一些等价矩阵（如 $(W_i^{UK})^T W_i^{UQ}$ 或与 $W^O W^{UV}$ 类似的组合矩阵），但这些不是新的模型参数，而是原参数的等价重排或预计算。

### 9.2 数学行为：理想情况下训练和推理等价

如果忽略浮点误差、kernel 实现差异、量化误差、dropout 等训练专用机制，那么 MLA 的训练形式和推理优化形式应该数学等价。即：显式恢复 key/value 的 MLA 与 latent-cache 优化后的 MLA，应当输出相同结果。

### 9.3 数据流差异

**训练时**：输入完整序列 $H = [h_1,\dots,h_T]$，并行计算所有 token 的 $Q^C, Q^R, K^C, V^C, K^R$，构造 $Q = [Q^C;Q^R], K = [K^C;K^R]$，做全序列 causal attention。训练时没有"历史 token 逐步生成"的概念，因此不需要 KV cache。

**推理时**：自回归生成，第 $t$ 步只输入当前 token 的 hidden state $h_t$。计算 $c_t^Q, q_t^C, q_t^R, c_t^{KV}, k_t^R$，然后把 $c_t^{KV}$ 和 $k_t^R$ 追加到 cache。对于 attention score，使用历史 cache $\{c_j^{KV}, k_j^R\}_{j=1}^t$，而不是历史完整 $K^C,V^C$。

### 9.4 为什么不同？

1. **训练要并行，推理要增量**：训练时已知完整序列可以一次性矩阵化计算，推理时逐 token 生成需要复用历史计算结果。
2. **训练关注吞吐，推理关注 KV cache 和内存带宽**：推理时瓶颈往往是 KV cache 显存占用、从显存读取 KV cache 的带宽、batch size 受 KV cache 限制。
3. **训练需要保留反向传播所需激活**：训练时需要保存某些中间激活用于 backward，推理时不需要 backward，只保留生成后续 token 必需的 cache。

---

## 10. 为什么 query latent 能减少激活值内存？推理时是不是不用 query latent？

**推理时通常仍然会计算 query latent $c_t^Q$，因为 $q_t^C$ 和 $q_t^R$ 都由它生成；但推理时不需要缓存它。**

### 10.1 Query latent 的维度优势

DeepSeek-V2 中 $d_c' = 1536$，而完整 content query 维度是 $n_h d_h = 128 \times 128 = 16384$，RoPE query 总维度是 $n_h d_h^R = 128 \times 64 = 8192$。所以如果直接保存完整 query 相关激活，维度会很大。

### 10.2 训练时的优势

训练反向传播时，线性层通常需要保存输入激活。MLA 中 query 路径有 bottleneck $h_t \rightarrow c_t^Q \rightarrow q_t$，中间的 $c_t^Q$ 维度远小于完整 query 维度。在配合 recomputation/checkpointing 时，可以保存较小的 $c_t^Q$，必要时重算 $q_t^C,q_t^R$，避免保存所有高维 query 激活。

### 10.3 推理时

推理当前步计算 $c_t^Q = W^{DQ}h_t$，然后 $q_t^C = W^{UQ}c_t^Q$，$q_t^R = \operatorname{RoPE}(W^{QR}c_t^Q)$，之后用它和历史 cache 算 attention。算完当前 token 后，$c_t^Q$ 没有必要保存，因为未来 token 不会用历史 token 的 query。未来需要的是历史 token 的 key/value 信息。

| 对象 | 训练时 | 推理时 |
|------|--------|--------|
| $c_t^Q$ | 可能保存用于反向传播或重算 | 现算现用，不缓存 |
| $q_t^C,q_t^R$ | 用于全序列 attention | 当前步现算现用 |
| $c_t^{KV}$ | 训练中作为中间激活 | 推理中必须缓存 |
| $k_t^R$ | 训练中作为中间激活 | 推理中必须缓存 |

---

## 11. 为什么 MLA 低秩压缩后性能反而可能强于 MHA？

**没有定理保证 MLA 一定比 MHA 强。** 论文的结论是经验性的：在他们的实验设置中，MLA 比 MHA 表现更好，同时 KV cache 小得多。但可以从以下角度分析为什么它有可能更强：

### 11.1 MLA 的低秩压缩本质

MLA 的本质是：**将每个 token 可供 attention 读取的信息压缩成一个共享 latent memory $c_t^{KV}$，再由不同 head 通过不同投影从这个 latent memory 中读取 key/value 表示。**

等价于 $W^K_{\text{MLA}} = W^{UK}W^{DKV}$，$W^V_{\text{MLA}} = W^{UV}W^{DKV}$，并且 $K,V$ 共享同一个下投影 $W^{DKV}$。

### 11.2 它不是简单地减少 head 数

MQA 中 $k_{t,1}=k_{t,2}=\cdots=k_{t,n_h}$，直接削弱了 head 的多样性。MLA 中虽然缓存的是同一个 $c_t^{KV}$，但每个 head 有自己的 $W_i^{UK}$ 和 $W_i^{UV}$，不同 head 仍然可以得到不同的 key/value。MLA 的压缩不是"让所有头共享同一套 KV"，而是**共享底层 latent memory，但保留 head-specific 的读取方式。** 这比 MQA/GQA 的共享方式更细腻。

### 11.3 低秩 bottleneck 可能起到正则化作用

MLA 强制 $W^K = W^{UK}W^{DKV}$ 且 $W^V = W^{UV}W^{DKV}$，这意味着 $K,V$ 必须通过同一个 latent space。这种结构约束模型不要把 key/value 学成完全无关的表示，而是要求它们共享底层语义信息。这带来类似正则化的效果：减少冗余、降低过拟合、提高泛化、促使模型学习更紧凑的 token memory 表示。

从线性代数角度看，MLA 要求 $\operatorname{Row}(W^K)$ 和 $\operatorname{Row}(W^V)$ 都包含在同一个低维子空间中，模型被迫学习一个共享的"信息子空间" $c_t^{KV}=W^{DKV}h_t$。

### 11.4 Key 与 Value 共享 latent 可能提升一致性

标准 MHA 中 $W^K$ 和 $W^V$ 完全独立。但在 attention 中，key 和 value 的功能是强相关的：key 决定这个 token 是否被关注，value 决定被关注后提供什么内容。MLA 让二者共享 $c_t^{KV}$，这意味着 token 被检索的依据和 token 被读取的内容都来自同一个 latent memory。类比数据库：key 是索引，value 是记录内容，MLA 强制索引和内容都由同一份底层记录生成。

### 11.5 Decoupled RoPE 把内容与位置解耦

attention score 变成 $(q_{t,i}^C)^T k_{j,i}^C + (q_{t,i}^R)^T k_j^R$，content 部分负责语义匹配，RoPE 部分负责位置关系。这种显式解耦可能让模型更容易分别学习"看什么内容"和"看什么位置"。

### 11.6 系统层面的收益

MLA 大幅降低 KV cache 后，推理时可以使用更大 batch、支持更长 context、内存带宽压力降低、长上下文任务中可使用更多上下文信息。

### 11.7 神经网络中有效维度可能本来就低

从经验上看，大型神经网络的许多权重矩阵和激活表示都有低有效秩现象。如果 $W^K,W^V$ 的有效奇异值谱衰减很快，那么用低秩结构表示它们损失很小。更重要的是，MLA 不是训练后压缩 MHA，而是从头训练低秩结构，模型会主动适应这种结构。

### 11.8 总结

MLA 的低秩压缩至少改变了四件事：
1. **KV 的存储对象**：从存储展开后的多头 KV 变成存储压缩 latent memory
2. **K/V 的参数化方式**：从 $h_t \rightarrow k_t$ 变成 $h_t \rightarrow c_t^{KV} \rightarrow k_t,v_t$，多了一个 bottleneck
3. **多头共享方式**：MQA/GQA 是直接共享 K/V，MLA 是共享 latent 但 head-specific 地生成 K/V
4. **内容与位置的耦合方式**：$\text{score} = \text{content score} + \text{position score}$，让内容和位置在结构上解耦

---

# 第五部分：Value-Output 吸收的 BlockDiag 分析

## 12. 为什么 $B = W^O W^{UV}$ 不是 MLA value-output 路径的等价吸收？

$W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}$ 和 $W^O \in \mathbb{R}^{d \times n_h d_h}$，所以 $B = W^O W^{UV} \in \mathbb{R}^{d \times d_c}$。这个乘法在维度上成立。

但它对应的计算是 $u_t = B z_t$，其中只有一个共享的 $z_t \in \mathbb{R}^{d_c}$。而真实 MLA 中不是一个 $z_t$，而是每个 head 一个 $z_{t,1}, z_{t,2}, \dots, z_{t,n_h}$。

**关键点**：不同 attention head 的注意力权重不同，所以每个 head 对 latent cache $c_j^{KV}$ 的加权和不同。因此吸收后的对象不是一个 $d \times d_c$ 的矩阵，而是一个作用在所有 head-specific latent 聚合结果上的矩阵，形状是 $d \times (n_h d_c)$。

### 12.1 MLA value 路径的真实计算

对第 $i$ 个 attention head：

$$
v_{j,i}^C = W_i^{UV} c_j^{KV},\quad W_i^{UV} \in \mathbb{R}^{d_h \times d_c}
$$

第 $i$ 个 head 的 attention 输出：

$$
o_{t,i} = \sum_{j=1}^t a_{t,j,i} W_i^{UV} c_j^{KV} = W_i^{UV}\left(\sum_{j=1}^t a_{t,j,i} c_j^{KV}\right)
$$

定义每个 head 自己的 latent 聚合结果：

$$
z_{t,i} = \sum_{j=1}^t a_{t,j,i} c_j^{KV} \in \mathbb{R}^{d_c}
$$

于是 $o_{t,i} = W_i^{UV} z_{t,i}$。注意：$z_{t,i}$ 是 head-specific 的。不同 head 的 attention weights 不同，所以 $z_{t,1}, z_{t,2}, \dots, z_{t,n_h}$ 通常都不同。

### 12.2 输出投影的真实形式

将 $W^O$ 按 head 切块：$W^O = [W_1^O, W_2^O, \dots, W_{n_h}^O]$，其中 $W_i^O \in \mathbb{R}^{d \times d_h}$。则：

$$
u_t = \sum_{i=1}^{n_h} W_i^O o_{t,i} = \sum_{i=1}^{n_h} W_i^O W_i^{UV} z_{t,i}
$$

令 $B_i = W_i^O W_i^{UV} \in \mathbb{R}^{d \times d_c}$，那么：

$$
u_t = \sum_{i=1}^{n_h} B_i z_{t,i} = [B_1, B_2, \dots, B_{n_h}] \begin{bmatrix} z_{t,1}\\ z_{t,2}\\ \vdots\\ z_{t,n_h} \end{bmatrix}
$$

所以正确的吸收形式是：

$$
B_{\text{abs}} = [B_1,B_2,\dots,B_{n_h}] \in \mathbb{R}^{d \times n_h d_c}
$$

### 12.3 为什么 $W^O W^{UV}$ 不对？

$W^O W^{UV}$ 对应的计算是 $u_t = (\sum_i W_i^O W_i^{UV}) z_t$，这只有在 $z_{t,1}=z_{t,2}=\cdots=z_{t,n_h}=z_t$ 时才等价。但一般情况不同 head 的 attention weights 不同，所以 $z_{t,1} \neq z_{t,2} \neq \cdots \neq z_{t,n_h}$。

### 12.4 用 BlockDiag 矩阵形式

令 $z_t = [z_{t,1}; z_{t,2}; \dots; z_{t,n_h}] \in \mathbb{R}^{n_h d_c}$。每个 head 先通过自己的 $W_i^{UV}$：

$$
o_t = [o_{t,1}; \dots; o_{t,n_h}] = \operatorname{BlockDiag}(W_1^{UV}, \dots, W_{n_h}^{UV}) z_t
$$

其中 $\operatorname{BlockDiag}(W_1^{UV}, \dots, W_{n_h}^{UV}) \in \mathbb{R}^{n_h d_h \times n_h d_c}$。

然后 $u_t = W^O o_t = W^O \operatorname{BlockDiag}(W_1^{UV}, \dots, W_{n_h}^{UV}) z_t$。

因此真正的 absorbed matrix 是：

$$
B_{\text{abs}} = W^O \operatorname{BlockDiag}(W_1^{UV}, \dots, W_{n_h}^{UV}) \in \mathbb{R}^{d \times n_h d_c}
$$

而不是 $W^O W^{UV} \in \mathbb{R}^{d \times d_c}$。

关键差别是：$W^{UV}$ 和 $\operatorname{BlockDiag}(W_1^{UV}, \dots, W_{n_h}^{UV})$ 不是同一个东西。

### 12.5 参数量对比

DeepSeek-V2 中 $d = 5120, n_h = 128, d_h = 128, d_c = 512$。

**原始 value-output 参数量**：
- $W^{UV}$: $128 \times 128 \times 512 = 8,388,608$ (约 8.39M)
- $W^O$: $5120 \times 128 \times 128 = 83,886,080$ (约 83.89M)
- 合计：约 92.28M

**正确 absorbed value-output 矩阵参数量**：
- $B_{\text{abs}} \in \mathbb{R}^{d \times n_h d_c}$: $5120 \times 128 \times 512 = 335,544,320$ (约 335.54M)
- 这正好等价于 $n_h$ 个 $B_i \in \mathbb{R}^{d \times d_c}$：$n_h \times d \times d_c = 128 \times 5120 \times 512 = 335.54M$

注意：absorbed 后的完整等价矩阵参数量是 $d \times n_h d_c$ 而不是 $d \times d_c$。$n_h$ 个 $B_i$ 并不是说模型原本真的有 $n_h$ 个独立大矩阵，而是说：如果要把 value up-projection 和 output projection 完全吸收成一个等价的直接映射，那么这个映射必须对每个 head 的 $z_{t,i}$ 分别作用，因此等价于 $n_h$ 个 $B_i$ 或一个大的 $d \times n_hd_c$ 矩阵。

$d \times d_c$ 对应的结构是 $u_t = B z_t$，这等价于所有 heads 共用同一个 attention 聚合结果。这不是 MLA，也不是标准 multi-head attention，它会把多头 attention 的 value-output 路径严重合并。

### 12.6 直观理解

标准 multi-head attention 的核心是：**每个 head 有自己的 attention weights，所以每个 head 从历史 token 中读取到的内容不同。** MLA 虽然把每个 token 的 value latent 压缩成 $c_j^{KV}$，但每个 head 仍然会用自己的 attention weights 读出不同的 latent summary $z_{t,i} = \sum_j a_{t,j,i} c_j^{KV}$。所以最后有 $n_h$ 个不同的 latent summaries，输出侧必须能够分别处理这些 head-specific summaries。如果只用一个 $B \in \mathbb{R}^{d \times d_c}$，那就相当于假设所有 heads 读出来的是同一个 summary，会丢掉 multi-head 的本质。

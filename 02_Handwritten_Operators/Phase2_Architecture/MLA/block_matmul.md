# 分块矩阵乘法完全指南

> 关联文档：[MLA_Paper_Notes.md](mla_paper_notes.md) · [MLA_Math_Proofs.md](mla_math_proofs.md) · [MLA_Code_Walkthrough.md](mla_code_walkthrough.md)
>
> 本文档是理解 MLA 权重吸收（W^UK→W^Q, W^UV→W^O）的前置数学基础。
> §6–§8 的块对角矩阵乘法是理解 MLA Value-Output 吸收中 BlockDiag 结构的关键。

## 目录

**矩阵分块乘法（1-12）**

1. [最一般的分块矩阵乘法](#1-最一般的分块矩阵乘法)
2. [2 x 2 分块形式](#2-2-times-2-分块形式)
3. [行分块乘列分块：最核心的”内积”形式](#3-行分块乘列分块最核心的内积形式)
4. [列分块乘完整矩阵：横向输出拼接](#4-列分块乘完整矩阵横向输出拼接)
5. [行分块矩阵乘完整矩阵：纵向输出拼接](#5-行分块矩阵乘完整矩阵纵向输出拼接)
6. [块对角矩阵乘法](#6-块对角矩阵乘法)
7. [块对角矩阵乘横向分块矩阵](#7-块对角矩阵乘横向分块矩阵)
8. [横向分块矩阵乘块对角矩阵](#8-横向分块矩阵乘块对角矩阵)
9. [纵向分块矩阵乘横向分块矩阵：外积形式](#9-纵向分块矩阵乘横向分块矩阵外积形式)
10. [两矩阵乘法的几种常用分块形式总结](#10-两矩阵乘法的几种常用分块形式总结)
11. [一个判断技巧：看输入是否共享](#11-一个判断技巧看输入是否共享)
12. [回到 MLA 的一句话总结](#12-回到-mla-的一句话总结)

**PyTorch 算子补充**

13. [expand 函数的用法](#expand函数的用法)
14. [PyTorch 张量广播机制完整过程](#pytorch中张量的广播机制完整过程)
15. [torch.bmm() 与 torch.matmul 的区别](#torchbmm函数的作用语法与torchmatmul的区别)
16. [为什么权重矩阵以转置形式存储](#为什么在pytorch中的模型权重矩阵在内存中是以转置的形式存储的)
17. [权重矩阵在乘法中为什么总是在右侧](#pytorch中权重矩阵在参与矩阵乘法运算时是不是几乎都是以右侧矩阵的形式出现的)

---

下面专门讲**两个矩阵相乘时的各种分块形式**。你可以把分块矩阵乘法理解成：

> 普通矩阵乘法是”标量乘加”；  
> 分块矩阵乘法是把标量换成”小矩阵”，然后做”矩阵乘加”。

只要分块维度对齐，计算规则完全类似普通矩阵乘法。

---

# 1. 最一般的分块矩阵乘法

设：

$$
A \in \mathbb{R}^{m \times n}
$$

$$
B \in \mathbb{R}^{n \times p}
$$

所以：

$$
C = AB \in \mathbb{R}^{m \times p}
$$

如果把 $A$ 按行、列分块，把 $B$ 按行、列分块：

$$
A =
\begin{bmatrix}
A_{11} & A_{12} & \cdots & A_{1s}\\
A_{21} & A_{22} & \cdots & A_{2s}\\
\vdots & \vdots & \ddots & \vdots\\
A_{r1} & A_{r2} & \cdots & A_{rs}
\end{bmatrix}
$$

$$
B =
\begin{bmatrix}
B_{11} & B_{12} & \cdots & B_{1q}\\
B_{21} & B_{22} & \cdots & B_{2q}\\
\vdots & \vdots & \ddots & \vdots\\
B_{s1} & B_{s2} & \cdots & B_{sq}
\end{bmatrix}
$$

注意中间的分块数都必须是 $s$。也就是说，$A$ 的列分块数量要等于 $B$ 的行分块数量。

那么：

$$
C = AB
$$

也可以分块为：

$$
C =
\begin{bmatrix}
C_{11} & C_{12} & \cdots & C_{1q}\\
C_{21} & C_{22} & \cdots & C_{2q}\\
\vdots & \vdots & \ddots & \vdots\\
C_{r1} & C_{r2} & \cdots & C_{rq}
\end{bmatrix}
$$

其中：

$$
C_{ij}
=
\sum_{k=1}^{s} A_{ik}B_{kj}
$$

这就是最一般的分块矩阵乘法公式。

它和普通矩阵乘法：

$$
c_{ij} = \sum_k a_{ik}b_{kj}
$$

完全同构。

唯一差别是：

- 普通矩阵乘法中 $a_{ik}, b_{kj}$ 是标量；
- 分块矩阵乘法中 $A_{ik}, B_{kj}$ 是矩阵块。

---

# 2. $2 \times 2$ 分块形式

最常见的是 $2 \times 2$ 块矩阵乘法。

设：

$$
A =
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
$$

$$
B =
\begin{bmatrix}
B_{11} & B_{12}\\
B_{21} & B_{22}
\end{bmatrix}
$$

则：

$$
AB =
\begin{bmatrix}
A_{11}B_{11}+A_{12}B_{21}
&
A_{11}B_{12}+A_{12}B_{22}
\\
A_{21}B_{11}+A_{22}B_{21}
&
A_{21}B_{12}+A_{22}B_{22}
\end{bmatrix}
$$

这个形式非常像普通 $2 \times 2$ 矩阵相乘：

$$
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
\begin{bmatrix}
e & f\\
g & h
\end{bmatrix}
=
\begin{bmatrix}
ae+bg & af+bh\\
ce+dg & cf+dh
\end{bmatrix}
$$

只不过把 $a,b,c,d,e,f,g,h$ 换成了矩阵块。

---

# 3. 行分块乘列分块：最核心的“内积”形式

设：

$$
A = [A_1,A_2,\dots,A_s]
$$

这是把 $A$ 横向分块，也就是按列分块。

再设：

$$
B =
\begin{bmatrix}
B_1\\
B_2\\
\vdots\\
B_s
\end{bmatrix}
$$

这是把 $B$ 纵向分块，也就是按行分块。

则：

$$
AB
=
[A_1,A_2,\dots,A_s]
\begin{bmatrix}
B_1\\
B_2\\
\vdots\\
B_s
\end{bmatrix}
=
\sum_{k=1}^{s} A_kB_k
$$

这个形式最重要。

它的意义是：

> 把 $A$ 看作若干个横向块，把 $B$ 看作若干个纵向块，二者相乘就是对应块相乘后求和。

---

## 3.1 维度要求

假设：

$$
A_k \in \mathbb{R}^{m \times n_k}
$$

$$
B_k \in \mathbb{R}^{n_k \times p}
$$

那么：

$$
A_kB_k \in \mathbb{R}^{m \times p}
$$

所有 $A_kB_k$ 形状相同，因此可以相加。

最终：

$$
AB \in \mathbb{R}^{m \times p}
$$

---

## 3.2 计算意义

这个形式对应普通矩阵乘法中的“内积”：

普通向量内积：

$$
[a_1,a_2,\dots,a_s]
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_s
\end{bmatrix}
=
\sum_k a_kb_k
$$

分块矩阵版本：

$$
[A_1,A_2,\dots,A_s]
\begin{bmatrix}
B_1\\
B_2\\
\vdots\\
B_s
\end{bmatrix}
=
\sum_k A_kB_k
$$

所以它可以叫作**块内积形式**。

---

## 3.3 在 attention 里的典型例子

输出投影：

$$
W^O = [W_1^O,W_2^O,\dots,W_{n_h}^O]
$$

多头输出：

$$
o =
\begin{bmatrix}
o_1\\
o_2\\
\vdots\\
o_{n_h}
\end{bmatrix}
$$

那么：

$$
W^O o
=
[W_1^O,\dots,W_{n_h}^O]
\begin{bmatrix}
o_1\\
\vdots\\
o_{n_h}
\end{bmatrix}
=
\sum_{i=1}^{n_h} W_i^O o_i
$$

这就是多头输出先拼接、再过输出矩阵的分块展开。

---

# 4. 列分块乘完整矩阵：横向输出拼接

设：

$$
B = [B_1,B_2,\dots,B_q]
$$

即 $B$ 横向分块。

那么：

$$
AB
=
A[B_1,B_2,\dots,B_q]
=
[AB_1,AB_2,\dots,AB_q]
$$

---

## 4.1 维度

如果：

$$
A \in \mathbb{R}^{m \times n}
$$

$$
B_j \in \mathbb{R}^{n \times p_j}
$$

那么：

$$
AB_j \in \mathbb{R}^{m \times p_j}
$$

所以：

$$
AB =
[AB_1,\dots,AB_q]
\in \mathbb{R}^{m \times (p_1+\cdots+p_q)}
$$

---

## 4.2 计算意义

这个形式表示：

> 如果右矩阵按列分成几组，那么左乘 $A$ 可以分别作用到每一组列上，最后把结果横向拼起来。

这在批处理、多组向量同时变换时很常见。

---

## 4.3 例子

设：

$$
X = [x_1,x_2,\dots,x_T]
$$

每列是一个 token 的 hidden state。

线性层：

$$
Y = WX
$$

则：

$$
WX
=
W[x_1,x_2,\dots,x_T]
=
[Wx_1,Wx_2,\dots,Wx_T]
$$

意义是：

> 对每个 token 的 hidden state 分别应用同一个线性变换 $W$。

---

# 5. 行分块矩阵乘完整矩阵：纵向输出拼接

设：

$$
A =
\begin{bmatrix}
A_1\\
A_2\\
\vdots\\
A_r
\end{bmatrix}
$$

即 $A$ 纵向分块。

则：

$$
AB
=
\begin{bmatrix}
A_1\\
A_2\\
\vdots\\
A_r
\end{bmatrix}
B
=
\begin{bmatrix}
A_1B\\
A_2B\\
\vdots\\
A_rB
\end{bmatrix}
$$

---

## 5.1 维度

如果：

$$
A_i \in \mathbb{R}^{m_i \times n}
$$

$$
B \in \mathbb{R}^{n \times p}
$$

那么：

$$
A_iB \in \mathbb{R}^{m_i \times p}
$$

所以：

$$
AB \in \mathbb{R}^{(m_1+\cdots+m_r)\times p}
$$

---

## 5.2 计算意义

这个形式表示：

> 如果左矩阵按行分成几组，那么它们可以分别作用到同一个右矩阵 $B$ 上，最后纵向拼接结果。

---

## 5.3 在 MLA 里的例子

MLA 中：

$$
W^{UV}
=
\begin{bmatrix}
W_1^{UV}\\
W_2^{UV}\\
\vdots\\
W_{n_h}^{UV}
\end{bmatrix}
$$

对同一个 latent：

$$
c \in \mathbb{R}^{d_c}
$$

有：

$$
W^{UV}c
=
\begin{bmatrix}
W_1^{UV}c\\
W_2^{UV}c\\
\vdots\\
W_{n_h}^{UV}c
\end{bmatrix}
$$

意义是：

> 同一个 latent $c$ 被不同 head 的 $W_i^{UV}$ 投影成不同 head 的 value。

---

# 6. 块对角矩阵乘法

块对角矩阵形如：

$$
D =
\operatorname{BlockDiag}(D_1,D_2,\dots,D_s)
$$

即：

$$
D =
\begin{bmatrix}
D_1 & 0 & \cdots & 0\\
0 & D_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & D_s
\end{bmatrix}
$$

如果：

$$
X =
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_s
\end{bmatrix}
$$

那么：

$$
DX
=
\begin{bmatrix}
D_1X_1\\
D_2X_2\\
\vdots\\
D_sX_s
\end{bmatrix}
$$

---

## 6.1 计算意义

块对角矩阵表示：

> 每个子矩阵 $D_i$ 只处理自己对应的输入 $X_i$，不同块之间不发生混合。

它是“分组独立线性变换”的矩阵表达。

---

## 6.2 在 multi-head attention 里的意义

每个 head 有自己的变换：

$$
W_i^{UV}
$$

而每个 head 有自己的输入：

$$
z_i
$$

所以：

$$
\begin{bmatrix}
W_1^{UV} & 0 & \cdots & 0\\
0 & W_2^{UV} & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & W_{n_h}^{UV}
\end{bmatrix}
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_{n_h}
\end{bmatrix}
=
\begin{bmatrix}
W_1^{UV}z_1\\
W_2^{UV}z_2\\
\vdots\\
W_{n_h}^{UV}z_{n_h}
\end{bmatrix}
$$

这就是为什么当每个 head 的 $z_i$ 不同时，必须用块对角矩阵，而不能用简单的纵向拼接矩阵。

---

# 7. 块对角矩阵乘横向分块矩阵

上面讲的是块对角乘分块向量。更一般地，如果：

$$
X =
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_s
\end{bmatrix}
$$

其中每个：

$$
X_i \in \mathbb{R}^{r_i \times p}
$$

那么：

$$
\operatorname{BlockDiag}(D_1,\dots,D_s)
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_s
\end{bmatrix}
=
\begin{bmatrix}
D_1X_1\\
D_2X_2\\
\vdots\\
D_sX_s
\end{bmatrix}
$$

这里 $X_i$ 可以是一个向量，也可以是一批向量组成的矩阵。

---

# 8. 横向分块矩阵乘块对角矩阵

设：

$$
A = [A_1,A_2,\dots,A_s]
$$

以及：

$$
D = \operatorname{BlockDiag}(D_1,\dots,D_s)
$$

则：

$$
AD
=
[A_1,A_2,\dots,A_s]
\begin{bmatrix}
D_1 & 0 & \cdots & 0\\
0 & D_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & D_s
\end{bmatrix}
$$

结果是：

$$
AD =
[A_1D_1,A_2D_2,\dots,A_sD_s]
$$

---

## 8.1 计算意义

这表示：

> 左矩阵的每个横向块 $A_i$ 分别和对应的 $D_i$ 相乘，结果仍然横向拼接。

---

## 8.2 MLA value-output 吸收正是这个形式

设：

$$
W^O = [W_1^O,W_2^O,\dots,W_{n_h}^O]
$$

以及：

$$
D =
\operatorname{BlockDiag}(W_1^{UV},\dots,W_{n_h}^{UV})
$$

那么：

$$
W^O D
=
[W_1^O W_1^{UV},W_2^O W_2^{UV},\dots,W_{n_h}^O W_{n_h}^{UV}]
$$

也就是：

$$
B_{\text{abs}}
=
[B_1,B_2,\dots,B_{n_h}]
$$

其中：

$$
B_i = W_i^O W_i^{UV}
$$

这就是之前 value-output 吸收的分块乘法依据。

---

# 9. 纵向分块矩阵乘横向分块矩阵：外积形式

设：

$$
A =
\begin{bmatrix}
A_1\\
A_2\\
\vdots\\
A_r
\end{bmatrix}
$$

$$
B =
[B_1,B_2,\dots,B_q]
$$

那么：

$$
AB
=
\begin{bmatrix}
A_1\\
A_2\\
\vdots\\
A_r
\end{bmatrix}
[B_1,B_2,\dots,B_q]
=
\begin{bmatrix}
A_1B_1 & A_1B_2 & \cdots & A_1B_q\\
A_2B_1 & A_2B_2 & \cdots & A_2B_q\\
\vdots & \vdots & \ddots & \vdots\\
A_rB_1 & A_rB_2 & \cdots & A_rB_q
\end{bmatrix}
$$

---

## 9.1 计算意义

这个形式类似向量外积。

普通向量外积：

$$
\begin{bmatrix}
a_1\\
a_2
\end{bmatrix}
[b_1,b_2]
=
\begin{bmatrix}
a_1b_1 & a_1b_2\\
a_2b_1 & a_2b_2
\end{bmatrix}
$$

分块矩阵外积：

$$
\begin{bmatrix}
A_1\\
A_2
\end{bmatrix}
[B_1,B_2]
=
\begin{bmatrix}
A_1B_1 & A_1B_2\\
A_2B_1 & A_2B_2
\end{bmatrix}
$$

---

# 10. 两矩阵乘法的几种常用分块形式总结

| 分块形式 | 公式 | 计算意义 |
|---|---|---|
| 一般块乘法 | $C_{ij}=\sum_k A_{ik}B_{kj}$ | 块级别的矩阵乘法 |
| 横向乘纵向 | $[A_1,\dots,A_s]\begin{bmatrix}B_1\\ \vdots\\ B_s\end{bmatrix}=\sum_i A_iB_i$ | 块内积、对应块乘后求和 |
| 完整矩阵乘横向分块 | $A[B_1,\dots,B_q]=[AB_1,\dots,AB_q]$ | 同一左变换作用到多组列 |
| 纵向分块乘完整矩阵 | $\begin{bmatrix}A_1\\ \vdots\\ A_r\end{bmatrix}B=\begin{bmatrix}A_1B\\ \vdots\\ A_rB\end{bmatrix}$ | 多组行分别作用同一右矩阵 |
| 块对角乘纵向分块 | $\operatorname{BlockDiag}(D_i)\begin{bmatrix}X_i\end{bmatrix}=\begin{bmatrix}D_iX_i\end{bmatrix}$ | 分组独立变换 |
| 横向分块乘块对角 | $[A_i]\operatorname{BlockDiag}(D_i)=[A_iD_i]$ | 每个横向块分别右乘对应块 |
| 纵向分块乘横向分块 | $\begin{bmatrix}A_i\end{bmatrix}[B_j]=[A_iB_j]_{ij}$ | 块外积 |

---

# 11. 一个判断技巧：看输入是否共享

这个技巧对理解 MLA 特别重要。

## 情况 A：所有 head 用同一个输入

如果所有 head 都用同一个 latent：

$$
c \in \mathbb{R}^{d_c}
$$

那么可以用纵向拼接：

$$
\begin{bmatrix}
W_1\\
W_2\\
\vdots\\
W_h
\end{bmatrix}
c
=
\begin{bmatrix}
W_1c\\
W_2c\\
\vdots\\
W_hc
\end{bmatrix}
$$

这对应：

> 同一个输入，经过多个不同投影，得到多个输出。

---

## 情况 B：每个 head 有自己的输入

如果每个 head 有不同输入：

$$
z_1,z_2,\dots,z_h
$$

那么必须用块对角：

$$
\operatorname{BlockDiag}(W_1,\dots,W_h)
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_h
\end{bmatrix}
=
\begin{bmatrix}
W_1z_1\\
W_2z_2\\
\vdots\\
W_hz_h
\end{bmatrix}
$$

这对应：

> 不同输入，分别经过对应投影，互不混合。

---

# 12. 回到 MLA 的一句话总结

在 MLA 中：

- 对单个 token 的 latent $c_j^{KV}$ 生成所有 head 的 value 时，所有 head 共享同一个输入 $c_j^{KV}$，所以可以用纵向拼接矩阵：

  $$
  W^{UV}c_j^{KV}
  =
  \begin{bmatrix}
  W_1^{UV}\\
  \vdots\\
  W_h^{UV}
  \end{bmatrix}
  c_j^{KV}
  $$

- 但在 attention 加权之后，每个 head 得到自己的 latent summary：

  $$
  z_i=\sum_j a_{ij}c_j^{KV}
  $$

  此时输入已经变成 $z_1,\dots,z_h$，每个 head 不同，所以需要块对角矩阵：

  $$
  \operatorname{BlockDiag}(W_1^{UV},\dots,W_h^{UV})
  \begin{bmatrix}
  z_1\\
  \vdots\\
  z_h
  \end{bmatrix}
  $$

这就是分块矩阵乘法在 MLA 里最关键的区别。

## expand函数的用法

在 PyTorch 中，`expand` 函数的核心作用是：**在不实际复制内存的情况下，将张量（Tensor）沿着大小为 1 的维度，扩展（广播/Broadcast）到更大的尺寸。**

在我们刚才写的 MLA 手撕代码中，有一句非常关键的代码：
`k_rope.expand(-1, -1, self.num_heads, -1)`
它完美地体现了 `expand` 的用法和价值。下面我为你详细拆解这个函数的用法和底层逻辑。

### 1. `expand` 的三大核心规则

1. **只能扩展大小为 `1` 的维度：** 如果某个维度的大小不是 `1`，你不能用 `expand` 改变它。
2. **使用 `-1` 保持不变：** 在参数中传入 `-1`，代表“这个维度的大小保持和原来一样”。
3. **零内存开销（返回视图 View）：** 这是 `expand` 最伟大的地方。它**不会**在内存中真正把数据复制 $N$ 份，而是通过底层修改“步长（Stride）”为 0 来实现欺骗视觉的效果。多个扩展出来的元素在底层指向同一块物理内存。

---

### 2. 直观的代码示例

#### 示例 A：基础二维扩充
假设我们有一个形状为 `(3, 1)` 的列向量，我们想把它扩展成 `(3, 4)` 的矩阵。

```python
import torch

x = torch.tensor([[1], 
                  [2], 
                  [3]])  # shape: (3, 1)

# 将第 1 维（大小为 3）保持不变，第 2 维（大小为 1）扩展为 4
y = x.expand(3, 4)       
# 或者写成 y = x.expand(-1, 4) 也是一样的

print(y)
"""
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
"""
print(y.shape) # torch.Size([3, 4])
```

#### 示例 B：MLA 代码中的真实场景（多头广播）
在刚才的 DeepSeek MLA 代码中，计算 Key 的旋转位置编码（RoPE）时：
```python
b, seq_len = 2, 10
num_heads = 128
d_rope = 64

# 初始时，所有注意力头共享同一个 RoPE 向量
k_rope = torch.randn(b, seq_len, 1, d_rope)  # shape: (2, 10, 1, 64)

# 此时我们要把它和 shape 为 (2, 10, 128, 128) 的 k_nope 进行拼接
# 所以必须先把 k_rope 的 head 维度从 1 扩展到 128
k_rope_expanded = k_rope.expand(-1, -1, num_heads, -1) 

print(k_rope_expanded.shape) 
# 输出: torch.Size([2, 10, 128, 64])
```
**内存魔法揭秘：** 
如果不使用 `expand`，直接把 1 份数据强行复制成 128 份，显存会多占用 127 倍！而使用 `expand`，逻辑上这 128 个头拥有各自的 64 维向量，但**物理显存上依然只有最初的那 1 份数据**。这就为训练和推理省下了极其可观的显存。

---

### 3. 面试高频考点：`expand` vs `repeat`

面试官非常喜欢问：“`expand` 和 `repeat` 有什么区别？”

| 特性 | `expand` | `repeat` |
| :--- | :--- | :--- |
| **内存分配** | **不分配新内存**，返回原张量的视图 (View) | **分配全新内存**，真正把数据复制 N 份 (Copy) |
| **限制条件** | 只能扩展原大小为 `1` 的维度 | 可以把**任意**大小的维度重复 N 次 |
| **参数含义** | 传入的是**目标绝对尺寸** (如 `expand(3, 4)`) | 传入的是**倍数** (如 `repeat(1, 4)` 表示第2维重复4次) |
| **适用场景** | 仅仅是为了让两个 Tensor 维度对齐以进行数学计算（加法、乘法等） | 确实需要生成一份独立的数据用于后续的**修改**操作 |

**一个隐藏的“坑”（需注意）：**
因为 `expand` 共享内存，所以如果你修改了扩展后的张量，会牵一发而动全身：
```python
x = torch.tensor([[1]])   # (1, 1)
y = x.expand(2, 2)        # (2, 2)
# y 是 [[1, 1], [1, 1]]

y[0, 0] = 999 
print(y)
# 输出: 
# [[999, 999],
#  [999, 999]] 
# (因为底层步长为0，所有元素都指向同一个物理地址)
```
所以 `expand` 通常只用于**前向传播的读操作和矩阵乘法**中，而不会用于原位修改。如果扩展后必须要改变它的值，需要调用 `.contiguous()` 或 `.clone()` 将它变成真正独立的内存。


## Pytorch中张量的广播机制完整过程

### 1. 对齐维度（从左侧补1）

- 比较两个张量的 `ndim`（维度数）。
- 如果 `ndim` 不同，则在**维度数较少**的那个张量的形状**左侧（最外层）补 1**，直到两者维度数相等。  
  例如：`a.shape = (4, 3)`，`b.shape = (3,)` → 将 `b` 对齐为 `(1, 3)`。

这一步只是逻辑上的形状对齐，不会实际改变张量数据或内存。

---

### 2. 从最后一个维度（最内层）向前逐维检查

对齐后，两个形状的维度数相同，令它们为 `shape_a` 和 `shape_b`。从最后一维（索引 -1）开始，向前遍历所有维度，对每一维的尺寸 `(d_a, d_b)` 要求：

- `d_a == d_b`，或
- `d_a == 1`，或
- `d_b == 1`

如果某一维不满足以上任何条件，则**广播失败**，抛出 `RuntimeError`。

---

### 3. 确定最终广播后的输出形状

输出张量的每个维度尺寸取 `max(d_a, d_b)`。  
也就是：如果一方为 1，则取另一方的尺寸；如果两者相等，取该尺寸；不可能出现其他情况（因为已经通过了检查）。  
这个输出形状在计算前就确定了，且是**虚拟的**，不会真正分配一个这么大的连续内存。

---

### 4. 实际运算：通过步长（stride）实现零拷贝广播

在真正执行逐元素运算（如 `+`, `*`，或 `matmul` 中涉及广播的环节）时，PyTorch 不会复制数据，而是为每个需要广播的维度设置一个**步长为 0** 的视图（view）。

例如：`a = torch.randn(3, 1)`，`b = torch.randn(1, 4)`  
- `a` 的形状 `(3,1)`，步长可能是 `(1, 1)` 或 `(1, 0)`？  
  实际上，原始 `a` 的 `stride` 是 `(1, 1)`（因为最后一维长度为 1，步长 1 也是合理的，但更关键是：当该维度参与广播到 4 时，可以通过将步长设为 0 来实现重复）。  
- 在计算 `a + b` 时，PyTorch 为 `a` 创建一个视图：形状 `(3,4)`，步长 `(1, 0)`；为 `b` 创建一个视图：形状 `(3,4)`，步长 `(0, 1)`。  
  这样在遍历元素时，索引 `[i, j]` 实际访问的是 `a[i, 0]` 和 `b[0, j]`，没有任何数据被复制。

因此，**修改步长，不分配新内存**，这正是广播高效的原因。`Tensor.expand()` 就是基于这一机制显式返回一个广播视图。

---

#### 总结：你的描述已很接近正确，只需强调：
- **补 1 是在左侧（最外层），不是右侧。**
- **输出形状是逐维取最大。**
- **广播通过设置步长为 0 实现，不复制内存。**
- 检查顺序确为从后向前（最内层开始）。


## torch.bmm()函数的作用语法,与torch.matmul的区别？

### `torch.bmm` 语法与作用

```python
torch.bmm(input, mat2) → Tensor
```

- **作用**：对两个**3维张量**执行批量矩阵乘法。
- **输入形状**：
  - `input`：`(b, n, m)`
  - `mat2`：`(b, m, p)`
- **输出形状**：`(b, n, p)`
- **严格限制**：
  - 输入**必须都是 3 维**。
  - 第 0 维（batch 维）大小必须相等。
  - **不支持广播**（broadcasting）。

典型用途：Transformer 中，当 `Q` 和 `K^T` 都已显式扩展成相同批次数时，可用 `bmm` 计算注意力分数：
```python
# Q: [batch, heads, seq, d] → 转置或 view 后使用
attn_scores = torch.bmm(Q, K.transpose(1, 2))   # Q: [b*h, s, d]  K: [b*h, d, s]
```
（这里通常需将 `heads` 并入 batch 维，形成 3 维张量再调用 `bmm`。）

---

### `torch.matmul` 对比

```python
torch.matmul(input, other) → Tensor
```

`matmul` 是一个**通用矩阵乘法**，根据输入维度自动判定运算类型：

| input 维度 | other 维度 | 运算 |
|------------|------------|------|
| 1-D | 1-D | 点积（标量） |
| 2-D | 1-D | 矩阵‑向量乘 |
| 1-D | 2-D | 向量‑矩阵乘 |
| 2-D | 2-D | 普通矩阵乘法 |
| ≥2-D | ≥2-D | **批量矩阵乘法**，将最后两维视为矩阵，前方维度视为 batch，且**支持广播** |

**关键区别**：
- `bmm` 只接受 3-D，且**不广播**；  
- `matmul` 可以处理 1-D 到高维输入，**自动广播**前方维度。  
  例：形状 `(2, 1, 3, 4)` × `(1, 5, 4, 6)` → 输出 `(2, 5, 3, 6)`，其中 batch 维 `(2,1)` 和 `(1,5)` 广播为 `(2,5)`。
- 当输入都是 3-D 且不需要广播时，`bmm` 与 `matmul` 等价，但 `bmm` 可能更明确表达意图。

**建议**：  
- 编写库代码或需要严格形状检查时，用 `bmm` 会更安全（一有维度错误会立即报错）。  
- 日常开发中多用 `matmul`，它的灵活性更高，代码更简洁。


## 为什么在Pytorch中的模型权重矩阵在内存中是以转置的形式存储的？

你的理解非常准确，确实抓住了核心原因之一：**转置存储让矩阵乘法中对权重矩阵的访问模式变成连续的行读取，从而充分利用现代硬件的缓存和向量化指令。**

下面我从内存布局、BLAS 库调用约定、梯度更新一致性三个层次详细解释，并验证你的猜想。

---

## 1. 前向计算的实际操作与内存布局

### 数学上的线性层
\[
y = xW + b
\]
- \(x\)：`[N, in_features]`
- \(W\)：`[in_features, out_features]`
- \(y\)：`[N, out_features]`

### PyTorch 中的实现
`nn.Linear(in_features, out_features)` 的权重 `weight` 实际形状是 **`(out_features, in_features)`**，即 \(W^T\)。
前向传播时，PyTorch 调用 `F.linear(input, weight, bias)`，内部等价于：
```python
output = input @ weight.t() + bias
```
因为 `weight.t()` 的形状是 `(in_features, out_features)`，所以计算结果与数学一致。

---

## 2. 为什么存成转置？ —— 内存访问连续性的本质原因

你的猜想完全正确：**转置存储后，在计算 `x @ W^T` 时，对权重矩阵的读取是按“行”进行的，而行在内存中是连续存储的，避免了跨步访问带来的性能损失。**

### 直观对比
假设我们直接用 `W` 形状 `(in, out)` 做乘法 `y = x @ W`，底层伪代码为：
```python
for i in range(N):
    for j in range(out):
        sum = 0
        for k in range(in):
            sum += x[i, k] * W[k, j]   # W[k, j] 每次跳跃 `out` 个元素
```
访问 `W[k, j]` 时，相邻的 `k` 对应的地址间隔是 `out` 个元素，导致缓存行利用率极低，无法使用向量化的连续加载指令（如 AVX 的 `_mm256_loadu_ps`）。

### 转置存储后
权重存储为 `W_stored` 形状 `(out, in)`，即 `W_stored = W^T`。
计算 `y = x @ W_stored.T`，底层伪码：
```python
for i in range(N):
    for j in range(out):
        sum = 0
        for k in range(in):
            sum += x[i, k] * W_stored[j, k]   # W_stored[j, k] 行连续
```
此时对于固定的输出神经元 `j`，`W_stored[j, :]` 是内存中**连续的一段**，可以一次加载一个 cache line，并利用 SIMD 指令同时计算多个 `k`。这种访问模式对 CPU 的预取、缓存和向量化极其友好，对 GPU 的合并内存访问（coalesced access）同样有利。

---

## 3. 与 cuBLAS / BLAS 库的配合

PyTorch 底层调用 cuBLAS（GPU）或 Intel MKL/OpenBLAS（CPU）等高度优化的矩阵乘法库。这些库通常遵循 BLAS 接口标准：

```
cublasGemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

以计算 `C = α op(A) op(B) + β C` 为例，`op` 可以是**不转置（N）**或**转置（T）**。

为了利用连续访存，PyTorch 将线性层映射为：
- `A = input`：形状 `(N, in_features)`，不转置。
- `B = weight`：形状 `(out_features, in_features)`，**指定为转置（T）**，这样 BLAS 库内部将其视为 `(in_features, out_features)` 的矩阵，且实际访问时读的是 `weight` 的行（连续）。
- 即调用 `Gemm(N, T, N, out_features, in_features, ...)`。

这种设计让 BLAS 库能够以自己擅长的方式组织分块和内存访问，同时避免在调用前对权重进行昂贵的显式转置。

---

## 4. 梯度更新的一致性

反向传播时，损失对权重的梯度需要与权重形状一致，才能进行优化器更新。

- 如果权重形状为 `(out, in)`，梯度 `dW` 也应该是 `(out, in)`。
- 前向：`y = x @ W_stored.T`。
- 反向：`dL/dW_stored = dL/dy.T @ x`，形状恰好为 `(out, in)`。
  这里 `dL/dy` 是 `(N, out)`，`x` 是 `(N, in)`，所以 `dL/dy.T @ x` 也是 `(out, in)`，无需额外转置，直接求和更新即可。

如果权重形状为 `(in, out)`，那么梯度 `dL/dW = x.T @ dL/dy` 形状为 `(in, out)`，与权重一致，但此时权重按 `(in, out)` 存储，乘法 `x.T @ dL/dy` 又会遇到跨步访问问题（读 `x` 的行连续，但写回梯度时可能还好）。所以从梯度计算和参数更新的整体流水线看，`(out, in)` 的布局都更优。

---

## 5. 卷积层的相似设计

这个原理不仅限于线性层，卷积层权重（Conv2d）形状通常为 `(out_channels, in_channels, kH, kW)`，同样是为了在 `im2col` 或 Winograd 等算法中，让输出通道维度的连续访问与内存布局对齐。

---

## 结论

你的猜测完全正确：**转置存储的根本目的是让矩阵乘法中右侧矩阵的读取变成连续的行访问，从而利用缓存、SIMD 和 GPU 的合并访存，大幅提升性能。** 同时，这种设计也与 BLAS 库的转置参数约定、梯度更新形状无缝契合，形成了整体高效的实现。


## Pytorch中权重矩阵在参与矩阵乘法运算时，是不是几乎都是以右侧矩阵的形式出现的？

**是的，在 PyTorch 中，模型的权重矩阵在参与矩阵乘法时，几乎总是以右侧矩阵的形式出现。**  
这一设计贯穿于线性层、卷积层、循环层等几乎所有核心模块。

---

## 1. 具体表现

- **线性层 `nn.Linear(in, out)`**  
  权重 `weight` 的形状是 `(out, in)`，前向计算为：
  ```python
  y = x @ weight.t()   # weight.t() 形状 (in, out) 位于右侧
  ```

- **卷积层 `nn.Conv2d(in_c, out_c, k)`**  
  权重形状 `(out_c, in_c, kH, kW)`，计算时底层常通过 `im2col` 把输入展成 `(N, in_c*kH*kW)`，权重 reshape 成 `(out_c, in_c*kH*kW)`，乘法仍然是：
  ```python
  col @ weight_2d.t()   # weight_2d.t() 形状 (in_c*kH*kW, out_c) 位于右侧
  ```

- **循环层 `nn.LSTM` 等**  
  输入权重 `W_ih` 形状 `(4*hidden, input)`，隐藏权重 `W_hh` 形状 `(4*hidden, hidden)`，计算均为：
  ```python
  gates = x @ W_ih.t() + h @ W_hh.t()
  ```
  权重依旧在右侧。

- **注意力中的 QKV 投影**  
  如 `nn.Linear(d_model, d_k * num_heads)`，同样右侧。

唯一例外是 `nn.Embedding`（通过查表而非矩阵乘法），但它不属于矩阵乘法场景。

---

## 2. 为什么选择右侧矩阵？

### 2.1 内存访问连续性
矩阵乘法 `C = A @ B` 中，对 **右侧矩阵 B** 的访问模式是：固定输出列 `j`，沿着约简维 `k` 遍历行元素 `B[k, j]`。  
如果 `B` 按列主序（Fortran 风格）存储，则 `B[k, j]` 在内存中连续，对 CPU 的缓存和 SIMD 极其友好。但在 PyTorch（C 风格，行主序）中，要使列访问变成连续访问，必须 **让内存中实际存储的是 `B^T`**。  
于是 PyTorch 将权重存储为 `B^T`，即 `(out_features, in_features)`，计算时用 `A @ B`（即 `x @ W_stored.T`），此时 BLAS 内部以“右侧矩阵的转置”方式调用，实际读取的是 `W_stored` 的行，完美实现了连续访存。

### 2.2 BLAS 库调用约定
cuBLAS / MKL 等底层库的标准接口为：
```
GEMM(transA, transB, m, n, k, A, lda, B, ldb, C, ldc)
```
PyTorch 将线性层映射为：
- `transA = N` （输入不转置）
- `transB = T` （权重转置，即 `weight.t()` 是实际参与运算的矩阵）
这样 BLAS 可以直接将 `weight` 视为右侧矩阵的转置形式，内部以最优方式访问，无需在调用前显式拷贝转置。

### 2.3 梯度计算的一致性
反向传播时，损失对权重的梯度 `dW` 必须与 `weight` 形状一致（`out × in`）。  
对于 `y = x @ W.t()`，有：
```
dL/dW_stored = (dL/dy)^T @ x
```
形状正好是 `(out, in)`，可直接用于优化器更新，无需额外的维度置换。  
如果权重存为 `(in, out)` 且计算 `x @ W`，则梯度为 `x^T @ dL/dy` 形状 `(in, out)`，虽然也一致，但前向计算时右侧读取 `W` 的列不连续，效率低下。

### 2.4 参数初始化和导出便利
将权重存储为 `(out, in)` 使得 **输出维度连续**，便于逐输出神经元做初始化（如 kaiming_uniform_ 按 fan_in 或 fan_out 计算），也方便导出为其他框架（很多框架都沿用这种布局，比如 TensorFlow 的 Dense 层权重也是 `[in, out]`，但注意 TF 默认计算是 `x @ W`，其权重形状为 `(in, out)`，但 PyTorch 选择了转置存储以适应其内存布局偏好）。

---

## 结论

**在 PyTorch 的矩阵乘法计算中，权重矩阵始终位于右侧，且以转置形式存储。**  
这一设计是由“行主序内存 + 右侧矩阵列连续访问”这一性能需求驱动的，同时与底层 BLAS 调用、梯度形状无缝契合，成为 PyTorch 高效张量计算的基础约定。
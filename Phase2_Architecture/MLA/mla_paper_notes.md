# MLA (Multi-head Latent Attention) 论文精读笔记

> 本文提取自 DeepSeek-V2 论文的核心内容，涵盖 MLA 架构设计、动机、对比与代码实现思路。
>
> 相关文件：
> - [MLA 数学证明](mla_math_proofs.md) -- 低秩 KV 压缩与权重吸收的严格数学推导
> - [MLA 分块矩阵乘法](block_matmul.md) -- 分块矩阵乘法基础
> - [MLA 代码走读](mla_code_walkthrough.md) -- 手撕代码详解

---

## 一、论文核心内容总结：DeepSeek-V2

DeepSeek-V2 是一个面向高性能、低成本训练和高效推理的大规模开源 Mixture-of-Experts (MoE) 语言模型。其核心贡献在于通过新的注意力机制 MLA 和新的 MoE 架构 DeepSeekMoE，在保持强大模型能力的同时显著降低训练和推理成本。

### 1. 模型总体定位

| 项目 | DeepSeek-V2 |
|------|-------------|
| 总参数量 | 236B |
| 每个 token 激活参数量 | 21B |
| 上下文长度 | 128K tokens |
| 模型类型 | MoE Transformer |
| 预训练数据量 | 8.1T tokens |
| 主要语言 | 中英双语 |

论文强调，DeepSeek-V2 虽然总参数达到 236B，但由于采用稀疏 MoE 结构，每个 token 只激活约 21B 参数，因此在训练和推理时比同规模 dense 模型更加经济。

### 2. 核心架构创新

1. **Multi-head Latent Attention (MLA)**
2. **DeepSeekMoE**

---

## 二、MLA：降低 KV Cache 的注意力机制

传统 Transformer 的 Multi-Head Attention (MHA) 在推理时需要缓存每一层、每个 token 的 Key 和 Value。随着上下文长度增加，KV cache 成为推理瓶颈。

### MLA 的核心思想

MLA 不直接缓存完整的 key 和 value，而是将 key 和 value 联合压缩为一个低维 latent 向量：

$$
c_t^{KV} = W^{DKV} h_t
$$

然后再通过上投影恢复 key 和 value：

$$
k_t^C = W^{UK} c_t^{KV}
$$

$$
v_t^C = W^{UV} c_t^{KV}
$$

推理时只需要缓存压缩后的 latent 向量，而不是完整的 key/value。

### MLA 的优势

- 显著减少 KV cache
- 性能不低于甚至优于 MHA

在 DeepSeek-V2 中，MLA 使 KV cache 相比 DeepSeek 67B 减少了 93.3%。

---

## 三、Decoupled RoPE：解决 RoPE 与低秩压缩的不兼容

DeepSeek-V2 继续使用 RoPE 位置编码，但 RoPE 与低秩 KV 压缩天然不兼容。

原因是，如果直接对压缩后的 key 使用 RoPE，位置相关矩阵会阻碍推理时的矩阵吸收优化，导致必须重新计算历史 token 的 key，降低推理效率。

论文提出 **Decoupled Rotary Position Embedding**：

- 普通内容信息由压缩后的 latent KV 表示；
- 位置信息由额外的 query/key 分支承载；
- 推理时缓存压缩 latent 向量和额外的 RoPE key。

因此，DeepSeek-V2 的 KV cache 规模为：

$$
(d_c + d_h^R)l
$$

其中 $d_c$ 是 KV 压缩维度，$d_h^R$ 是解耦 RoPE key 的维度，$l$ 是层数。

---

## 四、MLA 详细解释

### 1. 背景：为什么 KV cache 是推理瓶颈？

在自回归生成中，模型一次生成一个 token。假设当前生成第 $t$ 个 token，那么它需要 attend 到前面所有 token：

$$
1,2,\dots,t
$$

如果每一步都重新计算所有历史 token 的 key 和 value，代价会非常高。所以标准做法是对每个历史 token 计算一次 key/value，将它们缓存下来，后续生成时直接读取缓存。这就是 **KV cache**。

问题是：对于大模型和长上下文，KV cache 非常大。

在标准 MHA 中，每层每个 token 的 KV cache 元素数是：

$$
2 n_h d_h l
$$

其中系数 $2$ 来自 key 和 value。DeepSeek-V2 支持 128K context length，因此如果仍然用传统 MHA，KV cache 会非常庞大，限制最大 batch size、上下文长度、推理吞吐和显存利用率。

因此 MLA 的主要目标是：**用更小的缓存表示历史 token，同时尽可能保留 MHA 的表达能力。**

### 2. 标准 Multi-Head Attention 回顾

设第 $t$ 个 token 的输入 hidden state 为 $h_t \in \mathbb{R}^d$。

标准 MHA 通过三个线性映射生成 query、key、value：

$$
q_t = W^Q h_t,\quad k_t = W^K h_t,\quad v_t = W^V h_t
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}$。然后把它们切成 $n_h$ 个 head：

$$
[q_{t,1}; \dots; q_{t,n_h}] = q_t,\quad [k_{t,1}; \dots; k_{t,n_h}] = k_t,\quad [v_{t,1}; \dots; v_{t,n_h}] = v_t
$$

每个 head 维度为 $d_h$。第 $i$ 个 head 的 attention 输出为：

$$
o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h}}\right) v_{j,i}
$$

最后拼接所有 head，做输出投影：

$$
u_t = W^O [o_{t,1}; o_{t,2}; \dots; o_{t,n_h}]
$$

### 3. 现有替代方案：MQA 和 GQA

**Multi-Query Attention (MQA)**：query 仍有多个 head，但 key/value 只有一组被所有 query heads 共享。KV cache 从 $2 n_h d_h l$ 减少到 $2 d_h l$，非常省显存，但表达能力下降明显。

**Grouped-Query Attention (GQA)**：MHA 和 MQA 的折中。query 有 $n_h$ 个 head，key/value 有 $n_g$ 个 group，多个 query heads 共享同一组 key/value。KV cache 为 $2 n_g d_h l$。

论文认为 GQA 和 MQA 虽然减少了 KV cache，但性能往往不如 MHA。所以 DeepSeek-V2 希望提出一种机制：KV cache 接近 MQA/GQA 的低成本，性能达到甚至超过 MHA。

### 4. MLA 的核心思想：低秩 Key-Value 联合压缩

MLA 最重要的设计是：**不分别缓存完整 key 和 value，而是把 key 和 value 联合压缩成一个低维 latent vector。** 论文称之为 Low-Rank Key-Value Joint Compression。

MLA 先把 hidden state 压缩到一个低维 latent 向量：

$$
c_t^{KV} = W^{DKV} h_t
$$

其中 $c_t^{KV} \in \mathbb{R}^{d_c}$，$W^{DKV} \in \mathbb{R}^{d_c \times d}$，且 $d_c \ll d_h n_h$。

然后再通过 up-projection 得到 key 和 value：

$$
k_t^C = W^{UK} c_t^{KV},\quad v_t^C = W^{UV} c_t^{KV}
$$

其中 $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$。

### 5. 为什么叫"低秩压缩"？

MLA 中 key 的生成为 $k_t^C = W^{UK} W^{DKV} h_t$，这相当于把原来的 key projection matrix 分解成两个矩阵 $W^K \approx W^{UK} W^{DKV}$，其中中间维度是 $d_c$。如果 $d_c < d_h n_h$，这就是一种低秩分解形式。value 同理。更重要的是，**key 和 value 共享同一个压缩 latent $c_t^{KV}$**，这就是 Key-Value Joint Compression。

### 6. MLA 如何减少 KV cache？

在标准 MHA 中，每层每个 token 缓存 $2 n_h d_h$ 个元素。在 MLA 中，推理时只缓存 $c_t^{KV}$，所以每层每个 token 缓存 $d_c$ 个元素。

以 DeepSeek-V2 的参数为例（$n_h = 128, d_h = 128, d_c = 512$）：
- MHA 每层 KV cache 为 $2 \times 128 \times 128 = 32768$
- MLA 每层只缓存 512 个元素（暂时不考虑 RoPE 分支）

压缩比例非常大。

### 7. 进一步优化：推理时不需要显式恢复完整 key/value

论文中关键的一句：**during inference, since $W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$, we even do not need to compute keys and values out for attention.**

对于 key 侧：attention score $q_t^T k_j^C = q_t^T W^{UK} c_j^{KV} = (W^{UK^T} q_t)^T c_j^{KV}$，可以先把 query 投影到 latent KV 空间，直接和缓存的 $c_j^{KV}$ 做点积。

对于 value 侧：$o_t = \sum_j a_j v_j^C = \sum_j a_j W^{UV} c_j^{KV} = W^{UV}(\sum_j a_j c_j^{KV})$，可以把 $W^{UV}$ 吸收到 $W^O$ 中。

实际实现时有两种思路：
1. 训练时显式恢复 key/value，方便实现；
2. 推理时使用吸收后的权重，避免展开完整 KV。

### 8. Query 也进行低秩压缩，但目的不同

DeepSeek-V2 也对 query 做低秩压缩：

$$
c_t^Q = W^{DQ} h_t,\quad q_t^C = W^{UQ} c_t^Q
$$

其中 $c_t^Q \in \mathbb{R}^{d_c'}$。

注意：**query 压缩不能减少 KV cache**。原因是推理时只缓存历史 token 的 key/value，不缓存 query。每一步新 token 的 query 都是现算的。query 压缩的主要目的是减少训练时 activation memory。

DeepSeek-V2 中：$d_c = 512, d_c' = 1536, d = 5120, n_h = 128, d_h = 128$。完整 query 维度是 $n_h d_h = 16384$，通过 query 压缩，中间 latent 只有 1536 维。

### 9. RoPE 与低秩 KV 压缩的冲突

MLA 设计中最微妙的一点是 RoPE。论文指出：**RoPE is incompatible with low-rank KV compression.**

如果直接对压缩恢复后的 key 加 RoPE：$k_t = R_t W^{UK} c_t^{KV}$，其中 $R_t$ 是位置 $t$ 对应的 RoPE 旋转矩阵。attention score 是 $q_t^T R_j W^{UK} c_j^{KV}$。

问题在于，$R_j$ 和位置 $j$ 有关。之前能吸收 $W^{UK}$ 到 query projection，但现在中间多了一个位置相关矩阵 $R_j$，不能简单地把 $W^{UK}$ 吸收到 $W^Q$ 中。更糟糕的是，如果只缓存 $c_j^{KV}$，为了得到带 RoPE 的 key，就必须对所有历史 token 重新计算 $R_j W^{UK} c_j^{KV}$，会严重降低推理效率。

### 10. Decoupled RoPE：解耦位置和内容

为了解决这个问题，DeepSeek-V2 提出 **Decoupled Rotary Position Embedding**，将 query/key 分成两部分：

1. **内容部分 (content part)**：来自低秩压缩，不使用 RoPE；
2. **位置部分 (rotary part)**：单独的小维度分支，使用 RoPE。

**Query 的两部分：**

$$
q_{t,i} = [q_{t,i}^C; q_{t,i}^R]
$$

其中 $q_{t,i}^C$ 来自 query latent 和 $W^{UQ}$，$q_{t,i}^R = \text{RoPE}(W^{QR} c_t^Q)$ 的每个 head 独立部分。

**Key 的两部分：**

$$
k_{t,i} = [k_{t,i}^C; k_t^R]
$$

其中 $k_{t,i}^C$ 来自 KV latent 和 $W^{UK}$，$k_t^R = \text{RoPE}(W^{KR} h_t)$ 是 shared key，被所有 heads 共享。

### 11. MLA 的完整 attention 计算

对于第 $i$ 个 head：

$$
o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^C
$$

由于 query/key 是拼接的，点积等价于：

$$
q_{t,i}^T k_{j,i} = (q_{t,i}^C)^T k_{j,i}^C + (q_{t,i}^R)^T k_j^R
$$

即：内容匹配分数 + 位置相关分数。

### 12. MLA 推理时到底缓存什么？

因为采用 Decoupled RoPE，推理时需要缓存：

1. 压缩后的 KV latent：$c_t^{KV}$
2. Decoupled RoPE key：$k_t^R$

每层每个 token 的缓存量是 $d_c + d_h^R$。DeepSeek-V2 中 $d_c = 512, d_h^R = 64$，所以每层每个 token cache 为 $512 + 64 = 576$。

标准 MHA 中每层每个 token cache 为 $2 \times 128 \times 128 = 32768$。单层压缩比例约为 $\frac{576}{32768} \approx 1.76\%$。

论文还从与 GQA 的等价角度比较：MLA 的 KV cache 相当于 GQA 中只有 2.25 个 groups，但性能强于 MHA。

### 13. MLA 和 MHA/GQA/MQA 的对比

| Attention | KV cache per token | 能力 |
|-----------|-------------------|------|
| MHA | $2n_hd_hl$ | Strong |
| GQA | $2n_gd_hl$ | Moderate |
| MQA | $2d_hl$ | Weak |
| MLA | $(d_c+d_h^R)l \approx \frac{9}{2}d_hl$ | Stronger |

直观理解：
- MHA：每个 head 都有自己的 key/value，能力强，但 cache 大；
- MQA：所有 head 共享一套 key/value，cache 极小，但能力弱；
- GQA：多个 query head 共享一组 key/value，折中；
- MLA：每个 token 缓存一个 latent，理论上可以通过 up-projection 生成多头 key/value，cache 小但表达力强。

### 14. MLA 为什么可能比 MHA 还强？

1. **低秩压缩带来参数结构化约束**：MLA 将 key/value 生成为 $h_t \rightarrow c_t^{KV} \rightarrow k_t^C, v_t^C$，这种结构类似 bottleneck，起到正则化作用。
2. **Key 和 Value 共享 latent**：传统 MHA 中 key 和 value 是独立投影，MLA 中二者共享 $c_t^{KV}$，增强 key/value 的一致性。
3. **Decoupled RoPE 将内容和位置显式分离**：$q^C,k^C$ 负责内容语义，$q^R,k^R$ 负责位置信息，显式分工让模型更容易学习。

### 15. DeepSeek-V2 的 MLA 超参数

| 参数 | 含义 | 数值 |
|------|------|------|
| $d$ | hidden size | 5120 |
| $n_h$ | attention heads | 128 |
| $d_h$ | 每个 head 的 content 维度 | 128 |
| $d_c$ | KV compression dimension | 512 |
| $d_c'$ | query compression dimension | 1536 |
| $d_h^R$ | RoPE 分支每头维度 | 64 |
| layers | 层数 | 60 |

对应维度如下：
- 输入 $h_t \in \mathbb{R}^{5120}$
- Query latent $c_t^Q \in \mathbb{R}^{1536}$
- Content query $q_t^C \in \mathbb{R}^{128 \times 128} = 16384$
- RoPE query $q_t^R \in \mathbb{R}^{128 \times 64} = 8192$
- KV latent $c_t^{KV} \in \mathbb{R}^{512}$
- Content key/value $k_t^C, v_t^C \in \mathbb{R}^{128 \times 128}$
- RoPE key $k_t^R \in \mathbb{R}^{64}$（所有 heads 共享）
- 最终每个 head 的 query/key 维度：content 128 + RoPE 64 = 192
- attention score scaling：$\sqrt{128 + 64} = \sqrt{192}$

---

## 五、MLA 的 PyTorch 实现思路

### Query 路径

```
q_latent = q_down_proj(x)          # [B, S, q_lora_rank]
q_content = q_up_proj(q_latent)    # [B, S, H * D]
q_rope = q_rope_proj(q_latent)     # [B, S, H * R]

q_content = q_content.view(B, S, H, D)
q_rope = q_rope.view(B, S, H, R)
q_rope = apply_rope(q_rope)
```

### KV 路径

```
kv_latent = kv_down_proj(x)         # [B, S, kv_lora_rank]
k_content = k_up_proj(kv_latent)    # [B, S, H * D]
v_content = v_up_proj(kv_latent)    # [B, S, H * D]
k_rope = k_rope_proj(x)             # [B, S, R]

k_content = k_content.view(B, S, H, D)
v_content = v_content.view(B, S, H, D)
k_rope = k_rope.view(B, S, 1, R)    # shared across heads
k_rope = apply_rope(k_rope)
k_rope = k_rope.expand(B, S, H, R)  # 广播到所有 head
```

### 拼接 query/key

```
q = torch.cat([q_content, q_rope], dim=-1)  # [B, S, H, D + R]
k = torch.cat([k_content, k_rope], dim=-1)  # [B, S, H, D + R]
v = v_content                                # [B, S, H, D]
```

注意：value 不拼接 RoPE 分支，仍然是 content value。

### Attention

```
q = q.transpose(1, 2)  # [B, H, S, D + R]
k = k.transpose(1, 2)  # [B, H, S, D + R]
v = v.transpose(1, 2)  # [B, H, S, D]

scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D + R)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)  # [B, H, S, D]

out = out.transpose(1, 2).contiguous().view(B, S, H * D)
out = o_proj(out)
```

### 训练版 vs 推理优化版

手写时建议先实现训练版/直观版 MLA：
1. 显式计算 $c^{KV}$
2. 显式恢复 $k^C$、$v^C$
3. 显式拼接 $q^C/q^R$ 和 $k^C/k^R$
4. 做标准 attention

推理优化版进一步做权重吸收：
- 不显式恢复历史完整 $k^C$ 和 $v^C$
- 只缓存 $c^{KV}$ 和 $k^R$
- 通过矩阵结合律把 $W^{UK}$ 和 $W^{UV}$ 吸收到相关投影中

---

## 六、MLA 的核心公式汇总

### Query

$$
c_t^Q = W^{DQ}h_t
$$

$$
q_t^C = W^{UQ}c_t^Q
$$

$$
q_t^R = \text{RoPE}(W^{QR}c_t^Q)
$$

$$
q_{t,i} = [q_{t,i}^C; q_{t,i}^R]
$$

### Key/Value

$$
c_t^{KV} = W^{DKV}h_t
$$

$$
k_t^C = W^{UK}c_t^{KV}
$$

$$
v_t^C = W^{UV}c_t^{KV}
$$

$$
k_t^R = \text{RoPE}(W^{KR}h_t)
$$

$$
k_{t,i} = [k_{t,i}^C; k_t^R]
$$

### Attention

$$
o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^C
$$

$$
u_t = W^O[o_{t,1}; o_{t,2}; \dots; o_{t,n_h}]
$$

### 推理缓存

$$
\text{cache per token per layer} = d_c + d_h^R
$$

即缓存 $c_t^{KV}$ 和 $k_t^R$。

---

## 七、一句话理解 MLA

> MLA 将传统 MHA 中每个 token 需要缓存的大量 multi-head key/value，压缩成一个小的 latent KV 向量，并额外用一个小维度 RoPE key 保存位置信息；内容匹配通过 latent 压缩表示完成，位置信息通过 decoupled RoPE 分支完成，从而在保持强模型能力的同时大幅降低 KV cache。

手写实现时最重要的三点：
1. KV 不是直接投影成完整 key/value 后缓存，而是先压缩成 $c^{KV}$
2. RoPE 不作用在低秩 content key 上，而是单独用 $q^R,k^R$ 分支承载
3. 最终 attention 的 query/key 是 content 部分和 RoPE 部分拼接，value 只来自 content value

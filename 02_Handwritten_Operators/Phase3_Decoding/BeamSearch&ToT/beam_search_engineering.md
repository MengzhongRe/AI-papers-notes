# Beam Search 工程实现笔记

> 关联文档：[README.md](README.md) · [BeamSearch_knowledge_notes.md](BeamSearch_knowledge_notes.md)
> 关联代码：[beam_search.py](beam_search.py)
> 相关子目录：[Top-P_Decoding/](../Top-P_Decoding/)

---

## 一、整体架构与 API 设计

### 1.1 函数签名

```python
def beam_search(logits_fn, input_ids, num_beams=4, max_new_tokens=128,
                eos_token_id=None, length_penalty_alpha=0.6, top_k=None):
```

参数与 HF `model.generate(num_beams=B, do_sample=False)` 的对应：

| 本实现 | HF generate | 说明 |
|--------|-------------|------|
| `num_beams` | `num_beams` | 束宽度 B |
| `max_new_tokens` | `max_new_tokens` | 硬截断上限 |
| `eos_token_id` | `eos_token_id` | 序列结束符 |
| `length_penalty_alpha` | `length_penalty` | GNMT 长度归一化 |
| `top_k` | — | 每 beam 词汇剪枝（HF 内部隐式实现） |

核心差异：我们明确地把 `do_sample=False`（确定性 beam search）固定下来，不提供采样模式——beam search 天然是确定性的。

### 1.2 为什么是 `logits_fn` 而不是 `model`

```python
# 不是这样：
def beam_search(model, input_ids, ...):
    logits = model(input_ids)

# 而是这样：
def beam_search(logits_fn, input_ids, ...):
    logits = logits_fn(input_ids)
```

三层动机：

1. **解耦前向逻辑**：beam search 只关心"给定序列，返回 logits"，不关心模型是 GPT/Llama/玩具函数。KV cache、torch.compile、eval 模式等由调用方在 `logits_fn` 内部管理，beam search 本身是纯算法函数。

2. **可测试性**：测试时不需要加载真实模型，传一个返回预定义 logits 的玩具函数即可。测试覆盖的是搜索逻辑本身，不是模型质量。

3. **接口灵活性**：调用方可以自由选择传 `model`（HF 模型）、`model.forward`（自定义包装）、或任何满足 `Callable[[Tensor], Tensor]` 签名的函数。

### 1.3 四段式结构

```
探针（获取 V）→ 初始化（B 条 beam）→ 逐步解码循环 → 最终选取
```

整个函数是纯函数式的——无 `nn.Module` 继承、无全局状态。这与项目中 `generate_next_token.py` 的风格一致。

---

## 二、Beam 初始化的 Bootstrapping 技巧

```python
beam_seqs = input_ids.repeat(B, 1)                         # [B, prompt_len]
beam_scores = torch.full((B,), -float('inf'), device=device)
beam_scores[0] = 0.0
```

同一个 prompt 复制 B 份，但只给第一条有效分数 0，其余 `-inf`。

**第一步展开时**：

```
beam_scores = [0.0, -inf, -inf, -inf]    (B=4)
next_log_probs = [..., -0.5, -1.2, ...]  (唯一有效 log-P 来自 beam[0])

candidate_scores[0] = 0 + next_log_probs   → 正常分数
candidate_scores[1] = -inf + ...           → 全 -inf
candidate_scores[2] = -inf + ...           → 全 -inf
candidate_scores[3] = -inf + ...           → 全 -inf
```

从 B×V 个候选取 Top-B：beam[0] 的 V 个候选中排出前 4 名 → **恰好分裂为 B 条真正的 beam**。

**设计收益**：避免为第一步单独写 `1 → B` 的特殊分支。如果不用这个技巧，第一步需要从 1 条序列展开 V 个候选取 Top-B，而后续步从 B 条序列展开 B×V 个候选取 Top-B——两套逻辑。Bootstrapping 让第一步和后续步复用同一套 `展开 → 剪枝` 循环体。

即使将来加上 KV cache，这个技巧同样适用：初始化时 B 条 beam 共享同一份 prompt KV cache，第一步后分裂为 B 份独立的 cache。

---

## 三、并行前向与已终止 beam

### 3.1 Batch 并行

```python
logits = logits_fn(beam_seqs)                              # [B, cur_len, V]
next_logits = logits[:, -1, :]                             # [B, V]
```

B 条 beam 以 batch 维度一次喂给 `logits_fn`。GPU 对 batch 维度天然并行——B 从 1 到 5，前向耗时几乎不增长（sub-linear）。

### 3.2 已终止 beam 为何不跳过前向

```python
next_log_probs[done_flags] = -float('inf')   # 让它"绝育"，不是跳过前向
```

`done_flags=True` 的 beam 仍然参与了 `logits_fn(beam_seqs)` 的前向计算。理论上可以跳过：

```python
# 跳过版本（未采用）
active = beam_seqs[~done_flags]          # 取出活跃 beam
active_logits = logits_fn(active)         # 只算活跃的
# 还需要把结果填回 [B, V] 的 done_flags 行...
```

不这样做的原因：

- **无 KV cache 时**：已终止 beam 最多白跑几步就 `done_flags.all()` 跳出循环了，浪费的计算是常数级的。而活跃索引映射 + scatter 回填的代码复杂度却是持续的。
- **有 KV cache + CUDA graph 时**：跳过导致每步 batch size 变化 → CUDA kernel launch shapes 不稳定 → `torch.compile` 和 CUDA graph 缓存失效 → **代价远大于收益**。工业实现（vLLM、HF）通常宁可用 dummy padding 保持 batch 形状固定，也不做动态剔除。

### 3.3 早停条件

```python
if done_flags.all():
    break
```

所有 B 条 beam 都命中 EOS 即退出。注意这只是条件之一——如果 B 条都没 EOS 但达到了 `max_new_tokens` 上限，也会退出（`for _ in range(max_new_tokens)` 自然结束）。

knowledge_notes §4.2 讨论了更复杂的"分数优势早停"，本实现未纳入以保持代码简洁。

---

## 四、Top-K 词汇剪枝

### 4.1 动机

B=5、V=128k 时，每一步有 5 × 128,000 = 640,000 个候选需要排序（`candidate_scores.view(-1)` → `torch.topk(..., B)`）。实际上对给定前缀，绝大多数 token 的概率极低（<10^{-4}），保留 Top-K=50 即可。搜索空间从 640k 压缩到 250，排序开销降低 3 个数量级。

### 4.2 实现

```python
if K < V:
    topk_vals, topk_idx = torch.topk(next_logits, K, dim=-1)    # [B, K]
    next_logits = torch.full_like(next_logits, -float('inf'))
    next_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)  # 只恢复 Top-K
```

关键细节：**不改变张量形状**。用 `full_like(-inf)` + `scatter_` 保持 `[B, V]` 形状，而不是截断为 `[B, K]`。这是因为后续 `log_softmax` → `beam_scores.unsqueeze(1) + next_log_probs` 的广播机制要求 `[B, V]` 形状。

如果你熟悉 Top-P sampling 中 `scatter_` 的用法（`generate_next_token.py` 第 70 行），这里的模式完全一致——`scatter_` 按索引把值填回原位置，未被填到的保持 `-inf`。

---

## 五、候选展开与中间剪枝

### 5.1 Log-Softmax 的数值考量

```python
next_log_probs = F.log_softmax(next_logits, dim=-1)            # [B, V]
```

用 `log_softmax` 而非 `log(softmax(x))`。前者在 PyTorch 内部用 `logits - logsumexp(logits)` 实现，避免了 softmax 的 `exp / Σexp` 中间结果在 fp16 下可能溢出或丢失精度的问题。

值域 ≤ 0（log-概率），后续与 `beam_scores`（也是负值累加）相加后自然维持单调递减趋势。

### 5.2 中间剪枝为什么不用长度归一化

```python
candidate_scores = beam_scores.unsqueeze(1) + next_log_probs   # [B, V]
top_scores, top_flat_idx = torch.topk(candidate_scores.view(-1), B)
```

**剪枝时只用原始对数概率，不做长度归一化。**

原因：当前步所有活跃 beam 的序列长度相同（都是从同一个 prompt 出发、生成到第 t 步），归一化等价于全体同除 `t^α`，不改变 Top-B 的相对排序。长度归一化只在最终选取时才有意义——那时的候选序列长度各不相同（有的 beam 在第 2 步就 EOS 了，有的跑了 20 步）。

### 5.3 `view(-1)` + 整数除法的索引映射

```python
top_flat_idx = torch.topk(candidate_scores.view(-1), B)          # 展平后的下标
beam_indices = top_flat_idx // V     # 还原: 来源 beam
token_indices = top_flat_idx % V     # 还原: 新 token
```

这是一个经典的二维→一维映射技巧：

```
[B, V] 展平为 [B*V]
flat_idx = b * V + v
b = flat_idx // V       (整数除法)
v = flat_idx % V        (取余)
```

不需要维护额外的索引映射表，两行算术完成还原。

---

## 六、EOS "退役 + 回收"机制

这是 beam search 最容易被忽略的工程细节。详见 knowledge_notes §4.1.1，这里聚焦代码实现。

### 6.1 三步走

```python
done_flags = done_flags[beam_indices]      # Step 1: 继承来源 beam 的 done 状态
just_hit_eos = (token_indices == eos_token_id)
done_flags = done_flags | just_hit_eos     # Step 2: 更新

for b in range(B):
    if just_hit_eos[b]:
        finished_seqs.append(beam_seqs[b].clone())    # Step 3: 退役
        finished_scores.append(beam_scores[b].item())
```

**Step 1 — 继承**：`done_flags` 和 `beam_seqs` 一样，必须按 `beam_indices` 重排。如果这一步漏了 `done_flags = done_flags[beam_indices]`，done 状态会跟错 beam。

**Step 2 — 标记**：`just_hit_eos` 是本步刚命中 EOS 的 beam。`done_flags | just_hit_eos` 兼容量复命中（同一 beam 不会两度 EOS，`|` 是幂等的，但安全）。

**Step 3 — 克隆退役**：`.clone()` 是关键。如果不 clone，下一轮 `beam_seqs = beam_seqs[beam_indices]` 时该 beam 可能被覆盖或修改——finished 池里的序列就会"变质"。

### 6.2 座位回收

下一步循环中，退役 beam 的 `done_flags=True` → `next_log_probs[done_flags] = -float('inf')` → 候选全部落选 → 空出的席位被其他活跃 beam 的某条"第二后代"填补。**活跃席位始终恒定 B，但 finished 池随 EOS 事件持续增长。**

### 6.3 最终比较池大小

```
最终比较池 = finished_seqs (历史上所有 EOS 退役的) + 活跃 beam (硬截断时仍在跑的)
大小 ≥ B，理论上界 B × max_new_tokens
```

举例 B=3：
```
Step 2: beam[0] EOS → finished[0] 退役, beam[1] 分叉补位
Step 4: beam[2] EOS → finished[1] 退役, beam[0] 分叉补位
Step 7: 硬截断, 3 条活跃 beam 都没 EOS

最终: finished(2条) + 活跃(3条) = 5 条 > B=3
```

---

## 七、最终选取与长度归一化

### 7.1 合并

```python
all_seqs = list(finished_seqs)
all_raw = list(finished_scores)
for b in range(B):
    if not done_flags[b]:
        all_seqs.append(beam_seqs[b].clone())
        all_raw.append(beam_scores[b].item())
```

注意 `if not done_flags[b]`——已退役的 beam 不会重复计入（它们的 `done_flags` 在活跃池里仍为 True，但内容已经在 `finished_seqs` 中了）。

### 7.2 GNMT 归一化

```python
normed = [raw / max(seq.size(0) - prompt_len, 1) ** length_penalty_alpha
          for seq, raw in zip(all_seqs, all_raw)]
```

`seq.size(0) - prompt_len` 是新生成 token 数。`max(..., 1)` 防止生成 0 个 token 时除零。

关于 α 的选取：0 = 短序列霸榜，0.6 = 经验最优（GNMT 推荐），1.0 = 完全按长度平均。详见 knowledge_notes §3.4。

### 7.3 为什么用 `max(key=...)` 选优

```python
best_idx = max(range(len(normed)), key=lambda i: normed[i])
```

这是 Python 标准库的 argmax 惯用法——等价于 `numpy.argmax(normed)`，但不需要 import numpy。

更深层的原因：`all_seqs` 是异构列表（各序列长度不同），无法 `torch.stack` 或 `np.stack`。既然本来就要用 Python list 承载，argmax 自然也用 Python 原生写法。

---

## 八、形状流转速查表

| 步骤 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `input_ids` | `[1, P]` | 单条 prompt |
| 初始化 | `beam_seqs` | `[B, P]` | prompt 复制 B 份 |
| 初始化 | `beam_scores` | `[B]` | 仅 [0]=0，其余 -inf |
| 初始化 | `done_flags` | `[B]` | 全 False |
| 前向 | `logits` | `[B, L, V]` | B 条序列各长 L |
| 取末位 | `next_logits` | `[B, V]` | 只取最后位置 |
| Top-K | `next_logits` | `[B, V]` | 形状不变，只保留 K 个有效值 |
| Log-Softmax | `next_log_probs` | `[B, V]` | 对数概率 |
| 候选展开 | `candidate_scores` | `[B, V]` | beam分数 + 对数概率 |
| 展平剪枝 | `top_flat_idx` | `[B]` | 0 ~ B·V-1 的 flat index |
| 最终输出 | 返回值 | `[1, P+N]` | N 为生成 token 数 |

---

## 九、测试架构

5 个冒烟测试分别验证 beam search 的一个核心性质：

| 测试 | 验证点 | 玩具模型 |
|------|--------|----------|
| Test 1 | B=1 退化为贪心 | `make_model`：固定每步最高分 token 3 |
| Test 2 | B>1 找非贪心全局最优 | `ConditionalModel`：token 0 高分后平庸 vs token 1 低分后极优 |
| Test 3 | EOS 早停 | `make_model`：EOS 作为合法候选参与 Top-B |
| Test 4 | 长度归一化 | `LengthNormModel`：短路径(低质早EOS) vs 长路径(高质晚EOS) |
| Test 5 | Top-K 不误伤高分 | `make_model`：K=3 时 token 7 作为最高分不应丢失 |

两种玩具模型模式：

- **`make_model(logits_table)`**：位置确定性——第 t 步的 logits 只看序列长度，不关心具体内容。适合测试单步行为（贪心、EOS、Top-K）。
- **`ConditionalModel` / `LengthNormModel`**：历史条件性——根据上一步实际选了哪个 token 决定当前分布。适合测试多步路径（非贪心全局最优、长度归一化）。

为什么不需要真实模型：beam search 是**搜索算法**，它只消费 logits 分布，不关心模型内部参数。只要玩具模型能产生"高分陷阱"和"低分暗门"两类典型模式，就能充分验证搜索逻辑。

---

## 十、局限性与扩展方向

### 当前简化

- **无 KV cache**：每次前向传完整序列。加 KV cache 后，需要同步管理 B 份 cache，并在 `beam_indices` 重排时对 cache 做同样的重排。
- **无 CUDA graph / torch.compile 兼容**：动态 `torch.cat` 和 `for` 循环中的 Python list append 不适合捕捉为静态图。
- **单 prompt**：`input_ids` 只接受 `[1, P]`。多 prompt 需要外层 for 循环，工业上可扩展为 `[Batch, P]` 的向量化 beam search。

### 可加但不加的理由

- **Repetition Penalty**：是采样策略的扩展，不属于 beam search 核心管线。加它会让核心函数变长 ~15 行。
- **Diverse Beam Search**：将 B 分为 G 组加多样性惩罚，本质是多组 beam search + 评分修正。详见 knowledge_notes §6.1。
- **Constrained Beam Search**：需要词表前缀树或 FSM，输入依赖完全不同。

### 返回 Top-B 条序列

当前只返回最优 1 条。如需返回 Top-B 条，将函数末尾改为 `return sorted(all_seqs, key=..., reverse=True)[:B]` 即可。HF `model.generate(num_return_sequences=B)` 就做这个——但注意它返回的是最终 beam，不是采样多样性（见 knowledge_notes §5.6 关于 Gemini 多候选的讨论）。

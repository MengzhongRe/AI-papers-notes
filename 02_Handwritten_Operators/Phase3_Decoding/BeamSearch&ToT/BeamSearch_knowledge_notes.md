# Beam Search 系统性知识笔记

> **Day 46-47 理论学习前置材料** — 本文档聚焦 beam search 的理论基础、数学原理与算法直觉，不涉及具体代码实现。工程实现细节将在 `beam_search.py` 中以注释形式呈现。

---

## 第 1 章：问题背景与动机

### 1.1 自回归解码的形式化定义

给定一个自回归语言模型 $P_\theta$，解码任务是在给定输入上下文 $x$ 的条件下，逐 token 生成输出序列 $y = (y_1, y_2, \ldots, y_T)$。模型在每一步 $t$ 输出条件概率分布：

$$
P_\theta(y_t \mid y_{<t}, x)
$$

整个序列的联合概率为连乘形式：

$$
P_\theta(y \mid x) = \prod_{t=1}^{T} P_\theta(y_t \mid y_{<t}, x)
$$

解码的目标是找到使该联合概率最大化的序列：

$$
y^* = \arg\max_{y \in \mathcal{Y}} \log P_\theta(y \mid x) = \arg\max_{y \in \mathcal{Y}} \sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)
$$

其中 $\mathcal{Y}$ 是所有可能的序列集合。对于词表大小 $|V|$（通常 32k-128k）和序列长度 $T$（通常 512-4096），搜索空间大小为 $|V|^T$，这是一个组合爆炸问题。

### 1.2 贪心解码（Greedy Decoding）的局限

贪心解码在每一步独立地选择概率最高的 token：

$$
\hat{y}_t = \arg\max_{y_t} P_\theta(y_t \mid y_{<t}, x)
$$

**核心问题**：局部最优 $\neq$ 全局最优。一个早期的低概率 token 可能通向一个整体概率更高的序列。

**直觉示例**：考虑一个简化的生成场景：
- 第 1 步：模型预测 token A 概率 0.6，token B 概率 0.4
- 如果选了 A（贪心），后续最优路径的总 log-prob 为 -1.5
- 如果选了 B，后续最优路径的总 log-prob 为 -0.8（更高）

贪心解码会选择 A，错失更优的全局路径。这在机器翻译等任务中尤为明显，因为不同语序、措辞选择往往需要"先抑后扬"。

### 1.3 暴力搜索的不可行性

完全搜索需要评估 $|V|$ 种可能性的指数级组合：
- $|V| = 50{,}000$，$T = 10$：$50{,}000^{10} \approx 9.7 \times 10^{46}$ 条路径
- 即便只考虑 Top-100 token 的剪枝：$100^{20} = 10^{40}$

因此，需要一种在搜索广度和计算可行性之间取得平衡的近似搜索算法，beam search 应运而生。

---

## 第 2 章：Beam Search 的核心算法

### 2.1 基本思想与直觉

Beam search 是一种**宽度受限的广度优先搜索（BFS）**。它维护 $B$ 条当前最优的部分序列（hypotheses），每步将每条 beam 展开到 $V$ 个可能的下一个 token，从 $B \times V$ 个候选中剪枝保留 Top-$B$，进入下一步。

**核心直觉**：
- $B=1$ → 退化为贪心搜索
- $B=|V|$（每步全保留）→ 退化为完全的 BFS（但不可行）
- $B$ 是"搜索预算"——在计算开销与搜索精度之间折中

### 2.2 算法流程

设 beam width 为 $B$，最大生成长度为 $T_{\max}$：

```
初始化: beams = { (序列=[BOS], 分数=0) }   # B 条 beam，初始仅包含起始符

for t = 1 to T_max:
    candidates = []
    for each (序列, 分数) in beams:
        计算 P(y_t | 序列, x)  得到 |V| 个下一 token 及其概率
        for each 候选 token:
            新序列 = 序列 + token
            新分数 = 分数 + log P(token | 序列, x)
            candidates.append((新序列, 新分数))

    # 从所有 B × V 候选中选出分数最高的 B 个
    beams = top-B(candidates)  按分数降序

    if 所有 beam 都遇到 EOS:  提前终止
```

这个看似简单的算法有几个关键细节点，下面逐一展开。

### 2.3 Beam Width $B$ 的含义与权衡

| B 值 | 行为 | 计算量 | 搜索质量 |
|------|------|--------|----------|
| 1 | 退化为贪心搜索 | 最小 | 局部最优 |
| 3-5 | 轻量 beam search | 线性增长 | 明显改善 |
| 10-20 | 标准 beam search | 较大 | 翻译/摘要任务中接近饱和 |
| 50+ | 宽 beam search | 显著增长 | 可能因退化而**质量下降** |

**关键权衡**：
- **计算成本**：每步需评估 $B$ 个前缀 → 相当于 batch size $B$ 的前向传播
- **KV Cache 膨胀**：每条 beam 需独立维护 KV Cache，内存随 $B$ 线性增长
- **收益递减**：$B$ 从 1 增至 5 收益巨大；$B$ 从 10 增至 20 收益甚微；$B > 50$ 在某些任务中反而降低质量（见第 7 章）

### 2.4 搜索树的可视化理解

以一个微型词表 $V=\{a, b, c, d, e\}$、$B=2$ 为例：

```
Step 0:                       [BOS] (score=0)
                                 │
Step 1:             展开 B×V=10 个候选，取 Top-2
                       ┌─────────┴─────────┐
                     [a] (0.5)           [c] (0.3)
                       │                    │
Step 2:          各自展开 5 个候选        各自展开 5 个候选
              ┌──────┼──────┐        ┌──────┼──────┐
           [a,a]  [a,b]  [a,c]   [c,a]  [c,c]  [c,d]
           (0.4)  (0.8)  (0.6)   (0.7)  (0.2)  (0.65)
            ✗      ✓       ✗       ✓      ✗       ✗
            (被剪)  (保留)  (被剪)   (保留)  (被剪)   (被剪)
```

**关键观察**：
- 来自高分 beam `[a]` 的扩展 `[a,c]`（score=0.6）被剪掉了，而来自低分 beam `[c]` 的扩展 `[c,a]`（score=0.7）成功挤入 Top-2
- 这正是 beam search 相比贪心搜索（只保留最优前缀）的优势——**低分前缀可能在下一步"逆袭"**，贪心搜索会永远错过 `[c,a]` 这条路径
- 同时注意 `[a,a]`（重复 token a）虽然来自高分 beam，但分数很低（0.4），因为模型在当前前缀 `[a]` 下分配了极低的 $P(\texttt{a} \mid \texttt{[a]})$

### 2.5 候选剪枝与搜索空间压缩

在实践中，每一步展开 $B \times |V|$ 的完整空间是计算瓶颈——$B=5, |V|=50{,}000$ 意味着每一步要评估 250,000 个候选的分数。实际工程中有两种常见的剪枝策略：

#### Top-K 词汇表限制

每步展开时，不取完整的 $|V|$ 个 token，只取模型在当前 beam 的 logit 分布中概率最高的 $K$ 个 token（如 $K=50$）。这直接将搜索空间从 $B \times |V|$ 压缩至 $B \times K$。

**直觉合理性**：对于任意给定的前缀，模型在绝大多数 token 上的概率都极小（$\ll 10^{-4}$），真正有意义的候选通常仅在 Top-50 内。这也被 HuggingFace 等框架隐式采用（通过 `num_beams` 与内部优化结合）。

#### Score-based Pruning（GNMT 2016）

丢弃分数低于当前最优候选某个阈值 $\delta$ 的候选：

$$
\text{保留候选} \iff \text{score}(\text{候选}) \geq \text{score}(\text{最优候选}) - \delta
$$

这种策略破坏了 beam width 的固定预算（可能导致每步 beam 数不稳定），在现代 GPU 批处理中不常用，但其思想启发了后来的 dynamic beam allocation。

---

## 第 3 章：评分函数与长度归一化

### 3.1 原始评分：对数概率连乘

最直接的 beam 评分是对数概率之和：

$$
s(y) = \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x)
$$

因为每个 token 的 log-prob 都是负数（$\log P \leq 0$），所以 $s(y)$ 随序列长度 $|y|$ 单调递减。

### 3.2 短序列偏差（Length Bias）的数学分析

**核心问题**：$\mathbb{E}[s(y)]$ 随 $|y|$ 单调递减。因为每一步都加上一个负值，长序列的累积分数天然低于短序列。

**后果**：标准 beam search 在没有长度归一化时，倾向于：
- 过早生成 EOS（"I'm fine" 而非 "I'm doing well, thank you"）
- 输出异常短的序列
- 在翻译中省略源句的部分内容（欠翻译）

### 3.3 长度归一化（Length Normalization, GNMT 2016）

Google 的 GNMT 系统（Wu et al., 2016）提出了长度归一化的评分函数：

$$
\text{score}(y, x) = \frac{\sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x)}{(5 + |y|)^\alpha / (5 + 1)^\alpha}
$$

其中：
- $|y|$ 是序列长度（token 数）
- $\alpha \in [0, 1]$ 是长度惩罚系数
- 常数 $5$ 是一个平滑项，防止极短序列被过度惩罚

**三种边界情况的直觉**：
- $\alpha = 0$：退化为原始评分（短序列霸榜）
- $\alpha = 1$：完全按长度平均（每个 token 贡献权重相等）
- $\alpha \in (0.5, 0.7)$：实践中常用的折中区间

**简化形式**（现代很多实现使用）：

$$
\text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t})
$$

### 3.4 $\alpha$ 的超参数选择

| $\alpha$ 值 | 效果 | 适用场景 |
|-------------|------|----------|
| 0 | 无惩罚，偏好短序列 | 几乎不推荐 |
| 0.2-0.4 | 轻度惩罚 | 倾向于简洁的输出 |
| 0.6-0.7 | 经验最优区间 | 机器翻译、摘要（GNMT 推荐 $\alpha \approx 0.6$） |
| 1.0 | 完全按长度平均 | 偏向过长的序列 |
| > 1.0 | 过度惩罚短序列 | 可能导致"话痨"输出 |

**任务差异性**：
- 翻译/摘要（目标长度相对固定）→ $\alpha \approx 0.6-0.8$
- 开放式生成（长度不确定）→ 通常不推荐 beam search，若用则 $\alpha$ 偏小
- 代码生成（有明确结束标志）→ 长度归一化影响较小

### 3.5 Coverage Penalty（覆盖惩罚）——历史背景说明

GNMT 论文还引入了一种 coverage penalty：基于交叉注意力权重，惩罚未充分关注源句中某部分的解码路径。公式为：

$$
cp(X, Y) = \beta \sum_{i=1}^{|X|} \log\left(\min\left(\sum_{j=1}^{|Y|} p_{i,j},\ 1.0\right)\right)
$$

其中 $p_{i,j}$ 是第 $j$ 个目标 token 对第 $i$ 个源 token 的注意力概率。

**在现代 LLM 中的定位**：coverage penalty 是 encoder-decoder 架构的产物。现代 decoder-only LLM（GPT 系列）没有显式的 source-target attention，因此 coverage penalty **不直接适用**。其精神后代是 repetition penalty 和 n-gram blocking（见第 7.3 节）。

### 3.6 一个尖锐问题：长度归一化真的解决了 EOS 早停偏差吗？

**问题**：当一个 beam 较早生成 EOS 后停止分数累积，而其他 beam 仍在继续、每次加上负的 log-prob，原始评分下早期 EOS 的 beam 是否天然占优？长度归一化能扭转这个局面吗？

#### 数值实验

考虑两条竞争路径：

- **Beam A（短）**：5 步后生成 EOS，平均每步 log-prob = -0.6，原始分数 = -3.0
- **Beam B（长）**：20 步后生成 EOS，平均每步 log-prob = -0.5，原始分数 = -10.0

**无长度归一化时**：
```
Beam A: -3.0   vs   Beam B: -10.0   →   A 碾压胜出（差距 7.0）
```
Beam B 虽然 per-token 质量更高（-0.5 vs -0.6），但因为"多走了 15 步"被严重惩罚。

**加入长度归一化 ($\alpha = 0.6$) 后**：
```
Beam A: -3.0 / 5^0.6 = -3.0 / 2.63 ≈ -1.14
Beam B: -10.0 / 20^0.6 = -10.0 / 6.03 ≈ -1.66
差距 = 0.52  ← 大幅缩小
```

如果 Beam B 的 per-token 质量再稍好一点（比如平均 -0.38/token），就能翻盘：
```
Beam B: (-0.38 × 20) / 20^0.6 = -7.6 / 6.03 ≈ -1.26
Beam A: -3.0 / 2.63 ≈ -1.14  → A 仍胜，但极其接近
```
```
Beam B: (-0.35 × 20) / 20^0.6 = -7.0 / 6.03 ≈ -1.16  → 几乎平手
Beam B: (-0.32 × 20) / 20^0.6 = -6.4 / 6.03 ≈ -1.06  → B 反超！
```

#### 结论

长度归一化**不是**让长序列必胜，而是做了一件事：

> **把胜负的判定从"谁更短"转移到"谁的 per-token 质量更高"。**

- **没有归一化**：短序列几乎总是赢，无论质量多差
- **有归一化 ($\alpha=0.6$)**：短序列仍有 edge（分母幂次 < 1），但高质量长序列有机会翻盘
- **$\alpha=1.0$**：完全按长度平均，长短序列在"单位 token 质量"维度上公平竞争——但可能矫枉过正，导致模型偏好不必要地啰嗦

**$\alpha$ 的工程意义**：它控制的是"长度惩罚的强度"，本质上是声明"我们认为每增加一个 token 应该被惩罚多少"。$\alpha=0.6$ 是一个经验性折中——它足够弱以避免"话痨"，又足够强以给长序列一个公平机会。

#### 更根本的缓解：早停条件 2

即使有长度归一化，EOS 早停的"不公平"仍然在**解码过程中**存在——已终止 beam 停在原地，未终止 beam 持续被扣分。这就是为什么第 4.2 节的**分数优势条件早停**很重要：

> 如果在某个时刻，最优已完成序列的归一化分数超过了所有未完成 beam **可能达到的最高归一化分数**（考虑未来 token 最多贡献 log(1)=0），则直接终止所有搜索。

这个条件防止了"明知追不上还继续跑"的浪费，并且使得 EOS beam 的比较变得公平——因为终止条件是"已完成的已经足够好，未完成的无论怎么生成都超不过它"。

### 3.7 更细致的追问：归一化后的分数还会随长度增长而单调下降吗？

**答案：不一定。归一化后的分数可能在生成高质量 token 时**上升**。**

这与原始评分有本质区别。原始评分每步加一个负值，**严格单调递减**；而长度归一化后，分母也在增长，当新 token "质量高于历史平均水平"时，归一化分数会反弹。

#### 逐步实例：生成 "The cat sits on the mat"

假设一个模型逐 token 生成以下英文句子，每一步的真实概率如下：

```
Step 1: P("The" | [BOS])      = 0.80,  log = -0.223
Step 2: P("cat" | "The")      = 0.30,  log = -1.204
Step 3: P("sits" | "The cat") = 0.50,  log = -0.693
Step 4: P("on" | "...sits")   = 0.60,  log = -0.511
Step 5: P("the" | "...on")    = 0.70,  log = -0.357
Step 6: P("mat" | "...the")   = 0.40,  log = -0.916
Step 7: P(EOS  | "...mat")   = 0.60,  log = -0.511
```

**原始评分（纯累加）：严格单调递减**

| Step | Token | 单步 log-P | **累积 log-P** | 趋势 |
|------|-------|-----------|----------------|------|
| 1 | The | -0.223 | **-0.223** | — |
| 2 | cat | -1.204 | **-1.427** | ↓ |
| 3 | sits | -0.693 | **-2.120** | ↓ |
| 4 | on | -0.511 | **-2.631** | ↓ |
| 5 | the | -0.357 | **-2.988** | ↓ |
| 6 | mat | -0.916 | **-3.904** | ↓ |
| 7 | EOS | -0.511 | **-4.415** | ↓ |

每一步都必然下降——这就是短序列对长序列有结构性偏好的根源。

**长度归一化评分 ($\alpha = 0.6$)：出现了反弹**

| Step | 累积 log-P | $t$ | $t^{0.6}$ | **归一化分数** | 趋势 |
|------|-----------|-----|-----------|----------------|------|
| 1 | -0.223 | 1 | 1.000 | **-0.223** | — |
| 2 | -1.427 | 2 | 1.516 | **-0.941** | ↓ |
| 3 | -2.120 | 3 | 1.933 | **-1.097** | ↓ |
| 4 | -2.631 | 4 | 2.297 | **-1.145** | ↓ |
| 5 | -2.988 | 5 | 2.627 | **-1.137** | **↑ 反弹！** |
| 6 | -3.904 | 6 | 2.930 | **-1.332** | ↓ |
| 7 | -4.415 | 7 | 3.214 | **-1.374** | ↓ |

Step 5 的归一化分数 $-1.137$ 比 Step 4 的 $-1.145$ **更高**，尽管累积 log-P 在继续下降。

#### 为什么 Step 5 反弹了？

Step 5 的 token `"the"` 概率高达 0.70（log-P = -0.357），质量很高。虽然它在分子上加了 -0.357 的负值，但分母从 $4^{0.6}=2.297$ 增长到 $5^{0.6}=2.627$（增长了 14%），**把之前 Step 2（log-P=-1.204 的"cat"）造成的历史负担一起稀释了**。

直觉类比：**考试成绩的均分**。假设你已考 4 门课，均分 65。第 5 门考了 85 —— 均分会从 65 上升到 69，尽管 85 < 100。归一化评分同理：新 token 的"质量"高于历史平均时，分数就会上升。

#### 上升的数学条件

归一化分数从 step $t$ 到 $t+1$ 上升的充要条件：

$$
\frac{s_t + \log P_{t+1}}{(t+1)^\alpha} > \frac{s_t}{t^\alpha}
\;\iff\;
\log P_{t+1} > s_t \cdot \frac{(t+1)^\alpha - t^\alpha}{t^\alpha}
$$

因为 $s_t < 0$（历史累积为负），不等号右边是一个**负数**。例如在我们的例子中，Step 4 → 5 的阈值：

$$
s_4 = -2.631,\quad \frac{5^{0.6} - 4^{0.6}}{4^{0.6}} = \frac{2.627 - 2.297}{2.297} = 0.144
$$

阈值 $= -2.631 \times 0.144 = -0.379$。Step 5 的 $\log P = -0.357 > -0.379$，满足了上升条件。

#### 三种场景的直观对比

| 场景 | 原始 log-P 累加 | 归一化分数 ($\alpha=0.6$) |
|------|----------------|--------------------------|
| 每步都是高质量 token（$\log P \approx -0.1$） | 缓慢线性下降 | 几乎平坦或**轻微上升** |
| 每步都是中等 token（$\log P \approx -0.5$） | 稳定线性下降 | 缓慢下降（比原始慢得多） |
| 某一步突然低质量（$\log P \approx -3.0$） | 突然跌落 | 突然跌落（且分母还来不及变大来稀释） |

**关键洞察**：归一化评分不是简单地"把分数抬高"，而是改变了分数随长度的**变化率**。高质量的生成不会因为长度而被惩罚，低质量的 token 仍然会被惩罚。这让模型在选择"长但高质"和"短但平庸"之间有了公平的权衡。

---

## 第 4 章：EOS 早停与 Beam 动态管理

### 4.1 EOS Token 的语义

EOS 表示"序列完成"。当一个 beam 生成了 EOS：
- 该 beam **停止展开**（已完成）
- 但**保留在候选池中**，参与最终排序
- 其他未完成的 beam 继续展开

### 4.1.1 EOS "退役 + 回收"机制（关键工程细节）

一个容易被忽略的细节：EOS 不消灭活跃席位，而是触发"退役 + 回收再利用"。

1. **退役**：命中 EOS 的 beam 被 `clone()` 存入 `finished_seqs`，永久保留
2. **让位**：下一步该 beam 的候选行被全设为 `-inf`，在 Top-B 剪枝中自动淘汰
3. **回收**：空出的席位被某条其他活跃 beam 的**第二个后代**填补——即那条活跃 beam 会"分叉"，同时产出两个入选候选

**结果**：
- 每一步**活跃 beam 数恒定 = B**（永不缩水）
- 但 `finished_seqs` 随 EOS 事件**持续增长**
- **最终比较池 = finished_seqs + 硬截断时仍在跑的活跃 beam，总数 ≥ B**

举例 B=3：
```
Step 2: beam[0] 命中 EOS → finished_seqs[0] 退役
        空出席位被 beam[1] 的第二后代填补（beam[1] 分叉）
Step 4: beam[2] 命中 EOS → finished_seqs[1] 退役
        空出席位被 beam[0] 的第二后代填补
Step 7: 硬截断，beam[0]、beam[1]、beam[2] 全部未 EOS

最终比较池: finished_seqs(2条) + 活跃(3条) = 5 条 > B=3
```

理论上界：每个 step 最多 B 条 beam 同时命中 EOS，因此 `finished_seqs` 长度上界为 `B × max_new_tokens`（极端情况每步全体退役再全体分叉）。实际中极少出现。

### 4.2 早停策略（Early Stopping）

定义以下早停条件（满足任一即停止整体生成）：

1. **全终止条件**：所有 $B$ 条 beam 都生成了 EOS
2. **分数优势条件**：在某个时刻，当前最优的已完成序列的分数超过了所有未完成 beam 可能达到的最高分数（考虑到未来 token 最多贡献的 log-prob 有限）
3. **硬截断条件**：达到 `max_new_tokens`（工程安全阀）

条件 2 的直觉：如果最优已完成序列的分数是 -2.5，而当前仍活跃的 beam 中最好的分数是 -3.1，且每步最多增加 $\log(1)=0$ 的分数（实际上多为负数），那么未完成 beam 无论如何也追不上已完成序列，提前终止就是安全的。

### 4.3 Beam 的动态缩减

当部分 beam 终止后，有效 beam width 自然减少：
- 实际 beam 数从 $B$ 降至 $B - k$（其中 $k$ 是已终止 beam 数）
- 展开候选数从 $B \times |V|$ 降至 $(B - k) \times |V|$
- 减少了多余计算

**工程注意**：如果直接让 beam 数动态变化，GPU batch size 会不稳定。实践中常在已终止 beam 的位置填充"哑 beam"（dummy beam），保持 batch size 固定。

### 4.4 最终序列的选取

当 beam search 结束后，对所有已完成的序列（hit EOS 的 beam）按归一化分数排序，选分数最高者作为输出。如果没有任何 beam 生成 EOS（如达到 max_tokens 硬截断），则在活跃 beam 中选择分数最高者。

---

## 第 5 章：Beam Search 与其他解码策略的对比

### 5.1 解码策略的分类法

所有解码策略可以从两个维度分类：

| 维度 | 选项 | 含义 |
|------|------|------|
| 搜索方式 | 确定性搜索 vs 随机采样 | 是否引入随机性 |
| 目标 | 最大化概率 vs 多样性 | 追求最优解还是合理多样性 |

### 5.2 与贪心搜索（Greedy）的对比

贪心搜索是 $B=1$ 的 beam search 退化形式。

| 维度 | Greedy | Beam Search |
|------|--------|-------------|
| 搜索宽度 | 1 | B (通常 3-20) |
| 全局最优性 | 无保证 | 近似最优（$B$ 越大越接近） |
| 计算量 | $O(T \cdot f)$ | $O(T \cdot B \cdot f)$ |
| 确定性 | 确定 | 确定（相同 $B$ 下） |
| 输出多样性 | 单一输出 | $B$ 条候选输出 |

**注意**：两者都是确定性的。给定相同的模型和输入，输出固定。这一点区别于所有采样方法。

### 5.3 与 Top-K / Top-P 采样的对比

这是 beam search 与其最主要替代方案的对比：

| 维度 | Beam Search | Top-K / Top-P Sampling |
|------|-------------|------------------------|
| 核心思想 | 搜索全局最优 | 在高质量候选内随机采样 |
| 确定性 | 确定 | 随机 |
| 适用任务 | 翻译、摘要、ASR | 开放式对话、创意写作 |
| 输出特点 | 安全、可能重复 | 多样、可能不连贯 |
| 计算量 | $O(B \cdot T)$ | $O(T)$ |
| 每步操作 | 展开 + 排序 + 剪枝 | 截断 + 重归一化 + 采样 |

**为什么 beam search 不适合开放式生成**（Holtzman et al., ICLR 2020）：
- 人类自然文本**不是**模型概率最高的文本（否则信息量为零）
- 逐词最大化概率导致安全、模板化、重复的输出
- Beam search 的 perplexity 极低（远低于人类），但这恰恰说明它"过于可预测"

### 5.4 与对比搜索（Contrastive Search）的关系

对比搜索（Su et al., 2022）的评分函数为：

$$
\text{score}(y_t) = (1 - \beta) \cdot \underbrace{P_\theta(y_t \mid y_{<t}, x)}_{\text{模型置信度}} - \beta \cdot \underbrace{\max_{j < t} \text{sim}(h_t, h_j)}_{\text{退化惩罚}}
$$

其中 $\text{sim}(h_t, h_j)$ 是当前隐藏状态与历史隐藏状态的相似度。

**与 beam search 的对比**：
- 对比搜索**不维护多条 beam**，而是用退化惩罚项替代 beam search 的全局搜索
- 对比搜索的计算量接近贪心搜索（$O(T)$），但质量接近 beam search
- 它瞄准的是 beam search 的退化/重复问题

### 5.5 适用场景总结

| 任务类型 | 推荐策略 | 原因 |
|----------|----------|------|
| 机器翻译 | Beam Search ($B=4-8$) | 输出空间受限，"高概率解"通常质量好 |
| 文本摘要 | Beam Search ($B=3-5$) | 需要平衡覆盖率与简洁性 |
| 语音识别 (ASR) | Beam Search | 低熵任务，确定性搜索合适 |
| 图像描述 (Captioning) | Beam Search | 描述空间较集中 |
| 开放式对话 | Top-P / Top-K Sampling | 高熵任务，需要多样性 |
| 故事/创意写作 | Top-P + Temperature | 追求新意和多样性 |
| 代码生成 | Greedy 或 Beam Search | 确定性要求高，格式需精确 |
| 数学推理 | Beam Search / MCTS | 多步推理需探索多路径 |

### 5.6 一个工程上的追问：Beam Search 能流式输出吗？

**直觉问题**：beam search 需要等所有 beam 都生成 EOS、比较归一化分数后，才能决定最终输出哪条序列——那岂不是意味着首字延迟（TTFT）非常高？商业大模型为什么不这样做？

**澄清**：beam search 的延迟问题不是 TTFT，而是**总生成时间 + 流式兼容性**。

#### Beam search 的内部特性：同步批处理

Beam search 的每一步，所有 $B$ 条活跃 beam 做一次**批处理前向传播**，同时生出各自的下一 token。所以：

- **TTFT 几乎不受影响**：第一步与贪心/采样一样，做一次前向得到 logits。Beam search 的额外开销仅是从 logits 中取 Top-B 个 token。
- **每步 latency 增加**：每步的 batch size 是 $B$（而非 1），KV cache 大小是 $B$ 倍。但 GPU 对 batch 有天然并行性，所以每步时间通常**不是** $B$ 倍，而是 sub-linear 增长。
- **总生成时间更长**：$B \times T$ 的总 token 生成量（部分 beam 提前 EOS 会减少一些）。

#### 流式输出的两难

Beam search 理论上**可以**流式输出——每步输出当前 Top-1 beam 的最新 token。但这引入了"输出回溯"问题：

```
Step 1-5:  Beam A 领先 → 用户看到 "The cat sits on the"
Step 6-12: Beam C 反超 → 输出变成 "A feline rests upon"  ← 用户已看到的内容作废了!
```

| 流式策略 | 做法 | 用户体验 |
|----------|------|----------|
| 即时流式（naive streaming） | 每步推 Top-1 beam 的 token | 可能回溯，体验灾难 |
| 保守流式（prefix-locked） | 只推所有 beam 共享前缀的 token | 安全但低效（beam 往往快速分化） |
| 非流式（标准做法） | 等全部完成，返回最优序列 | 总耗时长，但输出一致可靠 |
| 折中 | 返回 Top-B 条完整序列给用户选择 | 适用于翻译建议等多候选场景 |

**一个常见的误解**：像 Gemini 这类模型偶尔给出两个不同解答让用户选择，这是 beam search 吗？

**大概率不是。** 要区分两种"多候选输出"：

| 机制 | 输出特征 | 多样性 |
|------|----------|--------|
| Beam Search (B=2) 返回 Top-2 | 高度相似，通常共享大量前缀 | 低 |
| 两次独立采样（不同 random seed + Top-P）| 结构可能迥异 | 高 |

**具体例子**：

```
Beam Search B=2 典型输出:
  Beam 1: "猫是一种常见的家养动物，以捕鼠能力著称。"
  Beam 2: "猫是一种常见的家养动物，性格独立且爱干净。"
           ^^^^^^^^^^^^^^^^^^^^^^ 前缀完全相同，只有尾巴不同

独立采样 ×2 典型输出:
  Response 1: "猫是家养动物，捕鼠能手。"
  Response 2: "从进化角度看，猫科动物在约1000万年前..."
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 从第一句就分岔了
```

只有第二种才能给用户"两个不同视角"的选择价值。第一种用户会觉得"你在逗我吗"。**Gemini 的多候选几乎确定是独立采样，而非 beam search。**

但 beam search 的"返回 Top-B 条序列"确实服务于翻译场景——Google Translate 偶尔展示替代表达，那些是 beam search 的产物。正是因为翻译中 beam 的同质性反而是加分项（都是合理译法，仅措辞不同），而非减分项。

#### 商业大模型的实际选择

| 场景 | 解码策略 | 为什么 |
|------|----------|--------|
| ChatGPT / Claude 流式聊天 | Top-P + Temperature | 每步一个 token，生成即"最终"，无回溯 |
| Google Translate API | Beam Search（非流式） | 用户期望等完整翻译结果 |
| GitHub Copilot 补全 | Greedy 或轻量 Beam Search（非流式） | 代码补全不流式展示 |
| Whisper / ASR API | Beam Search（非流式） | 语音识别一次性返回全文 |

**核心结论**：Beam search 在流式场景中的根本矛盾在于——它的"最终答案"在最后一个 EOS 生成之前是不确定的。而采样策略每一步都"落子无悔"。因此，**商业大模型的流式模式几乎从不使用 beam search**。它专属服务于"生成整段结果再返回"的非流式任务（翻译、摘要、ASR），这些任务中用户本来就预期等待完整输出。

---

## 第 6 章：Beam Search 的高级变体

### 6.1 Diverse Beam Search (AAAI 2018)

**动机**：标准 beam search 的 $B$ 条输出高度相似——通常共享大部分前缀，计算预算被浪费在探索极相似的路径上。

**核心思想**：将 $B$ 条 beam 划分为 $G$ 组（每组 $B/G$ 条），在组间施加多样性惩罚：

$$
\text{score}(y^{(g)}, x) = \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x) - \lambda \cdot \underbrace{\text{sim}(y^{(g)}, y^{(<g)})}_{\text{与前面组的相似度}}
$$

其中 $\text{sim}(\cdot, \cdot)$ 通常使用 n-gram 重叠度或嵌入余弦相似度，$\lambda$ 控制多样性强度。

**直觉**：
- 第 1 组：标准 beam search（找"最优"路径）
- 第 2 组：在最优路径附近做惩罚，被迫找"不同但也好"的路径
- 第 3 组：同时被前两组惩罚，进一步分散

**效果**：
- 在图像描述中，DBS 的 Top-1 准确率**高于**标准 beam search（因为多样性帮助探索，偶尔发现更好的路径）
- 额外计算开销极小（仅需计算相似度惩罚项）

### 6.2 Stochastic Beam Search

**动机**：beam search 的确定性使其无法应对"多条路径概率接近"的歧义场景。

**两种主要变体**：

1. **Sample-based Beam Search**：每步展开时，不是取 Top-$B$，而是从候选分布中按概率**采样** $B$ 个。这结合了 beam search 的多分支结构与采样的随机性。

2. **Conditional Poisson Stochastic Beam Search**（Kool et al., 2019）：使用泊松过程的 Gumbel 技巧，从模型分布中做无偏采样，保证多样性的同时维持理论性质。

**适用场景**：需要在"多条合理答案"之间做探索的任务（如开放式问答、多模态生成）。

### 6.3 Constrained Beam Search

**动机**：许多应用需要输出满足硬约束（某个词必须出现、必须符合 JSON Schema、不能包含某些 token 等）。

**做法**：在 beam 展开步骤中，对不满足约束的 token 直接将其概率置零（$-\infty$ logit），确保被剪枝。

**典型应用**：
- **词表前缀树（Trie）约束**：强制输出必须是一个合法的词典词
- **JSON Schema 约束**：用有限状态自动机（FSM）定义合法状态转移，每一步只允许与当前 JSON 状态兼容的 token
- **禁止词表**：过滤特定敏感词或任务不相关词

Constrained Beam Search 是现代 LLM 推理引擎（vLLM、SGLang、llama.cpp）中"结构化输出"功能的理论基础。

### 6.4 Beam Search 在各模态中的应用

- **语音识别**：最早使用 beam search 的领域之一（Graves, 2012），将 CTC 输出的帧级概率通过 beam search 解码为字符序列
- **图像描述**：CNN 编码 + RNN/LM 解码，beam search 找到视觉上最准确的描述
- **OCR**：与 ASR 类似，是从视觉特征序列到字符序列的序列转录任务
- **代码生成**：Beam search 结合语法约束，探索多行代码的不同实现路径

---

## 第 7 章：理论性质与局限性

### 7.1 Beam Search 的近似比分析

**悲观结论**：Beam search **没有最坏情况下的近似保证**。它可以任意差地近似 $\arg\max$ 序列（Cotterell et al., TACL 2020）。

**为什么实践中还行？**
- 语言模型的概率分布高度集中在少数路径上（低熵）
- Beam search 恰好利用了语言模型的低熵性质
- Chatfield et al.（2010）证明：如果每一步条件分布足够"尖"（低熵），beam search 接近最优

**NP 困难性**：精确求解 $\arg\max_y P(y|x)$ 是 NP-hard 的（某些假设下）。因此 beam search 是实践中不可避免的"可接受近似"。

### 7.2 Beam Search 的 Uniform Information Density 解释

Meister et al.（EMNLP 2020, Best Paper Honorable Mention）提出了一个新颖的解释：

> Beam search 成功的原因不在于它近似了 $\arg\max$ 序列，而在于它隐式地施加了**均匀信息密度（Uniform Information Density, UID）**的认知科学原理。

UID 原理认为：人类在语言产出时，倾向于**将信息均匀分布**在话语中，避免某一点信息过载。Beam search 的剪枝机制恰好"削平"了那些信息密度极不均匀的序列（如某个 token 概率极低但后续极高），使输出更接近人类语言的统计特性。

**实验支持**：Beam search 输出的 UID 程度与 BLEU 分数呈正相关。而精确的 MAP 解码（最大化概率）往往产生 UID 极不均匀的序列，质量反而更差。

### 7.3 "Beam Search 诅咒"：退化与重复

#### 现象

**反直觉事实**：$B$ 从 5 增至 50，BLEU 分数可能**下降**而非上升（Koehn & Knowles, 2017；Cohen & Beck, ICML 2019）。

#### 退化机制的三层分析

**第一层：模型误差**
- 模型本身对退化序列分配了过高的概率（训练数据偏差 + teacher forcing 的曝光偏差）
- Stahlberg & Byrne（EMNLP 2019）的震撼发现：在 **超过 50% 的句子**中，空字符串是模型下的全局最优解

**第二层：搜索误差**
- 更大的 beam width 使得搜索更深入模型置信度的"错误区域"
- 早期低概率 token + 后续高概率 token 的组合（称为"搜索偏离"search discrepancy）钻了模型评分机制的空子

**第三层：正反馈循环**
- 一旦模型生成了重复短语，自注意力机制会强化该模式
- "the current state of the art ... the current state of the art ..." 的局部概率越来越高
- Beam search 作为"最大化概率"的搜索，被这个正反馈循环困住

#### 缓解手段

**Repetition Penalty（重复惩罚）**：
对已出现在序列中的 token 施加分数衰减：
$$
\text{logit}(y_t) \leftarrow \begin{cases}
\text{logit}(y_t) / \theta & \text{if } y_t \text{ 已在 } y_{<t} \text{ 中出现} \\
\text{logit}(y_t) & \text{otherwise}
\end{cases}
$$
其中 $\theta \in (1.0, 1.5)$ 是惩罚强度。$\theta = 1.0$ 无惩罚，$\theta = 1.2$ 典型经验值。这不会完全禁止重复，而是让模型天然更有动力选择新 token。

**N-gram Blocking（N-gram 阻塞）**：
硬约束版本：如果某个 n-gram 已经出现过，直接将对应 token 的概率置零。简单粗暴但可能误伤合法重复（如诗歌的叠句、技术文档的术语重复）。

**两者对比**：
| 机制 | Repetition Penalty | N-gram Blocking |
|------|-------------------|-----------------|
| 类型 | 软约束（惩罚） | 硬约束（禁止） |
| 允许合理重复 | 是（仅降低概率） | 否（完全禁止） |
| 超参数 | $\theta$（连续） | n（离散，通常 n=3 或 4） |
| 使用频率 | 更常见 | 较少见 |

### 7.4 与 MCTS（蒙特卡洛树搜索）的对比

MCTS 是另一种树搜索方法，在 AlphaGo 中成名，近年被用于 LLM 推理（如 AlphaMath）。

| 维度 | Beam Search | MCTS |
|------|-------------|------|
| 搜索策略 | 确定性贪心展开 | 随机模拟 + 树策略引导 |
| 剪枝依据 | 立即评分（模型概率） | 模拟回报（rollout 结果） |
| 评估方式 | 单步 log-prob | 多步蒙特卡洛模拟 |
| 探索-利用 | 纯利用（取 Top-B） | UCB 平衡探索与利用 |
| 计算量 | $O(B \cdot T \cdot f)$ | $O(N \cdot T \cdot f)$，$N$ 为模拟次数 |
| 适用场景 | 翻译、摘要（低熵搜索） | 数学推理、棋类（需长程规划） |

**核心区别**：Beam search 用**单步模型概率**决定保留哪些路径（短视），MCTS 用**多步模拟的最终回报**来评估（远见）。在数学推理、代码生成等需要多步规划的任务中，MCTS 越来越受关注。

---

## 第 8 章：关键公式速查

### Beam Search 评分（原始）
$$
s(y) = \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x)
$$

### 长度归一化评分（GNMT）
$$
s_{\text{norm}}(y) = \frac{s(y)}{(5 + |y|)^\alpha} \quad \text{或简化为} \quad s_{\text{norm}}(y) = \frac{s(y)}{|y|^\alpha}
$$

### 每步展开与剪枝
$$
\text{beams}_{t} = \text{Top-}B\left( \bigcup_{i=1}^{B} \left\{ \left(\text{beam}_i \oplus v,\ \text{score}(\text{beam}_i) + \log P(v \mid \text{beam}_i) \right) \mid v \in V \right\} \right)
$$

### Diverse Beam Search 的多样性惩罚
$$
s_{\text{diverse}}(y^{(g)}) = s(y^{(g)}) - \lambda \sum_{h < g} \Delta(y^{(g)}, y^{(h)})
$$

---

## 附录 A：关键论文索引

按时间线排列：

| 年份 | 论文 | 核心贡献 |
|------|------|----------|
| 2006 | Graves et al., *Connectionist Temporal Classification*, ICML | 首次将 beam search 用于序列转录（ASR/OCR） |
| 2014 | Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS | 将 beam search 确立为 seq2seq 的标准解码方法 |
| 2016 | Wu et al., *Google's Neural Machine Translation System* (GNMT), arXiv:1609.08144 | **长度归一化** + coverage penalty + pruning 策略，定义了 NMT beam search 的工业标准 |
| 2018 | Vijayakumar et al., *Diverse Beam Search*, AAAI | 引入组间多样性惩罚，解决 beam 同质化问题 |
| 2019 | Cohen & Beck, *Empirical Analysis of Beam Search Performance Degradation*, ICML | 系统分析了 beam width 增大导致质量下降的机制（"搜索偏离"理论） |
| 2019 | Stahlberg & Byrne, *On NMT Search Errors and Model Errors*, EMNLP | 揭示空字符串常是模型全局最优的惊人结果 |
| 2020 | Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR | 证明 beam search 在开放式生成中不如 nucleus sampling；解释退化机制 |
| 2020 | Meister et al., *If Beam Search is the Answer, What Was the Question?*, EMNLP | Uniform Information Density 理论：beam search 的成功源于隐式施加 UID |
| 2022 | Su et al., *Contrastive Search Is What You Need For Neural Text Generation* | 提出对比搜索作为 beam search 的轻量替代 |

## 附录 B：术语对照表

| 英文 | 中文 | 含义 |
|------|------|------|
| Beam Search | 束搜索 | 维护 B 条最优候选路径的近似搜索算法 |
| Beam Width | 束宽度 | 同时维护的候选序列数量 B |
| Hypothesis | 假设/候选 | 一条正在扩展的部分序列 |
| Candidate | 候选 | 展开后的候选序列（B × V 个） |
| Length Penalty | 长度惩罚 | 对长序列的归一化系数 α |
| Length Normalization | 长度归一化 | $s/|y|^\alpha$，防止短序列霸榜 |
| EOS (End-of-Sequence) | 序列结束符 | 标记序列完成的特殊 token |
| Early Stopping | 早停 | 满足条件时提前终止搜索 |
| Search Discrepancy | 搜索偏离 | 相对于贪心路径的早期低概率 token 选择 |
| Degeneration | 退化 | 高 beam width 产生重复/低质输出的现象 |
| Repetition Penalty | 重复惩罚 | 对已出现 token 施加分数衰减 |
| N-gram Blocking | N-gram 阻塞 | 禁止已生成的 n-gram 再次出现 |
| Coverage Penalty | 覆盖惩罚 | encoder-decoder 架构中惩罚欠翻译的机制 |
| Pruning | 剪枝 | 从候选池中丢弃低分序列 |
| Diverse Beam Search | 多样性束搜索 | 组间加相似度惩罚的多组 beam search |

---

> **下一步**：基于本笔记的理论基础，阅读 [`beam_search.py`]() 的手撕实现。工程细节（最小堆调度、KV Cache 独立管理与裁剪、与 HF `model.generate()` 对齐）将在代码中以中文注释形式展开。

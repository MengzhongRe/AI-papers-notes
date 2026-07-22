# 交叉熵损失的严格学术推导：从 Shannon 公理到 LLM 预训练目标

> **定位**：系统性学术笔记，从第一性原理出发，构建交叉熵损失的完整知识体系。与 [`ce_loss_notes.md`](ce_loss_notes.md)（Q&A 风格）和 [`ce_loss_engineering.md`](ce_loss_engineering.md)（工程视角）互补。
>
> **核心命题**：回答「为什么 LLM 必须用交叉熵损失？」——这个问题需要从信息论、统计学、决策论三个维度，追溯到一个比 MLE 和 KL 散度都更底层的统一框架：**Proper Scoring Rule（恰当评分规则）**。

---

## 目录

- [Part 0 — 符号体系与前置定义](#part-0--符号体系与前置定义)
- [Part I — 信息论基石：从"惊讶"到"交叉熵"](#part-i--信息论基石从惊讶到交叉熵)
  - [§1.1 信息量：Shannon 的三大公理](#11-信息量shannon-的三大公理)
  - [§1.2 熵：平均惊讶度与编码下界](#12-熵平均惊讶度与编码下界)
  - [§1.3 KL 散度：分布间的"额外代价"](#13-kl-散度分布间的额外代价)
  - [§1.4 交叉熵的严格定义与核心恒等式](#14-交叉熵的严格定义与核心恒等式)
- [Part II — 统计学视角：MLE ⟺ 交叉熵](#part-ii--统计学视角mle--交叉熵)
  - [§2.1 极大似然估计的严格定义](#21-极大似然估计的严格定义)
  - [§2.2 经验分布视角下的等价性证明](#22-经验分布视角下的等价性证明)
- [Part III — 统合视角：哪个解释更本质？](#part-iii--统合视角哪个解释更本质)
  - [§3.1 两条路径的互推](#31-两条路径的互推)
  - [§3.2 Proper Scoring Rule：更底层的统一框架](#32-proper-scoring-rule更底层的统一框架)
  - [§3.3 最小描述长度原则](#33-最小描述长度原则)
- [Part IV — LLM 预训练的形式化数学目标](#part-iv--llm-预训练的形式化数学目标)
  - [§4.1 自回归语言建模的概率图模型](#41-自回归语言建模的概率图模型)
  - [§4.2 预训练损失函数的完整形式化](#42-预训练损失函数的完整形式化)
  - [§4.3 为什么这是一个合理的目标？](#43-为什么这是一个合理的目标)
- [Part V — Softmax + Cross-Entropy 的梯度魔法](#part-v--softmax--cross-entropy-的梯度魔法)
  - [§5.1 预备：Softmax 函数的 Jacobian 矩阵](#51-预备softmax-函数的-jacobian-矩阵)
  - [§5.2 Cross-Entropy + Softmax 合体求导](#52-cross-entropy--softmax-合体求导)
  - [§5.3 梯度下降的力学直觉：拉力、推力与零和博弈](#53-梯度下降的力学直觉拉力推力与零和博弈)
  - [§5.4 对比：MSE + Softmax 为什么失败](#54-对比mse--softmax-为什么失败)
  - [§5.5 从 Fisher 信息矩阵看交叉熵](#55-从-fisher-信息矩阵看交叉熵)
- [Part VI — 数值计算：LogSumExp 的数学与工程](#part-vi--数值计算logsumexp-的数学与工程)
  - [§6.1 朴素实现的溢出分析](#61-朴素实现的溢出分析)
  - [§6.2 LogSumExp 技巧的严格推导](#62-logsumexp-技巧的严格推导)
  - [§6.3 Safe Softmax 与 LogSumExp 的本质区别](#63-safe-softmax-与-logsumexp-的本质区别)
  - [§6.4 混合精度的失效边界](#64-混合精度的失效边界)
- [Part VII — 从损失到评价：困惑度与信息密度](#part-vii--从损失到评价困惑度与信息密度)
  - [§7.1 困惑度（Perplexity）](#71-困惑度perplexity)
  - [§7.2 交叉熵下界与语言的信息密度](#72-交叉熵下界与语言的信息密度)
- [附录 A：Proper Scoring Rule 的公理化推导](#附录-aproper-scoring-rule-的公理化推导)
- [附录 B：Label Smoothing 的正则化解释](#附录-blabel-smoothing-的正则化解释)
- [附录 C：温度参数化与知识蒸馏](#附录-c温度参数化与知识蒸馏)

---

## Part 0 — 符号体系与前置定义

为避免后文符号混乱，这里统一约定。以下约定除非特别说明，在全篇范围内一致使用。

**概率空间**：考虑有限离散样本空间 $\mathcal{X} = \{1, 2, \dots, V\}$，代表词表（vocabulary）。连续情况可通过 Lebesgue 积分类比，但语言模型本质是离散分布，故本文全程使用离散形式的求和。

| 符号 | 含义 |
|:-----|:-----|
| $P$ | 真实数据分布（data-generating distribution），通常不可知 |
| $\hat{P}$ | 经验分布（empirical distribution），从有限样本 $D$ 估计 |
| $Q$ 或 $P_\theta$ | 模型分布（model distribution），由参数 $\theta$ 参数化 |
| $\mathbf{y}$ | 真实标签的 one-hot 向量，$\mathbf{y} \in \{0, 1\}^V$，$\sum_j y_j = 1$ |
| $\mathbf{a}$ | 模型输出的概率向量（Softmax 归一化后），$\mathbf{a} \in (0, 1)^V$，$\sum_j a_j = 1$ |
| $\mathbf{z}$ | 模型输出的原始 logits（未经 Softmax），$\mathbf{z} \in \mathbb{R}^V$ |
| $c$ | 目标类别索引，$y_c = 1$，$y_{j \neq c} = 0$ |

**对数底数约定**：本文使用自然对数 $\ln$ (base $e$)，对应单位为 **nats**。在信息论中若使用 $\log_2$ 则单位为 **bits**。两者关系：$\log_2 x = \ln x / \ln 2$。在 LLM 文献和 PyTorch 源码中，交叉熵默认使用自然对数，本文遵循此惯例。困惑度（PPL）使用 $\exp(\cdot)$ 直接还原，对应 nats 的单位。

**期望记号**：$\mathbb{E}_{x \sim P}[f(x)] = \sum_{x \in \mathcal{X}} P(x) f(x)$。

---

## Part I — 信息论基石：从"惊讶"到"交叉熵"

交叉熵不是凭空定义的。它的逻辑起点是 Shannon (1948) 对 **信息量** 的公理化构造。我们按「信息量 → 熵 → KL 散度 → 交叉熵」的顺序逐级构建。

### 1.1 信息量：Shannon 的三大公理

**问题**：观测到事件 $x$ 发生，我们获得了多少「信息」？

Shannon 的洞见是：信息量只应依赖于事件发生的概率。设 $I: (0, 1] \to [0, +\infty)$ 为将概率映射为信息量的函数。

**公理 1（连续性）**：$I(p)$ 关于 $p$ 连续。概率的微小变化不应导致信息量的剧烈跳变。

**公理 2（单调性）**：若 $p_1 < p_2$，则 $I(p_1) > I(p_2)$。越不可能发生的事件，一旦发生，携带的信息量越大。

**公理 3（独立事件的可加性）**：若事件 $A$ 与 $B$ 独立，则 $P(A \cap B) = P(A)P(B)$，应满足：
$$I(P(A)P(B)) = I(P(A)) + I(P(B))$$

即两个独立事件同时发生的信息量，等于各自信息量之和。

**定理（Shannon, 1948）**：在满足公理 1-3 的条件下，$I(p)$ 必然具有形式：
$$I(p) = -k \log p$$

其中 $k > 0$ 是任意正比例常数。选取 $k = 1$ 和自然对数，我们得到标准定义：
$$\boxed{I(x) = -\ln p(x)}$$

**直觉**：$-\ln p(x)$ 度量了「观测到 $x$ 时，我们有多惊讶」（surprisal）。$p=1$（必然发生）→ 惊讶度为 $0$；$p \to 0$（极不可能）→ 惊讶度 $\to +\infty$。

**为什么必须是对数？——三条公理的深层含义**

Shannon 的三条公理不是凭空捏造的。它们分别对应了人类对「信息」的三个基本直觉，而这三个直觉一同**强迫**出对数形式：

- **公理 1（连续性）** 排除了一切「跳变」的可能性。如果概率从 $0.500$ 变成 $0.501$，信息量不应该从 $10$ 跳到 $0.3$。这保证了 $I(p)$ 是一个「光滑」的函数，可以用微积分处理。

- **公理 2（单调性）** 确立了信息的「方向」：越不可能 = 越令人惊讶 = 越多信息。这是一个定性的约束——但仅凭前两条公理，$I(p) = -\ln p$、$I(p) = 1/p - 1$、$I(p) = -\sqrt{p}$ 都满足（均为 $p$ 的连续且递减的函数）。

- **公理 3（独立事件可加性）** 是真正的「杀手锏」——它把信息变成了一个**可累加的量**。两个人各自独立地掷硬币，你同时得知两枚硬币的结果，获得的总信息量应该等于分别得知每个结果的信息量之和。这个看似无害的要求，在数学上等价于 Cauchy 函数方程 $f(xy) = f(x) + f(y)$，其唯一连续解就是对数函数。

**关键洞察**：公理 3 深刻地塑造了我们对「信息」的理解——它暗示信息是一个**广延量（extensive quantity）**，像质量、能量一样，独立来源的信息可以直接相加。如果你不接受公理 3（比如你认为两件独立事件同时发生应该带来「指数级」而非「加法级」的额外惊讶），那么你对信息的度量就不再是 Shannon 信息，而是另一个理论体系。但在所有满足这三条公理的理论中，对数形式是唯一的。这也是为什么 Shannon 称自己的理论为「A Mathematical Theory of Communication」而非「A Heuristic Model」——它的基础是公理化的，不是经验拟合的。

### 1.2 熵：平均惊讶度与编码下界

**定义（Shannon 熵）**：随机变量 $X \sim P$ 的熵定义为信息量的期望：
$$\boxed{H(P) \equiv \mathbb{E}_{x \sim P}\big[-\ln P(x)\big] = -\sum_{x \in \mathcal{X}} P(x) \ln P(x)}$$

其中约定 $0 \ln 0 = 0$（由 $\lim_{p \to 0^+} p \ln p = 0$ 保证连续性）。

**性质**：

1. **非负性**：$H(P) \geq 0$，等号当且仅当 $P$ 是退化分布（所有概率质量集中于一点）。
2. **上界**：$H(P) \leq \ln |\mathcal{X}|$，等号当且仅当 $P$ 是均匀分布。这由 Jensen 不等式直接推出：
   $$H(P) = -\sum P(x) \ln P(x) = \sum P(x) \ln \frac{1}{P(x)} \leq \ln \sum P(x) \cdot \frac{1}{P(x)} = \ln |\mathcal{X}|$$
3. **凹性**：$H(\lambda P_1 + (1-\lambda)P_2) \geq \lambda H(P_1) + (1-\lambda) H(P_2)$。混合增加不确定性。

**信息论的核心解释**：$H(P)$ 是**对来自分布 $P$ 的随机符号进行最优无损编码时，每个符号所需的平均比特数下界**（Shannon 信源编码定理）。这是为什么 $-\ln P(x)$ 可以理解为「编码 $x$ 所需的最短码长」—— 信息论将「信息」与「编码长度」等价。

**直觉展开——从「惊讶度」到「熵」的认知飞跃**

信息量 $I(x) = -\ln p(x)$ 衡量的是**单个事件**的惊讶程度。但一个随机变量的「整体不确定性」不能只看某一个事件——它是对所有可能事件的惊讶度的加权平均。这就是熵。

用掷骰子的例子建立直觉：

| 骰子类型 | 分布 | 熵 | 直觉 |
|:---------|:-----|:---|:-----|
| 作弊骰子（总是 6） | $P(6)=1$，其余 $=0$ | $H=0$ | 「零不确定性」——每次都知道会出什么 |
| 公平骰子（6 面） | 均匀分布 | $H=\ln 6 \approx 1.79$ nats | 「最大不确定性」——每次都在 6 个等可能结果中猜测 |
| 偏斜骰子 | 不均匀但非退化 | $0 < H < \ln 6$ | 「部分可预测」——有规律但不完全确定 |

这个例子揭示了熵的深层含义：**熵度量的是「在观测到实际结果之前，你平均有多不确定」**。等价地，它也度量了「观测到结果之后，你平均获得了多少信息」。

**熵与编码的桥梁**：为什么 $H(P)$ 同时也是最优码长？直觉如下：
- 对于概率为 $p(x)$ 的事件，如果我们给它分配一个长度为 $-\log_2 p(x)$ 比特的二进制编码（这是 Shannon-Fano 编码的核心思想），那么频繁发生的事件（高 $p$）获得短编码，罕见事件（低 $p$）获得长编码——这最小化了平均编码长度
- 平均编码长度 = $\sum p(x) \cdot (-\log_2 p(x)) = H(P)$ 比特
- 任何其他编码方案的平均码长都不可能低于这个值（信源编码定理），否则就会产生歧义（无法无损还原）

这解释了为什么「信息」和「编码长度」是同一个概念的两个侧面：信息是你**学到了什么**（消除的不确定性），编码长度是你**为存储或传输它付出了多少比特**。两者由同一个 $-\ln p$ 连接。

### 1.3 KL 散度：分布间的"额外代价"

**问题**：如果我们用分布 $Q$ 去编码实际来自分布 $P$ 的符号，比用 $P$ 自身编码，多付了多少代价？

**定义（KL 散度）**：
$$\boxed{D_{KL}(P \parallel Q) \equiv \sum_{x \in \mathcal{X}} P(x) \ln \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\ln \frac{P(x)}{Q(x)}\right]}$$

**核心性质 — Gibbs 不等式**：$D_{KL}(P \parallel Q) \geq 0$，等号当且仅当 $P = Q$（几乎处处）。

*证明*（使用 $\ln t \leq t - 1$，等号当且仅当 $t = 1$）：
$$\begin{aligned}
-D_{KL}(P \parallel Q) &= \sum_x P(x) \ln \frac{Q(x)}{P(x)} \\
&\leq \sum_x P(x) \left(\frac{Q(x)}{P(x)} - 1\right) \\
&= \sum_x Q(x) - \sum_x P(x) = 0
\end{aligned}$$

**关键理解**：KL 散度不是真正的「距离」（不满足对称性和三角不等式），但它度量了**用 $Q$ 替代 $P$ 时的信息损失**。注意不对称性：$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$ —— 前者惩罚 $Q$ 在 $P$ 有质量的地方太低，后者惩罚 $Q$ 在 $P$ 无质量的地方太高。

**直觉展开——不对称性的本质含义**

KL 散度的不对称性不是数学上的瑕疵，而是它有实际含义的体现。考虑一个具体的语言模型场景：

- $P$ = 真实的人类语言分布（「the」之后接「cat」的概率是 0.01）
- $Q$ = 你训练的模型分布（「the」之后接「cat」的概率是 $10^{-6}$）

$D_{KL}(P \parallel Q)$ 的含义是：**用模型 $Q$ 去压缩真实文本时，平均每个单词多浪费的比特数**。在这个方向下，$P$ 是「权威」，$Q$ 是被评估的对象。如果模型给某个真实常见的搭配（「the cat」）赋予了极低的概率，那么每当「the cat」真的出现时，模型需要极长的编码来描述它，KL 散度爆炸。

反过来，$D_{KL}(Q \parallel P)$ 的含义是：用真实分布 $P$ 来压缩模型 $Q$ 生成的文本。这关注的是「模型生成了什么而真实语言中几乎不存在」——即模型胡编乱造的内容。

**为什么 LLM 训练选 $D_{KL}(P \parallel Q)$ 而非 $D_{KL}(Q \parallel P)$？**

- $D_{KL}(P \parallel Q)$（等价于 CE）惩罚**遗漏**：真实语言中出现的搭配，模型必须给一定的概率。这对应了语言模型的**召回率**——不能遗漏人类会说的内容。
- $D_{KL}(Q \parallel P)$ 惩罚**虚报**：模型自己发明的搭配，真实语言中必须有。这对应了**精确率**——不能生成人类不会说的内容。

在预训练阶段，模型的容量远超训练数据的覆盖范围，最大的风险是模型「学不会」人类语言的所有模式（遗漏），而非「学太多」（虚报）。因此 $D_{KL}(P \parallel Q)$ 是自然的选择。但在 RLHF 的对齐阶段，$D_{KL}(Q \parallel P_{\text{ref}})$ 恰恰被用作约束项——防止模型为了讨好奖励模型而「胡编乱造」。

### 1.4 交叉熵的严格定义与核心恒等式

**定义（交叉熵）**：
$$\boxed{H(P, Q) \equiv -\sum_{x \in \mathcal{X}} P(x) \ln Q(x) = \mathbb{E}_{x \sim P}\big[-\ln Q(x)\big]}$$

**核心恒等式**（直接展开 $D_{KL}$ 的定义即可得）：
$$\begin{aligned}
D_{KL}(P \parallel Q) &= \sum_x P(x) \ln P(x) - \sum_x P(x) \ln Q(x) \\
&= -H(P) + H(P, Q)
\end{aligned}$$

整理得：
$$\boxed{H(P, Q) \equiv H(P) + D_{KL}(P \parallel Q)}$$

**这是理解交叉熵的最核心公式。** 它告诉我们：

- $H(P, Q)$ = 真实分布自身的不确定性 ($H(P)$) + 由模型近似的误差带来的额外不确定性 ($D_{KL}$)
- 由于 $H(P)$ 与模型参数 $\theta$ 无关，$$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta D_{KL}(P \parallel Q_\theta)$$
- **最小化交叉熵 $\equiv$ 最小化 KL 散度**，即让模型分布尽可能逼近真实分布

**直觉展开——一个具体的数值例子**

这个恒等式 $H(P, Q) = H(P) + D_{KL}(P \parallel Q)$ 极其深刻地解释了交叉熵的两个组成部分。让我们用一个例子感受它：

假设英语中「the」之后下一个词的**真实分布** $P$ 为：
- 「cat」: 0.08, 「dog」: 0.05, 「car」: 0.03, 「man」: 0.02, …（共 $V=50{,}000$ 个词）

那么 $H(P) \approx 6.5$ nats —— 这是语言本身固有的不确定性。即使你有一个完美模型 $Q = P$，交叉熵也至少是 6.5 nats。这是**不可约误差（irreducible error）**——是语言本身的熵。

现在假设你的模型 $Q$ 是一个不太好的近似。$D_{KL}(P \parallel Q) = 0.8$ nats 意味着模型在每个 token 上额外付出了 0.8 nats 的编码代价。
$$H(P, Q) = 6.5 + 0.8 = 7.3 \text{ nats}$$

如果模型改进，$D_{KL}$ 降至 0.3：
$$H(P, Q) = 6.5 + 0.3 = 6.8 \text{ nats}$$

如果模型是完美的（$Q = P$）：
$$H(P, Q) = 6.5 + 0 = 6.5 \text{ nats}$$

**关键洞察**：交叉熵把「语言有多难预测」（$H(P)$，固定）和「模型有多差」（$D_{KL}$，可优化）拆成了两个独立的可加项。训练过程中交叉熵的下降，100% 来自 $D_{KL}$ 项的减小。这解释了为什么不同语言/数据集的交叉熵下界不同（中文的下界可能高于英文，因为中文字符的熵更大），但无论如何优化，都无法突破 $H(P)$ 的下界。

---

## Part II — 统计学视角：MLE ⟺ 交叉熵

### 2.1 极大似然估计的严格定义

**设置**：我们有独立同分布 (i.i.d.) 样本 $D = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$，来自未知的真实分布 $P_{\text{data}}$。我们有一族参数化的概率模型 $\{P_\theta : \theta \in \Theta\}$。

**似然函数**：在参数 $\theta$ 下观测到数据集 $D$ 的联合概率：
$$L(\theta; D) = \prod_{i=1}^{N} P_\theta(x^{(i)})$$

**极大似然估计（MLE）** 选择使似然函数最大的参数：
$$\theta^*_{\text{MLE}} = \arg\max_\theta L(\theta; D)$$

**对数似然**：由于对数函数严格单调，最大化似然等价于最大化对数似然：
$$\theta^*_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^{N} \ln P_\theta(x^{(i)})$$

**为什么要取对数？**
1. **数值稳定性**：$N$ 个 $(0,1)$ 之间的数连乘 → 极快下溢为 0。对数将乘法变为加法。
2. **凸性保持**：对数似然常为凸函数，利于优化。
3. **与信息论的深层联系**：$\frac{1}{N}\sum \ln P_\theta(x^{(i)})$ 收敛到 $\mathbb{E}_{x \sim P_{\text{data}}}[\ln P_\theta(x)]$（大数定律）。

**直觉——MLE 到底在做什么？**

MLE 的哲学可以用一句话概括：**「选择一个参数 $\theta$，使得你实际观测到的数据，在这个参数下看起来『最不令人惊讶』。」**

想象你是一个侦探，在现场发现了三枚指纹。你有两个嫌疑人：
- 嫌疑人 A 的指纹出现概率：拇指 0.2%，食指 0.5%，中指 0.1%
- 嫌疑人 B 的指纹出现概率：拇指 0.01%，食指 0.02%，中指 0.005%

MLE 会选择嫌疑人 A——因为在你实际观测到的数据（这些指纹）下，A 的指纹模式使这些观测「看起来更合理」（有更高的联合概率）。这就是 MLE 的朴素直觉：**让已经发生的事情看起来是最可能发生的。**

但注意——MLE 不告诉你「A 是凶手的概率是多少」，它只是在 A 和 B 之间选择了那个使观测数据似然更大的人。这是频率学派和贝叶斯学派的核心分歧，但就参数估计而言，MLE 的性质已经足够优秀了。

**等价形式**：最小化**负对数似然（Negative Log-Likelihood, NLL）**：
$$\theta^*_{\text{MLE}} = \arg\min_\theta -\frac{1}{N} \sum_{i=1}^{N} \ln P_\theta(x^{(i)})$$

### 2.2 经验分布视角下的等价性证明

**经验分布**：从有限样本 $D$ 定义：
$$\hat{P}(x) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[x^{(i)} = x] = \frac{\text{count}(x)}{N}$$

这是对真实分布 $P_{\text{data}}$ 的非参数估计（直方图）。

**直觉——经验分布是 MLE 与交叉熵之间的「翻译官」**

这里的等价性证明揭示了一个重要的概念桥梁：**$\hat{P}$ 是将「有限的训练样本」翻译成「概率分布语言」的唯一自然方式。** 一旦我们接受经验分布作为对真实数据分布的估计，那么：

- MLE 说：选择 $\theta$ 使得样本的对数似然最大
- 信息论说：选择 $\theta$ 使得模型分布与经验分布的交叉熵最小

而数学告诉我们：这两个说法是**逐字逐句等价**的——它们在算术上是同一个表达式。这不是巧合，这是同一个优化问题的两种语言表述。MLE 用「概率」的语言说话，交叉熵用「信息」的语言说话，但经验分布 $\hat{P}$ 是连接这两种语言的词典。

**核心推导**：计算经验分布 $\hat{P}$ 与模型分布 $P_\theta$ 之间的交叉熵：
$$\begin{aligned}
H(\hat{P}, P_\theta) &= -\sum_{x \in \mathcal{X}} \hat{P}(x) \ln P_\theta(x) \\
&= -\sum_{x \in \mathcal{X}} \left(\frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[x^{(i)} = x]\right) \ln P_\theta(x) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \sum_{x \in \mathcal{X}} \mathbf{1}[x^{(i)} = x] \ln P_\theta(x) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \ln P_\theta(x^{(i)})
\end{aligned}$$

最后一步是因为内层求和中，只有当 $x$ 恰好等于 $x^{(i)}$ 时指示函数才为 $1$。

**结论**：
$$\boxed{H(\hat{P}, P_\theta) = -\frac{1}{N} \sum_{i=1}^{N} \ln P_\theta(x^{(i)}) = \text{NLL}(\theta)}$$

**最小化交叉熵（经验分布 vs 模型分布）严格等价于极大似然估计。** 这一等价性在离散和连续情形下都成立（对称嫡，参见 Akaike, 1973）。

---

## Part III — 统合视角：哪个解释更本质？

前面两个 Part 分别从信息论（交叉熵 = 编码代价）和统计学（MLE = 频率学派推断）给出解释。本节回答核心问题：**到底哪种解释是"第一性原理"？**

### 3.1 两条路径的互推

**从信息论到 MLE**：
$$D_{KL}(P_{\text{data}} \parallel P_\theta) = -H(P_{\text{data}}) + H(P_{\text{data}}, P_\theta)$$
由于 $H(P_{\text{data}})$ 与 $\theta$ 无关，$\arg\min_\theta H(P_{\text{data}}, P_\theta) = \arg\min_\theta D_{KL}(P_{\text{data}} \parallel P_\theta)$。
在有限样本上，用 $\hat{P}$ 替代 $P_{\text{data}}$（插件原理, plug-in principle），得到 $\arg\min_\theta H(\hat{P}, P_\theta)$，这恰好是 MLE。

**从 MLE 到信息论**：
$$\begin{aligned}
\theta^*_{\text{MLE}} &= \arg\max_\theta \frac{1}{N} \sum_i \ln P_\theta(x^{(i)}) \\
&= \arg\min_\theta -\frac{1}{N} \sum_i \ln P_\theta(x^{(i)}) \\
&= \arg\min_\theta H(\hat{P}, P_\theta)
\end{aligned}$$
当 $N \to \infty$，由强大数定律，$H(\hat{P}, P_\theta) \xrightarrow{\text{a.s.}} H(P_{\text{data}}, P_\theta)$。

**结论**：两种解释是同一个硬币的两面，互相蕴含。但它们都依赖于一个共同的前提——**「对数评分」本身是合理的**。这个前提的合法性由什么保证？

### 3.2 Proper Scoring Rule：更底层的统一框架

**评分规则（Scoring Rule）** 是比交叉熵更底层的概念。它来自决策论（decision theory），用于评估概率预测的质量。

**直觉——为什么要从「评分规则」的角度重新理解一切？**

在进入形式化定义之前，先理解这个概念为什么重要。考虑一个天气预报员的场景：

天气预报员每天报告：「明天下雨的概率是 30%」。作为评估者，你如何判断这个预报员是不是在胡说八道？你不能等明天看「下雨了」就说他错了——因为他说的不是「明天一定会下雨或一定不下雨」，而是「30%」。单一事件的结果无法评判一个概率预测的质量。

评分规则正是为了解决这个问题：**给定一个概率预测 $Q$ 和实际观测到的结果 $x$，给预测打一个分数（惩罚）。** 关键要求是——这个评分机制应该鼓励预报员诚实。如果预报员真的相信下雨概率是 30%，那么他报告「30%」应该比胡乱报告「80%」或「5%」获得更好的期望分数。

这就是「恰当性」（properness）的朴素含义：**评分机制本身不应该有「偏好」——它不应该让预报员觉得「虽然我真心觉得是 30%，但我报告 50% 能拿更高分」。**

**定义 1（评分规则）**：一个评分规则是一个函数 $S: \Delta_V \times \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$，其中 $\Delta_V$ 是 $V$ 类别的概率单纯形（所有可能概率分布的集合）。$S(Q, x)$ 表示：当你预测分布为 $Q$，而实际观测到 $x$ 时，你受到的「惩罚」（或负奖励）。

**定义 2（恰当评分规则, Proper Scoring Rule）**：$S$ 称为恰当的，如果对任意真实分布 $P \in \Delta_V$ 和任意预测分布 $Q \in \Delta_V$：
$$\mathbb{E}_{x \sim P}\big[S(P, x)\big] \leq \mathbb{E}_{x \sim P}\big[S(Q, x)\big]$$

即：**说真话永远是最优策略**。当你诚实报告你的真实信念 $P$ 时，你获得的期望惩罚最小。

$S$ 称为**严格恰当 (Strictly Proper)**，如果等号成立当且仅当 $Q = P$——即说真话是唯一最优策略。

**定义 3（局部评分规则, Local Scoring Rule）**：$S$ 是局部的，如果它在评估事件 $x$ 的预测质量时只依赖于 $Q(x)$（即只依赖于分配给实际发生事件的概率），而不依赖于 $Q$ 在其他未发生事件上的分配：
$$S(Q, x) = s(Q(x))$$
其中 $s: (0, 1] \to \mathbb{R}$。

**直觉——「局部性」为什么如此重要？**

局部性的要求看似技术性，实则极其深刻。它说：**当你事后评估「预测明天是否下雨」这个预报时，你只需要看预报员给「下雨」分配的概率——你不需要关心他给「下雪」、「晴天」、「冰雹」各自分配了多少概率。**

在 LLM 的语境下，局部性意味着：当模型预测下一个词是「cat」，我们事后只用看模型给「cat」分配的概率来评判它，不需要关心它给「dog」、「car」、「the」等其他 49,999 个词分别分配了多少。这个性质在 LLM 中至关重要，因为：

1. **计算可行性**：非局部评分规则（如 Brier score）需要在每个位置计算并存储所有 $V$ 个类别的得分，这在 $V = 128{,}000$ 时是灾难
2. **语义合理性**：如果正确答案是「cat」，模型说「80% cat, 20% dog」和「80% cat, 20% car」的语义质量显然是不同的——但局部评分规则只关心「cat 是 80%」这个事实，其余的信息由训练过程中其他样本的统计平均来体现

**定理（Savage, 1971 / Gneiting & Raftery, 2007）**：
在局部性和严格恰当的条件下，$S$ 必然是对数评分规则的仿射变换：
$$\boxed{S(Q, x) = -a \ln Q(x) + b(x)}$$
其中 $a > 0$ 且 $b: \mathcal{X} \to \mathbb{R}$ 是任意不依赖于 $Q$ 的函数。

*完整公理化推导见 [附录 A](#附录-aproper-scoring-rule-的公理化推导)。*

**这个定理的意义**：
- MLE（等价于最小化 $-\ln Q(x)$）不仅是"一种合理的方法"——它几乎是**唯一**满足可加性/局部性要求的严格恰当评分规则。
- 信息论中的交叉熵也恰好是这个形式——这不是巧合，而是因为 Shannon 的公理和 Savage 的公理在底层共享同一种数学结构（Cauchy 函数方程）。
- **交叉熵 ⟺ MLE ⟺ 对数评分规则 ⟺ 唯一局部严格恰当的评分规则。**

**层次关系**：
```
Proper Scoring Rule (最底层: 决策论)
    │
    ├── 对数评分 (唯一局部严格恰当)
    │       │
    │       ├── 信息论视角: 交叉熵 = 编码代价
    │       └── 统计学视角: MLE = 频率学派推断
    │
    └── Brier Score (非局部, 均方误差的恰当版本)
           → 不适用于 LLM 因为不满足局部性
```

### 3.3 最小描述长度原则

一个补充但不冗余的视角来自计算学习理论。

**最小描述长度（Minimum Description Length, MDL）原则**（Rissanen, 1978）：最优模型是能够用最短编码描述数据 + 模型自身的模型。

在 Shannon 编码框架下：
- 用模型 $P_\theta$ 编码数据 $D$ 的最短期望码长 = $-\sum_i \ln P_\theta(x^{(i)})$
- 加上模型自身的编码代价（正则化项，如参数的比特数）
- 最小化总码长 → 选择能最好压缩数据的模型

**压缩即理解**：如果模型能用少量比特精确描述数据，它必然捕捉到了数据的规律。交叉熵损失正是这个思想的数学实现——它迫使模型学习数据的统计结构，以便用尽可能短的编码长度描述它。

---

## Part IV — LLM 预训练的形式化数学目标

### 4.1 自回归语言建模的概率图模型

**记号**：

- 词表 $\mathcal{V} = \{1, 2, \dots, V\}$
- 一条长度为 $T$ 的文本序列 $\mathbf{x} = (x_1, x_2, \dots, x_T)$，$x_t \in \mathcal{V}$
- 约定 $x_{<t} = (x_1, \dots, x_{t-1})$，$x_{\leq t} = (x_1, \dots, x_t)$

**概率分解**：由条件概率的链式法则，任意序列的联合概率可分解为：
$$p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

这是一个完全一般的分解，不依赖任何 Markov 假设或独立假设。

**自回归参数化**：语言模型 $P_\theta$ 在每一个位置 $t$ 输出一个条件分布，参数化方式为：
$$P_\theta(x_t \mid x_{<t}) = \text{Softmax}\big(W \cdot h_\theta(x_{<t})\big)[x_t]$$

其中：
- $h_\theta: \mathcal{V}^{t-1} \to \mathbb{R}^d$ 是 Transformer 网络，将历史序列编码为一个 $d$ 维隐向量
- $W \in \mathbb{R}^{V \times d}$ 是 LM Head 的投影矩阵
- $\text{Softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^V \exp(z_j)}$

**为什么必须是 Softmax？** 由 Part III 的讨论，对数评分规则的严格恰当性要求预测必须是有效的概率分布（$\sum_i a_i = 1$，$a_i > 0$）。Softmax 是连接 logits 与概率单纯形最自然的连续可微双射。

### 4.2 预训练损失函数的完整形式化

**预训练语料**：$\mathcal{D} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\}$，每条序列 $\mathbf{x}^{(i)}$ 长度 $T_i$。

**单个序列的损失**：
$$\ell(\theta; \mathbf{x}^{(i)}) = -\frac{1}{T_i} \sum_{t=1}^{T_i} \ln P_\theta(x_t^{(i)} \mid x_{<t}^{(i)})$$

**整个语料的总损失**（按 token 加权的平均）：
$$\boxed{\mathcal{L}(\theta) = -\frac{1}{\sum_{i=1}^{N} T_i} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \ln P_\theta(x_t^{(i)} \mid x_{<t}^{(i)})}$$

**自回归的 Shift 逻辑**：在实际实现中（参见 [`lm_head.py`](lm_head.py)），一个批次的 $L$ 个 token 的 logits 通过与目标标签错位 $1$ 个位置来计算：

```python
shifted_logits = logits[..., :-1, :]    # [B, L-1, V]
shifted_labels = targets[:, 1:]          # [B, L-1]
```

这是因为位置 $t$ 的 logits 预测的是位置 $t+1$ 的 token。

**逐 Token 展开**（代入 Softmax）：
$$\mathcal{L}(\theta) = -\frac{1}{\sum_i T_i} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \ln \frac{\exp\big(z_{x_t^{(i)}}(x_{<t}^{(i)}; \theta)\big)}{\sum_{v=1}^{V} \exp\big(z_v(x_{<t}^{(i)}; \theta)\big)}$$

这个目标**同时等价于**：
1. 经验交叉熵最小化（信息论）
2. 极大似然估计（统计学）
3. 对数恰当评分规则的最小化（决策论）

### 4.3 为什么这是一个合理的目标？

我们需要严格论证：最小化交叉熵是否真的会使模型逼近「真实语言分布」？

**情况 1：模型族充分大（well-specified case）**

假设存在 $\theta^*$ 使得 $P_{\theta^*} = P_{\text{data}}$。由 MLE 的一致性（Wald, 1949），当 $N \to \infty$：
$$\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta^*$$

即交叉熵（= MLE）是**一致估计量**——数据越多，模型越接近真实。

**情况 2：模型族不完美（misspecified case）**——这更接近 LLM 的实际情况

即使没有任何 $\theta$ 能使 $P_\theta = P_{\text{data}}$，极小化交叉熵等价于极小化 $D_{KL}(P_{\text{data}} \parallel P_\theta)$。由 Sanov 定理和 M-估计理论（Huber, 1967; White, 1982），MLE 收敛到：
$$\theta^* = \arg\min_\theta D_{KL}(P_{\text{data}} \parallel P_\theta)$$

这是 KL 散度意义下「最接近真实分布」的模型——即**KL 投影（information projection）**。

**情况 3：如果用 MSE/Brier Score 替代交叉熵？——一个重要澄清**

一个常见的误解是「MSE 会收敛到错误的分布」。我们需要严格澄清这一点。

**命题：MSE/Brier Score 在总体水平上也是严格恰当的。**

*证明*：在总体水平上，MSE 损失为：
$$\mathcal{R}_{\text{MSE}}(\mathbf{a}) = \mathbb{E}_{x \sim P}\left[\frac{1}{2} \sum_{i=1}^{V} (a_i - y_i)^2\right]$$

其中 $\mathbf{y}$ 是 $x$ 的 one-hot 编码。展开期望：
$$\mathcal{R}_{\text{MSE}}(\mathbf{a}) = \frac{1}{2} \sum_{x \in \mathcal{X}} P(x) \sum_{i=1}^{V} (a_i - \delta_{i,x})^2$$

对每个 $i$ 分量求偏导（带约束 $\sum_i a_i = 1$，引入 Lagrange 乘子 $\lambda$）：
$$\frac{\partial}{\partial a_i}\left[\mathcal{R}_{\text{MSE}} + \lambda\left(1 - \sum_j a_j\right)\right] = \sum_{x} P(x)(a_i - \delta_{i,x}) - \lambda = 0$$
$$\Rightarrow a_i \sum_x P(x) - \sum_x P(x)\delta_{i,x} = \lambda$$
$$\Rightarrow a_i - P(i) = \lambda$$

对所有 $i$ 求和：$\sum_i a_i - \sum_i P(i) = 1 - 1 = 0 = V\lambda$，故 $\lambda = 0$，进而：
$$\boxed{a_i^* = P(i), \quad \forall i}$$

**结论**：MSE/Brier score 在总体水平（无限数据）下的最优解**恰好是真实分布 $P$**。它确实是一个严格恰当的评分规则。

**那么 MSE 的问题在哪？** 如果总体水平上两者都收敛到 $P$，那为什么 LLM 不用 MSE？

问题出在三个方面，它们共同导致 MSE 在实践中不可行：

**第一：局部性的缺失。** Brier score 不是局部评分规则——评估对事件 $c$ 的预测时，需要知道所有 $V$ 个类别的预测概率。在 $V = 128{,}000$ 的词表上，这意味着（1）必须在每个 token 位置存储完整的概率向量 $[V]$，显存需求爆炸；（2）反向传播的梯度流经 $V$ 条路径，计算量是 CE 的 $V$ 倍。

**第二：梯度消失（详见 §5.4）。** 虽然总体最优解相同，但 MSE + Softmax 组合的梯度包含 $a_i(1 - a_i)$ 因子。当模型对正确答案极度不确信（$a_c \approx 0$，正是训练初期和最需要学习的时候），梯度 $\approx 0$，模型完全停滞。CE 的梯度在此情况下是 $a_c - 1 \approx -1$——最强拉力。

**第三：优化景观的几何差异。** 即使在总体水平上两者共享同一个全局最小值 $P$，CE 的损失景观（loss landscape）在远离最优点时具有更大的曲率和更直接的梯度方向，而 MSE 的景观是「平坦 + 陡峭」的混合——当预测概率接近 0 或 1 时极其平坦（梯度消失），中间区域才是良态的。这种几何差异使得 SGD/Adam 在 MSE 上收敛速度慢数个数量级。

**总结对比**：

| 性质                | CE (Log Score)       | MSE / Brier Score      |
| :---------------- | :------------------- | :--------------------- |
| 严格恰当（总体水平收敛到 $P$） | ✅                    | ✅                      |
| 局部性               | ✅（只依赖 $Q(c)$）        | ❌（依赖所有 $Q(j)$）         |
| 与 Softmax 组合的梯度   | $a_i - y_i$（线性，永不消失） | 包含 $a_i(1-a_i)$（饱和区消失） |
| 每 token 计算复杂度     | $O(V)$               | $O(V)$                 |
| 每 token 显存（中间张量）  | $O(1)$（LogSumExp 融合） | $O(V)$（需存储全概率向量）       |
| LLM 中是否使用         | ✅ 标准                 | ❌ 从不使用                 |

---

## Part V — Softmax + Cross-Entropy 的梯度魔法

本节给出交叉熵与 Softmax 组合求导的完整数学推导。这个推导本身在工程文献中被反复引用，但通常缺乏对 Kronecker delta 分情况讨论的完整步骤。这里补全每一步。

### 5.1 预备：Softmax 函数的 Jacobian 矩阵

**定义（Softmax）**：对于 logits 向量 $\mathbf{z} = (z_1, \dots, z_V) \in \mathbb{R}^V$，Softmax 输出：
$$a_i = \text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{k=1}^{V} e^{z_k}}, \quad i = 1, \dots, V$$

我们需要计算 Jacobian 矩阵 $J \in \mathbb{R}^{V \times V}$，其中 $J_{ji} = \frac{\partial a_j}{\partial z_i}$。

**情况 1：$i = j$（对角元素）**

使用商法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$：

设 $u = e^{z_i}$，$v = \sum_{k=1}^{V} e^{z_k}$。
$$\frac{\partial a_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_k e^{z_k} - e^{z_i} \cdot e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{\sum_k e^{z_k} - e^{z_i}}{\sum_k e^{z_k}} = a_i(1 - a_i)$$

**情况 2：$i \neq j$（非对角元素）**

$$\frac{\partial a_j}{\partial z_i} = \frac{0 \cdot \sum_k e^{z_k} - e^{z_j} \cdot e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = - \frac{e^{z_j}}{\sum_k e^{z_k}} \cdot \frac{e^{z_i}}{\sum_k e^{z_k}} = -a_j a_i$$

**统一形式**：引入 Kronecker delta $\delta_{ji} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$：
$$\boxed{\frac{\partial a_j}{\partial z_i} = a_j \cdot (\delta_{ji} - a_i)}$$

这个形式将 $V^2$ 个偏导数优雅地压缩为一行。Jacobian 矩阵可写作：
$$J = \text{diag}(\mathbf{a}) - \mathbf{a} \mathbf{a}^\top$$

其中 $\text{diag}(\mathbf{a})$ 是以 $\mathbf{a}$ 为对角元素的对角矩阵，$\mathbf{a} \mathbf{a}^\top$ 是秩-1 外积矩阵。

### 5.2 Cross-Entropy + Softmax 合体求导

**设置**：真实标签 $\mathbf{y} \in \{0,1\}^V$，$\sum_j y_j = 1$（one-hot）。交叉熵损失：
$$L(\mathbf{z}) = -\sum_{j=1}^{V} y_j \ln a_j$$

其中 $a_j = \text{Softmax}(\mathbf{z})_j$。

**Step 1：$\frac{\partial L}{\partial a_j}$**
$$\frac{\partial L}{\partial a_j} = -\frac{y_j}{a_j}$$

**Step 2：全导数链式法则**
$$\frac{\partial L}{\partial z_i} = \sum_{j=1}^{V} \frac{\partial L}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_i}$$

**Step 3：代入 Jacobian 和导数，关键约分发生**
$$\begin{aligned}
\frac{\partial L}{\partial z_i} &= \sum_{j=1}^{V} \left(-\frac{y_j}{a_j}\right) \cdot a_j (\delta_{ji} - a_i) \quad\color{gray}{\text{—— $a_j$ 消去！}} \\
&= \sum_{j=1}^{V} -y_j (\delta_{ji} - a_i) \\
&= -\sum_{j=1}^{V} y_j \delta_{ji} + \sum_{j=1}^{V} y_j a_i
\end{aligned}$$

**Step 4：利用 Kronecker delta 的筛选性质和 one-hot 的归一化**
$$\begin{aligned}
\sum_{j=1}^{V} y_j \delta_{ji} &= y_i \quad\color{gray}{\text{（只有 $j = i$ 时 $\delta_{ji} = 1$，其余为零）}} \\
\sum_{j=1}^{V} y_j a_i &= a_i \sum_{j=1}^{V} y_j = a_i \cdot 1 = a_i
\end{aligned}$$

**Step 5：最终结果**
$$\boxed{\frac{\partial L}{\partial z_i} = a_i - y_i}$$

**这个公式的美感**：
- 它**完全是线性的**！MSE 的梯度含 $a_i(1-a_i)$ 的乘积饱和因子，而 CE 的对数求导 $\frac{1}{a_j}$ 恰好与 Softmax Jacobian 中的 $a_j$ 消去
- 如果模型极其确信正确（$a_c \approx 1$），则 $\frac{\partial L}{\partial z_c} \approx 0$——几乎不需要修正
- 如果模型极其确信错误（$a_c \approx 0$），则 $\frac{\partial L}{\partial z_c} \approx -1$——最大反向拉力
- 这个「线性梯度」的特性使优化器在任意 logit 尺度下都能获得稳定的信号，**彻底消灭了 Softmax 的饱和区导致的梯度消失**

### 5.3 梯度下降的力学直觉：拉力、推力与零和博弈

§5.2 推导了损失对 logits 的偏导数 $\frac{\partial L}{\partial z_i} = a_i - y_i$。但在梯度下降算法中，参数（以及对应的 $z_i$）的更新方向是梯度的**反方向**。因此，我们可以通过分析偏导数加上负号后的结果，来观察损失函数对每个 logit 值的实际作用力。

设更新步长（学习率）为 $\eta > 0$，对 logit $z_i$ 的更新方向可以表示为：

$$\Delta z_i \propto -\frac{\partial L}{\partial z_i} = y_i - a_i$$

这个简洁的表达式揭示了一个直观的力学图景：梯度下降在「拉高」正确的 logit，同时「压低」所有错误的 logit。我们可以将 token 分为「正确位置的 token」和「非正确位置的 token」两类来具体分析。

**1. 当该 token 是正确位置时（$y_i = 1$）**

- **更新方向**：$\Delta z_i \propto 1 - a_i$
- **作用机制**：因为预测概率 $a_i \in (0, 1)$，所以 $1 - a_i$ 永远为正数。这产生了一个**正向的拉力**，使正确 token 的原始分数 $z_i$ 变大。
- **力度自适应**：如果模型对正确 token 的预测概率 $a_i$ 很低（例如接近 0），则拉力 $1 - a_i \approx 1$ 非常大——模型在「亡羊补牢」；如果模型已经很自信（$a_i$ 接近 1），则拉力趋近于 0——「你已经对了，不必再调整」。

**2. 当该 token 不是正确位置时（$y_i = 0$）**

- **更新方向**：$\Delta z_i \propto -a_i$
- **作用机制**：因为 $a_i > 0$，所以 $-a_i$ 永远为负数。这产生了一个**负向的推力（压制力）**，迫使错误 token 的原始分数 $z_i$ 变小。
- **力度自适应**：如果模型在某个错误的 token 上给出了很高的预测概率 $a_i$，那么 $-a_i$ 的绝对值就会很大，从而产生一个极强的压制力将其 logits 压低——「你太自信了，这是错的，压下去！」；如果模型对该错误 token 的预测概率本来就很低（$a_i \approx 0$），那么压制力也会变得微乎其微——「你已经知道它不对了，不用再管」。

**零和博弈：拉力与推力的精确平衡**

从整体上看，所有 logits 受到的作用力之和为：

$$\sum_{i} (y_i - a_i) = \sum_{i} y_i - \sum_{i} a_i = 1 - 1 = 0$$

这个恒等式揭示了一个深刻的结构性质：在每一次更新中，**提升正确 token 分数的「拉力」，与压低所有错误 token 分数的「推力」总和是完全相等的**。这并非偶然——它是 Softmax 归一化（$\sum_i a_i = 1$）和 one-hot 标签归一化（$\sum_i y_i = 1$）的直接推论。

这种零和机制确保了 Softmax 概率分布的竞争性本质：要让正确的 token 脱颖而出，既需要拉高正确的，也需要压低错误的。两者的力量精确对等——每把正确的 logit 拉高一点，就必须把错误的 logits 总共压低同样的量。从这个意义上说，交叉熵损失的梯度下降过程，本质上是在 logits 空间中进行一场**守恒的力量重分配**。

### 5.4 对比：MSE + Softmax 为什么失败

**设置**：同样的 Softmax 输出 $\mathbf{a}$，MSE 损失：
$$L_{\text{MSE}}(\mathbf{z}) = \frac{1}{2} \sum_{j=1}^{V} (a_j - y_j)^2$$

**Step 1：$\frac{\partial L_{\text{MSE}}}{\partial a_j} = a_j - y_j$**

**Step 2：链式法则**
$$\begin{aligned}
\frac{\partial L_{\text{MSE}}}{\partial z_i} &= \sum_{j=1}^{V} (a_j - y_j) \cdot \frac{\partial a_j}{\partial z_i} \\
&= \sum_{j=1}^{V} (a_j - y_j) \cdot a_j (\delta_{ji} - a_i)
\end{aligned}$$

**Step 3：展开（分离对角和非对角项）**
$$\begin{aligned}
\frac{\partial L_{\text{MSE}}}{\partial z_i} &= \underbrace{(a_i - y_i) \cdot a_i (1 - a_i)}_{\text{对角项 } j=i} + \underbrace{\sum_{j \neq i} (a_j - y_j) \cdot a_j \cdot (-a_i)}_{\text{非对角项 } j \neq i} \\
&= (a_i - y_i) a_i (1 - a_i) - a_i \sum_{j \neq i} (a_j - y_j) a_j
\end{aligned}$$

**致命问题**：
- 当模型错得离谱时（$a_c \approx 0$，$y_c = 1$），对角项 $(0 - 1) \cdot 0 \cdot 1 = 0$
- 非对角项中每项都含 $a_i$（当 $i = c$ 时 $a_c \approx 0$），整体 $\approx 0$
- **梯度消失 → 模型卡死，无法从错误中学习**

**数学对比总结**：

| 损失函数 | $\partial L / \partial z_i$ | 当 $a_i \approx 0$ 且 $y_i = 1$ 时 | 是否为 Proper Scoring Rule |
|:---------|:----------------------------|:-----------------------------------|:---------------------------|
| **CE + Softmax** | $a_i - y_i$ | $\approx -1$（强梯度） | ✅ 是（局部） |
| **MSE + Softmax** | $(a_i - y_i)a_i(1-a_i) - a_i \sum_{j \neq i}(a_j - y_j)a_j$ | $\approx 0$（梯度消失） | ✅ 是（非局部，Brier score） |

**深层分析——梯度消失不是「bug」，是结构不匹配**

MSE 的梯度消失问题不能简单归结为「优化器不给力」。它揭示了 MSE 与 Softmax 参数化之间的一个**结构性不匹配**：

- Softmax 的设计哲学是：logits 的**相对差异**决定了概率分布。将 logits 整体加一个常数，概率分布不变。这是一种「平移不变」的参数化。
- CE 损失的梯度 $\partial L / \partial z_i = a_i - y_i$ **继承了这种平移不变性**：如果将 logits 整体加常数 $c$，Softmax 输出不变，CE 损失不变，梯度也不变。CE 的梯度「生活」在 Softmax 的概率单纯形上，天然匹配。
- MSE 损失的梯度 **破坏了这种平移不变性**：虽然 Softmax 输出本身是平移不变的，但 MSE 关于 $a_i$ 的导数是 $a_i - y_i$，经过 Softmax 的 Jacobian 链式传播后，梯度被额外的 $a_i(1-a_i)$ 和 $a_i$ 因子「污染」了。这些因子反映了**Softmax 将 logits 空间映射到概率单纯形时的局部收缩/膨胀**——在概率极端区域（$a_i \approx 0$ 或 $1$），logits 的一个单位变化只引起概率的极小变化，这是 Softmax 的固有能力：通过将大范围的 logits 差异「压缩」进有界的概率区间来稳定输出。MSE 没有对这种压缩做「反补偿」，导致梯度信号在饱和区被压垮。CE 通过对数求导的 $\frac{1}{a_j}$ 项恰好做了这个反补偿，使得梯度信号不随 Softmax 的局部伸缩而衰减。

这就是为什么「CE + Softmax 是天作之合」——不是巧合，而是对数评分规则在 Softmax 参数化下的梯度恰好自洽。

### 5.5 从 Fisher 信息矩阵看交叉熵

**Fisher 信息矩阵**定义为负对数似然的 Hessian 的期望：
$$\mathcal{I}(\theta)_{kl} = \mathbb{E}_{x \sim P_\theta}\left[-\frac{\partial^2 \ln P_\theta(x)}{\partial \theta_k \partial \theta_l}\right]$$

对于 Softmax + CE 组合，关于 logits $\mathbf{z}$ 的 Fisher 信息矩阵为：
$$\mathcal{I}(\mathbf{z})_{ij} = \mathbb{E}_{\mathbf{a} \sim \text{Softmax}(\mathbf{z})}\left[-\frac{\partial^2 L}{\partial z_i \partial z_j}\right] = a_i(\delta_{ij} - a_j)$$

恰好等于 Softmax 输出的**协方差矩阵**。这意味着：
- 在自然梯度下降（Amari, 1998）中，用 $\mathcal{I}^{-1}$ 作为预条件子，等价于在模型分布的内在 Riemann 流形上做最速下降
- CE 损失下的 Fisher 信息矩阵具有如此简洁的形式，这是使自然梯度实用化的关键因素
- 这也是为什么 AdamW 等自适应优化器在 CE 损失上效果特别好的深层原因——对角近似的 Fisher 信息矩阵天然地对齐了 Softmax 的曲率结构

---

## Part VI — 数值计算：LogSumExp 的数学与工程

Part V 给出了完美的数学梯度，但在浮点运算中，直接计算 Softmax 和 log 会遇到灾难性的溢出问题。本节严格推导 LogSumExp 技巧。

### 6.1 朴素实现的溢出分析

**设置**：logits $\mathbf{z} \in \mathbb{R}^V$，通常数值范围可达 $[-100, 100]$ 甚至更宽（深层 Transformer 的输出方差很大）。

**朴素实现**：$L = -\ln\left(\frac{e^{z_c}}{\sum_{j=1}^V e^{z_j}}\right)$

**FP32 的数值边界**：

| 现象 | 条件 | 结果 |
|:-----|:-----|:-----|
| 上溢 | $z_i > \ln(\text{FP32}_{\max}) \approx 88.72$ | $e^{z_i} \to +\infty$（inf） |
| 下溢 | $z_i < \ln(\text{FP32}_{\min+}) \approx -87.34$ | $e^{z_i} \to 0.0$ |
| $\ln(0)$ | 目标词 $e^{z_c}$ 已下溢 | $-\infty$ → 梯度 NaN |

**BF16 的数值边界**（更严苛）：

| 现象 | 条件 | 结果 |
|:-----|:-----|:-----|
| 上溢 | $z_i > \ln(\text{BF16}_{\max}) \approx 88.72$ | $+\infty$（与 FP32 相同，因为指数位相同） |
| 下溢 | $z_i < \ln(\text{BF16}_{\min+}) \approx -87.34$ | 0.0 |

BF16 的致命问题不在溢出而在精度：7 位尾数位导致累加时 $1.0 + 0.0067 \approx 1.0$（大数吃小数），在 10 万词表累加时灾难性失真。

### 6.2 LogSumExp 技巧的严格推导

**Step 1：将对数从分数中拆出**
$$L = -\ln e^{z_c} + \ln \sum_{j=1}^V e^{z_j} = -z_c + \ln \sum_{j=1}^V e^{z_j}$$

这个恒等式 $\ln(A/B) = \ln A - \ln B$ 已经消除了分母中的 $e^{z_c}$ 带来的下溢风险（因为它是减去了一个数而不是除以一个接近 0 的数）。

**Step 2：减去最大值，确保指数安全**
令 $M = \max_j z_j$。利用 $e^{z_j} = e^{z_j - M} \cdot e^{M}$：
$$\begin{aligned}
\ln \sum_{j=1}^V e^{z_j} &= \ln\left( e^M \sum_{j=1}^V e^{z_j - M} \right) \\
&= \ln e^M + \ln \sum_{j=1}^V e^{z_j - M} \\
&= M + \ln \sum_{j=1}^V e^{z_j - M}
\end{aligned}$$

**关键**：$z_j - M \leq 0$（对所有 $j$），最大的指数项为 $e^0 = 1$，其余的 $e^{z_j - M} \in (0, 1]$。这保证了：
- 所有指数项 $\leq 1$，不会上溢
- 求和中有至少一个 $e^0 = 1$，$\ln(\text{求和}) \geq 0$，不会出现 $\ln(0)$

**Step 3：复合到交叉熵损失**
$$\begin{aligned}
L &= -z_c + M + \ln \sum_{j=1}^V e^{z_j - M}
\end{aligned}$$

**更优雅的形式（对数减去最大值）**：
由于 $z_c - M \leq 0$，而 $-(z_c - M) = M - z_c \geq 0$：
$$\boxed{L = -(z_c - M) + \ln \sum_{j=1}^V e^{z_j - M}}$$

**在代码中的实现**（参见 [`lm_head.py:66-85`](lm_head.py)）：
```python
max_logits, _ = torch.max(valid_logits_fp32, dim=-1, keepdim=True)  # [M, 1]
safe_logits = valid_logits_fp32 - max_logits                        # [M, V]
true_safe_logits = safe_logits.gather(dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
exp_logits = torch.exp(safe_logits)                                  # [M, V]
log_sum_exp = torch.log(torch.sum(exp_logits, dim=-1))              # [M]
loss = (-true_safe_logits + log_sum_exp).mean()
```

**等价性验证**：PyTorch 的 `F.cross_entropy` 内部等价于 `F.log_softmax` + `F.nll_loss`，而 `F.log_softmax` 内部使用的正是上述 LogSumExp 算法。手写实现与官方的数值误差应满足 $\text{atol} < 10^{-4}$。

### 6.3 Safe Softmax 与 LogSumExp 的本质区别

**Safe Softmax**（分步实现）：
```python
probs = safe_softmax(logits)      # Step 1: 计算所有概率
loss = -torch.log(probs[target])  # Step 2: 取 log
```

其中 `safe_softmax` 的实现为：
$$\text{safe\_softmax}(\mathbf{z})_i = \frac{e^{z_i - M}}{\sum_j e^{z_j - M}}$$

**区别 1：显存**。Safe Softmax 产生完整的 $[B, L, V]$ 概率矩阵写入 HBM（高达 10-20 GB），LogSumExp 在寄存器内一步完成，无需写出中间张量。

**区别 2：$\log(0)$ 风险**。当 $z_c - M$ 极度负（如 $-110$），Safe Softmax 输出的 $P(c)$ 在 FP16/BF16 下会被截断为绝对 $0.0$，后续 $\ln(0.0) = -\infty$。而 LogSumExp 的合并公式中该项变为 $-(z_c - M) + \cdots = 110 + \cdots$——恶劣预测转化为巨大的线性惩罚，而非致命的无穷大。

**数学上完全等价，工程上完全不同。** 这类似于 FlashAttention 的核心理念：计算图等价，但内存访问模式根本不同。

### 6.4 混合精度的失效边界

大模型训练的标准策略：所有矩阵乘法用 BF16（速度），但跨量级求和操作必须转为 FP32（精度）。

**BF16 在 LogSumExp 中失效的严格条件**：

设 $M = \max_j z_j$，我们需要计算：
$$S = e^0 + \sum_{j: z_j < M} e^{z_j - M}$$

BF16 的机器精度（machine epsilon）在 $1.0$ 附近为 $\epsilon_{\text{BF16}} = 2^{-7} = 0.0078125$。

任何 $e^{z_j - M} < \epsilon_{\text{BF16}}$ 的项，在累加到 $1.0$ 时会被截断为 0（swamping effect）。临界条件：
$$z_j - M < \ln(0.0078125) \approx -4.85$$

即：**任何 logit 低于最大值 4.85 nats 以上的词，在 BF16 累加中完全消失。**

对于 $V = 128,000$ 的词表，若有 99% 的词（约 126,720 个）低于此阈值，且它们的平均 $e^{z_j - M} \approx e^{-8} \approx 0.000335$，则数学上它们合计贡献 $126720 \times 0.000335 \approx 42.5$，但在 BF16 中全部被吞没——LogSumExp 的求和值从真实的 $\ln(1 + 42.5 + \cdots) \approx 3.77$ 偏小到 $\ln(1) = 0$，误差不可接受。

**结论**：LogSumExp 的求和阶段**必须**在 FP32 下进行。FP32 的 $\epsilon_{\text{FP32}} \approx 1.19 \times 10^{-7}$，吞噬阈值约为 $\ln(10^{-7}) \approx -16.1$，足以覆盖几乎所有实际 logit 差异。

---

## Part VII — 从损失到评价：困惑度与信息密度

交叉熵损失是训练时的优化目标，但在论文和基准测试中，极少直接报告「交叉熵 = 2.35 nats/token」。取而代之的是**困惑度（Perplexity, PPL）**。本节系统性地解释困惑度的定义、直觉、数学性质、以及它与语言信息密度的深层联系。

### 7.1 困惑度（Perplexity）的严格定义

**定义**：给定一个长度为 $T$ 的测试序列 $\mathbf{x} = (x_1, \dots, x_T)$，语言模型 $P_\theta$ 分配的困惑度为：

$$\boxed{\text{PPL}(\mathbf{x}) = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\ln P_\theta(x_t \mid x_{<t})\right) = \exp\big(\mathcal{L}_{\text{CE}}(\mathbf{x})\big)}$$

其中 $\mathcal{L}_{\text{CE}}(\mathbf{x}) = -\frac{1}{T} \sum_{t=1}^{T} \ln P_\theta(x_t \mid x_{<t})$ 是序列上每 token 的平均交叉熵（单位：nats）。PPL 的单位与 $\mathcal{L}_{\text{CE}}$ 中 log 的底数一致——使用自然对数时，PPL 在「nats 的指数」意义下定义；若改用 $\log_2$，则 PPL 是在「bits 的 $2^{\mathcal{L}}$」意义下定义。在实际研究中，约定俗成是使用 $\exp(\mathcal{L}_{\text{CE}})$（nats）。

**等价形式**：取对数后可以直接看出 PPL 是序列联合概率的几何平均的倒数：
$$\text{PPL}(\mathbf{x}) = \left(\prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})\right)^{-1/T} = \frac{1}{\sqrt[T]{\prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})}}$$

### 7.2 困惑度的直觉——「有效分支因子」

困惑度最核心的直觉是：**PPL 可以理解为模型在每个预测位置上「平均面对多少个等可能的候选」。**

**极端情况建立直觉**：

| 场景        | 模型行为               | $\mathcal{L}_{\text{CE}}$  | PPL             | 解释                |
| :-------- | :----------------- | :------------------------- | :-------------- | :---------------- |
| 完美预测      | 每次给正确答案概率 1.0      | 0.0                        | $e^0 = 1$       | 从不纠结，永远知道下一个词是什么  |
| 二选一       | 每次在两个等可能词中选        | $\ln 2 \approx 0.693$      | $e^{\ln 2} = 2$ | 平均在 2 个候选中纠结      |
| 均匀猜测（小词表） | $V=10$，均匀随机        | $\ln 10 \approx 2.30$      | 10              | 平均在 10 个候选中纠结     |
| 均匀猜测（大词表） | $V=128{,}000$，均匀随机 | $\ln 128000 \approx 11.76$ | 128,000         | 完全随机，每个 token 都是猜 |
| 典型预训练模型   | LLaMA-3 8B 级别的预测能力 | ~2.1                       | ~8.2            | 平均在约 8 个候选词中纠结    |

**关键理解——PPL 是对「均匀猜测」的校准**：如果模型在每个位置的损失是 $\mathcal{L}$，那么困惑度 $e^{\mathcal{L}}$ 就是在问：「如果模型在每个位置都在 $K$ 个等可能的候选中做均匀猜测，且这个均匀猜测的交叉熵恰好也是 $\mathcal{L}$，那么 $K$ 是多少？」答案是 $K = e^{\mathcal{L}}$。这就是 PPL。

### 7.3 困惑度的优越性——为什么不用原始的 CE 值？

困惑度相比原始的交叉熵数值，有几个实际的优越性：

**可解释性**：「PPL = 50」比「CE = 3.91 nats」更容易让非信息论背景的人建立直觉——前者可以直接理解为「模型平均在 50 个词中纠结」，后者则需要额外的思维转换。

**跨词表的近似可比性**：虽然不同词表的 token 定义不同，不能直接比较 PPL 的绝对值，但 PPL 至少给出了一个与词表大小无关的「归一化」：均匀猜测时 $\text{PPL} = V$（与直觉一致），而 CE 的均匀猜测值是 $\ln V$（取决于词表大小）。这使得研究者在看到 PPL 时能快速判断模型离「随机猜测」有多远。

**数学上的简洁关系**：PPL 与 token 级别的平均「确定性」有直接关系。如果模型的平均预测概率为 $\bar{p}$（几何平均），则 $\text{PPL} = 1 / \bar{p}$。PPL = 10 意味着模型平均给每个正确答案的概率约为 0.1。

### 7.4 困惑度的分解——它到底衡量了什么？

将交叉熵的核心恒等式代入 PPL 的定义：
$$\begin{aligned}
\text{PPL} &= \exp\big(H(P) + D_{KL}(P \parallel P_\theta)\big) \\
&= \underbrace{\exp(H(P))}_{\text{数据的固有困惑度}} \times \underbrace{\exp(D_{KL}(P \parallel P_\theta))}_{\text{模型误差带来的额外困惑度放大}}
\end{aligned}$$

这个分解具有深远的意义：

- **$\exp(H(P))$** 是**不可约困惑度（irreducible perplexity）**——这是语言本身的「固有难度」。即使有一个完美的语言模型，它也无法将 PPL 降到这个值以下。这个值由语言的内在不确定性决定：语法歧义（「I saw a man with a telescope」——是「我用望远镜看到一个人」还是「我看到一个拿望远镜的人」？）、语义模糊、以及本质上不可预测的创造性表达，都贡献了这个下界。

- **$\exp(D_{KL}(P \parallel P_\theta)) \geq 1$** 是**模型误差放大因子**。当 $D_{KL} = 0.5$ nats 时，$\exp(0.5) \approx 1.65$，意味着模型的缺陷使困惑度比理论下界膨胀了 65%。

**由此可以理解**：
- 英语的自然困惑度下限（字符级）的经典 Shannon (1951) 估计约为 $\exp(1.2 \text{ bits}) \approx 2.3$ 的「分支因子」——这个数太低了，是因为 Shannon 实验使用字符级（letter-level）预测，而我们讨论的是 token-level。对于 BPE tokenization，由于 token 本身已经携带了大量信息，token-level 的 PPL 一般在 5-50 量级。
- 现代大模型的 PPL 在 benchmark 上不断下降（从 GPT-2 的 ~35 到 LLaMA-3 的 ~8 在 WikiText 上），本质上是在逼近 $\exp(H(P))$。每个点的下降都越来越「贵」——因为 $D_{KL}$ 的减少变得越来越困难。

### 7.5 困惑度的局限性与注意事项

**不同 tokenizer 的 PPL 不可直接比较**。这是最常见的误用。一个使用 BPE（词表 50K）的模型和另一个使用 SentencePiece（词表 32K）的模型在同一个测试集上的 PPL，即使模型能力完全相同，也会因为 token 的平均信息密度不同而产生差异。补救方法之一是使用 **bits-per-byte (BPB)** 或 **bits-per-character (BPC)** 进行归一化：
$$\text{BPB} = \frac{\mathcal{L}_{\text{CE}} / \ln 2}{\text{平均每 token 的 UTF-8 字节数}}$$

BPB 将交叉熵归一化到「每个原始字节」的水平，消除了 tokenization 方案的影响，使不同模型可以公平比较。

**PPL 对罕见 token 高度敏感**。由于 PPL 取了几何平均，一个极其糟糕的预测（如给正确答案 $10^{-6}$ 的概率）会在指数函数放大下剧烈推高 PPL。这意味着 PPL 对模型在长尾词汇上的表现特别敏感——而这通常恰恰是模型最薄弱的地方。从这个意义上说，PPL 是一个「严厉」的指标。

**极低 PPL 不一定意味着好模型**。一个在训练集上严重过拟合的语言模型可能会在域内测试集上报告极低的 PPL（因为记住了训练数据），但面对稍微偏离分布的文本时表现崩溃。PPL 必须在分布外（out-of-distribution）数据上评估才有真正的诊断价值。

### 7.6 困惑度的实际数值与经验参考

以下是一些有参考价值的典型数值（WikiText-103 / 类似基准）：

| 模型规模 | 典型 PPL | 典型 CE (nats) | 年代 |
|:---------|:---------|:---------------|:-----|
| 均匀猜测 (V=50K) | 50,000 | 10.82 | — |
| GPT-2 Small (124M) | ~35 | ~3.56 | 2019 |
| GPT-2 XL (1.5B) | ~18 | ~2.89 | 2019 |
| LLaMA-7B | ~7.5 | ~2.01 | 2023 |
| LLaMA-2 70B | ~4.5 | ~1.50 | 2023 |
| LLaMA-3 8B | ~8.2 | ~2.10 | 2024 |
| 理论上限 $\exp(H(P))$ | 未知（估计 3-6） | 未知（估计 1.1-1.8） | — |

**解读经验数值**：
- 2019 年到 2023 年，PPL 从 ~35 降至 ~5，下降约 7 倍。但 CE 仅从 3.56 降至 ~1.6 nats——因为 PPL 是指数标度，后期的每个 bit 的改进（比如从 CE=2.0 → 1.8）对应的 PPL 变化看起来比前期小（从 7.4 → 6.0 的绝对降幅不如 35 → 25），但实际上难度更大。
- 理论上限 $\exp(H(P))$ 的具体数值是一个活跃的研究问题。不同语料、不同语言、不同领域的 $\exp(H(P))$ 不同。代码的熵通常低于自然语言（因为代码的句法更严格），对话的熵高于百科文本（因为对话更不可预测）。
- 值得注意的是：如果 PPL 的剩余下降空间（从当前 ~5 到理论下界 ~4 的差距）越来越小，这意味着 $D_{KL}$ 的改进空间也在收窄——模型在统计层面已经接近捕捉了语言的大部分可预测结构，剩余的不确定性更多来自语言本身而非模型的不足。但这恰恰说明了「下一个 token 预测」作为一个统计学习任务也在接近其能力的极限——这为 Phase 3-6 的推理和对齐研究提供了背景动机。

### 7.7 交叉熵与语言的信息密度

**Shannon 游戏与现代 LLM**：Shannon (1951) 的经典实验要求人类被试逐个猜测一句话的下一个字母，记录猜对的次数。他发现英语的字符级熵约为 0.6-1.3 bits/character。半个多世纪后，LLM 在 token 级别上自动完成了 Shannon 当年用人类实验想测量的东西——而两者的结果高度一致。这本身就是对「语言建模即压缩」哲学的一个强有力的实证验证。

**为什么「下一个 token 预测」如此困难但仍然是最有效的范式？**

自然语言同时具有两个看似矛盾的特性：
1. **高局部可预测性**：给定充分上下文，「the cat sat on the ____」中下一个词的候选被强烈约束在少数几个选项（mat, floor, chair...）
2. **长尾不可预测性**：但在大量真实文本中，下一个 token 常常是创造性的、信息密集的（新的人名、数字、代码符号、专有名词），模型在这些情况下几乎是在做「均匀猜测」

交叉熵将这两个极端统一在一个连贯的度量框架下：低损失（~0.1 nats）来自可预测的结构化上下文，高损失（~5-10 nats）来自长尾的新颖内容。平均下来（~2 nats），即 PPL ≈ 7-8，恰好反映了语言作为「半结构化符号序列」的本质——既不是完全随机的噪声，也不是完全确定的程序。

---

## 附录 A：Proper Scoring Rule 的公理化推导

本附录给出 Savage (1971) 对局部严格恰当评分规则的完整公理化推导。这是 Part III §3.2 中引用定理的严格证明。

### A.1 问题设定

- 样本空间 $\mathcal{X} = \{1, 2, \dots, V\}$（有限，$V \geq 2$）
- 概率单纯形 $\Delta_V = \{(p_1, \dots, p_V) : p_i > 0, \sum p_i = 1\}$（内部点，以排除退化的边界情况）
- 评分规则 $S: \Delta_V \times \mathcal{X} \to \mathbb{R}$

**局部性假设**：$S(Q, x) = s(Q(x))$，其中 $s: (0, 1] \to \mathbb{R}$。即惩罚只依赖于分配给实际发生事件的概率 $Q(x)$，不依赖 $Q$ 在其他未发生事件上的分配。

### A.2 恰当性的数学条件

$S$ 是恰当的，当且仅当对所有 $P, Q \in \Delta_V$：
$$\sum_{x=1}^{V} P(x) \cdot s(P(x)) \leq \sum_{x=1}^{V} P(x) \cdot s(Q(x))$$

即：诚实报告 $P$ 是使期望损失最小化的策略。

令函数 $G(p) = p \cdot s(p)$（乘以概率后的加权损失）。恰当性条件等价于：对所有 $P, Q \in \Delta_V$：
$$\sum_{x} G(P(x)) \leq \sum_{x} P(x) \cdot s(Q(x))$$

### A.3 Euler 方程的导出

考虑对 $Q$ 的微小扰动。固定 $P$，设 $Q = P + \epsilon \cdot \eta$，其中 $\eta$ 满足 $\sum_x \eta_x = 0$（保持 $\sum Q(x) = 1$），且 $\epsilon$ 充分小以保证 $Q$ 仍在 $\Delta_V$ 内部。

定义 $\Phi(Q) = \sum_x P(x) \cdot s(Q(x))$。在 $Q = P$ 处取最优化的一阶条件（变分导数）：
$$\left.\frac{d}{d\epsilon} \Phi(P + \epsilon \eta)\right|_{\epsilon=0} = 0$$

展开：
$$\frac{d}{d\epsilon} \sum_x P(x) \cdot s(P(x) + \epsilon \eta_x) = \sum_x P(x) \cdot s'(P(x)) \cdot \eta_x$$

令 $f(p) = p \cdot s'(p)$。一阶条件为：对所有满足 $\sum \eta_x = 0$ 的扰动 $\eta$：
$$\sum_x f(P(x)) \cdot \eta_x = 0$$

### A.4 从一阶条件到函数方程

上述条件意味着 $f(P(x))$ 在约束 $\sum \eta_x = 0$ 下与 $\eta$ 正交。由线性代数基本引理，这意味着 $f(P(x))$ 对所有 $x$ 取相同的值——即 $f(p)$ 是常数函数。

**证明**：若存在 $x_1, x_2$ 使得 $f(P(x_1)) \neq f(P(x_2))$，构造 $\eta$ 为 $\eta_{x_1} = 1$，$\eta_{x_2} = -1$，其余为零，则 $\sum f(P(x))\eta_x \neq 0$，与条件矛盾（注意 $\sum \eta_x = 0$ 满足）。故 $f(p) = c$（常数）。

即：
$$p \cdot s'(p) = c, \quad \forall p \in (0, 1]$$

### A.5 微分方程求解

$$s'(p) = \frac{c}{p}$$

积分得：
$$s(p) = c \ln p + d$$

其中 $d$ 是积分常数。由于 $s$ 随 $p$ 单调递减（越高概率应越小惩罚），我们有 $c < 0$。令 $a = -c > 0$：
$$s(p) = -a \ln p + d$$

**严格恰当性的验证**（二阶条件）：计算 $\Phi(Q) - \Phi(P)$：
$$\begin{aligned}
\Phi(Q) - \Phi(P) &= \sum_x P(x)\big[s(Q(x)) - s(P(x))\big] \\
&= -a \sum_x P(x) \ln \frac{Q(x)}{P(x)} \\
&= a \cdot D_{KL}(P \parallel Q) \geq 0
\end{aligned}$$

由 Gibbs 不等式，等号当且仅当 $P = Q$。故 $a > 0$ 保证严格恰当性。

### A.6 最终结论

**局部严格恰当的评分规则必然具有以下形式**：
$$\boxed{S(Q, x) = -a \ln Q(x) + b(x)}$$

其中 $a > 0$，$b: \mathcal{X} \to \mathbb{R}$ 是任意不依赖于 $Q$ 的函数。

- 取 $a = 1, b(x) \equiv 0$ 得到**对数评分规则**：$S_{\log}(Q, x) = -\ln Q(x)$
- 加上与 $Q$ 无关的项 $b(x)$ 不影响优化问题（argmin/argmax 不变），故交叉熵/对数评分在仿射等价意义下是唯一的
- 任何其他局部评分规则（如 Brier score $S(Q, x) = (1 - Q(x))^2 + \sum_{y \neq x} Q(y)^2$）都不是局部的——Brier score 的评估同时依赖于所有 $Q(y)$

---

## 附录 B：Label Smoothing 的正则化解释

Part II 中提到 CE 损失对「过度自信的错误」惩罚极重，Label Smoothing 是缓解这一问题的正则化技术。

### B.1 标准 CE 的尖峰问题

考虑 one-hot 标签 $\mathbf{y} = (0, \dots, 0, 1, 0, \dots, 0)$。CE 损失 $L = -\ln a_c$ 的唯一目标是最小化 $a_c$ 与 $1$ 的差距。最优解要求 $a_c \to 1$，但这在 Softmax 下等价于 $z_c \to +\infty$ 且 $z_{j \neq c} \to -\infty$——logits 无限发散，模型极度 "overconfident"。

这导致两个问题：
1. **过拟合**：模型对训练集中见过的 token 过于自信，泛化到未见 token 时反而不可靠
2. **对噪声敏感**：标注错误或歧义 token 会使模型剧烈修正

### B.2 Label Smoothing 的形式化

将 one-hot 标签替换为平滑标签（Szegedy et al., 2016）：
$$\tilde{y}_j = (1 - \epsilon) \cdot y_j + \frac{\epsilon}{V}, \quad j = 1, \dots, V$$

其中 $\epsilon \in [0, 1]$（通常取 $\epsilon = 0.1$），$V$ 是词表大小。

**平滑后的交叉熵**：
$$\begin{aligned}
\tilde{L} &= -\sum_{j=1}^{V} \tilde{y}_j \ln a_j \\
&= -(1 - \epsilon) \ln a_c - \frac{\epsilon}{V} \sum_{j=1}^{V} \ln a_j \\
&= (1 - \epsilon) \cdot L_{\text{CE}} + \epsilon \cdot H(\text{Uniform}, P_\theta)
\end{aligned}$$

即：Label Smoothing = 标准 CE（权重 $1 - \epsilon$）+ 与均匀分布的交叉熵（权重 $\epsilon$）。

**最优解变为**：$a_c^* = 1 - \epsilon + \epsilon/V$，$a_{j \neq c}^* = \epsilon/V$。模型不再把 logits 推向无穷，而是收敛到有限值（由 $\epsilon$ 决定的平衡态）。

### B.3 与 Proper Scoring Rule 的联系

Label Smoothing 改变了目标分布 $P$（从 $\hat{P}$ 变为 $\tilde{P}$）。从信息论角度看，它等价于在 KL 散度中加入均匀先验的惩罚：
$$\min_\theta D_{KL}(\tilde{P} \parallel P_\theta) = \min_\theta \left[ (1-\epsilon) D_{KL}(\hat{P} \parallel P_\theta) + \epsilon D_{KL}(\text{Uniform} \parallel P_\theta) \right]$$

这恰好是贝叶斯视角下的 MAP 估计——均匀分布作为 Dirichlet 先验的超参数。

---

## 附录 C：温度参数化与知识蒸馏

### C.1 温度调节的 Softmax

引入温度参数 $T > 0$：
$$\text{Softmax}_T(\mathbf{z})_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T \to 0$：趋于 argmax（硬决策），分布变为 one-hot
- $T = 1$：标准 Softmax
- $T \to +\infty$：趋于均匀分布

**温度对交叉熵损失的影响**：$L_T = -\ln a_c(T) = -\frac{z_c}{T} + \ln \sum_j e^{z_j / T}$（需配合温度缩放调整 LogSumExp 的数值稳定逻辑）。

### C.2 知识蒸馏中的温度交叉熵

在知识蒸馏（Hinton et al., 2015）中，学生模型的训练目标包含两部分：

1. **硬标签损失**（标准 CE）：$L_{\text{hard}} = -\ln P_\theta^{\text{student}}(c \mid \mathbf{x})$
2. **软标签损失**（温度化的 KL 散度）：用教师模型在高温 $T$ 下的概率分布作为目标

**软标签损失**：
$$L_{\text{soft}} = T^2 \cdot D_{KL}\left(P_{T}^{\text{teacher}} \parallel P_{T}^{\text{student}}\right)$$

乘以 $T^2$ 是因为温度缩放使得 Softmax 的梯度缩小 $1/T$ 倍，乘以 $T^2$ 恢复梯度的量级。

**教师分布的「暗知识」**：高温下教师模型输出的概率分布不仅包含「第一名」（正确答案），还包含了第二名、第三名等错误答案之间的相对关系——这些关系编码了类别间的相似性结构（如 $\ln P_{\text{teacher}}(\texttt{dog} \mid \dots) > \ln P_{\text{teacher}}(\texttt{car} \mid \dots)$，尽管两者都不如正确答案），学生模型从这些相对关系中学到了比 one-hot 标签更丰富的知识。

---

## 参考文献

1. Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
2. Savage, L. J. (1971). Elicitation of Personal Probabilities and Expectations. *Journal of the American Statistical Association*.
3. Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*.
4. Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica*.
5. Wald, A. (1949). Note on the Consistency of the Maximum Likelihood Estimate. *Annals of Mathematical Statistics*.
6. White, H. (1982). Maximum Likelihood Estimation of Misspecified Models. *Econometrica*.
7. Huber, P. J. (1967). The Behavior of Maximum Likelihood Estimates Under Nonstandard Conditions. *Proceedings of the Fifth Berkeley Symposium*.
8. Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*.
9. Szegedy, C. et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*.
10. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop*.

---

> 📖 **相关笔记**：
> - 工程实现与数值细节：[`ce_loss_engineering.md`](ce_loss_engineering.md) — 显存分析、Tensor Cores 对齐、分布式 LogSumExp
> - Q&A 风格知识点：[`ce_loss_notes.md`](ce_loss_notes.md) — `torch.gather()` 用法、Weight Tying 历史、双栖 API 设计
> - 手撕代码：[`lm_head.py`](lm_head.py) — LogSumExp CE Loss 的完整 PyTorch 实现与测试

---

*本笔记最后更新于 2026 年 6 月。*

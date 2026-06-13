# 大模型标准解码管线深度解析

> 相关代码：[basic_sampler.py](basic_sampler.py)（nn.Module 封装），[generate_next_token.py](generate_next_token.py)（函数式面试版）

---

## 一、LM Head 的几何本质

输入端的 Embedding 是 $O(1)$ 查表取第 42 行，输出端的 LM Head 则是**全词表相似度检索**——输出向量 `[1, 4096]` 与权重矩阵 `[4096, 128256]` 相乘，本质上在逐一计算输出向量与 128,256 个"锚点向量"的**内积（未归一化的余弦相似度）**。结果 `[1, 128256]` 就是 Logits。

拿到 Logits 后，两条分岔路：
- **训练**：走 CrossEntropyLoss，反向传播把输出向量往正确锚点方向掰
- **推理**：走采样策略（Softmax → 概率 → 选词），核心抉择是 argmax 还是带随机性的 Top-K/Top-P

---

## 二、为什么需要解码策略：长尾词的威胁

词表服从**齐夫定律**——少数高频词占据 80%+ 概率质量，90% 的词是长尾噪音（生僻字、BPE 副产物）。若不做截断，`multinomial` 有 ~20% 概率抽中乱码，模型随后强行圆谎导致逻辑崩溃。

Top-K 和 Top-P 的共同使命：**把低频噪音设为 `-inf`，将概率确定性还给正常逻辑。**

---

## 三、Top-K vs Top-P

| | Top-K | Top-P（核采样） |
|---|---|---|
| **逻辑** | 固定保留前 K 个 | 累加概率达到 P 就停 |
| **优点** | 简单可控 | 自适应上下文 |
| **缺点** | 僵硬——不论语境候选池固定 | 需全局 sort |

**核采样命名出处**：*The Curious Case of Neural Text Degeneration (2019)*。前几十个词的概率累加已达 0.9，它们是概率宇宙的"原子核"——数量极少却占据绝大部分概率质量，外围 3 万多词只是"电子云"。

**两个极端语境**：
- "The capital of France is " → Paris 概率 0.95，Top-P 核收缩到 1，Top-K 却死板拉进 49 个垃圾词
- "I went to the supermarket and bought some " → 数百合理选项，Top-P 核膨胀到 300，Top-K 却一刀切断了多样性

---

## 四、生产级解码管线

标准六阶段流水线：

```
[B, V] Logits
  ├─ 1. 贪婪特判 ──────── temperature < 1e-5 → argmax 直接返回
  ├─ 2. 温度缩放 ──────── logits / temperature
  ├─ 3. Top-K 截断 ────── 保留前 K 大，其余设为 -inf
  ├─ 4. Top-P 核采样 ──── sort → cumsum → 截断 → scatter_ 还原
  ├─ 5. Softmax 归一化 ── -inf → 概率 = 0.0
  └─ 6. multinomial ──── 掷骰子，返回 Token ID
```

```python
def generate_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    assert logits.dim() == 2, "logits 必须是 2D 张量 [Batch, Vocab]"

    # 1. 贪婪解码特判（短路优化）
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. 温度缩放（创建新张量，保护原始 logits）
    logits = logits / temperature

    # 3. Top-K 截断
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))   # 防越界
        top_values, _ = torch.topk(logits, top_k, dim=-1)
        kth_values = top_values[:, -1:]        # 保持 [B, 1] 维度以广播
        logits = logits.masked_fill(logits < kth_values, float('-inf'))

    # 4. Top-P 核采样
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits.float(), dim=-1)  # fp32 防精度溢出
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 右移：放过边界词
        sorted_indices_to_remove[..., 0] = False  # 至少保留 1 个词

        # scatter_: 从 sorted 顺序还原回 vocab 顺序
        indices_to_remove = torch.empty_like(sorted_indices_to_remove).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits.masked_fill_(indices_to_remove, float('-inf'))

    # 5-6. Softmax + 采样
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

### Mask 右移的核心逻辑

假设 `top_p=0.8`，排序后概率 `[0.5, 0.2, 0.15, 0.1, 0.05]`，cumsum = `[0.5, 0.7, 0.85, 0.95, 1.0]`。

不右移：`cumsum > 0.8` → `[F, F, T, T, T]`，只保留前两个词，累加仅 0.7（不足 0.8！）

右移：`mask[1:] = mask[:-1]; mask[0] = False` → `[F, F, F, T, T]`，第三项被"抢救"回来，累加 0.85 ≥ 0.8。

三个细节：`.clone()` 防止视图覆盖；索引 0 强设 `False` 保证至少 1 个候选；Softmax/cumsum 前转 `fp32` 防精度溢出。

### 为什么先 Top-K 再 Top-P？

Top-P 需要全词表 `torch.sort()`，词表 10 万维时开销大。先 Top-K 把大部分 Logits 截断为 `-inf`，推理引擎此时只对 K 个有效词做排序，大幅降低时间复杂度。**顺序不可逆。**

---

## 五、索引对齐：为什么必须保留 [B, V] 形状

**核心痛点**：`multinomial` 返回的是**坐标**（0~31999），而非概率值。Token ID 就是物理坐标——`logits[0, 1024]` = 词表第 1024 号词。

若缩小张量（Top-K 后只保留 50 个值），`multinomial` 返回 0~49 的局部索引，与真实 Token ID 脱钩。要么走 `gather` 多层映射（丑陋易错），要么保持 `[B, V]` 用 `-inf` 掩码。

**设计哲学**：Top-K 和 Top-P 不是"接力传递缩小后的张量"，而是**对同一块 `[B, V]` 画布先后涂抹**。无论多少过滤策略，画布维度不变，各策略彻底解耦互不干扰。

---

## 六、核心算子速查

### torch.scatter_

**本质**：根据"索引密码表"把打乱的数据物归原主。
- `dim=-1`（跨列）：同一行内左右交换——用于 Top-P 中将 sorted 顺序还原回 vocab 顺序
- `dim=0`（跨行）：锁定列，上下跳跃——用于 MoE 路由分发 Token 到不同 Expert

### torch.multinomial

**物理直觉**：GPU 按概率大小划分轮盘面积，然后扔飞镖。`argmax` 永远选最大的，`multinomial` 赋予创造力和多样性。

**底层算法**：逆变换采样——概率值 → cumsum 得 CDF → 生成均匀随机数 `r ∈ [0,1)` → 二分查找 `r` 落在哪个区间 → 返回该区间索引。

---

## 七、工业界踩坑

**Tensor Parallelism 下的 multinomial 蝴蝶效应**：多卡各自调用 `multinomial`，随机种子不一致 → GPU 0 抽 Token A、GPU 1 抽 Token B → 下一步隐藏状态完全错乱。

**解决**：强同步（只让 GPU 0 采样，NCCL 广播结果）或种子对齐（强制所有卡随机种子一致）。

---

## 八、面试追问速查

- **Weight Tying**：早期模型（GPT-2）的 Embedding 和 LM Head 共享权重（互为转置）。现代 LLM 不绑定，LM Head 独立学习。
- **Top-K 和 Top-P 顺序**：不可反过来，先 Top-K 再 Top-P 是为了缩小 sort 范围。
- **Mask 右移**：放过恰好把 cumsum 推过 P 阈值的边界词。索引 0 强设 `False` 保证至少 1 个候选。
- **[B, V] 不缩小**：`-inf` 掩码比缩小张量更解耦，过滤策略间互不干扰。

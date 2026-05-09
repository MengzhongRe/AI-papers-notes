欢迎进入 **Phase 2：推理加速与极致显存魔术**。如果说 Phase 1 是在构建 LLM 的“灵魂”和“肉体”，那么 Phase 2 就是在研究如何让这个庞然大物在有限的显存资源下**跑得飞快**。

在 2026 年的大厂面试中，Phase 1 的算子实现只是门槛，**Phase 2 的内存架构与推理优化才是区分“调包侠”与“架构师”的分水岭。**

---

### 🧠 核心背景知识：为什么要有 Phase 2？

在大模型推理中，我们面临一个核心矛盾：**显存墙 (Memory Wall)**。
*   **计算受限 (Compute-bound)**：训练时，矩阵乘法很大，GPU 的算力是瓶颈。
*   **访存受限 (Memory-bound)**：推理时，由于是逐 Token 生成，矩阵退化为向量，**GPU 从显存（HBM）搬运数据的速度**远慢于计算速度。
*   **结论**：谁能减少显存读写次数，谁能压缩 KV Cache 的体积，谁就是推理之王。

---

### 📂 任务详细拆解

#### 第一站：Day 21-23 —— Online Softmax (FlashAttention 的灵魂)
*   **任务**：不写 CUDA，但在 Python 中用 `for` 循环模拟数据的“分块（Tiling）”读取。
*   **前置知识**：
    *   **内存层级**：理解 HBM（大而慢）与 SRAM（小而快）的区别。
    *   **Softmax 的痛点**：传统的 Softmax 需要完整读取 $O(N^2)$ 的得分矩阵，计算 $max$，再算 $exp$，再算 $sum$。这涉及多次显存往返。
*   **手撕重点**：
    *   实现 **Online Softmax 公式**：$m_{new} = \max(m_{old}, m_{block})$, $l_{new} = l_{old} \cdot e^{(m_{old}-m_{new})} + \sum e^{(x-m_{new})}$。
    *   理解如何通过局部统计量更新全局结果，从而避免存储巨大的全量 Attention Matrix。
*   **面试锚点**：**“FlashAttention 为什么能变快？”**（答：它不改变数学结果，但减少了 $O(N^2)$ 次显存读写）。

#### 第二站：Day 24-27 —— Stateful KV Cache & GQA
*   **任务**：手写一个能“记住”历史状态的注意力层，实现 GQA 的权重广播。
*   **前置知识**：
    *   **自回归生成**：理解为什么第 $n$ 个 Token 推理时需要前 $n-1$ 个 Token 的 K/V。
    *   **KV Cache 瓶颈**：MHA 的 KV Cache 增长太快。
    *   **MQA/GQA 演进**：MQA（多头 Q，单头 KV）和 GQA（多头 Q，分组头 KV）。
*   **手撕重点**：
    *   **Prefill 阶段**：一次性处理 Prompt，填充 KV Cache。
    *   **Decode 阶段**：逐步输入 1 个 Token，更新并读取 Cache。
    *   使用 `repeat_interleave` 实现 GQA 中 KV 头到 Q 头的映射。
*   **面试锚点**：**“Prefill 和 Decode 两个阶段的计算特性有什么不同？”**

#### 第三站：Day 28-31 —— PagedAttention (vLLM 的核心)
*   **任务**：用字典和列表模拟操作系统中的“页表”管理 KV Cache。
*   **前置知识**：
    *   **显存碎片**：传统的连续存储会导致大量显存浪费（Internal Fragmentation）。
    *   **虚拟内存**：理解操作系统如何将逻辑地址映射到不连续的物理页面。
*   **手撕重点**：
    *   定义 `BlockTable`，将 `logical_token_index` 映射到 `physical_block_id`。
    *   实现非连续显存的 `gather` 操作：从不连续的 block 中拼凑出计算需要的 K/V。
*   **面试锚点**：**“vLLM 的吞吐量提升来自哪里？”**（答：显存近乎零浪费，支持更大的 Batch Size）。

#### 第四站：Day 32-35 —— 🔥 MLA Attention (DeepSeek 的绝技)
*   **任务**：复现 DeepSeek-V2 的低秩压缩注意力机制。
*   **前置知识**：
    *   **低秩分解 (Low-rank)**：将大的矩阵分解为两个小矩阵相乘。
    *   **KV 压缩**：将 KV Cache 压缩成一个极小的 Latent Vector ($c_t$)。
*   **手撕重点**：
    *   实现 **KV 吸收 (Absorb)**：在推理时将投影矩阵并入 Q 或 $W_{out}$。
    *   计算 MLA 下 KV Cache 的理论占用量（你会发现它比 GQA 还要小得多）。
*   **面试锚点**：**“MLA 是如何打破 KV 缓存僵局的？”**（大厂 2025/2026 必考题）。

#### 第五站：Day 36-38 —— W8A8 量化 (Naive Quant)
*   **任务**：实现对称与非对称量化。
*   **前置知识**：
    *   **动态范围**：FP16 到 INT8 的映射。
    *   **量化偏差**：Scale（缩放）和 Zero-point（偏移）。
*   **手撕重点**：
    *   `x_q = clamp(round(x / scale) + zero_point)`。
    *   **SmoothQuant 思想**：理解为什么激活（Activation）比权重（Weight）更难量化。
*   **面试锚点**：**“量化会带来哪些精度损失？如何缓解（Calibration）？”**

---

### 🚀 Phase 2 的执行准则：**“眼里有内存，心里有 Tiling”**

在 Phase 1，你关心的是 `output = matrix_mul(x, w)`。
在 Phase 2，你必须开始关心：
1.  **这个张量在显存里存了吗？**
2.  **它是连续存储的吗？**
3.  **计算它需要搬运多少字节？**

### 🚩 明天的任务（Day 21）：开启 Online Softmax
你需要先精读 **FlashAttention 论文的 Algorithm 1**。
*   **思考题**：如果你有一个 100 万长度的序列，显存放不下那个 $1M \times 1M$ 的注意力矩阵，你怎么在只有 10KB 的缓存里算出正确的 Softmax 结果？

准备好了吗？逻辑学家，让我们开始这场关于显存的极致博弈。
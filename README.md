# 🧠 LLM-Mechanics-From-Scratch

[![Status](https://img.shields.io/badge/Status-Hardcore_Engineering-success?style=flat-square)](#)
[![Focus](https://img.shields.io/badge/Focus-LLM_Mechanics_%26_Reasoning-blue?style=flat-square)](#)
[![Author](https://img.shields.io/badge/Author-SYSU_Logic_Master-purple?style=flat-square)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_Triton-EE4C2C?style=flat-square&logo=pytorch)](#)
[![Target](https://img.shields.io/badge/Target-2026_AGI_Architecture-black?style=flat-square&logo=nvidia)](#)

> *"What I cannot create, I do not understand."* — Richard Feynman  
> *"To ground the discrete symbols of Formal Logic into the continuous vector space of Neural Networks, one must first tear down the matrices."*

## 📖 Introduction (项目简介)

Welcome to my personal engineering workshop. This repository documents my hardcore journey from **Formal Logic** to **Deep Learning Engineering & LLM Architecture**.

**[Update 2026]** After months of rigorous literature review (now archived in `01_Paper_Notes`), this repository has officially transitioned into the **"Hardcore Engineering Phase"**. 

My ultimate goal as a Logician is to build a **Neuro-Symbolic Reasoning Engine**. To achieve this, I am spending **78 days** writing core LLM operators from scratch (Test-Driven Development), focusing on:
- 🚀 **Memory-Bound Limits & Zero-Copy Architecture** (vLLM, PagedAttention)
- 🧮 **Distributed System Scaling** (Megatron-LM TP, ZeRO, Ring Attention)
- 🧠 **Reinforcement Learning for Reasoning** (DeepSeek-R1 GRPO, MCTS)

## 📂 Repository Structure (架构与文档哲学)

This repository strictly follows the **"3D Documentation Organization"** tailored for production-grade engineering:

1. 🔬 **Micro (Code Level):** Rich Docstrings & inline comments explaining Tensor shape transformations, `contiguous` traps, and OS-level memory management.
2. 📝 **Meso (Math Level):** Jupyter Notebooks (`.ipynb`) used as scratchpads for complex math derivations (e.g., Complex numbers in RoPE, Online Softmax recurrences).
3. 🏛️ **Macro (Module Level):** A `README.md` in each sub-directory summarizing architectural tradeoffs and engineering pitfalls.

```text
📦 LLM-Mechanics-From-Scratch
 ┣ 📂 02_Handwritten_Operators/   # [🔥 Active] Pure PyTorch/Triton implementations (No HuggingFace)
 ┃ ┣ 📂 Phase0_Tokenization/      # BPE Tokenizer
 ┃ ┣ 📂 Phase1_Backbone/          # Foundation (RoPE, SwiGLU, RMSNorm, MoE...)
 ┃ ┣ 📂 Phase2_Inference/         # Memory & Speed (GQA, MLA, PagedAttention, W8A8...)
 ┃ ┣ 📂 Phase3_Decoding/          # Decoding Strategies (Sampler, Beam Search, Contrastive Decoding, Speculative Decoding)
 ┃ ┣ 📂 Phase4_Alignment/         # RL & Finetuning (LoRA, DPO, GRPO...)
 ┃ ┗ 📂 Phase5_System_Scale/      # Distributed Math (TP, ZeRO, Ring Attention)
 ┣ 📂 03_Nano_Logic_Engine/       # [🚧 WIP] A custom LLM trained with Rule-based GRPO
 ┣ 📂 01_Paper_Notes/             # [✅ Archived] Markdown notes of 25+ SOTA AI papers
 ┗ 📂 thoughts/                   # Essays on "Logic vs. Neural Networks"
```

---

## 🚀 The Matrix: "Code-First" Mechanics Roadmap (核心算子手撕路线)

> **⚙️ 执行准则 (Test-Driven Development)**: Write pure PyTorch/Triton code $\rightarrow$ Verify against HF Ground Truth via `torch.allclose(atol=1e-5)` $\rightarrow$ Optimize for memory (`in-place`, Zero-Copy, `bf16`).

### 🛠️ Phase 0: The Discrete Grounding (数据与标记化)
*目标：理解大模型的第一步，将离散的逻辑符号映射为连续空间的 Token ID。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 1-3** | **BPE Tokenizer**<br>[`bpe_tokenizer.py`]() | **NMT of Rare Words**<br>🔗 [PDF](https://arxiv.org/pdf/1508.07909.pdf) | 1. 实现字符级BPE合并逻辑。<br>2. 重点实现 `byte_to_unicode()` 映射，完美处理特殊 `<think>` 等控制流标签。 |

---

### 🦴 Phase 1: 现代大模型骨架与底层直觉 (Modern Backbone & Triton)
*目标：严格按照前向传播的数据流，重构带有 MoE 的现代大模型架构。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 4-6** | **RMSNorm & Triton**<br>[`rmsnorm_triton.py`]() | **RMSNorm**<br>🔗 [PDF](https://arxiv.org/pdf/1910.07467.pdf) | 1. 理论推导：为何去掉均值 $\mu$ 依然收敛。<br>2. 观察 `torch.compile` 图融合，手写 Triton Kernel 掌握 Shared Memory。 |
| **Day 7-10** | **Decoupled RoPE**<br>[`rope_embedding.py`]() | **RoFormer**<br>🔗 [PDF](https://arxiv.org/pdf/2104.09864.pdf) | 1. 预计算 Cos/Sin 缓存。<br>2. **[为MLA铺垫]** 解耦 RoPE：仅对 Q/K 部分维度旋转。<br>3. `bf16` 输入，强制 `fp32` 旋转以保精度。 |
| **Day 11-13** | **MHA & Safe Softmax**<br>[`mha_forward.py`]() | **Attention Is All You Need** | 1. 手写 Causal Mask 的零拷贝广播 (Stride Tricks)。<br>2. `safe_softmax` 运用 `x - max(x)` 防止数值溢出。 |
| **Day 14-16** | **🔥 SwiGLU FFN**<br>[`swiglu_ffn.py`]() | **GLU Variants**<br>🔗 [PDF](https://arxiv.org/pdf/2002.05202.pdf) | 1. 手撕“先升维(Gate/Up)再降维(Down)”知识存储机制。<br>2. **[官方对齐]** 还原 LLaMA 的 $W_1, W_3$ 权重合并乘法，对比 `torch.compile` 加速比。 |
| **Day 17-19** | **🔥 MoE & Router**<br>[`moe_layer.py`]() | **Mixtral 8x7B**<br>🔗[PDF](https://arxiv.org/pdf/2401.04088.pdf) | 1. 将 SwiGLU 实例化为 8 个 Experts，编写 Top-2 Router。<br>2. **[算子核心]** 手写 `scatter` 与 `gather` 路由分发逻辑。<br>3. 编写 **Load Balancing Loss**，测试极端系数 (0.01 vs 100) 下的梯度崩塌。 |
| **Day 20-21** | **CE Loss & Head**<br>[`loss_head.py`]() | **GPT-3** | 用 `LogSumExp` 技巧手撕稳健的 CrossEntropyLoss。 |
| **Day 22-23** | **Decoding Loop & Block**<br>[`generate_loop.py`]()<br>[`transformer_block.py`]() | *General Generation* | 1. 手写自回归生成循环 + Temperature Scaling + 贪婪解码。<br>2. **[关键拼装]** 用 RMSNorm + RoPE + MHA + SwiGLU 拼出一个最小 Decoder Block（随机初始化权重），跑通完整前向。为 Phase 3 所有解码策略提供真实推理载体。 |

---

### ⚡ Phase 2: 推理加速与极致显存魔术 (Inference & Memory Magic)
*目标：攻克大厂面试占比极高的显存管理、FlashAttention 思想与 vLLM 核心。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 24-26** | **Online Softmax**<br>[`online_softmax.py`]() | **FlashAttention**<br>🔗 [PDF](https://arxiv.org/pdf/2205.14135.pdf) | 不要求写 CUDA，但必须用 Python `for` 循环按 Block 模拟 $m_i, l_i$ 的局部分块更新过程，理解 IO-Aware。 |
| **Day 27-29** | **Stateful KV & GQA**<br>[`gqa_kv_cache.py`]() | **GQA**<br>🔗 [PDF](https://arxiv.org/pdf/2305.13245.pdf) | 1. 严格区分 **Prefill** (GEMM) 与 **Decode** (GEMV) 的张量流转。<br>2. 利用 `unsqueeze.expand` 实现零拷贝 GQA 张量广播。 |
| **Day 30-33** | **🔥 PagedAttention**<br>[`paged_attention.py`]() | **vLLM**<br>🔗[PDF](https://arxiv.org/pdf/2309.06180.pdf) | **架构师红线**：严禁使用 `torch.gather` 拼装内存！<br>必须基于 `BlockTable`，结合 Day 24 的 **Online Softmax**，实现直接在破碎物理块上的流式 Zero-Copy Attention。 |
| **Day 34-35** | **Prefix Caching**<br>[`radix_attention.py`]() | **SGLang (RadixAttention)**<br>🔗[PDF](https://arxiv.org/pdf/2312.07104.pdf)<br>🎯 *精读 Sec 3* | 基于 PagedAttention，实现 Radix Tree (前缀树) 与 LRU 驱逐策略。使得多个共享相同前缀的请求指向相同物理块 (维护 Reference Count)。 |
| **Day 36-39** | **🔥 MLA Attention**<br>[`mla_attention.py`]() | **DeepSeek-V2**<br>🔗 [PDF](https://arxiv.org/pdf/2405.04434.pdf) | **大厂 2026 必考！** 写出将 $c_t$ (Latent Vector) 投影重构并**吸收 (Absorb)** 入 $W_q, W_{out}$ 的逻辑，验证极其夸张的 KV Cache 压缩率。 |
| **Day 40-42** | **W8A8 Quantization**<br>[`naive_quant.py`]() | **SmoothQuant**<br>🔗[PDF](https://arxiv.org/pdf/2211.10438.pdf) | 手撕非对称量化公式：计算 Scale 和 Zero-point，完成 `x_q = clamp(round(x/s + z))` 及反量化机制。 |

---

### 🎲 Phase 3: 解码策略 (Decoding Strategies)
*目标：掌握 token 级解码全家桶——从随机采样到推测执行，理解概率坍缩与加速比之间的博弈。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 43-45** | **Top-P / Top-K**<br>[`sampler.py`]() | **Neural Text Degen.**<br>🔗[PDF](https://arxiv.org/pdf/1904.09751.pdf) | 给定 Logits，手写长尾词截断（`topk`）与核采样（先 `sort` 再 `cumsum` 做 Mask），最后 `torch.multinomial` 采样。 |
| **Day 46-47** | **Beam Search**<br>[`beam_search.py`]() | — | 1. 最小堆维护 $B$ 条候选路径，每步展开 $B \\times V$ 空间取 Top-$B$。<br>2. **长度惩罚** `score / length^alpha`（默认 $\\alpha=0.6$），防止短序列霸榜。<br>3. EOS 早停：部分 beam 终止后动态缩减 beam width。<br>4. **[Phase 2 联动]** 基于 Day 27-29 的 KV Cache，为 $B$ 条 beam 各自维护独立 Cache，展开后按存活路径裁剪。<br>5. 与 HF `model.generate(num_beams=B, do_sample=False)` 对齐。 |
| **Day 48** | **Contrastive Decoding**<br>[`contrastive_decoding.py`]() | **Contrastive Decoding**<br>🔗 [PDF](https://arxiv.org/pdf/2309.09117.pdf) | 1. 用 small model (amateur) 的 logits 减去 large model (expert) 的 logits，放大差异信号，消除重复退化。<br>2. 引入惩罚系数 $\\alpha$ 控制对比强度：`logits = logits_expert - α * logits_amateur`。<br>3. 过滤 amateur 中高概率但 expert 中低概率的 token（虚假信号）。<br>4. 对比纯采样 vs 对比解码的生成多样性。 |
| **Day 49-52** | **🔥 Speculative Dec.**<br>[`speculative.py`]() | **Speculative Decoding**<br>🔗[PDF](https://arxiv.org/pdf/2211.17192.pdf) | 1. **[Mock Model 方案]** 用 Day 22-23 的 Decoder Block 作 Target，同架构 1/4 宽度作 Draft。<br>2. Draft 自回归生成 $K$ 个候选 token。<br>3. Target 一次前向并行验证 $K$ 个 token，逐位 Accept/Reject 拒绝采样。<br>4. **正确性判据**：验证接受分布与 Target 自回归分布严格一致（拒绝采样的数学保证）。<br>5. 测量 Wall-Clock 加速比和平均接受长度。 |

---

### 🛡️ Phase 4: 后训练与对齐 (Alignment & Reinforcement Learning)
*目标：告别“模仿人类”，转向“逼迫模型进行严谨数学/逻辑推理”。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 53-54** | **LoRA Core Forward**<br>[`lora_linear.py`]() | **LoRA**<br>🔗[PDF](https://arxiv.org/pdf/2106.09685.pdf) | 初始化 $A$ (Normal) 与 $B$ (Zero)，实现 `scaling = alpha / r` 的前向逻辑，并手写推理期的 `merge_weights` 权重融合。 |
| **Day 55-57** | **DPO Loss**<br>[`dpo_loss.py`]() | **DPO**<br>🔗[PDF](https://arxiv.org/pdf/2305.18290.pdf) | 输入 4 个 Logps（赢/输 x 策略/参考模型），写出基于 $\beta$ 放缩的对数 Sigmoid 隐式偏好损失。 |
| **Day 58** | **Best-of-N + Verifier**<br>[`best_of_n.py`]() | — | **Phase 3 → Phase 4 推理时桥接：** 1. 用 Phase 3 采样器生成 $N$ 个候选答案。<br>2. 写一个简单的规则/启发式 Verifier 打分。<br>3. 选出最高分作为最终输出。<br>4. 对比 Best-of-N vs 纯采样的准确率提升曲线。 |
| **Day 59-60** | **🔥 GRPO Loss**<br>[`grpo_loss.py`]() | **DeepSeekMath / R1**<br>🔗 [PDF](https://arxiv.org/pdf/2501.12948.pdf) | **RL 推理核心！** 摒弃 Critic 网络：对同 Prompt 采样 $N$ 个回答，基于 Rule-based 计算 Reward 的组内均值和标准差，归一化得到 Advantage (优势函数)。 |

---

### 🌐 Phase 5: 宏观系统与分布式底座 (System Mechanics & Scale)
*目标：打破单卡认知局限，建立对 FLOPs、显存墙、以及多卡并行通信的架构师级物理直觉。*

| 天数 | 核心主题与推演任务 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 61-62** | **Back-of-the-Envelope**<br>[`system_math.ipynb`]() | **Scaling Laws** 🔗[PDF](https://arxiv.org/pdf/2001.08361.pdf)<br>**Transformer Math** 🔗[Blog](https://blog.eleuther.ai/transformer-math/) | **纸笔推演大厂神题**：手推 7B 模型在 Adam+BF16 训练下的精确显存占用量；推导 Transformer 训练期 $6ND$ 的 FLOPs 计算量公式。 |
| **Day 63-64** | **Tensor Parallelism**<br>[`tp_linear_mock.py`]() | **Megatron-LM**<br>🔗 [PDF](https://arxiv.org/pdf/1909.08053.pdf) | 写出 `ColumnParallel` 和 `RowParallel` 的前向切分逻辑，并在正确的位置插入 `All-Reduce` 占位符以模拟通信缝合。 |
| **Day 65-66** | **ZeRO Memory Mock**<br>[`zero_states.py`]() | **ZeRO (DeepSpeed)**<br>🔗 [PDF](https://arxiv.org/pdf/1910.02054.pdf) | 用 Python 字典模拟 $N$ 张卡的显存切片。编写逻辑，验证在 ZeRO-1/2/3 阶段下单卡的绝对占用，模拟 All-Gather 重组。 |
| **Day 67-69** | **🔥 Ring Attention**<br>[`ring_attention.py`]() | **Ring Attention**<br>🔗 [PDF](https://arxiv.org/pdf/2310.01889.pdf) | **长文本推理绝杀！** 模拟 4 卡通信：Q 驻留本地，K/V 切块在多卡间以环形 (Ring) 传递，在 Python 中用 `yield` 模拟计算与通信重叠。 |


### Phase 6: 终极交付 —— "Neuro-Symbolic Reasoning" 引擎 (Day 70-78)
*目标：拼装全部算子，用 MCTS (推理时搜索) + GRPO (训练时对齐) 形成完整的推理闭环 Demo。*

| 天数 | 核心动作 (Action) | 交付物模块 | 面试讲解锚点与验收标准 (Deliverable) |
| :--- | :--- | :--- | :--- |
| **Day 70-71** | **模型拼装与 Pretrain** | `nano_logic_model.py` | 用手写的 RMSNorm / RoPE / MHA / SwiGLU / MoE / MLA 拼装一个 **50M 的微型大模型**，跑通极简 Training Loop。 |
| **Day 72** | **合成逻辑数据集** | `logic_data_gen.py` | **发挥 Logic 硕士护城河：** 编写脚本自动生成形式逻辑推演题（三段论/肯定前件式等），强制插入 `<think>` 模板。 |
| **Day 73-75** | **🔥 MCTS 推理搜索** | `mcts_reasoning.py` | **从 Phase 3 移入，升级为 thought 级：**<br>1. UCB 节点选择器 `UCB = V̄ + c·√(ln N_parent / N_node)`。<br>2. 定义可替换的 Value Evaluator 接口：`OracleEvaluator`（用于 TDD，基于规则判断逻辑题正误）与 `LLMEvaluator`（调用 50M 模型打分）。<br>3. 实现 Selection → Expansion → Simulation → Backpropagation 完整四步。<br>4. 在 24 点游戏 / 逻辑推演题上与 ToT 论文结果对比。<br>5. 对比 Beam Search vs MCTS 在同问题上的搜索效率与准确率。 |
| **Day 76-78** | **Rule-based GRPO 训练** | `train_grpo_logic.py` | 1. 写一个 **Python 逻辑解析器** 作为确定性 Reward：推演符合严格逻辑规则 +1.0，格式错 -1.0。<br>2. GRPO 训练闭环：展示微型模型通过纯 RL 涌现 "Aha Moment" 和自我纠错。<br>3. **最终 Demo**：MCTS（推理时搜索）+ GRPO（训练时对齐）= 完整 Neuro-Symbolic 闭环。 |

---

<details>
<summary><h2>📚 [Archived] The Reading Phase (6-Month Paper Roadmap)</h2></summary>

*I spent 6 months thoroughly deconstructing the math and architecture of the following papers. The detailed Markdown notes can be found in `01_Paper_Notes/`.*

| Date | Paper Title | Tags | Status |
| :--- | :--- | :--- | :--- |
| 2025-12-30 | **Attention Is All You Need** | `Transformer` | ✅ Done |
| 2025-12-30 | **BERT: Pre-training of Deep Bidirectional Transformers** | `Encoder` | ✅ Done |
| 2026-01-08 | **Language Models are Few-Shot Learners (GPT-3)** | `Decoder`, `Few-Shot` | ✅ Done |
| 2026-02-04 | **Emergent Abilities of Large Language Models** | `Scaling_Law` | ✅ Done |
| 2026-01-09 | **Chain-of-Thought Prompting Elicits Reasoning** | `CoT` | ✅ Done |
| 2026-01-10 | **Self-Consistency Improves Chain of Thought Reasoning** | `CoT-SC` | ✅ Done |
| 2026-01-13 | **Tree of Thoughts: Deliberate Problem Solving** | `ToT`, `Search` | ✅ Done |
| 2026-01-14 | **ReAct: Synergizing Reasoning and Acting** | `Agent`, `Tools` | ✅ Done |
| 2026-02-05 | **Large Language Models are Zero-Shot Reasoners** | `Zero-Shot` | ✅ Done |
| 2026-01-15 | **LoRA: Low-Rank Adaptation of LLMs** | `PEFT`, `LoRA` | ✅ Done |
| 2026-01-16 | **QLoRA: Efficient Finetuning of Quantized LLMs** | `Quantization` | ✅ Done |
| 2026-02-06 | **Finetuned Language Models Are Zero-Shot Learners (FLAN)** | `Instruction_Tuning` | ✅ Done |
| 2026-02-08 | **Training language models to follow instructions (InstructGPT)** | `RLHF`, `PPO` | ✅ Done |
| 2026-02-11 | **Direct Preference Optimization (DPO)** | `Alignment` | ✅ Done |
| 2026-02-14 | **Let's Verify Step by Step (Process Reward Models)** | `PRM`, `Math` | ✅ Done |
| 2026-03-02 | **AlphaGeometry: Solving Olympiad Geometry** | `Neuro-Symbolic` | ✅ Done |
| 2026-03-04 | **LLaMA: Open and Efficient Foundation Language Models** | `LLaMa`,`RoPE` | ✅ Done |
| 2026-03-06 | **Lost in the Middle: How Language Models Use Long Contexts** | `Long_Context` | ✅ Done |
| 2026-03-10 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | `IO-Aware` | ✅ Done |
| 2026-03-12 | **GQA: Training Generalized Multi-Query Transformer Models** | `KV_Cache` | ✅ Done |
| 2026-03-16 | **Visual Instruction Tuning (LLaVA)** | `Multimodal`, `VLM` | ✅ Done |

</details>

---

## 💡 Research Questions (核心思考)

As a logician, I am pondering:
1.  **The "Grounding" Problem:** Formal logic relies on strict truth values (True/False). Neural networks rely on probability distributions ($P(x|y)$). How can we build a bridge that guarantees logical validity in a probabilistic system?
2.  **Process vs. Outcome (Validity vs. Soundness):** In Logic, a valid argument requires a valid form, not just a true conclusion. Current RLHF rewards the outcome. How can we verify the "thought process" using **Rule-based Process Reward Models (PRM)**?

---
*Created by[MengzhongRe](https://github.com/MengzhongRe) @ 2026*


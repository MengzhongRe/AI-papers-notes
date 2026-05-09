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

My ultimate goal as a Logician is to build a **Neuro-Symbolic Reasoning Engine**. To achieve this, I am spending **70 days** writing core LLM operators from scratch (Test-Driven Development), focusing on:
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
 ┃ ┣ 📂 Phase3_Decoding/          # Search & Sampling (Speculative Decoding, MCTS...)
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
| **Day 22** | **Decoding Loop**<br>[`generate_loop.py`]() | *General Generation* | 1. 手写自回归生成循环。<br>2. 实现 Temperature Scaling、贪婪解码逻辑。 |
| **Day 23** | **Inverted Dropout**<br>[`inverted_dropout.py`]() | *Legacy Concept* | *注：现代 LLM 预训练已极少使用 Dropout，保留此算子仅为强化底层张量与正则化直觉。* |

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

### 🎲 Phase 3: 解码、推测与树搜索 (Decoding & Speculative Execution)
*目标：掌握大厂最省机器算力、提升逻辑推演能力的解码黑科技。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 43-45** | **Top-P / Top-K**<br>[`sampler.py`]() | **Neural Text Degen.**<br>🔗[PDF](https://arxiv.org/pdf/1904.09751.pdf) | 给定 Logits，手写长尾词截断（`topk`）与核采样（先 `sort` 再 `cumsum` 做 Mask），最后 `torch.multinomial` 采样。 |
| **Day 46-49** | **Beam Search & MCTS**<br>[`beam_mcts.py`]() | **Tree of Thoughts**<br>🔗 [PDF](https://arxiv.org/pdf/2305.10601.pdf) | 1. 维护大小为 $B$ 的堆进行束搜索。<br>2. **[o1前置]** 基于 UCB (Upper Confidence Bound) 写一个启发式节点选择器。 |
| **Day 50-53** | **🔥 Speculative Dec.**<br>[`speculative.py`]() | **Speculative Decoding**<br>🔗[PDF](https://arxiv.org/pdf/2211.17192.pdf) | 初始化 Draft 和 Target 模型。实现 Draft 生成 $K$ 个 token，Target 一次前向后进行 **Accept/Reject 并行拒绝采样** 的概率纠正。 |

---

### 🛡️ Phase 4: 后训练与对齐 (Alignment & Reinforcement Learning)
*目标：告别“模仿人类”，转向“逼迫模型进行严谨数学/逻辑推理”。*

| 天数 | 核心主题与手撕代码 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 54-55** | **LoRA Core Forward**<br>[`lora_linear.py`]() | **LoRA**<br>🔗[PDF](https://arxiv.org/pdf/2106.09685.pdf) | 初始化 $A$ (Normal) 与 $B$ (Zero)，实现 `scaling = alpha / r` 的前向逻辑，并手写推理期的 `merge_weights` 权重融合。 |
| **Day 56-58** | **DPO Loss**<br>[`dpo_loss.py`]() | **DPO**<br>🔗[PDF](https://arxiv.org/pdf/2305.18290.pdf) | 输入 4 个 Logps（赢/输 x 策略/参考模型），写出基于 $\beta$ 放缩的对数 Sigmoid 隐式偏好损失。 |
| **Day 59-61** | **🔥 GRPO Loss**<br>[`grpo_loss.py`]() | **DeepSeekMath / R1**<br>🔗 [PDF](https://arxiv.org/pdf/2501.12948.pdf) | **RL 推理核心！** 摒弃 Critic 网络：对同 Prompt 采样 $N$ 个回答，基于 Rule-based 计算 Reward 的组内均值和标准差，归一化得到 Advantage (优势函数)。 |

---

### 🌐 Phase 5: 宏观系统与分布式底座 (System Mechanics & Scale)
*目标：打破单卡认知局限，建立对 FLOPs、显存墙、以及多卡并行通信的架构师级物理直觉。*

| 天数 | 核心主题与推演任务 | 核心论文 / 直达链接 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 62-63** | **Back-of-the-Envelope**<br>[`system_math.ipynb`]() | **Scaling Laws** 🔗[PDF](https://arxiv.org/pdf/2001.08361.pdf)<br>**Transformer Math** 🔗[Blog](https://blog.eleuther.ai/transformer-math/) | **纸笔推演大厂神题**：手推 7B 模型在 Adam+BF16 训练下的精确显存占用量；推导 Transformer 训练期 $6ND$ 的 FLOPs 计算量公式。 |
| **Day 64-65** | **Tensor Parallelism**<br>[`tp_linear_mock.py`]() | **Megatron-LM**<br>🔗 [PDF](https://arxiv.org/pdf/1909.08053.pdf) | 写出 `ColumnParallel` 和 `RowParallel` 的前向切分逻辑，并在正确的位置插入 `All-Reduce` 占位符以模拟通信缝合。 |
| **Day 66-67** | **ZeRO Memory Mock**<br>[`zero_states.py`]() | **ZeRO (DeepSpeed)**<br>🔗 [PDF](https://arxiv.org/pdf/1910.02054.pdf) | 用 Python 字典模拟 $N$ 张卡的显存切片。编写逻辑，验证在 ZeRO-1/2/3 阶段下单卡的绝对占用，模拟 All-Gather 重组。 |
| **Day 68-70** | **🔥 Ring Attention**<br>[`ring_attention.py`]() | **Ring Attention**<br>🔗 [PDF](https://arxiv.org/pdf/2310.01889.pdf) | **长文本推理绝杀！** 模拟 4 卡通信：Q 驻留本地，K/V 切块在多卡间以环形 (Ring) 传递，在 Python 中用 `yield` 模拟计算与通信重叠。 |


### Phase 6: 终极交付 —— "Neuro-Symbolic Reasoning" 引擎
*目标：拼装前 70 天的算子，结合逻辑学背景，打造极具辨识度的 RL-for-Reasoning 闭环 Demo。*

| 天数 | 核心动作 (Action) | 交付物模块 | 面试讲解锚点与验收标准 (Deliverable) |
| :--- | :--- | :--- | :--- |
| **Day 61-62** | **模型拼装与 Pretrain** | `nano_logic_model.py` | 用手写的 RoPE/RMSNorm/MoE/MLA 拼装一个 **50M 的微型大模型**，跑通极简 Training Loop。 |
| **Day 63** | **合成逻辑数据集** | `logic_data_gen.py` | **发挥 Logic 硕士护城河：** 编写脚本自动生成形式逻辑推演题（三段论/肯定前件式等），强制插入 `<think>` 模板。 |
| **Day 64-65** | **Rule-based GRPO** | `train_grpo_logic.py` | 1. 写一个 **Python 逻辑解析器** 作为确定性 Reward：推演符合严格逻辑规则 +1.0，格式错 -1.0。<br>2. 运行 GRPO 训练，展示微型模型如何通过纯 RL 涌现出 "Aha Moment" 和自我纠错能力。 |

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


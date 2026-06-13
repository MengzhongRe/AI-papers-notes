# 01_Paper_Notes —— 论文笔记

本目录收录了 LLM 全链路关键论文的精读笔记。为便于检索和体系化学习，所有笔记按**六大类别**组织，覆盖从经典基础架构到前沿推理智能体的完整知识谱系。

## 为什么是这六个类别？

这六类的划分逻辑是沿着 **"模型从哪里来 -> 模型如何更高效 -> 模型如何听话 -> 模型如何思考 -> 模型如何对齐人类 -> 模型如何符号推理"** 的认知递进线：

1. **Foundation** — 一切的起点：Transformer 架构的诞生与 Scaling Law 的发现。
2. **Modern_Architecture** — 工程化落地：让大模型真正能"跑起来"且"跑得快"的关键架构创新。
3. **Fine_tuning** — 高效适配：如何用极少资源让通用大模型适配特定任务。
4. **Reasoning_Agent** — 思维链与智能体：让模型学会"一步步想"并调用外部工具。
5. **Alignment_Human_Preference** — 对齐人类偏好：从"能说话"到"说人话、说对话"的 RLHF 技术栈。
6. **Neuro_Symbolic** — 神经符号推理：融合深度学习与符号逻辑，本仓库的终极目标方向。

## 论文清单

| 类别 | 论文 |
| :--- | :--- |
| **Foundation**<br>基础模型 | `2017_Attention.md` — Attention Is All You Need<br>`2018_BERT.md` — BERT: Pre-training of Deep Bidirectional Transformers<br>`2020_GPT3.md` — Language Models are Few-Shot Learners<br>`2022_Emergence.md` — Emergent Abilities of Large Language Models |
| **Modern_Architecture**<br>现代架构 | `2022_Flash_Attention.md` — FlashAttention: Fast and Memory-Efficient Exact Attention<br>`2023_GQA.md` — GQA: Training Generalized Multi-Query Transformers<br>`2023_LLaVA.md` — Visual Instruction Tuning<br>`2023_LLama.md` — LLaMA: Open and Efficient Foundation Language Models<br>`2023_Lost_In_The_Middle.md` — Lost in the Middle: How Language Models Use Long Contexts<br>`2023_Mamba.md` — Mamba: Linear-Time Sequence Modeling with Selective State Spaces<br>`2024_SMoE.md` — Mixtral of Experts |
| **Fine_tuning**<br>微调技术 | `2021_LoRA.md` — LoRA: Low-Rank Adaptation of Large Language Models<br>`2022_FLAN_Instruct_Tuning.md` — Finetuned Language Models Are Zero-Shot Learners<br>`2023_QLoRA.md` — QLoRA: Efficient Finetuning of Quantized LLMs<br>`2024_LongLoRA.md` — LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models |
| **Reasoning_Agent**<br>推理与智能体 | `2022_CoT.md` — Chain-of-Thought Prompting Elicits Reasoning<br>`2022_Zero_Shot_Learners.md` — Large Language Models are Zero-Shot Learners<br>`2023_ReAct.md` — ReAct: Synergizing Reasoning and Acting<br>`2023_Self_Consistency.md` — Self-Consistency Improves Chain of Thought Reasoning<br>`2023_Tree_of_Thoughts.md` — Tree of Thoughts: Deliberate Problem Solving |
| **Alignment_Human_Preference**<br>对齐与人类偏好 | `2021_Training_Verifiers.md` — Training Verifiers to Solve Math Word Problems<br>`2022_InstructGPT.md` — Training Language Models to Follow Instructions<br>`2022_RLAIF.md` — Constitutional AI: Harmlessness from AI Feedback<br>`2023_DPO.md` — Direct Preference Optimization |
| **Neuro_Symbolic**<br>神经符号推理 | `2022_Least_To_Most.md` — Least-to-Most Prompting Enables Complex Reasoning<br>`2022_MaieuticPrompting.md` — Maieutic Prompting: Logically Consistent Reasoning<br>`2022_PAL.md` — PAL: Program-Aided Language Models<br>`2023_Verify_Step_By_Step.md` — Let's Verify Step by Step<br>`2024_AlphaGeometry.md` — Solving Olympiad Geometry Without Human Demonstrations |

## 命名规范

所有笔记文件遵循统一的命名格式：

```
YYYY_Topic.md
```

- **YYYY** — 论文发表年份（4 位数字）
- **Topic** — 论文核心主题的英文简称（下划线分隔）
- 例如：`2017_Attention.md`、`2023_LLaVA.md`、`2024_SMoE.md`

这样命名的好处是按文件名排序即可自动获得按类别内的**时间线排序**，方便追溯技术演进脉络。

# Transformer Math 101 — EleutherAI 博客阅读笔记

> 原文：[Transformer Math 101](https://blog.eleuther.ai/transformer-math/) · Quentin Anthony, Stella Biderman, Hailey Schoelkopf · 2023 年 4 月 18 日
> 定位：忠实记录原文所有公式、表格和工程经验。每节前半部分是对原文的严格复现，末尾附结构化总结。

---

## 目录

- [1. 引言](#1-引言)
- [2. 计算需求 (Compute Requirements)](#2-计算需求-compute-requirements)
- [3. 参数量与数据集大小的权衡](#3-参数量与数据集大小的权衡)
- [4. 计算成本的工程经验](#4-计算成本的工程经验)
- [5. 显存需求 (Memory Requirements)](#5-显存需求-memory-requirements)
  - [5.1 推理 (Inference)](#51-推理-inference)
  - [5.2 训练 (Training)](#52-训练-training)
  - [5.3 模型参数](#53-模型参数)
  - [5.4 优化器状态](#54-优化器状态)
  - [5.5 梯度](#55-梯度)
  - [5.6 激活值与批次大小](#56-激活值与批次大小)
  - [5.7 训练总显存](#57-训练总显存)
- [6. 分布式训练](#6-分布式训练)
  - [6.1 分片优化器 (Sharded Optimizers)](#61-分片优化器-shattered-optimizers)
  - [6.2 3D 并行](#62-3d-并行)
  - [6.3 分片优化器 + 3D 并行](#63-分片优化器--3d-并行)
- [7. 结论](#7-结论)
- [附录：作者的结构化总结与知识整合](#附录作者的结构化总结与知识整合)

---

## 1. 引言

**原文要点**：关于 transformer 语言模型的很多基础而重要的信息可以通过简单的计算得出。不幸的是，这些计算公式在 NLP 社区中并未被广泛了解。本文档的目的就是收集这些公式，连同它们的来源和重要性。

> **说明**：本文主要关注**训练成本**——这由 GPU 显存（VRAM）主导。推理成本的类比讨论（关注延迟）见 [Kipply 的博客](https://kipp.ly/transformer-inference-arithmetic/)。

---

## 2. 计算需求 (Compute Requirements)

### 核心公式

训练 transformer 模型所需计算量的基本方程：

$$C \approx \tau T = 6PD$$

其中：

| 符号 | 含义 | 单位 |
|------|------|------|
| $C$ | 训练所需的总计算量 | 总 FLOPs |
| $C_{forward}$ | 前向传播计算量 | $\approx 2PD$ FLOPs |
| $C_{backward}$ | 反向传播计算量 | $\approx 4PD$ FLOPs |
| $\tau$ | 硬件聚合吞吐量 | FLOPs/s，$\tau = (\text{GPU 数量}) \times (\text{单 GPU 实际 FLOPs})$ |
| $T$ | 训练耗时 | 秒 |
| $P$ | 模型参数量 | — |
| $D$ | 数据集大小 | tokens |

这些方程由 [OpenAI 的 scaling laws 论文](https://arxiv.org/abs/2001.08361) 和 [DeepMind 的 scaling laws 论文](https://arxiv.org/abs/2203.15556) 提出并实验验证。

### 关于 $C$ 的单位

原文特别讨论了 $C$ 可以用多种单位衡量：

- **FLOP-秒**：$\text{FLOPs/s} \times \text{秒}$
- **GPU-小时**：$\text{GPU 数量} \times \text{小时}$
- **PetaFLOP-天**：Scaling laws 论文常用此单位。$1 \text{ PetaFLOP-day} = 10^{15} \times 24 \times 3600$ 总浮点运算

### 理论 FLOPs vs 实际 FLOPs

原文强调了一个重要区分：GPU 白皮书通常宣传的是**理论 FLOPs**，但这些在实践中永远达不到（尤其是在分布式设置中）。后面「计算成本」部分会报告一些常见的**实际 FLOPs** 值。

### 参数 vs 数据量权衡

原文讨论了「计算最优」（compute optimal）语言模型的概念。通常被称为 「Chinchilla scaling laws」，计算最优模型满足近似关系：

$$D = 20P$$

这在一个非常特定的意义下是「最优」的：在「1000 个 GPU 跑 1 小时」和「1 个 GPU 跑 1000 小时」成本相同的假设下，如果你的目标是用最小的 GPU 小时成本最大化性能，就应该使用上述方程。

**原文的重要建议**：

> **我们不建议用少于 200B tokens 来训练 LLM。** 尽管这对很多模型来说是「Chinchilla 最优」的，但得到的模型通常相当差。对于几乎所有应用，我们建议先确定可接受的推理成本，然后在该推理成本约束下训练尽可能大的模型、用尽可能多的 tokens。

---

## 3. 计算成本的工程经验

原文给出了以下具体的工程数字和经验法则：

### 实际 FLOPs 参考值

| 配置 | 实际 TFLOP/s/A100 |
|------|-------------------|
| GPT-NeoX（标准 attention） | 150 |
| GPT-NeoX（Flash Attention） | 180 |
| Megatron-DS（文献报告范围） | 137 - 163 |

### 经验法则

> **一般经验法则：你应该始终能在大约 120 TFLOP/s/A100 的水平上运行。如果低于 115 TFLOP/s/A100，你的模型或硬件配置可能有问题。**

### 数据并行的扩展性

- 使用高质量互联（如 InfiniBand），可以在数据并行维度上实现**线性或亚线性扩展**（增加数据并行度应能线性提升总吞吐量）
- 原文附了一张使用 GPT-NeoX 在 Oak Ridge Summit 超算上测试的图（V100 on x-axis，但原文中大部分数值例子用 A100）

---

## 4. 显存需求 (Memory Requirements)

> Transformer 通常用参数数量来描述其「大小」。但确定一个模型能否在给定的计算资源上运行时，需要知道模型占用的**字节数**。

### 4.1 推理 (Inference)

#### 模型权重

大多数 Transformer 以**混合精度**训练（fp16 + fp32 或 bf16 + fp32）。推理时可以将模型从 fp32 转换为 fp16 甚至 int8，不会遭受显著的性能损失：

| 精度 | 每参数字节数 |
|------|------------|
| int8 | 1 byte/param |
| fp16 / bf16 | 2 bytes/param |
| fp32 | 4 bytes/param |

#### 推理总显存

原文指出除了模型权重外，前向传播过程中还有少量额外开销（≤ 20%）。因此：

$$\text{Total Memory}_{Inference} \approx 1.2 \times \text{Model Memory}$$

这个开销通常对「最大能放多大模型」的判断影响不大。

### 4.2 训练 (Training)

训练需要的显存远超推理。除了模型参数本身，训练还需要存储**优化器状态**和**梯度**。

### 4.3 模型参数（训练时）

| 精度 | 每参数字节数 |
|------|------------|
| 纯 fp32 | 4 bytes/param |
| 纯 fp16 | 2 bytes/param |
| 混合精度（fp16/bf16 + fp32） | 2 bytes/param (fp16 副本) + 4 bytes/param (fp32 副本，计入优化器状态) |

**原文说明**：混合精度需要同时在内存中存储 fp16/bf16 和 fp32 版本的模型。fp32 副本（4 bytes/param）被计入优化器状态。

混合精度被广泛使用的原因：
1. fp32 训练稳定，但显存开销高，且不利用 NVIDIA GPU Tensor Core
2. fp16 训练快，但难以收敛
3. 混合精度结合了两者优势——利用 Tensor Core 加速同时保持 fp32 的数值稳定性

### 4.4 优化器状态

Adam 很强大，但显存效率极低。除了模型参数和梯度外，还需要额外保存三份梯度参数：

| 优化器 | 每参数字节数 | 拆解 |
|--------|------------|------|
| **Vanilla AdamW** | **12 bytes/param** | fp32 参数副本: 4 + Momentum: 4 + Variance: 4 |
| 8-bit Adam（如 bitsandbytes）| 6 bytes/param | fp32 参数副本: 4 + Momentum: 1 + Variance: 1 |
| SGD with momentum | 8 bytes/param | fp32 参数副本: 4 + Momentum: 4 |

### 4.5 梯度

| 精度 | 每参数字节数 |
|------|------------|
| fp32 | 4 bytes/param |
| fp16 | 2 bytes/param |

**原文说明**：梯度数据类型通常与模型数据类型匹配。在 fp16 混合精度训练中，梯度通常以 fp16 存储。

### 4.6 激活值与批次大小

现代 GPU 训练 LLM 时通常是**显存瓶颈**而非 FLOPs 瓶颈。因此 **activation recomputation / checkpointing** 是极受欢迎的技术——用额外的计算成本换更低的显存成本。

基本原理：不保存所有层的激活值，而是选择性地清除某些层的激活值，在反向传播需要时重新计算。

原文给出了 Megatron 的选择性重计算方案的图示（红色虚线 = A100-80GB 显存容量，"present work" = 应用选择性重计算后的显存需求）。

原文给出激活值显存的三个公式（假定 activations 以 fp16 存储，无序列并行）：

**无重计算 (No Recomputation)**：

$$M_{activations} = sbhL\left(10 + \frac{24}{t} + 5 \cdot \frac{a \cdot s}{h \cdot t}\right) \text{ bytes}$$

**选择性重计算 (Selective Recomputation)**：

$$M_{activations} = sbhL\left(10 + \frac{24}{t}\right) \text{ bytes}$$

**全重计算 (Full Recomputation)**：

$$M_{activations} = 2 \cdot sbhL \text{ bytes}$$

其中：

| 符号 | 含义 |
|------|------|
| $s$ | 序列长度 (sequence length)，单位 tokens |
| $b$ | 每 GPU 的 batch size |
| $h$ | Transformer 层的 hidden size |
| $L$ | Transformer 层数 |
| $a$ | attention heads 数量 |
| $t$ | tensor parallelism 度数（不使用 TP 则为 1） |

重计算带来的额外前向计算量：

$$2PD \leq C_{forward} \leq 4PD$$

### 4.7 训练总显存

$$\text{Total Memory}_{Training} = \text{Model Memory} + \text{Optimiser Memory} + \text{Activation Memory} + \text{Gradient Memory}$$

---

## 5. 分布式训练

### 5.1 分片优化器 (Sharded Optimizers)

优化器的巨大显存开销是 ZeRO 和 FSDP 等分片优化器的主要动机。分片策略将优化器开销除以 GPU 数量——这解释了为什么同样的模型配置在大规模下能跑、小规模下 OOM。

原文给出了 ZeRO paper 中的图示，说明 $P_{os}$（ZeRO-1）、$P_{os+g}$（ZeRO-2）、$P_{os+g+p}$（ZeRO-3）的递进关系。

**用本博客的符号体系表达（假定混合精度 + Adam）**：

#### ZeRO-1

$$\text{Total Memory}_{Training} \approx \text{Model Memory} + \frac{\text{Optimizer Memory}}{\text{No. GPUs}} + \text{Activation Memory} + \text{Gradient Memory}$$

#### ZeRO-2

$$\text{Total Memory}_{Training} \approx \text{Model Memory} + \text{Activation Memory} + \frac{\text{Optimizer Memory} + \text{Gradient Memory}}{\text{No. GPUs}}$$

#### ZeRO-3

$$\text{Total Memory}_{Training} \approx \text{Activation Memory} + \frac{\text{Model Memory} + \text{Optimizer Memory} + \text{Gradient Memory}}{\text{No. GPUs}} + \text{(ZeRO-3 Live Params)}$$

其中 DP Degree 在没有 pipeline/tensor parallelism 时就是 GPU 总数。

**ZeRO-3 Live Params**：ZeRO-3 引入了一组配置选项（`stage3_max_live_parameters`、`stage3_max_reuse_distance`、`stage3_prefetch_bucket_size`、`stage3_param_persistence_threshold`）来控制同一时刻 GPU 内存中保留多少参数。较大的值占用更多显存但需要更少通信。这些参数对总 GPU 显存有显著影响。

**ZeRO-R（激活值分区）**：ZeRO 还可以通过 ZeRO-R 在数据并行 rank 上对激活值进行分区。这会把 $M_{activations}$ 除以 tensor parallelism 度数 $t$。更多细节见 ZeRO 论文和相关配置选项（GPT-NeoX 中对应 `partition_activations` 标志）。

使用 ZeRO-R + ZeRO-1 时的显存公式：

$$\text{Total Memory}_{Training} \approx \text{Model Memory} + \frac{\text{Optimizer Memory}}{\text{No. GPUs}} + \frac{\text{Activation Memory}}{\text{No. GPUs}} + \text{Gradient Memory}$$

### 5.2 3D 并行

LLM 的并行主要有三种形式：

**数据并行 (Data Parallelism)**：在（可能是模型并行的）模型副本之间拆分数据。

**流水线并行 / 张量/模型并行 (Pipeline / Tensor/Model Parallelism)**：这些并行方案将模型参数**拆分**到 GPU 上。需要显著的通信开销，但显存缩减大约为：

$$M_{model}^{\text{w/ parallelism}} \approx \frac{\text{Model Memory}}{\text{Pipe-Parallel-Size} \times \text{Tensor-Parallel-Size}}$$

$$M_{gradients}^{\text{w/ parallelism}} \approx \frac{\text{Gradient Memory}}{\text{Pipe-Parallel-Size}}$$

**原文警告**：这两个公式是**近似**的，原因：
1. 流水线并行**不**减少激活值的显存占用
2. 流水线并行要求所有 GPU 存储所有 micro-batch 飞行中的激活值——这对大模型来说变得很显著
3. GPU 需要临时存储并行方案所需的额外通信缓冲区

### 5.3 分片优化器 + 3D 并行

当 ZeRO 与 tensor/pipeline 并行结合时，形成的并行策略形成一个网格。

**DP Degree 的计算**：

$$\text{DP Degree} = \frac{\text{No. GPUs}}{(\text{Pipe-Parallel-Size}) \times (\text{Tensor-Parallel-Size})}$$

DP degree 对于计算训练的全局 batch size 至关重要。

**原文的经验总结**：

- 流水线并行和所有 ZeRO stages 理论上是兼容的。但流水线并行与 ZeRO-2/3 的梯度分片结合时**难以维持效率**（因为 ZeRO-2 分片梯度，但 PP 累积梯度。可以仔细定义流水线调度并重叠通信来维持效率，但难度高到 DeepSpeed 目前禁止这种组合——[链接到源码](https://github.com/microsoft/DeepSpeed/blob/v0.10.1/deepspeed/runtime/pipe/engine.py#L71)）
- Tensor 并行与所有 ZeRO stages 都是互补的（原文解释了原因：ZeRO-3 从其他 rank 收集完整层参数然后处理完整输入；TP 从其他 rank 收集远程激活值然后处理输入的**局部**分片）
- EleutherAI 的大部分工作在 pipeline + tensor 并行的基础上使用 **ZeRO-1**。原因是他们发现在大规模下 ZeRO-3 对硬件来说通信过重，因此改为跨节点使用 pipeline 并行、节点内使用 tensor 并行。

**一个典型的 3D 并行 ZeRO-1 + 激活值分区的完整公式**：

$$\text{Total Memory}_{Training} \approx \frac{\text{Model Memory}}{\text{PP} \times \text{TP}} + \frac{\text{Optimizer Memory}}{\text{No. GPUs}} + \frac{\text{Activation Memory}}{\text{TP}} + \frac{\text{Gradient Memory}}{\text{PP}}$$

---

## 6. 结论

原文以以下声明收尾：

> EleutherAI 的工程师经常使用上述启发式方法来规划高效的模型训练和调试分布式运行。作者希望为这些常被忽视的实现细节提供一些清晰度。

引用格式（BibTeX）也在原文末尾给出。

---

## 附录：结构化总结与知识整合

以下是我个人的归纳组织，将原文分散在各节的公式和概念整合为可速查的体系。

### A. 核心公式一览

| 公式 | 说明 | 来源（博客原文节） |
|------|------|------------------|
| $C \approx 6PD$ | 训练总 FLOPs | §2 Compute Requirements |
| $C_{forward} \approx 2PD$ | 前向 FLOPs | §2 |
| $C_{backward} \approx 4PD$ | 反向 FLOPs | §2 |
| $D = 20P$ | Chinchilla 最优（参数-数据比） | §2 Parameter vs Dataset Tradeoffs |
| $M_{inference} \approx 1.2 \times M_{model}$ | 推理显存 | §4.1 |
| $M_{training} = M_{model} + M_{optimizer} + M_{activations} + M_{gradients}$ | 训练总显存 | §4.7 |
| $M_{optimizer}^{AdamW} = 12 \cdot P$ bytes | AdamW 优化器状态 | §4.4 |
| $M_{activations}^{selective} = sbhL(10 + 24/t)$ | 选择性重计算激活值 | §4.6 |
| $M_{activations}^{full} = 2 \cdot sbhL$ | 全重计算激活值 | §4.6 |

### B. 精度 vs 字节速查

| 场景 | 精度 | 字节/参数 |
|------|------|----------|
| 推理（int8 量化） | int8 | 1 |
| 推理/训练前向（bf16/fp16） | bf16/fp16 | 2 |
| 训练（参数副本、优化器状态、梯度）| fp32 | 4 |

### C. 优化器字节开销对比

| 优化器 | 每参数字节 | = 参数副本 + 动量 + 方差 |
|--------|-----------|------------------------|
| AdamW | 12 | 4 + 4 + 4 |
| 8-bit Adam | 6 | 4 + 1 + 1 |
| SGD + Momentum | 8 | 4 + 4 |

### D. ZeRO Stages 递进

| Stage | 分片内容 | 关键公式 |
|-------|---------|---------|
| ZeRO-1 ($P_{os}$) | Optimizer States | $M_{total} \approx M_{model} + \frac{M_{optimizer}}{N_{GPU}} + M_{act} + M_{grad}$ |
| ZeRO-2 ($P_{os+g}$) | + Gradients | $M_{total} \approx M_{model} + M_{act} + \frac{M_{optimizer} + M_{grad}}{N_{GPU}}$ |
| ZeRO-3 ($P_{os+g+p}$) | + Model Parameters | $M_{total} \approx M_{act} + \frac{M_{model} + M_{optimizer} + M_{grad}}{N_{GPU}}$ |

### E. 与原博客的对比：我们后续写的 system_math_notes 增加了什么

| 维度 | Transformer Math 101 | 本项目的 system_math_notes |
|------|---------------------|--------------------------|
| 模型基准 | 通用（未指定模型） | Qwen3-8B 为具体算例 |
| 非 matmul 操作分析 | 未涉及 | 第 3 章：Softmax/SiLU/RMSNorm 逐一量化 |
| $2ND$ 近似推导 | 一句带过 | 第 1 章：从内积到求和完整推导 |
| 激活值分解 | 三个公式（无推导） | 第 6 章：逐张量拆解 + FlashAttention 校正 |
| AC/GA 数学 | 未涉及 | 第 7-8 章：checkpoint interval 的完整 trade-off |
| 反向 FLOPs 证明 | 只给出结论 | 第 1.3 节：给 weight grad 和 input grad 各写了一遍 matmul |
| GQA 对 FLOPs 的影响 | 未涉及 | 第 2.1 节：GQA 4:1 省 50% QKV FLOPs 的完整计算 |

**简言之**：Transformer Math 101 给出了「地图」——全局公式和工程经验。本项目的 system_math_notes 是地图上每条路径的「实地勘测」——代入具体模型、逐层推导、每题验证。

# Phase 1：现代大模型骨架 — 手撕实现

本目录是 LLM 核心算子手撕工程的 **Phase 1**，对应 70 天路线图中的 **Day 4–23**。目标是用纯 PyTorch/Triton 从零实现一个 Decoder-only 大模型（GPT/LLaMA 风格）的全部骨架组件——从 Token 嵌入到最终 Loss 输出，不依赖 HuggingFace。所有算子通过 `torch.allclose` 与官方实现严格对齐。

核心哲学：**"What I cannot create, I do not understand."** — 每个算子的数学推导、工程取舍和显存优化都记录在 `_notes.md` 知识库中。

## 目录结构

```
Phase1_Backbone/
├── README.md                          # 本文件——目录总览与学习路径
├── CLAUDE.md                          # 本目录的任务指令（重组与审阅笔记）
├── Embedding/                         # 词嵌入：One-Hot + Matmul 等价性、查表法 vs 矩阵乘法
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── embedding_notes.md             #   完整知识库（7 个专题）
│   └── my_embedding.py
├── RMSNorm_Compilation/               # RMSNorm：从 PyTorch 原生到 Triton 融合 Kernel
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── Pytorch_Native/
│   │   ├── README.md                  #     子目录索引
│   │   ├── rmsnorm_notes.md           #     完整知识库：尺度不变性、梯度推导、torch.compile
│   │   ├── my_layer_norm.py / my_RMSNorm.py / my_rmsnorm_compile.py
│   └── Triton/
│       ├── README.md                  #     子目录索引
│       ├── triton_kernel_notes.md     #     硬件数据流、SPMD 范式、Kernel 逐行解析
│       ├── my_rmsnorm_triton.py / benchmark_rmsnorm.py
├── Attention/                         # 多头因果掩码注意力 + Safe Softmax + 张量基础
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── attention_notes.md             #   核心知识库：复杂度分析、QK 缩放、因果掩码三场景
│   ├── softmax_notes.md               #   Safe Softmax 避坑点与 FP16 浮点表示
│   ├── tensor_mechanics.md            #   张量基础：clone/view/contiguous/reshape 显存语义
│   ├── multi_heads_attention.py / safe_softmax.py
├── FFN/                               # SwiGLU 前馈网络：三矩阵合并优化 + 参数/算力占比分析
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── ffn_notes.md                   #   FFN 架构基础：Position-wise、升维降维、参数/算力占比
│   ├── SwiGLU.md                      #   SwiGLU 专题：设计哲学、激活函数对比、权重初始化
│   ├── ffn_engineering.md             #   现代工程实践：Bias-free、硬件对齐、显存管理、API 辨析
│   ├── swiglu_ffn.py
├── LM_Head_CE_Loss/                   # LM Head + 数值稳定交叉熵损失
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── CE_Loss_Notes.md               #   理论与工程笔记：CE 基础 / 浮点精度 / LM Head 架构 / Q&A
│   ├── CE_Loss_Academic_Notes.md      #   学术深度推导：Shannon 公理 → Proper Scoring Rule
│   └── lm_head.py
├── RoPE/                              # 旋转位置编码：解耦旋转 + 混合精度保护
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── rope_notes.md                  #   完整知识库：词袋子问题 → Long Term Decay 证明
│   ├── rope_utils.py                  #   共享工具：precompute_freqs_cos_sin + rotate_half
│   ├── rope_embedding.py / rope_embedding_position.py
├── MoE/                               # Mixtral 8x7B 稀疏专家层：负载均衡 + Grouped GEMM
│   ├── README.md                      #   目录索引 + 学习路径
│   ├── moe_notes.md                   #   完整知识库：Aux Loss 三种证明、Router 设计、梯度校验
│   ├── moe_utils.py                   #   共享工具：compute_aux_loss（负载均衡损失）
│   ├── moe_layer.py / moe_layer_naive.py
└── Dropout/                           # 倒置 Dropout + autograd.Function 全链路（选读）
    ├── README.md                      #   目录索引 + 学习路径
    ├── dropout_notes.md               #   完整知识库：期望值对齐、Type Promotion、Autograd 引擎
    └── inverted_dropout.py
```

## 组件速览

| 组件 | 一句话定位 | 核心实现 | 知识笔记 |
| :--- | :--- | :--- | :--- |
| **Embedding** | 离散符号到连续空间的映射字典 | `my_embedding.py` | [embedding_notes.md](Embedding/embedding_notes.md) |
| **RMSNorm** | 砍掉均值的归一化，LLaMA 标配 | `my_RMSNorm.py` / `my_rmsnorm_triton.py` | [rmsnorm_notes.md](RMSNorm_Compilation/Pytorch_Native/rmsnorm_notes.md) / [triton_kernel_notes.md](RMSNorm_Compilation/Triton/triton_kernel_notes.md) |
| **Attention** | 多头因果掩码注意力 + Safe Softmax | `multi_heads_attention.py` / `safe_softmax.py` | [attention_notes.md](Attention/attention_notes.md) / [softmax_notes.md](Attention/softmax_notes.md) / [tensor_mechanics.md](Attention/tensor_mechanics.md) |
| **FFN** | SwiGLU 前馈网络，占模型 2/3 参数 | `swiglu_ffn.py` | [ffn_notes.md](FFN/ffn_notes.md) / [SwiGLU.md](FFN/SwiGLU.md) / [ffn_engineering.md](FFN/ffn_engineering.md) |
| **LM Head + CE Loss** | 词表概率输出 + 数值稳定损失 | `lm_head.py` | [CE_Loss_Notes.md](LM_Head_CE_Loss/CE_Loss_Notes.md) / [CE_Loss_Academic_Notes.md](LM_Head_CE_Loss/CE_Loss_Academic_Notes.md) |
| **RoPE** | 解耦旋转位置编码，长文本泛化关键 | `rope_embedding.py` | [rope_notes.md](RoPE/rope_notes.md) |
| **MoE** | 稀疏专家混合，8×7B 参数的训练秘诀 | `moe_layer.py` | [moe_notes.md](MoE/moe_notes.md) |
| **Dropout** | 倒置 Dropout + Autograd 引擎全链路（现代 LLM 已少用） | `inverted_dropout.py` | [dropout_notes.md](Dropout/dropout_notes.md) |

## 建议学习路径

推荐按以下顺序线性推进，每个阶段先读笔记理解"为什么"，再看代码验证"怎么做"：

### 阶段一：基础算子（Day 4–12）

| 顺序 | 组件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [Embedding](Embedding/) | 离散 Token 怎么变成连续向量？为什么查表法等价于 One-Hot + Linear？ |
| 2 | [RMSNorm](RMSNorm_Compilation/) | 为什么 LayerNorm 被淘汰？"尺度不变性"怎么自动阻尼梯度？ |
| 3 | [Attention](Attention/) | QK^T 为什么要除以 √d_k？因果掩码在训练/Prefill/Decode 三阶段有何不同？ |
| 4 | [FFN](FFN/) | 为什么 FFN 占 2/3 参数？SwiGLU 的三矩阵合并省了什么？ |

> **穿插选读**：[Dropout](Dropout/) — 虽然现代 LLM 已不再使用 Dropout，但它是理解 `autograd.Function`、`ctx.save_for_backward` 和 `backward` 执行机制的最佳入口。

### 阶段二：组装与 Loss（Day 13–16）

| 顺序 | 组件 | 核心问题 |
| :--- | :--- | :--- |
| 5 | [LM Head + CE Loss](LM_Head_CE_Loss/) | 为什么交叉熵 + Softmax 是天作之合？LogSumExp 怎么同时解决溢出和 log(0)？ |

至此，你已拥有拼装一个完整 Decoder-only 模型所需的全部算子。

### 阶段三：高级架构（Day 17–23）

| 顺序 | 组件 | 核心问题 |
| :--- | :--- | :--- |
| 6 | [RoPE](RoPE/) | 绝对位置编码 vs 相对位置编码？怎么用复数旋转实现远程衰减？ |
| 7 | [MoE](MoE/) | 负载均衡损失如何防止专家坍缩？为什么 for 循环遍历专家必须消除？ |

### 阶段四：编译器与底层（穿插于阶段一）

| 顺序 | 组件 | 核心问题 |
| :--- | :--- | :--- |
| — | [torch.compile](RMSNorm_Compilation/Pytorch_Native/rmsnorm_notes.md) | Memory Wall 是什么？编译器怎么自动做算子融合？ |
| — | [Triton Kernel](RMSNorm_Compilation/Triton/) | SPMD 范式怎么用？为什么 `rsqrt` 比除法快 30%？ |

## 快速运行

每个子目录的 `.py` 文件自带 `if __name__ == '__main__':` 冒烟测试，可直接运行：

```bash
# 基础算子
python Embedding/my_embedding.py
python RMSNorm_Compilation/Pytorch_Native/my_RMSNorm.py
python Attention/multi_heads_attention.py
python Attention/safe_softmax.py
python FFN/swiglu_ffn.py

# Loss 与位置编码
python LM_Head_CE_Loss/lm_head.py
python RoPE/rope_embedding.py

# 高级架构
python MoE/moe_layer.py

# Autograd 进阶（选读）
python Dropout/inverted_dropout.py

# Triton Kernel（需 Linux + CUDA）
python RMSNorm_Compilation/Triton/my_rmsnorm_triton.py
python RMSNorm_Compilation/Triton/benchmark_rmsnorm.py
```

或通过项目根目录的 pytest 批量运行：

```bash
pytest tests/test_phase1_backbone.py -v
```

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 每个子目录的**精简索引**：文件清单 + 一句话说明 + 运行指令 + 链接到知识库 |
| **`{topic}_notes.md`** | **完整知识库**：理论 → 实现 → 工程细节，按 `Part I/II/III` 分主题组织（最多 3 层） |
| **`plan.md`** | **实现路线图**（仅 MoE、Dropout）：分 Day 的任务分解，面向动手实践 |
| **代码注释** | 中文 docstring + 关键公式用 LaTeX + 张量形状流转注释（如 `# [B, L, D] -> [B, num_heads, L, head_dim]`） |
| **冒烟测试** | 每个 `.py` 自带 `__main__` 块，包含形状检查、数值断言和友好中文输出 |

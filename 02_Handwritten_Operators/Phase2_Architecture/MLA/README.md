# MLA (Multi-head Latent Attention)：DeepSeek-V2 低秩注意力 — 手撕实现

本目录从零实现 DeepSeek-V2 的核心创新——Multi-head Latent Attention。通过低秩压缩将 KV Cache 压缩为极小的 Latent Vector，再通过"吸收矩阵"技术将投影权重在推理时合并，实现比 GQA 更极致的显存节省。

## 目录结构

```
MLA/
├── README.md                        # 本文件 — 目录索引与学习路径
├── multi_head_latent_attention.py   #   MLA 完整前向实现（含吸收矩阵）
├── causal_mask_padding_mask.py      #   因果掩码 + 填充掩码的组合掩码工具
├── mla_paper_notes.md               #   论文精读：DeepSeek-V2 架构 → Decoupled RoPE → KV 压缩率
├── mla_math_proofs.md               #   数学证明：低秩分解 → 权重吸收推导 → FLOPs 对比
├── mla_code_walkthrough.md          #   代码走读：einsum 教程 → forward 流程 → mask 组合逻辑
└── block_matmul.md                  #   前置数学：分块矩阵乘法基础（BlockDiag 形式化）
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [multi_head_latent_attention.py](multi_head_latent_attention.py) | MLA 完整实现：低秩投影 + 吸收矩阵 + KV Cache 管理 | 对着代码看实现 |
| [causal_mask_padding_mask.py](causal_mask_padding_mask.py) | 因果掩码 + 填充掩码的组合工具函数 | 配套理解 mask 逻辑 |
| [mla_paper_notes.md](mla_paper_notes.md) | DeepSeek-V2 论文精读：架构总览、Decoupled RoPE 设计动机、KV 压缩率论证 | 先看这里理解 MLA 是什么 |
| [mla_math_proofs.md](mla_math_proofs.md) | 严谨数学推导：低秩分解 → weight absorption → FLOPs/参数量对比 | 追数学细节时看 |
| [mla_code_walkthrough.md](mla_code_walkthrough.md) | 代码逐行走读：einsum 语法教程 → forward 流程拆解 → mask 组合逻辑 | 对照代码理解实现 |
| [block_matmul.md](block_matmul.md) | 前置数学：分块矩阵乘法基础、BlockDiag 形式化 | 需要用到的背景知识 |

## 快速运行

```bash
python causal_mask_padding_mask.py
python multi_head_latent_attention.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [mla_paper_notes.md](mla_paper_notes.md) | MLA 的整体架构是什么？Decoupled RoPE 为什么需要？KV 能压缩多少？ |
| 2 | [block_matmul.md](block_matmul.md) | 分块矩阵乘法基础——理解吸收矩阵的前置知识 |
| 3 | [mla_math_proofs.md](mla_math_proofs.md) | W^UK 怎么被吸收到 W^Q？W^UV 怎么被吸收到 W^O？FLOPs 对比 |
| 4 | [causal_mask_padding_mask.py](causal_mask_padding_mask.py) | 因果掩码和 padding 掩码怎么组合？tril 约定是什么？ |
| 5 | [mla_code_walkthrough.md](mla_code_walkthrough.md) | einsum 语法怎么用？forward 每一步对应论文中哪个公式？ |
| 6 | [multi_head_latent_attention.py](multi_head_latent_attention.py) | 完整前向传播：低秩投影 → KV Cache → 吸收 → 输出 |

## 文档约定

五个 `.md` 文件各自有独立的领地，交叉链接而非重复内容：
- **Paper_Notes**：论文架构解读，不深入数学推导
- **Math_Proofs**：严格数学证明，不重复论文背景
- **Code_Walkthrough**：代码逐行解读，嵌入关键代码片段但不嵌入完整代码
- **block_matmul**：独立的前置数学，被上述文件引用
- **README.md**：本文档——目录索引

# MLA (Multi-head Latent Attention) 笔记索引

> 关联代码：[multi_head_latent_attention.py](multi_head_latent_attention.py) · [causal_mask_padding_mask.py](causal_mask_padding_mask.py)

MLA 的深度笔记拆分为四个专题文件，各司其职、互不重复。完整的架构介绍与学习路径见本目录的 [README.md](README.md)。

## 文件导航

### 笔记（.md）

| 文件 | 职责 | 适合场景 |
| :--- | :--- | :--- |
| [mla_paper_notes.md](mla_paper_notes.md) | DeepSeek-V2 论文精读：架构总览、Decoupled RoPE 动机、KV 压缩率、与 MHA/GQA/MQA 对比、超参数设计 | 先看这个，理解 MLA 是什么 |
| [mla_math_proofs.md](mla_math_proofs.md) | 严格数学推导：低秩分解 → SVD 最优近似 → 权重吸收 W^UK→W^Q 和 W^UV→W^O → FLOPs 对比 → BlockDiag 修正 | 追数学细节时看 |
| [block_matmul.md](block_matmul.md) | 前置数学：分块矩阵乘法基础与 BlockDiag 形式化 | 需要背景知识时查阅 |
| [mla_code_walkthrough.md](mla_code_walkthrough.md) | 代码逐行走读：einsum 语法教程 → forward 流程拆解 → mask 组合逻辑 → 推理公式与代码对应 | 对照代码理解实现 |

### 代码（.py）

- [multi_head_latent_attention.py](multi_head_latent_attention.py) — MLA 完整实现：训练版（`forward_training`）+ 推理版（`forward_infer` + `absorb_weights`），**自带 `__main__` 冒烟测试**
- [causal_mask_padding_mask.py](causal_mask_padding_mask.py) — 注意力掩码工具：causal mask + padding mask 组合

### 边界约定

四个笔记文件各自有严格领地，交叉链接而非重复内容：
- **Paper_Notes**：论文架构解读，不深入数学推导——涉及证明时引向 Math_Proofs
- **Math_Proofs**：严格数学证明，不重复论文背景——涉及代码实现时引向 Code_Walkthrough
- **Code_Walkthrough**：代码逐行走读与 einsum 教程，不重复完整证明——涉及理论基础时引向 Math_Proofs 或 Paper_Notes
- **block_matmul**：独立前置知识，被上述文件引用

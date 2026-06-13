# 词嵌入（Word Embedding）— 手撕实现

本目录包含词嵌入层的完整手写实现，展示数学等价（One-Hot + Matmul）与工程查表（高级索引）两种实现方式。

> 完整的理论背景和工程细节请参见 [embedding_notes.md](embedding_notes.md)。

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `my_embedding.py` | **两种等价实现** — `forward_math_equivalent`（One-Hot + Matmul）与 `forward_engineering_real`（查表法），验证两者完全等价 |
| `embedding_notes.md` | **完整知识库** — 符号连续化、`nn.Parameter`、`F.one_hot`、查表 vs 矩阵乘法、Weight Tying 等 7 个专题 |

## 快速运行

```bash
python my_embedding.py
```

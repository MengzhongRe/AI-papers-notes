# LM Head 与交叉熵损失（CE Loss）— 手撕实现

本目录包含语言模型头（LM Head）与数值稳定交叉熵损失的完整手写实现，涵盖自回归 Shift 逻辑、LogSumExp 技巧、Padding 过滤和分布式并行议题。

> 完整的理论推导和工程细节请参见以下两份笔记，各有侧重、互补不重复：
> - [`CE_Loss_Notes.md`](CE_Loss_Notes.md) — **理论与工程笔记**（本目录主知识库），四大部分：CE 理论基础 / 浮点精度与显存 / LM Head 架构与工程边界 / 面试 Q&A
> - [`CE_Loss_Academic_Notes.md`](CE_Loss_Academic_Notes.md) — **学术深度推导**，从 Shannon 公理到 Proper Scoring Rule 的系统性理论

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `lm_head.py` | **手撕实现** — 自回归 Shift 逻辑、Padding 过滤、数值稳定 CE Loss，含五重测试 |
| `CE_Loss_Notes.md` | **理论与工程笔记**（本目录主知识库）— 四大部分：CE 理论基础 / 浮点精度与显存 / LM Head 架构与工程边界 / 面试 Q&A |
| `CE_Loss_Academic_Notes.md` | **学术深度推导** — 从第一性原理出发：信息论基础、MLE 等价性、Proper Scoring Rule 统一框架、LLM 形式化目标、梯度魔法、LogSumExp、困惑度，含三份附录 |

## 快速运行

```bash
python lm_head.py
```

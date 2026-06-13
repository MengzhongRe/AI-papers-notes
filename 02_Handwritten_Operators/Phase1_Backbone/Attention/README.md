# Attention：多头因果掩码注意力 — 手撕实现

本目录包含 Multi-Head Attention 的手撕实现，以及 Safe Softmax 和张量力学的配套知识笔记。

## 目录结构

```
Attention/
├── README.md                  # 本文件 — 目录索引与学习路径
├── multi_heads_attention.py   #   多头因果掩码注意力实现
├── safe_softmax.py            #   Safe Softmax 手撕实现（数值安全版）
├── attention_notes.md         #   深度知识库：MHA 核心笔记（复杂度、因果掩码、Dropout）
├── softmax_notes.md           #   深度知识库：Safe Softmax 避坑 + FP16 精度
└── tensor_mechanics.md        #   深度知识库：PyTorch 张量基础（clone/view/contiguous/reshape）
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [multi_heads_attention.py](multi_heads_attention.py) | 多头因果掩码注意力，支持 causal/padding 掩码 | 对着代码看实现 |
| [safe_softmax.py](safe_softmax.py) | Safe Softmax 手撕实现（与 `torch.softmax` 对齐） | 理解数值安全 |
| [attention_notes.md](attention_notes.md) | MHA 核心：复杂度分析、QK 缩放、因果掩码三场景、Dropout | 先看这里 |
| [softmax_notes.md](softmax_notes.md) | Safe Softmax 避坑 + FP16 半精度浮点表示原理 | 深入数值细节 |
| [tensor_mechanics.md](tensor_mechanics.md) | 张量基础：clone/view/contiguous/reshape 显存语义 | 需要时查阅 |

## 快速运行

```bash
python multi_heads_attention.py
python safe_softmax.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [tensor_mechanics.md](tensor_mechanics.md) | view vs reshape 的区别？contiguous() 做了什么？ |
| 2 | [attention_notes.md](attention_notes.md) | QK^T 为什么除以 √d_k？因果掩码在训练/Prefill/Decode 三阶段有何不同？ |
| 3 | [softmax_notes.md](softmax_notes.md) | Safe Softmax 怎么防止指数溢出？FP16 的精度陷阱？ |
| 4 | [multi_heads_attention.py](multi_heads_attention.py) | 多头拆分怎么用 view+transpose？mask 怎么应用？ |
| 5 | [safe_softmax.py](safe_softmax.py) | 手写版和 `torch.softmax` 差了多远？ |

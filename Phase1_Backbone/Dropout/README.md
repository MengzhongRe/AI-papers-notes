# Dropout：倒置 Dropout — 手撕 autograd 全链路

本目录是 Phase1_Backbone 中关于 Dropout 的手撕实现与知识库。现代 LLM（如 Llama）虽已不再使用 Dropout，但它是理解 `autograd.Function`、`ctx.save_for_backward` 和 `backward` 执行机制的最佳入口。

## 目录结构

```
Dropout/
├── README.md              # 本文件 — 目录索引
├── inverted_dropout.py    #   手撕 Inverted Dropout（含 CustomDropout + DropoutFunc）
└── dropout_notes.md       #   深度知识库：期望值对齐、TPE、Autograd 全链路等 11 个专题
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [inverted_dropout.py](inverted_dropout.py) | 手写 Forward/Backward 梯度流 + CustomDropout 封装 | 对着代码看 autograd |
| [dropout_notes.md](dropout_notes.md) | 完整知识库：数学/工程原理 → 原地操作 → autograd.Function 全链路 | 先看这里理解为什么 |

## 快速运行

```bash
python inverted_dropout.py
```

> 📖 也请参考 Attention 目录中关于注意力 Dropout 的专项讨论：[../Attention/attention_notes.md](../Attention/attention_notes.md)

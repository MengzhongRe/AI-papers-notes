# MoE：Mixtral 8x7B 稀疏专家层 — 手撕实现

本目录包含 MoE（Mixture of Experts）的手撕实现，涵盖 Top-K 路由、负载均衡损失、以及 Naive for-loop 与 Vectorized scatter-gather 两种实现。

## 目录结构

```
MoE/
├── README.md              # 本文件 — 目录索引与学习路径
├── moe_utils.py            #   共享工具函数：compute_aux_loss（负载均衡损失）
├── moe_layer.py           #   Vectorized MoE 实现（scatter-gather 分发）
├── moe_layer_naive.py     #   Naive for-loop 版本（教学原型）
└── moe_notes.md           #   深度知识库：负载均衡损失推导、Router 设计、梯度校验
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [moe_layer_naive.py](moe_layer_naive.py) | Naive 实现：`for` 循环遍历专家 — 教学起点 | 先看这里理解 MoE 流程 |
| [moe_layer.py](moe_layer.py) | Vectorized 实现：scatter-gather 无 for 循环 — 工业写法 | 理解性能关键 |
| [moe_notes.md](moe_notes.md) | 完整知识库：Aux Loss 三种证明 → Router 设计 → 梯度校验 | 通关后精读 |

## 快速运行

```bash
python moe_layer_naive.py
python moe_layer.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [moe_layer_naive.py](moe_layer_naive.py) | MoE 的基本流程是什么？for 循环怎么遍历专家？ |
| 2 | [moe_notes.md Part I](moe_notes.md) | 负载均衡损失为什么必要？Aux Loss 怎么推导？ |
| 3 | [moe_layer.py](moe_layer.py) | 为什么 for 循环不行？scatter-gather 怎么消除循环？ |
| 4 | [moe_notes.md Part II-III](moe_notes.md) | Router 梯度怎么传？为什么 MoE 训练还占显存？ |

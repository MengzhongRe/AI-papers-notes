# RoPE：旋转位置编码 — 手撕实现

本目录包含 RoPE（Rotary Position Embedding）的手撕实现，涵盖固定频率预计算、解耦旋转、以及 Prefill/Decode 双阶段的位置编码应用。

## 目录结构

```
RoPE/
├── README.md                   # 本文件 — 目录索引与学习路径
├── rope_embedding.py           #   基础 RoPE 实现（固定位置频率表）
├── rope_embedding_position.py  #   扩展版：支持 position_ids 的动态推理
└── rope_notes.md               #   深度知识库：位置编码编年史 → RoPE 理论 → 工程实现
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [rope_utils.py](rope_utils.py) | 共享工具函数：`precompute_freqs_cos_sin` + `rotate_half` | 被下面两个文件共同引用 |
| [rope_embedding.py](rope_embedding.py) | 基础 RoPE：使用共享函数实现 `apply_rotary_emb` | 先看这里理解基础 |
| [rope_embedding_position.py](rope_embedding_position.py) | 动态推理版：支持 `position_ids`，演示 Prefill vs Decode | 理解推理时位置编码 |
| [rope_notes.md](rope_notes.md) | 完整知识库：词袋子问题 → Long Term Decay → 混合精度保护 → PyTorch 细节 | 通关后精读 |

## 快速运行

```bash
python rope_embedding.py
python rope_embedding_position.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [rope_notes.md Part I](rope_notes.md) | 为什么需要位置编码？绝对 vs 相对？ |
| 2 | [rope_notes.md Part II](rope_notes.md) | RoPE 核心理论：复数旋转、远程衰减证明 |
| 3 | [rope_embedding.py](rope_embedding.py) | precompute_freqs_cos_sin 怎么预计算？rotate_half 怎么实现解耦？ |
| 4 | [rope_embedding_position.py](rope_embedding_position.py) | 推理时 position_ids 怎么用？Prefill vs Decode 的区别？ |

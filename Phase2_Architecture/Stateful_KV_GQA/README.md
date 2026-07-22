# Stateful KV Cache & GQA：分组查询注意力 + 两阶段推理 — 手撕实现

GQA（分组查询注意力）通过让多个 Query 头共享 KV 头，大幅缩小 KV Cache 体积。本目录提供两种 KV Cache 策略（动态拼接 vs 静态预分配）的完整实现，并辅以 Prefill/Decode 两阶段推理的数学推导笔记。

## 目录结构

```
Stateful_KV_GQA/
├── README.md                           # 本文件 — 目录索引与学习路径
├── gqa_attention_base.py               #   GQA 共享基类：QKV 投影 + KV 头广播 + 注意力计算
├── grouped_query_attention.py           #   GQA 动态缓存版：torch.cat 拼接历史 KV
├── stateful_kvcache_gqa.py             #   GQA 静态预分配版：register_buffer + start_pos 就地写入
└── kv_cache_prefill_decode_notes.md    #   深度知识库：O(N³)→O(N²) / Prefill vs Decode / 算术强度
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [gqa_attention_base.py](gqa_attention_base.py) | GQA 共享基类：QKV 投影 + KV 头广播 + 注意力计算 + 多头合并 | 了解公共逻辑 |
| [grouped_query_attention.py](grouped_query_attention.py) | GQA 动态缓存版：每次 forward 用 `torch.cat` 拼接历史 KV，简单直观 | 先看这个理解两阶段行为 |
| [stateful_kvcache_gqa.py](stateful_kvcache_gqa.py) | GQA 静态预分配版：`register_buffer` + `start_pos` 索引就地写入，兼容 torch.compile | 看生产级实现 |
| [kv_cache_prefill_decode_notes.md](kv_cache_prefill_decode_notes.md) | 深度知识库：10 个专题覆盖 KV Cache / Prefill vs Decode / 算术强度 / Static Cache | 先看笔记理解为什么 |

## 快速运行

```bash
python grouped_query_attention.py
python stateful_kvcache_gqa.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [kv_cache_prefill_decode_notes.md](kv_cache_prefill_decode_notes.md) | 为什么需要 KV Cache？Prefill 和 Decode 的算术强度差多大？ |
| 2 | [grouped_query_attention.py](grouped_query_attention.py) | GQA 的 KV 头怎么广播到 Q 头？动态 cache 如何用 torch.cat 实现？ |
| 3 | [stateful_kvcache_gqa.py](stateful_kvcache_gqa.py) | 静态预分配为什么更快？start_pos 索引怎么统一 Prefill 和 Decode？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 |
| **代码注释** | 中文注释 + 张量形状流转标注 |
| **冒烟测试** | 每个 `.py` 自带 `__main__` 块，含 assert 验证形状和数值等价性 |

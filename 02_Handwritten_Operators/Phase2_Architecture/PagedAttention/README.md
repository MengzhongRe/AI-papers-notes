# PagedAttention：vLLM 物理块表 + 零拷贝碎片化 Attention — 手撕实现

本目录从零实现 vLLM 推理引擎的核心——PagedAttention。通过字典和列表模拟操作系统的"页表"，将 KV Cache 存储在非连续的物理块中，实现近乎零浪费的显存管理。

## 目录结构

```
PagedAttention/
├── README.md                    # 本文件 — 目录索引与学习路径
├── paged_attention.py           #   完整实现：PhysicalKVMemoryPool + PagedAttention
└── paged_attention_notes.md     #   深度知识库：vLLM 架构 → 虚拟内存映射 → 面试 Q&A
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [paged_attention.py](paged_attention.py) | `PhysicalKVMemoryPool`（物理块池）+ `PagedAttention`（碎片化 Online Softmax） | 对着代码看实现 |
| [paged_attention_notes.md](paged_attention_notes.md) | 完整知识库：显存碎片问题 → 页表映射 → Block Table → Online Softmax 流式计算 | 先看这里理解为什么 |

## 快速运行

```bash
python paged_attention.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [paged_attention_notes.md](paged_attention_notes.md) | 传统连续 KV Cache 的显存碎片问题？"页"的概念怎么引入？ |
| 2 | [paged_attention.py](paged_attention.py) | Block Table 怎么把 logical_token_index 映射到 physical_block_id？ |
| 3 | [paged_attention.py](paged_attention.py) | 如何在不连续物理块上做 Online Softmax 流式计算？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 |
| **代码注释** | 中文注释 + 张量形状流转标注（如 `# [num_heads, head_dim]`） |
| **冒烟测试** | `paged_attention.py` 自带 `__main__` 块，与拼接后暴力 Attention 对比验证 |
| **笔记** | `paged_attention_notes.md` 按 16 个专题覆盖 vLLM 全部核心知识点 |

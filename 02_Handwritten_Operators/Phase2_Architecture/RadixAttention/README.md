# RadixAttention：SGLang 前缀树缓存 + LRU 驱逐 — 手撕实现

本目录从零实现 SGLang 推理引擎的核心——RadixAttention（Radix Tree 前缀缓存）。通过前缀树（Radix Tree / Patricia Trie）管理跨请求的 KV Cache 共享，用 LRU 驱逐策略在显存耗尽时回收最久未用的缓存。

## 目录结构

```
RadixAttention/
├── README.md                      # 本文件 — 目录索引与学习路径
├── radix_attention.py             #   RadixTree 核心实现：节点分裂 + 前缀匹配 + LRU 驱逐
├── radix_attention_notes.md       #   深度知识库：PagedAttention vs RadixAttention → LRU 策略 → 调度
├── radix_code_walkthrough.md      #   代码走读：match_prefix / insert(split) / evict 逐函数解析
├── radix_tree_structure.png       #   前缀树结构示意图
└── cache_lifecycle.png            #   PagedAttention vs RadixAttention 缓存生命周期对比图
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [radix_attention.py](radix_attention.py) | `RadixNode` + `RadixCacheTree` 完整实现：匹配/插入分裂/LRU 驱逐，自带断言冒烟测试 | 对着代码看实现 |
| [radix_attention_notes.md](radix_attention_notes.md) | 深度知识库：PagedAttention vs RadixAttention 区别、LRU 为什么优于 LFU、缓存感知调度 | 先看这里理解为什么 |
| [radix_code_walkthrough.md](radix_code_walkthrough.md) | 代码逐行走读：四大核心逻辑（节点存储/前缀匹配/插入分裂/驱逐）的图文拆解 | 对照代码理解实现 |

## 快速运行

```bash
python radix_attention.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [radix_attention_notes.md](radix_attention_notes.md) | PagedAttention 的 `ref_count` 不是已经能共享了吗？为什么还需要 RadixAttention？ |
| 2 | [radix_code_walkthrough.md](radix_code_walkthrough.md) | Split 分裂时为什么截断后的父节点 ref_count=0，旧分支继承 ref_count=1？ |
| 3 | [radix_attention.py](radix_attention.py) | `match_prefix → insert → evict` 三个操作在真实多轮对话中如何串联？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 |
| **代码注释** | 中文注释 + 类/方法 docstring |
| **冒烟测试** | `radix_attention.py` 自带 `__main__` 块，含多个 assert 验证树结构正确性 |

# Phase 0：分词器 (Tokenization) — 手撕 BPE 全链路

本目录是 LLM 核心算子手撕工程的 **Phase 0**，对应 70 天路线图中的 **Day 1–3**。目标是用纯 Python 从零实现一个符合现代工业标准的 BPE 分词器——从 UTF-8 字节映射到贪心合并训练，不依赖 HuggingFace `tokenizers`。

核心哲学：**"What I cannot create, I do not understand."** — 先理解字符级 BPE 的教学原型，再将其升级为字节级 BBPE 的工业实现。

## 目录结构

```
Phase0_tokenization/
├── README.md                        # 本文件 — 目录索引与学习路径
├── base_bpe.py                      #   共享基类（字符级 & 字节级的公共骨架）
├── bpe_tokenizer.py                 #   字符级 BPE — 教学原型（继承 base_bpe）
├── bbpe_tokenizer.py                #   字节级 BBPE — 工业标准（继承 base_bpe）
├── byte_to_unicode_map.py           #   bytes_to_unicode() 工具函数
├── BPE_Pipeline_Notes.md            #   深度知识库：工业级 BPE 六步流水线全流程
└── BBPE.md                          #   深度知识库：BBPE 底层原理 + Unicode/UTF-8
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [base_bpe.py](base_bpe.py) | 提取共享逻辑 — 统计、合并、编码循环、特殊 token 统一处理 | 先看这里理解架构 |
| [bpe_tokenizer.py](bpe_tokenizer.py) | 字符级 BPE，从训练字符集出发合并，OOV 回退 `<UNK>` | 从这里开始手撕 |
| [bbpe_tokenizer.py](bbpe_tokenizer.py) | 字节级 BBPE，UTF-8 + 256 基础字节 → 永无 OOV | 理解现代分词器底层 |
| [byte_to_unicode_map.py](byte_to_unicode_map.py) | GPT-2 经典映射：0–255 → 可见 Unicode 字符 | 跟着 BBPE 一起看 |
| [BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) | 六步流水线 + 规范化四维度 + 预分词 + 正则分支优先级 | 通关后精读 |
| [BBPE.md](BBPE.md) | Unicode/UTF-8 基础 → 字节映射 → 升级指南 → 面试 Q&A | 深入 BBPE 专项 |

## 快速运行

```bash
# 字符级 BPE（教学原型）
python bpe_tokenizer.py

# 字节级 BBPE（工业标准 — 中文/Emoji 全覆盖）
python bbpe_tokenizer.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [base_bpe.py](base_bpe.py) | BPE 的公共骨架长什么样？类结构怎么设计？ |
| 2 | [bpe_tokenizer.py](bpe_tokenizer.py) | 字符级 BPE 怎么从几个字符开始合并？遇到 OOV 怎么办？ |
| 3 | [byte_to_unicode_map.py](byte_to_unicode_map.py) | 256 字节怎么映射成可见字符？为什么不能直接对原始字节做正则？ |
| 4 | [bbpe_tokenizer.py](bbpe_tokenizer.py) | BBPE 如何通过仅 4 个钩子把字符级升级为字节级？ |
| 5 | [BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) | 工业界六步流水线全貌？规范化四个维度？预分词正则怎么设计？ |
| 6 | [BBPE.md](BBPE.md) | Unicode vs UTF-8 底层关系？字节映射怎么实现？面试怎么答？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 运行指令 + 学习路径 + 链接到深度文档 |
| **`{Topic}_Notes.md`** | 深度知识库：理论 → 实现 → 工程 → 面试，按 Part I/II/III 分层 |
| **代码注释** | 中文 docstring + 关键步骤解释 + 张量/数据结构形状注释 |
| **冒烟测试** | 每个 `.py` 自带 `__main__` 块，包含编解码断言和友好中文输出 |
| **交叉链接** | 所有 `.md` 之间互相链接，README 是入口导航 |

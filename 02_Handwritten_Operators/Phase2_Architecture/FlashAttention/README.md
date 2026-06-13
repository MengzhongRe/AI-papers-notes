# FlashAttention：Online Softmax 分块注意力 — 手撕实现

本目录用纯 Python for-循环模拟 FlashAttention 的"分块（Tiling）"读取逻辑，不写 CUDA 也能理解 GPU Kernel 的核心思想。

## 目录结构

```
FlashAttention/
├── README.md                      # 本文件 — 目录索引与学习路径
├── flash_attention_forward.py     #   完整 Tiling 循环实现（外层 Q，内层 K/V）
├── online_softmax.py              #   单块 Online Softmax 更新函数（教学用）
└── flash_attention_notes.md       #   深度知识库：GPU 矩阵乘法 → Online Softmax → 面试 Q&A
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [flash_attention_forward.py](flash_attention_forward.py) | 完整双层循环实现：外层 Q 循环 + 内层 K/V 循环 + 因果掩码 | 先看这里理解全流程 |
| [online_softmax.py](online_softmax.py) | 单块更新函数：演示一次 Online Softmax 的三个统计量更新 | 理解数学核心 |
| [flash_attention_notes.md](flash_attention_notes.md) | 深度知识库：GPU 架构 → Online Softmax → FlashAttention 前向 | 通关后精读 |

## 快速运行

```bash
python online_softmax.py
python flash_attention_forward.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [online_softmax.py](online_softmax.py) | m_new / l_new / O_new 三个统计量怎么用旧值递推？ |
| 2 | [flash_attention_forward.py](flash_attention_forward.py) | Tiling 怎么组织？外层 Q 内层 K/V 的循环结构为什么这样设计？ |
| 3 | [flash_attention_notes.md](flash_attention_notes.md) | 为什么 GPU 做矩阵乘法是分块的？SRAM vs HBM 的延迟差多大？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 |
| **代码注释** | 中文注释 + 张量形状流转标注（如 `# [B, H, Br, d]`） |
| **冒烟测试** | 每个 `.py` 自带 `__main__` 块，与标准注意力对比验证 |

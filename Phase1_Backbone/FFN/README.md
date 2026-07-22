# FFN：前馈神经网络 — 手撕 SwiGLU

本目录是 Phase1_Backbone 中关于 FFN（Feed-Forward Network）的手撕实现与知识库。
FFN 是大模型的"参数蓄水池"与"知识存储库"，占据模型约 2/3 的参数量。

## 目录结构

```
FFN/
├── README.md                # 本文件 — 目录索引与学习路径
├── swiglu_ffn.py            #   SwiGLU FFN 实现（含 w13 三矩阵合并优化 + torch.compile 性能测试）
├── ffn_notes.md             #   深度知识库：FFN 架构基础（Position-wise、升维降维、参数/算力占比）
├── SwiGLU.md                #   深度知识库：SwiGLU 专题（设计哲学、激活函数对比、权重初始化）
└── ffn_engineering.md       #   深度知识库：现代工程实践（Bias-free、硬件对齐、显存管理、API 辨析）
```

## 组件速览

| 文件                                       | 一句话定位                                                                                    | 读者                 |
| :--------------------------------------- | :--------------------------------------------------------------------------------------- | :----------------- |
| [swiglu_ffn.py](swiglu_ffn.py)           | Vanilla FFN + SwiGLU FFN（含 w13 三矩阵合并优化），附带梯度回传、等价性、torch.compile 性能测试                    | 对着代码看实现            |
| [ffn_notes.md](ffn_notes.md)             | FFN 架构基础：Position-wise 的含义、升维降维原理、参数/算力占比分析                                              | 先看这里理解 FFN 是什么     |
| [SwiGLU.md](SwiGLU.md)                   | SwiGLU 专题：激活函数演进（ReLU→GELU→Swish→SwiGLU）、门控机制、权重初始化                                      | 理解 SwiGLU 为什么是最优选择 |
| [ffn_engineering.md](ffn_engineering.md) | 现代工程决策：Bias-free 设计、`hidden_dim` 硬件对齐、`torch.no_grad()`、原地操作与显存管理、`F.linear` vs `matmul` | 写生产级代码时看           |

## 快速运行

```bash
python swiglu_ffn.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [ffn_notes.md](ffn_notes.md) | FFN 是什么？为什么叫 Position-wise？为什么要先升维再降维？ |
| 2 | [ffn_notes.md §Part II](ffn_notes.md#part-ii-参数与计算量分析) | 为什么 FFN 占模型 2/3 参数？在训练/Prefill/Decode 各阶段算力占比如何？ |
| 3 | [SwiGLU.md](SwiGLU.md) | 为什么从 ReLU 过渡到 SwiGLU？Swish 比 ReLU 好在哪里？ |
| 4 | [SwiGLU.md §Part III](SwiGLU.md#第三部分权重初始化) | W2 为什么需要特殊的 1.414 缩放？std=0.02 的由来？ |
| 5 | [swiglu_ffn.py](swiglu_ffn.py) | 三矩阵合并（w13）如何省掉一次 GEMM？`chunk` 怎么用？ |
| 6 | [ffn_engineering.md](ffn_engineering.md) | 为什么要去掉偏置？`hidden_dim` 怎么对齐到 256 倍数？`mul_()` 和 `del` 在训练和推理中行为有何不同？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 运行 + 学习路径 + 链接 |
| **`{topic}.md`** | 深度知识库：理论 → 实现 → 工程 → 面试，按 Part I/II/III 分层 |
| **代码注释** | 中文 docstring + 张量形状标注 `# [B, L, D]` |
| **冒烟测试** | `swiglu_ffn.py` 自带 `__main__` 块，包含参数量守恒、梯度回传、等价性验证 |
| **交叉链接** | 三个 md 文件顶部互相链接，每个文件底部有目录文件索引 |

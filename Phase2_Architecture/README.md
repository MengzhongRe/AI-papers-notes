# Phase 2：推理加速与显存管理

从零实现 LLM 推理加速的核心算子：Online Softmax 分块注意力、KV Cache/GQA 状态管理、PagedAttention 虚拟内存映射、RadixAttention 前缀树缓存、MLA 低秩注意力、W8A8 量化。

## 目录结构

```
Phase2_Architecture/
├── README.md                        # 本文件 — 目录索引与学习路线
├── FlashAttention/                  # Online Softmax 分块注意力
│   ├── README.md
│   ├── online_softmax.py
│   ├── flash_attention_forward.py
│   └── flash_attention_notes.md
├── Stateful_KV_GQA/                 # 状态化 KV Cache + 分组查询注意力
│   ├── README.md
│   ├── gqa_attention_base.py
│   ├── grouped_query_attention.py
│   ├── stateful_kvcache_gqa.py
│   └── kv_cache_prefill_decode_notes.md
├── PagedAttention/                  # vLLM 物理块表 + 零拷贝碎片化 Attention
│   ├── README.md
│   ├── paged_attention.py
│   └── paged_attention_notes.md
├── RadixAttention/                  # SGLang 前缀树缓存 + LRU 驱逐
│   ├── README.md
│   ├── radix_attention.py
│   ├── radix_code_walkthrough.md
│   ├── radix_attention_notes.md
│   ├── radix_tree_structure.png
│   └── cache_lifecycle.png
├── MLA/                             # DeepSeek-V2 多头潜在注意力
│   ├── README.md
│   ├── multi_head_latent_attention.py
│   ├── causal_mask_padding_mask.py
│   ├── mla_paper_notes.md
│   ├── mla_math_proofs.md
│   ├── mla_code_walkthrough.md
│   ├── block_matmul.md
│   └── mla_notes.md
└── W8A8/                            # 对称/非对称量化 + SmoothQuant
    ├── README.md
    ├── quant_primitives.py
    ├── w8a8_gemm_mock.py
    ├── smooth_quant.py
    ├── SmoothQuant_Paper_Notes.md
    └── run_all_tests.py
```

## 组件速览

| 子目录 | 一句话定位 | 核心面试锚点 |
| :--- | :--- | :--- |
| [FlashAttention/](FlashAttention/) | 双层循环模拟分块 Online Softmax，不写 CUDA 理解 FlashAttention | "为什么 FlashAttention 能变快？—— IO 感知，减少 O(N²) 显存读写" |
| [Stateful_KV_GQA/](Stateful_KV_GQA/) | GQA KV 头广播 + Prefill/Decode 两阶段 KV Cache | "Prefill 和 Decode 的计算特性有什么不同？—— GEMM vs GEMV" |
| [PagedAttention/](PagedAttention/) | vLLM 核心：物理块池 + BlockTable + 碎片化 Online Softmax | "vLLM 吞吐量提升来自哪里？—— 显存近乎零浪费，Batch Size 翻倍" |
| [RadixAttention/](RadixAttention/) | SGLang 前缀树跨请求 KV Cache 共享 + LRU 驱逐 | "RadixAttention 和 PagedAttention 什么关系？—— 上层缓存管理 vs 底层物理机制" |
| [MLA/](MLA/) | DeepSeek-V2 低秩 KV 压缩 + 权重吸收 | "MLA 如何打破 KV 缓存僵局？—— latent cache 比 GQA 还小数十倍" |
| [W8A8/](W8A8/) | INT8 量化原语 + 离群值灾难 + SmoothQuant 难度迁移 | "量化为什么掉精度？—— outlier 拉大 scale，正常值被抹平" |

## 建议学习路径

| 顺序 | 目录 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [FlashAttention/](FlashAttention/) | GPU 为什么用分块做矩阵乘法？Online Softmax 怎么用局部统计量更新全局？ |
| 2 | [Stateful_KV_GQA/](Stateful_KV_GQA/) | KV Cache 为什么是 O(N³)→O(N²) 的关键？Prefill vs Decode 算术强度差多大？ |
| 3 | [PagedAttention/](PagedAttention/) | 显存碎片怎么被页表消除？不连续物理块上怎么做流式 Attention？ |
| 4 | [RadixAttention/](RadixAttention/) | 前缀树怎么跨请求复用 KV Cache？LRU 驱逐 vs LFU 为什么选 LRU？ |
| 5 | [MLA/](MLA/) | 低秩压缩怎么把 KV Cache 压到极小？吸收矩阵怎么绕过显式恢复 K/V？ |
| 6 | [W8A8/](W8A8/) | INT8 量化怎么算 scale/zero-point？SmoothQuant 怎么把难度从 A 迁移到 W？ |

## 快速运行

```bash
# FlashAttention
python FlashAttention/online_softmax.py
python FlashAttention/flash_attention_forward.py

# Stateful KV Cache & GQA
python Stateful_KV_GQA/grouped_query_attention.py
python Stateful_KV_GQA/stateful_kvcache_gqa.py

# PagedAttention
python PagedAttention/paged_attention.py

# RadixAttention
python RadixAttention/radix_attention.py

# MLA
python MLA/causal_mask_padding_mask.py
python MLA/multi_head_latent_attention.py

# W8A8
python W8A8/run_all_tests.py
```

## 文档约定

每个子目录遵循统一规范：

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 + 快速运行 + 交叉链接 |
| **`{topic}.py`** | 手撕代码，自带 `if __name__ == '__main__':` 冒烟测试（含断言） |
| **`{topic}_notes.md`** | 深度知识库：Part I 为什么 → Part II 怎么做 → Part III 工程细节 → Part IV 面试 |
| **代码注释** | 中文注释 + 张量形状流转标注（如 `# [B, H, N, d]`） |

## 前置背景

Phase 2 的核心矛盾是 **显存墙 (Memory Wall)**：推理逐 Token 生成时，矩阵退化为向量，GPU 搬运数据的速度远慢于计算速度。谁能减少显存读写、压缩 KV Cache 体积，谁就是推理之王。

- **Compute-bound**：训练/Prefill 时矩阵乘法大，GPU 算力是瓶颈
- **Memory-bound**：Decode 时每次只输入 1 个 Token，GPU 90%+ 时间在等数据
- **结论**：Phase 2 所有优化（FlashAttention / GQA / PagedAttention / MLA / W8A8）都在回答同一个问题 —— 如何少搬数据、搬快数据

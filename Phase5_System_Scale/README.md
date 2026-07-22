# Phase 5: 宏观系统与分布式底座

> 关联文档：[PLAN.md](PLAN.md) · [../README.md](../README.md)
> 关联子目录：[System_Math](System_Math/) · [Collectives](Collectives/) · [Tensor_Parallel](Tensor_Parallel/) · [Pipeline_Parallel](Pipeline_Parallel/) · [ZeRO](ZeRO/) · [Ring_Attention](Ring_Attention/) · [Expert_Parallel](Expert_Parallel/) · [Parallelism_Recipe](Parallelism_Recipe/)

---

## 一句话定位

Phase 5 回答一个核心问题：**当单卡装不下模型时，怎么用多张 GPU 一起算？** 从「算一笔账需要多少 FLOPs 和显存」开始，到理解四种并行策略（TP/PP/DP/EP）的原理和组合方式，最终能对一个给定的模型和集群给出合理的策略配置。承上（Phase 2 单卡推理瓶颈 → 为什么需要多卡）启下（Phase 6 拼装微型大模型 → 理解它背后的训练基础设施是怎么运转的）。

---

## 目录树

```
Phase5_System_Scale/
├── README.md                         # 本文件 — 模块速览 + 面试视角 + 联动索引
├── PLAN.md                           # 逐日执行指南（16 天，含资源标注）
├── System_Math/                      # Day 61-62: FLOPs 推演 + 显存建模
│   └── system_math.ipynb
├── Collectives/                      # Day 63-64: 通信原语 + Ring-AllReduce
│   ├── ring_allreduce.py
│   └── collectives_notes.md
├── Tensor_Parallel/                  # Day 65-66: Megatron-LM 风格 TP
│   ├── tp_linear.py
│   └── tp_notes.md
├── Pipeline_Parallel/                # Day 67-68: GPipe / 1F1B / DualPipe
│   └── pipeline_notes.md
├── ZeRO/                             # Day 69-70: ZeRO-1/2/3 显存模拟
│   ├── zero_sim.py
│   └── zero_notes.md
├── Ring_Attention/                   # Day 71-72: 环形 KV 传递 + Online Softmax
│   ├── ring_attention.py
│   └── ring_attention_notes.md
├── Expert_Parallel/                  # Day 73-74: MoE 分布式路由
│   ├── expert_parallel.py
│   └── expert_parallel_notes.md
└── Parallelism_Recipe/               # Day 75-76: 混合策略推演 + 工业前沿
    └── parallelism_recipe_notes.md
```

---

## 学习路径图

模块按依赖关系排列（从上到下，箭头表示前置知识）：

```
System_Math (FLOPs + Memory)
    │
    ├──→ Collectives (通信原语)
    │        │
    │        ├──→ Tensor_Parallel (TP)
    │        │        │
    │        │        └──→ Pipeline_Parallel (PP)
    │        │                 │
    │        │                 └──→ 三者并列，无严格依赖
    │        │
    │        ├──→ ZeRO (显存优化) ──→ 可独立学习，也可与 TP/PP 并行
    │        │
    │        ├──→ Ring_Attention ──→ 需要 Online Softmax (Phase 2) + 通信基础
    │        │
    │        └──→ Expert_Parallel ──→ 需要 MoE (Phase 1) + All-to-All (Day 63)
    │
    └──→ Parallelism_Recipe (混合推演) ──→ 依赖以上所有模块
```

**建议阅读顺序**：严格按 Day 61→76 走，因为 Collectives 是 TP/ZeRO/Ring Attention 的共同地基，不能跳过。

---

## 模块速览表

| 模块                 | 天数        | 定位                           | 学习方式                               | 难度    | 面试相关度 |
| ------------------ | --------- | ---------------------------- | ---------------------------------- | ----- | ----- |
| System_Math        | Day 61-62 | 算力账 + 显存账——知道为什么要分布式         | 📐 Jupyter 数学推导                    | ⭐⭐⭐   | ⭐⭐⭐⭐⭐ |
| Collectives        | Day 63-64 | 多卡对话的语言——6 种通信操作的语义          | 💻 手撕 Ring-AllReduce + 📐 带宽直觉     | ⭐⭐⭐   | ⭐⭐⭐⭐  |
| Tensor_Parallel    | Day 65-66 | 切单层矩阵——最优雅的并行                | 💻 ColumnParallel / RowParallel 手撕 | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ |
| Pipeline_Parallel  | Day 67-68 | 切层序列——跨节点扩展的核心               | 📐 Gantt 图 + 泡率公式推导                | ⭐⭐⭐⭐  | ⭐⭐⭐⭐  |
| ZeRO               | Day 69-70 | 省显存的三级递进——从切 Adam 到切参数       | 💻 Dict 模拟显存变化                     | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ |
| Ring_Attention     | Day 71-72 | 序列维度并行——长文本的解决方案             | 💻 Online Softmax + 环形 KV          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  |
| Expert_Parallel    | Day 73-74 | MoE 的规模化——每个 GPU 只持部分 expert | 💻 All-to-All dispatch 手撕          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  |
| Parallelism_Recipe | Day 75-76 | 四维策略混合 → 最佳配置                | 📐 案例推演 + 阅读                       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 面试视角速查

### FLOPs & Memory

**Q: 「7B 模型 bf16 训练，单卡 A100-80G 能跑吗？」**
> 参数 14GB + 梯度 14GB + Adam(fp32 m+v+master) 84GB = 模型状态 112GB → 已超 80GB，还没算激活值。引出 ZeRO/Activation Ckpt 的必要性。关键是展示推导过程，不只报数字。

### Ring-AllReduce

**Q: 「AllReduce 的通信量和 GPU 数的关系？」**
> 每卡发 2(N-1)/N × D ≈ 2D，几乎不随 N 增长。总通信量 2(N-1)D 分布到 N 张卡。这是 Ring-AllReduce 能 scale 到千卡的根本原因。

### Tensor Parallelism

**Q: 「TP 为什么只在节点内用？」**
> 每层前向 2 次 AllReduce，通信量 ∝ B×S×d。跨节点 InfiniBand (400 GB/s) 只有 NVLink (900 GB/s) 的 1/2，通信成为瓶颈。结论：TP 在 NVLink 域内（≤8 卡）使用。

**Q: 「MLP 为什么是 ColumnParallel → RowParallel 而不是反过来？」**
> ColumnParallel 的 partial 输出 [B,S,d_ff/N] 正好是 RowParallel 需要的列切分输入 → 中间省一次 AllReduce。反过来要多一次。

### Pipeline Parallelism

**Q: 「PP 的泡率怎么算？从 GPipe 到 DualPipe 优化的主线是什么？」**
> 泡率 = (P-1)/(P-1+M)。增加 M 降低泡率但增大显存。优化主线是「填缝」：1F1B 交错插入反向 → Interleaved 细化粒度 → ZB 拆分反向 → DualPipe 双向注入。

### ZeRO

**Q: 「ZeRO-3 为什么通信量是 DP 的 1.5×？」**
> 多了 2 次前向 AllGather（前后向各拼一次参数，每次 1/Np × 全量）。加反向 ReduceScatter(1×)。总计 ≈ 1+2/Np ≈ 1×(Np 大时)。Np=8 约 1.25×。

**Q: 「Activation Checkpointing 和 ZeRO 分别省什么？为什么能一起用？」**
> AC 省激活值（ZeRO 不管），ZeRO 省模型状态（AC 不管）。互不重叠，组合使用。

### Ring Attention

**Q: 「为什么 Ring Attention 需要 Online Softmax？」**
> KV 块分批到达，每块独立 softmax 分母不同。需要 (m,l,O) 三元组维护全局状态，新块到达时做修正。不做则各块 softmax 无法拼出正确 overall attention。

### Expert Parallelism

**Q: 「EP 和 TP 的区别？为什么 MoE 层用 EP？」**
> TP 每层所有 GPU 参与（密集 AllReduce），EP 每 token 只去 1-2 expert 所在 GPU（稀疏 All-to-All）。MoE 用 TP 会导致非 expert 的 GPU 做无用计算。

**Q: 「为什么 DeepSeek-V3 训练只要 $5.57M？」**
> MoE 每个 token 只激活 ~37B/671B + DualPipe MFU 51% + FP8 blockwise + Aux-loss-free balancing。

### 混合并行

**Q: 「64 张 A100 训 175B 模型，设计并行策略」**
> Step 1 显存：175B×2=350GB → 必须 TP。TP=8 节点内 → 每卡 43.75GB(fit)。Step 2 跨节点：8 节点 × 8 卡 = 64，PP=8 跨节点。Step 3 ZeRO：Adam 84GB/卡 → ZeRO-3 进一步压缩。最终：TP=8 + PP=8 + ZeRO-3 = 64 GPU。展示推导过程的系统性。

---

## 跨 Phase 联动

| Phase 5 模块 | 依赖的先前内容 |
|-------------|--------------|
| Ring Attention (Day 71) | **Phase 2** `FlashAttention/online_softmax.py` — (m,l,O) Online Softmax 更新逻辑 |
| Expert Parallelism (Day 73) | **Phase 1** `MoE/moe_layer.py` — 单卡 Top-K routing + load balancing loss |
| Tensor Parallelism (Day 65) | **Phase 1** `Attention/` + `FFN/` — 理解 MLP 和 Attention 的矩阵运算结构 |

---

## 快速运行指令

```bash
# 通信原语
cd Collectives && python ring_allreduce.py

# Tensor Parallelism
cd Tensor_Parallel && python tp_linear.py

# ZeRO 显存模拟
cd ZeRO && python zero_sim.py --model 7B --gpus 8

# Ring Attention
cd Ring_Attention && python ring_attention.py

# Expert Parallelism
cd Expert_Parallel && python expert_parallel.py
```

每个 `.py` 自带 `if __name__ == '__main__':` 冒烟测试，验证与参考实现的数值对齐。

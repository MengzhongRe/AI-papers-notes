# Phase 5 分布式系统 — 逐日执行计划

> **属于**：[README.md](README.md) Phase 5 模块
> **目录**：`Phase5_System_Scale/`
> **目标**：建立对 FLOPs、显存墙、多卡并行通信的架构师级物理直觉，能对给定模型和集群给出合理的并行策略配置

---

## 16 天总览

```
Part 1（Day 61-64）：地基 — 算力账、显存账、通信账
Part 2（Day 65-70）：模型切分 — TP / PP / ZeRO
Part 3（Day 71-74）：序列与专家 — Ring Attention / Expert Parallelism
Part 4（Day 75-76）：整合 — 混合策略推演 + 知识收束
```

|  天  | 日期  | 主题                                                |     类型      | 预计时间 | 状态  |
| :-: | --- | ------------------------------------------------- | :---------: | :--: | :-: |
| 61  | —   | FLOPs 推演 — Transformer 前向计算量                      |   📐 数学推导   |  3h  |  ⬜  |
| 62  | —   | 显存建模 — 为什么单卡装不下                                   |   📐 数学推导   |  3h  |  ⬜  |
| 63  | —   | 通信原语 — 6 种集合通信 + Ring-AllReduce 手撕                |    💻 编码    |  3h  |  ⬜  |
| 64  | —   | 通信带宽直觉 — 计算/通信比率                                  |  📐 推导+💻   | 2.5h |  ⬜  |
| 65  | —   | Tensor Parallelism — ColumnParallel + RowParallel |    💻 编码    |  3h  |  ⬜  |
| 66  | —   | TP 进阶 — 完整 Transformer Block 切分                   | 💻 编码+📐 画图 |  3h  |  ⬜  |
| 67  | —   | Pipeline Parallelism — GPipe 与泡率公式                |  📐 画图+公式   |  3h  |  ⬜  |
| 68  | —   | PP 进阶 — 1F1B · Interleaved · DualPipe             |  📐 画图+阅读   |  3h  |  ⬜  |
| 69  | —   | ZeRO — 三级显存递进手推                                   | 💻 Dict 模拟  |  3h  |  ⬜  |
| 70  | —   | ZeRO 实战 — FSDP2 + 组合策略                            |  💻 编码+阅读   | 2.5h |  ⬜  |
| 71  | —   | Ring Attention — 核心算法手撕                           |    💻 编码    | 3.5h |  ⬜  |
| 72  | —   | Ring Attention — Causal 变体 + Ulysses 对比           |  💻 编码+阅读   |  3h  |  ⬜  |
| 73  | —   | Expert Parallelism — MoE 分布式路由                    |    💻 编码    |  3h  |  ⬜  |
| 74  | —   | EP 进阶 — Capacity Factor + DeepSeek 实战             |  💻 编码+阅读   |  3h  |  ⬜  |
| 75  | —   | 混合并行 — 策略推演                                       |   📐 案例推演   |  3h  |  ⬜  |
| 76  | —   | 收束 — 知识地图 + 工业前沿巡览                                |  📐 画图+写作   | 2.5h |  ⬜  |

---

## Part 1：地基 — 算力账、显存账、通信账（Day 61-64）

> 不知道单步花多少计算、占多少显存、通信要多久，就没法判断并行策略是赚是亏。这四天建立最底层的物理直觉。

---

### Day 61 — FLOPs 推演：Transformer 一次前向的计算量

**目标**：手推 LLaMA-7B 一次前向的精确 FLOPs 分解。理解 `6ND` 公式中每个因子的物理含义。知道 Attention 和 FFN 各占多少比例、seq_len 变化时谁成为瓶颈。

**预计时间**：3 小时（阅读 30min + Jupyter 推导 2h + 思考 30min）

---

**任务清单**：

#### 第一步：读 Transformer Math 博客（约 20 分钟）

- [x] **阅读** [Transformer Math (EleutherAI)](https://blog.eleuther.ai/transformer-math/)（必读，全文约 15 分钟）
  - 阅读时关注三个数字的来源：
    - matmul FLOPs = 2×M×K×N —— 为什么有个 2？（一次乘一次加）
    - 前向总 FLOPs ≈ 2ND —— D 是训练 tokens，N 是参数量。为什么系数是 2 而不是 4？
    - 训练总 FLOPs ≈ 6ND —— 为什么反向 ≈ 前向的 2 倍？
  - 不用记推导细节，但要能用一句话说清楚「6ND 里的 2 和 4 怎么来的」

- [ ] **选读** [Scaling Laws (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) §2.1-2.2（15 分钟）
  - 不看实验细节，只看 Kaplan 是怎么从参数 N 和 tokens D 推到总计算量 C 的
  - 对比 [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)（5 分钟扫一眼 Abstract + 图 1）
    - 核心修正：Kaplan 说模型大小比数据重要，Chinchilla 说参数和 tokens 应该等比增长。知道这个争议就够了

#### 第二步：Jupyter 推导（约 2 小时）

- [x] **打开 `System_Math/compute_flops_notes.md` 阅读笔记**，对照推导 Jupyter notebook
  - 写出每一层的矩阵形状：
    - Q/K/V/O：各 4096×4096
    - gate/up：各 4096×14336
    - down：14336×4096
    - embedding + lm_head：vocab_size×4096
  - 算单层总参数量 = 4×4096² + 3×4096×14336 ≈ 243M（不含 embedding）

- [x] **创建第二节「前向 FLOPs 分解」**
  - 逐一推导 Attention 的 4 部分 FLOPs（代入 B=1, S=2048）：
    - QKV 投影 = 2 × B × S × 4096 × (3×4096)
    - Attention Scores = 2 × B × S² × 4096（Q×K^T 的 matmul，S² 是关键的瓶颈）
    - Weighted Sum = 2 × B × S² × 4096（softmax(A)×V）
    - Output 投影 = 2 × B × S × 4096 × 4096
  - 逐一推导 FFN 的 3 部分 FLOPs：
    - gate_proj = 2 × B × S × 4096 × 14336
    - up_proj = 同上
    - down_proj = 2 × B × S × 14336 × 4096
  - **关键对比**：S=2048 时 Attention FLOPs vs FFN FLOPs 各占百分之多少？

- [x] **创建第三节「S 变化时的瓶颈转移」**
  - 分别代入 S=512 / 2048 / 8192 / 32768
  - 画一个堆叠柱状图：不同 S 下 Attention FLOPs vs FFN FLOPs 的占比变化
  - **关键洞察**：S=512 时 FFN 占 > 80%（短序列瓶颈在 FFN）。S=32768 时 Attention 占比爆炸（长序列瓶颈在 Attention）。这个 flip 点在哪里？

- [x] **创建第四节「训练总 FLOPs」**
  - 推导 C_fwd ≈ 2ND
  - 推导 C_bwd ≈ 4ND（weight grad 需 2ND + input grad 需 2ND）
  - 推导 C_total ≈ 6ND
  - 代入 N=7B, D=2T 算总量，与 GPT-3 论文的 3.14×10^23 FLOPs 对照

#### 第三步：思考与收束（约 30 分钟）

- [x] **思考题**（不强制写下来，但认真想）：
  1. 为什么 scaling laws 中 embedding 层的 FLOPs 被忽略？（查表操作不是 matmul，FLOPs 占比 < 0.1%）
  2. 如果你要训练 1T tokens 的 70B 模型，最少需要多少 H100-GPU-hours？（用 6ND 公式 + H100 的 989 TFLOPS bf16）
  3. 这个数字和你记忆中的 Llama 2 70B 训练时间一致吗？

**完成标准**：
- `compute_flops_notes.md` 有完整 FLOPs 推导（参数表 / 前向分解 / S 变化 / 训练总量）
- 能脱稿说出 6ND 中 2 和 4 的物理来源
- 能说出 S=2048 时 Attention 和 FFN 各占 FLOPs 的大致比例（约 28% vs 72%）

---

### Day 62 — 显存建模：为什么单卡装不下

**目标**：手推 7B 模型 Adam+bf16 训练时的精确显存占用。算出具体数字，让「单卡装不下」不再是一个抽象结论，而是一道可以逐项验证的账本。

**预计时间**：3 小时（阅读 30min + Jupyter 推导 2h + 思考 30min）

---

**任务清单**：

#### 第一步：读 ZeRO 论文的显存分析（约 30 分钟）

- [x] **精读** [ZeRO Paper](https://arxiv.org/abs/1910.02054) §2「Where Did All the Memory Go?」（必读，20 分钟）
  - **图 1**：显存分解饼图——这张图背下来。参数(青色)、梯度(橙色)、Adam m(绿色)、Adam v(红色)、激活值(蓝色)
  - **表 1**：逐项数据和百分比——把数字抄到你的 notebook 里
  - 读完问自己：为什么 Activation（蓝色）的占比随模型规模变化？什么情况下 Activation 反而比 Adam 状态占得多？
  - 跳过 §3（DP 基础）——你已经懂了
  - **选读** [Gradient Checkpointing (HF Docs)](https://huggingface.co/docs/transformers/v5.6.2/grad_checkpointing) 或 [Ultra-scale Playbook Notes](https://desh2608.github.io/2025-10-30-ultrascale-notes/)（5 分钟）
    - 理解一句话：每隔 N 层存一次激活值 → 反向时重算 → 显存省了，FLOPs 多了 33%

#### 第二步：Jupyter 显存账单（约 2 小时）

- [x] **阅读 `System_Math/memory_modeling_notes.md` 做对照**
  - 模型状态（Model States）：
    - 参数 bf16：7B × 2 bytes = 14 GB
    - 梯度 bf16：14 GB
    - Adam m(fp32)：7B × 4 bytes = 28 GB
    - Adam v(fp32)：28 GB
    - master weights(fp32)：28 GB
    - **模型状态合计 = 112 GB**——已经超过 A100 80G！还没算激活值
  - Residual States（激活值 + 临时缓冲）：
    - 单层 SwiGLU FFN 的中间激活 ≈ B×S×14336×2 bytes
    - 32 层 × 上面这个数 + residual states + RMSNorm 输入 ≈ ?
    - **总计 > 120 GB**
  - 画一个和 ZeRO 论文图 1 一样的饼图

- [ ] **创建第二节「什么配置刚好溢出？」**
  - 固定 seq_len=2048，依次代入 bs=1/2/4/8/16/32 → 画柱状图
  - 固定 bs=1，依次代入 S=512/1024/2048/4096/8192 → 画柱状图
  - 标出「A100-80G 红线」——哪些配置能跑、哪些溢出
  - **关键洞察**：bs=1, S=2048 时模型状态（112GB）本身就超了，激活值还没算！所以 ZeRO 要解决的首敌是 Adam 状态，不是激活值

- [ ] **创建第三节「加入 Activation Checkpointing」**
  - 假设每 2 层设 1 个 checkpoint → 激活值降为原来的 1/2
  - 假设每层都 checkpoint（极致）→ 激活值接近 0，但重算 FLOPs +33%
  - 画出显存-FLOPs trade-off 曲线
  - **关键洞察**：AC 省激活值，但模型状态 112GB 纹丝不动——所以 AC 从来不是单卡训练的解药，只是 ZeRO/TP 的补充

- [ ] **创建第四节「Gradient Accumulation」**
  - 推导：peak memory 不变，但 effective batch_size = steps × micro_batch_size
  - 为什么有用？（不需要存多份激活值，因为微批次是串行的）
  - 一个具体的例子：想达到 effective bs=64，但单卡只能 bs=2 → GA=32 → peak memory 依然是 bs=2 的量

- [ ] **选做**：在 notebook 里加 `ipywidgets` 交互 slider——拖动 bs 和 seq_len，实时更新饼图

#### 第三步：思考与收束（约 30 分钟）

- [ ] **思考题**：
  1. 为什么推理（inference）的显存远少于训练？（不需要存梯度 + Adam + 激活值。推理只需要参数 14GB + KV Cache）
  2. 如果 batch_size=1 时单卡依然跑不了训练，问题出在哪？问题出在 Adam——它占了 84GB，而单卡参数才 14GB！ZeRO 的第一刀一定是砍向 Adam

**完成标准**：
- `system_math.ipynb` 有 4 节完整推导（逐项账单 / 溢出条件 / AC 模拟 / GA 推导）
- 能脱稿说出 7B 模型训练的显存占用五项和各自数量级
- 能解释「为什么 AC + ZeRO-3 是互补组合」——AC 省激活值，ZeRO-3 省模型状态，互不重叠

---

### Day 63 — 通信原语：6 种集合通信 + Ring-AllReduce 手撕

**目标**：理解 NCCL 的 6 种集合通信操作语义，手撕 Ring-AllReduce 的两阶段算法。这是 TP/ZeRO/Ring Attention 的共同地基——不先掌握通信，后面每个并行策略都是「知其然不知其所以然」。

**预计时间**：3 小时（画图+阅读 50min + 编码 1.5h + 验证+推导 40min）

---

**任务清单**：

#### 第一步：画图理解 6 种集合通信（约 30 分钟）

- [ ] **先在纸上画出 6 种集合通信的数据流向**（不写代码，先画图）：
  - AllReduce：每个 GPU 输入不同数据 → **每个 GPU 输出相同结果**
  - AllGather：每个 GPU 输入不同数据 → **每个 GPU 输出所有数据的拼接**
  - ReduceScatter：每个 GPU 输入不同数据 → **每个 GPU 输出结果的 1/N 分片**
  - Broadcast：1 个 GPU 输入数据 → **所有 GPU 输出相同数据**
  - Scatter：1 个 GPU 输入数据 → **切成 N 片分给 N 个 GPU**
  - All-to-All：每个 GPU 的 N 个 chunks → **分别发给所有 N 个 GPU**
  - 每张图标注：每个 GPU 手里有什么（输入） → 每个 GPU 最后得到什么（输出）。能区分这 6 种通信的输入输出差异

- [ ] **阅读** [BaPipe Ring AllReduce 图解](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)（必读，10 分钟）
  - 重点看图 2 和图 3：Scatter-Reduce 阶段（环形传递 + 部分和累加）和 AllGather 阶段（环形传递最终结果）
  - 每个 GPU 在每一步(step) 只和邻居 GPU 通信——不是 all-to-all 广播

- [ ] **快速浏览** [NCCL User Guide - Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)（5 分钟）
  - 确认你画的 6 张图和官方文档一致

#### 第二步：手撕 Ring-AllReduce（约 1.5 小时）

- [ ] **创建 `Collectives/ring_allreduce.py`**
  - 实现 `ring_allreduce(tensor, rank, world_size)` 函数
  - 两阶段各 world_size-1 轮：
    - **Scatter-Reduce**：每轮 send(当前累加值 → next_rank) + recv(partial_sum ← prev_rank) + 累加到自己的值
    - **AllGather**：每轮 send(最终结果 → next_rank) + recv(邻居的结果 ← prev_rank)
  - 用 `torch.distributed.send/recv` 实现
  - 用 `torch.multiprocessing` 启动 world_size 个进程模拟多卡

- [ ] **在 `__main__` 中验证**：
  - 4 卡模拟（world_size=4），随机张量
  - 与 `torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)` 对比
  - 打印每轮每张卡的中间张量值，可视化 Scatter-Reduce 和 AllGather 两阶段

- [ ] **推导通信量公式**：
  - 每卡在每个 phase 发 (N-1) × (D/N) = (N-1)D/N 数据
  - 两个 phase → 2(N-1)D/N
  - N 大时 ≈ 2D → **每卡发 2 份全量数据，不随 GPU 数增长**

#### 第三步：输出笔记（约 20 分钟）

- [ ] **写 `Collectives/collectives_notes.md`**（不需要长篇，结构清晰即可）：
  - 第 1 节：6 种集合通信的输入输出关系图
  - 第 2 节：Ring-AllReduce 两阶段的数据流推导
  - 第 3 节：通信量公式推导 + "N 大时 ≈ 2D"的含义

- [ ] **选读** [NCCL 源码 `src/collectives.cc`](https://github.com/NVIDIA/nccl/blob/master/src/collectives.cc)（10 分钟）
  - 搜索 `ncclAllReduce` 函数，看 Ring 和 Tree 两种路径的选择逻辑
  - 这不是为了读懂 C++，而是建立信心：「那些高级 API 底层确实是我手写的 Ring」

**完成标准**：
- 能不看笔记画出 6 种集合通信的输入输出
- `ring_allreduce.py` 跑通，与 `torch.distributed.all_reduce` 结果一致
- 能推导通信量 = 2(N-1)/N × D
- 运行命令：`cd Collectives && python ring_allreduce.py`

---

### Day 64 — 通信带宽直觉：从纸上走到物理世界

**目标**：理解 HBM > NVLink > InfiniBand > PCIe 的带宽数量级差异，掌握计算/通信比的计算方法，建立「为什么 TP 只在节点内、PP 能跨节点」的物理直觉。

**预计时间**：2.5 小时（阅读 20min + 计算推导 1h + 编码完善 1h + 思考 20min）

---

**任务清单**：

#### 第一步：建立带宽数量级直觉（约 20 分钟）

- [ ] **阅读** [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)（5 分钟）
  - 只找带宽数字：HBM3e 带宽、NVLink 4.0 带宽、FP8 TFLOPS
  - **背下这四个数字**：
    - HBM3e（H100 片内）：3.35 TB/s
    - NVLink 4.0（节点内，单向）：900 GB/s
    - InfiniBand NDR400（跨节点）：400 GB/s
    - PCIe 5.0 ×16：64 GB/s

- [ ] **选读** [NVIDIA DGX H100 Network Architecture](https://docs.nvidia.com/dgx/dgxh100-user-guide/network-architecture.html)（10 分钟）
  - 理解 NVSwitch：节点内 8 卡不是环形连接，而是通过 NVSwitch 全互联
  - 为什么 NVSwitch 域外还是要回到 Ring/Tree？跨节点没有 NVSwitch

#### 第二步：计算/通信比率（约 1 小时）

- [ ] **以 7B 模型的一个 FFN 层为例，算两个数字**：

  1. **matmul 计算时间**：
     - gate_proj FLOPs = 2 × B × S × 4096 × 14336
     - H100 bf16 = 989 TFLOPS(non-sparse) → 时间 = FLOPs / 989e12
     - 代入 B=1, S=2048 → 约 0.24 ms

  2. **AllReduce 通信时间**：
     - 激活值大小 = B × S × 4096 × 2 bytes(bf16) = 1 × 2048 × 4096 × 2 ≈ 16 MB
     - NVLink 内：16MB / 900GB/s ≈ 18 μs
     - InfiniBand 跨节点：16MB / 400GB/s ≈ 40 μs（+ overhead 可能 100-200 μs）

  3. **比值**：
     - 节点内 NVLink：0.24ms / 0.018ms ≈ 13:1 → TP 是合算的
     - 跨 InfiniBand：0.24ms / 0.1ms ≈ 2.4:1 → TP 开始悬了
     - 跨 PCIe：0.24ms / 0.25ms ≈ 1:1 → TP 完全不可用

  **这就是为什么 TP 只在节点内、PP 能跨节点的物理根源。**

- [ ] **扩展思考**：如果 B=8（大 batch），计算/通信比怎么变？
  - 计算量 ∝ B，通信量 ∝ B → 比值不变？不对——FLOPs 中的 S² 项不随 B 增长……但这里的通信是 AllReduce 激活值，大小 ∝ B×S×d。计算量中的 matmul 也 ∝ B×S×d。所以比例大致不变。**真正的变化来自 attention scores（O(S²)不随 B 增长但通信也不含它）。这就是为什么长序列场景下计算/通信比更好——多出来的计算是本地 S²，不需要通信。**

- [ ] **Ring vs Tree 的选择逻辑**：
  - Ring：延迟 ∝ N，每步只传 D/N → 适合小消息 + 多 GPU
  - Tree：延迟 ∝ log(N)，大消息带宽更优 → 适合大消息 + 多节点
  - NCCL 的阈值约 256KB——小于它用 Ring，大于它用 Tree

#### 第三步：完善代码（约 1 小时）

- [ ] **完善 `ring_allreduce.py`**：
  - 添加通信量统计：记录每轮 send/recv 的字节数
  - 添加计时功能：记录两个 phase 各耗时多少
  - `__main__` 中对比不同 world_size（2/4/8）下每卡通信量和耗时

- [ ] **写一个简单的 `Collectives/comm_calc_ratio.py`**：
  - 输入：模型大小、batch_size、seq_len、GPU 型号
  - 输出：计算时间 / 通信时间（NVLink 内）+ 跨 InfiniBand 的比值
  - 判断结论：「TP 在 NVLink 域内合算 / 跨节点不合算」

- [ ] **选做**：从 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 拉代码跑 `all_reduce_perf`
  - 单卡也能跑（单卡 all_reduce 就是本地 memcpy 的 aliasing），但能看到带宽理论峰值
  - 如果有 2+ GPU 的机器，跑真实多卡通信测带宽

#### 第四步：思考与收束（约 20 分钟）

- [ ] **面试视角**——自问自答：
  > Q: "AllReduce 的通信量随 GPU 数量怎么变化？"
  > A: 每卡发 2(N-1)/N × D ≈ 2D，不随 N 增长。总通信量 2(N-1)D 分布在 N 张卡上，单卡带宽压力恒定。这是 Ring-AllReduce 能 scale 到千卡的根本原因。对比参数服务器：中心节点带宽 O(N)，N 大时中心节点被撑爆。

**完成标准**：
- 能默写 HBM/NVLink/InfiniBand/PCIe 四个带宽数字
- 能随手算出一个场景的 NVLink 内 / InfiniBand 跨节点的计算通信比
- `Collectives/` 目录完整（`ring_allreduce.py` + `collectives_notes.md` + `comm_calc_ratio.py`）

---

## Part 2：模型切分 — TP / PP / ZeRO（Day 65-70）

> 单卡装不下模型，怎么办？三种切法的本质区别：TP 切单层、PP 切层序列、ZeRO 切优化器状态。各解决一个问题、各有自己的 trade-off。

---

### Day 65 — Tensor Parallelism：ColumnParallel + RowParallel 手撕

**目标**：手撕 Megatron-LM 风格的 TP 线性层。理解 MLP 的 ColumnParallel → RowParallel 配对为什么恰好省去中间一次 AllReduce——这是 TP 最精妙的设计。

**预计时间**：3 小时（阅读 45min + 编码 1.5h + 验证+笔记 45min）

---

**任务清单**：

#### 第一步：读 Megatron-LM（约 45 分钟）

- [ ] **精读** [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) §3「Model Parallel Transformers」（必读，20 分钟）
  - **图 2**：ColumnParallel 和 RowParallel 的配对示意图——这张图读懂了，TP 就懂了一半
  - 关注 f 和 g 两个函数：f = identity（ColumnParallel 的前向不收集结果），g = AllReduce（RowParallel 的前向必须要同步）
  - 为什么 ColumnParallel 接 RowParallel 时 f 不用做任何事？因为 partial 输出恰好是 RowParallel 的切分输入

- [ ] **浏览工业源码** [Megatron-LM `layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py)（15 分钟）
  - 搜索 `gather_output` 参数：这个 bool 控制「ColumnParallel 输出时是否 AllGather 完整结果」
  - 搜索 `skip_bias_add`：bias 在分布式下怎么处理——ColumnParallel 的 bias 只加到 partial 输出
  - 不用全读懂，重点是看工业代码做了哪些你还没想到的优化（如 fused AllReduce + GEMM）

- [ ] **选读** [DTensor Tutorial (PyTorch)](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)（10 分钟）
  - 对比你手写的 TP 和 PyTorch 原生 `distribute_tensor` API：你的代码要管理 2 份 W_col[0] 和 W_col[1]，DTensor 只需要一个张量 + placement spec

#### 第二步：手撕 TP 线性层（约 1.5 小时）

- [ ] **创建 `Tensor_Parallel/tp_linear.py`**

  - 实现 `ColumnParallelLinear`：
    ```python
    # W [d_in, d_out] → W[0] [d_in, d_out//TP], W[1] [d_in, d_out//TP]
    # 每卡输入相同 X，各自算 partial Y[i] = X @ W[i]
    # gather_output=True → AllGather(Y[0], Y[1]) 拼成完整输出
    # gather_output=False → 返回 partial Y[i]（给 RowParallel 直接用）
    ```

  - 实现 `RowParallelLinear`：
    ```python
    # W [d_in, d_out] → W[0] [d_in//TP, d_out], W[1] [d_in//TP, d_out]
    # 输入 X 沿 d_in 维切分 → 每卡 X[i] @ W[i] → AllReduce(partial_sum)
    # 注意：X 的切分恰好来自上一层 ColumnParallel(gather_output=False)
    ```

  - AllReduce 模拟：用 `torch.sum([partial_0, partial_1], dim=0)` 标注「此处应为 NCCL AllReduce」

- [ ] **拼一个完整的 TP-MLP**：
  - ColumnParallel(gate, gather_output=False) + ColumnParallel(up, gather_output=False) → SiLU(element-wise) × gate_output → RowParallel(down)
  - **验证关键洞察**：gate/up 的 partial 输出经过 SiLU 乘法后仍然是 [B,S,d_ff/TP] 形状 → 恰好是 RowParallel 的列切分输入 → 中间不需要 AllGather/AllReduce

- [ ] **`__main__` 中验证**：
  - 随机初始化权重
  - TP=2 MLP 前向 vs 完整 `nn.Linear` MLP 前向 → `torch.allclose(atol=1e-5)`
  - 验证 bit-exact 等价性

#### 第三步：输出笔记（约 45 分钟）

- [ ] **写 `Tensor_Parallel/tp_notes.md`**：
  - 第 1 节：ColumnParallel 和 RowParallel 的矩阵切分图示
  - 第 2 节：MLP 完整 TP 路径的数据流——标注「这里省了一次 AllReduce」
  - 第 3 节：通信量分析——每次 AllReduce 传 B×S×d 的激活值

- [ ] **思考题**：
  1. 为什么 ColumnParallel 后面必须是 RowParallel？（反过来 RowParallel → ColumnParallel 会多一次 AllReduce——画一下矩阵切分就清楚了）
  2. GQA（Q heads != KV heads）做 TP 有什么额外限制？

**完成标准**：
- `tp_linear.py` 的 TP=2 MLP 输出与完整 nn.Linear 严格对齐
- 能画出 MLP 的完整 TP 数据流图，标注「这里省了一次 AllReduce」
- 运行命令：`cd Tensor_Parallel && python tp_linear.py`

---

### Day 66 — TP 进阶：完整 Transformer Block 切分

**目标**：把 TP 应用到完整 Block。画出整个 Block 的 TP 数据流图，数清 2 次 AllReduce 的位置。思考推理场景下 TP 为什么退化。

**预计时间**：3 小时（阅读 20min + 画图+编码 1.5h + 分析+思考 1h）

---

**任务清单**：

#### 第一步：读 Megatron-LM §4（约 20 分钟）

- [ ] **阅读** [Megatron-LM](https://arxiv.org/abs/1909.08053) §4「Layout of Entire Model」（15 分钟）
  - **图 3**：对照理解 Attention + FFN + Residual 的完整 TP 布局
  - 注意 LayerNorm 的位置——**Norm 不做 TP**，因为它的输入已经是上一层 RowParallel AllReduce 后的完整副本了
  - 注意 Residual Connection——每卡各自加（因为 RowParallel 输出已经 AllReduce 统一了）

- [ ] **选读** [GQA Paper](https://arxiv.org/abs/2305.13245)（5 分钟扫一眼）
  - GQA 下 KV heads < Q heads → TP 不能超过 KV heads 数。如果 N_kv_heads=4, TP=8，有些卡上没有 KV head

#### 第二步：画图 + 编码（约 1.5 小时）

- [ ] **在纸上画出完整 Block 的 TP 数据流图**：
  ```
  Input (完整) → RMSNorm (完整，每卡各自算)
    → Q: ColumnParallel → [B,S,d_head×N_q/TP] —— 每卡只有部分 Q heads
    → K: ColumnParallel → [B,S,d_head×N_kv/TP] —— 注意 GQA！KV heads 少时切分受限
    → V: ColumnParallel → [B,S,d_head×N_kv/TP]
    → Attention Scores (每卡各自算自己的 heads)
    → O: RowParallel → AllReduce → [B,S,d_model] 完整 ← 第 1 次 AllReduce
    → + Residual (每卡各自加)
    → RMSNorm (完整，每卡各自算)
    → Gate: ColumnParallel(gather_output=False)
    → Up: ColumnParallel(gather_output=False)
    → SiLU × Gate
    → Down: RowParallel → AllReduce → [B,S,d_model] 完整 ← 第 2 次 AllReduce
    → + Residual (每卡各自加)
  ```
  - 标注图中每一处「完整副本」「partial 输出」「AllReduce」

- [ ] **用 Day 65 的类拼出一个迷你 Transformer Block 的 TP 版本**（不涉及真实 Block，只做拼装验证）：
  - Input → Norm → Attention(QKV 的 ColPar+RowPar) → Residual → Norm → FFN(Gate/Up 的 ColPar+Down 的 RowPar) → Residual
  - 验证：TP=2 和 TP=4 的 Block 输出与完整 Block 一致（随机权重初始化）

#### 第三步：分析与思考（约 1 小时）

- [ ] **分析推理场景 (bs=1, seq=1)**：
  - Attention: 1 × 1 × 4096 的 GEMM → 矩阵太小，GPU SM 利用率 < 5%
  - FFN: 1 × 1 × 4096 × 14336 → 约 117M FLOPs，H100 耗时 < 1 μs
  - 而 2 次 AllReduce 传输 2 × 1 × 1 × 4096 × 2 = 16KB → NVLink 耗时 < 0.02 μs
  - **矛盾不在通信——在计算**：每卡 GEMM 太小，GPU 跑不满。**推理场景不用 TP，用 ZeRO-3 或 PP**

- [ ] **面试视角**——自问自答：
  > Q: "TP 最大能到多少？"
  > A: 三个限制——(1) attention heads 数必须 ≥ TP（否则部分 GPU 没 head）；(2) 每卡 GEMM 矩阵太小 → GPU 利用率低；(3) 通信量 ∝ B×S×d，大 batch/长序列时通信耗时占比上升。TP=8 通常是 H100 单节点内的甜点。

- [ ] **思考题**：
  - 为什么 RMSNorm 不做 TP？（Norm 的计算只有逐元素乘加 → 通信/计算比极高。每卡输入已经是完整副本，各自 Norm 即可）

**完成标准**：
- 能脱稿画出完整 Block 的 TP 数据流图，标注每层的切分方式和 2 处 AllReduce
- Block TP 输出与完整版本对齐
- `Tensor_Parallel/` 目录完整（`tp_linear.py` + `tp_notes.md`）

---

### Day 67 — Pipeline Parallelism：GPipe 与泡率公式

**目标**：理解 PP 为什么是「跨节点扩展的核心手段」——TP 通信量随 batch_size 增长，跨节点 InfiniBand 扛不住。PP 只在层边界传 activation，通信量不随层内计算增长。手画 GPipe Gantt 图，推导泡率公式。

**预计时间**：3 小时（阅读 30min + 画图推导 1.5h + 笔记输出 1h）

---

**任务清单**：

#### 第一步：读 GPipe 论文（约 30 分钟）

- [ ] **精读** [GPipe Paper](https://arxiv.org/abs/1811.06965) §3.1（必读，20 分钟）
  - 关注 micro-batch 的概念：一个大 batch 切成 M 个小块 → 块与块之间可以流水线重叠
  - 理解 Pipeline 的「建立阶段」：前 M 个 micro-batch 流过 P 个 stage 需要 P+M-1 步
  - 理解 GPipe 的局限：必须等全部前向完成才能反向 → 像公路只有一个方向交替通车

- [ ] **回顾 Day 64 的动机**（5 分钟）：
  - TP 通信量 ∝ B×S×d，跨 InfiniBand 时通信时间接近计算时间
  - PP 传的也是一整层的 activation（B×S×d），但**只在 P 个层边界传 P-1 次**
  - 总通信量(P-1)×B×S×d 不随层内 GEMM 的计算量增长

#### 第二步：画 Gantt 图 + 推导泡率（约 1.5 小时）

- [ ] **用 matplotlib 画 GPipe Gantt 图**（P=4 stages, M=8 micro-batches）：
  - x 轴 = 时间步，y 轴 = GPU(stage)
  - 前向阶段：F1→F2→F3→F4（GPU1 做完 F1 传给 GPU2 做 F1，流水线建立中）
  - 反向阶段：B4→B3→B2→B1（全部前向完成后才开始反向）
  - 用不同颜色区分「计算中」「空闲等待」「通信中」
  - **直观看到**：大量灰色空闲——这就是「泡」

- [ ] **推导泡率公式**：
  - 前向建立：GPU1 需要先建 F1, GPU2 接收后建 F2……P 步后所有 GPU 都在干活
  - 总步数 = P + M - 1（P 步建立 + M-1 步流过）
  - 泡（= 某个 GPU 空闲的时间步）= P-1 步（每个 stage 在建立阶段需要等待前面 stage）
  - 泡率 = (P-1)/(P-1+M)
  - P=4, M=8 → 泡率 = 3/11 ≈ 27%
  - M=32 时泡率 = 3/35 ≈ 8.6%——但代价是存 M 个微批次的激活值

- [ ] **对比 TP**：
  - TP 通信量 = 2×AllReduce × N_layers × B×S×d（每层都要通信）
  - PP 通信量 = (P-1) × B×S×d（总共只传 P-1 次）
  - TP 通信 ∝ 层数，PP 通信 ∝ stage 数。层数 32 > stage 数 4 → PP 通信少得多

#### 第三步：输出笔记（约 1 小时）

- [ ] **写 `Pipeline_Parallel/pipeline_notes.md`**：
  - 第 1 节：PP 的动机——与 TP 通信量的对比
  - 第 2 节：GPipe Gantt 图（用 matplotlib 图 + 箭头标注）
  - 第 3 节：泡率公式推导——从直观到数学
  - 第 4 节：M 的 trade-off——降低泡率 vs 增加显存

**完成标准**：
- `pipeline_notes.md` 含 matplotlib 绘制的 Gantt 图 + 泡率推导过程
- 能解释「PP 为什么不像 TP 那样需要 NVLink」
- 能推导泡率公式并解释 M >> P 时的极限行为

---

### Day 68 — PP 进阶：1F1B · Interleaved · Zero-Bubble · DualPipe

**目标**：理解「填缝」这个核心思想如何贯穿 PP 的整个优化史——从 GPipe 到 DualPipe，每一步都在找一个可以让 GPU 不空闲的间隙。

**预计时间**：3 小时（阅读 1h + 画图 1.5h + 思考 30min）

**注意**：今天不写 Python 代码。PP 的调度逻辑用 Gantt 图理解比代码直观得多。

---

**任务清单**：

#### 第一步：逐级读论文（约 1 小时）

- [ ] **1F1B**：[PipeDream Paper](https://arxiv.org/abs/1806.03377) §3-4（选读，10 分钟）
  - 核心洞察：不需要等所有前向完成。前向跑够 P 步建立流水线后，每完成一个前向立刻插一个反向

- [ ] **Interleaved**：[Megatron-LM 2 Paper](https://arxiv.org/abs/2104.04473) §2.2（必读，15 分钟）
  - 每层切成 N 个 chunks → 总阶段数 = P × N（更多阶段 = 更细的粒度 = 更少的等待）
  - 泡从 15%(1F1B) → ~5%(Interleaved)

- [ ] **Zero-Bubble**：[Zero-Bubble Paper](https://arxiv.org/abs/2401.10241) §2-3（选读，10 分钟）
  - 把反向拆成 B(参数梯度，需要通信) 和 W(输入梯度，不需要通信)
  - 用 W 填满原本空闲的 slot

- [ ] **DualPipe**：[DeepSeek-V3 §3.1](https://arxiv.org/abs/2412.19437) + [DualPipe GitHub](https://github.com/deepseek-ai/DualPipe)（必读，20 分钟）
  - 双向注入：两条数据流从 pipeline 两端对向流动
  - 像双向车道——前向从 A→B，反向从 B→A，在中间相遇时交错
  - 结果：泡 39%→8%，MFU 51%，在 2048 H800 集群上验证

#### 第二步：画三张 Gantt 图（约 1.5 小时）

- [ ] **图 1：1F1B Gantt 图**（P=4, M=8）
  - 前向跑 P 步建立 → 交替 F/B/F/B……
  - 对比 Day 67 的 GPipe 图——灰色区域明显减少

- [ ] **图 2：Interleaved 1F1B Gantt 图**（P=4, 每 stage 切成 2 chunks, M=8）
  - 总阶段数 = 8。粒度更细 → 泡更少
  - 标注和 1F1B 相比泡减少的位置

- [ ] **图 3：DualPipe 概念示意图**（不用 Gantt，画概念图即可）
  - 两条数据流对向流动（各有一个方向的前向 + 另一个方向的反向）
  - 标注计算/通信重叠区域
  - 用一句话解释核心思想

- [ ] **在所有图的下方标注泡率**：GPipe 47% → 1F1B 15% → Interleaved 5% → ZB <1% → DualPipe 8%（注：DualPipe 的 8% 不含纯泡——它把泡用通信重叠填了）

#### 第三步：思考与收束（约 30 分钟）

- [ ] **面试视角**——自问自答：
  > Q: "从 GPipe 到 DualPipe，优化的主线是什么？"
  > A: 填缝——让 GPU 永远不闲着。1F1B 把反向交错插入前向的间隙；Interleaved 把粒度做细减少等待时间；Zero-Bubble 把反向拆成 B/W 两部分精准填入空隙；DualPipe 双向注入，利用相反方向的数据流填充对方的空闲时间，实现计算和通信的完全重叠。

- [ ] **思考题**：
  - DualPipe 的「双向注入」在单数据流场景（如单用户在线推理）下会退化吗？
  - Interleaved 的 chunks 越多越好吗？什么限制了 chunk 数？（显存：每个 chunk 都要存激活值。chunk 越多 = 需要 hold 的激活值越多。Activation checkpointing 可以缓解，但不解决根本问题）

**完成标准**：
- 能画出 GPipe / 1F1B / Interleaved 三种 Gantt 图并准确标注泡率差异
- 能用一句话说清 DualPipe 的核心创新
- `Pipeline_Parallel/` 目录完整

---

### Day 69 — ZeRO：三级显存递进手推

**目标**：用 Python dict 模拟 ZeRO-1/2/3 的显存变化。三级递进本质是一个精妙设计：每多切一件事，显存降一档，通信多一档。理解这个渐进的 trade-off 序列。

**预计时间**：3 小时（阅读 30min + 编码 1.5h + 推导+笔记 1h）

---

**任务清单**：

#### 第一步：读 ZeRO 论文 §4-6（约 30 分钟）

- [ ] **精读** [ZeRO Paper](https://arxiv.org/abs/1910.02054) §4-6（Stage 1-3）（必读，25 分钟）
  - **图 4**：ZeRO 三级显存对比——这张图和 Day 62 的图 1 是同一组作者，风格一致
  - §4（P_os）：只切 optimizer states → 每卡 Adam = 12N/Np bytes → 显存从 16N 降至 4N+12N/Np
  - §5（P_os+g）：加切 gradients → ReduceScatter 替换 AllReduce → 通信量不变，显存降到 2N+14N/Np
  - §6（P_os+g+p）：连 parameters 也切 → 前向 AllGather 拼参数 → 用完即弃 → 反向再 AllGather → 通信 ×1.5，显存降到 16N/Np
  - 跳过 §7（ZeRO-R, Residual State 优化）——选读标记

#### 第二步：用 dict 模拟显存（约 1.5 小时）

- [ ] **创建 `ZeRO/zero_sim.py`**（不涉及真实分布式，纯 Python dict 模拟）

  - 定义 `ZeROStage1Sim(Np, model_params)`：
    ```python
    # 初始化：per_gpu_os = total_os / Np（Adam m,v,master 各切 1/Np）
    # 训练一步的通信：AllReduce（全量 gradient + 全量参数）
    # peak_memory = 2N(param+grad) + 12N/Np(os)
    ```

  - 定义 `ZeROStage2Sim`：
    ```python
    # 额外切 gradient：反向时 ReduceScatter 替换 AllReduce
    # 每卡只保留自己的 1/Np gradient
    # peak_memory = 2N(params) + 2N/Np(grad) + 12N/Np(os)
    ```

  - 定义 `ZeROStage3Sim`：
    ```python
    # 连参数也切：前向 AllGather(参数) → gc → 反向 AllGather(参数) → gc
    # peak_memory = 2N/Np(param) + 2N/Np(grad) + 12N/Np(os) = 16N/Np ← 净省 Np 倍！
    # 但通信量 = AllGather(fwd) + AllGather(bwd) + ReduceScatter(grad) ≈ 1.5× DP
    ```

  - 模拟 Np=4 时每步的内存变化：
    - 打印每张 GPU 的 peak memory（对比 DP 基准）
    - 打印每张 GPU 的通信量（对比 DP 基准）

- [ ] **在 `__main__` 中测试**：
  - 输入 7B 模型、Np=4：ZeRO-1/2/3 三种策略下每卡显存和通信量
  - 输入 70B 模型、Np=8：同上
  - 验证显存 sum 守恒：所有 GPU 的总 model states = 16N bytes（和 DP 一样——只是分布式了）

#### 第三步：推导 + 笔记（约 1 小时）

- [ ] **推导三级通信量递进**：
  - ZeRO-1：1× 通信量（AllReduce 不变，只是 optimizer state 更新时各做各的。等一等——optimizer step 不需要通信！Adam 更新是本地操作，因为每个参数只属于一张卡。通信只在 AllReduce gradient 环节，1×）
  - ZeRO-2：1× 通信量（ReduceScatter 和 AllReduce 通信量相同。为什么？ReduceScatter 的 data 流动量和 AllReduce 一样——都是 scatter-reduce 阶段做完，只是 ReduceScatter 不做最后的 allgather）
  - ZeRO-3：约 1.5×（+2 次 AllGather 参数各 1/Np × 全量。每次 AllGather = (Np-1)/Np × D 通信量。两次 = 2(Np-1)/Np × 2N ≈ 4N。ReduceScatter grad = (Np-1)/Np × 2N ≈ 2N。合计 ≈ 6N vs DP AllReduce = 2N(参数) + 2N(grad) = 4N。所以 = 6N/4N = 1.5×）

- [ ] **记决策直觉**：
  - 7B：模型小，ZeRO-1 够用（Adam 12N cut 到 12N/Np, 显存从 16N 到 4N+12N/Np ≈ 53GB for Np=4）
  - 13-70B：ZeRO-3 / FSDP2
  - 175B+：ZeRO-3 + TP + PP 组合
  - ZeRO 不切激活值 → AC 是 ZeRO 的天然互补

- [ ] **写 `ZeRO/zero_notes.md`**：
  - 第 1 节：三级递进表（每级的显存公式 + 通信倍数 + 适用模型规模）
  - 第 2 节：dict 模拟的输出截图/表格
  - 第 3 节：决策树

- [ ] **选做**：用 matplotlib 画显存-通信 pareto 曲线（DP/Z-1/Z-2/Z-3 四个点，x=显存，y=通信）

**完成标准**：
- `zero_sim.py` 能打印 ZeRO-1/2/3 下每卡 peak memory 和通信量
- 能口头推出「ZeRO-3 省 Np× 显存、多 0.5× 通信」的结论
- 运行命令：`cd ZeRO && python zero_sim.py`
- `ZeRO/` 目录完整

---

### Day 70 — ZeRO 实战：FSDP2 + 组合策略

**目标**：理解 FSDP2 如何等价于 ZeRO-3。理解 ZeRO 如何与 TP/PP 共存。能回答「这个规模的模型，在这个集群上，该用什么组合」的判断题。

**预计时间**：2.5 小时（阅读 45min + 编码完善 1h + 思考 45min）

---

**任务清单**：

#### 第一步：读 FSDP2 + 对比资料（约 45 分钟）

- [ ] **阅读** [FSDP2 Technical Blog (PyTorch)](https://pytorch.org/blog/fsdp2/)（必读，20 分钟）
  - `reshard_after_forward=True`：前向 AllGather 参数 → 算完立即释放 → 反向时重新 AllGather。这等价于 ZeRO-3
  - DTensor per-parameter sharding：不需要 FlatParameter（FSDP1 的局限）
  - `fully_shard` API：对每个 submodule 调用，粒度比 FSDP1 的「整模型一个策略」灵活得多

- [ ] **阅读** [FSDP vs DeepSpeed (Spheron, 2026)](https://www.spheron.network/blog/distributed-llm-training-fsdp-deepspeed-megatron-multi-node/)（必读，15 分钟）
  - 决策矩阵：DeepSpeed 优势 = CPU/NVMe offload（ZeRO-Infinity）。FSDP2 优势 = PyTorch 原生 + DTensor + 与 torch.compile 的 better compatibility
  - 一般规则：单节点 8 卡以内用 FSDP2；需要 offload、混合精度、NVMe → DeepSpeed

- [ ] **选读** [PyTorch FSDP2 for 100B+ Models (TechBytes, 2026)](https://techbytes.app/posts/fsdp2-100b-models-multi-node-training-deep-dive-2026/)（10 分钟）
  - 看一个真实的大模型训练配置

#### 第二步：完善 zero_sim + 理解组合（约 1 小时）

- [ ] **完善 `zero_sim.py`**：
  - 加命令行接口：`python zero_sim.py --model 7B --gpus 8` → 输出推荐策略和预期显存
  - 加入简单可视化：显存-通信 trade-off 曲线（DP/Z-1/Z-2/Z-3 四个点）

- [ ] **理解 ZeRO-3 + TP 共存**：
  - DTensor 的 `DeviceMesh`：把 GPUs 分组成 tp_groups（每 group 做 TP）→ tp_groups 之间做 FSDP/ZeRO
  - 例如：8 GPUs, TP=2 → 4 个 tp_groups。tp_group_0 做 Attention QKV 的 ColParallel → RowParallel。所有 4 个 tp_groups 之间做 FSDP 切参数
  - 通信量 = TP 的 AllReduce（tp_group 内，NVLink）+ FSDP 的 AllGather/ReduceScatter（tp_group 间，可能跨 InfiniBand）

- [ ] **Activation Checkpointing vs ZeRO 对比总结**：
  - AC：省激活值（存每 N 层 checkpoint → 反向时重算。ZeRO 不管激活值）
  - ZeRO：省模型状态（参数 + 梯度 + Adam。AC 不管模型状态）
  - 两者互补：ZeRO-3 + AC 是目前 100B+ 模型训练的标配

#### 第三步：思考与收束（约 45 分钟）

- [ ] **面试视角**——自问自答：
  > Q: "Activation Checkpointing 和 ZeRO 分别省什么显存？为什么能一起用？"
  > A: AC 省激活值（ZeRO 不切激活值），ZeRO 省模型状态（AC 不管参数/梯度/Adam）。互不重叠，组合使用：ZeRO-3 把模型状态降到 16N/Np，AC 把激活值降到 O(√L) checkpoint。两者一起 → 显存从 120GB+ → 可能在 80GB 以内。

- [ ] **思考题**：
  - 如果有 NVMe SSD（8 GB/s 读），ZeRO-Infinity 能 offload 什么？（Adam states、梯度、甚至参数。比 CPU offload 快 10× 但比 HBM 慢 400×。适用场景：极少数超大 batch 的训练）

**完成标准**：
- `zero_sim.py` 有 CLI + 可视化
- 能解释 FSDP2 = ZeRO-3 的等价关系
- 能解释 ZeRO-3 + TP 的 DeviceMesh 共存机制
- `ZeRO/` 目录完整
- 运行命令：`cd ZeRO && python zero_sim.py --model 7B --gpus 8`

---

## Part 3：序列维度与专家路由（Day 71-74）

> 前两种并行切的是参数和层。还有两个维度可以切：序列（Ring Attention）和专家（Expert Parallelism）。前者是长文本的解决方案，后者是 MoE 模型的规模化手段。

---

### Day 71 — Ring Attention：核心算法手撕

**目标**：手撕 Ring Attention 的核心循环。这是 Phase 5 最硬核的模块——但本质是 Phase 2 Online Softmax 在「多卡间」的推广。你把单卡 SRAM 里的分块 softmax 扩展到「多卡 HBM 之间环形通信」，就是 Ring Attention。

**预计时间**：3.5 小时（回顾+阅读 45min + 编码 2h + 验证+笔记 45min）

---

**任务清单**：

#### 第一步：回顾 Phase 2 + 读 Ring Attention 论文（约 45 分钟）

- [ ] **回顾 Phase 2** `FlashAttention/online_softmax.py`（10 分钟）
  - 重读 (m, l, O) 三元组的更新公式：
    - m = max(m_old, m_local)
    - l = exp(m_old - m) × l_old + exp(m_local - m) × l_local
    - O = (l_old/l) × exp(m_old - m) × O_old + (l_local/l) × O_local
  - 如果你不记得这个公式了，先花 15 分钟重新理解——它是 Ring Attention 的数学基础

- [ ] **精读** [Ring Attention Paper](https://arxiv.org/abs/2310.01889) §3（必读，25 分钟）
  - 核心：4 张 GPU，各持有 (Q_i, K_i, V_i)。Round 1→4 环形传递 K,V，每轮用新到的 KV 更新 online softmax
  - 注意：Q 驻留本地不动，只有 K,V 在环上移动
  - 通信量：每轮传 seq_len/Np × head_dim × 2（K+V）× Np 轮 = 2 × seq_len × head_dim × Np

- [ ] **选读** [Ring Attention 图解 (Spheron, 2026)](https://www.spheron.network/blog/ring-attention-tree-attention-sequence-parallelism-gpu-cloud/)（10 分钟）
  - 可视化和动画很有帮助

#### 第二步：手撕核心循环（约 2 小时）

- [ ] **创建 `Ring_Attention/ring_attention.py`**

  - 实现 `block_attention_with_online_softmax(Q, K, V, m_old, l_old, O_old)`：
    ```python
    # 输入：Q [B, n_heads, S_i, d_head] —— 本地 Q 块
    #      K [B, n_heads, S_j, d_head] —— 新到达的 K 块
    #      V [B, n_heads, S_j, d_head] —— 新到达的 V 块
    #      m_old, l_old, O_old —— 当前全局 online softmax 状态
    # 输出：m_new, l_new, O_new —— 更新后的全局状态
    ```

  - 实现 `ring_attention_step(rank, world_size, Q_chunks, K_chunks, V_chunks)`：
    ```python
    # 循环 world_size 轮：
    # for round in range(world_size):
    #    src_kv = (rank - round) % world_size  # 本轮的 KV 来自这张卡
    #    用 K_chunks[src_kv], V_chunks[src_kv] 更新 online softmax
    #    send(K_chunks[src_kv], V_chunks[src_kv] → next_rank)
    #    recv(下一张卡的 KV ← prev_rank)
    ```

  - 不涉及真实的 `torch.distributed.send/recv`（太复杂 + 单卡跑不了）。用函数调用来模拟环形传递——`round=0` 时用的 KV 是 `chunks[(rank-0)%Np]`，`round=1` 时是 `chunks[(rank-1)%Np]`，依此类推

- [ ] **`__main__` 中验证**：
  - 4 卡模拟：随机 seq_len=1024，切成 4 块每卡 256 tokens
  - Ring Attention 输出 vs `F.scaled_dot_product_attention`(完整序列) → `torch.allclose(atol=1e-4)`
  - 打印每轮每卡的 (m, l, O) 最大/最小值变化过程，可视化「逐块修正」

#### 第三步：输出笔记（约 45 分钟）

- [ ] **写 `Ring_Attention/ring_attention_notes.md`**：
  - 第 1 节：Ring Attention 的核心思想——Q 驻留，KV 环形传递
  - 第 2 节：Online Softmax 修正公式（从 Phase 2 的单卡→多卡推广）
  - 第 3 节：通信量 = 2 × seq_len × head_dim × Np（为什么是 2×？K+V 各传一份）
  - 第 4 节：与 TP 通信量的对比（RA：O(S)，TP：O(B×S)。长序列下 RA 计算/通信比更好）

**完成标准**：
- Ring Attention 输出与完整 attention `torch.allclose(atol=1e-4)`
- 能画出 4 轮环形传递中每张卡的 KV 持有情况
- 运行命令：`cd Ring_Attention && python ring_attention.py`

---

### Day 72 — Ring Attention：Causal 变体 + Ulysses 对比

**目标**：理解 Causal Ring Attention 的负载不均衡（后面 GPU 需要 attend 更多 KV）以及 DeepSpeed Ulysses 的 All-to-All 替代方案。建立「给定场景选 Ring 还是 Ulysses」的判断力。

**预计时间**：3 小时（阅读 45min + 编码 1.5h + 思考+笔记 45min）

---

**任务清单**：

#### 第一步：读 Ulysses + 理解对比（约 45 分钟）

- [ ] **阅读** [DeepSpeed Ulysses Paper](https://arxiv.org/abs/2309.14509) §2-3（必读，20 分钟）
  - 核心：2 次 All-to-All。第一次把序列并行 ([B,N_heads,S/Np,d_head]) 变成 head 并行 ([B,N_heads/Np,S,d_head])。各算各的 attention（完整 S），再 All-to-All 变回序列并行
  - 限制：N_heads 必须 ≥ Np，否则无法均分。且 GQA 下 KV heads < Q heads → 更严格

- [ ] **阅读** [Striped Attention](https://arxiv.org/abs/2311.09431)（选读，10 分钟）
  - Ring Attention 的 causal 场景修复：zigzag token 排列让每张 GPU 的计算量接近相等

- [ ] **画对比图**（纸上画即可，5 分钟）：
  - Ring：沿 S 切 → 每卡有 (Q_i, K_i, V_i) → KV 环形传 → 各轮更新 online softmax
  - Ulysses：沿 S 切 → All-to-All → 每卡有完整 S 但 N_heads/Np heads → local attention → All-to-All 恢复

#### 第二步：编码 causal ring attention（约 1.5 小时）

- [ ] **在 `ring_attention.py` 中添加 causal mask 版本**：
  - 每轮 attention 计算时加 causal mask：Q_i 不需要 attend 还未到来的 KV
  - 具体做法：round=0 用 K[(rank-0)%Np] 时，如果 (rank-0)%Np > rank（即 KV 来自后面 GPU），causal mask 会完全 mask 掉
  - 测量 4 卡每张 GPU 的 FLOPs 差异（写一个简单的 FLOPs counter）：
    - GPU 0（序列前 1/4）：只 attend 自己的 K_0 → 1 轮有效计算
    - GPU 1：attend K_0 + K_1 → 2 轮
    - GPU 3：attend 全部 K → 4 轮
    - **负载比 1:2:3:4**——最后一张 GPU 的计算量是第一张的 4 倍！

- [ ] **在 `__main__` 中对比**：
  - 非 causal（bidirectional attention）vs causal ring attention 的输出差异
  - 验证 causal ring attention 的每卡 FLOPs 确实随 rank 递增

#### 第三步：建立判断力（约 45 分钟）

- [ ] **总结 Ring vs Ulysses 选择矩阵**：
  | 场景 | 推荐 | 原因 |
  |------|------|------|
  | N_heads ≥ Np 且非 GQA | Ulysses | 通信量小（2×All-to-All ≤ Ring 的 Np 轮 KV 传输） |
  | GQA | Ring Attention | KV heads 少时 Ulysses 退化 |
  | 长序列（128K+）| Zigzag Ring | Causal 负载更均匀 |
  | 异构集群 | Ring | 更 flexible（不要求 head 数约束） |

- [ ] **面试视角**——自问自答：
  > Q: "GQA 为什么对 Ulysses 是挑战？"
  > A: Ulysses 的第一次 All-to-All 把序列切分→head 切分，要求 all_heads = N_q_heads + N_kv_heads 能被 Np 整除。GQA 下 N_kv_heads << N_q_heads（如 40+8=48, Np=16 → 48/16=3 整除，但 KV head 只有 8 个 → Np=16 时每卡分到的 KV head 数 = 8/16 = 0.5 → 不完整）。且某些卡分不到 KV heads → 无法独立做 attention。Ring Attention 没这个限制——它沿序列切，不沿 head 切。

- [ ] **写 `Ring_Attention/ring_attention_notes.md` 补充**：
  - 第 5 节：Causal Ring Attn 的负载不均衡分析
  - 第 6 节：Ulysses 的原理 + GQA 退化 + 选择矩阵

**完成标准**：
- Causal Ring Attention 完成，能展示每卡 FLOPs 差异
- 能画出 Ring vs Ulysses 的数据流对比
- `Ring_Attention/` 目录完整

---

### Day 73 — Expert Parallelism：MoE 的分布式路由

**目标**：手撕 EP 的四步循环。Phase 1 学会了单卡 MoE（8 experts 挤一块 GPU），EP 让你理解规模化 MoE 的真正形态——每个 GPU 只持有一部分 experts，tokens 通过 All-to-All 路由到正确 GPU。

**预计时间**：3 小时（回顾+阅读 45min + 编码 1.5h + 验证+笔记 45min）

---

**任务清单**：

#### 第一步：回顾 Phase 1 + 读 DeepSeek-V2（约 45 分钟）

- [ ] **回顾 Phase 1** `MoE/moe_layer.py`（10 分钟）
  - 单卡 Top-K routing：`router_logits = self.gate(x)` → `topk_indices, topk_weights`
  - Load balancing loss：`L_aux = N_experts × sum(f_i × P_i)`
  - 这些在 EP 场景中仍然存在——只是 experts 分布在多卡上

- [ ] **阅读** [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434) §3.1-3.2「MoE Architecture」（必读，20 分钟）
  - 关注 Token-to-Expert 的分配机制
  - DeepSeekMoE 的 Shared Expert + Routed Expert 设计

- [ ] **阅读** [Megatron Parallel Folding](https://arxiv.org/abs/2603.07685)（选读，10 分钟）
  - 核心：Attention 和 MoE 用不同并行拓扑——Attention 用 TP，MoE 用 EP。两者独立，不互相锁定

#### 第二步：手撕 EP 四步循环（约 1.5 小时）

- [ ] **创建 `Expert_Parallel/expert_parallel.py`**

  - 实现 4 个核心函数：
    ```python
    def expert_dispatch(tokens, expert_ids, world_size):
        """Step 1+2: 按 expert_ids 分组 tokens → 模拟 All-to-All 发送到对应 GPU"""

    def expert_compute(tokens, expert_weights, expert_biases):
        """Step 3: 每 GPU 对分到的 tokens 执行本地 experts 的前向（nn.Linear→SwiGLU→nn.Linear）"""

    def expert_combine(outputs, original_indices, world_size):
        """Step 4: 模拟 All-to-All 按原 token 序号重组输出"""

    def ep_moe_forward(x, gate, experts_per_gpu, world_size):
        """完整 EP 前向：Routing → Dispatch → Compute → Combine"""
    ```

  - 模拟 2 GPU × 4 experts（每 GPU 2 experts）：
    - 随机输入 tokens（B=4, S=8, d_model=32），随机初始化 gate + expert 权重
    - Top-2 routing → dispatch → compute → combine
    - Residual connection：x_final = x + combine_output

- [ ] **`__main__` 中验证**：
  - EP 版本（2 GPU × 2 experts each）vs 单卡完整 MoE（4 experts 全在同一 GPU）
  - `torch.allclose(atol=1e-5)`（因为只有 All-to-All 的数据重组，计算结果 should be bit-exact）

#### 第三步：输出笔记（约 45 分钟）

- [ ] **写 `Expert_Parallel/expert_parallel_notes.md`**：
  - 第 1 节：EP 四步循环的数据流图（每步标注输入/输出形状）
  - 第 2 节：为什么 MoE 层用 EP 而 Attention 层用 TP？（EP 稀疏——每 token 只去 1-2 experts；TP 密集——所有 GPU 都参与。MoE 用 TP 会导致非 expert 的 GPU 做无用计算）
  - 第 3 节：All-to-All 通信量 vs TP 的 AllReduce 通信量对比

**完成标准**：
- EP 输出与单卡 MoE 严格对齐
- 能画出 EP 四步循环的数据流
- 运行命令：`cd Expert_Parallel && python expert_parallel.py`

---

### Day 74 — EP 进阶：Capacity Factor + DeepSeek 工程实践

**目标**：理解 hot expert 溢出的处理策略（Capacity Factor）。速览 DeepSeek-V3 在 2048 GPU 上的 EP 实战配置，理解每个并行维度的数字是怎么选出来的。

**预计时间**：3 小时（编码 1h + 阅读 1.5h + 思考 30min）

---

**任务清单**：

#### 第一步：实现 Capacity Factor + Load Balancing（约 1 小时）

- [ ] **在 `expert_parallel.py` 中添加 capacity_factor 处理**：
  - capacity = tokens_total / N_experts × capacity_factor
  - 超出的 token 被 drop → residual 直通（不经过 expert）
  - 创建 hot expert 场景：人工让 80% tokens 路由到 expert 0 → overflow → drop 到 residual

- [ ] **实现 bias-based load balancing**（模仿 DeepSeek 方案）：
  - 每个 expert 维护一个 bias（初始 0），路由时 gate_logits = original_logits + bias
  - 每 N 步根据历史负载统计更新 bias：负载过重的 expert 降低 bias
  - **关键优势**：不用 auxiliary loss → 不干扰主训练 loss + 没有 scale 超参

#### 第二步：读 DeepSeek-V3 工程配置（约 1.5 小时）

- [ ] **精读** [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437) §2.1-2.3（必读，30 分钟）
  - §2.1 架构：256 experts per layer, Top-8 routing, shared experts
  - §2.2 Multi-Token Prediction：训练时预测未来 n 个 token → 训练信号更密集 → 训练质量 ↑
  - §2.3 FP8 blockwise training：每 128×128 子矩阵独立 scale → outlier 只污染局部，不波及整个张量

- [ ] **配置解读**（15 分钟）：
  - 2048 H800 GPUs：TP=4, EP=64, PP=16, DP=2
  - EP=64 为什么是最大头？256 experts / 64 = 4 experts per GPU。experts 数量远超 GPU 数，切分收益最大
  - TP=4 只在节点内（8 卡节点，NVLink）
  - PP=16 跨节点（2048/8=256 节点，分成 16 个 PP stage → 每 stage 16 节点）
  - DP=2 很少——因为 MoE 每个 token 只激活少量参数，数据并行度不需要很高

- [ ] **速览 DeepSeek 五大连创新**（20 分钟）：
  | 创新 | 解决的问题 | Phase 5 的知识根基 |
  |------|-----------|-----------------|
  | DualPipe | PP 的泡（39%→8%） | Day 67-68 的泡率公式 |
  | FP8 Blockwise | 训练精度/速度（50% 加速） | 按子矩阵而非整张量量化 |
  | Aux-loss-free Balancing | MoE 负载不均（loss scale 问题） | bias-based routing |
  | Multi-Token Prediction | 训练密度 + inference 加速（1.8×） | 预测未来 n token |
  | DeepEP | EP 通信（GPU-initiated, 41% 延迟 ↓） | All-to-All + bypass CPU |

- [ ] **选读** [DeepSeek-V4 Analysis (LMSYS, 2026)](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)（15 分钟）
  - V4 新增：Anticipatory Routing（用旧权重监督路由决策防止 loss spike）+ On-Policy Distillation（全词表 logit 匹配）

- [ ] **选读** [DeepEP](https://github.com/deepseek-ai/DeepEP) README（5 分钟）
  - GPU kernel 直接 push/pull MoE 数据，bypass CPU → 41% 延迟降低

#### 第三步：思考与收束（约 30 分钟）

- [ ] **面试视角**——自问自答：
  > Q: "为什么 DeepSeek-V3 训练只要 ~$5.57M？"
  > A: 四个因素——(1) MoE 每个 token 只激活 ~37B/671B 参数；(2) DualPipe MFU 51%（vs 同期 Meta ~35%）；(3) FP8 blockwise training 精度够且速度翻倍；(4) Aux-loss-free balancing 减少了超参调优和收敛问题。

- [ ] **思考题**：
  - 为什么 MoE 模型中 EP 通常 >> TP？（因为 expert 数量是独立的并行维度——不受 attention heads 或 hidden_dim 限制。TP 最大 8，EP 可以到 64 甚至 128）
  - Auxiliary load balancing loss 的问题是什么？（scale 是超参——太小没效果，太大干扰主任务收敛。DeepSeek 用 bias 替代 loss 是更干净的方案）

**完成标准**：
- Capacity factor overflow 处理完成，hot expert 模拟跑通
- Bias-based load balancing 原型完成
- 能解读 DeepSeek-V3 2048 GPU 配置中每个数字的物理含义
- `Expert_Parallel/` 目录完整

---

## Part 4：整合 — 混合策略与知识收束（Day 75-76）

> 把学过的四种并行策略拼起来。给定一个模型规模和集群配置，你能像机械师一样给出推荐方案。

---

### Day 75 — 混合并行：策略选择与案例推演

**目标**：建立「模型 & 集群 → 最佳并行策略」的决策能力。三个真实案例完整推导。

**预计时间**：3 小时（整理约束 30min + 案例推演 1.5h + 笔记输出 1h）

---

**任务清单**：

#### 第一步：整理决策约束（约 30 分钟）

- [ ] **列出策略选择的硬约束**（写到笔记里）：
  1. 单卡显存：模型参数 + Adam + 激活值 < HBM（A100=80G, H100=80G, H800=80G）
  2. TP ≤ N_attention_heads（且 N_heads % TP = 0）
  3. N_experts % EP = 0（All-to-All 需要均匀分布）
  4. TP 在 NVLink 域内（H100 单节点 8 卡 → TP ≤ 8）
  5. DP = N_total / (TP × PP × EP)
  6. PP stage 数 × M(micro-batches) 的激活值不能炸显存

- [ ] **列出软约束**（优先级从高到低）：
  1. 优先用 ZeRO/FSDP（通信开销最小、代码侵入性最低）
  2. TP 仅在 ZeRO 不够时叠加（且只在节点内）
  3. PP 仅在需跨节点时使用（泡是代价）
  4. EP 仅在 MoE 架构时使用（dense 模型不需要）

#### 第二步：三个案例完整推导（约 1.5 小时）

- [ ] **案例 1：7B Dense @ 8×A100-80G**
  - 单卡参数 14GB → fit 80GB
  - 但 Adam 84GB + 激活值 ~2GB → 单卡训练不可行
  - 不需要 TP（模型足够小）
  - ZeRO-3 → 模型状态 16N/Np = 16×7B/8 = 14GB 每卡 + 激活值 ~2GB = 16GB
  - **推荐：FSDP/ZeRO-3, DP=8**

- [ ] **案例 2：70B Dense @ 32×H100-80G**
  - 参数 140GB bf16 → 单卡完全不可能
  - TP=4 → 每卡 35GB（参数）+ 激活值 ~5GB → fit 80GB
  - 32 GPUs / TP=4 = 8 DP groups → 每个 DP group 含 4 卡做 TP
  - ZeRO-3 across DP groups：Adam 状态 12N/N_dp
  - **推荐：TP=4, ZeRO-3(DP=8), PP=1（不需要跨节点）**
  - 验证：8 节点 × 4 卡 = 32 卡 ✓。TP=4 在 NVSwitch 域内 ✓

- [ ] **案例 3：671B MoE (DeepSeek-V3) @ 2048×H800-80G**
  - 256 experts per layer——这是显存主力
  - EP=64（256/64=4 experts per GPU。EP 是最大头——专家切分收益最高）
  - Attention 层 TP=4（节点内 NVLink）
  - PP=16（跨节点——2048/8=256 节点，16 个 PP stage → 每 stage 16 节点
  - DP=(2048) / (4 × 64 × 16) = 2048/4096 = 0.5 → 不对。重新算：2048/(4×64×16) = 2048/4096=0.5。DeepSeek 的 DP=2 → 说明他们的 formula 不是全乘，而是 TP=4 用于 Attention, EP=64 用于 MoE，两者不互斥（Parallel Folding）

#### 第三步：输出笔记（约 1 小时）

- [ ] **创建 `Parallelism_Recipe/parallelism_recipe_notes.md`**：
  - 第 1 节：决策树（flowchart）——输入 模型类型+规模+GPU 数 → 逐步排除 → 最终策略
  - 第 2 节：三个案例的逐步推导（含每一步的计算数字和约束检查）
  - 第 3 节：并行策略速查表（矩阵形式——行=模型规模，列=GPU 数 → 推荐策略组合）

- [ ] **面试视角**——准备一个 5 分钟推演：
  > 题目："64 张 A100 训 175B 模型，设计并行策略"
  > Step 1 显存：175B×2=350GB → 不可能单卡 → TP 必须。
  > Step 2 TP：TP=8（节点内 NVLink）→ 每卡 43.75GB（fit）+ 激活值 ~5GB = 48.75GB
  > Step 3 剩余：64/8=8 → 还需跨节点或数据并行
  > Step 4 PP：PP=4, DP=2 → 总计 8×4×2=64 GPU。跨节点 PP 通信量 = B×S×d — 小数据量
  > Step 5 ZeRO：Adam 84GB/64×ZeRO-3 → 每卡 ~1.3GB。但 TP 已经切了参数，ZeRO 只用于 DP dim
  > 最终：TP=8 + PP=4 + DP=2 + ZeRO-2 = 64 GPUs
  > 关键：过程中的系统性推演比精确数字重要 100 倍

**完成标准**：
- 三个案例推导过程完整，每个数字有来源
- 决策树和速查表可用
- `Parallelism_Recipe/` 目录完整

---

### Day 76 — 收束：知识地图 + 工业前沿巡览

**目标**：把 16 天的学习串联成一张知识地图。速览 DeepSeek 家族创新。回头看 Phase 2——分布式视角能给单卡推理什么新洞察。

**预计时间**：2.5 小时（画地图 1h + 写作 1h + 自查 30min）

---

**任务清单**：

#### 第一步：画知识地图（约 1 小时）

- [ ] **画一张 A4 大小的知识地图**（Mind Map 或 Mermaid 图）：
  ```
  分布式训练（核心问题：单卡装不下）
  ├── 地基
  │   ├── FLOPs 账单（6ND 公式 + Attention vs FFN 占比）
  │   ├── 显存账单（参数/梯度/Adam/激活值/临时缓冲）
  │   └── 通信语言（AllReduce/AllGather/ReduceScatter/Broadcast/Scatter/All-to-All）
  ├── 模型切分
  │   ├── TP —— 切单层矩阵（NVLink 域内，通信 ∝ B×S×d）
  │   │   └── ColumnParallel → RowParallel（省一次 AllReduce）
  │   ├── PP —— 切层序列（跨节点，泡率 = (P-1)/(P-1+M)）
  │   │   └── GPipe → 1F1B → Interleaved → ZB → DualPipe（填缝史）
  │   └── ZeRO —— 切优化器/梯度/参数（3 级递进，通信 ↑ 显存 ↓）
  │       └── FSDP2 = PyTorch 原生 ZeRO-3
  ├── 序列与专家
  │   ├── Ring Attention —— 序列维度并行（Online Softmax + 环形 KV）
  │   │   └── vs Ulysses（All-to-All vs Ring，GQA 场景选择）
  │   └── Expert Parallelism —— MoE 的规模化（All-to-All dispatch）
  │       └── Capacity Factor + Aux-loss-free Balancing
  └── 整合
      └── 混合策略决策树（7B/70B/671B 三档案例 + 速查表）
  ```

- [ ] **DeepSeek 创新归纳表**（五大创新 + 对应的 Phase 5 知识根基）：

#### 第二步：写 Phase5_System_Scale/README.md（约 1 小时）

- [ ] 如果还没写，现在写（Day 75-76 的整合任务）。内容涵盖：
  - 定位 → 目录树 → 学习路径图 → 模块速览 → 面试视角速查 → 跨 Phase 联动 → 运行指令

- [ ] 回头看 Phase 2——写下你分布式视角的新洞察：
  - PagedAttention 的 block table 在 EP/TP 场景下如何共享？每个 expert GPU 需要自己的 KV blocks 吗？
  - MLA（Multi-head Latent Attention）的 latent KV 在 TP 切分时——是按 latent dim 切还是按 head 切？
  - 这些是你学完 Phase 5 才能提出的问题——之前根本看不到

#### 第三步：自查清单（约 30 分钟）

- [ ] **每个子目录的交付物检查**:
  - [ ] System_Math/：`system_math.ipynb` 是否有 FLOPs + Memory 推导？
  - [ ] Collectives/：`ring_allreduce.py` 是否跑通？`collectives_notes.md` 是否含 6 种通信图？
  - [ ] Tensor_Parallel/：`tp_linear.py` 是否与 nn.Linear 对齐？`tp_notes.md` 是否完整？
  - [ ] Pipeline_Parallel/：`pipeline_notes.md` 是否含 Gantt 图 + 泡率推导？
  - [ ] ZeRO/：`zero_sim.py` 是否输出 ZeRO-1/2/3 的显存对比？`zero_notes.md` 是否含决策树？
  - [ ] Ring_Attention/：`ring_attention.py` 是否与 F.sdpa 对齐？`ring_attention_notes.md` 是否含 Ring/Ulysses 对比？
  - [ ] Expert_Parallel/：`expert_parallel.py` 是否与单卡 MoE 对齐？`expert_parallel_notes.md` 是否完整？
  - [ ] Parallelism_Recipe/：`parallelism_recipe_notes.md` 是否含三个案例 + 决策树？

- [ ] **项目根 README.md 的 Phase 5 表格是否已更新？**（Day 1 已经更新了，确认一下）

- [ ] **面试自我模拟**（5 分钟）：
  > "用 2 分钟讲清楚分布式训练？"
  > 核心问题（单卡装不下）→ 四大并行按切分对象（数据/矩阵/层/expert）→ 每种的计算/通信/显存三角 → 2026 趋势（MoE+EP 主流、DualPipe 空泡消除、FP8 降成本）

**完成标准**：
- 知识地图清晰可读（A4 一页纸）
- Phase5 README.md 写毕
- 自查清单全部打勾
- 能完成 2 分钟分布式训练概述

---

## 每日节奏建议

```
阅读 (30-60min) → 推导/编码 (1.5-2h) → 笔记/思考 (30-45min)
      ↓                    ↓                       ↓
  带着问题读           动手验证理解             写下来内化
```

- **不要跳过的环节**：笔记（30min 写笔记）。推过公式、写过代码，如果不写下来，一周后忘大半
- **遇到算不出来的数字**：回到 Day 61-62 的 Jupyter notebook，用自己的公式重新代一遍
- **代码卡住**：先自己 Debug 15min → 看报错信息仔细想 → 再查文档 or 问 AI

---

## 本文档与其他文件的关系

```
Phase5_System_Scale/
├── PLAN.md              ← 你正在读的文件（日级执行计划）
├── README.md            ← 目录说明和设计概览
├── System_Math/         ← Day 61-62 产出
├── Collectives/         ← Day 63-64 产出
├── Tensor_Parallel/     ← Day 65-66 产出
├── Pipeline_Parallel/   ← Day 67-68 产出
├── ZeRO/                ← Day 69-70 产出
├── Ring_Attention/      ← Day 71-72 产出
├── Expert_Parallel/     ← Day 73-74 产出
└── Parallelism_Recipe/  ← Day 75-76 产出
```

每完成一天的任务，在 TASK-PLAN.md 对应的 checklist 中打勾。16 天后，这份文档就是你 Phase 5 的学习档案。

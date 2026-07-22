# Triton RMSNorm 融合 Kernel

本子目录包含 RMSNorm 的 Triton 手写融合 Kernel 及性能 Benchmark，将多次显存读写融合为单次，突破 Memory-Bound 瓶颈。

> 完整的硬件数据流、SPMD 范式、代码逐行解析请参见 [triton_kernel_notes.md](triton_kernel_notes.md)。

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `my_rmsnorm_triton.py` | **Triton 融合 Kernel** — `_rmsnorm_fwd_kernel` + CPU Wrapper，一次 SRAM 驻留完成全部 RMSNorm 计算 |
| `benchmark_rmsnorm.py` | **性能 Benchmark** — 对比 torch_native vs compile vs triton vs my_torch 四种实现 |
| `triton_kernel_notes.md` | **完整知识库** — 变量物理字典、史诗级数据流、Kernel 代码逐行解剖、`tl.constexpr` 编译期常量深度解析 |
| `results_old/` | 旧版 Benchmark 结果（性能图表 + CSV） |
| `results_new/` | 新版 Benchmark 结果（性能图表 + CSV） |

## 快速运行

```bash
# Triton Kernel 冒烟测试
python my_rmsnorm_triton.py

# 性能 Benchmark
python benchmark_rmsnorm.py
```

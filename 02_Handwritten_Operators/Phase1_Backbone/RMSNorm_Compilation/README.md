# RMSNorm 编译优化与算子融合

本目录涵盖 RMSNorm 的完整实现链路——从纯 PyTorch 数学实现、`torch.compile` 图融合截获，到最终的 Triton 手写融合 Kernel。主线思路是：先用 Python 写出数学上正确的版本，再通过编译器或手写 Kernel 将多次显存读写融合为单次，突破 Memory-Bound 瓶颈。

## 目录结构

```
RMSNorm_Compilation/
├── README.md                      # 本文件——目录总览
├── Pytorch_Native/                # PyTorch 原生实现与 torch.compile 图融合
│   ├── README.md                  #   精简索引——文件清单与运行指令
│   ├── rmsnorm_notes.md           #   完整知识库：LayerNorm vs RMSNorm、尺度不变性、梯度推导
│   ├── my_layer_norm.py           #   手撕 LayerNorm（含 fp32 精度保护）
│   ├── my_RMSNorm.py              #   手撕 RMSNorm（与 nn.RMSNorm 对齐）
│   └── my_rmsnorm_compile.py      #   torch.compile 截获脚本——打印底层 Triton 代码
└── Triton/                        # Triton 手写融合 Kernel
    ├── README.md                  #   精简索引——文件清单与运行指令
    ├── triton_kernel_notes.md     #   完整知识库：硬件数据流、SPMD 范式、tl.constexpr 语义
    ├── my_rmsnorm_triton.py       #   Triton RMSNorm 融合算子 + CPU Wrapper
    ├── benchmark_rmsnorm.py       #   性能 Benchmark（torch_native vs compile vs triton vs my_torch）
    ├── results_old/               #   旧版 Benchmark 结果（性能图表 + CSV）
    └── results_new/               #   新版 Benchmark 结果（性能图表 + CSV）
```

## 学习路径

1. **入门**：从 `Pytorch_Native/my_layer_norm.py` 开始，理解 LayerNorm 的数学计算流。
2. **简化**：阅读 `Pytorch_Native/my_RMSNorm.py`，理解 RMSNorm 如何砍掉均值计算。
3. **理论**：深入 `Pytorch_Native/rmsnorm_notes.md`，掌握尺度不变性、梯度平滑、LayerNorm vs RMSNorm 对比。
4. **编译**：运行 `Pytorch_Native/my_rmsnorm_compile.py`，截获 `torch.compile` 生成的底层 Triton 代码。
5. **手写 Kernel**：阅读 `Triton/triton_kernel_notes.md` 和 `Triton/my_rmsnorm_triton.py`，理解如何手写 GPU Kernel 融合一次完成 RMSNorm。
6. **性能验证**：运行 `Triton/benchmark_rmsnorm.py`，对比四种实现在不同维度下的显存带宽吞吐。

## 相关目录

- [MoE/](../MoE/) — MoE 混合专家层（内部使用 SwiGLUFFN）
- [RoPE/](../RoPE/) — 旋转位置编码

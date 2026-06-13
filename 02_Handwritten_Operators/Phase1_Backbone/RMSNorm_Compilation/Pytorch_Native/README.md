# RMSNorm — PyTorch 原生实现

本子目录包含 LayerNorm / RMSNorm 的纯 PyTorch 手写实现，以及 `torch.compile` 图融合截获脚本。

> 完整的数学推导、尺度不变性理论、梯度分析及 `torch.compile` 实操指南请参见 [rmsnorm_notes.md](rmsnorm_notes.md)。

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `my_layer_norm.py` | **手撕 LayerNorm**（含 fp32 精度保护），与 `nn.LayerNorm` 对齐 |
| `my_RMSNorm.py` | **手撕 RMSNorm**（砍掉均值计算），与 `nn.RMSNorm` 对齐 |
| `my_rmsnorm_compile.py` | **torch.compile 截获脚本** — 打印编译器生成的底层 Triton 代码（需 CUDA 环境） |
| `rmsnorm_notes.md` | **完整知识库** — LayerNorm vs RMSNorm 理论、尺度不变性、梯度推导、`torch.compile` 图融合探秘 |

## 快速运行

```bash
# LayerNorm 冒烟测试
python my_layer_norm.py

# RMSNorm 冒烟测试
python my_RMSNorm.py

# torch.compile 截获（需 CUDA）
python my_rmsnorm_compile.py
```

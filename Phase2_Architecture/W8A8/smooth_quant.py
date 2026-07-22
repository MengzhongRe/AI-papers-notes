import torch
import torch.nn.functional as F
from quant_primitives import TensorQuantizer


def test_smooth_quant():
    """
    Day 38 TDD Check: The SmoothQuant Math Migration
    """
    print("\n--- Day 38: Applying SmoothQuant Math ---")
    torch.manual_seed(42)
    X = torch.randn(1, 128, 4096, dtype=torch.float32)
    W = torch.randn(4096, 4096, dtype=torch.float32)
    X[:, :, 1024] *= 150.0  # 依然注入 Outlier

    Y_gt = F.linear(X, W)

    # --- SmoothQuant 核心算法开始 ---
    print("\n[SmoothQuant] Computing smooth migration factors...")
    # 1. 寻找平滑因子 S (alpha 通常取 0.5)
    alpha = 0.5

    # 计算 X 沿 SeqLen 维度的最大绝对值 (形状: [4096])
    x_max = torch.max(torch.abs(X), dim=1)[0].squeeze()  # [1, 128, 4096] -> [1, 4096] -> [4096]

    # 计算 W 沿 Out 维度的最大绝对值 (形状: [4096])
    w_max = torch.max(torch.abs(W), dim=0)[0]  # [4096, 4096] -> [4096]

    # s = (x_max ** alpha) / (w_max ** (1-alpha) + 1e-5)
    smooth_scales = (x_max ** alpha) / (w_max ** (1 - alpha) + 1e-5)

    print(f"Smooth scales range: [{smooth_scales.min():.6f}, {smooth_scales.max():.6f}]")

    # 2. 难度迁移 (等价代换: X_new = X / S, W_new = W * S)
    # 注意维度广播
    print("Applying difficulty migration...")
    X_smoothed = X / smooth_scales.view(1, 1, -1)
    W_smoothed = W * smooth_scales.view(1, -1)

    print(f"Smoothed X range: [{X_smoothed.min():.6f}, {X_smoothed.max():.6f}]")
    print(f"Smoothed W range: [{W_smoothed.min():.6f}, {W_smoothed.max():.6f}]")

    # 3. 对平滑后的张量进行 W8A8 量化
    print("\n[SmoothQuant W8A8] Quantizing smoothed tensors...")
    quantizer = TensorQuantizer(symmetric=True)

    # 量化 X_smoothed
    sx, zx = quantizer.get_scale_and_zp(X_smoothed, dim=None)
    X_q = quantizer.quantize(X_smoothed, sx, zx)
    X_s_deq = quantizer.dequantize(X_q, sx, zx)

    # 量化 W_smoothed
    sw, zw = quantizer.get_scale_and_zp(W_smoothed, dim=0)
    W_q = quantizer.quantize(W_smoothed, sw, zw)
    W_s_deq = quantizer.dequantize(W_q, sw, zw)

    print(f"Activation scale: {sx.item():.6f}")
    print(f"Weight scale range: [{sw.min():.6f}, {sw.max():.6f}]")

    # 4. 执行矩阵乘法
    print("\nPerforming GEMM with quantized tensors...")
    Y_smooth_w8a8 = F.linear(X_s_deq, W_s_deq)

    mse_smooth = F.mse_loss(Y_gt, Y_smooth_w8a8)
    print(f"[Smooth W8A8] MSE Loss: {mse_smooth.item():.6e} (Miraculously fixed!)")

    return Y_gt, Y_smooth_w8a8, mse_smooth


def compare_all_methods():
    """
    综合对比: 纯 FP32, W8A16, Naive W8A8, SmoothQuant W8A8
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON: All Quantization Methods")
    print("=" * 70)

    torch.manual_seed(42)
    X = torch.randn(1, 128, 4096, dtype=torch.float32)
    W = torch.randn(4096, 4096, dtype=torch.float32)
    X[:, :, 1024] *= 150.0  # Outlier

    Y_gt = F.linear(X, W)

    # 1. W8A16
    w_quantizer = TensorQuantizer(symmetric=True)
    scale_w, zp_w = w_quantizer.get_scale_and_zp(W, dim=0)
    W_q = w_quantizer.quantize(W, scale_w, zp_w)
    W_deq = w_quantizer.dequantize(W_q, scale_w, zp_w)
    Y_w8a16 = F.linear(X, W_deq)
    mse_w8a16 = F.mse_loss(Y_gt, Y_w8a16)

    # 2. Naive W8A8
    x_quantizer = TensorQuantizer(symmetric=True)
    scale_x, zp_x = x_quantizer.get_scale_and_zp(X, dim=None)
    X_q = x_quantizer.quantize(X, scale_x, zp_x)
    X_deq = x_quantizer.dequantize(X_q, scale_x, zp_x)
    Y_w8a8_naive = F.linear(X_deq, W_deq)
    mse_w8a8_naive = F.mse_loss(Y_gt, Y_w8a8_naive)

    # 3. SmoothQuant W8A8
    alpha = 0.5
    x_max = torch.max(torch.abs(X), dim=1)[0].squeeze()
    w_max = torch.max(torch.abs(W), dim=0)[0]
    smooth_scales = (x_max ** alpha) / (w_max ** (1 - alpha) + 1e-5)

    X_smoothed = X / smooth_scales.view(1, 1, -1)
    W_smoothed = W * smooth_scales.view(1, -1)

    quantizer = TensorQuantizer(symmetric=True)

    sx, zx = quantizer.get_scale_and_zp(X_smoothed, dim=None)
    X_q = quantizer.quantize(X_smoothed, sx, zx)
    X_s_deq = quantizer.dequantize(X_q, sx, zx)

    sw, zw = quantizer.get_scale_and_zp(W_smoothed, dim=0)
    W_q = quantizer.quantize(W_smoothed, sw, zw)
    W_s_deq = quantizer.dequantize(W_q, sw, zw)

    Y_smooth_w8a8 = F.linear(X_s_deq, W_s_deq)
    mse_smooth = F.mse_loss(Y_gt, Y_smooth_w8a8)

    # 输出对比表
    print(f"\n{'Method':<20} {'MSE Loss':<15} {'Degradation vs FP32':<20}")
    print("-" * 70)
    print(f"{'FP32 Baseline':<20} {0.0:<15.6e} {'1.0x':<20}")
    print(f"{'W8A16':<20} {mse_w8a16.item():<15.6e} {f'{(mse_w8a16 / (mse_w8a16 + 1e-8)).item():.2f}x':<20}")
    print(f"{'Naive W8A8':<20} {mse_w8a8_naive.item():<15.6e} {f'{(mse_w8a8_naive / (mse_w8a16 + 1e-8)).item():.2e}x':<20}")
    print(f"{'SmoothQuant W8A8':<20} {mse_smooth.item():<15.6e} {f'{(mse_smooth / (mse_w8a16 + 1e-8)).item():.2f}x':<20}")
    print("=" * 70)

    print(f"\n✅ SmoothQuant recovers {((mse_w8a8_naive - mse_smooth) / mse_w8a8_naive * 100).item():.1f}% of performance loss!")


if __name__ == "__main__":
    test_smooth_quant()
    compare_all_methods()

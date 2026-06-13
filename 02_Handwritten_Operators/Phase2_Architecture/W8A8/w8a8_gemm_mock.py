import torch
import torch.nn.functional as F
from quant_primitives import TensorQuantizer


def test_outlier_collapse():
    """
    Day 37 TDD Check: W8A16 vs Naive W8A8 under Outliers
    """
    print("\n--- Day 37: Testing Outlier Collapse ---")
    torch.manual_seed(42)

    # 1. 模拟数据 (Batch=1, SeqLen=128, Hidden=4096)
    X = torch.randn(1, 128, 4096, dtype=torch.float32)
    W = torch.randn(4096, 4096, dtype=torch.float32)  # [Out, In]

    # 【高能预警】注入 LLM 特有的极端离群值 (Outliers)
    # 我们让第 1024 个特征通道异常巨大
    X[:, :, 1024] *= 150.0

    # Ground Truth (纯 FP32)
    Y_gt = F.linear(X, W)

    # 2. 模拟 W8A16 (Weight-Only Quantization)
    print("\n[W8A16] Quantizing weights only...")
    w_quantizer = TensorQuantizer(symmetric=True)
    scale_w, zp_w = w_quantizer.get_scale_and_zp(W, dim=0)  # Per-Channel 权重
    W_q = w_quantizer.quantize(W, scale_w, zp_w)
    W_deq = w_quantizer.dequantize(W_q, scale_w, zp_w)

    Y_w8a16 = F.linear(X, W_deq)  # W8A16 采用 On-the-fly 反量化后相乘
    mse_w8a16 = F.mse_loss(Y_gt, Y_w8a16)
    print(f"[W8A16] MSE Loss: {mse_w8a16.item():.6e} (Should be very low)")

    # 3. 模拟 Naive W8A8 (全量化，Tensor-wise 激活值)
    print("\n[Naive W8A8] Quantizing both weights and activations...")
    x_quantizer = TensorQuantizer(symmetric=True)
    scale_x, zp_x = x_quantizer.get_scale_and_zp(X, dim=None)  # 激活值用全局
    X_q = x_quantizer.quantize(X, scale_x, zp_x)
    X_deq = x_quantizer.dequantize(X_q, scale_x, zp_x)

    # 观察 X_deq。你会发现由于 scale_x 极大，原本正常的数值被挤压成了 0。
    print(f"Original X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"After dequant X range: [{X_deq.min():.6f}, {X_deq.max():.6f}]")
    print(f"X quantized to INT8 range: [{X_q.min()}, {X_q.max()}]")
    print(f"Scale X: {scale_x.item():.6f} (Huge scale squeezes normal values!)")

    Y_w8a8_naive = F.linear(X_deq, W_deq)
    mse_w8a8 = F.mse_loss(Y_gt, Y_w8a8_naive)
    print(f"[Naive W8A8] MSE Loss: {mse_w8a8.item():.6e} (Should EXPLODE!)")

    # 对比
    print("\n" + "=" * 60)
    print(f"MSE Ratio (Naive W8A8 / W8A16): {(mse_w8a8 / mse_w8a16).item():.2e}x")
    print("=" * 60)


if __name__ == "__main__":
    test_outlier_collapse()

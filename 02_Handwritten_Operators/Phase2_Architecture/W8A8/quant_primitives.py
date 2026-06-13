import torch
import torch.nn.functional as F


class TensorQuantizer:
    """
    Day 36: Base Quantization Primitives
    Goal: Map FP16/FP32 tensor to INT8 and back.
    """
    def __init__(self, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.symmetric = symmetric

        # 物理边界定义 (INT8)
        if self.symmetric:
            self.q_min = -(2**(num_bits - 1)) + 1  # -127 (严格对称, 舍弃-128)
            self.q_max = 2**(num_bits - 1) - 1     # 127
        else:
            self.q_min = 0                         # UINT8
            self.q_max = 2**num_bits - 1           # 255

    def get_scale_and_zp(self, x: torch.Tensor, dim=None):
        """计算 Scale 和 Zero-point。dim 指定量化粒度(None为全局, -1为per-channel)"""
        if self.symmetric:
            # 对称量化下，寻找绝对值的最大值
            x_max = torch.max(torch.abs(x), dim=dim, keepdim=True)[0] if dim is not None else torch.max(torch.abs(x))
            # scale = x_max / q_max. zp 强制为 0
            scale = x_max / self.q_max
            zp = torch.zeros_like(scale)
        else:
            # 非对称量化，找 x_min 和 x_max
            x_min = torch.min(x, dim=dim, keepdim=True)[0] if dim is not None else torch.min(x)
            x_max = torch.max(x, dim=dim, keepdim=True)[0] if dim is not None else torch.max(x)
            # scale = (x_max - x_min) / (q_max - q_min)
            scale = (x_max - x_min) / (self.q_max - self.q_min)
            # zp = round(q_min - x_min / scale)
            zp = torch.round(self.q_min - x_min / (scale + 1e-8))

        return scale, zp

    def quantize(self, x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
        """数学映射: X_q = clamp(round(X / S + Z), min, max)"""
        # 执行缩放、平移、四舍五入、和截断 (clamp)
        x_q = torch.round(x / (scale + 1e-8) + zp)
        x_q = torch.clamp(x_q, self.q_min, self.q_max)
        # 转为 int8 以模拟物理存储
        return x_q.to(torch.int8)

    def dequantize(self, x_q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
        """反向恢复: X_fp = (X_q - Z) * S"""
        # 将 x_q 转回浮点并执行反向公式
        x_fp = (x_q.float() - zp) * scale
        return x_fp


def test_primitives():
    """Day 36 TDD Check: Quantize and Dequantize with Low MSE"""
    print("\n--- Day 36: Testing Base Quantizer ---")
    torch.manual_seed(42)

    # 创建一个正态分布张量
    x = torch.randn(100, 100, dtype=torch.float32)

    # 对称量化
    quantizer_sym = TensorQuantizer(symmetric=True)
    scale, zp = quantizer_sym.get_scale_and_zp(x)
    x_q = quantizer_sym.quantize(x, scale, zp)
    x_deq = quantizer_sym.dequantize(x_q, scale, zp)

    mse = F.mse_loss(x, x_deq)
    print(f"[Symmetric] MSE Loss: {mse.item():.6e} (Should be < 1e-4)")

    # 非对称量化
    quantizer_asym = TensorQuantizer(symmetric=False)
    scale, zp = quantizer_asym.get_scale_and_zp(x)
    x_q = quantizer_asym.quantize(x, scale, zp)
    x_deq = quantizer_asym.dequantize(x_q, scale, zp)

    mse = F.mse_loss(x, x_deq)
    print(f"[Asymmetric] MSE Loss: {mse.item():.6e} (Should be < 1e-3)")


if __name__ == "__main__":
    test_primitives()

# RoPE 共享工具函数 — 预计算频率表 + 交错翻转
#
# 被 rope_embedding.py 和 rope_embedding_position.py 共同引用。
# 提取共享代码，消除 ~40 行重复。

import torch


def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE 的 cos/sin 矩阵（公式 34 的实数派实现）。

    Args:
        dim: 参与旋转的特征维度 (rope_dim)
        end: 最大支持的序列长度 (Seq_len)
        theta: 频率基数

    Returns:
        cos, sin: 形状均为 [end, dim] 的浮点张量
    """
    # 1. 计算 dim/2 个复平面的角速度: ω_i = 1 / (theta ** (2i / d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # 2. 生成绝对位置 t: [0, 1, 2, ..., end-1]
    t = torch.arange(end, dtype=torch.float32)

    # 3. 矩阵外积: 计算每个位置 × 每个平面的旋转角度 → [end, dim/2]
    freqs_outer = torch.outer(t, freqs)

    # 4. 对齐维度: [end, dim/2] → [end, dim]
    freqs_outer = torch.cat([freqs_outer, freqs_outer], dim=-1)

    # 5. 生成 cos/sin 常驻内存，推理时直接查表
    cos = torch.cos(freqs_outer)
    sin = torch.sin(freqs_outer)

    return cos, sin


def rotate_half(x: torch.Tensor):
    """
    实现公式 34 中的 [-x₂, x₁] 张量。
    将向量在最后一个维度上切成两半，后半取负放前面，前半放后面。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

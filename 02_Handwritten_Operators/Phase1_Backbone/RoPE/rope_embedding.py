# 手撕 RoPE（旋转位置嵌入向量）— 基础版
#
# 使用 rope_utils.py 中的共享函数 precompute_freqs_cos_sin 和 rotate_half。

import torch
from rope_utils import precompute_freqs_cos_sin, rotate_half


# =====================================================
# 模块：应用 RoPE 前向传播
# =====================================================
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = None):
    """
    应用 RoPE。支持 DeepSeek 的解耦机制（只旋转部分维度）和混合精度保护。

    Args:
        x: [Batch_size, Seq_len, num_heads, head_dim]
        cos, sin: [Seq_len, rope_dim]
        rope_dim: 参与旋转的维度大小。为 None 则全部旋转
    """
    head_dim = x.shape[-1]
    if rope_dim is None:
        rope_dim = head_dim

    # 1. 解耦机制：切开张量
    x_rot = x[..., :rope_dim]   # 需要接受旋转的维度
    x_pass = x[..., rope_dim:]   # 不需要旋转的维度

    # 2. 精度保护：强制提升为 fp32
    x_rot_fp32 = x_rot.float()

    # 3. 将 cos/sin 的形状从 [Seq_len, rope_dim] 广播为 [1, Seq_len, 1, rope_dim]
    seq_len = x_rot_fp32.shape[1]
    cos = cos[:seq_len].view(1, seq_len, 1, cos.shape[1]).to(x_rot_fp32.device)
    sin = sin[:seq_len].view(1, seq_len, 1, sin.shape[1]).to(x_rot_fp32.device)

    # 4. 计算公式 34，实现实数矩阵的旋转
    x_rotated_fp32 = (x_rot_fp32 * cos) + (rotate_half(x_rot_fp32) * sin)

    # 5. 还原现场：降级并拼接
    x_rotated = x_rotated_fp32.to(x.dtype)

    if rope_dim < head_dim:
        return torch.cat([x_rotated, x_pass], dim=-1)
    else:
        return x_rotated


# ====================================================================
# 测试用例
# =====================================================================
if __name__ == '__main__':
    # 1. 定义全局变量
    BATCH_SIZE = 2
    SEQ_LEN = 1024
    DIM = 4096
    NUM_HEADS = 32
    HEAD_DIM = DIM // NUM_HEADS
    ROPE_DIM = HEAD_DIM // 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print('启动 RoPE 工业级算子！')

    # 2. 构造输入数据
    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=dtype, device=device)
    cos, sin = precompute_freqs_cos_sin(dim=ROPE_DIM, end=SEQ_LEN, theta=10000.0)

    # 3. 对输入数据进行旋转
    q_rotated = apply_rotary_emb(q, cos, sin, ROPE_DIM)

    print(f'[*] 输入 Query 数据形状为: {q.shape}')
    print(f'[*] 输出 Query 形状: {q_rotated.shape}')
    print(f'cos/sin 缓存形状: {cos.shape}')

    # 4. 验证解耦机制是否生效
    diff_rotated = (q[..., :ROPE_DIM] - q_rotated[..., :ROPE_DIM]).abs().max().item()
    diff_pass = (q[..., ROPE_DIM:] - q_rotated[..., ROPE_DIM:]).abs().max().item()

    print('\n边界验证！')
    print(f' -> 旋转部分的变化量（前 {ROPE_DIM} 维）: {diff_rotated:.4f} (期望 > 0)')
    print(f' -> 未旋转部分的变化量（后 {ROPE_DIM} 维）: {diff_pass:.4f}')

    if diff_pass == 0.0 and diff_rotated > 0.1:
        print('\n伟大的胜利！解耦 RoPE 成功！')
    else:
        print('失败，请检查代码！')

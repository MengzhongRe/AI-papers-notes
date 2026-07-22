# 手撕 RoPE（旋转位置编码）+ position_ids 动态感知 — 适配 Decode 生成阶段
#
# 使用 rope_utils.py 中的共享函数，同时支持 Prefill 和 Decode 两个阶段。

import torch
from rope_utils import precompute_freqs_cos_sin, rotate_half


# ==================================================================================
# 模块：应用 RoPE 前向传播（动态 position_ids 版）
# ==================================================================================
def apply_rope_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   position_ids: torch.Tensor, rope_dim: int = None):
    """
    应用 RoPE，支持推理时的动态 position_ids。

    Args:
        x: [Batch_size, Seq_len, num_heads, head_dim]
        cos, sin: [max_seq_len, rope_dim] 预先计算好的整个缓存表
        position_ids: [Batch_size, Seq_len] 每个 token 的绝对位置
        rope_dim: 参与旋转的维度大小。为 None 则全部旋转
    """
    head_dim = x.shape[-1]
    if rope_dim is None:
        rope_dim = head_dim

    # 1. 解耦机制：切开张量
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # 2. 精度保护：把旋转的部分转换为 float32
    x_rot_fp32 = x_rot.float()

    # 3. 根据当前输入的 position_ids，从缓存表中抽取对应的 cos/sin
    #    position_ids 形状: [Batch_size, Seq_len]
    #    抽取后形状: [Batch_size, Seq_len, rope_dim]
    cos_sliced = cos[position_ids].to(x.device)
    sin_sliced = sin[position_ids].to(x.device)

    # 4. 广播: 插入 num_heads 维度 → [Batch_size, Seq_len, 1, rope_dim]
    cos_sliced = cos_sliced.unsqueeze(2)
    sin_sliced = sin_sliced.unsqueeze(2)

    # 5. 计算公式 34：实现实数矩阵的旋转
    x_rotated_fp32 = (x_rot_fp32 * cos_sliced) + (rotate_half(x_rot_fp32) * sin_sliced)

    # 6. 还原现场：降级并拼接
    x_rotated = x_rotated_fp32.to(x.dtype)

    if rope_dim < head_dim:
        return torch.cat([x_rotated, x_pass], dim=-1)
    else:
        return x_rotated


# ============================================================================
# 测试用例
# ============================================================================
if __name__ == '__main__':
    # 1. 定义全局变量
    BATCH_SIZE = 2
    SEQ_LEN = 1024   # 预填充阶段长度
    MAX_LEN = 2048   # 系统支持的最大长度
    NUM_HEADS = 32
    HEAD_DIM = 128
    ROPE_DIM = HEAD_DIM // 2
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('启动 RoPE 工业级算子（支持动态推理）...')

    # 预计算整个生命周期的三角函数并放入设备
    cos, sin = precompute_freqs_cos_sin(dim=ROPE_DIM, end=MAX_LEN, theta=10000.0)
    cos, sin = cos.to(device), sin.to(device)

    # ===========================================================================
    # 模拟 1：Prefill 阶段（处理一段完整的 Prompt）
    # ===========================================================================
    print('\n--- 模拟 Phase 2: Prefill 阶段 ---')
    q_prefill = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=dtype, device=device)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(BATCH_SIZE, -1)

    q_rotated_prefill = apply_rope_emb(q_prefill, cos, sin, position_ids, ROPE_DIM)
    print(f'[*] Prefill 阶段的输入形状为: {q_prefill.shape}')
    print(f'[*] Prefill 阶段的输出形状为: {q_rotated_prefill.shape}')

    # 验证解耦
    diff_rotated = (q_prefill[..., :ROPE_DIM] - q_rotated_prefill[..., :ROPE_DIM]).abs().max().item()
    diff_pass = (q_prefill[..., ROPE_DIM:] - q_rotated_prefill[..., ROPE_DIM:]).abs().max().item()
    assert diff_rotated > 0.1 and diff_pass == 0.0, 'Prefill 解耦失败'
    print('✅ Prefill 阶段解耦测试通过')

    # ===========================================================================
    # 模拟 2：Decode 阶段（自回归生成下一个 Token）
    # ===========================================================================
    print('\n--- 模拟 Phase 3: Decode 阶段 ---')
    q_decode = torch.randn((BATCH_SIZE, 1, NUM_HEADS, HEAD_DIM), dtype=dtype, device=device)
    # 关键：传入这个 Token 在全量文本中的绝对位置（第 1024 个位置）
    position_ids = torch.tensor([[1024], [1024]], dtype=torch.long, device=device)

    q_decode_rotated = apply_rope_emb(q_decode, cos, sin, position_ids, ROPE_DIM)
    print(f'[*] Decode 输入形状为: {q_decode.shape}')
    print(f'[*] Decode 输出形状为: {q_decode_rotated.shape}')

    # 验证 Decode 阶段的解耦
    diff_rotated_d = (q_decode[..., :ROPE_DIM] - q_decode_rotated[..., :ROPE_DIM]).abs().max().item()
    diff_pass_d = (q_decode[..., ROPE_DIM:] - q_decode_rotated[..., ROPE_DIM:]).abs().max().item()
    assert diff_rotated_d > 0.1 and diff_pass_d == 0.0, 'Decode 解耦失败'
    print('✅ Decode 流水线测试通过（Seq_len=1 正确广播，解耦验证通过）')

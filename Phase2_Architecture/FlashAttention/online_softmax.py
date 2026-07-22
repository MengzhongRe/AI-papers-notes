# 手撕 FlashAttention 前向传播 — Online Softmax 分块更新

import torch
import math


def flash_attention_forward_block(Q_block, K_block, V_block, O_old, m_old, l_old, mask_block=None):
    """
    模拟 FlashAttention 中 SRAM 里的一个块更新逻辑，外层 Q 循环，内层 K/V 循环。

    维度说明：
        B: 批量大小
        H: 注意力头数
        Br: Q 的行块大小
        Bc: K/V 的列块大小
        d: 注意力头维度

        Q_block:  [B, H, Br, d]
        K_block:  [B, H, Bc, d]
        V_block:  [B, H, Bc, d]
        O_old:    [B, H, Br, d]   — 之前的累积输出
        m_old:    [B, H, Br, 1]   — 之前的逐行最大值
        l_old:    [B, H, Br, 1]   — 之前的逐行 softmax 分母
    """
    d = Q_block.shape[-1]

    # 1. 计算注意力局部分数并缩放
    # [B, H, Br, d] @ [B, H, d, Bc] -> [B, H, Br, Bc]
    S_local = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d)

    if mask_block is not None:
        S_local = S_local.masked_fill(mask_block, -1e9)

    # 2. 行最大值
    # torch.max(dim=...) 返回 (values, indices) 元组，取 [0] 得到 values
    # m_local: [B, H, Br, 1]
    m_local = torch.max(S_local, dim=-1, keepdim=True)[0]

    # 3. 更新全局最大值（逐元素取 max）
    # m_new: [B, H, Br, 1]
    m_new = torch.maximum(m_old, m_local)
    decay = torch.exp(m_old - m_new)

    # 4. 局部 softmax（减去 m_new 防溢出）
    # P_local: [B, H, Br, Bc]
    P_local = torch.exp(S_local - m_new)
    # l_local: [B, H, Br, 1]
    l_local = torch.sum(P_local, dim=-1, keepdim=True)

    # 5. 更新全局分母：旧分母乘衰减系数 + 新分母
    # l_new: [B, H, Br, 1]
    l_new = l_old * decay + l_local

    # 6. 更新输出 O
    # 旧 O 需要乘回旧分母（去归一化），再乘衰减系数，加上新的 P_local @ V_block
    # P_local @ V_block: [B, H, Br, Bc] @ [B, H, Bc, d] -> [B, H, Br, d]
    O_unnormalized = O_old * l_old * decay + torch.matmul(P_local, V_block)

    # 7. 用新分母归一化
    # O_new: [B, H, Br, d]
    O_new = O_unnormalized / l_new

    return O_new, m_new, l_new


# ============================================================
# 冒烟测试
# ============================================================
if __name__ == '__main__':
    print('========== Online Softmax 分块更新 — 冒烟测试 ==========')
    B, H, N, d = 2, 4, 128, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)

    # 单块测试：取前 32 个 token 作为一块
    Br = 32
    Q_block = Q[:, :, :Br, :]
    K_block = K[:, :, :Br, :]
    V_block = V[:, :, :Br, :]

    O_old = torch.zeros(B, H, Br, d, device=device)
    m_old = torch.full((B, H, Br, 1), float('-inf'), device=device)
    l_old = torch.zeros(B, H, Br, 1, device=device)

    O_new, m_new, l_new = flash_attention_forward_block(
        Q_block, K_block, V_block, O_old, m_old, l_old
    )
    print(f'[*] O_new 形状: {O_new.shape}  (期望: [{B}, {H}, {Br}, {d}])')
    print(f'[*] m_new 形状: {m_new.shape}')
    print(f'[*] l_new 形状: {l_new.shape}')
    assert not torch.isnan(O_new).any(), 'O_new 包含 NaN!'
    assert not torch.isinf(O_new).any(), 'O_new 包含 Inf!'

    # 与标准 softmax 对比
    scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d)
    O_std = torch.matmul(torch.softmax(scores, dim=-1), V_block)
    diff = (O_new - O_std).abs().max().item()
    print(f'[*] 与标准 softmax 的最大误差: {diff:.6e}')
    assert torch.allclose(O_new, O_std, atol=1e-5), '实现与标准 softmax 不一致！'
    print('✅ Online Softmax 单块更新验证通过')

# 手动实现FlashAttetion-2,外层Q,O；内层K,V的循环。数值稳定性，实现快速因果掩码减少计算量

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: bool = True, block_size: int = 64) -> torch.Tensor:
    """
    实现FlashAttention-v2,外层Q,O循环;内层K,V,并兼容快速掩码矩阵的前向传播函数
    参数:
        Q: [B,H,N,d],HBM中的query矩阵
        K: [B,H,N,d],HBM中的key矩阵
        V: [B,H,N,d],HBM中的value矩阵
        mask: 是否实现因果掩码
        block_size: Bc,Br的大小
    返回:
        O: [B,H,N,d],结果张量
    """
    B,H,N,d = Q.shape
    device = Q.device

    # 在HBM中初始化结果张量O,全局统计量m,l
    O = torch.zeros_like(Q)
    m = torch.full((B,H,N,1), float('-inf'), device=device)
    l = torch.zeros((B,H,N,1),device=device)

    # 计算行块数，列块数
    Tr = (N + block_size - 1) // block_size
    Tc = (N + block_size - 1) // block_size

    # 外层循环qi,oi,mi,li,
    for i in range(Tr):
        r_start = i * block_size
        r_end = min((i + 1) * block_size, N)
        # 计算实际行块大小
        Br = r_end - r_start

        # 从HBM中搬运初始化对应行的qi,oi,li,mi
        qi = Q[:, :, r_start:r_end, :]  # [B,H,Br,d]
        oi = O[:, :, r_start:r_end, :].clone()   # [B,H,Br,d]
        mi = m[:, :, r_start:r_end, :].clone()
        li = l[:, :, r_start:r_end, :].clone() 

        # 内层循环kj,vj
        for j in range(Tc):
            # 如果是全掩蔽，直接跳过
            if mask and (j > i):
                continue
            # 计算列块的起始和结束索引
            c_start = j * block_size
            c_end = min((j + 1) * block_size, N)
            # 计算列块实际大小
            Bc = c_end - c_start

            # 从HBM中读取kj,vj
            kj = K[:, :, c_start:c_end, :] # [B,H,Bc,d]
            vj = V[:, :, c_start:c_end, :] # [B,H,Bc,d]
            # 计算局部注意力分数sij: [Br,Bc]
            sij = torch.matmul(qi,kj.transpose(-2,-1)) / math.sqrt(d)

            # --------------- 块内掩蔽操作 ----------------------------
            # 如果是黄灯块，实现掩蔽
            if mask and (i == j):
                # 计算块内元素的全局实际行列索引(row_idx, col_idx)
                # 如果row_idx >= col_idx则不掩蔽，否则掩蔽
                # row_idx: [Br,1]
                row_idx = torch.arange(r_start,r_end,device=device).view(-1,1)
                # col_idx: [Bc]
                col_idx = torch.arange(c_start,c_end,device=device)
                sij = torch.where(row_idx >= col_idx, sij, float('-inf'))
            
            # ---------------- online softmax 核心公式 -------------------------
            m_block = torch.max(sij,dim=-1,keepdim=True)[0] # [B,H,Br,1]
            m_new = torch.maximum(mi,m_block) # [B,H,Br,1]
            # 计算指数衰减系数
            alpha = torch.exp(mi - m_new)
            p_tilde = torch.exp(sij - m_new)    # [B,H,Br,Bc]
            l_block = torch.sum(p_tilde,dim=-1,keepdim=True) # [B,H,Br,1]

            # 更新未归一化的累加和oi: [B,H,Br,d]
            oi = oi * alpha + torch.matmul(p_tilde, vj)
            # 更新li,mi
            li = li * alpha + l_block
            mi = m_new
        
        # 所有内层循环结束后，将完整的oi,li,mi写入大矩阵O,m,l
        # oi写入前需要除以li归一化
        O[:, :, r_start:r_end, :] = oi / li
        m[:, :, r_start:r_end, :] = mi
        l[:, :, r_start:r_end, :] = li
    
    return O

def test_flash_attention():
    print('-------- 开始进行FlashAttention 算子测试 ------------------')
    # 1.定义全局变量
    B = 2
    H = 32
    N = 1024
    d = 512
    block_size = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # 2.初始化Q,K,V
    Q = torch.randn((B,H,N,d),device=device)
    K = torch.randn((B,H,N,d),device=device)
    V = torch.randn((B,H,N,d),device=device)

    print(f'[*] 输入矩阵形状为: {Q.shape}')

    # 3.计算flash_attention的O
    O_flash = flash_attention_forward(Q,K,V,mask=True,block_size=64)

    # 4.计算标准注意力的输出O_std: [B,H,N,d]
    scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d)
    # 创造下三角因果掩码矩阵
    causal_mask = torch.tril(torch.ones(1,1,N,N)).to(device)
    scores.masked_fill_(causal_mask==0,float('-inf'))
    attention_weights = torch.softmax(scores,dim=-1)
    O_std = torch.matmul(attention_weights,V)

    print(f'[*] 输出矩阵形状为: {O_flash.shape}')
    diff = (O_flash - O_std).abs().max().item()
    print(f'最大精度误差为: {diff:.4e}')
    assert torch.allclose(O_flash,O_std,atol=1e-5),'❌ FlashAttetion实现和标准注意力实现不一致！'
    print('✅ FlashAttetion实现和标准注意力实现一致')

if __name__ == '__main__':
    test_flash_attention()






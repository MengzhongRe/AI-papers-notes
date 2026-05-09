import torch
import math

# 用pytorch手撕实现Flashattention的前向传播逻辑，尤其是在线softmax部份（外层Q视角）
def flash_attention_forward_block(Q_block,K_block,V_block,O_old,m_old,l_old,mask_block=None):
    """
    模拟Flashattention中SRAM里的一个块的更新逻辑，外层Q循环，内存K，V循环

    维度说明：
    B:批量大小
    H：注意力头数
    Br：Q的行块大小
    Bc：K，V的列块大小
    d:注意力头维度

    Q_block:[B,H,Br,d]
    K_blocl:[B,H,Bc,d]
    V_block:[B,H,Bc,d]
    O_old:[B,H,Br,d]
    m_old:[B,H,Br,1]
    l_old:[B,H,Br,1]
    """
    d = Q_block.shape[-1]
    # 1. 计算注意力局部分数同时缩放
    # [B,H,Br,d] @ [B,H,d,Bc] -> [B,H,Br,Bc]
    S_local = torch.matmul(Q_block,K_block.transpose(-2,-1)) / math.sqrt(d)

    # 掩蔽
    if mask_block is not None:
        S_local = S_local.masked_fill(mask_block,-1e9)

    # 2. 求S_local的每行的最大值，即m_local
    # torch.max()沿着某个维度求最大值的时候，返回的是(values,indices)的元组！我们必须取第一个值
    # m_local: [B,H,Br,1]
    m_local = torch.max(S_local,dim=-1,keepdim=True)[0]

    # 3. 算出新的全局最大值：不用torch.max,用torch.amximum实现两个相同维度张量的逐元素比较
    # m_new: [B,H,Br,1]
    m_new = torch.maximum(m_old,m_local)
    decay = torch.exp(m_old - m_new) # 计算指数衰减因子

    # 4. 计算局部概率和分母（直接减去m_new）
    # P_local: [B,H,Br,Bc]
    P_local = torch.exp(S_local - m_new)
    # l_local: [B,H,Br,1]
    l_local = torch.sum(P_local,dim=-1,keepdim=True)

    # 4.更新全局分母：旧分母需乘以指数衰减系数再加上新分母
    # l_new: [B,H,Br,1]
    l_new = l_old * decay + l_local

    # 5. 更新输出O
    # 旧的O需要乘回旧分母，再乘指数衰减系数，加上新的P_local @ V_block
    # P_local @ V_block: [B,H,Br,Bc] @ [B,H,Bc,d] -> [B,H,Br,d]
    O_unnormalized = O_old * l_old * decay + torch.matmul(P_local,V_block)

    # 6. 对输出用新分母和进行规约
    # O_new：[B,H,Br,d]
    O_new = O_unnormalized / l_new 

    return O_new,m_new,l_new


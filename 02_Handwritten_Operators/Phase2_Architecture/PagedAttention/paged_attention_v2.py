import torch
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PhysicalKVMemoryPool():
    def __init__(self, num_blocks: int, block_size: int,
                num_kv_heads: int, head_dim: int, device: str = DEVICE):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 分配num_blocks个每个放block_size个token的对应num_kv_heads头数量
        # 以及head_dim维度的kv cache物理块
        # 实际中会提前分配超大块的连续显存
        self.k_pool = torch.zeros((num_blocks,num_kv_heads,block_size,head_dim),device=device)
        self.v_pool = torch.zeros((num_blocks,num_kv_heads,block_size,head_dim),device=device)
    
def paged_attention(q: torch.Tensor, kv_pool: PhysicalKVMemoryPool, 
                    block_table: list[int], context_len: int) -> torch.Tensor:
    num_heads,head_dim = q.size()
    block_size = kv_pool.block_size
    scale = 1.0 / math.sqrt(head_dim)
    # 分配Online Softmax的m,l,o
    m_global = torch.full((num_heads,1),float('-inf'),device=DEVICE)
    l_global = torch.zeros((num_heads,1),device=DEVICE)
    out_global = torch.zeros((num_heads,head_dim),device=DEVICE)
    q_unsqueezed = q.unsqueeze(1)

    # 从页表映射中分块读取当前块的k,v cache
    for logical_idx, physical_block_dix in enumerate(block_table):
        # 从kv_pool切片读取当前块的kv cache
        k_block = kv_pool.k_pool[physical_block_dix]
        v_block = kv_pool.v_pool[physical_block_dix]
        # 注意力计算
        scores = torch.matmul(q_unsqueezed,k_block.transpose(1,2)) * scale
        scores = scores.squeeze(1)
        # 对最后一块可能越界的token进行掩蔽
        start_token_idx = logical_idx * block_size
        end_token_idx = start_token_idx + block_size
        if end_token_idx > context_len:
            val_len_in_block = context_len - start_token_idx
            scores[:, val_len_in_block:] = float('-inf')
        
        # Online Softmax核心逻辑
        m_block = torch.max(scores,dim=-1,keepdim=True)[0]  # [num_heads,1]
        m_new = torch.maximum(m_global,m_block)
        alpha = torch.exp(m_global - m_new)
        p_tilde = torch.exp(scores - m_new)
        l_block = torch.sum(p_tilde,dim=-1,keepdim=True)

        out_global = out_global * alpha
        out_block = torch.matmul(p_tilde.unsqueeze(1),v_block).squeeze(1)
        out_global += out_block
        l_global = l_global * alpha + l_block
        m_global = m_new
    
    final_out = out_global / l_global
    return final_out

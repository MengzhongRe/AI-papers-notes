# 用Python实现PagedAttention算法
import torch
import math

# =====================================================
# 模块一: GPU上的全局物理显存池
# =====================================================
class PhysicalKVMemoryPool():
    """
    模拟GPU上的全局KV cache显存池
    在实际的vLLM引擎中这是预先分配好的巨大的连续的显存
    """
    def __init__(self, num_blocks: int, block_size: int,
                num_kv_heads: int, head_dim: int, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 分配对应num_blocks个每个大小固定block_size的带有num_kv_heads个注意力头数的K V缓冲池
        self.k_pool = torch.zeros((num_blocks,num_kv_heads,block_size,head_dim),device=device)
        self.v_pool = torch.zeros((num_blocks,num_kv_heads,block_size,head_dim),device=device)
    
# =========================================================
# 模块二: PagedAttention 核心算法
# ========================================================
class PagedAttention():
    
    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward_decode(self, q: torch.Tensor, kv_pool: PhysicalKVMemoryPool,
                    block_table: list[int], context_len: int):
        """
        执行Decode阶段的 PagedAttention计算(即生成一个新token)
        参数:
            q: [num_q_heads,head_dim],当前 step 生成的query向量,
                这里简化处理,bacth_size为1,并且不考虑GQA,因此q_heads = kv_heads
            kv_pool: 全局物理显存池
            block_table: 当前请求的列表,例如[7,1,3] 表示逻辑块0,1,2在物理池中的对应
            context_len: 当前请求的历史总长度,包括历史和当前token
        """
        num_heads = q.size(0)
        block_size = kv_pool.block_size
        device = q.device

        # ============================================
        # 初始化Online Softmax的全局变量
        # 这里由于query长度为1,因此m,l,o的长度也是1
        # m_global 初始化为全负无穷
        m_global = torch.full((num_heads,1),float('-inf'),device=device)
        # l_global 初始化为全0
        l_global = torch.zeros((num_heads,1),device=device)
        out_global = torch.zeros((num_heads,self.head_dim),device=device)
        q_unsqueezed = q.unsqueeze(1)   # [num_heads,1,head_dim]

        # 遍历该请求页表中的每一个物理块
        for logical_idx, physical_block_idx in enumerate(block_table):
            # 从全局显存池中零拷贝切片当前块的KV
            # k_block: [num_heads,block_size,head_dim]
            # v_block: [num_heads,block_size,head_dim]
            k_block = kv_pool.k_pool[physical_block_idx]
            v_block = kv_pool.v_pool[physical_block_idx]

            # 注意力分数计算
            # 计算query与当前K块的注意力分数
            # q: [num_heads,head_dim], k_block: [num_hedas,block_size,head_dim]
            # 需要将q在中间增加一个维度,由于q是循环外变量我们最好在循环外变更其形状
            scores = torch.matmul(q_unsqueezed,k_block.transpose(-2,-1)) * self.scale  # [num_heads,1,block_size]
            scores = scores.squeeze(1)  # [num_heads,block_size]

            # Mask:边界处理
            # 如果这是最后一块,可能没有被填满，我们需要将超出的部分mask掉
            start_token_idx = logical_idx * block_size
            end_token_idx = start_token_idx + block_size
            if end_token_idx > context_len:
                # 计算当前块的有效token数量
                valid_len_in_block = context_len - start_token_idx
                # 对无效部分的注意力取负无穷
                scores[:, valid_len_in_block:] = float('-inf')
            
            # Online Softmax核心模块
            # 计算当前块的最大值
            # torch.max()会返回(values,indices)元组，我们仅返回第一个值
            m_local = torch.max(scores,dim=-1,keepdim=True)[0]  # [num_heads,1]
            # 计算全局最大值
            m_new = torch.maximum(m_global,m_local) # [num_heads,1]
            # 计算指数衰减系数
            alpha = torch.exp(m_global - m_new) # [num_heads,1]
            p_tilde = torch.exp(scores - m_new) # [num_heads,block_size]
            # 计算当前块的指数和
            l_local = torch.sum(p_tilde,dim=-1,keepdim=True) # [num_heads,1]
            # [num_heads,1,block_size] @ [num_heads,block_size,head_dim]
            out_local = torch.matmul(p_tilde.unsqueeze(1),v_block).squeeze(1)  # [num_heads,head_dim]
            # 更新out_global: 对以前的out乘alpha归一化
            out_global = out_global * alpha + out_local
            # 更新全局指数和
            l_global = l_global * alpha + l_local
            # 更新全局最大值
            m_global = m_new
        # 当内层循环所有的k,v块迭代结束后,则本query的输出完,最后需要将输出归一化
        final_out = out_global / l_global
        return final_out    # [num_heads,head_dim]
            

# ===============================================
# 测试用例
# =============================================
def test_paged_attention():
    # 1.设置全局变量
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_BLOCKS = 8
    BLOCK_SIZE = 4
    NUM_HEADS = 32
    HEAD_DIM = 128
    # 初始化物理kv缓冲池
    kv_pool = PhysicalKVMemoryPool(NUM_BLOCKS,BLOCK_SIZE,NUM_HEADS,HEAD_DIM,device=DEVICE)
    # 在物理块7,1,3构造kv cache
    kv_pool.k_pool[7] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    kv_pool.v_pool[7] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    kv_pool.k_pool[1] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    kv_pool.v_pool[1] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    kv_pool.k_pool[3] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    kv_pool.v_pool[3] = torch.randn((NUM_HEADS,BLOCK_SIZE,HEAD_DIM))
    # 构造页表
    block_table = [7,1,3]
    # 三个物理块，我们仅构造11长度的context_len,测试能否掩蔽最后一个物理块的最后一个token值
    context_len = 11
    # 实例化PagedAttention类
    paged_attention_layer = PagedAttention(HEAD_DIM)
    # 构造query张量
    q = torch.randn((NUM_HEADS,HEAD_DIM),device=DEVICE)
    print(f'输入query张量的形状为: {q.shape}')
    out = paged_attention_layer.forward_decode(q,kv_pool,block_table,context_len)
    print(f'输出张量形状为: {out.shape}, 应该为: {(NUM_HEADS,HEAD_DIM)}')

    # ----- 逻辑等价性校验 -------------
    # 我们为了验证算法中online softmax以及mask逻辑是否正确,等价于直接验证最终数值的正确
    # 最终的数值数学上就等于将几个k,v拼接起然后截断掉最后一个token再直接与q做MHA的结果一致
    k_list = [kv_pool.k_pool[idx] for idx in block_table]
    v_list = [kv_pool.v_pool[idx] for idx in block_table]

    k_cat = torch.cat(k_list,dim=1) # [num_heads,12,head_dim]
    v_cat = torch.cat(v_list,dim=1) 
    # 直接把最后一个tokrn截断不要
    k_val = k_cat[:, :context_len, :]   # [num_heads,11,head_dim]
    v_val = v_cat[:, :context_len, :]   #   
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores_ref = torch.matmul(q.unsqueeze(1),k_val.transpose(1,2)) * scale  # [num_heads,1,11]
    atten_ref = torch.softmax(scores_ref,dim=-1)    # [num_heads,1,11]
    out_ref = (atten_ref @ v_val).squeeze(1)    # [num_heads,head_dim]
    diff = (out - out_ref).abs().max().item()
    print(f'与标准实现的最大误差为: {diff:.4e}')
    assert torch.allclose(out,out_ref,atol=1e-4),' ❌ 与官方实现误差过大!'
    print(' ✅ 恭喜你,你的实现成功了！')


if __name__ == '__main__':
    test_paged_attention()


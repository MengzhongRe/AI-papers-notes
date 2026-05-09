import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self,d_model,num_heads,num_kv_heads,dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0,'d_model must be divisible by num_heads'
        assert num_heads % num_kv_heads == 0,'num_heads must be divisible by num_kv_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads

        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model,num_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(d_model,num_kv_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(d_model,num_kv_heads * self.head_dim,bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim,d_model,bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,past_key_value=None,use_cache=False):
        """
        x: [Batch_size,Seq_len,d_model]
        past_key_value:tuple of (past_keys,past_values)
        use_cache:是否返回当前的KVCache供下一步解码使用
        """
        B,N_curr,_ = x.size()
        # ====================
        # 1.投影并改变视图
        # ====================
        # q:[B,N_curr,num_heads * head_dim] -> [B,N_curr,num_heads,head_dim] -> [B,num_heads,N_curr,head_dim]
        q = self.q_proj(x).view(B,N_curr,self.num_heads,self.head_dim).transpose(1,2)
        # k,v:[B,N_curr,num_kv_heads * head_dim] -> [B,N_curr,num_kv_heads,head_dim]-> [B,num_kv_heads,N_curr,head_dim]
        k = self.k_proj(x).view(B,N_curr,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,N_curr,self.num_kv_heads,self.head_dim).transpose(1,2)

        # ====================
        # 2.处理KV缓存（如果有）
        # ====================
        if past_key_value is not None:
            past_k,past_v = past_key_value
            # 将当前的k,v与缓存中的k,v拼接
            # past_k: [B,num_kv_heads,N_past,head_dim]
            # 拼接后的k,v的维度:[B,num_kv_heads,N_past + N_curr,head_dim]
            k = torch.cat([past_k,k],dim=2)
            v = torch.cat([past_v,v],dim=2)

        # 如果需要缓存把当前最新完整的K，V保存下来
        present_key_value = (k,v) if use_cache else None

        # 记录当前key序列的长度
        N_kv = k.size(2)

        # =========================
        # 3. GQA维度对齐
        # 目标：k,v:[B,num_kv_heads,N_kv,head_dim] -> [B,num_heads,N_kv,head_dim]
        # ===================================================================
        # 步骤一:unsqueeze(2)变成[B,num_kv_heads,1,N_kv,head_dim]
        # 步骤二:expand扩展为[B,num_kv_heads,num_queries_per_kv,N_kv,head_dim]，此操作不分配新物理内存
        # 步骤三：reshape压平为[B,num_heads,N_kv,head_dim]
        k_expanded = k.unsqueeze(2).expand(B,self.num_kv_heads,self.num_queries_per_kv,N_kv,self.head_dim).view(B,self.num_heads,N_kv,self.head_dim)
        v_expanded = v.unsqueeze(2).expand(B,self.num_kv_heads,self.num_queries_per_kv,N_kv,self.head_dim).view(B,self.num_heads,N_kv,self.head_dim)

        # ===============================
        # 4. 计算缩放点积注意力
        # q:[B,num_heads,N_curr,head_dim]
        # k_expanded:[B,num_heads,N_kv,head_dim] -> [B,num_heads,head_dim,N_kv]
        # scores = q @ K^ -> [B,num_heads,N_curr,N_kv]
        scores = torch.matmul(q,k_expanded.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # Causal Mask(因果掩码)处理
        # 仅当N_curr > 1 即模型处于训练或预填充阶段作因果掩蔽
        if N_curr > 1:
            # 生成一个下三角为True，右上角为False的矩阵
            # mask: [N_curr,N_kv]
            mask = torch.tril(torch.ones(N_curr,N_kv,device=x.device)).bool()
            scores = scores.masked_fill(~mask,-1e9)
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores,dim=-1)
        # 正则化
        attention_weights = self.dropout(attention_weights)

        # 计算加权和
        # out:[B,num_heads,N_curr,N_kv] @ [B,num_heads,N_kv,head_dim] -> [B,num_heads,N_curr,heda_dim]
        out = torch.matmul(attention_weights,v_expanded)

        # 5.合并多头并输出
        out = out.transpose(1,2).contiguous().view(B,N_curr,-1)
        out = self.o_proj(out)

        return out,present_key_value

# =============================
# 测试用例
# =============================
if __name__ == '__main__':
    B = 2
    d_model = 4096
    num_heads = 32
    num_kv_heads = 8

    gqa = GroupedQueryAttention(d_model=d_model,num_heads=num_heads,num_kv_heads=num_kv_heads)

    print('=================阶段一:Prefill(预填充阶段)=================')
    # 用户输入了长度为10的prompt
    seq_len = 10
    x_prefill = torch.randn(B,seq_len,d_model)

    # 首次forward，没有历史cache,但需要返回cache
    out_prefill,kv_cache = gqa(x_prefill,past_key_value=None,use_cache=True)

    print(f'Prefill 输出维度： {out_prefill.shape}')  # 期望： [B,seq_len,d_model]即[2,10,4096]
    print(f'kv_cache 输出维度： {kv_cache[0].shape}') # 期望：[B,num_kv_heads,N_kv,head_dim]即[2,8,10,128]

    print('\n================阶段二:Decoding(逐字解码阶段)================')
    # 模型开始吐第十一个词，此时输入的序列长度永远是1
    x_decode_step_1 = torch.randn(B,1,d_model)

    # 传入刚生成的1个词，以及刚才存下来的kv_cahce
    out_decode_1,kv_cache = gqa(x_decode_step_1,past_key_value=kv_cache,use_cache=True)

    print(f'Decoding Step 1 输出维度:{out_decode_1.shape}') # 期望：[2,1,4096]
    print(f'更新后的KV cahce的维度: {kv_cache[0].shape}') # 期望: [2,8,11,128]

    # 模型吐出第十二个词
    x_decode_step_2 = torch.randn(B,1,d_model)
    out_decode_2,kv_cache = gqa(x_decode_step_2,past_key_value=kv_cache,use_cache=True)
    print(f'Decoding Step 2 输出维度:{out_decode_2.shape}') # 期望：[2,1,4096]
    print(f'更新后的KV cahce的维度: {kv_cache[0].shape}') # 期望: [2,8,12,128]
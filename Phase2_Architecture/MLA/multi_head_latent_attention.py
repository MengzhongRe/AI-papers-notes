# 用pytorch手撕Deepseek-v2 MLA(multi_head_latent_attention)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple,Optional

def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    对 x 应用旋转位置编码（RoPE）。

    注意：当前为教学占位实现（identity），仅保证 MLA 架构逻辑完整。
    完整的 RoPE 实现见 Phase1_Backbone/RoPE/rope_embedding.py。
    如需集成，将本函数替换为 `apply_rotary_emb` 调用即可。
    """
    return x

class MultiHeadLatentAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_head: int, d_rope: int,
                q_lora_rank: int, kv_lora_rank: int, dropout_p: float):
        """
        参数:
            d_model: 模型的隐藏层维度
            num_heads: 注意力头数量
            d_head: 单头注意力维度
            d_rope: 每个注意力头的位置编码的维度
            q_lora_rank: query_latent即q被低秩压缩后的维度
            kv_lora_rank: kv-joint-latent低秩压缩后的维度
        权重:
            1. Query的压缩以及旋转投影矩阵:
                W_dq: [d_model, q_lora_rank], Query的Down projection矩阵
                W_uq: [q_lora_rank, num_heads * d_head], Query latent的Up projection
                W_qr: [q_lora_rank, num_heads * d_rope], Query latent的旋转位置矩阵
            2. KV的压缩及其旋转投影矩阵
                W_dkv: [d_model, kv_lora_rank], KV的联合压缩矩阵
                W_uk: [kv_lora_rank, num_heads * d_head], K的Up projection
                W_uv: [kv_lora_rank, num_heads * d_head], V的Up projection
                W_kr: [d_model, d_rope], K的单头旋转位置投影矩阵,Key的所有头共享一个位置向量编码
            3. Out投影矩阵
                Wo: [num_heads * d_head, d_model], 最终输出的投影矩阵
            
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_rope = d_rope
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # 1. Query的压缩及其位置旋转矩阵
        self.W_dq = nn.Linear(d_model, q_lora_rank, bias=False)
        self.W_uq = nn.Linear(q_lora_rank, num_heads * d_head, bias=False)
        self.W_qr = nn.Linear(q_lora_rank, num_heads * d_rope,bias=False)
        # 2. KV的压缩及key的旋转位置矩阵
        # 生成c_tkv,kv-joint-latent
        self.W_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.W_uk = nn.Linear(kv_lora_rank, num_heads * d_head, bias=False)
        self.W_uv = nn.Linear(kv_lora_rank, num_heads * d_head, bias=False)
        # K 的解耦 RoPE 投影 (注意：DeepSeek-v2中，K的RoPE是所有头共享的！)
        self.W_kr = nn.Linear(d_model, d_rope, bias=False)
        # 3. Out的投影矩阵
        self.W_o = nn.Linear(num_heads * d_head, d_model, bias=False)

        # 4.存放推理时吸收后的矩阵
        self.absorbed_W_q = None
        self.absorbed_W_o = None

        self.dropout = nn.Dropout(dropout_p)


    def forward_training(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        训练阶段没有吸收的前向传播函数(适合理解完整的MLA逻辑与数据流)
        参数:
            x: [Batch_size,Seq_len,d_model],上一层的输出
            is_causal: 是否开启对注意力分数的因果掩码
        返回:
            out: [B,S,d_model]注意力层的输出结果
        """
        B,S = x.size(0),x.size(1)
        # 1. 计算query latent,q_content,q_rope并拼接成为完整的query
        # 压缩：计算压缩后的q_latent
        q_latent = self.W_dq(x) # [Batch_size,Seq_len,q_lora_rank]
        # 还原与分头：将压缩后的q_lantent还原并将不同的注意力头分开
        # [B,S,num_heads * d_head] -> [B,S,num_heads,d_head]
        q_content = self.W_uq(q_latent).view(B,S,self.num_heads,-1) 
        # 位置：计算q_rope
        # [B,S,num_heads * d_rope] -> [B,S,num_heads,d_rope]
        q_rope = self.W_qr(q_latent).view(B,S,self.num_heads,-1)
        # 应用旋转位置编码函数
        q_rope = apply_rope(q_rope)
        # 拼接content与rope:将q_content和q_rope在最后一个维度上拼接成为完整的query
        # [B,S,num_heads,d_head+d_rope]
        query = torch.cat([q_content,q_rope], dim=-1)

        # 2. 计算kv-joint-latent
        kv_latent = self.W_dkv(x) # [B,S,kv_lora_rank]
        # 计算k_content并分头:[B,S,num_heads * d_head] -> [B,S,num_heads,d_head]
        k_content = self.W_uk(kv_latent).view(B,S,self.num_heads,-1) 
        # 计算key单头位置信息
        k_rope = self.W_kr(x).unsqueeze(2) # [B,S,1,d_rope]
        # 对k_rope应用RoPE
        k_rope = apply_rope(k_rope)
        # 由于所有key的头共享同一个k_rope因此我们需要将k_rope在所有头上扩展
        # [B,S,num_heads,d_rope]
        k_rope_expanded = k_rope.expand(-1,-1,self.num_heads,-1)
        # 然后将k_content和k_rope在最后一个维度上拼接
        # [B,S,num_heads,d_head + d_rope]
        key = torch.cat([k_content,k_rope_expanded],dim=-1)
        # 计算v并分头:[B,S,num_heads * d_head] -> [B,S,num_heads,d_head]
        v = self.W_uv(kv_latent).view(B,S,self.num_heads,-1) 
        
        # 3.计算注意力结果
        # 3.1计算注意力分数
        q = query.transpose(1,2) # [B,num_heads,S,d_head+d_rope]
        k = key.transpose(1,2) # [B,num_heads,S,d_head+d_rope]
        v = v.transpose(1,2)  # [B,num_heads,S,d_head]

        scale = 1.0 / math.sqrt(self.d_head + self.d_rope)
        scores = torch.matmul(q, k.transpose(-2,-1)) * scale # [B,num_heads,S,S]

        # 3.2实现因果掩蔽
        if is_causal:
            causal_mask = torch.tril(torch.ones(1,1,S,S,device=x.device)).bool()
            # ~按位取反符号只能用于bool或integer类型张量,因此将causal_mask转换为bool张量
            scores = scores.masked_fill(~causal_mask,float('-inf'))

        # 3.3 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        # 对注意力权重正则化
        attention_weights = self.dropout(attention_weights) # [B,num_heads,S,S]
        # 3.4计算加权和
        out = attention_weights @ v # [B,num_heads,S,d_head]
        out = out.transpose(1,2).contiguous().view(B,S,-1) # [B,S,num_heads * d_head]
        return self.W_o(out) # [B,S,d_model]

    
    def absorb_weight(self):
        # 1. W_uk和W_uq被吸收进W_q
        # W_q = W_uq^ @ W_uk: [num_heads,q_lora_rank,d_head] @ [num_heads,d_head,kv_lora_rank]
        # [num_heads, q_lora_rank, kv_lora_rank]
        # W_uk_w: [num_heads * d_head, kv_loran_rank] -> [num_heads, d_head, kv_lora_rank]
        W_uk_w = self.W_uk.weight
        W_uk_w = W_uk_w.view(self.num_heads, self.d_head, -1)
        # W_uq_w: [num_heads * d_head, q_lora_rank] -> [num_heads, d_head, q_lora_rank]
        W_uq_w = self.W_uq.weight
        W_uq_w = W_uq_w.view(self.num_heads, self.d_head, -1)
        # 数学推导: 对于每个注意力头i，scores_i为
        # q_content @ k_content^: (q_latent @ W_uq^) @ (kv_latent @ W_uk^)^
        # q_latent @ (W_uq^ @ W_uk) @ kv_latent^
        # 因此吸收后的矩阵为W_q = W_uq^ @ W_uk
        # [num_heads,q_lora_rank,d_head] @ [num_heads,d_head,kv_lora_rank]
        # -> [num_heads,q_lora_rank,kv_lora_rank]
        # 注意：这里我们使用 bmm 批量矩阵乘法对每个头进行吸收
        self.absorbed_W_q = torch.bmm(W_uq_w.transpose(1,2), W_uk_w)

        # 2. W_uv和W_o被吸收进new W_o
        W_uv_w = self.W_uv.weight # [num_heads * d_head, kv_lora_rank]
        # [num_heads * d_head, kv_lora_rank] -> [num_heads,d_head,kv_lora_rank]
        W_uv_w = W_uv_w.view(self.num_heads, self.d_head, -1)
        W_o_w = self.W_o.weight # [d_model, num_heads * d_head]  
        # -> [d_model, num_heads, d_head]
        W_o_w = W_o_w.view(self.d_model, self.num_heads, -1) 
        # -> [num_heads, d_head, d_model]
        W_o_transpose = W_o_w.permute(1,2,0)
        # [num_heads, kv_lora_rank, d_head] @ [num_heads, d_head, d_model]
        # -> [num_heads, kv_lora_rank, d_model]
        self.absorbed_W_o = torch.bmm(W_uv_w.transpose(1,2), W_o_transpose)

    def forward_infer(self, x: torch.Tensor, 
                    cache_c_kv: torch.Tensor=None,
                    cache_k_rope: torch.Tensor=None) \
                    -> Tuple[torch.Tensor,Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MLA推理时的前向传播函数,利用了被吸收后的矩阵,不显式恢复key,value
        参数:
            x: [B,1,d_model],模型上一层的输出隐藏张量
            cache_c_kv: [B,L,kv_lora_rank],前L个被压缩后的kv cache
            cache_k_rope: [B,L,d_rope],前L个key每个头的位置向量
        返回:
            out: [B,1,d_model],注意力层的输出结果
        """
        b,s,_ = x.shape
        assert s == 1,'推理阶段只允许一个token一个token生成!'
        assert self.absorbed_W_q is not None and self.absorbed_W_o is not None, \
            '权重未被吸收，请先调用 absorb_weight()'

        # 1. 计算c_q,q_rope
        # c_q: [b,1,q_lora_rank]
        c_q = self.W_dq(x)
        # q_rope分头: [b,1,num_heads * d_rope] -> [b,1,num_heads,d_rope]
        q_rope = self.W_qr(c_q).view(b,1,self.num_heads,-1)
        q_rope = apply_rope(q_rope) # [b,1,num_heads,d_rope]

        # 2. 计算当前token的c_kv以及k_rope,并存入缓存!
        # [b,1,kv_lora_rank]
        current_c_kv = self.W_dkv(x)
        current_k_rope = self.W_kr(x).unsqueeze(2) # [b,1,1,d_rope]
        current_k_rope = apply_rope(current_k_rope)

        if cache_c_kv is not None:
            # 在序列长度维度上拼接
            cache_c_kv = torch.cat([cache_c_kv,current_c_kv],dim=1)
            cache_k_rope = torch.cat([cache_k_rope,current_k_rope],dim=1)
        else:
            cache_c_kv, cache_k_rope = current_c_kv, current_k_rope
        
        # ========= 使用吸收后的权重进行注意力计算 =================
        # 3. 计算score_content(直接在latent空间内算)
        # c_q: [b,1,q_lora_rank],absorbed_W_q: [num_heads,q_lora_rank,kv_lora_rank]
        # 相当于用c_q乘上被吸收后的W_q得到特制的q_latent: [b,num_heads,1,kv_lora_rank]
        # 函数表示相当于用
        q_latent = torch.einsum('b s q , h q k -> b h s k',c_q, self.absorbed_W_q)
        # 和缓存中的c_kv做点积
        # c_kv: [b,l,kv_lora_rank] -> [b,kv_lora_rank,l] -> [b,1,kv_lora_rank,l] -> \
        # 在矩阵乘法时将c_kv的第二维度自动广播到了num_heads[b,num_heads,kv_lora_rank,l]
        # [b,num_heads,1,kv_lora_rank] @ [b,num_heads,kv_lora_rank,l] -> [b,num_heads,1,l]
        # 最后得到了每个注意力头的该query对l个token的内容注意力分数
        score_content = torch.matmul(q_latent, cache_c_kv.transpose(-2,-1).unsqueeze(1))

        # 4. 计算score_rope部分
        # q_rope @ k_rope
        # q_rope: [b,1,num_heads,d_rope],k_rope: [b,l,1,d_rope]
        score_rope = torch.einsum('b s h r , b l q r -> b h s l', q_rope,cache_k_rope)

        # 总得分为内容部分的注意力分数 + rope部分的注意力分数
        scale = 1.0 / math.sqrt(self.d_head + self.d_rope)
        atten_scores = (score_content + score_rope) * scale
        atten_weights = F.softmax(atten_scores,dim=-1) # [b,num_heads,1,l]

        # 直接用得到的注意力权重,去计算c_kv的加权和
        # c_kv: [b,l,kv_lora_rank] -> [b,1,l,kv_lora_rank] 
        # -> 广播到 [b,num_heads,l,kv_lora_rank]
        # [b,num_heads,1,l] @ [b,1,l,kv_lora_rank] -> [b,num_heads,1,kv_lora_rank]
        attended_c_kv = torch.matmul(atten_weights, cache_c_kv.unsqueeze(1))

        # 直接将加权和后的c_kv与吸收后的W_o做矩阵乘法
        # 用einsum直接将每一个头的投影结果求和
        # attended_c_kv: [b,num_heads,1,kv_lora_rank]
        # W_o: [num_heads,kv_lora_rank,d_model]
        # out: [b,1,d_model]
        out = torch.einsum('b h s k , h k d -> b s d', attended_c_kv, self.absorbed_W_o)

        return out, cache_c_kv, cache_k_rope

# ============================================================
# 冒烟测试
# ============================================================
if __name__ == '__main__':
    print('========== MLA（Multi-head Latent Attention）冒烟测试 ==========')
    B, S, d_model = 2, 8, 512
    num_heads, d_head, d_rope = 8, 64, 64
    q_lora_rank, kv_lora_rank = 256, 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mla = MultiHeadLatentAttention(
        d_model=d_model, num_heads=num_heads, d_head=d_head, d_rope=d_rope,
        q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank, dropout_p=0.0
    ).to(device)

    x = torch.randn(B, S, d_model, device=device)

    print('[*] Test 1: 训练模式前向传播...')
    out_train = mla.forward_training(x, is_causal=True)
    assert out_train.shape == (B, S, d_model), f'形状错误: {out_train.shape}'
    assert not torch.isnan(out_train).any(), '输出包含 NaN!'
    print(f'   输出形状: {out_train.shape} OK')

    print('[*] Test 2: 权重吸收...')
    mla.absorb_weight()
    assert mla.absorbed_W_q is not None
    assert mla.absorbed_W_o is not None
    print(f'   absorbed_W_q 形状: {mla.absorbed_W_q.shape}')
    print(f'   absorbed_W_o 形状: {mla.absorbed_W_o.shape} OK')

    print('[*] Test 3: 推理模式前向传播（单 token decode）...')
    x_decode = torch.randn(B, 1, d_model, device=device)
    cache_c_kv = torch.randn(B, S, kv_lora_rank, device=device)
    cache_k_rope = torch.randn(B, S, 1, d_rope, device=device)

    out_infer, new_c_kv, new_k_rope = mla.forward_infer(
        x_decode, cache_c_kv=cache_c_kv, cache_k_rope=cache_k_rope
    )
    assert out_infer.shape == (B, 1, d_model), f'形状错误: {out_infer.shape}'
    assert new_c_kv.shape == (B, S + 1, kv_lora_rank)
    assert new_k_rope.shape == (B, S + 1, 1, d_rope)
    assert not torch.isnan(out_infer).any(), '输出包含 NaN!'
    print(f'   输出形状: {out_infer.shape}')
    print(f'   更新后的 KV Cache 形状: {new_c_kv.shape} OK')

    print('All MLA tests passed!')


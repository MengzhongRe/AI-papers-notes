# 用Pytorch手动实现带状态的GQA(GropuedQueryAttention),预分配缓冲池，兼容大模型预训练、推理
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StatefulGQA(nn.Module):
    def __init__(self,d_model: int, num_q_heads: int, num_kv_heads: int, max_batch_size: int, max_seq_len: int, dropout_p: float):
        super().__init__()
        assert d_model % num_q_heads == 0,' ❌ d_model must be divisible by num_q_heads!'
        assert num_q_heads % num_kv_heads == 0,' ❌ num_q_heads must be divisible by num_kv_heads!'

        # 1.初始化Python变量
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        
        # 2.初始化模型权重
        self.Wq = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p)

        # 3.预分配静态缓存(Static Cache)
        # 用nn.Module的第三种模型类属性Buffer预分配，其中加上persistent=False可以使得其不保存在model.state_dict()中
        self.register_buffer('k_cache', torch.zeros(max_batch_size,num_kv_heads,max_seq_len,self.head_dim), persistent=False)
        self.register_buffer('v_cache', torch.zeros(max_batch_size,num_kv_heads,max_seq_len,self.head_dim), persistent=False)
    

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        适用于推理阶段的前向传播函数
        参数:
            x:[Batch,N_curr,d_model],上一层的输出隐藏层状态
            start_pos:x输入在整个推理序列中的位置
                Prefill阶段: start_pos == 0
                Decode阶段: start_pos > 0
        返回：
            [Batch,N_curr,d_model]GQA层的输出状态
        """
        B,N_curr,_ = x.size()
        # ---------- 1. 将输入x送入投影层得到当前的q,k,v -------------
        q = self.Wq(x).view(B,N_curr,self.num_q_heads,self.head_dim).transpose(1,2)
        k = self.Wk(x).view(B,N_curr,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = self.Wv(x).view(B,N_curr,self.num_kv_heads,self.head_dim).transpose(1,2)

        # ------------ 2.将新的k,v通过缓冲池切片存入到相应的位置 -----------------------
        self.k_cache[:B, :, start_pos : start_pos + N_curr, :] = k
        self.v_cache[:B, :, start_pos : start_pos + N_curr, :] = v

        # ------------- 3.提取历史总kv cache  ---------------------------------
        # 如果是Decode阶段,即start_pos > 0,则从Cache中切片提取截止到当前总的kv cache
        if start_pos > 0:
            k = self.k_cache[:B, :, : start_pos + N_curr, :]
            v = self.v_cache[:B, :, : start_pos + N_curr, :]

        # 记录当前k的总长度
        N_kv = k.size(2)
        
        # ------------- 4. kv cache 维度扩展 -----------------------------
        # 将kv caceh的维度从num_kv_heads扩展到num_q_heads，才能进行后续的点积计算
        k_expanded = k.unsqueeze(2).expand(B,self.num_kv_heads,self.num_queries_per_kv,
                        N_kv,-1).contiguous().view(B,self.num_q_heads,N_kv,-1)
        v_expanded = v.unsqueeze(2).expand(B,self.num_kv_heads,self.num_queries_per_kv,
                        N_kv,-1).contiguous().view(B,self.num_q_heads,N_kv,-1)

        # ------------ 5.做点积计算 ----------------------------------------
        # 前置准备工作做完，进行真正的注意力分数计算
        scores = torch.matmul(q,k_expanded.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # Causal Mask 因果掩码
        # 仅当模型处于Prefill阶段即N_curr > 1时才进行因果掩码
        # 这里我们通过torch.tril配合torch.ones生成下三角矩阵
        # 然后通过masked_fill将上三角的注意力分数全部设置为负无穷大
        # causal_mask: [N_curr,N_kv]
        if N_curr > 1:
            causal_mask = torch.tril(torch.ones(N_curr,N_kv, device=x.device)).bool()
            scores = scores.masked_fill(~causal_mask,float('-inf'))
        
        # 计算出归一化后的注意力权重
        attention_weights = torch.softmax(scores,dim=-1)
        # 将注意力权重送入Dropout随机失活
        attention_weights = self.dropout(attention_weights)
        # 计算加权和
        out = attention_weights @ v_expanded
        # 合并注意力头
        out = out.transpose(1,2).contiguous().view(B,N_curr,-1)
        
        return self.Wo(out)


# ==================================================================
# 测试用例(Test Case)
# =================================================================
def test_gqa():
    print(f' 🚀 开始进行手撕代码的GQA测试......')
    # --------- 1.全局变量配置 -----------------
    d_model = 4096
    num_q_heads = 32
    num_kv_heads = 8

    # ------- Test1: 断言正确性测试 ------------
    print('[*] Test 1: 断言正确性测试 --- ')


    # ------- Test 2: 输出形状正确性测试 -----------
    print('[*] Test 2: 输出形状正确性测试 --- ')

    # ------- Test 3: 状正确性测试 -----------
    print('[*] Test 3: 输出形状正确性测试 --- ')

    # ------- Test 2: 输出形状正确性测试 -----------
    print('[*] Test 3: 输出形状正确性测试 --- ')



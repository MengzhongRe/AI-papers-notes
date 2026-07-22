import torch
import math
import torch.nn as nn
# 手动实现多头掩蔽注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0,"d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model,bias=False)
        self.W_k = nn.Linear(d_model,d_model,bias=False)
        self.W_v = nn.Linear(d_model,d_model,bias=False)
        self.W_o = nn.Linear(d_model,d_model,bias=False)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self,q,k,v,mask=None):
        # q,k,v: [Batch_size, Seq_len, d_model]
        # 先根据输入获取批量大小
        batch_size = q.size(0)
        # 先把q输入W_q得到Q:[batch_size,seq_len,d_model] -> [batch_size,seq_len,d_model]
        # 再用view函数分头-> [batch_size,seq_len,num_heads,d_k]
        # 再用tranpose函数调换seq_len 和 num_heads维度实现注意力头的并行计算-> [batch_size,num_heads,seq_len,d_k]
        Q = self.W_q(q).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        # 计算点积:Q @ K^：[B,num_heads,L,head_dim] @ [B,num_heads,head_dim,L]
        # scores: [B,num_heads,L,L]
        # 计算点积后需要除以根号下head_dim，防止方差扩大到原来的d_K倍，后续经过softmax后概率值过于尖锐
        # 导致梯度消失，训练失败
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        # 实现掩蔽注意力:把mask 为0的token赋值为负无穷大，使得其经过softmax之后概率为0
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        # 将scores送入softmax得到注意力权重
        # attention_weights:[B,num_heads,L,L]
        attention_weights = torch.softmax(scores,dim=-1)
        # 对注意力权重正则化
        attention_weights = self.dropout(attention_weights)
        # 计算输出:attention_weights @ V:[B,num_heads,L,L] @ [B,num_heads,L,head_dim]
        # out:[B,num_heads,L,head_dim]
        out = torch.matmul(attention_weights,V)
        # 在将结果输入到线性层W_o之前，需要合并多个注意力头，因为W_o只接受d_model
        # 在合并注意力头之前，需要用contiguous函数开辟新的内存空间，以供view函数调整张量，否则会报错
        # out:[B,num_heads,L,head_dim] -> [B,L,num_heads,head_dim] -> [B,L,d_model]
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        # out: [B,L,d_model] -> [B,L,d_model]
        out = self.W_o(out)
        return out

# ================================================
# 测试用例
# ================================================
if __name__ == '__main__':
    print('--- 开始测试手撕的多头因果掩蔽注意力 ---')
    # 1.定义全局变量
    BATCH_SIZE = 2
    SEQ_LEN = 1024
    D_MODEL = 4096
    NUM_HEADS = 32
    HEAD_DIM = D_MODEL // NUM_HEADS
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2.初始化注意力类并构造输入数据
    mha = MultiHeadAttention(D_MODEL,NUM_HEADS,0.1).to(device)
    # 关键：测试前向传播必须开启eval模式，关闭Dropout,否则结果具有随机性，无法进行验证
    mha.eval()
    x1 = torch.randn((BATCH_SIZE,SEQ_LEN,D_MODEL),dtype=dtype,device=device)
    print(f'[*] 输入的张量形状为: {x1.shape}')

    # 构造对比输入(把x1复制一份，但篡改最后一个token的所有特征)
    # torch.clone()会直接对张量进行深拷贝，完全开辟一块新内存把原先的数据搬进去，修改x2不会改变x1!!!
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn((BATCH_SIZE,D_MODEL),dtype=dtype,device=device)

    # 3. 构造因果因果掩码矩阵mask:[SEQ_LEN,SEQ_LEN] -> [1,1,SEQ_LEN,SEQ_LEN]
    # 以便利用广播机制
    # 在类GPT的Decoder-only大模型中，其在训练和预填充(Prefill)阶段不可以看到后面的token
    # 我们利用torch.tril生成下三角矩阵，即下三角全为1，其他全为0，其中1为保持原状，0为掩蔽
    causal_mask = torch.tril(torch.ones(SEQ_LEN,SEQ_LEN)).unsqueeze(0).unsqueeze(0).to(device)
    print(f'[*] 生成的因果掩码矩阵形状为: {causal_mask.shape}')

    # 4.前向传播测试（推理阶段不需要计算梯度值，因此torch.no_grad()减少显存占用）
    with torch.no_grad():
        out1 = mha(x1,x1,x1,causal_mask)
        out2 = mha(x2,x2,x2,causal_mask)
        assert out1.shape == (BATCH_SIZE,SEQ_LEN,D_MODEL),f'形状错误,期望形状为: {(BATCH_SIZE,SEQ_LEN,D_MODEL)},\
            实际为: {out1.shape}'

        # 获取除最后个token外的所有输出
        out1_past = out1[:, :-1, :]
        out2_past = out2[:, :-1, :]

        # 获取最后一个token的输出
        out1_last = out1[:, -1, :]
        out2_last = out2[:, -1, :]

        diff_past = (out1_past - out2_past).abs().max().item()
        print(f'[*] 历史词汇的差异极值为: {diff_past:.8f}')
        assert diff_past < 1e-6,'严重Bug!现在的词看到了未来的词！'

        diff_last = (out1_last - out2_last).abs().max().item()
        print(f'[*] 新词汇的差异极值: {diff_last:.8f}')
        assert diff_last > 1e-4,'严重Bug：输入被篡改，但最新输出没有变化！'

        print(f'\n伟大的胜利！因果掩蔽机制实现成功！！！')

    

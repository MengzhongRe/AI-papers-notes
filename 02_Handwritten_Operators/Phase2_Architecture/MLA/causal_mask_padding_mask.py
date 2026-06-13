# 在大模型的训练过程中,由于每个batch长度不一,为了利用GPU的并行性,需要对长度较小的batch pad
# 到最长bacth的长度。在注意力层计算注意力权重时，显然每一个query不能对被pad的key进行检索关注
# 因此，我们需要实现padding mask。同时训练阶段还会有causal mask,二者如何实现?
#
# 相关文档:
#   mla_code_walkthrough.md — 第1-2节详细讨论了双重掩码的实现原理与 NaN 陷阱
#   multi_head_latent_attention.py — MLA 完整实现（训练版 + 推理版含权重吸收）
import torch
import torch.nn.functional as F
import math

def attention_with_combined_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        padding_mask: torch.Tensor=None, is_causal: bool = True):
    """
    参数:
        q: [b,h,l,dh],query张量
        k: [b,h,l,dh],key张量
        v: [b,h,l,dh],value张量
        padding_mask: [b,l], True表示对应batch的对应位置不是<pad> token,False则是<pad> token
        is_causal: 是否应用因果掩码
    返回:

    """ 
    b,h,l,dh = q.size()
    # 计算q @ k^原始注意力分数
    scale = 1.0 / math.sqrt(dh)
    scores = torch.matmul(q, k.transpose(-2,-1)) * scale

    mask = None
    # 构建因果掩码矩阵
    if is_causal:
        mask = torch.tril(torch.ones(l,l,device=q.device)).bool() # [l,l]

    # 如果传入padding_mask矩阵,扩展其维度
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [b,1,1l]
        mask = padding_mask if mask is None else (mask & padding_mask)
    
    if mask is not None:
        scores = scores.masked_fill(~mask,float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    out = attention_weights @ v
    return out, attention_weights

if __name__ == '__main__':
    B,H,L,D = 2,1,4,8
    # 三元表达式确定device
    device = 'cuda' if torch.cuda.is_available() else 'mps' \
        if torch.backends.mps.is_available() else 'cpu'
    size = (B,H,L,D)
    q = torch.randn(size,device=device)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    padding_mask = torch.tensor([[True,True,True,True],[True,True,False,False]]).to(device)

    out,attention_weights = attention_with_combined_mask(q,k,v,padding_mask,is_causal=True)
    # 序列1：无 padding，全部 True
    print("序列1的 Attention Weights Matrix:")
    print(attention_weights[0, 0])
    # 断言：序列1 权重矩阵应为下三角
    w1 = attention_weights[0, 0]
    assert w1.shape == (L, L), f"权重形状错误: {w1.shape}"
    assert torch.allclose(w1[0].sum(), torch.tensor(1.0), atol=1e-5), "每行权重和应为 1"
    assert torch.all(w1.triu(diagonal=1) == 0), "因果掩码未生效—上三角应全为 0"

    # 序列2：后2个位置被 padding，权重应只分配到前2个真实 token
    print("序列2的 Attention Weights Matrix:")
    w2 = attention_weights[1, 0]
    print(w2)
    assert w2[:, 2:].sum() == 0, "padding 位置应分配 0 权重"
    assert torch.allclose(w2[:2, :2].sum(dim=1), torch.tensor(1.0), atol=1e-5), \
        "真实 token 的权重和应为 1"
    assert not torch.isnan(w2).any(), "权重矩阵包含 NaN！"
    print("✅ 双重掩码 (causal + padding) 功能正常")


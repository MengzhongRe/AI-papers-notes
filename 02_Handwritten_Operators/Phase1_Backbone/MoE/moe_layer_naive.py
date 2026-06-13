# 手动实现MoE(Mixtral of Experts)稀疏专家8 * 7b网络
# 公式:$$ MoE(x) = \sum_{i=0}^{n-1} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x) $$
# 实例化8个实现过的SwiGLUFFN,手动实现Rouer路由机制（Wg），把一个token从dim维度映射到xWg,即num_experts
# 再通过Top K(一般为2)截断，把所有后面的输出logits赋值为负无穷，然后用softmax归一化，将x输入到激活的SwiGLU_i()中
# 根据概率值求和即可
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import sys
from pathlib import Path
from moe_utils import compute_aux_loss
try:
    from FFN.swiglu_ffn import SwiGLUFFN
except ImportError:
    current_dir = Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.append(str(parent_dir))
    from FFN.swiglu_ffn import SwiGLUFFN

# =============================================================
# 模块二:实现MoELayer(Naive Loop)类
# ==============================================================
class NaiveMoELayer(nn.Module):
    def __init__(self,dim: int, hidden_dim: int, num_experts: int = 8, topk: int = 2):
        super().__init__()
        # 实例化8个专家SwiGLU
        self.experts = nn.ModuleList([SwiGLUFFN(dim,hidden_dim) for _ in range(num_experts)])
        # 实例化路由层(bias-free)
        self.router = nn.Linear(dim,num_experts,bias=False)
        self.topk = topk

    def forward(self,x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """
        输入:
            x: [Batch_size,Seq_len,dim]
        输出
            (MoE(x),Aux_Loss): [Batch_size,Seq_len,dim],标亮aux_loss
        """
        # 获取x的各个分量，将x展平为:[B * L,dim]
        batch_size,seq_len,dim = x.shape
        x_flat = x.view(-1,dim) # [B * L,dim]
        total_tokens = x_flat.shape[0]    # B * L

        # ===========================================================
        # 1. 计算路由分数：输入 x 送入路由层计算 xWg，再用 topk 截断和 softmax 归一化
        # ===========================================================
        logits = self.router(x_flat)

        # 用 torch.topk 将输入的原始 logits 根据 topk 截断
        weights, indices = torch.topk(logits, self.topk, dim=-1)  # [total_tokens, topk]
        weights = F.softmax(weights, dim=-1).type_as(x)

        # 计算辅助损失
        aux_loss = compute_aux_loss(logits, indices, len(self.experts))

        # 2.初始化输出
        # 我们需要一个和x_flat形状一样的全零张量来累加结果
        final_output = torch.zeros_like(x_flat) # [total_tokens,dim]


        # ============================================================
        # 3.分发(Naive Loop)
        # ====================================================
        for i in range(self.topk):
            # 提取当前第i大的专家权重及其索引
            expert_idx = indices[:, i]   # [total_tokens] 每个token对应的第i大的专家索引
            expert_weight = weights[:, i]    # [total_tokens] 每个token对应的第i大的专家权重

            # 遍历八个专家
            for j,expert in enumerate(self.experts):
                # 创建一个索引，标记哪些token的专家索引等于当前专家j
                mask = (expert_idx == j) #布尔掩码[total_tokens]
                
                if mask.any():
                    # 提取属于该专家的Token并计算
                    # expert(x[mask])的结果形状是: [num_mask_tokens,dim]
                    expert_output = expert(x_flat[mask])

                    # 将专家输出乘以对应的权重，并累加到最终输出中
                    # expert_weight[mask]的形状是: [num_mask_tokens]
                    # 需要将其形状调整为[ num_mask_tokens,1]以便利用Pytorch的广播机制与expert_output相乘
                    final_output[mask] += expert_output * expert_weight[mask].unsqueeze(-1)

        return final_output.view(batch_size, seq_len, dim), aux_loss

# ===========================================================
# 模块三:测试MoELayer
# ==========================================================
def test_run():
    # 1.定义全局变量
    batch_size = 2
    seq_len = 128
    total_tokens = batch_size * seq_len
    dim = 4096
    hidden_dim = int(4 * dim)
    num_experts = 8
    topk = 2
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.实例化MoELayer
    moe_layer = NaiveMoELayer(dim,hidden_dim,num_experts,topk).to(device)
    # ==================================================================
    # 测试一:参数注册校验
    # ==================================================================
    # 确保所有专家和路由层的参数都被正确注册到模型中
    params = list(moe_layer.parameters())
    print(f'参数张量数量：{len(params)} (期望17个)')
    single_expert_params = sum(p.numel() for p in moe_layer.experts[0].parameters())
    print(f'单个专家模型参数量: {single_expert_params / 1e6:.4f} Million')
    print(f'单个专家模型参数显存占用: {4 * single_expert_params / 1024**3:.2f} GB')
    params_total = sum(p.numel() for p in params)
    print(f'模型总参数量: {params_total/1e9:.4f} Billion')
    print(f'模型参数显存占用: {4 * params_total / 1024**3:.2f} GB')


    # =====================================================================
    # 测试二：形状与梯度流校验
    # =========================================================================
    # 3.创建输入张量:为了测试能否把梯度回传到输入x，我们需要设置x.reguires_grad = True
    x = torch.randn((batch_size,seq_len,dim),dtype=dtype,device=device,requires_grad=True)
    # 4.前向传播
    output,aux_loss = moe_layer(x)
    # 伪造损失值
    loss = output.pow(2).mean() + 0.01 * aux_loss
    loss.backward()    
    # 打印输出形状和参数量
    print(f'输入x的形状是: {x.shape}')
    print(f'输出的形状是: {output.shape}')
    print(f'负载均衡aux_loss损失是: {aux_loss:.4f}')
    assert x.grad is not None,'❌ 梯度无法回传到输入端'
    assert moe_layer.router.weight.grad is not None,'❌ 梯度无法回传到路由层'
    assert moe_layer.experts[0].w13.weight.grad is not None,'❌ 梯度无法回传到专家层'

    # =======================================================
    # 测试三：Naive 版本推理耗时
    # ======================================================
    # 第一个torch.cuda.synchronize用于清空之前GPU执行的其他任务，避免影响后续的计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):
        _ = moe_layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / 10
    print(f'Naive版本推理平均耗时为: {avg_time * 1e3:.4f} ms')

    # ========================================================
    # 测试四：稀疏性激活验证
    # =======================================================
    # 我们需要计算每个专家接到的活，也就是激活的tokens数量
    logits = moe_layer.router(x.view(-1,dim))
    weights,indices = torch.topk(logits,topk,dim=-1)

    expert_counts = torch.zeros(num_experts,device=device)

    for i in range(topk):
        for j in range(num_experts):
            count = (indices[:, i] == j).sum()
            expert_counts[j] += count
    total_expert_totkens = expert_counts.sum().item()
    assert total_expert_totkens == total_tokens * topk,'❌ 专家激活量过高，请重试！'
    print(f'总专家激活量为: {total_expert_totkens},理论稠密激活量为: {total_tokens * num_experts}')
    print(f'每个专家的激活数量为: {expert_counts.tolist()}')



if __name__ == '__main__':
    test_run()

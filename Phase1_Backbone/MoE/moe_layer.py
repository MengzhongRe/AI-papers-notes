# 手撕向量化MoE(Mixtral of Experts)稀疏专家网络

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import sys

from moe_utils import compute_aux_loss

# 导入SwiGLUFFN (仅在必要时修改sys.path)
try:
    from FFN.swiglu_ffn import SwiGLUFFN
except ImportError:
    cur_dir = Path(__file__).resolve()
    parent_dir = cur_dir.parent.parent
    sys.path.append(str(parent_dir))
    from FFN.swiglu_ffn import SwiGLUFFN

# ========================================================
# 模块一:实现向量化MoE(8 * 7B)类
# =========================================================
class VectorizedMoELayer(nn.Module):
    def __init__(self,dim: int, hidden_dim: int, num_experts: int = 8, topk: int = 2):
        """
        参数：
            dim: 模型的维度
            hidden_dim: 每一个SwiGLU专家的隐藏层维度
            num_experts: 总专家数量
            topk: 每个token激活的专家数量
        """
        super().__init__()
        self.experts = nn.ModuleList([SwiGLUFFN(dim,hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim,num_experts,bias=False)
        self.topk = topk
    
    def forward(self,x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """
        参数:
            x: 输入张量:[Batch_size,Seq_len,dim],一般为注意力层的输出
        返回:
            (output,aux_loss): 一个MoE(x),一个负载均衡损失
        """
        # 1.获取输入张量形状
        batch_size,seq_len,dim = x.shape
        # 将张量在tokens维度展平，方便后续输入专家
        x_flat = x.view(-1,dim) # [num_tokens,dim]
        num_tokens = x_flat.shape[0] # 获取总输入tokens数

        # 2.将x_flat送入路由层得到原始logits值
        logits = self.router(x_flat).float()   # logits:[num_tokens,num_experts]

        # 计算真实的topk
        # torch.topk返回(weights,indices)元组，其中weights是前k大的logits值
        # indices是前k大的索引
        weights,indices = torch.topk(logits,self.topk,dim=-1)
        weights = F.softmax(weights,dim=-1).type_as(x)

        # 3. 计算 aux_loss 负载均衡损失
        aux_loss = compute_aux_loss(logits, indices, len(self.experts))

        # 准备输出画布
        final_out = torch.zeros_like(x_flat)

        # 开始num_experts的循环
        for i in range(len(self.experts)):
            # 用torch.where把选择了i专家的tokens聚合
            # torch.where输入一个布尔张量时，会返回(行，列)元素索引，表示为True的行，列号
            tokens_indices,k_levels = torch.where(indices == i)

            if tokens_indices.numel() > 0:
                # 1.聚合
                candidate_tokens = x_flat[tokens_indices]   # [selected_tokens,dim]
                # 将聚合后的tokens送入专家i计算输出
                expert_output = self.experts[i](candidate_tokens)    # [selected_tokens,dim]
                # 计算每个token对专家i的权重
                # [num_tokens,topk2] -> [selected_tokens] -> [selected_tokens,1]
                selected_weights = weights[tokens_indices,k_levels].unsqueeze(-1) 
                # 原地操作：使用mul_进行逐元素乘法原地操作覆盖expert_output 而非*可以节省内存
                expert_output.mul_(selected_weights)    # [slected_tokens,dim]
                # 累加写回：使用final_output.index_add_()原地在维度0上对tokens_indices对应的行累加结果
                final_out.index_add_(0,tokens_indices,expert_output)

        return final_out.view(batch_size,seq_len,-1), aux_loss

# =======================================================================================
# 模块二: 测试
# ======================================================================================
from moe_layer_naive import NaiveMoELayer
def test_run():
    # 1.定义全局变量
    batch_size = 2
    seq_len = 256
    total_tokens = batch_size * seq_len
    dim = 4096
    hidden_dim = int(4 * dim)
    test_time = 1000
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('--------------- 开始进行MoE代码测试 -----------------')
    print('[*] 正在实例化Naive MoE与工业级MoE...')
    # 实例化Naive版本和向量化版本的MoELayer
    moe_layer = VectorizedMoELayer(dim,hidden_dim).to(dtype).to(device)
    naive_moe_layer = NaiveMoELayer(dim,hidden_dim).to(dtype).to(device)

    moe_layer.eval()
    naive_moe_layer.eval()

    # 构造输入张量x
    x = torch.randn((batch_size,seq_len,dim),dtype=dtype,device=device)
    print(f'\n输入x的形状为: {x.shape}')
    # ==================================================
    # 测试一:逻辑等价性校验
    # =================================================
    # 把Naive版本的权重复制给向量化版本
    moe_layer.load_state_dict(naive_moe_layer.state_dict())

    # 将x输入给两个版本得到输出
    with torch.no_grad():
        out_naive,loss_naive = naive_moe_layer(x)
        out_moe,loss_moe = moe_layer(x)
    
    diff = (out_naive - out_moe).abs().max().item()
    print(f'\n[*] Naive和向量化版本最大绝对误差为: {diff:.4f}')
    assert torch.allclose(out_naive,out_moe,atol=1e-5),'❌ 两种版本输出差距过大！'

    # ================================================
    # 测试二：Unit Gradient Test 单元梯度测试
    # 测试梯度能否正确回传到路由和权重层
    # ================================================
    # 初始化优化器(SGD or Adam)
    optimizer = torch.optim.Adam(moe_layer.parameters(),lr=1e-3)
    # 构造输入并进行梯度跟踪
    x = torch.randn((batch_size,seq_len,dim),dtype=dtype,device=device,requires_grad=True)
    moe_layer.train()

    out_put1,aux_loss1 = moe_layer(x)
    # 伪造loss
    loss1 = out_put1.pow(2).mean() + 0.01 * aux_loss1
    # 清空遗留梯度
    optimizer.zero_grad()
    # 计算梯度
    loss1.backward()
    # 记录更新前的Router和Expert权重
    # 注意：一定要用clone()深拷贝一份，同时用detach()将拷贝从计算图分离，以释放缓存
    old_router_grad = moe_layer.router.weight.grad.clone().abs().max().item()
    old_expert_grad = moe_layer.experts[0].w13.weight.grad.clone().abs().max().item()
    print(f'\n[*] 负载均衡系数为0.01时Router 权重梯度最大值为: {old_router_grad:.4e}')
    print(f'[*] 负载均衡系数为0.01时Expert 权重梯度最大值为: {old_expert_grad:.4e}')

    # 清空梯度
    moe_layer.zero_grad()

    out_put2,aux_loss2 = moe_layer(x)
    # 伪造loss
    loss2 = out_put2.pow(2).mean() + 100.0 * aux_loss2
    # 计算梯度
    loss2.backward()
    # 记录更新前的Router和Expert权重
    # 注意：一定要用clone()深拷贝一份，同时用detach()将拷贝从计算图分离，以释放缓存
    new_router_grad = moe_layer.router.weight.grad.clone().abs().max().item()
    new_expert_grad = moe_layer.experts[0].w13.weight.grad.clone().abs().max().item()
    print(f'[*] 负载均衡系数为100.0 时Router 权重梯度最大值为: {new_router_grad:.4e}')
    print(f'[*] 负载均衡系数为100.0 时Expert 权重梯度最大值为: {new_expert_grad:.4e}')
    print(f'\n[⚖️ 梯度流向分析]')
    print(f'Router 梯度放大了: {new_router_grad / (old_router_grad + 1e-9):.2f} 倍')
    print(f'Expert 梯度放大了: {new_expert_grad / (old_expert_grad + 1e-9):.2f} 倍')


    # ================================================
    # 测试三：测试Naive和向量化推理速度
    # ===============================================
    # 预热
    moe_layer.eval()
    naive_moe_layer.eval()
    for _ in range(10):
        _ = naive_moe_layer(x)
        
    # 1.测试Naive推理耗时
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(test_time):
            _ = naive_moe_layer(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / test_time
        print(f'Naive 版本{test_time}次推理总耗时为：{end - start:.2f} s, 平均耗时为: {avg_time * 1e3:.4f} ms')

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(test_time):
            _ = moe_layer(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / test_time
        print(f'向量化 版本{test_time}次推理总耗时为：{end - start:.2f} s, 平均耗时为: {avg_time * 1e3:.4f} ms')

if __name__ == '__main__':
    test_run()



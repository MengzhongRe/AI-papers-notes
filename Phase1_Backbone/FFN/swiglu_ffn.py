# 手撕Transformer中前馈神经网络层(feed-forward network)的演进代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ==========================================================
# 代码一：VanillaFFN(Attention is all you need的源代码骨架)
# ===========================================================
# 传统FFN需要对当前层输入进行升维度（一般为4倍），然后再降维
class VanillaFFN(nn.Module):
    def __init__(self,d_model: int, hidden_dim: int):
        super().__init__()
        # 1. 升维矩阵（知识的Key）
        self.w1 = nn.Linear(d_model,hidden_dim)
        # 2. 激活函数
        self.act = nn.ReLU()
        # 3. 降维矩阵(知识的Value)
        self.w2 = nn.Linear(hidden_dim,d_model)
        # 总参数量为2 * d_model * hidden_dim
    
    def forward(self,x: torch.Tensor):
        # 先升维再激活最后降
        return self.w2(self.act(self.w1(x)))

# ======================================================
# 代码二：实现SwiGLU（Swish + GLU(门控线性单元)）
# ======================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network(大模型的记忆细胞)
    设计哲学:
        1. 三矩阵架构:Gate(w1),Up(w3),Down(w2)
        2. Bias-free:现代大模型为了训练稳定性和归一化对称性，全面抛弃了偏置
        3. 维度对齐： 确保隐藏层维度是256的倍数，优化GPU算子对齐
    """
    def __init__(self,dim: int, hidden_dim: int, multiple_of: int = 256, init_std: float = 0.02):
        super().__init__()
        # ---------步骤一:维度对齐 ----------
        # 传统FFN是4d.SwiGLU为了保持总参数量不变，将隐藏层维度降低为8/3d
        # 公式: hidden_dim = 2 * hidden_dim / 3
        hidden_dim = int(2 * hidden_dim / 3)
        # 确保hidden_dim是multiple_of的倍数
        # 向上取整到multiple_of的倍数，方便利用硬件加速
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # ---------------步骤二：定义线性投影层 --------------
        # 核心优化：将Gate(w1)和Up(w3)合并为一个大矩阵(dim,2 * hidden_dim)
        # 一次性算出Gate\Up的结果，减少一次矩阵乘法的调度开销
        self.w13 = nn.Linear(dim,2 * hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)

        self._resnet_parameters(init_std)
    
    def _resnet_parameters(self,init_std):
        # W13使用标准初始化
        nn.init.trunc_normal_(self.w13.weight,mean=0.0,std=init_std)
        # W2使用较小的初始化，稳定训练初期的梯度
        nn.init.trunc_normal_(self.w2.weight,mean=0.0,std=init_std / 1.414)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        实现SwiGLUFFN的前向传播
        FFN_SwiGLU(x) = (SiLU(xW1) * xW3)W2
        输入输出维度:
            x: [batch_size, seq_len, dim]
            输出: [batch_size, seq_len, dim]
        """
        # 1.一次性计算W1,W3的结果
        combined_proj = self.w13(x)

        # 2.使用torch.chunk(x,2,dim=-1)将combined-proj在hidden_dim维度切分为两半
        # 正好对应Gate和Up的结果
        gate_proj,up_proj = combined_proj.chunk(2,dim=-1)

        # 3.计算Gate的激活值和Up的乘积，并降低维度
        return self.w2(F.silu(gate_proj) * up_proj)

# ==========================================================
# 测试用例
# =========================================================
def run_test():
    print('开始SwiGLUFFN的测试...')
    # 1.定义全局变量
    dim = 4096
    batch_size = 2
    seq_len = 128
    hidden_dim = int(dim * 4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # 2. 实例化模型
    swiglu_ffn = SwiGLUFFN(dim,hidden_dim,multiple_of=256).to(device).to(dtype)
    swiglu_ffn.eval()  # 切换到评估模式，关闭dropout等训练特定行为
    # 3.构造输入向量x: [batch-size,seq_len,dim]
    x = torch.randn((batch_size,seq_len,dim),device=device,dtype=dtype)

    # 3.验证一:参数量守恒
    # SwiGLUFFN的总参数量应该与传统FFN基本相当
    params_total = sum(p.numel() for p in swiglu_ffn.parameters())
    expected_params = 8 * dim * dim
    ratio = params_total / expected_params
    print(f'模型总参数量为: {params_total/1e6:.2f} M | 与传统FFN 的参数比为: {ratio:.2f}')
    assert 0.95 < ratio < 1.5,'参数比例异常，请检查维度！'

    # 4.验证二:前向传播输出维度正确
    output = swiglu_ffn(x)
    assert output.shape == x.shape,f'错误❌ 输出张量形状期望为： {x.shape}, 实际为: {output.shape}'
    print(f'✅ 形状校验通过！ {output.shape}')

    # 5.验证三:反向传播测试
    try:
        loss = output.pow(2).mean()
        loss.backward()
        print('✅ 梯度回传测试通过！')
    except RuntimeError as e:
        print(f'❌ 梯度回传失败: {e}')
    
    # 6.验证四:逻辑等价性验证
    #验证合并矩阵w13得到的结果是否与w1,w3的结果相等
    with torch.no_grad():
        w1_weight,w3_weight = torch.chunk(swiglu_ffn.w13.weight,2,dim=0)
        gate_proj = F.linear(x,w1_weight)
        gate_proj = F.silu(gate_proj)
        up_proj = F.linear(x,w3_weight)
        ref_inter = gate_proj * up_proj
        ref_out = F.linear(ref_inter,swiglu_ffn.w2.weight)

        diff = (output - ref_out).abs().max().item()
        #{diff:.2e}表示返回diff的保留两位小数的科学计数法
        # 例子diff = 0.0000123456,则{diff:.2e} = 1.23e-5
        print(f'权重合并等价性误差: {diff:.2e}')
        assert diff < 1e-4,'❌ 权重合并与独立版本差异过大，复现失败！'
    
    # 7.验证五：torch.compile编译版本和与手撕版本对比
    import time

    compiled_swiglu = torch.compile(swiglu_ffn)
    # 预热
    for _ in range(100):
        _ = compiled_swiglu(x)

    # 计时测试
    start = time.time()
    for _ in range(10000):
        _ = compiled_swiglu(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'编译版本 10000 次耗时: {time.time() - start:.4f}s')

    start = time.time()
    for _ in range(10000):
        _ = swiglu_ffn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'原始版本 10000 次耗时: {time.time() - start:.4f}s')
    
    print('\n 恭喜！所有测试均通过！')

if __name__ == '__main__':
    run_test()






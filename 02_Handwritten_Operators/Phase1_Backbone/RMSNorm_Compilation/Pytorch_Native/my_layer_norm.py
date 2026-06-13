# 用Pytorch手动实现LayerNorm

import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self,dim: int,eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # \\gamma 缩放系数
        self.bias = nn.Parameter(torch.zeros(dim)) # \\beta 平移系数

    def forward(self,x):
        # 输入x: [Batch,Seq_len,dim]
        # 强制转换为float32,防止求和溢出
        x_fp32 = x.float()
        # 求均值mu = [B,L,1]
        mu = x_fp32.mean(dim=-1,keepdim=True)
        # 求差值
        diff = x_fp32 - mu
        # 求方差
        var = (diff ** 2).mean(dim=-1,keepdim=True)
        # 归一化并转回原类型
        # x_nomr: [B,L,dim]
        x_norm = (diff * torch.rsqrt(var + self.eps)).to(x.dtype)
        # 归一化后乘以缩放系数和偏移量返回[B,L,dim]
        return x_norm * self.weight + self.bias

# ===========================================
# 测试用例
# ==========================================
if __name__ == '__main__':

    # 1.定义全局变量
    Batch_size = 2
    Seq_len = 1024
    Dim = 4096
    dtype = torch.bfloat16

    # 2.构造输入数据
    x = torch.randn((Batch_size,Seq_len,Dim),dtype=dtype)

    # 3.实例化手撕的LayerNorm和官方torch.nn.LayerNorm类
    my_layernorm = MyLayerNorm(Dim,eps=1e-5).to(dtype)
    official_layernorm = nn.LayerNorm(Dim).to(dtype)
    # 强制两者的权重和偏置相等
    official_layernorm.weight.data = my_layernorm.weight.data.clone()
    official_layernorm.bias.data = my_layernorm.bias.data.clone()
    # 4.执行测试
    print(f'[*] 测试开始...')
    print(f'\t输入x的维度: {x.shape}')

    y_my = my_layernorm(x)
    y_official = official_layernorm(x)

    print(f'\t我的输出维度: {y_my.shape},输出类型为: {y_my.dtype}')
    print(f'\t官方的输出维度为: {y_official.shape},输出类型为: {y_official.dtype}')

    max_diff = (y_my - y_official).abs().max().item()
    print(f'[*] 最大绝对误差: {max_diff}')

    assert torch.allclose(y_my,y_official,rtol=1e-2,atol=1e-2),'输出结果不一致，实现失败！'
    print(f'[*] 实现成功！')



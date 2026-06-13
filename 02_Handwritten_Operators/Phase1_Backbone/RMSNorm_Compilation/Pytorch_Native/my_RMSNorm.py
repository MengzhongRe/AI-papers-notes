# ==========================================
# 手撕RMSNorm（root mean squre normalization）
# ==========================================
import torch
import torch.nn as nn

class MyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # \\gamma,缩放因子，每个维度独立训练一个缩放因子
        # RMSNorm只有缩放因子gamma，没有平移因子beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入维度: [B,L,dim]
        # 先将输入数据从原数据类型(fp16、bf16)转换为fp32，防止后续在计算平方和时数值溢出,NaN
        x_fp32 = x.float()
        # 直接求均方根(砍掉了减去均值的逻辑)
        rms_sq = torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps
        # 归一化并转换回原数据类型(fp16,bf16)
        x_norm = (x_fp32 * torch.rsqrt(rms_sq)).to(x.dtype)
        # 返回时乘上缩放系数
        return x_norm * self.weight
# ========================================
# 测试用例
# ========================================
if __name__ == '__main__':
    print('========== 手撕 RMSNorm 单元测试 ==========')
    # 1.定义全局变量
    Batch_size = 2
    Seq_len = 1024
    Dim = 4096
    dtype = torch.bfloat16
    # 2.实例化自己手写的RMSNorm类和pytorch官方的torch.nn.RMSNorm
    my_rmsnorm = MyRMSNorm(Dim,eps=1e-6).to(dtype)
    official_rmsnorm = nn.RMSNorm(Dim).to(dtype)
    # 强制对齐两者的权重初始值
    official_rmsnorm.weight.data = my_rmsnorm.weight.data.clone()

    # 3.构造输入数据(bf16)x: [B,L,D]
    x = torch.randn((Batch_size,Seq_len,Dim),dtype=dtype)

    # 4.输入数据得到输出
    y_my = my_rmsnorm(x)
    y_official = official_rmsnorm(x)
    print(f'[*] 我的输出维度: {y_my.shape}')
    print(f'[*] 官方实现的输出维度: {y_official.shape}')
    max_diff = (y_my - y_official).abs().max().item()
    print(f'[*] 最大绝对误差: {max_diff}')

    assert torch.allclose(y_my,y_official,rtol=1e-2,atol=1e-2),'实现错误，两者的输出结果并不相等！'
    print('RMSNorm 实现成功！')

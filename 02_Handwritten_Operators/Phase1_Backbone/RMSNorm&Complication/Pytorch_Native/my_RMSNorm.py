# ==========================================
# 手撕RMSNorm（root mean squre normalization）
# ==========================================
import torch
import torch.nn as nn
import torch._dynamo
# 把重编译上限提高到 64 次（默认是 8）
torch._dynamo.config.recompile_limit = 64

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
        rms = torch.sqrt(torch.mean(x_fp32 ** 2,dim=-1,keepdim=True) + self.eps)
        # 归一化并转换回原数据类型(fp16,bf16)
        x_norm =(x_fp32 / rms).to(x.dtype)
        # 返回时乘上缩放系数
        return x_norm * self.weight
# ========================================
# 测试用例
# ========================================
if __name__ == '__main__':
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
    print('RMSNorm实现成功！')




# ### 🔍 破案：为什么 `bf16` 容不下 `1e-5` 的误差？

# 在深度学习的浮点数世界里，不同的数据类型拥有不同的**机器极小值 (Machine Epsilon)**，也就是它能分辨的最小刻度：

# 1. **`float32` (FP32)**: 有 23 位尾数，精度极高，能分辨到约 **`1e-7`** 的差异。所以你用 `atol=1e-5` 去测 `fp32` 毫无问题。
# 2. **`float16` (FP16)**: 有 10 位尾数，能分辨到约 **`1e-3`**。
# 3. **`bfloat16` (BF16)**: 为了换取和 FP32 一样巨大的表示范围（防止溢出），它极其残忍地砍掉了尾数，**只保留了 7 位尾数！
# ** 它的物理极限精度只有约 **`1e-2` 到 `1e-3`**！

# **🔥 核心直觉：**
# 你要求两个 `bfloat16` 张量的误差小于 `0.00001` (`1e-5`)，就相当于**你拿着一把最小刻度只有 1厘米 的尺子，
# 要求它测出两根头发丝的厚度差异。** 即使 PyTorch 官方底层的 C++ 代码和你写的 Python 代码在数学逻辑上 100% 等价，
# 但因为底层计算顺序的微小不同（比如先除后乘、还是先乘后除），在 `bf16` 的低精度下，产生的截断误差会轻松达到 `0.01` 级别！

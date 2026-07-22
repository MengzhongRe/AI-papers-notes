import torch
import torch.nn as nn
import triton
import triton.language as tl

# ======================================================
#GPU Kernel：真正泡在成千上万个流式多处理器(SM)上的底层代码
# ======================================================

@triton.jit
def _rmsnorm_fwd_fused(
    X_ptr, Y_ptr, W_ptr,        # [指针] 输入X,输出Y,权重\gamma的显存起始地址
    stride_x_row, stride_y_row, # [步长] 内存中跳到下一行（下一个token）需要跨越多少个元素
    N, eps,                     # [标量] 特征维度大小和防溢出最小值
    BLOCK_SIZE: tl.constexpr   # [常量] 案板大小（必须是2的幂），tl.constexpr叫做"编译期常量"是Triton编译器在编译时
    # 就写死的数据，不可以再变更，其作用是向GPU申请固定大小的SRAM空间，否则GPU会罢工
):  
    # 获取当前线程ID
    # 假设输入有2048个token,GPU就会派出2048个Block
    # program_id(0)获取的是当前正在处理第几个token（也就是第几行）  
    row_idx = tl.program_id(axis = 0)

    # 2.找到属于我的那一堆食材
    # 起始地址 + 行号 * 每行跨度
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # 3.在SRAM上划定格子(生成0,1,2,...Block_size - 1的数组)
    cols = tl.arange(0,BLOCK_SIZE)
    # 防越界保护：如果N是4096，BLOCK_SIZE也是4096，安全
    # 但如果N是4000，我们要把多出来的格子盖住
    mask = cols < N

    # 4.从HBM取出这一行的4096个token元素，一口气捞到SRAM上
    # X_row_ptr + cols会生成一维向量，表示拿取的所有内存地址位置
    # mask意思就是比N大的那些内存地址不用取，直接将其值赋为other = 0.0
    x = tl.load(X_row_ptr + cols,mask,other=0.0)
    # 从HBM取出RMSNorm的缩放系数也就是gamma
    w = tl.load(W_ptr + cols,mask,other=0.0)
    # 5.将输入转换为tl.float32防止求平方和时溢出
    x_fp32 = x.to(tl.float32)

    # 6.核心数学运算：求平方和 -> 算均值 ->加eps -> 算根号
    variance = tl.sum(x_fp32 * x_fp32,axis=0) / N
    # rsqrt就是 1 /sqrt,由于乘法比除法块
    rsqrt = tl.math.rsqrt(variance + eps)
    # 7.归一化：乘上倒数，转换回原数据类型(bf16,fp16),乘上缩放系数
    y = (x_fp32 * rsqrt).to(x.dtype) * w
    # 8.把结果从SRAM搬运到HBM
    # mask是一个布尔数组，告诉GPU 如果是True的地方就放到内存指针对应的HBM位置，否则就不放
    # 如果不加mask，可能会导致内存越界，污染下一个token的数据甚至导致驱动重启
    tl.store(Y_row_ptr + cols,y,mask=mask)

# ==============================================================
# CPU Wrapper： Pytorch调度器，负责计算Grid 和 Block大小并调用 Kernel
# ===============================================================
def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps:float=1e-5) -> torch.Tensor:
    # 1.无论是[B,L,D]还是[B,L,K,D]，我们统统把前面压扁成 2D: [tokens总数,Dim]
    x_2d = x.view(-1,x.shape[-1])
    n_tokens,dim = x_2d.shape

    # 2.提前开辟好一个新的输出张量的显存位置
    y_2d = torch.empty_like(x_2d)

    # 3.确定案板大小，即Block_size,必须是大于等于dim的最小的2的幂
    # 比如dim=4096,则BlOCK_SIZE=4096,dim=4000,BLOCK_SIZE依然等于4096
    BLOCK_SIZE = triton.next_power_of_2(dim)

    # 4.确定并发数量(Grid):一共有n_tokens个token,那么就派发n_tokens个线程块
    # grid必须定义为1到3个元素的元组
    grid = (n_tokens,)

    # 5.发射指令，把输入张量、输出张量、缩放系数权重矩阵的内存指针、行步长及其他常量、块大小传给GPU Kernel
    _rmsnorm_fwd_fused[grid](
        x_2d, y_2d, weight,
        x_2d.stride(0), y_2d.stride(0),
        dim, eps,
        BLOCK_SIZE
    )
    # 6.将输出张量从压扁的2D变回原来的3D,4D 的形状
    return y_2d.view_as(x)

# =================================================
# 测试用例
# =================================================
if __name__ == '__main__':
    print('========== Triton RMSNorm 融合算子单元测试 ==========')
    # 全局变量配置
    B,L,D = 2, 1024, 4096 # 模仿LLaMa配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # 初始化输入张量和缩放权重
    x = torch.randn((B,L,D),dtype=dtype,device=device)
    weight = torch.ones(D,dtype=dtype,device=device)

    # 1.跑pytorch官方的RMSNorm
    official_rmsnorm = nn.RMSNorm(D).to(device).to(dtype)
    # 把其权重矩阵和我们的权重矩阵对齐
    official_rmsnorm.weight.data = weight.clone()
    y_official = official_rmsnorm(x)

    # 跑Triton版融合算子
    y_triton = triton_rmsnorm(x,weight,eps=1e-5)

    # 计算最大绝对误差
    max_diff = (y_triton - y_official).abs().max().item()
    print(f'[*] 你的Triton最大绝对误差为: {max_diff}')

    assert torch.allclose(y_triton,y_official,atol=1e-2,rtol=1e-2),'你的Triton算子实现错误'
    print(f'你成功实现了工业级的Triton RMSNorm 融合算子！！！')
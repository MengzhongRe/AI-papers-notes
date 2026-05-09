import torch
import triton
import triton.testing
import sys
from pathlib import Path

# 导入写好的triton RMSNrom 融合算子
from my_rmsnorm_triton import triton_rmsnorm
# ===============================
# 解决导入路径问题
# ================================
# 获取当前文件路径的上两级目录路径
ROOT = Path(__file__).parent.parent
# 将该路径加入到python搜索空间中
sys.path.append(str(ROOT))
from Pytorch_Native.my_RMSNorm import MyRMSNorm

# ==================================================
# 跑分配置： 我们要测试在不同特征维度下(dim)的性能表现
# ==================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['DIM'],    # 横坐标： 特征维度大小
        x_vals=[512 * i for i in range(2,17)],  # 测试从1024到8192的不同维度
        line_arg='provider', # 图例：我们要对比哪几种实现
        line_vals=['torch_native','torch_compile','triton','my_torch'],  # 四位参赛选手
        line_names=['Pytorch Native','Torch.compile','My triton','My Torch'],   # 图例名称
        styles=[('blue','-'), ('green','--'), ('red','-.'), ('yellow',':')], # 颜色和线型
        ylabel='GB/s',   #显存带宽吞吐量
        plot_name='RMSNorm-Performance', # 图表保存的名字
        args={'B': 4, 'L': 2048},   # 固定Batch_size和Seq_len
    )
)
def benchmark(B,L,DIM,provider):
    # 初始化输入数据，确保在GPU上，数据类型为bf16
    dtype = torch.bfloat16
    device = 'cuda'
    x = torch.randn((B,L,DIM),dtype=dtype,device=device)
    # 初始化缩放系数权重
    weight = torch.ones(DIM,dtype=dtype,device=device)
    
    # 工业级要求：在测速时要算quantiles,去除异常波动
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch_native':
        # 选手1：Pytoch官方原声C++实现
        ms,min_ms,max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.rms_norm(x,(DIM,), weight, 1e-5),
            quantiles=quantiles
        )
    elif provider == 'torch_compile':
        # 选手2:torch.compile编译的图融合算子
        my_rmsnorm = MyRMSNorm(dim=DIM,eps=1e-5).to(device).to(dtype)
        compiled_my_rmsnorm = torch.compile(my_rmsnorm)
        # 编译器需要热身，先跑一下触发编译
        for _ in range(2):
            compiled_my_rmsnorm(x)
        ms,min_ms,max_ms = triton.testing.do_bench(
            lambda: compiled_my_rmsnorm(x),
            quantiles = quantiles
        )
    elif provider == 'triton':
        # 选手3:自己手写的Triton Kernel
        ms,min_ms,max_ms = triton.testing.do_bench(
            lambda: triton_rmsnorm(x,weight,1e-5),
            quantiles=quantiles
        )
    elif provider == 'my_torch':
        my_rmsnorm = MyRMSNorm(DIM,1e-5).to(dtype).to(device)
        ms,min_ms,max_ms = triton.testing.do_bench(
            lambda: my_rmsnorm(x),
            quantiles=quantiles
        )
    
    # 算账：我们应该如何计算GB/s?
    # 我们读入了一次x,写入了一次y，两者维度一模一样，所以总访问量是2 * x的体积
    # 体积 = 元素个数(numel) * 每个元素的字节数(bf16是两个字节)
    # 我们定义一个匿名函数，函数接受一个ms的参数，代表完成一次测试所需要的毫秒时间，函数计算返回在改时间下
    # 处理2 * x体积量的操作的带宽速度，单位为GB/s
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    # x.numel()返回张量 总元素个数，例如 shape [1024,1024] → numel () = 1024×1024 = 1,048,576
    # x.element_size()作用：返回 每个元素占多少字节float32 → 4 字节，float16 → 2 字节，int8 → 1 字节
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    print('正在启动压榨RTX 5070ti 极限带宽的Bench mark...(这可能需要一两分钟，请耐心等待)')
    # 这行代码会在当前目录下生成图表文件和数据
    benchmark.run(print_data=True, save_path=f'{ROOT}/Triton/results_new')
    print('跑分结束！请查看当前目录下的图片文件.')
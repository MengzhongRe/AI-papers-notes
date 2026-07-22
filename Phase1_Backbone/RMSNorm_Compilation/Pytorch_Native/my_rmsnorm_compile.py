# ==================================
# Day5 截获torch.compile 的底层Triton代码
# ====================================
import torch
import os

from my_RMSNorm import MyRMSNorm

# 核心魔法咒语
# 该环境变量会强迫Pytorch将它在底层生成的Open AI Triton(类C++)代码直接打印到终端
os.environ['TORCH_LOGS'] = 'output_code'

if __name__ == '__main__':
    # 强制检查GPU（TRiton 编译强依赖于GPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        raise RuntimeError('编译任务必须在Cuda环境下运行')
    
    print(f'[Step 1] 初始化MyRMSNorm 并送入GPU...')
    Dim = 4096
    dtype = torch.bfloat16

    my_norm = MyRMSNorm(Dim,eps=1e-5).to(device).to(dtype)

    # 伪造一个LLama隐藏层状态张量
    x = torch.randn((2,1024,Dim),dtype=dtype,device=device)

    print(f'[Step 2] 调用torch.compile编译刚才实例化的模型')
    compiled_norm = torch.compile(my_norm)

    print(f'[Step 3] 第一次前向传播(触发编译引擎)')
    y = compiled_norm(x)

    print('\n编译截获完成！！')
# 手动实现反向神经元失活Inverted Dropout

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
# 模块一：底层算子实现（手写Forward和Backward）
# =====================================================================
class DropoutFunc(torch.autograd.Function):
    """
    工业级 Inverted Dropout 底层算子实现(手动推导梯度流)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        """
        前向传播
        参数：
            ctx: 上下文对象，用于存储反向传播需要的变量
            x: 输入张量
            p: 丢弃概率
            training: 当前是否是训练模式
        """
        # 1. 推理模式 或 丢弃概率为0
        if not training or p == 0.0:
            # save_for_backward 必须存入 Tensor 或 None
            ctx.save_for_backward(None)
            ctx.scale_factor = 1.0
            return x

        # 2. 生成Dropout Mask
        # torch.rand_like(x) 生成与x同形状的[0,1) 的均匀分布
        # 如果 rand > p,则判断为True(存活),否则为False(死亡)
        # mask是布尔变量，仅仅占1Byte/元素
        mask = torch.rand_like(x) > p

        # 3.计算Inverted Dropout 缩放因子
        scale_factor = 1.0 / (1.0 - p)

        # 4. Inverted Dropout核心逻辑
        # 【引擎机制】：x(FP32) * mask(Bool) * scale_factor(Float标量)
        # 触发 Type Promotion，mask 在寄存器里瞬间被当作 1.0/0.0 计算，不额外开辟显存！
        output = x * mask * scale_factor

        # 5. 保存上下文，供反向传播时使用
        ctx.save_for_backward(mask) # 把布尔掩码矩阵存进储物柜
        ctx.scale_factor = scale_factor   # 保存缩放概率

        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        反向传播: 手动指定梯度流向
        grad_output: 损失函数对当前层输出的梯度
        """
        # 从储物柜中取出保存的mask
        # ctx.saved_tensor返回的是元组，需要解包
        mask, = ctx.saved_tensors
        scale_factor = ctx.scale_factor

        # 1. 如果mask 为None,说明前向是推理模式或p=0,梯度无损原样返回
        if mask is None:
            grad_input = grad_output
        else:
            # 2. 梯度计算公式
            # 死亡的节点(mask=False) -> 梯度被强行乘 0 截断(没干活不配拿梯度)
            # 存活的节点(mask = True) -> 梯度原样传回，并放大的 scale_factor倍数
            grad_input = grad_output * mask * scale_factor
        
        # 必须严格对应 forward的输入参数数量(x, p , traning) 返回梯度
        # 因为p  和 training是普通的 Python 标量/布尔值，不可导，返回None
        return grad_input, None, None

# ======================================================================
# 模块二： 封装为nn.Moudle 提供给上层网络使用
# =====================================================================
class CustomDropout(nn.Module):
    """
    包装器: 将底层 Autograd 算子包装成标准的网络层
    """
    def __init__(self, p: float):
        super().__init__()
        assert 0.0 <= p < 1.0, 'p must be in [0,1)'
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.training 是 nn.Module 底层自带的属性
        # 当你调用 model.train() 时它是 True，调用 model.eval() 时它是 False
        return DropoutFunc.apply(x, self.p, self.training)

# ==========================================================
# 模块三: 测试用例
# ===========================================================
def test_dropout():
    print('🚀 开始进行 Inverted Dropout 算子测试...\n')
    torch.manual_seed(100) # 设置torch随机种子，以便进行复现

    x = torch.randn(4,10,requires_grad=True)
    p = 0.5

    # 实例化官方和我们手写的Dropout
    my_dropout = CustomDropout(p)
    official_dropout = nn.Dropout(p)

    # ===================================================
    # Test 1: 推理模式测试
    # ===================================================
    print('-> [Test 1] 推理模式一致性测试')
    # 先把dropout设置为推理模式eval()
    my_dropout.eval()
    my_eval_output = my_dropout(x)

    assert torch.allclose(my_eval_output,x,atol=1e-6),'❌ 推理模式下 输出必须严格等于输入！'
    print(' ✅ 推理模式测试通过！\n')

    # ===============================================
    # Test 2: 训练模式 缩放与mask测试
    # ===================================================
    print('-> [Test 2] 训练模式 缩放与mask 测试')
    my_dropout.train()
    my_train_output = my_dropout(x)

    zero_count = (my_train_output == 0.0).sum().item()
    zero_count_x = (x == 0.0).sum().item()
    total_count = x.numel()
    print(f' [*] 总元素为:{total_count}, 输入张量为0的数量为: {zero_count_x}, \
          被Drop的元素数为: {zero_count}, 理论期望值为: int({total_count * p})')
    assert zero_count > 0,'❌ 没有元素被 drop'

    # 检查存活的元素是否被正放缩
    survivor_mask = my_train_output != 0.0
    expected_scale = 1.0 / (1.0 - p)
    actual_scale = my_train_output[survivor_mask] / x[survivor_mask]
    expected_scale_tensor = torch.full_like(actual_scale,expected_scale)
    assert torch.allclose(actual_scale,expected_scale_tensor,atol=1e-4),'❌ 放缩系数不正确！'
    print(' ✅ 所有存活的元素的放缩系数都正确！\n')

    # ==================================================================
    # Test 3: 训练模式强对齐测试
    # ==================================================================
    # print('-> [Test 3] 训练模式强对齐测试,直接与官方Pytorch C++ 算子对比结果')
    # official_dropout.train()
    # # 【劫持pytorch随机数状态】
    # rng_state = torch.get_rng_state()
    # my_out_train = my_dropout(x)
    # # 恢复随机数状态，跑官方算子
    # torch.set_rng_state(rng_state)
    # official_out_train = official_dropout(x_official)

    # # 现在它们生成的随机 Mask 绝对是一模一样的！
    # assert torch.allclose(my_out_train,official_out_train,atol=1e-6),' ❌ 你的输出和官方nn.Dropout算子输出不一致!'
    # print("  ✅ 成功验证：在相同的随机数状态下，输出特征与官方 100% 对齐！\n")

    # ===================================================================
    # Test 4: 反向传播梯度测试
    # =================================================================
    print('-> [Test 4] 纯手工反向传播梯度流测试')
    # 伪造一个loss
    loss = my_train_output.sum()
    loss.backward()

    my_grad = x.grad

    # 死亡节点的梯度必须是0
    assert torch.all(my_grad[~survivor_mask] == 0.0),'❌ 死亡节点的梯度必须是0'
    assert torch.allclose(my_grad[survivor_mask],torch.full_like(my_grad[survivor_mask],expected_scale)),' ❌ 存活节点梯度不正确！'

    print("  ✅ 反向传播梯度实现符合微积分逻辑！\n")

    print('🎉🎉🎉 你手写的Inverted Dropout 算子全部测试通过！ ')

if __name__ == "__main__":
    test_dropout()


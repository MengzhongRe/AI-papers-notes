# 手撕基于减去行最大值的safe_softmax函数
import torch

# ======================================================
# 错误示范：不减去最大值的sfotmax
# =====================================================
def naive_softmax(x: torch.Tensor, dim: int = -1):
    """
    错误示范：由于不减去最大值可能会导致数值溢出!
    """
    # 1.取指数
    exp_x = torch.exp(x)
    # 2.求分母指数和
    sum_exp_x = torch.sum(exp_x,dim=dim,keepdim=True)
    # 3.归一化
    return exp_x / sum_exp_x

# =======================================================
# 核心算子:工业级safe_softmax
# =======================================================
def safe_softmax(x: torch.Tensor, dim: int = -1):
    """
    参数:
        x: 归一化前的张量，一般为原始logits值
        dim: 归一化的维度,一般为最后一个维度-1
    返回值：
        行归一化后的张量
    """
    # 1.x中的每个值必须减去改行的最大值,以防止后续取指数后数值爆炸，导致数值溢出NaN
    # torch.max会返回元组(values,indices)，我们只需要values,取第一个值即可
    # 避坑点：必须keepdim=True,保留张量最后一个维度，以便之后减去最大值时能够利用pytorch的广播机制
    x_max = torch.max(x,dim=dim,keepdim=True).values
    # 2.减去最大值:此时除了最大值为0，其余都为负数
    x_safe = x - x_max

    # 3.对每个分量取指数，此时所有值都在(0,1]之间
    exp_x = torch.exp(x_safe)

    # 4.求分母指数和，必须keepdim=True,以便后续利用广播机制与exp_x进行除法
    sum_exp_x = torch.sum(exp_x,dim=dim,keepdim=True)

    # 5.分子/分母归一化概率
    return exp_x / sum_exp_x

# ======================================================
# 测试用例
# ======================================================
if __name__ == '__main__':
    print('--- 启动Safe Softmax工业级验证 ---')
    BATCH_SIZE = 2
    NUM_HEADS = 32
    SEQ_LEN = 128

    # ===================================================
    # 测试一：与Pytorch官方底层C++实现的等价性验证(fp32)
    # ===================================================
    # 输入数据是attnetion层中的注意力分数scores: [B,num_heads,L,L]
    print('[测试一] 与Pytorch官方的等价性验证---')
    scores = torch.randn((BATCH_SIZE,NUM_HEADS,SEQ_LEN,SEQ_LEN),dtype=torch.float32)

    # 与官方torch.nn.functional.softmax对比
    out_official = torch.nn.functional.softmax(scores,dim=-1)
    out_safe = safe_softmax(scores,dim=-1)
    # 计算两者的最大差值
    max_diff = (out_official - out_safe).abs().max().item()
    print(f'[*] 两者的最大差值为: {max_diff:.8f}')
    assert max_diff < 1e-6,'实现错误，误差过大！'
    print(f'-> 测试一通过：等价性完美!')

    # =====================================================
    # 测试二：半精度防爆测试(FP1极值爆炸)
    # ====================================================
    print('\n[测试二] 半精度数值防爆测试')
    # 构造极端大值exp(15)
    # exp(15)对于fp32不算大，但是fp16会瞬间数值溢出
    extreme_scores = torch.tensor([[1.0,2.0,15.0,3.0,4.0]],dtype=torch.float16)
    print(f'-> 危险输入！ {extreme_scores.tolist()}')

    out_fp16_naive = naive_softmax(extreme_scores,dim=-1)
    print(f'-> Naive Softmax输出: {out_fp16_naive.tolist()}')

    out_fp16_safe = safe_softmax(extreme_scores,dim=-1)
    print(f'-> Safe Softmax输出: {out_fp16_safe.tolist()}')

    # 检测Naive 和 Safe是否存在NaN 
    assert torch.isnan(out_fp16_naive).any(),'测试设计失败：Naive居然没有炸'
    assert not torch.isnan(out_fp16_safe).any(),'测试失败，Safe居然炸了'
    print(f'-> 测试二通过：成功拦截了NaN灾难！')

    # ===================================================================
    # 测试三:概率总和验证
    # =================================================================
    print('\n[测试三] 概率总和验证')
    # 对于任意输入，沿dim=-1维度概率总和必定为1
    prob_sums = out_safe.sum(dim=-1)
    print(f'-> Safe Softmax概率总和为: {prob_sums.tolist()}')

    # fp16精度下有极小的误差
    assert torch.allclose(prob_sums,torch.ones_like(prob_sums),atol=1e-3),'失败，总和不为1'
    print('->. 测试三通过: 严格符合概率分布')

    print('\n伟大的胜利！全部实现成功！！！')




    

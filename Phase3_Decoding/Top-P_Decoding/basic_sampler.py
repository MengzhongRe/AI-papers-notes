# 手动实现大模型轻量级带温度缩放、Top-K 截断和 Top-P 采样的解码器（nn.Module 封装）
#
# 本文件将采样逻辑封装为 nn.Module 子类（BasicSampler），包含完整的：
#   贪婪特判 -> 温度缩放 -> Top-K 截断 -> Top-P 核采样 -> Softmax 归一化 -> 多项式采样
#   BasicSampler 可作为子模块嵌入模型的 forward 流程中直接调用。
#
# 与 generate_next_token.py 的关系：
#   generate_next_token.py 将同样的核心管线实现为独立函数 generate_next_token()，
#   适合面试场景直接手撕。两者核心管线完全一致（贪婪特判 -> 温度缩放 -> Top-K -> Top-P -> Softmax -> multinomial），
#   区别仅在于代码组织方式（类 vs 函数）。generate_next_token.py 的工业级边界保护更完善
#   （如 top_k 越界 clamp、维度自适应兼容等），本文件则更适合作为模型组件集成。

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 模块一：实现轻量级采样器类
# ==============================================================================
class BasicSampler(nn.Module):
    def __init__(self,temperature: float = 0.9, top_k: int = 50, top_p: float = 0.9):
        super().__init__()
        # 取temperature最小为1e-5,防止除0错误
        self.temperature = max(temperature,1e-5)
        self.top_k = top_k
        self.top_p = top_p

    def forward(self,logits: torch.Tensor) -> torch.Tensor:
        """
        参数:
            logits: LMHead的将隐状态映射到模型词表之后的原始输出分数[B,L,V]
        返回:
            next_token: 模型的下一个输出ID [B,1]
        """
        # 自回归阶段，模型是一个词一个词往外蹦的，我们只关心最后一个时间步的logit值
        # [B,L,V] -> [B,V]
        logits = logits[:,-1]
        # ===================================================================
        # 1.贪婪解码特判：如果温度值特别低，直接取最大值作为输出，避免复杂计算
        # ====================================================================
        if self.temperature < 1e-5:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # ===================================================================
        # 2.温度缩放
        # ===================================================================
        logits = logits / self.temperature

        # ====================================================================
        # 3.Top_k 斩断长尾词:用fill_把logits填全-inf,再用scatter_把
        # 想要的topk写上去
        # ====================================================================
        # 用torch.topk获取前k大的值和索引
        top_k_values,top_k_indices = torch.topk(logits,self.top_k,dim=-1)
        # 先用fill_把Logits重置为全-inf,再用scatter_把前k大的值按照原先的索引重新填上去
        logits.fill_(float('-inf')).scatter_(dim=-1,index=top_k_indices,src=top_k_values)

        # ==================================================================
        # 4.Top P 核采样
        # ==================================================================
        # 用torch.sort对logits降序排序，并记录好对应索引(后面要根据索引还原)
        sorted_logits,sorted_indices = torch.sort(logits,dim=-1,descending=True)

        # 计算排序好的softmax归一化概率
        sorted_probs = F.softmax(sorted_logits,dim=-1)

        # 计算累加和概率
        cumulative_probs = torch.cumsum(sorted_probs,dim=-1)

        # Mask: 创建掩码矩阵，标记那些累加概率和大于topp的索引
        sorted_indices_to_remove = cumulative_probs > self.top_p

        # Mask右移：如果用上面的那个掩码矩阵，则累加概率刚好超过topp的logit也会被掩蔽
        # 我们需要将mask的值向右移一位，同时永远保证第一个词不会被遮蔽
        # 在Python中的list切片操作会开品新的内存，但是pytorch的张量切片操作则不会
        # 其仅仅是原张量的视图，底层指向的是同一块内存，因此修改切片会修改原张量
        # 这里如果不对视图深拷贝，旧进行赋值操作会导致内存覆盖错误
        sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0] = False

        # 把不需要的词的排序之后的logits设为-inf
        sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))

        # 把修改后的Logits【还原会原本的词表顺序】
        # 使用torch.scatter_算子原地操作
        logits.scatter_(dim=-1,index=sorted_indices,src=sorted_logits)

        # ===============================================================
        # 5.终极审判: Softmax归一化 + 多样式采样
        # ===============================================================
        probs = F.softmax(logits,dim=-1)

        # 根据概率掷骰子，num_samples决定最后一个维度的抽取数量
        # next_token: [Batch,1]
        next_token = torch.multinomial(probs,num_samples=1)

        return next_token

# =============================================================================
# 冒烟测试
# ==============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('BasicSampler 轻量级采样器（nn.Module 封装）冒烟测试')
    print('=' * 50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    batch_size = 4
    vocab_size = 50257

    logits = torch.randn(batch_size, 16, vocab_size, dtype=dtype, device=device)
    logits_decode = torch.randn(batch_size, 1, vocab_size, dtype=dtype, device=device)

    # 测试1: 贪婪解码
    sampler = BasicSampler(temperature=0.0)
    token = sampler(logits)
    assert token.shape == (batch_size, 1), f'贪婪输出形状错误: {token.shape}'
    print(f'[PASS] 测试1 贪婪解码: {token.shape}')

    # 测试2: 标准采样
    sampler = BasicSampler(temperature=0.9, top_k=50, top_p=0.9)
    token = sampler(logits)
    assert token.shape == (batch_size, 1), f'标准采样输出形状错误: {token.shape}'
    print(f'[PASS] 测试2 标准采样: {token.shape}')

    # 测试3: 解码阶段(seq_len=1)
    token = sampler(logits_decode)
    assert token.shape == (batch_size, 1), f'解码阶段输出形状错误: {token.shape}'
    print(f'[PASS] 测试3 解码阶段: {token.shape}')

    print('全部 BasicSampler 测试通过！')

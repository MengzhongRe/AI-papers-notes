# 手动实现语言模型输出头和LogSumExp版本的交叉熵损失
#
# 本文件与同目录文件的关系：
#   README.md — LM Head 与 CE Loss 核心知识笔记（交叉熵 vs MSE、LogSumExp、浮点精度、torch.gather()、双栖 API、Weight Tying 等 8 个专题）
#   plan.md   — Day 20 实现计划：显存核弹分析、词表对齐（Tensor Cores）、分布式 LogSumExp（Vocab Parallelism）等扩展讨论
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# 模块一:LogSumExp数值稳定的交叉熵损失实现，避免logits过大导致的溢出问题
# ===========================================================================
class LMHead(nn.Module):
    def __init__(self,hidden_dim: int, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

        # 词表映射层：将特征维度映射到词表概率空间
        # vocab_size常pad到8或64的倍数，方便计算优化
        self.head = nn.Linear(hidden_dim,vocab_size,bias=False)

    def forward(self,hidden_states: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor,torch.Tensor]:
        """
        参数：
            hidden_states: [Batch_size,Seq_len,Hidden_dim]
            targets: [Batch_size,Seq_len],对应真实的Token ID
        返回:
            logits: [batch_size,Seq_len,vocab_size]
            loss: 标量，交叉熵损失
        """
        # 1.映射到词表空间
        logits = self.head(hidden_states) # [B,L,V]

        loss = None

        if targets is not None:
            # ===================================
            # 知识点1：自回归Shift逻辑（错位预测）
            # ===================================
            # 语言模型的训练目标是预测下一个token，因此需要将输入和标签错位对齐
            # 输入hidden_states的第i个位置对应标签targets的第i+1个位置
            shifted_logits = logits[..., :-1, :].contiguous() # [B,L-1,V]
            shifted_labels = targets[:, 1:].contiguous() # [B,L-1,V]

            # 将张量展平，方便计算loss
            flat_logits = shifted_logits.view(-1,self.vocab_size)
            flat_labels = shifted_labels.view(-1)

            # 过滤掉padding的token
            valid_mask = (flat_labels != self.ignore_index) # [M]
            valid_logits = flat_logits[valid_mask] # [M,V]
            valid_labels = flat_labels[valid_mask] # [M]

            # 如果全是-100，直接返回0梯度
            if valid_labels.numel() == 0:
                return logits, torch.tensor(0.0,device=logits.device,requires_grad=True)

            # ===================================
            # 知识点2：数值稳定的交叉熵计算(LogSumExp版本)
            # ===================================
            # 计算交叉熵损失，避免数值溢出
            # 先把valid_logits转换为fp32,防止后续10万词表累加导致精度下溢出
            valid_logits_fp32 = valid_logits.float()
            # 1.计算最大值
            # torch.max()会返回(values,indices),我们只需要values
            max_logits,_ = torch.max(valid_logits_fp32,dim=-1,keepdim=True) #[M,1]
            # 2.减去最大值，得到数值稳定的logits
            safe_logits = valid_logits_fp32 - max_logits # [M,V]
            # 3.提取目标词的true_logits
            # 用torch.gather()函数提取每个token真实的logit值
            # gather函数：沿着dim维度，根据index索引提取元素，unsqueeze(-1)把valid_labels从[M]变成[M,1]，方便gather操作
            # gather之后的返回值也是[M,1],需要squeeze(-1)把它变回[M]
            # 优化：直接在safe_logits上gather,这样后面计算Loss就不需要再加M了
            # 因为-（xc - M） = xc - M, 只要在safe_logits上gather就已经包含了减去M的操作了
            true_safe_logits = safe_logits.gather(dim=-1,index=valid_labels.unsqueeze(-1)).squeeze(-1) # [M]

            # 4.求指数
            exp_logits = torch.exp(safe_logits) # [M,V]
            # 5.求指数和并取对数
            log_sum_exp = torch.log(torch.sum(exp_logits,dim=-1)) # [M]

            # 6.计算交叉熵损失
            loss = (-true_safe_logits + log_sum_exp).mean() # 标量

        return logits,loss

# ======================================================================
# 模块二：测试用例
# ======================================================================
def test_lm_head():
    print('🚀 开始进行 LMHead & LogSumExp 极限抗压测试...\n')

    # 1.定义全局变量
    Bacth_size = 2
    Seq_len = 128
    Hidden_dim = 512
    V = 32000
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.实例化LMHead
    lm_head = LMHead(Hidden_dim,V).to(device).to(dtype)

    # 2.创建随机输入数据
    hidden_states = torch.randn((Bacth_size,Seq_len,Hidden_dim),dtype=dtype,device=device,requires_grad=True)
    print(f'[*] 输入隐藏层状态形状为: {hidden_states.shape}')
    targets = torch.randint(0,V,(Bacth_size,Seq_len),device=device)
    # 人工制造一些Padding
    targets[0,100:] = -100
    targets[1,120:] = -100


    # ====================================================================
    # 测试1 & 2: 前向数值对其 与 反向梯度校验
    # =====================================================================
    print('-> [Test 1 & 2] 数值与梯度对齐测试')
    # --- 我的模型 ---
    my_logits,my_loss = lm_head(hidden_states,targets)
    my_loss.backward()
    my_grad_weight = lm_head.head.weight.grad.clone()
    my_grad_input = hidden_states.grad.clone()

    # 清空梯度
    lm_head.zero_grad()
    hidden_states.grad.zero_()

    # ------ Pytorch 官方对照组-----
    gt_logits = lm_head.head(hidden_states)
    shift_logits = gt_logits[..., :-1, :].contiguous().view(-1,V)
    shift_labels = targets[..., 1:].contiguous().view(-1)
    # 计算官方的LogSumExp Loss
    gt_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=lm_head.ignore_index)
    gt_loss.backward()
    gt_grad_weight = lm_head.head.weight.grad.clone()
    gt_grad_input = hidden_states.grad.clone()

    # 校验loss
    loss_diff = (my_loss - gt_loss).abs().item()
    assert torch.allclose(my_loss.float(),gt_loss.float(),atol=1e-4),f'loss 校验失败！Diff: {loss_diff}'
    print(f' ✅️ 前向Loss 完全一致 (误差: {loss_diff:.2e})')

    # 校验梯度
    weight_grad_diff = (my_grad_weight - gt_grad_weight).abs().max().item()
    input_grad_diff = (my_grad_input - gt_grad_input).abs().max().item()
    assert torch.allclose(my_grad_weight,gt_grad_weight,atol=1e-4),'权重梯度校验失败！'
    assert torch.allclose(my_grad_input,gt_grad_input,atol=1e-4),'输入梯度校验失败！'
    print(f' ✅️ 反向梯度完全一致。其中权重误差：{weight_grad_diff:.2e},输入梯度误差为 {input_grad_diff:.2e}')

    # ====================================================
    # 测试3：极端数值稳定性测试
    # ==================================================
    print('-> [Test 3] 极端数值免疫测试')

    lm_head.zero_grad()
    # 强行给隐状态注入极端大值或小值(1000,-1000)
    nuclear_hidden = hidden_states.detach().clone()
    nuclear_hidden[0,0,0] = 1000.0
    nuclear_hidden[0,0,1] = -1000.0
    nuclear_hidden.requires_grad_(True)

    _,nuclear_loss = lm_head(nuclear_hidden,targets)

    assert not torch.isnan(nuclear_loss),'❌️ 发生NaN!'
    assert not torch.isinf(nuclear_loss),'❌️ 发生Inf'
    print(f' ✅️ 成功抵御核弹级数字异常。损失为: {nuclear_loss.item():.4f}\n')

    # ====================================================================
    # 测试 4： 全mask 边界测试
    # ===================================================================
    print('-> [Test 4] 全 Padding 边界测试')
    lm_head.zero_grad()
    # 用torch.full(size,number)构造一个全-100的标签
    empty_targets = torch.full((Bacth_size,Seq_len),-100,device=device)
    _,empty_loss = lm_head(hidden_states,empty_targets)
    # 尝试反向传播，看是否会引发requires_grad 报错
    empty_loss.backward()

    assert empty_loss.item() == 0.0,'全mask时loss应为0.0!'
    print(' ✅️ 成功处理全mask空标签，返回Loss 0.0且未触发引擎崩溃\n')

    # ========================================================================
    # 测试 5：推理模式测试
    # =======================================================================
    print('-> [Test 5] 推理模式测试')
    with torch.no_grad():
        # 推理时Loss应返回为None
        inf_logits,inf_loss = lm_head(hidden_states)

    assert inf_logits.shape == (Bacth_size,Seq_len,V),'Logits形状错误'
    assert inf_loss is None,'推理时Loss 应为None'
    print(' ✅️ 推理模式 API 验证通过 \n')

    print(' 恭喜!你的LMHead 通过了所有工业级极限测试！')


if __name__ == '__main__':
    test_lm_head()

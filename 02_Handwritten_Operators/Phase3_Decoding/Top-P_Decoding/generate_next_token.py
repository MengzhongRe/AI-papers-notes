# 大模型解码管线代码(Decoding Pipeline)
# 用Pytorch实现从 贪婪特判 -> 温度缩放 -> Top-K长尾词截断 -> Top-P核采样(Nuclear Sampling)
import torch
import torch.nn.functional as F

# =======================================================
# 模块一: 函数式解码策略
# ======================================================
def generate_next_token(logits: torch.Tensor, temperature: float= 0.9, \
            top_k: int=50, top_p: float=0.9) -> torch.Tensor:
    """
    实现大模型解码管线
    参数:
        logits: [Batch,L,vocab_size] 语言模型头输出的原始得分
        temperature: 控制温度缩放的系数,温度越小,词表分布越尖锐;温度越大,词表分布越平滑
        top_k: 斩断k之后的所有长尾词
        top_p: 斩断累加概率为top_p之后的长尾词(保留刚刚累加过p的token)
    返回:
        next_token: [Batch,1] 模型最终输出的词元ID
    """
    # =====================================================
    # 0. 维度判断
    # ====================================================
    if logits.dim() == 3:
        logits = logits[:, -1]
    elif logits.dim() == 2:
        pass
    else:
        raise ValueError(f'❌ logits必须是 2 或 3 维张量,但收到了{logits.dim()}维张量!')

    # 1. 贪婪特判(如果温度极低，直接选最大值，省去复杂计算)
    if temperature < 1e-5:
        # [Batch,1]
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. 温度缩放
    logits = logits / temperature

    # ========================================================
    # 3. Top-K长尾词截断
    # ========================================================
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # 找出logits中前k大的词(也就是第k大的值),torch.topk按照降序返回(值索引)结果
        # [Batch,k]
        topk_values, _ = torch.topk(logits,top_k,dim=-1)
        kth_values = topk_values[:, -1:]  # [Batch,1]
        # 将小于k_value的值都赋值为负无穷
        logits.masked_fill_(logits < kth_values, float('-inf'))
    
    # =====================================================
    # 4. Top-P核采样
    # ====================================================
    if 0.0 < top_p < 1.0:
        # 4.1 将logits降序排列,需要记住原索引方便后续还原
        sorted_logits, sorted_indices = torch.sort(logits,descending= True, dim=-1)
        # 4.2 计算排好序后的概率值
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        # 4.3 计算排好序后的累加概率值
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 4.4 计算需要被剔除的词元索引
        indices_to_removed = cumulative_probs > top_p
        # 4.5 Mask 右移
        indices_to_removed[:, 1:] = indices_to_removed[:, :-1].clone()
        indices_to_removed[:, 0] = False
        # 4.6 把不需要的词排序后的logits设为-inf
        sorted_logits.masked_fill_(indices_to_removed, float('-inf'))
        # 4.7 把修改后的 Logits 【还原回原本的词表顺序】
        # 使用 scatter_ 算子：按照 sorted_indices，把 sorted_logits 填回空张量中
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    # ===========================================================
    # 5. 重归一化与多项式采样
    # ===========================================================
    probs = F.softmax(logits, dim=-1)
    # 用multinomial进行多项式采样
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


# ==============================================================
# 模块二：测试用例
# ===============================================================
def test_sampler():
    print('=' * 50)
    print('generate_next_token 采样管线极限抗压测试')
    print('=' * 50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 为了保证部分测试的可复现性，固定随机种子
    torch.manual_seed(42)

    # ===============================================
    # Test 1: 绝对贪婪测试
    # ================================================
    print(f'-> [Test 1] 绝对贪婪解码测试 (Temperature < 1e-5)')
    logits = torch.tensor([[10.0,20.0,1.0,2.0,5.0]],device=device)
    greedy_next_token = generate_next_token(logits,temperature=0.0)
    assert greedy_next_token.item() == 1,f'❌️ 贪婪解码失败！返回Tokens数为: {greedy_next_token.item()}'
    print('✅️ 贪婪验证测试成功！')

    # ===================================================
    # Test 2: Top-K 暴力截断测试(长尾词过滤)
    # =================================================
    print('-> [Test 2]  Top-K 长尾词过滤测试(K = 2)')
    logits = torch.tensor([[10.0,9.0,1.0,2.0,4.0]],device=device)
    sampled_tokens = []
    for _ in range(100):
        t = generate_next_token(logits,temperature=1.0,top_k=2,top_p=1.0)
        sampled_tokens.append(t.item())

    assert all(token in [0,1] for token in sampled_tokens),'❌️ Top-K 错误，采样到了长尾词!'
    print(f"  ✅ 成功验证：100次采样结果完全被锁定在 Top-2 内。样本展示: {sampled_tokens[:10]}\n")

    # =========================================================
    # Test 3: Top-P 右移测试
    # ==========================================================
    print('-> [Test 3] Top-P 边界测试')
    probs = torch.tensor([[0.5,0.3,0.2]],device=device)
    logits = torch.log(probs)

    # 设定top-p=0.7
    # 此时如果没有右移，则1号词元永远不会被采样到
    p_sampled = []
    for _ in range(100):
        t = generate_next_token(logits,temperature=1.0,top_k=0.0,top_p=0.7)
        p_sampled.append(t.item())

    assert 2 not in p_sampled,'❌️ 漏放了：Top-P掩码失效，把0.2的词也放了进来!'
    assert 1 in p_sampled,'❌️ 1号词没有被采样到，右移操作失效！'
    print("  ✅ 成功验证：右移魔术生效，刚好越界的边界词被完美保留！\n")

    # =========================================================
    # Test 4: Batch 独立性测试
    # ========================================================
    print('-> [Test 4] 多 Batch 并发独立性测试')
    # Batch 0: 词汇 2 最强
    # Batch 1: 词汇 0 最强
    # Batch 2: 词汇 3 最强
    logits = torch.tensor([[0.0, 0.0, 100.0, 0.0],[100.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 100.0]
    ], device=device)

    batch_tokens = generate_next_token(logits,temperature=1.0,top_k=50,top_p=0.9)
    assert batch_tokens[0,0].item() == 2,'Bacth 0 错误'
    assert batch_tokens[1,0].item() == 0,'Bacth 1 错误'
    assert batch_tokens[2,0].item() == 3,'Batch 2 错误'
    print("  ✅ 成功验证：多 Batch 并发时各行采样互不干扰！\n")

    # ===========================================================
    # Test 5: 参数越界抗压测试
    # ===========================================================
    print('-> [Test 5] 越界参数抗压测试')
    # 词表共五个词，但是用户硬要传top-k=100
    logits = torch.tensor([[1.0,2.0,3.0,5.0,5.0]],device=device)
    try:
        generate_next_token(logits,temperature=1.0,top_k=100,top_p=1.2)
        print("  ✅ 成功验证：代码能够自适应并兼容 K 超出词表 和 P 不规范的情况，无崩溃。\n")
    except Exception as e:
        print(f'❌️ 代码崩溃: {e}')

    print("🎉🎉🎉 恭喜！你的采样管线已通过全套大厂级验证，概率坍缩完美运行！")

if __name__ == '__main__':
    test_sampler()

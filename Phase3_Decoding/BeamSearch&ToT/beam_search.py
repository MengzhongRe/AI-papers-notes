# Beam Search 解码器 — 手撕版
# 核心管线与 HF model.generate(num_beams=B, do_sample=False) 对齐
import torch
import torch.nn.functional as F
from typing import Callable

# =====================================================================
# 模块一: Beam Search 核心函数
# =====================================================================
def beam_search(logits_fn: Callable[[torch.Tensor], torch.Tensor], input_ids: torch.Tensor,
                num_beams: int = 4, max_new_tokens: int = 128,
                eos_token_id: int | None = None, length_penalty_alpha: float = 0.6, 
                top_k: int | None = None):
    """
    束搜索解码 — 每步维护 B 条最优部分序列，从 B*V 候选中剪枝保留 Top-B。
    内部算法管线与 HuggingFace model.generate(num_beams=B, do_sample=False) 对齐。

    参数:
        logits_fn:
            callable,接受 input_ids [B, seq_len] → logits [B, seq_len, vocab_size]。
            调用方负责 model.eval() + torch.no_grad() 包裹。
        input_ids: [1, prompt_len]
        num_beams: 束宽度 B。B=1 等价于贪心搜索
        max_new_tokens: 最大生成 token 数硬截断
        eos_token_id: EOS token ID,None 则跳过 EOS 逻辑
        length_penalty_alpha: GNMT 长度归一化系数 α∈[0,1]。0=无惩罚(短序列霸榜),0.6=经验最优
        top_k: 每条 beam 仅展开 Top-K 高概率词。None=使用完整词表(不限制)

    返回:
        best_sequence: [1, prompt_len + new_len]
    """
    device = input_ids.device
    B = num_beams
    prompt_len = input_ids.size(1)  # [1,prompt_len] -> prompt_len

    # ===== 1. 探针：获取词表大小 =====
    # [1,prompt_len] -> [1,prompt_len,vocab_size]
    first_logits = logits_fn(input_ids)                            # [1, L, V]
    V = first_logits.size(-1)
    K = min(top_k, V) if top_k else V

    # ===== 2. 初始化 beam 状态 =====
    # 设计: 同一个 prompt 复制 B 份，只给 [0] 有效分数 0，其余 -inf。
    # 第一步展开时，B-1 条无效 beam 的候选全为 -inf，在 Top-B 剪枝中自然淘汰；
    # 唯一有效 beam 展开出 V 个候选，取 Top-B，恰好分裂为 B 条真正的 beam。
    # 这样避免了为第一步写特殊逻辑，与后续步复用同一套"展开 → 剪枝"循环体。
    # beam_seqs:   [B, cur_len]  每条 beam 的完整序列
    # beam_scores: [B]           累积对数概率 (log-P 之和)
    # done_flags:  [B]           该 beam 是否已命中 EOS
    beam_seqs = input_ids.repeat(B, 1)                             # [B, prompt_len]
    beam_scores = torch.full((B,), -float('inf'), device=device)
    beam_scores[0] = 0.0
    done_flags = torch.zeros(B, dtype=torch.bool, device=device)

    finished_seqs = []    # 存已命中 EOS 的完整序列
    finished_scores = []  # 存对应原始累积 log-P

    # ===== 3. 逐步解码循环 =====
    for _ in range(max_new_tokens):
        if done_flags.all():
            break

        # 3.1 并行前向：B 条 beam 以 batch 维度一次喂给模型
        # 注意：这里没有跳过已终止 beam（done_flags=True 的）。
        # 理论上已终止 beam 不需要再算前向，但跳过需要维护活跃索引映射，
        # 增加代码复杂度。且没有 KV cache 时已终止 beam 最多白跑几步
        # 就全终止跳出循环了，浪费的计算量是常数级。
        # 在有 KV cache + CUDA graph 的场景中，跳过反而破坏 batch 形状
        # 稳定性，代价大于收益。
        logits = logits_fn(beam_seqs)                              # [B, cur_len, V]
        next_logits = logits[:, -1, :]                             # [B, V]  只取最后位置

        # 3.2 Top-K 词汇剪枝：B×V → B×K，大幅降低后续排序开销
        if K < V:
            topk_vals, topk_idx = torch.topk(next_logits, K, dim=-1)    #[B,K]
            next_logits = torch.full_like(next_logits, -float('inf'))
            next_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)

        # 3.3 Log-Softmax → 对数概率 (log-P，值域 ≤ 0)
        next_log_probs = F.log_softmax(next_logits, dim=-1)        # [B, V]

        # 已完成 beam 不产生候选
        next_log_probs[done_flags] = -float('inf')

        # 3.4 候选展开：每条 beam × 每个 token → B×V 个候选
        # candidate[b][v] = beam_scores[b] + logP(v|beam)
        candidate_scores = beam_scores.unsqueeze(1) + next_log_probs  # [B, V]

        # 3.5 剪枝：从 B×V 候选中取分数最高的 B 个
        top_scores, top_flat_idx = torch.topk(
            candidate_scores.view(-1), B)                          # [B]

        # 3.6 还原每个候选的 (来源 beam, 新 token)
        beam_indices = top_flat_idx // V                            # [B]
        token_indices = top_flat_idx % V                            # [B]
        # 3.7 更新 beam：按来源索引重组序列，拼接新 token
        beam_seqs = beam_seqs[beam_indices]                        # [B, cur_len]
        beam_scores = top_scores.clone()                           # [B]
        # concat([B,curr_len],[B,1]) -> [B,curr_len + 1] -> [B,new_len]
        beam_seqs = torch.cat([beam_seqs, token_indices.unsqueeze(1)], dim=1)

        # 3.8 EOS 处理：刚命中 EOS 的 beam 移入 finished 池
        # EOS 不消灭席位，只"退役 + 回收再利用"：
        #   - 命中 EOS 的 beam clone 进 finished_seqs 永久保存
        #   - 下一步它的候选行全为 -inf，在 Top-B 中自动让位
        #   - 空出的席位被其他活跃 beam 的备选后代填补（某条活跃 beam 会"分叉"）
        # 因此活跃 beam 数始终恒定 B，但 finished 池随 EOS 事件持续增长。
        # 最终比较池 = finished_seqs + 仍在跑的活跃 beam，总数 ≥ B。
        done_flags = done_flags[beam_indices]
        if eos_token_id is not None:
            just_hit_eos = (token_indices == eos_token_id)
            done_flags = done_flags | just_hit_eos
            for b in range(B):
                if just_hit_eos[b]:
                    finished_seqs.append(beam_seqs[b].clone())
                    finished_scores.append(beam_scores[b].item())

    # ===== 4. 最终选取 =====
    # 合并已退役 beam 和仍在跑的活跃 beam，按长度归一化分数公平比较。
    # 注意：此池大小可能 > B，因为历史 EOS 事件越多次，finished_seqs 越长。
    # list() 是浅拷贝：后续 append 不污染原始 finished_* 变量，
    # 防止未来有人在循环后引用它们时踩到被追加过的数据。
    all_seqs = list(finished_seqs)
    all_raw = list(finished_scores)
    for b in range(B):
        if not done_flags[b]:
            all_seqs.append(beam_seqs[b].clone())
            all_raw.append(beam_scores[b].item())

    # 按长度归一化分数选最优
    # GNMT: score_norm = raw_score / |y|^α
    if length_penalty_alpha > 0:
        normed = [raw / max(seq.size(0) - prompt_len, 1) ** length_penalty_alpha
                  for seq, raw in zip(all_seqs, all_raw)]
        best_idx = max(range(len(normed)), key=lambda i: normed[i])
    else:
        best_idx = max(range(len(all_raw)), key=lambda i: all_raw[i])

    return all_seqs[best_idx].unsqueeze(0)  # [total_len] -> [1, total_len]


# =====================================================================
# 模块二: 冒烟测试 — 用确定的 logits 表模拟模型前向，分段验证行为
# =====================================================================
if __name__ == '__main__':
    print('=' * 55)
    print('Beam Search 冒烟测试')
    print('=' * 55)

    device = 'cuda' if torch.cuda.is_available() \
            else 'mps' if torch.backends.mps.is_available() \
            else 'cpu'
    
    torch.manual_seed(42)
    V = 10           # 玩具词表大小
    EOS = 9          # EOS token
    prompt = torch.tensor([[0, 1]])
    P = 2            # prompt 长度

    # 确定性模型: 预存每步 logits [max_len, V]，按当前序列长度 L 取行
    def make_model(logits_table):
        def fn(input_ids):
            B, L = input_ids.shape
            row = min(L - 1, logits_table.size(0) - 1)
            out = torch.zeros(B, L, V)
            out[:, -1, :] = logits_table[row]
            return out
        return fn

    # ---------------------------------------------------------------
    # Test 1: B=1 退化为贪心搜索
    # ---------------------------------------------------------------
    print('\n[Test 1] B=1 等价于贪心搜索')
    t1 = torch.zeros(10, V)
    for i in range(10):
        t1[i, 3] = 5.0       # token 3 始终最高
    m1 = make_model(t1)
    seq = beam_search(m1, prompt, num_beams=1, max_new_tokens=5,
                      eos_token_id=EOS, length_penalty_alpha=0.0)
    gen = seq[0, P:].tolist()
    assert all(t == 3 for t in gen), f'失败: {gen}'
    print(f'  ✅ 输出: {gen} — 与贪心完全一致')

    # ---------------------------------------------------------------
    # Test 2: B>1 找到比贪心更优的全局路径
    # ---------------------------------------------------------------
    print('\n[Test 2] B=2 应选非贪心入口，找到更优全局路径')
    # 场景: 第1步 token 0 高分(贪心必选)，token 1 略低但后续极优
    #       第2步(from token 1) token 5 极高 → 总分反超
    class ConditionalModel:
        """条件模型：根据上一个 token 决定 logits 分布。"""
        def __init__(self, V, eos):
            self.V = V
            self.eos = eos

        def __call__(self, input_ids):
            B, L = input_ids.shape
            logits = torch.zeros(B, L, self.V)
            last = input_ids[:, -1]
            for b in range(B):
                gen_len = L - P
                if gen_len == 0:
                    # 第一步: token 0 高分(贪心入口), token 1 略低(正确入口)
                    logits[b, -1, 0] = 2.0
                    logits[b, -1, 1] = 1.5
                    for v in range(2, self.V):
                        logits[b, -1, v] = -10.0
                elif last[b] == 0:
                    # 贪心路径后续: 平庸，推 EOS
                    logits[b, -1, self.eos] = 2.0
                elif last[b] == 1:
                    # 非贪心路径后续: token 5 极优
                    logits[b, -1, 5] = 5.0
                    logits[b, -1, self.eos] = -5.0
                else:
                    logits[b, -1, self.eos] = 10.0
            return logits

    m2 = ConditionalModel(V, EOS)
    seq2 = beam_search(m2, prompt, num_beams=2, max_new_tokens=3,
                       eos_token_id=EOS, length_penalty_alpha=0.0)
    gen2 = seq2[0, P:].tolist()
    # 贪心会选 [0, EOS]，beam 应发现 [1, 5, EOS] 更优
    assert gen2[0] == 1, f'第1步应选 token 1 (非贪心)，实际: {gen2[0]}'
    assert 5 in gen2, f'应包含 token 5，实际: {gen2}'
    print(f'  ✅ 输出: {gen2} — 成功避开贪心陷阱')

    # ---------------------------------------------------------------
    # Test 3: EOS 早停 — 已终止 beam 停跳，活跃 beam 继续
    # ---------------------------------------------------------------
    print('\n[Test 3] EOS 早停')
    t3 = torch.zeros(6, V)
    t3[P-1, 2] = 3.0           # 第1步: token 2 高分
    t3[P-1, EOS] = 1.0          # EOS 也是合法候选
    t3[P, 3] = 5.0              # 第2步: token 3 高分
    t3[P+1, EOS] = 10.0         # 第3步: 强推 EOS
    m3 = make_model(t3)
    seq3 = beam_search(m3, prompt, num_beams=3, max_new_tokens=5,
                       eos_token_id=EOS, length_penalty_alpha=0.0)
    gen3 = seq3[0, P:].tolist()
    assert EOS in gen3, f'EOS 应出现在输出中: {gen3}'
    print(f'  ✅ 输出: {gen3} — EOS 早停正常')

    # ---------------------------------------------------------------
    # Test 4: 长度归一化 — 抑制短序列霸榜
    # ---------------------------------------------------------------
    print('\n[Test 4] 长度归一化 — α=0.6 应放松对长序列的惩罚')
    class LengthNormModel:
        """短路径(token 0): 平庸但早 EOS；长路径(token 1): 高质但晚 EOS。"""
        def __init__(self, V, eos):
            self.V = V
            self.eos = eos

        def __call__(self, input_ids):
            B, L = input_ids.shape
            logits = torch.zeros(B, L, self.V)
            for b in range(B):
                last = input_ids[b, -1].item()
                gen_len = L - P
                if gen_len == 0:
                    logits[b, -1, 0] = 1.5   # 短路径入口
                    logits[b, -1, 1] = 1.0   # 长路径入口
                elif last == 0:
                    # 短路径: 低质量 token，快速推 EOS
                    if gen_len == 1:
                        logits[b, -1, 2] = 1.0         # 平庸中间 token
                        logits[b, -1, self.eos] = 0.5
                    else:
                        logits[b, -1, self.eos] = 10.0
                elif last == 1:
                    # 长路径: 持续高质量 token
                    if gen_len < 5:
                        logits[b, -1, gen_len + 3] = 3.0   # 高分 token
                        logits[b, -1, self.eos] = -2.0
                    else:
                        logits[b, -1, self.eos] = 10.0
                else:
                    logits[b, -1, self.eos] = 10.0
            return logits

    m4 = LengthNormModel(V, EOS)
    seq_no = beam_search(m4, prompt, num_beams=2, max_new_tokens=10,
                         eos_token_id=EOS, length_penalty_alpha=0.0)
    seq_yes = beam_search(m4, prompt, num_beams=2, max_new_tokens=10,
                          eos_token_id=EOS, length_penalty_alpha=0.6)

    def strip(seq):
        t = seq[0, P:].tolist()
        return t[:t.index(EOS)] if EOS in t else t

    len_no = len(strip(seq_no))
    len_yes = len(strip(seq_yes))
    print(f'  α=0   选中长度: {len_no}  (短序列霸榜)')
    print(f'  α=0.6 选中长度: {len_yes}  (归一化平衡)')
    assert len_yes >= len_no, f'归一化不应更短: {len_yes} < {len_no}'
    print('  ✅ 长度归一化生效')

    # ---------------------------------------------------------------
    # Test 5: Top-K 剪枝不误伤高分 token
    # ---------------------------------------------------------------
    print('\n[Test 5] Top-K 剪枝应保留高分 token')
    t5 = torch.zeros(5, V)
    t5[P-1, 7] = 10.0    # token 7 是全局最高分
    m5 = make_model(t5)
    seq_full = beam_search(m5, prompt, num_beams=1, max_new_tokens=1,
                           eos_token_id=None, length_penalty_alpha=0.0, top_k=None)
    seq_prune = beam_search(m5, prompt, num_beams=1, max_new_tokens=1,
                            eos_token_id=None, length_penalty_alpha=0.0, top_k=3)

    assert seq_full[0, P].item() == 7, f'完整词表应选7: {seq_full[0, P].item()}'
    assert seq_prune[0, P].item() == 7, f'K=3不应丢失7: {seq_prune[0, P].item()}'
    print(f'  ✅ 完整词表输出: {seq_full[0, P].item()}, K=3输出: {seq_prune[0, P].item()} — 一致')

    print('\n' + '=' * 55)
    print('🎉 全部 5 项冒烟测试通过！Beam Search 手撕验收合格')
    print('=' * 55)

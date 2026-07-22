# 状态化 GQA（分组查询注意力）— 静态预分配缓存版本
#
# 继承 gqa_attention_base.py 的共享基类，实现 register_buffer 预分配的
# KV Cache 策略。推理时通过 start_pos 索引进行就地写入 (In-place)，
# 避免动态 torch.cat 的内存碎片和重复分配开销。
#
# 关键特性:
#   1. register_buffer 预分配: 静态缓存不随 optimizer.step() 更新
#   2. persistent=False: KV Cache 不保存在 state_dict 中
#   3. start_pos 索引: Prefill (start_pos=0) 和 Decode (start_pos>0) 统一接口
#   4. 固定形状兼容 torch.compile / CUDA Graph
#
# 相关文件：grouped_query_attention.py（动态缓存版本，适合教学理解 GQA 机制）

import torch
import torch.nn as nn
import math
from gqa_attention_base import GroupedQueryAttentionBase


class StatefulGQA(GroupedQueryAttentionBase):
    """静态预分配缓存 GQA — 适用于生产推理环境。"""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 max_batch_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__(d_model, num_heads, num_kv_heads, dropout)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # register_buffer 预分配静态缓存，persistent=False 不写进 state_dict
        self.register_buffer(
            'k_cache',
            torch.zeros(max_batch_size, num_kv_heads, max_seq_len, self.head_dim),
            persistent=False
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(max_batch_size, num_kv_heads, max_seq_len, self.head_dim),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        Args:
            x: [Batch, N_curr, d_model]
            start_pos: 输入在全量序列中的起始位置
                       Prefill 阶段: start_pos == 0
                       Decode 阶段: start_pos > 0
        Returns:
            [Batch, N_curr, d_model]
        """
        q, k, v, B, N_curr = self._project_qkv(x)

        # ── 就地写入 KV Cache ──
        self.k_cache[:B, :, start_pos: start_pos + N_curr, :] = k
        self.v_cache[:B, :, start_pos: start_pos + N_curr, :] = v

        # ── 提取历史 KV ──
        if start_pos > 0:
            k = self.k_cache[:B, :, : start_pos + N_curr, :]
            v = self.v_cache[:B, :, : start_pos + N_curr, :]

        k, v = self._expand_kv_heads(k, v)
        return self._attention(q, k, v, N_curr, device=x.device)


# ==================================================================
# 测试用例
# ==================================================================
def test_gqa():
    print(' 🚀 开始进行手撕代码的 GQA 测试......')
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d_model = 128
    num_q_heads = 8
    num_kv_heads = 2
    max_batch_size = 4
    max_seq_len = 32
    head_dim = d_model // num_q_heads

    gqa = StatefulGQA(
        d_model=d_model,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dropout=0.0
    ).to(device)

    B, S = 2, 5
    x = torch.randn(B, S, d_model, device=device)

    # ── Test 1: Prefill 阶段形状校验 ──
    print('[*] Test 1: Prefill 阶段输出形状校验...')
    out_prefill = gqa(x, start_pos=0)
    assert out_prefill.shape == (B, S, d_model), f'❌ Prefill 输出形状错误: {out_prefill.shape}'
    print(f'   Prefill 输出形状: {out_prefill.shape} ✅')

    # ── Test 2: Decode 阶段形状校验 ──
    print('[*] Test 2: Decode 阶段输出形状校验...')
    x_decode = torch.randn(B, 1, d_model, device=device)
    out_decode = gqa(x_decode, start_pos=S)
    assert out_decode.shape == (B, 1, d_model), f'❌ Decode 输出形状错误: {out_decode.shape}'
    print(f'   Decode 输出形状: {out_decode.shape} ✅')

    # ── Test 3: 数值等价性校验（Prefill vs 手动 GQA）──
    print('[*] Test 3: 数值等价性校验（手动 GQA 对比）...')
    gqa2 = StatefulGQA(
        d_model=d_model, num_heads=num_q_heads, num_kv_heads=num_kv_heads,
        max_batch_size=max_batch_size, max_seq_len=max_seq_len, dropout=0.0
    ).to(device)

    Wq = gqa2.Wq.weight.data
    Wk = gqa2.Wk.weight.data
    Wv = gqa2.Wv.weight.data
    Wo = gqa2.Wo.weight.data

    B2, S2 = 2, 4
    x2 = torch.randn(B2, S2, d_model, device=device)
    num_queries_per_kv = num_q_heads // num_kv_heads

    # 手动计算 GQA
    q_manual = (x2 @ Wq.T).view(B2, S2, num_q_heads, head_dim).transpose(1, 2)
    k_manual = (x2 @ Wk.T).view(B2, S2, num_kv_heads, head_dim).transpose(1, 2)
    v_manual = (x2 @ Wv.T).view(B2, S2, num_kv_heads, head_dim).transpose(1, 2)

    k_exp = k_manual.unsqueeze(2).expand(B2, num_kv_heads, num_queries_per_kv, S2, -1)
    k_exp = k_exp.contiguous().view(B2, num_q_heads, S2, -1)
    v_exp = v_manual.unsqueeze(2).expand(B2, num_kv_heads, num_queries_per_kv, S2, -1)
    v_exp = v_exp.contiguous().view(B2, num_q_heads, S2, -1)

    scores_manual = torch.matmul(q_manual, k_exp.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.tril(torch.ones(S2, S2, device=device)).bool()
    scores_manual = scores_manual.masked_fill(~causal_mask, float('-inf'))
    attn_manual = torch.softmax(scores_manual, dim=-1)
    out_manual = attn_manual @ v_exp
    out_manual = out_manual.transpose(1, 2).contiguous().view(B2, S2, -1)
    out_manual = out_manual @ Wo.T

    out_model = gqa2(x2, start_pos=0)

    max_diff = (out_model - out_manual).abs().max().item()
    print(f'   与手动 GQA 的最大误差: {max_diff:.6e}')
    assert torch.allclose(out_model, out_manual, atol=1e-5), '❌ 与手动 GQA 计算不一致!'

    print(' 🎉 所有测试通过!')


if __name__ == '__main__':
    test_gqa()

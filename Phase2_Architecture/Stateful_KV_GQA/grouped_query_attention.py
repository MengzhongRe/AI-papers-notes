# GQA（分组查询注意力）— 动态缓存版本
#
# 继承 gqa_attention_base.py 的共享基类，只需实现动态 cache 管理策略：
#   每次 forward 通过 torch.cat 拼接历史 KV Cache，
#   逻辑简单直观，适合教学理解 GQA 核心机制和 Prefill/Decode 的行为差异。
#
# 相关文件：stateful_kvcache_gqa.py（静态预分配缓存版本，生产环境推荐）

import torch
import torch.nn as nn
from gqa_attention_base import GroupedQueryAttentionBase


class GroupedQueryAttention(GroupedQueryAttentionBase):
    """动态缓存 GQA — 每次 forward 通过 torch.cat 追加历史 KV 到当前 KV。"""

    def forward(self, x, past_key_value=None, use_cache=False):
        """
        Args:
            x: [Batch_size, Seq_len, d_model]
            past_key_value: (past_keys, past_values) 元组
            use_cache: 是否返回当前 KV Cache 供下一步解码使用
        """
        q, k, v, B, N_curr = self._project_qkv(x)

        # ── 动态缓存：拼接历史 KV ──
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        k, v = self._expand_kv_heads(k, v)
        out = self._attention(q, k, v, N_curr, device=x.device)
        return out, present_key_value


# =============================
# 测试用例
# =============================
if __name__ == '__main__':
    print('=' * 60)
    print(' GQA（分组查询注意力）— 动态缓存版本 测试')
    print('=' * 60)
    B = 2
    d_model = 4096
    num_heads = 32
    num_kv_heads = 8

    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    head_dim = d_model // num_heads

    print('================= 阶段一: Prefill（预填充阶段）=================')
    seq_len = 10
    x_prefill = torch.randn(B, seq_len, d_model)

    out_prefill, kv_cache = gqa(x_prefill, past_key_value=None, use_cache=True)

    print(f'Prefill 输出维度：{out_prefill.shape}')
    print(f'KV Cache 维度：{kv_cache[0].shape}')

    assert out_prefill.shape == (B, seq_len, d_model)
    assert kv_cache[0].shape == (B, num_kv_heads, seq_len, head_dim)
    assert not torch.isnan(out_prefill).any(), 'Prefill 输出包含 NaN!'
    assert not torch.isinf(out_prefill).any(), 'Prefill 输出包含 Inf!'

    print('\n================= 阶段二: Decoding（逐字解码阶段）=================')
    x_decode = torch.randn(B, 1, d_model)

    out_decode, kv_cache = gqa(x_decode, past_key_value=kv_cache, use_cache=True)

    print(f'Decoding Step 1 输出维度：{out_decode.shape}')
    print(f'更新后的 KV Cache 维度：{kv_cache[0].shape}')

    assert out_decode.shape == (B, 1, d_model)
    assert kv_cache[0].shape == (B, num_kv_heads, seq_len + 1, head_dim)
    assert not torch.isnan(out_decode).any(), 'Decoding 输出包含 NaN!'

    # 第二步
    x_decode2 = torch.randn(B, 1, d_model)
    out_decode2, kv_cache = gqa(x_decode2, past_key_value=kv_cache, use_cache=True)
    assert out_decode2.shape == (B, 1, d_model)
    assert kv_cache[0].size(2) == seq_len + 2, 'KV Cache 序列长度应为 12'

    print('\n🎉 所有断言通过！GQA 动态缓存版本验证完成')

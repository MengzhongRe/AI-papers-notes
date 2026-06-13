# GQA（分组查询注意力）共享基类
#
# 提取 grouped_query_attention.py 和 stateful_kvcache_gqa.py 的共同逻辑：
#   - QKV 投影 + 视图变换
#   - GQA KV 头广播扩展
#   - 缩放点积注意力 + 因果掩码 + Softmax + Dropout
#   - 多头合并 + 输出投影
#
# 子类在 forward() 中直接实现各自的 cache 管理策略
# （动态 torch.cat 或静态 register_buffer），无需重写钩子。

import torch
import torch.nn as nn
import math


class GroupedQueryAttentionBase(nn.Module):
    """GQA 共享基类。"""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        assert num_heads % num_kv_heads == 0, 'num_heads must be divisible by num_kv_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.Wq = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ── GQA KV 头广播 ──────────────────────────────────────────────

    def _project_qkv(self, x: torch.Tensor):
        """QKV 投影 + 视图变换。"""
        B, N_curr, _ = x.size()
        q = self.Wq(x).view(B, N_curr, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N_curr, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N_curr, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v, B, N_curr

    def _expand_kv_heads(self, k: torch.Tensor, v: torch.Tensor):
        """零拷贝广播：将 KV 从 num_kv_heads 扩展到 num_heads。

        注意：expand 后调用 contiguous().view() 会触发显存拷贝——
        这是为了实现 .view() 的代价，在实际生产中可以用 reshape 替代。
        """
        B, _, N_kv, _ = k.shape
        k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv,
                                   N_kv, self.head_dim).reshape(B, self.num_heads, N_kv, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv,
                                   N_kv, self.head_dim).reshape(B, self.num_heads, N_kv, self.head_dim)
        return k, v

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   N_curr: int, device):
        """缩放点积注意力 + 因果掩码 + Softmax + Dropout + 多头合并。"""
        N_kv = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if N_curr > 1:
            mask = torch.tril(torch.ones(N_curr, N_kv, device=device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # [B, H, N_curr, head_dim]
        out = out.transpose(1, 2).contiguous().view(q.size(0), N_curr, -1)
        return self.Wo(out)

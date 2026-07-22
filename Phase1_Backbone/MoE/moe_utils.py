# MoE 共享工具函数 — 辅助损失（Auxiliary Loss）计算
#
# 被 moe_layer.py 和 moe_layer_naive.py 共同引用。
# 消除 ~15 行重复的 aux_loss 逻辑。

import torch
import torch.nn.functional as F


def compute_aux_loss(logits: torch.Tensor, indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """计算负载均衡辅助损失（Load Balancing Auxiliary Loss）。

    公式: aux_loss = N * sum(f * P)
    其中 P 是每个专家被选中的全局路由概率，f 是每个专家实际被选为 top-k 的频率。

    Args:
        logits: 路由层原始输出 [num_tokens, num_experts]
        indices: top-k 选中的专家索引 [num_tokens, top_k]
        num_experts: 专家总数 N

    Returns:
        aux_loss: 标量损失值
    """
    # 全局路由概率 P = mean(Softmax(logits))
    prob = F.softmax(logits, dim=-1)          # [num_tokens, num_experts]
    P = prob.mean(dim=0)                       # [num_experts]

    # 每个专家实际被选为 top-k 的频率 f
    mask = torch.zeros_like(logits)
    mask.scatter_(1, indices, 1.0)
    f = mask.mean(dim=0)                       # [num_experts]

    return num_experts * torch.sum(f * P)

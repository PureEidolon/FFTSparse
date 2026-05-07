import torch
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda


def Sparge_prefill(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        topk: float = 0.5,
        is_causal: bool = False,
):
    # spas_sage_attn 输入格式: [batch, nheads, seqlen, headdim]
    # query_states 已经是 [B, H, L, D] 格式，直接使用

    attn_output = spas_sage2_attn_meansim_topk_cuda(
        query_states,
        key_states,
        value_states,
        topk=topk,
        is_causal=is_causal,
    )  # [B, H, L, D]

    return attn_output
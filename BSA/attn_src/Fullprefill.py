import torch
from flash_attn import flash_attn_func

def Full_prefill(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        causal: bool = True,
        attention_mask=None,
):
    # flash_attn 输入格式: [batch, seqlen, nheads, headdim]
    q = query_states.transpose(1, 2)  # [B, L, H, D]
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    attn_output = flash_attn_func(
        q, k, v,
        causal=causal,
    )  # [B, L, H, D]

    return attn_output.transpose(1, 2)  # [B, H, L, D]
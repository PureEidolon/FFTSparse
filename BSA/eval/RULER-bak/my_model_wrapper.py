import sys
import torch
import types
import math
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, nn
from model_wrappers import HuggingFaceModel

# 加入项目根目录
sys.path.insert(0, "/root/cjh/pro/BSA")

from attn_src.fft_attn.my_attn_v7 import myattn_prefill
from attn_src.xattn.src.Xattention import Xattention_prefill
from attn_src.Flexprefill import Flexprefill_prefill
from attn_src.Minference import Minference_prefill
from attn_src.Sparge import Sparge_prefill
from flash_attn import flash_attn_func


@torch.no_grad()
def new_attention_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, output_attentions=False, use_cache=False,
    cache_position=None, position_embeddings=None, **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:  # prefill 阶段
        method = self.method
        cfg    = self.attn_cfg

        if method == "full":
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)

        elif method == "myattn":
            attn_output = myattn_prefill(
                query_states, key_states, value_states,
                layer_idx=self.layer_idx,
                block_size=cfg.get("block_size", 128),
                is_causal=True,
                sink_ratio=cfg.get("sink_ratio", 0.01),
                recent_ratio=cfg.get("recent_ratio", 0.01),
                local_span_ratio=cfg.get("local_ratio", 0.02),
                enable_correlation_mask=cfg.get("use_cor", False),
                correlation_selection_mode='threshold',
                corr_threshold=cfg.get("corr_thres", 1.0),
                enable_last_block_mask=cfg.get("enable_last_block", False),
                last_block_threshold=cfg.get("last_block_thres", 0.01),
                is_visual=False,
                collect_corr_stats=False,
            ).reshape(1, -1, 32, 128).permute(0, 2, 1, 3)

        elif method == "xattn":
            attn_output = Xattention_prefill(
                query_states, key_states, value_states,
                norm=1, stride=8,
                threshold=self.threshold.to(key_states.device),
                use_triton=True, keep_sink=True, keep_recent=True,
            )

        elif method == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=0.9, tau=0.1,
            ).transpose(1, 2)

        elif method == "minference":
            attn_output = Minference_prefill(query_states, key_states, value_states)

        elif method == "sparge":
            attn_output = Sparge_prefill(
                query_states, key_states, value_states,
                topk=0.5, is_causal=False,
            )

    else:  # decode 阶段，标准 attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output  = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


class MyCustomModel(HuggingFaceModel):
    def __init__(self, name_or_path: str, method: str = "full", attn_cfg: dict = None, **generation_kwargs):
        # 不调用父类 __init__，自己加载并注入 attention
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="auto", attn_implementation="eager",
        )

        # 注入自定义 attention
        attn_cfg = attn_cfg or {}
        for name, module in self.model.named_modules():
            if name.split(".")[-1] == "self_attn":
                module.method   = method
                module.attn_cfg = attn_cfg
                if method == "xattn":
                    from ratio import max
                    layer_idx = int(name.split(".")[2])
                    module.threshold = torch.tensor(max[layer_idx])
                module.forward = types.MethodType(new_attention_forward, module)

        self.model.eval()
        self.pipeline = None
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop', [])

        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side  = 'left'
            self.tokenizer.pad_token     = self.tokenizer.eos_token
            self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id

        print(f"✅ 模型加载完成，method={method}, attn_cfg={attn_cfg}")

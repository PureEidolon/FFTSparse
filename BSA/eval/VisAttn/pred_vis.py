# eval/VisAttn/pred_vis.py
#
# 最简推理脚本，专门用于注意力可视化。
# 只跑少量样本，不保存预测结果，不评估指标。
#
# 使用方式：
#   python -u eval/VisAttn/pred_vis.py \
#       --model llama-3.1-8b-instruct \
#       --task hotpotqa \
#       --num_samples 1 \
#       --vis_layers 0,8 \
#       --vis_heads 0,1 \
#       --vis_dir ./vis_attn

import sys

# == 第一步：先加项目根目录到 sys.path ======================================
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 0是父目录，1是爷目录
print("Path(__file__):",Path(__file__))
sys.path.insert(0, str(ROOT_DIR))
print(f"✅ 已把路径 '{ROOT_DIR}' 加入 sys.path")
# =======================================================================


import json
import math
import types
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, nn
from transformers.cache_utils import Cache

from eval.VisAttn.full_attn_vis import maybe_vis_full_attn



# =============================================================================
# 参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, required=True)
    parser.add_argument("--task",           type=str, required=True)
    parser.add_argument("--num_samples",    type=int, default=1)
    parser.add_argument("--vis_layers",     type=str, default="0")
    parser.add_argument("--vis_heads",      type=str, default="0")
    parser.add_argument("--vis_dir",        type=str, default="./vis_attn")
    parser.add_argument("--vis_downsample", type=int, default=512)
    return parser.parse_args()


# =============================================================================
# Attention forward：prefilling 时触发可视化，其余走标准计算
# =============================================================================

@torch.no_grad()
def vis_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states   = repeat_kv(key_states,   self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # prefilling 阶段（Q、K 等长）→ 触发可视化
    if key_states.shape[2] == query_states.shape[2]:
        if self.layer_idx in self.vis_cfg["vis_layers"]:
            maybe_vis_full_attn(query_states, key_states, self.layer_idx, self.vis_cfg)

    # 标准 attention 计算
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output  = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


# =============================================================================
# 主流程
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    if args.vis_layers.strip() == "-1":
        vis_layers = set(range(32))
    else:
        vis_layers = set(int(x) for x in args.vis_layers.split(","))

    vis_cfg = dict(
        vis_layers    = vis_layers,
        vis_heads     = [int(x) for x in args.vis_heads.split(",")],
        vis_dir       = args.vis_dir,
        vis_downsample= args.vis_downsample,
    )

    # 加载模型
    model2path = json.load(open(str(ROOT_DIR)+"/eval/LongBench/config/model2path.json"))
    tokenizer  = AutoTokenizer.from_pretrained(model2path[args.model], use_fast=False)
    model      = AutoModelForCausalLM.from_pretrained(
        model2path[args.model],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    # patch attention，注入 vis_cfg
    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            module.vis_cfg = vis_cfg
            module.forward = types.MethodType(vis_attention_forward, module)

    # 加载数据
    dataset2prompt = json.load(open(str(ROOT_DIR)+"/eval/LongBench/config/dataset2prompt.json"))
    data_path      = f"/root/cjh/pro/resources/datasets/LongBench/{args.task}.jsonl"
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))

    # 只跑 prefilling，不需要生成
    for idx, sample in enumerate(samples):
        print(f"\n===== Sample {idx} =====")
        prompt = dataset2prompt[args.task].format(**sample)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7500).to("cuda")
        print(f"seq_len = {inputs.input_ids.shape[-1]}")
        with torch.no_grad():
            model(**inputs, use_cache=False)
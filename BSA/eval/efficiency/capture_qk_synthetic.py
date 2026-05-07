#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
capture_qk_synthetic.py

"大海捞针" QK 数据生成脚本：
1. 用无关文字（"大海"）+ 有用文字（"针"）拼接成指定长度的文本
2. 一次性 forward 整个序列（不做 chunk 切分，避免 chunk 边界导致的分块伪影）
3. 保存格式与 capture_qk_real.py 完全一致：
   - query_layer{L}_{seq_len}.pkl  →  tensor [1, H, L, D]
   - key_layer{L}_{seq_len}.pkl    →  tensor [1, H, L, D]
   - needle_meta_{seq_len}.json    →  针的位置等元信息

使用方式：
  python capture_qk_synthetic.py \
      --model_path /path/to/llama-3.1-8b-instruct \
      --layers 0 3 7 11 15 19 23 27 30 31 \
      --lens 8 16 32 64 \
      --needle_position random \
      --out_dir ./output/needle_qk_data
"""

import os
import sys
import math
import json
import types
import pickle
import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# 参数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Needle-in-a-haystack QK capture")
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace 模型路径")
    p.add_argument("--layers", type=int, nargs="+", default=[0],
                   help="要截取的层，空格分隔")
    p.add_argument("--lens", type=int, nargs="+", default=[8, 16, 32, 64],
                   help="目标长度（单位 K），空格分隔")
    p.add_argument("--needle_position", type=str, default="middle",
                   choices=["beginning", "middle", "end", "quarter", "three_quarter", "random"],
                   help="针插入的位置")
    p.add_argument("--out_dir", type=str, default="./needle_qk_data")

    # 自定义文本（可选）
    p.add_argument("--haystack_text", type=str, default=None,
                   help="自定义大海文本（不指定则用默认）")
    p.add_argument("--needle_text", type=str, default=None,
                   help="自定义针文本（不指定则用默认）")
    p.add_argument("--question", type=str, default=None,
                   help="自定义问题（不指定则用默认）")
    return p.parse_args()


# =============================================================================
# 默认文本素材
# =============================================================================

DEFAULT_HAYSTACK = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Birds were singing in the trees as the morning sun rose above the horizon. "
    "A gentle breeze carried the scent of wildflowers across the open meadow. "
    "The old wooden bridge creaked under the weight of passing travelers. "
    "Children played in the park while their parents watched from nearby benches. "
    "The library was filled with ancient books covered in a thin layer of dust. "
    "Raindrops tapped against the window pane creating a soothing rhythm. "
    "The mountain trail wound through dense forests and rocky outcrops. "
    "Fishermen cast their lines into the calm waters of the lake at dawn. "
    "The marketplace bustled with vendors selling fruits, spices, and handmade crafts. "
)

DEFAULT_NEEDLE = (
    "The secret password to access the hidden vault is 'Strawberry Sunshine 7492'. "
    "Remember this carefully as it will be needed later."
)

DEFAULT_QUESTION = (
    "What is the secret password to access the hidden vault? "
    "Answer based on the information provided in the text above."
)


# =============================================================================
# 文本拼接
# =============================================================================

def build_needle_in_haystack_prompt(
    tokenizer,
    target_length: int,
    haystack_text: str,
    needle_text: str,
    question: str,
    needle_position: str = "middle",
):
    prefix = "Read the following text carefully and answer the question at the end.\n\n"
    suffix = f"\n\n{question}\nAnswer:"

    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle_text, add_special_tokens=False))
    special_tokens = 1

    haystack_budget = target_length - prefix_tokens - suffix_tokens - needle_tokens - special_tokens

    if haystack_budget <= 0:
        print(f"  [警告] target_length={target_length} 太小，无法容纳所有内容")
        haystack_budget = 100

    haystack_unit_tokens = len(tokenizer.encode(haystack_text, add_special_tokens=False))
    repeats_needed = (haystack_budget // haystack_unit_tokens) + 2
    full_haystack = haystack_text * repeats_needed

    haystack_token_ids = tokenizer.encode(full_haystack, add_special_tokens=False)[:haystack_budget]

    position_map = {
        "beginning": 0.0,
        "quarter": 0.25,
        "middle": 0.5,
        "three_quarter": 0.75,
        "end": 1.0,
    }

    if needle_position == "random":
        ratio = np.random.uniform(0.05, 0.95)
        print(f"  随机插入位置: {ratio:.2%}")
    else:
        ratio = position_map[needle_position]

    split_point = int(len(haystack_token_ids) * ratio)
    haystack_before = tokenizer.decode(haystack_token_ids[:split_point], skip_special_tokens=True)
    haystack_after = tokenizer.decode(haystack_token_ids[split_point:], skip_special_tokens=True)

    prompt = prefix + haystack_before + " " + needle_text + " " + haystack_after + suffix

    needle_token_start = prefix_tokens + split_point
    needle_token_end = needle_token_start + needle_tokens

    return prompt, needle_token_start, needle_token_end


# =============================================================================
# forward_to_save（一次性 forward，不用 chunk）
# =============================================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def forward_to_save(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)

    # 一次性 forward，直接做 causal attention
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states_full, value_states_full, is_causal=True)

    # 保存 Q 和 K
    if self.layer_idx in self.layers_to_save:
        query_path = os.path.join(self.save_dir, f"query_layer{self.layer_idx}_{self.target_len}.pkl")
        key_path = os.path.join(self.save_dir, f"key_layer{self.layer_idx}_{self.target_len}.pkl")

        with open(query_path, "wb") as f:
            pickle.dump(query_states.detach().cpu(), f)
        with open(key_path, "wb") as f:
            pickle.dump(key_states_full.detach().cpu(), f)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# =============================================================================
# 模型加载与 patch
# =============================================================================

def load_patched_model(model_path, layers_to_save, target_len, save_dir):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    # 启用 gradient checkpointing 以节省显存
    model.gradient_checkpointing_enable()

    for layer in model.model.layers:
        layer.self_attn.layers_to_save = layers_to_save
        layer.self_attn.target_len = target_len
        layer.self_attn.save_dir = save_dir
        layer.self_attn.forward = forward_to_save.__get__(layer.self_attn)

    return model, tokenizer


# =============================================================================
# 主流程
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_layers = args.layers
    target_lengths = [x * 1024 for x in args.lens]

    haystack_text = args.haystack_text or DEFAULT_HAYSTACK
    needle_text = args.needle_text or DEFAULT_NEEDLE
    question = args.question or DEFAULT_QUESTION

    print(f"目标层: {target_layers}")
    print(f"目标长度: {[f'{x//1024}K={x}' for x in target_lengths]}")
    print(f"针的位置: {args.needle_position}")

    # ---- 对每个目标长度进行处理 ----
    for target_len in target_lengths:
        seq_len = target_len
        len_k = target_len // 1024

        print(f"\n{'='*60}")
        print(f"处理目标长度: {len_k}K ({seq_len} tokens)")
        print(f"{'='*60}")

        # 检查是否已生成
        all_exist = all(
            os.path.exists(os.path.join(args.out_dir, f"query_layer{l}_{seq_len}.pkl"))
            and os.path.exists(os.path.join(args.out_dir, f"key_layer{l}_{seq_len}.pkl"))
            for l in target_layers
        )
        if all_exist:
            print(f"  数据已存在，跳过...")
            continue

        # 清理上一轮可能残留的不完整文件
        for l in target_layers:
            for prefix in ["query_layer", "key_layer"]:
                path = os.path.join(args.out_dir, f"{prefix}{l}_{seq_len}.pkl")
                if os.path.exists(path):
                    os.remove(path)

        # 加载模型（每个长度重新加载，因为 target_len 不同）
        model, tokenizer = load_patched_model(
            model_path=args.model_path,
            layers_to_save=target_layers,
            target_len=seq_len,
            save_dir=args.out_dir,
        )

        # 构建 prompt
        prompt, needle_start, needle_end = build_needle_in_haystack_prompt(
            tokenizer=tokenizer,
            target_length=seq_len,
            haystack_text=haystack_text,
            needle_text=needle_text,
            question=question,
            needle_position=args.needle_position,
        )

        # Token 化并截断到目标长度
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seq_len
        ).input_ids.to("cuda")
        actual_len = input_ids.shape[1]
        print(f"  实际 token 数: {actual_len}")
        print(f"  针的位置: token {needle_start} ~ {needle_end}")

        # 一次性 forward 整个序列，不做 chunk 切分
        print(f"  一次性 forward (seq_len={seq_len})")
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                use_cache=False,
                num_logits_to_keep=1,
            )
        print(f"  forward done")

        # 验证每层数据
        for l in target_layers:
            qp = os.path.join(args.out_dir, f"query_layer{l}_{seq_len}.pkl")
            kp = os.path.join(args.out_dir, f"key_layer{l}_{seq_len}.pkl")
            with open(qp, "rb") as f:
                q = pickle.load(f)
            with open(kp, "rb") as f:
                k = pickle.load(f)
            assert q.shape[-2] == seq_len, f"layer {l} q mismatch: {q.shape}"
            assert k.shape[-2] == seq_len, f"layer {l} k mismatch: {k.shape}"
            print(f"  验证 Layer {l}: q={q.shape}, k={k.shape}")
            del q, k

        # 保存针的元信息
        meta = {
            "model_path": args.model_path,
            "target_length": seq_len,
            "actual_length": actual_len,
            "needle_position": args.needle_position,
            "needle_token_start": needle_start,
            "needle_token_end": needle_end,
            "needle_text": needle_text,
            "question": question,
        }
        meta_path = os.path.join(args.out_dir, f"needle_meta_{seq_len}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 清理
        del model, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ 全部完成，数据保存在 {args.out_dir}/")


if __name__ == "__main__":
    main()
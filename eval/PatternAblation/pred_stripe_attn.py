#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pred_stripe_attn.py

端到端实验脚本：比较三种 attention 模式的生成质量。
  1) Baseline: local window + sink + recent
  2) Baseline + Stripes (top 5%/10%/20% 对角线条纹)
  3) Full attention (上界)

流程（每个样本）：
  Step 1: Full attention prefilling → 截获每层每头的 Q、K
  Step 2: 分析每层每头的 top 对角线条纹
  Step 3: 分别用各种 mask 模式跑 generation（lazy 构建 mask，不预存）
  Step 4: 保存生成结果到 jsonl

使用方式：
  python pred_stripe_attn.py \
      --model llama-3.1-8b-instruct \
      --task 2wikimqa \
      --num_samples 50 \
      --out_dir ./stripe_exp_results \
      --model2path ./config/model2path.json \
      --dataset2prompt ./config/dataset2prompt.json \
      --dataset2maxlen ./config/dataset2maxlen.json
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT_DIR))

import os
import json
import math
import types
import argparse
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, nn
from transformers.cache_utils import Cache
from typing import Optional, Tuple


# =============================================================================
# 参数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stripe attention experiment")
    p.add_argument("--model",        type=str, required=True)
    p.add_argument("--task",         type=str, required=True)
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--max_length",   type=int, default=9000,  help="tokenizer truncation")
    p.add_argument("--max_gen",      type=int, default=256,   help="max new tokens to generate")
    p.add_argument("--out_dir",      type=str, default="./stripe_exp_results")
    p.add_argument("--stripe_fracs", type=str, default="0.05,0.10,0.20",
                   help="comma-separated stripe fractions to test")
    # 稀疏 attention 参数（占序列长度的比例）
    p.add_argument("--local_ratio",  type=float, default=0.02, help="local window ratio")
    p.add_argument("--sink_ratio",   type=float, default=0.01, help="sink token ratio")
    p.add_argument("--recent_ratio", type=float, default=0.01, help="recent token ratio")
    # 路径
    p.add_argument("--model2path",       type=str, default=None)
    p.add_argument("--dataset2prompt",   type=str, default=None)
    p.add_argument("--dataset2maxlen",   type=str, default="./config/dataset2maxlen.json",
                   help="path to dataset2maxlen.json")
    p.add_argument("--data_root",        type=str,
                   default="/root/cjh/pro/resources/datasets/LongBench")
    return p.parse_args()


# =============================================================================
# Step 1: Full attention forward，截获所有层所有头的 Q、K
# =============================================================================

class AllLayerQKCatcher:
    """截获所有层的 Q、K（仅 prefilling 阶段）。"""
    def __init__(self):
        self.qk_dict = {}  # {layer_idx: (q, k)}  q,k shape: [num_heads, L, D]

    def clear(self):
        self.qk_dict.clear()
        gc.collect()


def make_capture_all_forward(catcher):
    """Patch forward：截获每层的 Q、K（RoPE 之后、repeat_kv 之后）。"""

    @torch.no_grad()
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
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
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 截获（仅 prefilling）
        if key_states.shape[2] == query_states.shape[2]:
            catcher.qk_dict[self.layer_idx] = (
                query_states[0].float().cpu(),  # [H, L, D]
                key_states[0].float().cpu(),
            )

        # 标准 attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    return patched_forward


# =============================================================================
# Step 2: 分析条纹 → 得到每层每头的 top 对角线
# =============================================================================

def analyze_all_stripes(catcher, stripe_fracs):
    """
    对所有层所有头分析条纹，逐头 GPU 加速。
    返回: {frac: {layer_idx: {head_idx: set_of_diag_offsets}}}
    """
    result = {frac: {} for frac in stripe_fracs}
    total_layers = len(catcher.qk_dict)

    for layer_idx, (q_all, k_all) in catcher.qk_dict.items():
        num_heads, L, D = q_all.shape
        for frac in stripe_fracs:
            result[frac][layer_idx] = {}

        t0 = time.time()
        for head_idx in range(num_heads):
            print(f"  Layer {layer_idx}/{total_layers-1}, Head {head_idx}/{num_heads-1}, L={L}", end="\r")

            q_h = q_all[head_idx].to("cuda")  # [L, D]
            k_h = k_all[head_idx].to("cuda")
            scores = (q_h @ k_h.T) / math.sqrt(D)  # [L, L] on GPU

            diag_mean = torch.zeros(L, device="cuda", dtype=torch.float32)
            for d in range(L):
                diag_mean[d] = torch.diag(scores, diagonal=-d).mean()

            ranked = torch.argsort(-diag_mean).cpu().numpy()
            del scores, q_h, k_h, diag_mean

            for frac in stripe_fracs:
                K = max(1, int(frac * L))
                result[frac][layer_idx][head_idx] = set(ranked[:K].tolist())

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx}/{total_layers-1}: done ({num_heads} heads, {elapsed:.1f}s)          ")

    torch.cuda.empty_cache()
    return result


# =============================================================================
# Step 3: Lazy masked attention forward（按需构建 mask，不预存，不 OOM）
# =============================================================================

def build_baseline_mask(L, local_w, sink_n, recent_n):
    """构建 baseline mask [L, L] bool tensor（CPU）。"""
    mask = torch.zeros(L, L, dtype=torch.bool)
    # Sink
    mask[:, :sink_n] = True
    # Local window
    rows = torch.arange(L).unsqueeze(1)
    cols = torch.arange(L).unsqueeze(0)
    mask |= (cols >= rows - local_w + 1) & (cols <= rows)
    # Recent
    if recent_n > 0:
        mask[-recent_n:, :] = True
    # Causal
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    return mask & causal


def make_masked_forward_lazy(mode, local_w, sink_n, recent_n, stripe_diags_by_layer=None):
    """
    Lazy 版 masked attention forward。
    - baseline mask 缓存一次（所有层共用）
    - stripe mask 在 prefilling 时按需构建当前层的，用完释放
    - decoding 阶段走标准 causal mask
    """
    # 预先把 set 转成 sorted list
    stripe_lists = {}
    if stripe_diags_by_layer:
        for layer_idx, head_dict in stripe_diags_by_layer.items():
            stripe_lists[layer_idx] = {}
            for h, s in head_dict.items():
                stripe_lists[layer_idx][h] = sorted(s)

    _baseline_cache = {}  # {L: [1,1,L,L] on GPU}

    @torch.no_grad()
    def masked_forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs,
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
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        kv_len = key_states.shape[2]

        if q_len == kv_len and mode != "full":
            # ---- Prefilling 阶段 ----

            # baseline mask：只构建一次，缓存
            if q_len not in _baseline_cache:
                base = build_baseline_mask(q_len, local_w, sink_n, recent_n)
                _baseline_cache[q_len] = base.unsqueeze(0).unsqueeze(0).cuda()  # [1,1,L,L]

            if mode == "baseline":
                attn_weights = attn_weights.masked_fill(~_baseline_cache[q_len], float("-inf"))

            elif mode == "stripe" and self.layer_idx in stripe_lists:
                print(f"    [stripe mask] Layer {self.layer_idx}...", end="\r")
                base_cpu = build_baseline_mask(q_len, local_w, sink_n, recent_n)
                causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))

                num_heads = query_states.shape[1]
                head_masks = []
                for h in range(num_heads):
                    offsets = stripe_lists[self.layer_idx].get(h, [])
                    if offsets:
                        stripe_mask = torch.zeros(q_len, q_len, dtype=torch.bool)
                        for d in offsets:
                            length = q_len - d
                            if length > 0:
                                idx = torch.arange(length)
                                stripe_mask[idx + d, idx] = True
                        head_masks.append((base_cpu | stripe_mask) & causal)
                    else:
                        head_masks.append(base_cpu)

                # 只有当前层的 mask 上 GPU，用完释放
                combined = torch.stack(head_masks).unsqueeze(0).cuda()  # [1, H, L, L]
                attn_weights = attn_weights.masked_fill(~combined, float("-inf"))
                del combined, head_masks, base_cpu, causal
                torch.cuda.empty_cache()
        else:
            # ---- Decoding 阶段 或 Full mode ----
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, :kv_len]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    return masked_forward


# =============================================================================
# 工具函数
# =============================================================================

def patch_model(model, forward_fn):
    """给模型所有 self_attn 层 patch forward。"""
    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            module.forward = types.MethodType(forward_fn, module)


def generate_text(model, tokenizer, input_ids, max_gen=256):
    """简单的 greedy generation。"""
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_gen,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =============================================================================
# 主流程
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stripe_fracs = [float(x) for x in args.stripe_fracs.split(",")]

    # ---- 路径 ----
    if args.model2path:
        model2path_file = args.model2path
    else:
        candidates = [
            str(ROOT_DIR) + "/eval/LongBench/config/model2path.json",
            "./config/model2path.json",
        ]
        model2path_file = next((c for c in candidates if os.path.exists(c)), candidates[0])

    if args.dataset2prompt:
        dataset2prompt_file = args.dataset2prompt
    else:
        candidates = [
            str(ROOT_DIR) + "/eval/LongBench/config/dataset2prompt.json",
            "./config/dataset2prompt.json",
        ]
        dataset2prompt_file = next((c for c in candidates if os.path.exists(c)), candidates[0])

    model2path     = json.load(open(model2path_file))
    dataset2prompt = json.load(open(dataset2prompt_file))


    task_key = args.task
    if task_key not in dataset2prompt:
        for k in sorted(dataset2prompt.keys(), key=len, reverse=True):
            if k in task_key:
                task_key = k
                break
        print(f"  task '{args.task}' 匹配到 prompt key: '{task_key}'")
    # 加载 dataset2maxlen，自动设置 max_gen
    if os.path.exists(args.dataset2maxlen):
        dataset2maxlen = json.load(open(args.dataset2maxlen))
        if task_key in dataset2maxlen:
            args.max_gen = dataset2maxlen[task_key]
            print(f"从 dataset2maxlen.json 读取 max_gen={args.max_gen}")

    # ---- 加载模型 ----
    model_path = model2path[args.model]
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    print(f"模型: {num_layers} layers, {num_heads} heads")

    # ---- 加载数据 ----
    data_path = os.path.join(args.data_root, f"{args.task}.jsonl")
    print(f"数据集: {data_path}")
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))
    print(f"共 {len(samples)} 个样本")

    # ---- 准备输出文件（检查断点续推） ----
    mode_names = ["full", "baseline"] + [f"baseline+stripe_{int(f * 100)}pct" for f in stripe_fracs]

    # 检查已完成的样本数（以 full 文件为准）
    full_fp = os.path.join(args.out_dir, f"{args.task}_full.jsonl")
    done_count = 0
    if os.path.exists(full_fp):
        with open(full_fp, "r", encoding="utf-8") as f:
            done_count = sum(1 for _ in f)
        print(f"🔄 检测到已有结果，已完成 {done_count} 个样本，从第 {done_count} 个继续")

    # 验证所有模式文件行数一致，不一致则截断到最小值
    for mn in mode_names:
        fp = os.path.join(args.out_dir, f"{args.task}_{mn}.jsonl")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
            if n != done_count:
                print(f"⚠️ {mn} 文件有 {n} 行，与 full 的 {done_count} 行不一致，截断到 {min(n, done_count)} 行")
                done_count = min(done_count, n)

    for mn in mode_names:
        fp = os.path.join(args.out_dir, f"{args.task}_{mn}.jsonl")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > done_count:
                with open(fp, "w", encoding="utf-8") as f:
                    f.writelines(lines[:done_count])

    # 以追加模式打开
    writers = {}
    for mn in mode_names:
        fp = os.path.join(args.out_dir, f"{args.task}_{mn}.jsonl")
        writers[mn] = open(fp, "a", encoding="utf-8")
        print(f"  输出: {fp}")

    # ---- 逐样本处理 ----
    for idx, sample in enumerate(samples):
        if idx < done_count:
            continue
        print(f"\n{'='*60}")
        print(f"Sample {idx}/{len(samples)}")
        print(f"{'='*60}")


        task_key = args.task
        if task_key not in dataset2prompt:
            for k in sorted(dataset2prompt.keys(), key=len, reverse=True):
                if k in task_key:
                    task_key = k
                    break
            print(f"  task '{args.task}' 匹配到 prompt key: '{task_key}'")
        prompt = dataset2prompt[task_key].format(**sample)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_length).to("cuda")
        seq_len = inputs.input_ids.shape[-1]
        print(f"seq_len = {seq_len}")


        # 稀疏参数（按序列长度计算）
        local_w  = max(1, int(args.local_ratio  * seq_len))
        sink_n   = max(1, int(args.sink_ratio   * seq_len))
        recent_n = max(1, int(args.recent_ratio  * seq_len))
        print(f"local_w={local_w}, sink_n={sink_n}, recent_n={recent_n}")

        # ======== Step 1: Full attention prefilling → 截获 Q、K ========
        print("Step 1: Full attention prefilling (截获 Q、K)...")
        t0 = time.time()
        catcher = AllLayerQKCatcher()
        capture_fn = make_capture_all_forward(catcher)
        patch_model(model, capture_fn)

        with torch.no_grad():
            model(**inputs, use_cache=False)

        print(f"  截获 {len(catcher.qk_dict)} 层的 Q、K ({time.time()-t0:.1f}s)")

        # ======== Step 2: 分析条纹 ========
        print("Step 2: 分析每层每头的条纹...")
        t0 = time.time()
        stripe_info = analyze_all_stripes(catcher, stripe_fracs)
        catcher.clear()
        torch.cuda.empty_cache()
        print(f"  条纹分析完成 ({time.time()-t0:.1f}s)")

        # ======== Step 3: 各模式生成 ========

        # --- 3a: Full attention ---
        print("Step 3a: Full attention generation...")
        t0 = time.time()
        full_fn = make_masked_forward_lazy("full", local_w, sink_n, recent_n)
        patch_model(model, full_fn)
        pred_full = generate_text(model, tokenizer, inputs.input_ids, args.max_gen)
        print(f"  Full: {pred_full[:80]}... ({time.time()-t0:.1f}s)")
        record = {"pred": pred_full, "answers": sample.get("answers", [sample.get("answer", "")]),
                  "all_classes": sample.get("all_classes", None), "length": seq_len}
        writers["full"].write(json.dumps(record, ensure_ascii=False) + "\n")
        writers["full"].flush()

        # --- 3b: Baseline (local + sink + recent) ---
        print("Step 3b: Baseline generation...")
        t0 = time.time()
        baseline_fn = make_masked_forward_lazy("baseline", local_w, sink_n, recent_n)
        patch_model(model, baseline_fn)
        pred_baseline = generate_text(model, tokenizer, inputs.input_ids, args.max_gen)
        print(f"  Baseline: {pred_baseline[:80]}... ({time.time()-t0:.1f}s)")
        record = {"pred": pred_baseline, "answers": sample.get("answers", [sample.get("answer", "")]),
                  "all_classes": sample.get("all_classes", None), "length": seq_len}
        writers["baseline"].write(json.dumps(record, ensure_ascii=False) + "\n")
        writers["baseline"].flush()

        # --- 3c: Baseline + Stripes ---
        for frac in stripe_fracs:
            frac_name = f"baseline+stripe_{int(frac*100)}pct"
            print(f"Step 3c: {frac_name}...")
            t0 = time.time()

            stripe_fn = make_masked_forward_lazy(
                "stripe", local_w, sink_n, recent_n,
                stripe_diags_by_layer=stripe_info[frac],
            )
            patch_model(model, stripe_fn)
            pred_stripe = generate_text(model, tokenizer, inputs.input_ids, args.max_gen)

            torch.cuda.empty_cache()

            print(f"  {frac_name}: {pred_stripe[:80]}... ({time.time()-t0:.1f}s)")
            record = {"pred": pred_stripe,
                      "answers": sample.get("answers", [sample.get("answer", "")]),
                      "all_classes": sample.get("all_classes", None), "length": seq_len}
            writers[frac_name].write(json.dumps(record, ensure_ascii=False) + "\n")
            writers[frac_name].flush()

        
        # 清理
        torch.cuda.empty_cache()
        gc.collect()

    # ---- 关闭文件 ----
    for w in writers.values():
        w.close()

    print(f"\n✅ 实验完成！结果保存在 {args.out_dir}/")
    print(f"输出文件：")
    for mn in mode_names:
        fp = os.path.join(args.out_dir, f"{args.task}_{mn}.jsonl")
        print(f"  {fp}")


if __name__ == "__main__":
    main()
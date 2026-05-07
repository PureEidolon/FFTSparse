#./eval/LVEval/pred.py
"""
LV-Eval 数据集推理脚本
- 支持所有 LV-Eval 数据集和长度级别（16k/32k/64k/128k/256k）
- 支持 full/myattn/xattn/flex/minference/sparge 方法
- 支持 attention 计时 (--timing)
- 支持 4bit 量化 (--load_4bit)
- 支持断点续跑
"""
import os
import re
import sys
import pdb
import time
import json
import math
import types
import random
import inspect


import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
print("Path(__file__):", Path(__file__))
sys.path.insert(0, str(ROOT_DIR))
print(f"✅ 已把路径 '{ROOT_DIR}' 加入 sys.path")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DynamicCache,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    apply_rotary_pos_emb,
)

from attn_src.xattn.src.Xattention import Xattention_prefill
from attn_src.fft_attn.my_attn_v7 import myattn_prefill
print(inspect.signature(myattn_prefill))
from attn_src.Flexprefill import Flexprefill_prefill
from attn_src.Minference import Minference_prefill
#from attn_src.Sparge import Sparge_prefill
from flash_attn import flash_attn_func
from ratio import max as xattn_max_threshold

# 导入 LV-Eval 配置
from config import DATASET_PROMPT, DATASET_MAXGEN


# ============ 全局计时器 ============
ATTN_TIMES = {}


def reset_attn_times():
    global ATTN_TIMES
    ATTN_TIMES = {}


def get_total_attn_time():
    return sum(ATTN_TIMES.values())

def get_dataset_length(task_name):
    """从 task 名解析长度档位，返回 token 数。
    如 'loogle_CR_mixup_64k' -> 65536
    若解析不到则返回 None
    """
    m = re.search(r'_(\d+)k$', task_name)
    if m is None:
        return None
    return int(m.group(1)) * 1024   # LV-Eval 官方按 1024 计


# ============ 参数解析 ============
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--task", type=str, required=True,
                        help="如 hotpotwikiqa_mixup_32k")
    parser.add_argument("--data_dir", type=str,
                        default="/root/cjh/pro/resources/datasets/LVEval",
                        help="LV-Eval 数据根目录")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")


    # myattn 参数
    parser.add_argument("--use_cor", action="store_true")
    parser.add_argument("--sink_ratio", type=float, default=0.01)
    parser.add_argument("--recent_ratio", type=float, default=0.01)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--local_ratio", type=float, default=0.02)
    parser.add_argument("--corr_thres", type=float, default=1.0)
    parser.add_argument("--enable_last_block", action="store_true")
    parser.add_argument("--last_block_thres", type=float, default=0.01)
    parser.add_argument("--is_visual", action="store_true", help="是否进行可视化")
    parser.add_argument("--corr_selection_mode", type=str, default="threshold",choices=["threshold", "topk"])
    parser.add_argument("--corr_topk_ratio", type=float, default=0.2)
    parser.add_argument("--enable_column_mask", action="store_true", default=True, help="是否启用列重要性mask")
    parser.add_argument("--column_topk_ratio", type=float, default=0.1, help="列重要性mask的topk比例")

    # v7 新增: 列重要性 mask 范围
    parser.add_argument("--column_start_exclude_ratio", type=float, default=0.1)
    parser.add_argument("--column_end_exclude_ratio", type=float, default=0.2)

    # v7 新增: 条带/弥散自适应
    parser.add_argument("--diag_sample_ratio", type=float, default=0.15)
    parser.add_argument("--min_diag_samples", type=int, default=5)
    parser.add_argument("--max_diag_samples", type=int, default=64)
    parser.add_argument("--stripe_threshold", type=float, default=0.3)
    parser.add_argument("--qk_topk_ratio", type=float, default=0.2)



    return parser.parse_args(args)


# ============ 工具函数 ============
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_dataset_base(task_name):
    """去掉长度后缀: hotpotwikiqa_mixup_32k -> hotpotwikiqa_mixup"""
    return re.split(r'_\d+k$', task_name)[0]


def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "qwen" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "qwen" in model_name.lower():
        response = response.split("<|im_end|>")[0].strip()
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    return response


def load_data(data_path, num_samples=-1):
    """加载 LV-Eval jsonl 数据"""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_samples != -1 and i >= num_samples:
                break
            obj = json.loads(line)
            # 字段兼容
            if "answers" not in obj and "answer" in obj:
                obj["answers"] = obj["answer"] if isinstance(obj["answer"], list) else [obj["answer"]]
            if "all_classes" not in obj:
                obj["all_classes"] = None
            if "length" not in obj:
                obj["length"] = 0
            samples.append(obj)
    return samples


def load_model_and_tokenizer(path, model_name, load_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )

    if load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("📦 使用 4bit 量化加载模型")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("📦 使用 bf16 加载模型")

    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)

    mem_bytes = torch.cuda.memory_allocated()
    print(f"📊 模型显存占用: {mem_bytes / 1024**3:.2f} GB")

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    return model, tokenizer, eos_token_ids


# ============ Attention Forward ============
@torch.no_grad()
def new_attention_forward(
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        # ===== prefill 阶段 =====
        if args.timing:
            torch.cuda.synchronize()
            attn_start = time.time()

        if self.method == "myattn":
            attn_output = myattn_prefill(
                query_states,
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                block_size=args.block_size,
                is_causal=True,
                is_visual=args.is_visual,
                # 先验 mask
                sink_ratio=args.sink_ratio,
                recent_ratio=args.recent_ratio,
                local_span_ratio=args.local_ratio,
                # FFT 相关性 mask
                enable_correlation_mask=args.use_cor,
                correlation_selection_mode=args.corr_selection_mode,
                correlation_topk_ratio=args.corr_topk_ratio,
                corr_threshold=args.corr_thres,
                collect_corr_stats=False,
                # 列重要性 mask
                enable_column_mask=args.enable_column_mask,
                column_topk_ratio=args.column_topk_ratio,
                column_start_exclude_ratio=args.column_start_exclude_ratio,
                column_end_exclude_ratio=args.column_end_exclude_ratio,
                # last block mask
                enable_last_block_mask=args.enable_last_block,
                last_block_threshold=args.last_block_thres,
                # 条带/弥散自适应 (v7 新增)
                diag_sample_ratio=args.diag_sample_ratio,
                min_diag_samples=args.min_diag_samples,
                max_diag_samples=args.max_diag_samples,
                stripe_threshold=args.stripe_threshold,
                qk_topk_ratio=args.qk_topk_ratio,
                # 可视化
                attention_vis_heads="0",
                attention_vis_dir='./vis_attn/output',
            )
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        elif self.method == "xattn":
            self.threshold = self.threshold.to(key_states.device)
            attn_output = Xattention_prefill(
                query_states, key_states, value_states,
                norm=1, stride=8,
                threshold=self.threshold,
                use_triton=True,
                keep_sink=True,
                keep_recent=True,
            )

        elif self.method == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=0.9, tau=0.1,
            ).transpose(1, 2)

        elif self.method == "minference":
            attn_output = Minference_prefill(query_states, key_states, value_states)

        #elif self.method == "sparge":
        #    attn_output = Sparge_prefill(
        #        query_states, key_states, value_states,
        #        topk=0.5, is_causal=False,
        #    )

        elif self.method == "full":
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)

        if args.timing:
            torch.cuda.synchronize()
            attn_elapsed = time.time() - attn_start
            global ATTN_TIMES
            if self.layer_idx not in ATTN_TIMES:
                ATTN_TIMES[self.layer_idx] = 0.0
            ATTN_TIMES[self.layer_idx] += attn_elapsed

        '''
        if self.layer_idx == 31:
            print(" == ATTN_TIMES:\n",ATTN_TIMES)
            pdb.set_trace()
        '''
        
    else:
        # ===== decode 阶段 =====
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ============ 推理主函数 ============
def get_pred(
    model, tokenizer, eos_token_ids,
    data, max_length, max_gen,
    prompt_format, model_name, out_path,
):
    # 断点续跑
    existing_preds = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_preds.append(json.loads(line))
        start_idx = len(existing_preds)
        print(f"📂 读取已有结果 {start_idx} 条，从第 {start_idx} 个样本继续")
    else:
        start_idx = 0
        print(f"📂 未找到已有结果，从第 0 个样本开始")

    preds = []
    pbar = tqdm(data)

    for idx, json_obj in enumerate(pbar):
        if idx < start_idx:
            continue

        prompt = prompt_format.format(**json_obj)

        # 中间截断（保留首尾）
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True
            )

        prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")
        seq_len = input.input_ids.shape[-1]
        pbar.set_description(f"样本 {idx}, len={seq_len}")

        # 清零计时器
        reset_attn_times()

        with torch.no_grad():
            past_key_values = DynamicCache()

            torch.cuda.synchronize()
            prefill_start = time.time()


            output = model(
                input_ids=input.input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )

            torch.cuda.synchronize()
            prefill_elapsed = time.time() - prefill_start

            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]

            for _ in tqdm(range(max_gen - 1), desc="Decoding", leave=False):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

            pred = tokenizer.decode(generated_content, skip_special_tokens=True)
            pred = post_process(pred, model_name)



        # 打印
        reference = json_obj["answers"]
        if isinstance(reference, list) and reference:
            ref_str = reference[0][:100]
        else:
            ref_str = "(无标准答案)"
        print(f"\n🔍Reference: {ref_str}")
        print(f"🤖Prediction: {pred[:100]}")

        # 构建结果
        pred_item = {
            "pred": pred,
            "answers": json_obj["answers"],
            "gold_ans": json_obj.get("answer_keywords", None),
            "input": json_obj["input"],
            "all_classes": json_obj.get("all_classes", None),
            "length": json_obj.get("length", 0),
        }

        # 计时统计
        if args.timing:
            total_attn_time = get_total_attn_time()
            pred_item["input_len"] = seq_len
            pred_item["prefill_ms"] = round(prefill_elapsed * 1000, 2)
            pred_item["attn_ms"] = round(total_attn_time * 1000, 2)
            attn_pct = total_attn_time / prefill_elapsed * 100 if prefill_elapsed > 0 else 0
            print(f"⏱️  样本 {idx} (len={seq_len}): "
                  f"prefill={prefill_elapsed * 1000:.2f}ms, "
                  f"attn={total_attn_time * 1000:.2f}ms, "
                  f"attn占比={attn_pct:.1f}%")

        preds.append(pred_item)

        # 立刻追加写入
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(pred_item, f, ensure_ascii=False)
            f.write("\n")

        # 释放显存
        try:
            del past_key_values, output, pred_token_idx, input
            if 'outputs' in locals():
                del outputs
        except:
            pass
        torch.cuda.empty_cache()

    # 汇总
    all_preds = existing_preds + preds

    if args.timing:
        attn_times = [p["attn_ms"] for p in all_preds if "attn_ms" in p and p["attn_ms"] > 0]
        prefill_times = [p["prefill_ms"] for p in all_preds if "prefill_ms" in p and p["prefill_ms"] > 0]
        input_lens = [p["input_len"] for p in all_preds if "input_len" in p]

        if attn_times:
            print(f"\n{'=' * 60}")
            print(f"TIMING SUMMARY ({len(attn_times)} samples)")
            print(f"{'=' * 60}")
            print(f"  avg input_len:   {np.mean(input_lens):.0f}")
            print(f"  avg prefill_ms:  {np.mean(prefill_times):.2f}")
            print(f"  avg attn_ms:     {np.mean(attn_times):.2f}")
            print(f"  avg attn ratio:  {np.mean(attn_times) / np.mean(prefill_times) * 100:.1f}%")
            print(f"{'=' * 60}")

    return all_preds


# ============ Main ============
if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    print("=" * 60)
    print("LV-Eval 参数配置")
    print("=" * 60)
    print(f"  模型:              {args.model}")
    print(f"  方法:              {args.method}")
    print(f"  任务:              {args.task}")
    print(f"  数据目录:          {args.data_dir}")
    print(f"  样本数:            {args.num_samples}")
    print(f"  计时:              {args.timing}")
    print(f"  4bit量化:          {args.load_4bit}")
    if args.method == "myattn":
        print(f"  use_cor:           {args.use_cor}")
        print(f"  sink_ratio:        {args.sink_ratio}")
        print(f"  recent_ratio:      {args.recent_ratio}")
        print(f"  local_ratio:       {args.local_ratio}")
        print(f"  block_size:        {args.block_size}")
        print(f"  corr_thres:        {args.corr_thres}")
        print(f"  corr_sel_mode:     {args.corr_selection_mode}")
        print(f"  corr_topk_ratio:   {args.corr_topk_ratio}")
        print(f"  enable_last_block: {args.enable_last_block}")
        print(f"  last_block_thres:  {args.last_block_thres}")
        print(f"  {'enable_column_mask':<20}: {args.enable_column_mask}")
        print(f"  {'column_topk_ratio':<20}: {args.column_topk_ratio}")
        print(f"  {'stripe_threshold':<20}: {args.stripe_threshold}")
        print(f"  {'qk_topk_ratio':<20}: {args.qk_topk_ratio}")
        print(f"  {'diag_sample_ratio':<20}: {args.diag_sample_ratio}")
        print(f"  {'min/max_diag_samples':<20}: {args.min_diag_samples}/{args.max_diag_samples}")
        print(f"  {'col_exclude_ratio':<20}: {args.column_start_exclude_ratio}/{args.column_end_exclude_ratio}")
    print("=" * 60)

    # 模型路径
    model2path = json.load(open("eval/LVEval/model_config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LVEval/model_config/model2maxlen.json", "r"))

    model_name = args.model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name, load_4bit=args.load_4bit
    )

    # Monkey-patch attention
    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            layer_idx = int(name.split(".")[2])
            module.method = args.method
            if args.method == "xattn":
                module.threshold = torch.tensor(xattn_max_threshold[layer_idx])
            module.forward = types.MethodType(new_attention_forward, module)

    # 解析 task 名
    task = args.task
    dataset_base = get_dataset_base(task)
    prompt_format = DATASET_PROMPT[dataset_base]
    max_gen = DATASET_MAXGEN[dataset_base]

    # 按数据集长度档位限制输入
    task_max_len = get_dataset_length(task)
    model_max_len = model2maxlen[model_name]

    if task_max_len is not None:
        # 取两者较小值，且给 chat template + max_gen 留余量
        max_length = min(task_max_len, model_max_len) - max_gen - 64
        print(f"📏 任务长度档位: {task_max_len} tokens, "
              f"模型上限: {model_max_len}, "
              f"最终 max_length: {max_length}")
    else:
        max_length = model_max_len
        print(f"3📏 未识别长度档位，使用模型上限: {max_length}")

    prompt_format = DATASET_PROMPT[dataset_base]
    max_gen = DATASET_MAXGEN[dataset_base]

    # 数据路径: data_dir/hotpotwikiqa_mixup/hotpotwikiqa_mixup_32k.jsonl
    data_path = os.path.join(args.data_dir, dataset_base, f"{task}.jsonl")
    print(f" == DATA_PATH: {data_path}")

    data = load_data(data_path, num_samples=args.num_samples)

    # 输出目录
    pred_dir = f"eval/LVEval/pred/{model_name}"
    os.makedirs(pred_dir, exist_ok=True)

    # 输出文件名
    if args.method == "full":
        out_path = f"{pred_dir}/{task}-full.jsonl"
    elif args.method == "myattn":
        if args.use_cor:
            param_list = [
                args.sink_ratio, args.recent_ratio, args.local_ratio,
                "bs-" + str(args.block_size)
            ]
            if args.corr_selection_mode == "threshold":
                param_list.append(f"thres-{args.corr_thres}")
            else:
                param_list.append(f"topk-{args.corr_topk_ratio}")
            if args.enable_last_block:
                param_list.append(f"lb-{args.last_block_thres}")
            if args.enable_column_mask:
                param_list.append(f"col-{args.column_topk_ratio}")
            # v7 新增
            param_list.append(f"st-{args.stripe_threshold}")
            param_list.append(f"qk-{args.qk_topk_ratio}")
            param_str = ", ".join(map(str, param_list))
        else:
            param_list = [args.sink_ratio, args.recent_ratio, args.local_ratio]
            if args.enable_last_block:
                param_list.append(f"lb-{args.last_block_thres}")
            if args.enable_column_mask:
                param_list.append(f"col-{args.column_topk_ratio}")
            param_str = ", ".join(map(str, param_list))
        out_path = f"eval/LVEval/pred/{model_name}/{task}-myattn_v7-[ {param_str} ].jsonl"
    elif args.method == "xattn":
        out_path = f"{pred_dir}/{task}-xattn-stride=8.jsonl"
    elif args.method == "flex":
        out_path = f"{pred_dir}/{task}-flex.jsonl"
    elif args.method == "minference":
        out_path = f"{pred_dir}/{task}-minference.jsonl"
    elif args.method == "sparge":
        out_path = f"{pred_dir}/{task}-sparge.jsonl"

    print(f" == OUT_PATH: {out_path}")

    preds = get_pred(
        model, tokenizer, eos_token_ids,
        data, max_length, max_gen,
        prompt_format, model_name, out_path,
    )
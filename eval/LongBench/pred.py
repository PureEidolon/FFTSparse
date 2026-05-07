#./eval/LongBench/pred.py
import os
import sys
import time
import pdb
# == 第一步：先加项目根目录到 sys.path ======================================
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 0是父目录，1是爷目录
print("Path(__file__):",Path(__file__))
sys.path.insert(0, str(ROOT_DIR))
print(f"✅ 已把路径 '{ROOT_DIR}' 加入 sys.path")
# =======================================================================


from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DynamicCache,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    apply_rotary_pos_emb,
    nn,
)
import math
from attn_src.xattn.src.Xattention import Xattention_prefill
from attn_src.fft_attn.my_attn_v7 import myattn_prefill
from attn_src.Flexprefill import Flexprefill_prefill
from attn_src.Minference import Minference_prefill
#from attn_src.Sparge import Sparge_prefill

from flash_attn import flash_attn_func
import types
from ratio import max_ratio, max


# ============ 全局计时器 ============
ATTN_TIMES = {}        # {layer_idx: 累计 attn 时间}
PREFILL_TIME = 0.0     # 当前样本的 prefill 总时间


def reset_attn_times():
    """清零所有层的 attention 计时"""
    global ATTN_TIMES, PREFILL_TIME
    ATTN_TIMES = {}
    PREFILL_TIME = 0.0


def get_total_attn_time():
    """获取所有层 attention 时间之和（秒）"""
    return sum(ATTN_TIMES.values())


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument("--method",type=str,default="full",)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--tasks", type=str, nargs="+", help="task names", required=True)
    parser.add_argument("--num_samples", type=int, help="samles", required=True)
    parser.add_argument("--timing", action="store_true", help="是否统计 attention 计时")
    parser.add_argument("--is_visual", action="store_true", help="是否进行可视化")



    parser.add_argument("--use_cor", action="store_true", help="是否使用相关性计算")
    parser.add_argument("--sink_ratio", type=float, default=0.01)
    parser.add_argument("--recent_ratio", type=float, default=0.01)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--local_ratio", type=float, default=0.02)
    parser.add_argument("--corr_thres", type=float, default=1.0)

    parser.add_argument('--corr_selection_mode', type=str, default='threshold',choices=['threshold', 'topk'],help='相关性 mask 选择模式：threshold 或 topk')
    parser.add_argument('--corr_topk_ratio', type=float, default=0.2,help='topk 模式下保留的远程延迟比例')

    parser.add_argument("--enable_column_mask", action="store_true", help="是否启用列重要性mask")
    parser.add_argument("--column_topk_ratio", type=float, default=0.1, help="列重要性mask的topk比例")

    parser.add_argument("--enable_last_block", action="store_true", help="是否保留最后一个query对key的重要块")
    parser.add_argument("--last_block_thres", type=float, default=0.01, help="最后一个block的阈值")

    # pred.py
    parser.add_argument("--diag_sample_ratio", type=float, default=0.15,help="条带判断时对角线（tau方向）的采样比例，相对于可用远程对角线总数")
    parser.add_argument("--min_diag_samples", type=int, default=5,help="对角线采样数下限")
    parser.add_argument("--max_diag_samples", type=int, default=64,help="对角线采样数上限")
    parser.add_argument("--stripe_threshold", type=float, default=0.3, help="用条纹进行模式判断的阈值")
    parser.add_argument("--qk_topk_ratio", type=float, default=0.3, help="条纹模式下每行的采样比例")

    parser.add_argument("--load_4bit", action="store_true", help="是否使用4bit量化加载模型")
    parser.add_argument("--data_dir", type=str, default=None, help="数据集目录，默认用LongBench路径")

    return parser.parse_args(args)


# This is the customized building prompt for chat models
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


def load_longbench_data(path: str, num_samples: int = -1):
    """加载 LongBench 格式的 .jsonl 数据，-1表示全部"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_samples != -1 and i >= num_samples:
                break
            obj = json.loads(line)

            # InfiniteBench 字段适配
            if "answers" not in obj and "answer" in obj:
                obj["answers"] = obj["answer"] if isinstance(obj["answer"], list) else [obj["answer"]]
            if "all_classes" not in obj:
                obj["all_classes"] = obj.get("options", None)
            if "length" not in obj:
                obj["length"] = obj.get("token_length", 0)

            samples.append(obj)
    return samples


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
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        # ===== prefill 阶段：计时 =====
        if args.timing:
            torch.cuda.synchronize()
            attn_start = time.time()

        if self.method == "myattn":
            if self.layer_idx == 0: print("执行 MyAttn （prefilling 阶段）")

            attn_output = myattn_prefill(
                query_states,
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                block_size=args.block_size,
                is_causal=True,
                sink_ratio=args.sink_ratio,
                recent_ratio=args.recent_ratio,
                local_span_ratio=args.local_ratio,

                enable_correlation_mask=args.use_cor,
                correlation_selection_mode=args.corr_selection_mode,
                correlation_topk_ratio=args.corr_topk_ratio,
                corr_threshold=args.corr_thres,
                collect_corr_stats=False,

                enable_column_mask=args.enable_column_mask,
                column_topk_ratio=args.column_topk_ratio,

                enable_last_block_mask=args.enable_last_block,
                last_block_threshold=args.last_block_thres,

                diag_sample_ratio=args.diag_sample_ratio,
                min_diag_samples=args.min_diag_samples,
                max_diag_samples=args.max_diag_samples,

                stripe_threshold=args.effective_stripe_threshold,
                qk_topk_ratio=args.effective_qk_topk_ratio,

                is_visual=args.is_visual,
                attention_vis_heads="0",
                attention_vis_dir='./vis_attn/output',
            )
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        elif self.method == "xattn":
            self.threshold = self.threshold.to(key_states.device)
            threshold = self.threshold
            attn_output = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                norm=1,
                stride=8,
                threshold=threshold,
                use_triton=True,
                keep_sink=True,
                keep_recent=True,
            )

        elif self.method == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=0.9,
                tau=0.1,
            ).transpose(1, 2)

        elif self.method == "minference":
            attn_output = Minference_prefill(query_states, key_states, value_states)

        #elif self.method == "sparge":
        #    attn_output = Sparge_prefill(
        #        query_states,
        #        key_states,
        #        value_states,
        #        topk=0.5,
        #        is_causal=False,
        #    )

        elif self.method == "full":
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)

        # ===== 计时结束 =====
        if args.timing:
            torch.cuda.synchronize()
            attn_elapsed = time.time() - attn_start
            global ATTN_TIMES
            if self.layer_idx not in ATTN_TIMES:
                ATTN_TIMES[self.layer_idx] = 0.0
            ATTN_TIMES[self.layer_idx] += attn_elapsed

    else:
        # ===== decode 阶段：不计时 =====
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
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    out_path,
):
    # ============================================================
    # 断点续跑：读取已有结果
    # ============================================================
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
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
        ]:
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")
        seq_len = input.input_ids.shape[-1]
        pbar.set_description(f"Generating for {idx}, len = {seq_len}")

        if seq_len >= 45_000:
            print("input 过长，截断至 45_000")
            input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=45_000).to("cuda")
            seq_len = input.input_ids.shape[-1]

        # ===== 清零计时器 =====
        reset_attn_times()




        with torch.no_grad():
            past_key_values = DynamicCache()

            # ===== Prefill 计时 =====
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


            # ===== 打印 =====
            reference = json_obj["answers"]
            print(f"\n🔍Reference: {reference[0][:100]}")
            print(f"🤖Prediction: {pred[:100]}")

            # ===== 构建结果 =====
            pred_item = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }

            # ===== 计时统计（可选） =====
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

            # ===== 立刻追加写入文件 =====
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(pred_item, f, ensure_ascii=False)
                f.write("\n")

            # ===== 释放显存 =====
            del past_key_values, output, pred_token_idx, input
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()

    # ============================================================
    # 推理结束，打印汇总统计
    # ============================================================
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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


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

    # 打印模型显存占用
    mem_bytes = torch.cuda.memory_allocated()
    print(f"📊 模型显存占用: {mem_bytes / 1024**3:.2f} GB")

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    return model, tokenizer, eos_token_ids




# ===== 任务类型判断（基于模型） =====

RETRIEVAL_TASK_CACHE = {}

# 明确属于检索类的任务类型
RETRIEVAL_TYPES = {"retrieval", "counting"}

def is_retrieval_task(model, tokenizer, dataset_name, prompt_format):
    if dataset_name in RETRIEVAL_TASK_CACHE:
        return RETRIEVAL_TASK_CACHE[dataset_name]

    classify_prompt = f"""You are an NLP task classifier. Classify the following task into exactly one of these categories:
                        - retrieval: finding a specific passage or document from many candidates
                        - counting: counting occurrences or unique items
                        - multi_hop_qa: answering questions that require reasoning across multiple passages
                        - extractive_qa: answering questions by extracting from a given passage
                        - summarization: condensing long text into a shorter summary
                        - code_completion: completing or continuing code
                        - classification: assigning a label or category to text
                        
                        Task name: {dataset_name}
                        Task prompt template: {prompt_format[:300]}
                        
                        Answer with exactly one word from the list above.
                        Answer:"""

    inputs = tokenizer(classify_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
        )
    generated = tokenizer.decode(
        output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    ).strip().lower()

    # 模糊匹配
    task_type = "other"
    for t in ["retrieval", "counting", "multi_hop_qa", "extractive_qa",
              "summarization", "code_completion", "classification"]:
        if t in generated:
            task_type = t
            break

    result = task_type in RETRIEVAL_TYPES
    RETRIEVAL_TASK_CACHE[dataset_name] = result
    print(f"📋 任务类型判断: {dataset_name} => {task_type} => {'检索类，调整参数' if result else '非检索类，保持原始参数'} (模型输出: '{generated}')")
    return result





if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # 打印参数配置
    print("=" * 60)
    print("参数配置")
    print("=" * 60)
    print(f"  {'模型':<20}: {args.model}")
    print(f"  {'方法':<20}: {args.method}")
    print(f"  {'任务':<20}: {args.tasks}")
    print(f"  {'样本数':<20}: {args.num_samples}")
    print(f"  {'LongBench-E':<20}: {args.e}")
    print("-" * 60)
    print("Attention 参数:")
    print(f"  {'use_cor':<20}: {args.use_cor}")
    print(f"  {'sink_ratio':<20}: {args.sink_ratio}")
    print(f"  {'recent_ratio':<20}: {args.recent_ratio}")
    print(f"  {'local_ratio':<20}: {args.local_ratio}")
    print(f"  {'block_size':<20}: {args.block_size}")
    print(f"  {'corr_thres':<20}: {args.corr_thres}")
    print("最后一个QueryBlock 参数-2:")
    print(f"  {'enable_last_block':<20}: {args.enable_last_block}")
    print(f"  {'last_block_thres':<20}: {args.last_block_thres}")
    print("-" * 60)
    print("列采样 参数-3:")
    print(f"  {'enable_column_mask':<20}: {args.enable_column_mask}")
    print(f"  {'column_topk_ratio':<20}: {args.column_topk_ratio}")
    print("条纹 参数:")
    print(f"  {'diag_sample_ratio':<20}: {args.diag_sample_ratio}")
    print(f"  {'min_diag_samples':<20}: {args.min_diag_samples}")
    print(f"  {'max_diag_samples':<20}: {args.max_diag_samples}")
    print(f"  {'stripe_threshold':<20}: {args.stripe_threshold}")
    print(f"  {'qk_topk_ratio':<20}: {args.qk_topk_ratio}")
    print("-" * 60)




    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name, load_4bit=args.load_4bit
    )

    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            layer_idx = int(name.split(".")[2])
            module.method = args.method
            if args.method == "xattn":
                module.threshold = torch.tensor(max[layer_idx])
            module.forward = types.MethodType(new_attention_forward, module)

    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "gov_report", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
        ]
    else:
        datasets = args.tasks  # 直接用传入的列表

    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))




    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")

    for dataset in datasets:
        if args.data_dir:
            DATA_PATH = os.path.join(args.data_dir, dataset + ".jsonl")
        else:
            DATA_PATH = "/root/cjh/pro/resources/datasets/LongBench/" + dataset + ".jsonl"

        print("\n" + "==" * 30)
        print(f"🚀  当前数据集: [{dataset}]  ({datasets.index(dataset) + 1}/{len(datasets)})")
        print(" 🚀  数据集路径:", DATA_PATH)
        print("==" * 30 + "\n")


        data = load_longbench_data(DATA_PATH, num_samples=args.num_samples)
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")

        if args.method == "full":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-full.jsonl"
        elif args.method == "myattn":
            # 1-4: 空间先验参数
            param_list = [
                f"{args.sink_ratio}",
                f"{args.recent_ratio}",
                f"{args.local_ratio}",
                f"{args.last_block_thres if args.enable_last_block else 'OFF'}"
            ]

            # 判断是否为 "no_dynamic" (纯静态模式)
            if not args.use_cor:
                # 如果关闭了动态掩码，加上醒目的标识，并跳过无关的动态参数
                param_list.append("Dynamic-OFF")
                # 列掩码 (Col) 在静态模式下依然起作用，所以保留它的记录
                param_list.append(f"ColR-{args.column_topk_ratio if args.enable_column_mask else 'OFF'}")
            else:
                # ===== 正常的动态模式 =====
                # 5: 相关性选择模式 (Threshold 或 Topk)
                if args.corr_selection_mode == "threshold":
                    param_list.append(f"Thres-{args.corr_thres}")
                else:
                    param_list.append(f"FFTtopk-{args.corr_topk_ratio}")

                # 6-9: 列掩码、采样与条纹参数
                param_list.append(f"ColR-{args.column_topk_ratio if args.enable_column_mask else 'OFF'}")
                param_list.append(f"SR-{args.diag_sample_ratio}")
                param_list.append(f"ST-{args.stripe_threshold}")
                param_list.append(f"QKtopk-{args.qk_topk_ratio}")

            # 组合成最终字符串
            param_str = ", ".join(map(str, param_list))
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-myattn_v7-[ {param_str} ].jsonl"
            
        elif args.method == "xattn":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-xattn-stride=8.jsonl"
        elif args.method == "flex":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-flex.jsonl"
        elif args.method == "minference":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-minference.jsonl"
        elif args.method == "sparge":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-sparge.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        # 先用默认值初始化
        args.effective_stripe_threshold = args.stripe_threshold
        args.effective_qk_topk_ratio = args.qk_topk_ratio


        ''' 
        task_is_retrieval = is_retrieval_task(model, tokenizer, dataset, prompt_format)
        if task_is_retrieval:
            args.effective_stripe_threshold = args.stripe_threshold * 0.5
            args.effective_qk_topk_ratio = min(1.0, args.qk_topk_ratio * 2)
            print(f"🔍 检索任务，stripe_threshold {args.stripe_threshold} => {args.effective_stripe_threshold}, "
                  f"qk_topk_ratio {args.qk_topk_ratio} => {args.effective_qk_topk_ratio}")
        else:
            print(f"✍️  非检索任务，保持原始参数")
        '''

        preds = get_pred(
            model, tokenizer, eos_token_ids,
            data, max_length, max_gen,
            prompt_format, dataset, model_name,
            out_path,
        )
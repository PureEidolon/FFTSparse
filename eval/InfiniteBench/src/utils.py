# utils.py
import torch
import numpy as np
import random
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def seed_everything(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_infinitebench_data(path: str, num_samples: int = -1) -> List[Dict]:
    """加载数据"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_samples != -1 and i >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def build_prompt(template: str, data: Dict, task: str) -> str:
    """构建 prompt"""
    template_vars = {}

    if "context" in data:
        template_vars["context"] = data["context"]
    if "input" in data:
        template_vars["input"] = data["input"]

    if task in ["longbook_qa_eng", "longbook_qa_chn", "longbook_choice_eng"]:
        if "question" in data:
            template_vars["question"] = data["question"]
        elif "input" in data:
            template_vars["question"] = data["input"]

    if task in ["longbook_choice_eng", "code_debug"]:
        if "options" in data and len(data["options"]) >= 4:
            template_vars["OPTION_A"] = data["options"][0]
            template_vars["OPTION_B"] = data["options"][1]
            template_vars["OPTION_C"] = data["options"][2]
            template_vars["OPTION_D"] = data["options"][3]

    if task == "math_find":
        template_vars["prefix"] = data.get("prefix", "")

    if task == "code_run":
        if "func" in data:
            template_vars["func"] = data["func"]
        if "func_call" in data:
            template_vars["func_call"] = data["func_call"]

    return template.format(**template_vars)


def build_chat(tokenizer, prompt: str, model_name: str) -> str:
    """构建聊天格式"""
    model_name_lower = model_name.lower()
    if "llama-2" in model_name_lower:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response: str, model_name: str) -> str:
    """后处理输出"""
    model_name_lower = model_name.lower()

    if "xgen" in model_name_lower:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name_lower:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "llama-2" in model_name_lower:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )

    return response.strip()


def load_model_and_tokenizer(path: str, model_name: str, use_4bit=False, use_8bit=False, model_gpu=None):
    """加载模型和分词器"""
    print(f"📦 正在加载模型: {model_name}")
    print(f"   路径: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    quantization_config = None
    if use_4bit:
        print("🔧 使用 4-bit 量化")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        print("🔧 使用 8-bit 量化")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    if model_gpu is not None:
        device_map = f"cuda:{model_gpu}"
    else:
        device_map = "cuda:0" if (use_4bit or use_8bit) else "auto"

    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    # 禁用 causal mask 生成
    import types
    def _update_causal_mask_dummy(self, attention_mask, input_tensor, cache_position,
                                  past_key_values, output_attentions):
        return None

    model.model._update_causal_mask = types.MethodType(_update_causal_mask_dummy, model.model)

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    print(f"✅ 模型加载完成")

    return model, tokenizer, eos_token_ids


def save_csv(preds, csv_path):
    """单独保存 CSV 计时数据"""
    import csv

    with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
        fieldnames = ['sample_id', 'input_length', 'prefill_time_s',
                      'prefill_throughput_tokens_per_s', 'decode_time_s',
                      'generated_tokens', 'decode_throughput_tokens_per_s',
                      'total_time_s', 'prediction', 'reference']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pred in preds:
            prefill_throughput = pred['input_length'] / pred['prefill_time'] if pred['prefill_time'] > 0 else 0
            decode_throughput = pred['generated_tokens'] / pred['decode_time'] if pred['decode_time'] > 0 else 0
            reference = pred['answer'][0] if pred['answer'] else 'N/A'

            writer.writerow({
                'sample_id': pred['id'],
                'input_length': pred['input_length'],
                'prefill_time_s': f"{pred['prefill_time']:.4f}",
                'prefill_throughput_tokens_per_s': f"{prefill_throughput:.2f}",
                'decode_time_s': f"{pred['decode_time']:.4f}",
                'generated_tokens': pred['generated_tokens'],
                'decode_throughput_tokens_per_s': f"{decode_throughput:.2f}",
                'total_time_s': f"{pred['prefill_time'] + pred['decode_time']:.4f}",
                'prediction': pred['pred'][:100],
                'reference': reference[:100] if isinstance(reference, str) else str(reference)[:100]
            })

    print(f"📊 计时数据已保存到: {csv_path}")

def save_results(preds, out_path, csv_path, task, method):
    """保存结果到 JSONL 和 CSV"""
    # 保存 JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ 预测完成，结果已保存到: {out_path}")

    # 复用 save_csv
    save_csv(preds, csv_path)
    



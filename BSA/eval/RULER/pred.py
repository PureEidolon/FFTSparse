# eval/RULER/pred.py
import os, sys, json, types, math, argparse, importlib
from pathlib import Path
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, nn

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from attn_src.fft_attn.my_attn_v7 import myattn_prefill
from attn_src.xattn.src.Xattention import Xattention_prefill
from attn_src.Flexprefill import Flexprefill_prefill
from attn_src.Minference import Minference_prefill
from flash_attn import flash_attn_func
from ratio import max as xattn_max



def parse_args():
    p = argparse.ArgumentParser()
    # 路径与任务
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True, help="<seq_len>/data/")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="synthetic")
    p.add_argument("--subset", type=str, default="validation")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--ruler_scripts", type=str,default="/backup01/cjh/projects/resources/datasets/RULER/scripts")

    # 方法
    p.add_argument("--method", type=str, default="full")

    # myattn 参数(对齐 LongBench)
    p.add_argument("--sink_ratio", type=float, default=0.01)
    p.add_argument("--recent_ratio", type=float, default=0.01)
    p.add_argument("--local_ratio", type=float, default=0.02)
    p.add_argument("--block_size", type=int, default=128)

    p.add_argument("--use_cor", action="store_true")
    p.add_argument("--corr_selection_mode", type=str, default="threshold", choices=["threshold", "topk"])
    p.add_argument("--corr_thres", type=float, default=1.0)
    p.add_argument("--corr_topk_ratio", type=float, default=0.2)

    p.add_argument("--enable_column_mask", action="store_true", default=True)
    p.add_argument("--column_topk_ratio", type=float, default=0.1)

    p.add_argument("--enable_last_block", action="store_true")
    p.add_argument("--last_block_thres", type=float, default=0.01)

    p.add_argument("--diag_sample_ratio", type=float, default=0.15)
    p.add_argument("--min_diag_samples", type=int, default=5)
    p.add_argument("--max_diag_samples", type=int, default=64)
    p.add_argument("--stripe_threshold", type=float, default=0.3)
    p.add_argument("--qk_topk_ratio", type=float, default=0.3)
    p.add_argument("--load_4bit", action="store_true")

    return p.parse_args()


@torch.no_grad()
def new_attention_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, output_attentions=False, use_cache=False,
    cache_position=None, position_embeddings=None, **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    if k.shape[2] == q.shape[2]:  # prefill
        m = self.method
        if m == "full":
            o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True).transpose(1, 2)
        elif m == "myattn":
            o = myattn_prefill(
                q, k, v,
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
                stripe_threshold=args.stripe_threshold,
                qk_topk_ratio=args.qk_topk_ratio,
                is_visual=False,
            ).reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        elif m == "xattn":
            o = Xattention_prefill(q, k, v, norm=1, stride=8,
                                   threshold=self.threshold.to(k.device),
                                   use_triton=True, keep_sink=True, keep_recent=True)
        elif m == "flex":
            o = Flexprefill_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                    gamma=0.9, tau=0.1).transpose(1, 2)
        elif m == "minference":
            o = Minference_prefill(q, k, v)
        else:
            raise ValueError(f"unknown method: {m}")
    else:  # decode
        w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, :k.shape[-2]]
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, v)

    o = o.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    return self.o_proj(o), None, past_key_value


def load_model(path, method, load_4bit=False):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    kwargs = dict(
        trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="auto", attn_implementation="eager",
    )
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("📦 4bit 加载")
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

    for name, mod in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            mod.method = method
            if method == "xattn":
                from ratio import max as xattn_max
                layer_idx = int(name.split(".")[2])
                mod.threshold = torch.tensor(xattn_max[layer_idx])
            mod.forward = types.MethodType(new_attention_forward, mod)
    model.eval()

    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    return model, tok, eos


def load_data(data_dir, task, subset, num_samples):
    path = Path(data_dir) / task / f"{subset}.jsonl"
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if num_samples != -1 and i >= num_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def get_task_config(task, benchmark, ruler_scripts):
    sys.path.insert(0, ruler_scripts)
    base = importlib.import_module(f"data.{benchmark}.constants").TASKS
    with open(os.path.join(ruler_scripts, f"{benchmark}.yaml")) as f:
        custom = yaml.safe_load(f)
    if task not in custom:
        raise ValueError(f"{task} not in {benchmark}.yaml")
    cfg = custom[task]
    cfg.update(base[cfg["task"]])
    return cfg


def main():
    global args
    args = parse_args()

    print("=" * 60)
    print(f"task={args.task}  method={args.method}")
    print(f"data_dir={args.data_dir}")
    print(f"save_dir={args.save_dir}")
    if args.method == "myattn":
        print(f"  sink={args.sink_ratio} recent={args.recent_ratio} local={args.local_ratio} bs={args.block_size}")
        print(f"  use_cor={args.use_cor} mode={args.corr_selection_mode} thres={args.corr_thres} topk={args.corr_topk_ratio}")
        print(f"  col_mask={args.enable_column_mask} col_topk={args.column_topk_ratio}")
        print(f"  last_block={args.enable_last_block} lb_thres={args.last_block_thres}")
        print(f"  stripe={args.stripe_threshold} qk_topk={args.qk_topk_ratio} diag={args.diag_sample_ratio}")
    print("=" * 60)

    cfg = get_task_config(args.task, args.benchmark, args.ruler_scripts)
    max_gen = cfg["tokens_to_generate"]

    model, tok, eos = load_model(args.model_path, args.method, args.load_4bit)
    max_pos = model.config.max_position_embeddings

    data = load_data(args.data_dir, args.task, args.subset, args.num_samples)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.task}.jsonl"

    # 断点续跑
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["index"])
        print(f"📂 已完成 {len(done)} 条,继续剩余样本")

    with open(out_path, "a", encoding="utf-8", buffering=1) as fout:
        for sample in tqdm(data, desc=args.task):
            if sample["index"] in done:
                continue

            enc = tok(sample["input"], return_tensors="pt").to(model.device)
            ids = enc.input_ids
            attn_mask = enc.attention_mask
            seq_len = ids.shape[-1]
            #print(f"  [sample {sample['index']}] input_tokens={seq_len}", flush=True)
            #print(f"  [before forward] alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB", flush=True)

            if seq_len > max_pos:
                raise RuntimeError(f"sample {sample['index']} len={seq_len} > max_pos={max_pos}")

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    eos_token_id=eos,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )

            torch.cuda.synchronize()
            #print(f"  [after generate] alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

            gen = out_ids[0, ids.shape[-1]:].tolist()
            pred = tok.decode(gen, skip_special_tokens=True)

            fout.write(json.dumps({
                "index": sample["index"],
                "pred": pred,
                "input": sample["input"],
                "outputs": sample["outputs"],
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }) + "\n")

            del out_ids, ids, attn_mask, enc
            torch.cuda.empty_cache()

    print(f"✅ saved to {out_path}")


if __name__ == "__main__":
    main()
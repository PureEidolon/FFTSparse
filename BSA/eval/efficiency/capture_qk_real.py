#capture_qk_real.py
"""
生成不同规模的 query/key/value 数据并保存
注意：使用一次性 forward 而非 chunk 推理，避免 chunk 边界导致的分块伪影
"""
import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import argparse
from flash_attn import flash_attn_func
sys.path.append("../..")

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_dir", type=str, default="real_texts")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--lens", type=int, nargs='+', default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--layers_to_save", type=int, nargs='+',
                        default=[0, 3, 7, 11, 15, 19, 23, 27, 30, 31])
    return parser.parse_args()


def generate_prompt(tokenizer, target_len, text_dir):
    text_path = os.path.join(text_dir, f"text_{target_len}.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"真实文本不存在: {text_path}")

    print(f"  使用真实文本: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=target_len).input_ids.to("cuda")
    print(f"  input_ids shape: {input_ids.shape}")
    return input_ids


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
    print(f"  [Layer {self.layer_idx:2d} / {self.total_layers - 1}] forward...", flush=True)


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

    # 保存 Q/K（在 attention 计算之前，数据已经准备好了）
    if self.layer_idx in self.layers_to_save:
        query_path = os.path.join(self.save_dir, f"query_layer{self.layer_idx}_{self.target_len}.pkl")
        key_path = os.path.join(self.save_dir, f"key_layer{self.layer_idx}_{self.target_len}.pkl")
        with open(query_path, "wb") as f:
            pickle.dump(query_states.detach().cpu(), f)
        with open(key_path, "wb") as f:
            pickle.dump(key_states_full.detach().cpu(), f)

    # 用 flash attention 替代 sdpa
    # flash_attn_func 输入格式: (B, S, H, D)
    q = query_states.transpose(1, 2)        # [B, S, H, D]
    k = key_states_full.transpose(1, 2)
    v = value_states_full.transpose(1, 2)
    attn_output = flash_attn_func(q, k, v, causal=True)  # [B, S, H, D]
    attn_output = attn_output.reshape(bsz, q_len, -1)  # [B, S, H*D]
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def load_fake_model(name_or_path, layers_to_save, target_len, save_dir):
    print(f"Loading tokenizer from {name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    print(f"Loading model from {name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    # 启用 gradient checkpointing 以节省显存
    # 这样中间层的激活不会全部保留在显存中
    model.gradient_checkpointing_enable()

    total_layers = len(model.model.layers)
    for layer in model.model.layers:
        layer.self_attn.layers_to_save = layers_to_save
        layer.self_attn.target_len = target_len
        layer.self_attn.save_dir = save_dir
        layer.self_attn.total_layers = total_layers  # 移到这里
        layer.self_attn.forward = forward_to_save.__get__(layer.self_attn)
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for length in args.lens:
        seq_len = length * 1024
        print(f"\n{'=' * 50}")
        print(f"Generating data for {length}K tokens (seq_len={seq_len})")
        print(f"{'=' * 50}")

        # 检查所有层是否都已生成
        all_exist = all(
            os.path.exists(os.path.join(args.output_dir, f"query_layer{l}_{seq_len}.pkl"))
            and os.path.exists(os.path.join(args.output_dir, f"key_layer{l}_{seq_len}.pkl"))
            for l in args.layers_to_save
        )
        if all_exist:
            print(f"Data already exists for {length}K, skipping...")
            continue

        model, tokenizer = load_fake_model(
            name_or_path=args.model_path,
            layers_to_save=args.layers_to_save,
            target_len=seq_len,
            save_dir=args.output_dir,
        )

        input_ids = generate_prompt(tokenizer, seq_len, args.text_dir)
        print(f"Input shape: {input_ids.shape}")

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
        for l in args.layers_to_save:
            qp = os.path.join(args.output_dir, f"query_layer{l}_{seq_len}.pkl")
            kp = os.path.join(args.output_dir, f"key_layer{l}_{seq_len}.pkl")
            with open(qp, "rb") as f:
                q = pickle.load(f)
            with open(kp, "rb") as f:
                k = pickle.load(f)
            assert q.shape[-2] == seq_len, f"layer {l} q mismatch: {q.shape}"
            assert k.shape[-2] == seq_len, f"layer {l} k mismatch: {k.shape}"
            print(f"  Layer {l}: q={q.shape}, k={k.shape}")
            del q, k

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print("=" * 50)
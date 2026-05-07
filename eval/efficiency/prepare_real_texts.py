"""
从 InfiniteBench 的 longbook_sum_eng.jsonl 读取真实长文本，
截取不同长度保存到 real_texts/ 目录
"""
import os
import json
import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="InfiniteBench jsonl 文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="tokenizer 模型路径")
    parser.add_argument("--output_dir", type=str, default="real_texts")
    parser.add_argument("--target_lens", type=int, nargs='+',
                        default=[4, 8, 16, 32, 64, 128],
                        help="目标长度列表（单位K）")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="使用第几条样本")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print(f"加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 读取指定样本的 context
    print(f"读取数据: {args.data_path} (样本 {args.sample_idx})")
    with open(args.data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.sample_idx:
                obj = json.loads(line)
                context = obj["context"]
                break

    # tokenize 整篇文本
    all_ids = tokenizer.encode(context, add_special_tokens=False)
    total_tokens = len(all_ids)
    print(f"原文总 token 数: {total_tokens} ({total_tokens / 1024:.1f}K)")

    # 截取不同长度
    for length_k in args.target_lens:
        target_tokens = length_k * 1024
        out_path = os.path.join(args.output_dir, f"text_{target_tokens}.txt")

        if os.path.exists(out_path):
            print(f"\n{length_k}K: 已存在，跳过")
            continue

        if target_tokens > total_tokens:
            print(f"\n{length_k}K: 需要 {target_tokens} tokens，但原文只有 {total_tokens}，跳过")
            continue

        # 直接截取前 target_tokens 个 token
        truncated_ids = all_ids[:target_tokens + 20]
        text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"{length_k}K: 截取 {len(truncated_ids)} tokens → {out_path}")

    print("\n全部完成！")
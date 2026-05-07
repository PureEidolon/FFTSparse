#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
filter_by_length.py

按 tokenize 后的序列长度筛选数据集样本，保存到当前目录。

使用方式：
  python filter_by_length.py \
      --model llama-3.1-8b-instruct \
      --task 2wikimqa \
      --min_len 0 \
      --max_len 5000 \
      --model2path ./config/model2path.json \
      --dataset2prompt ./config/dataset2prompt.json
"""

import os
import json
import argparse
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Filter dataset by tokenized length")
    p.add_argument("--model",          type=str, required=True)
    p.add_argument("--task",           type=str, required=True)
    p.add_argument("--min_len",        type=int, default=0,    help="最小序列长度（含）")
    p.add_argument("--max_len",        type=int, default=5000, help="最大序列长度（含）")
    p.add_argument("--model2path",     type=str, default="./config/model2path.json")
    p.add_argument("--dataset2prompt", type=str, default="./config/dataset2prompt.json")
    p.add_argument("--data_root",      type=str,
                   default="/root/cjh/pro/resources/datasets/LongBench")
    p.add_argument("--out_dir",        type=str, default="./filtered_data")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model2path     = json.load(open(args.model2path))
    dataset2prompt = json.load(open(args.dataset2prompt))

    print(f"加载 tokenizer: {model2path[args.model]}")
    tokenizer = AutoTokenizer.from_pretrained(model2path[args.model], use_fast=False)

    data_path = os.path.join(args.data_root, f"{args.task}.jsonl")
    print(f"数据集: {data_path}")
    print(f"筛选范围: [{args.min_len}, {args.max_len}]")

    kept = []
    total = 0
    with open(data_path) as f:
        for line in f:
            total += 1
            sample = json.loads(line)
            prompt = dataset2prompt[args.task].format(**sample)
            tokens = tokenizer(prompt, truncation=False)
            seq_len = len(tokens.input_ids)

            if args.min_len <= seq_len <= args.max_len:
                sample["_seq_len"] = seq_len
                kept.append(sample)
                print(f"  [{total}] seq_len={seq_len} ✓", end="\r")
            else:
                print(f"  [{total}] seq_len={seq_len} ✗ (跳过)", end="\r")

    print(f"\n\n总样本: {total}, 筛选后: {len(kept)}")

    out_path = os.path.join(args.out_dir, f"{args.task}_{args.min_len}-{args.max_len}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 保存到: {out_path}")

    # 打印长度分布
    lengths = [s["_seq_len"] for s in kept]
    if lengths:
        print(f"\n长度分布:")
        print(f"  min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
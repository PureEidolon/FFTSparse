#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_stripe_exp.py

评估 stripe_attn_experiment.py 的输出结果。
读取 out_dir 下所有 jsonl，计算指标并打印对比表格。

使用方式：
  python eval_stripe_exp.py --results_path ./stripe_exp_results

需要 metrics.py 在同目录或 sys.path 中。
"""

import os
import sys
import json
import argparse
import numpy as np

# 尝试从多个位置导入 metrics
try:
    from metrics import (
        qa_f1_score, rouge_zh_score, qa_f1_zh_score, rouge_score,
        classification_score, retrieval_score, retrieval_zh_score,
        count_score, code_sim_score,
    )
except ImportError:
    # 如果当前目录没有，尝试从 LongBench eval 目录导入
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[0]
    candidates = [
        str(ROOT / "../../eval/LongBench"),
        str(ROOT / "../LongBench"),
        str(ROOT),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "metrics.py")):
            sys.path.insert(0, c)
            break
    from metrics import (
        qa_f1_score, rouge_zh_score, qa_f1_zh_score, rouge_score,
        classification_score, retrieval_score, retrieval_zh_score,
        count_score, code_sim_score,
    )


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "longbook_qa_eng_88K": qa_f1_score,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate stripe attention experiment results")
    p.add_argument("--results_path", type=str, required=True,
                   help="directory containing the output jsonl files")
    return p.parse_args()


def extract_dataset_name(filename):
    """
    从文件名提取 dataset 名。
    例: 2wikimqa_full.jsonl → 2wikimqa
         2wikimqa_baseline+stripe_5pct.jsonl → 2wikimqa
    """
    # 去掉 .jsonl 后缀
    name = filename.replace(".jsonl", "")
    # 已知的模式后缀
    suffixes = [
        "_full", "_baseline+stripe_20pct", "_baseline+stripe_10pct",
        "_baseline+stripe_5pct", "_baseline",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # fallback: 第一个 _ 之前
    return name.split("_")[0]


def extract_mode_name(filename):
    """
    从文件名提取模式名。
    例: 2wikimqa_full.jsonl → full
         2wikimqa_baseline+stripe_5pct.jsonl → baseline+stripe_5pct
    """
    name = filename.replace(".jsonl", "")
    dataset = extract_dataset_name(filename)
    mode = name[len(dataset) + 1:]  # +1 for the underscore
    return mode


def scorer(dataset, predictions, answers, all_classes):
    """与原 LongBench eval.py 保持一致的评分逻辑。"""
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        prediction = (
            prediction.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        if dataset in ["multifieldqa_zh", "dureader"]:
            prediction = prediction.split("问题：")[0].strip()
        if dataset in ["lsht"]:
            prediction = prediction.split("新闻内容：")[0].strip()
        if dataset in ["passage_retrieval_zh"]:
            prediction = prediction.split("请问")[0].split("提示")[0].strip()
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def main():
    args = parse_args()
    path = args.results_path

    # 收集所有 jsonl 文件
    all_files = sorted([f for f in os.listdir(path) if f.endswith(".jsonl")])
    if not all_files:
        print(f"没有找到 jsonl 文件: {path}")
        return

    print(f"评估目录: {path}")
    print(f"找到 {len(all_files)} 个文件\n")

    # 按 dataset 分组
    results = {}  # {dataset: {mode: score}}

    for filename in all_files:
        dataset_raw = extract_dataset_name(filename)
        mode = extract_mode_name(filename)

        # 模糊匹配 dataset2metric 的 key
        dataset = dataset_raw
        if dataset not in dataset2metric:
            for k in sorted(dataset2metric.keys(), key=len, reverse=True):
                if k in dataset:
                    dataset = k
                    break

        if dataset not in dataset2metric:
            print(f"⚠️  跳过未知数据集: {dataset_raw} ({filename})")
            continue
            
        predictions, answers = [], []
        all_classes = None

        filepath = os.path.join(path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                if "all_classes" in data:
                    all_classes = data["all_classes"]

        if len(predictions) == 0:
            print(f"⚠️  空文件: {filename}")
            continue

        score = scorer(dataset, predictions, answers, all_classes)

        if dataset not in results:
            results[dataset] = {}
        results[dataset][mode] = score

        print(f"  {filename}: {score}")

    # ---- 打印对比表格 ----
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)

    # 确定所有模式（按固定顺序）
    mode_order = ["full", "baseline", "baseline+stripe_5pct",
                  "baseline+stripe_10pct", "baseline+stripe_20pct"]

    for dataset in sorted(results.keys()):
        print(f"\n📊 {dataset}:")
        modes = results[dataset]

        full_score = modes.get("full", None)

        # 表头
        print(f"  {'Mode':<30} {'Score':>8} {'vs Full':>10}")
        print(f"  {'-'*30} {'-'*8} {'-'*10}")

        for m in mode_order:
            if m not in modes:
                continue
            s = modes[m]
            if full_score is not None and m != "full":
                diff = s - full_score
                diff_str = f"{diff:+.2f}"
            else:
                diff_str = "-"
            print(f"  {m:<30} {s:>8.2f} {diff_str:>10}")

    # ---- 保存结果 ----
    out_path = os.path.join(path, "eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from config import DATASET_METRIC
from utils import ensure_dir


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None)
    return parser.parse_args(args)


def custom_sort(s):
    letters = re.findall('[a-zA-Z]+', s)
    numbers = re.findall('\d+', s)
    return (letters, int(numbers[0])) if numbers else (letters, 0)


# scorer 返回值加上 total_sample
def scorer(task_part, predictions, answers, gold_anss):
    dataset_name = re.split(r'_\d+k$', task_part)[0]
    total_score = 0.
    total_sample = 0
    scores = {DATASET_METRIC[dataset_name].__name__: []}
    for (prediction, ground_truths, gold_ans) in zip(predictions, answers, gold_anss):
        total_sample += 1
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, DATASET_METRIC[dataset_name](prediction, ground_truth, gold_ans))
        total_score += score
        scores[DATASET_METRIC[dataset_name].__name__].append(score)
    return total_sample, round(100 * total_score / total_sample, 2), scores


if __name__ == '__main__':
    args = parse_args()
    path = args.input_dir.rstrip("/")
    save_dir = f"{path}/eval_result/"
    ensure_dir(save_dir)

    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    all_files.sort(key=custom_sort)

    all_results = dict()
    all_scores = dict()

    # 表头
    print(f"\n  {'文件':<75} {'samples':>7} {'score':>6} {'input_len':>10} {'prefill_ms':>12} {'attn_ms':>10} {'attn%':>6}")
    print(f"  {'─' * 75} {'─' * 7} {'─' * 6} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 6}")

    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, gold_anss, datas = [], [], [], []
        attn_times, prefill_times, input_lens = [], [], []

        # 解析文件名
        key = filename[:-6]  # 去掉 .jsonl
        first_dash = key.index('-')
        task_part = key[:first_dash]                         # hotpotwikiqa_mixup_16k
        method_part = key[first_dash + 1:]                   # full 或 myattn_v2-[ ... ]
        dataset_name = re.split(r'_\d+k$', task_part)[0]    # hotpotwikiqa_mixup
        length = task_part.split('_')[-1]                    # 16k

        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                gold_ans = data['gold_ans'] if 'gold_ans' in data else None
                gold_anss.append(gold_ans)

                # 收集时间信息
                if "attn_ms" in data:
                    attn_times.append(data["attn_ms"])
                if "prefill_ms" in data:
                    prefill_times.append(data["prefill_ms"])
                if "input_len" in data:
                    input_lens.append(data["input_len"])

        # 接收返回值
        total_sample, score_mean, metric_scores = scorer(task_part, predictions, answers, gold_anss)

        # result_entry 加 samples
        result_entry = {"samples": total_sample, "score": score_mean}
        
        if attn_times:
            result_entry["avg_input_len"] = round(np.mean(input_lens), 0)
            result_entry["avg_prefill_ms"] = round(np.mean(prefill_times), 2)
            result_entry["avg_attn_ms"] = round(np.mean(attn_times), 2)
            result_entry["attn_ratio"] = round(np.mean(attn_times) / np.mean(prefill_times) * 100, 1)

        all_scores[key] = result_entry

        # 打印
        short_name = f"{task_part} ({method_part})"
        if attn_times:
            print(f"  {short_name:<75} {total_sample:>7} {score_mean:>6} {np.mean(input_lens):>10.0f} {np.mean(prefill_times):>12.2f} {np.mean(attn_times):>10.2f} {np.mean(attn_times) / np.mean(prefill_times) * 100:>5.1f}%")
        else:
            print(f"  {short_name:<75} {total_sample:>7} {score_mean:>6}")

        # 汇总到 all_results
        length_entry = {"score": score_mean}
        if attn_times:
            length_entry["avg_attn_ms"] = round(np.mean(attn_times), 2)
            length_entry["avg_prefill_ms"] = round(np.mean(prefill_times), 2)

        if dataset_name in all_results:
            all_results[dataset_name].append({length: length_entry})
        else:
            all_results[dataset_name] = [{length: length_entry}]

    # 保存 JSON
    out_path = os.path.join(save_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)
    print(f"\nJSON saved to {out_path}")

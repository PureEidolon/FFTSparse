import json
import re
import csv
from collections import defaultdict

INPUT_FILE = "result.json"
OUTPUT_FILE = "result.csv"

KNOWN_DATASETS = [
    "passage_retrieval_en", "2wikimqa", "hotpotqa", "passage_count",
    "qasper", "triviaqa", "musique", "trec", "lsht",
    "multifieldqa_en", "multifieldqa_zh", "lcc", "repobench-p",
    "multi_news", "samsum", "vcsum", "qmsum", "dureader", "gov_report",
    "narrativeqa","passage_retrieval_zh",
]

data = defaultdict(dict)
methods_order = []
datasets_order = []

def parse_key(key):
    k = re.sub(r"\(.*?\)\s*$", "", key).strip()  # 去掉末尾 (200 samples)
    k = re.sub(r"\.jsonl$", "", k)               # 去掉 .jsonl
    # 长名字优先匹配，避免 repobench 误匹配 repobench-p 这类情况
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if k.startswith(ds + "-"):
            return ds, k[len(ds) + 1:]
    raise ValueError(f"无法识别 dataset: {key}")


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    obj = json.load(f)

for key, score in obj.items():
    dataset, method = parse_key(key)
    if dataset not in datasets_order:
        datasets_order.append(dataset)
    if method not in methods_order:
        methods_order.append(method)
    data[dataset][method] = score

# 行=方法，列=数据集
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["method"] + datasets_order)
    for m in methods_order:
        row = [m] + [data[ds].get(m, "") for ds in datasets_order]
        writer.writerow(row)

print(f"Done! 写入 {OUTPUT_FILE}")
print(f"数据集数: {len(datasets_order)}, 方法数: {len(methods_order)}")
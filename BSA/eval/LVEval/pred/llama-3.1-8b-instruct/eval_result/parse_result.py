"""
从本地 result.json 读取评估结果，按数据集和方法拆分，汇总各项指标并保存到 CSV。

用法：
    python parse_benchmark.py                          # 默认读取 result.json，输出 benchmark_results.csv
    python parse_benchmark.py -i my_result.json        # 指定输入文件
    python parse_benchmark.py -o output.csv            # 指定输出文件
"""

import re
import csv
import json
import math
import argparse


def parse_key(key: str):
    """
    将 key 拆分为 (dataset, method)。
    数据集名称到 XXk 结束，之后第一个 '-' 后面的部分为方法名。
    例如:
        loogle_CR_mixup_16k-full -> (loogle_CR_mixup_16k, full)
        loogle_CR_mixup_32k-myattn_v2-[...] -> (loogle_CR_mixup_32k, myattn_v2-[...])
    """
    m = re.match(r'^(.*?_\d+k)-(.*)', key)
    if m:
        return m.group(1), m.group(2)
    return key, "unknown"


def parse_all(data: dict) -> list[dict]:
    """解析所有条目，返回行列表。"""
    # 第一遍：解析所有条目
    raw_rows = []
    for key, metrics in data.items():
        dataset, method = parse_key(key)
        raw_rows.append({"dataset": dataset, "method": method, "metrics": metrics})

    # 收集每个数据集下 full 方法的基准值
    full_baseline = {}  # dataset -> {avg_prefill_ms, avg_attn_ms}
    for r in raw_rows:
        if r["method"] == "full":
            full_baseline[r["dataset"]] = {
                "avg_prefill_ms": r["metrics"].get("avg_prefill_ms", 0),
                "avg_attn_ms": r["metrics"].get("avg_attn_ms", 0),
            }

    # 第二遍：构建最终行，插入 speed_up 列
    rows = []
    for r in raw_rows:
        dataset, method, metrics = r["dataset"], r["method"], r["metrics"]
        baseline = full_baseline.get(dataset, {})
        base_prefill = baseline.get("avg_prefill_ms", 0)
        base_attn = baseline.get("avg_attn_ms", 0)

        row = {"dataset": dataset, "method": method}
        for metric_name, value in metrics.items():
            # NaN -> 空字符串，方便 CSV 展示
            if isinstance(value, float) and math.isnan(value):
                row[metric_name] = ""
            else:
                row[metric_name] = value

            # 在 avg_prefill_ms 后紧跟 avg_prefill_speed_up
            if metric_name == "avg_prefill_ms":
                if base_prefill > 0 and isinstance(value, (int, float)) and value > 0 and not math.isnan(value):
                    row["avg_prefill_speed_up"] = round(base_prefill / value, 2)
                else:
                    row["avg_prefill_speed_up"] = ""

            # 在 avg_attn_ms 后紧跟 avg_attn_speed_up
            if metric_name == "avg_attn_ms":
                if base_attn > 0 and isinstance(value, (int, float)) and value > 0 and not math.isnan(value):
                    row["avg_attn_speed_up"] = round(base_attn / value, 2)
                else:
                    row["avg_attn_speed_up"] = ""

        rows.append(row)

    rows.sort(key=lambda x: (x["dataset"], x["method"]))
    return rows


def save_csv(rows: list[dict], output_path: str):
    """保存到 CSV 文件。"""
    if not rows:
        print("没有数据可保存。")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已保存到 {output_path}，共 {len(rows)} 条记录。")


def print_table(rows: list[dict]):
    """在终端打印对齐的表格。"""
    if not rows:
        return
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    header_line = " | ".join(h.rjust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(row[h]).rjust(col_widths[h]) for h in headers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 benchmark 评估结果 JSON，输出 CSV")
    parser.add_argument("-i", "--input", default="result.json", help="输入 JSON 文件路径 (默认: result.json)")
    parser.add_argument("-o", "--output", default="benchmark_results.csv", help="输出 CSV 文件路径 (默认: benchmark_results.csv)")
    args = parser.parse_args()

    # 读取 JSON（处理 NaN：JSON 标准不支持 NaN，用 parse_constant 兼容）
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f, parse_constant=lambda x: float("nan"))

    rows = parse_all(data)
    print_table(rows)
    print()
    save_csv(rows, args.output)
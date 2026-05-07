# eval/RULER/summarize.py
import os, sys, csv, argparse, subprocess
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str,
                   default="/root/cjh/pro/BSA/eval/RULER/results")
    p.add_argument("--model", type=str, default="llama-3.1-8b-instruct")
    p.add_argument("--force", action="store_true",default=True,
                   help="强制重跑 evaluate(忽略已有 summary)")
    return p.parse_args()


def read_summary(csv_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    tasks = rows[1][1:]
    scores = rows[2][1:]
    nulls = rows[3][1:] if len(rows) > 3 else [""] * len(tasks)
    out = {}
    for t, s, nu in zip(tasks, scores, nulls):
        try: n = int(nu.split("/")[1])
        except: n = None
        out[t] = (s, n)
    return out


def run_evaluate(pred_dir, force=False):
    """如果没有 summary csv 就调 evaluate.py 生成"""
    has_summary = any(pred_dir.glob("summary*.csv"))
    has_jsonl = any(pred_dir.glob("*.jsonl"))
    if not has_jsonl:
        return
    if has_summary and not force:
        return

    print(f"▶ 运行 evaluate: {pred_dir}")
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--data_dir", str(pred_dir),
        "--benchmark", "synthetic",
    ]
    # 屏蔽 evaluate.py 自己的 stdout(那一堆 "not found"),保留错误
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  evaluate 失败: {result.stderr[-500:]}")


def collect(pred_dir):
    """从一个 pred/ 目录读所有 summary csv,合并成 {task: score}"""
    merged = {}
    for csv_file in pred_dir.glob("summary*.csv"):
        try:
            merged.update(read_summary(csv_file))
        except Exception as e:
            print(f"⚠️  跳过 {csv_file}: {e}")
    return merged


def main():
    args = parse_args()
    if args.results_dir is None:
        base = Path(__file__).parent / "results" / args.model
    else:
        base = Path(args.results_dir) / args.model
    if not base.exists():
        print(f"❌ {base} 不存在"); return

    # 1) 先扫一遍,所有缺 summary 的 pred 目录都补跑 evaluate
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir(): continue
        for seq_dir in sorted(exp_dir.iterdir()):
            if not seq_dir.is_dir(): continue
            pred_dir = seq_dir / "pred"
            if pred_dir.exists():
                run_evaluate(pred_dir, args.force)

    # 2) 收集所有结果
    results = defaultdict(lambda: defaultdict(dict))
    all_tasks = set()

    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir(): continue
        exp_name = exp_dir.name

        for seq_dir in sorted(exp_dir.iterdir(),
                              key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not seq_dir.is_dir(): continue
            seq_len = seq_dir.name
            pred_dir = seq_dir / "pred"
            if not pred_dir.exists(): continue

            scores = collect(pred_dir)
            if scores:
                results[exp_name][seq_len] = scores
                all_tasks.update(scores.keys())

    if not results:
        print("❌ 没有找到任何结果"); return

    all_tasks = sorted(all_tasks)
    seq_lens  = sorted({s for exp in results.values() for s in exp.keys()},
                       key=lambda x: int(x))

    # 3) 终端打印
    for seq_len in seq_lens:
        print(f"\n{'='*70}")
        print(f"  SEQ_LEN = {seq_len}")
        print(f"{'='*70}")
        col_w = max(40, max(len(e) for e in results.keys()) + 2)
        header = f"{'Method':<{col_w}}" + "".join(f"{t:>16}" for t in all_tasks) + f"{'Avg':>10}"
        print(header)
        print("-" * len(header))

        for exp_name in sorted(results.keys()):
            row_scores = results[exp_name].get(seq_len, {})
            if not row_scores: continue
            cells, valid = [], []
            for t in all_tasks:
                v = row_scores.get(t, "-")
                if isinstance(v, tuple):
                    s, n = v
                    cell = f"{s}({n})" if n is not None else str(s)
                    try:valid.append(float(s))
                    except:pass
                else:
                    cell = "-"
                cells.append(f"{cell:>16}")

            avg = f"{sum(valid)/len(valid):.2f}" if valid else "-"
            print(f"{exp_name:<{col_w}}" + "".join(cells) + f"{avg:>10}")

    # 4) 写 CSV
    out_path = base / "all_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "method"] + all_tasks + ["avg"])
        for seq_len in seq_lens:
            for exp_name in sorted(results.keys()):
                row_scores = results[exp_name].get(seq_len, {})
                if not row_scores: continue
                row = [seq_len, exp_name]
                valid = []
                for t in all_tasks:
                    v = row_scores.get(t, "")
                    if isinstance(v, tuple):
                        s, n = v
                        row.append(f"{s}({n})" if n is not None else s)
                        try:valid.append(float(s))
                        except:pass
                    else:
                        row.append("")
                row.append(f"{sum(valid)/len(valid):.2f}" if valid else "")
                w.writerow(row)

    print(f"\n✅ 汇总已保存: {out_path}")


if __name__ == "__main__":
    main()
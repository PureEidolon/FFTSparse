#!/usr/bin/env python3
"""
计时对比脚本
比较不同 Attention 方法在各个任务上的性能
"""

import pandas as pd
import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="对比不同方法的计时性能")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="预测结果目录路径")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instruct",
                        help="模型名称")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（可选，默认打印到终端）")
    return parser.parse_args()


def extract_method_from_filename(filename):
    """从文件名提取方法名"""
    stem = filename.stem  # 去掉扩展名

    if stem.endswith('_timing'):
        stem = stem[:-7]  # 去掉 '_timing'

    print(" ==== stem:", stem)
    # 解析不同格式
    if '-full' in stem:
        return 'full'
    elif '-xattn' in stem:
        return 'xattn'
    elif '-flex' in stem:
        return 'flex'
    elif '-minference' in stem:
        return 'minference'
    elif '-myattn' in stem:
        # 提取参数
        return stem.split('-', 1)[1]  # 返回完整的 myattn 配置
    else:
        return 'unknown'


def load_timing_data(csv_path):
    """加载单个 CSV 文件的计时数据"""
    try:
        df = pd.read_csv(csv_path)

        stats = {
            'num_samples': len(df),
            'avg_input_length': df['input_length'].mean(),
            'avg_prefill_time': df['prefill_time_s'].mean(),
            'avg_prefill_throughput': df['prefill_throughput_tokens_per_s'].mean(),
            'avg_decode_time': df['decode_time_s'].mean(),
            'avg_decode_throughput': df['decode_throughput_tokens_per_s'].mean(),
            'avg_total_time': df['total_time_s'].mean(),
            'avg_generated_tokens': df['generated_tokens'].mean(),
        }

        return stats
    except Exception as e:
        print(f"⚠️  加载 {csv_path} 失败: {e}")
        return None


def compare_timings(pred_dir, model_name):
    """对比所有方法的计时"""
    pred_path = Path(pred_dir) / model_name

    if not pred_path.exists():
        print(f"❌ 目录不存在: {pred_path}")
        return None

    # 查找所有 timing.csv 文件
    timing_files = list(pred_path.glob("*_timing.csv"))

    if not timing_files:
        print(f"❌ 未找到任何 _timing.csv 文件在: {pred_path}")
        return None

    print(f"📂 找到 {len(timing_files)} 个计时文件")

    # 按任务组织数据
    tasks_data = {}

    for csv_file in timing_files:
        # 提取任务名（文件名第一部分）
        task = csv_file.stem.split('-')[0]
        method = extract_method_from_filename(csv_file)

        # 加载数据
        stats = load_timing_data(csv_file)

        if stats:
            if task not in tasks_data:
                tasks_data[task] = {}

            tasks_data[task][method] = stats

    return tasks_data


def format_comparison_table(tasks_data):
    """格式化对比表格"""
    for task, methods_data in tasks_data.items():
        print("\n" + "=" * 120)
        print(f"📊 任务: {task}")
        print("=" * 120)

        # 创建表格
        rows = []
        for method, stats in methods_data.items():
            rows.append({
                'Method': method,
                'Samples': stats['num_samples'],
                'Avg Input': f"{stats['avg_input_length']:.0f}",
                'Prefill (s)': f"{stats['avg_prefill_time']:.3f}",
                'Prefill (tok/s)': f"{stats['avg_prefill_throughput']:.1f}",
                'Decode (s)': f"{stats['avg_decode_time']:.3f}",
                'Decode (tok/s)': f"{stats['avg_decode_throughput']:.1f}",
                'Total (s)': f"{stats['avg_total_time']:.3f}",
                'Gen Tokens': f"{stats['avg_generated_tokens']:.1f}",
            })

        df = pd.DataFrame(rows)

        # 按 Prefill 时间排序
        df = df.sort_values('Prefill (s)')

        print(df.to_string(index=False))

        # 找出最快的方法
        prefill_times = {row['Method']: float(row['Prefill (s)']) for _, row in df.iterrows()}
        fastest = min(prefill_times, key=prefill_times.get)

        print(f"\n✨ 最快方法: {fastest} ({prefill_times[fastest]:.3f}s)")

        # 计算加速比
        if 'full' in prefill_times:
            baseline = prefill_times['full']
            print(f"\n📈 相对于 Full Attention 的加速:")
            for method, time in sorted(prefill_times.items()):
                if method != 'full':
                    speedup = baseline / time
                    print(f"   {method:<30}: {speedup:.2f}x")


def save_comparison_to_csv(tasks_data, output_path):
    """保存对比结果到 CSV"""
    all_rows = []

    for task, methods_data in tasks_data.items():
        for method, stats in methods_data.items():
            all_rows.append({
                'task': task,
                'method': method,
                'num_samples': stats['num_samples'],
                'avg_input_length': stats['avg_input_length'],
                'avg_prefill_time_s': stats['avg_prefill_time'],
                'avg_prefill_throughput': stats['avg_prefill_throughput'],
                'avg_decode_time_s': stats['avg_decode_time'],
                'avg_decode_throughput': stats['avg_decode_throughput'],
                'avg_total_time_s': stats['avg_total_time'],
                'avg_generated_tokens': stats['avg_generated_tokens'],
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\n💾 对比结果已保存到: {output_path}")


def main():
    args = parse_args()

    print("=" * 120)
    print("⏱️  Attention 方法性能对比")
    print("=" * 120)

    # 加载并对比数据
    tasks_data = compare_timings(args.pred_dir, args.model)

    if not tasks_data:
        return

    # 打印对比表格
    format_comparison_table(tasks_data)

    # 保存到文件（如果指定）
    if args.output:
        save_comparison_to_csv(tasks_data, args.output)
    else:
        # 默认保存到预测目录
        default_output = Path(args.pred_dir) / args.model / "timing_comparison.csv"
        save_comparison_to_csv(tasks_data, default_output)


if __name__ == "__main__":
    main()
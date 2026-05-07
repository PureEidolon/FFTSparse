"""
Benchmark: 对比 myattn, xattn, fullattn 等方法的纯注意力计算效率（多层汇总）
- 加载多层预生成的 Q/K 数据
- 对每个方法，遍历所有层做注意力计算，累加耗时
- warmup 2 次，测试 10 次取平均
"""
import os
import gc
import pdb
import pickle
import json
import time
import torch
import sys
import argparse

from pathlib import Path

NOW_DIR = Path(__file__).resolve().parents[0]
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
print(f"✅ 已把路径 '{ROOT_DIR}' 加入 sys.path")

# ============ 导入各种注意力实现 ============
from flash_attn import flash_attn_func
import numpy as np


try:
    from attn_src.fft_attn.my_attn_v7 import myattn_prefill
    #from attn_src.fft_attn.my_attn_v6_timed import myattn_prefill
    MYATTN_AVAILABLE = True
except ImportError:
    MYATTN_AVAILABLE = False
    print("Warning: myattn_prefill not available")

try:
    from attn_src.xattn.src.Xattention import Xattention_prefill
    XATTN_AVAILABLE = True
except ImportError:
    XATTN_AVAILABLE = False
    print("Warning: Xattention_prefill not available")

try:
    from attn_src.Fullprefill import Full_prefill
    FULL_AVAILABLE = True
except ImportError:
    FULL_AVAILABLE = False
    print("Warning: Full_prefill not available")

try:
    from attn_src.Flexprefill import Flexprefill_prefill
    FLEXPREFILL_AVAILABLE = True
except ImportError:
    FLEXPREFILL_AVAILABLE = False
    print("Warning: Flexprefill_prefill not available")

try:
    from attn_src.Minference import Minference_prefill
    MINFERENCE_AVAILABLE = True
except ImportError:
    MINFERENCE_AVAILABLE = False
    print("Warning: Minference_prefill not available")

try:
    from attn_src.Sparge import Sparge_prefill
    SPARGE_AVAILABLE = False
except ImportError:
    SPARGE_AVAILABLE = False
    print("Warning: Sparge_prefill not available")

# 阈值
from attn_src.xattn.threshold.llama_threshold import llama_fuse_8, llama_fuse_16


# ============ 参数解析 ============
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(NOW_DIR) + "/output")
    parser.add_argument("--lens", type=int, nargs='+', default=[8, 16, 32, 64, 128])
    parser.add_argument("--layers", type=int, nargs='+',default=[0, 3, 7, 11, 15, 19, 23, 27, 30, 31])
    parser.add_argument("--layers_per_len", type=str, nargs='+', default=None,help="按长度指定层数,格式 'LEN:L1,L2,...'")
    parser.add_argument("--num_warmups", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="eval_result/benchmark_attn_multilayer.json")
    parser.add_argument("--is_visual", action="store_true", help="是否进行可视化")

    # myattn 参数
    parser.add_argument("--use_cor", action="store_true")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--sink_ratio", type=float, default=0.01)
    parser.add_argument("--recent_ratio", type=float, default=0.01)
    parser.add_argument("--local_ratio", type=float, default=0.02)
    parser.add_argument("--corr_thres", type=float, nargs='+', default=[1.0])
    parser.add_argument("--corr_selection_mode", type=str, default="threshold",choices=["threshold", "topk"],help="相关性 mask 选择模式：threshold 或 topk")
    parser.add_argument("--corr_topk_ratios", type=float, nargs='+', default=[0.2],help="topk 模式下保留的远程延迟比例列表")

    parser.add_argument("--enable_last_block", action="store_true", help="是否保留最后一个query对key的重要块")
    parser.add_argument("--last_block_thres", type=float, default=0.01, help="最后一个block的阈值")

    parser.add_argument("--enable_column_mask", action="store_true", default=True,
                        help="是否启用列重要性mask")
    parser.add_argument("--column_topk_ratio", type=float, default=0.1,
                        help="列重要性mask的topk比例")

    parser.add_argument("--diag_sample_ratio", type=float, default=0.15, help="对角线采样比例")
    parser.add_argument("--min_diag_samples", type=int, default=5, help="对角线采样数下限")
    parser.add_argument("--max_diag_samples", type=int, default=64, help="对角线采样数上限")
    parser.add_argument("--qk_topk_ratio", type=float, default=0.2, help="弥散模式每行采样比例")
    parser.add_argument("--stripe_threshold", type=float, default=0.3, help="用条纹进行模式判断的阈值")

    parser.add_argument("--target_sparsities", type=float, nargs='+', default=None,
                        help="目标稀疏率列表，指定后自动校准 corr_val")

    # xattn 参数
    parser.add_argument("--xattn_stride", type=int, nargs='+', default=[8, 16])

    # flex 参数
    parser.add_argument("--flex_gamma", type=float, default=0.95)
    parser.add_argument("--flex_tau", type=float, default=0.1)

    return parser.parse_args(args)


# ============ 数据加载 ============
def load_qkv_data(data_dir, seq_len, layer_idx):
    """加载指定层的预生成 Q, K 数据，V 用随机生成"""
    # 将 seq_len 转为 "XK" 格式（如 8192 → "8K"）
    length_in_k = seq_len // 1024
    actual_data_dir = os.path.join(data_dir, f"{length_in_k}K")

    query_path = os.path.join(actual_data_dir, f"query_layer{layer_idx}_{seq_len}.pkl")
    key_path = os.path.join(actual_data_dir, f"key_layer{layer_idx}_{seq_len}.pkl")

    with open(query_path, "rb") as f:
        q = pickle.load(f)
    with open(key_path, "rb") as f:
        k = pickle.load(f)

    q = q.to("cuda").contiguous()
    k = k.to("cuda").contiguous()

    # V 用随机生成
    v = torch.randn_like(q, dtype=torch.bfloat16, device="cuda").contiguous()

    # GQA repeat
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads != kv_heads:
        assert q_heads % kv_heads == 0
        repeat_factor = q_heads // kv_heads
        slen = k.shape[2]
        head_dim = k.shape[3]
        k = k.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1) \
            .reshape(1, q_heads, slen, head_dim).contiguous()
        v = v[:, :kv_heads, :, :].unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1) \
            .reshape(1, q_heads, slen, head_dim).contiguous()

    return q, k, v


def preload_all_layers(data_dir, seq_len, layers):
    """预加载所有层的 Q/K/V 到 GPU，避免反复读磁盘"""
    data = {}
    total = len(layers)
    for i, layer_idx in enumerate(layers, 1):
        print(f"  [Progress {i}/{total}] Loading layer {layer_idx} (seq_len={seq_len})...", end='\r')
        q, k, v = load_qkv_data(data_dir, seq_len, layer_idx)
        data[layer_idx] = (q, k, v)
    print(f"\n  ✅ Finished loading {total} layers.")  # 换行并显示完成
    return data






# ============ 校准函数 ============
def calibrate_from_qk(data_dict, layers, args, target_sparsity):
    """
    直接用预加载的 Q/K 数据二分搜索校准 corr_val，
    使 corr_mask 保留率接近 target_sparsity。
    """
    from attn_src.fft_attn import my_attn_v4 as myattn_module

    def measure(val):
        


        myattn_module._sparsity_collector = {'corr': [], 'final': []}
        with torch.no_grad():
            for layer_idx in layers:
                q, k, v = data_dict[layer_idx]
                if args.corr_selection_mode == "threshold":
                    thres, topk_ratio = val, 0.2
                else:
                    thres, topk_ratio = 1.0, val
                myattn_prefill(
                    q, k, v,
                    layer_idx=layer_idx,
                    block_size=args.block_size,
                    is_causal=True,
                    sink_ratio=args.sink_ratio,
                    recent_ratio=args.recent_ratio,
                    local_span_ratio=args.local_ratio,
                    enable_correlation_mask=True,
                    correlation_selection_mode=args.corr_selection_mode,
                    correlation_topk_ratio=topk_ratio,
                    corr_threshold=thres,
                    enable_column_mask=args.enable_column_mask,
                    column_topk_ratio=args.column_topk_ratio,
                    enable_last_block_mask=args.enable_last_block,
                    last_block_threshold=args.last_block_thres,
                    diag_sample_ratio=args.diag_sample_ratio,
                    min_diag_samples=args.min_diag_samples,
                    max_diag_samples=args.max_diag_samples,
                    qk_topk_ratio=args.qk_topk_ratio,
                    stripe_threshold=args.stripe_threshold,
                    is_visual=False,
                    attention_vis_heads="0",
                    attention_vis_dir='./vis_attn/output',
                )
        corr_ret = float(np.mean(myattn_module._sparsity_collector['corr']))
        final_ret = float(np.mean(myattn_module._sparsity_collector['final']))
        myattn_module._sparsity_collector = {'corr': [], 'final': []}
        return corr_ret, final_ret

    lo, hi = 0.0, 1.0
    best_val = 0.5
    mode_name = "corr_thres" if args.corr_selection_mode == "threshold" else "topk_ratio"

    for step in range(20):
        mid = (lo + hi) / 2
        corr_ret, final_ret = measure(mid)
        print(f"  step {step+1}/20: {mode_name}={mid:.4f}, "
              f"corr保留率={corr_ret:.4f}, final保留率={final_ret:.4f} "
              f"(target={target_sparsity})")

        if abs(corr_ret - target_sparsity) < 0.01:
            best_val = mid
            break

        if args.corr_selection_mode == "threshold":
            if corr_ret > target_sparsity:
                lo = mid
            else:
                hi = mid
        else:
            if corr_ret > target_sparsity:
                hi = mid
            else:
                lo = mid
        best_val = mid

    corr_ret, final_ret = measure(best_val)
    print(f"\n  校准结果: {mode_name}={best_val:.4f}, "
          f"corr保留率={corr_ret:.4f}, final保留率={final_ret:.4f}")

    if hasattr(myattn_module, '_sparsity_collector'):
        del myattn_module._sparsity_collector

    return best_val, corr_ret, final_ret
















# ============ 多层 Benchmark ============
def benchmark_multilayer(method, data_dict, layers, args,
                         corr_val=None,
                         xattn_stride=8,
                         num_warmups=1, num_iterations=3):
    """
    对一个方法，遍历所有层做注意力计算，累加耗时。
    warmup num_warmups 次，测试 num_iterations 次取平均。

    Args:
        corr_val: threshold 模式下的阈值，或 topk 模式下的 ratio（所有层共用同一个值）

    返回: avg_time, times_list, success
    """

    def run_one_layer(q, k, v, layer_idx):
        if method == "full":
            flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                causal=True,
            )
        elif method == "myattn":
            if args.corr_selection_mode == "threshold":
                thres = corr_val
                topk_ratio = 0.2
            else:
                thres = 1.0
                topk_ratio = corr_val

            myattn_prefill(
                q, k, v,
                layer_idx=layer_idx,
                block_size=args.block_size,
                is_causal=True,
                sink_ratio=args.sink_ratio,
                recent_ratio=args.recent_ratio,
                local_span_ratio=args.local_ratio,
                enable_correlation_mask=args.use_cor,
                correlation_selection_mode=args.corr_selection_mode,
                correlation_topk_ratio=topk_ratio,
                corr_threshold=thres,
                enable_column_mask=args.enable_column_mask,
                column_topk_ratio=args.column_topk_ratio,
                enable_last_block_mask=args.enable_last_block,
                last_block_threshold=args.last_block_thres,
                diag_sample_ratio=args.diag_sample_ratio,
                min_diag_samples=args.min_diag_samples,
                max_diag_samples=args.max_diag_samples,
                qk_topk_ratio=args.qk_topk_ratio,
                stripe_threshold=args.stripe_threshold,
                is_visual=args.is_visual,
                attention_vis_heads="0",
                attention_vis_dir='./vis_attn/output',
            )
        elif method == "xattn":
            th_map = {8: llama_fuse_8, 16: llama_fuse_16}
            threshold = torch.tensor(th_map[xattn_stride])[layer_idx].to(q.device)
            Xattention_prefill(
                q, k, v,
                stride=xattn_stride,
                threshold=threshold,
                use_triton=True,
                chunk_size=min(32768, q.shape[2]),
            )
        elif method == "flex":
            Flexprefill_prefill(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                args.flex_gamma, args.flex_tau,
            )
        elif method == "minference":
            Minference_prefill(q, k, v)
        elif method == "sparge":
            Sparge_prefill(q, k, v, topk=0.5, is_causal=False)
        else:
            raise ValueError(f"Unknown method: {method}")

    # warmup
    for _ in range(num_warmups):
        print(f"    [Warmup {_+1}/{num_warmups}] Running all layers...", end='\r')
        for layer_idx in layers:
            q, k, v = data_dict[layer_idx]

            run_one_layer(q, k, v, layer_idx)

        torch.cuda.synchronize()

    # 测试
    times = []
    for i in range(num_iterations):
        print(f"    [Test {i + 1}/{num_iterations}] Running all layers...", end='\r')
        total_time = 0.0
        events = []

        for layer_idx in layers:
            q, k, v = data_dict[layer_idx]

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            run_one_layer(q, k, v, layer_idx)
            end_event.record()

            events.append((start_event, end_event))

        # 所有层跑完后统一 sync 一次，再汇总时间
        torch.cuda.synchronize()
        for start_event, end_event in events:
            total_time += start_event.elapsed_time(end_event)

        times.append(total_time)



    avg_time = sum(times) / len(times)

    gc.collect()
    torch.cuda.empty_cache()

    return avg_time, times, True


# ============ 可用性检查 ============
def check_availability(method):
    avail = {
        "full": True,
        "myattn": MYATTN_AVAILABLE,
        "xattn": XATTN_AVAILABLE,
        "flex": FLEXPREFILL_AVAILABLE,
        "minference": MINFERENCE_AVAILABLE,
        "sparge": SPARGE_AVAILABLE,
    }
    return avail.get(method, False)


# ============ Main ============
if __name__ == "__main__":
    args = parse_args()

    # ===== 解析 layers_per_len =========================================
    layers_per_len_map = {}
    if args.layers_per_len is not None:
        for item in args.layers_per_len:
            try:
                len_str, layers_str = item.split(":")
                layers_per_len_map[int(len_str)] = [int(x) for x in layers_str.split(",")]
            except ValueError:
                raise ValueError(f"--layers_per_len 格式错误: '{item}',应该是 'LEN:L1,L2,...'")
        print(f"  layers_per_len map: {layers_per_len_map}")
    # ==================================================================

    print("=" * 60)
    print("Benchmark Configuration (Multi-Layer Attention Only)")
    print("=" * 60)
    print(f"  data_dir:          {args.data_dir}")
    print(f"  lens:              {args.lens}")
    print(f"  layers:            {args.layers}")
    print(f"  num_warmups:       {args.num_warmups}")
    print(f"  num_iterations:    {args.num_iterations}")
    print(f"  block_size:        {args.block_size}")
    print(f"  use_cor:           {args.use_cor}")
    print(f"  corr_sel_mode:     {args.corr_selection_mode}")
    print(f"  corr_thres:        {args.corr_thres}")
    print(f"  corr_topk_ratios:  {args.corr_topk_ratios}")
    print(f"  qk_topk_ratio:     {args.qk_topk_ratio}")
    print(f"  stripe_threshold:  {args.stripe_threshold}")


    print(f"  xattn_stride:      {args.xattn_stride}")
    print(f"  flex_gamma:        {args.flex_gamma}")
    print(f"  flex_tau:          {args.flex_tau}")

    print("\nAvailability:")
    methods_to_test = ["full", "myattn", "xattn", "flex", "minference", "sparge"]
    for m in methods_to_test:
        status = "✓" if check_availability(m) else "✗"
        print(f"  {m:<15} {status}")

    # ============ 结果存储 ============
    all_results = {}

    for length_k in args.lens:
        seq_len = length_k * 1024

        # ===== 决定本次用哪些层 ==============
        if length_k in layers_per_len_map:
            current_layers = layers_per_len_map[length_k]
        else:
            current_layers = args.layers
        # ===================================

        print(f"\n{'=' * 60}")
        print(f"Testing {length_k}K tokens (seq_len={seq_len})")
        print(f"{'=' * 60}")

        # 预加载所有层数据

        print(f"  Loading {len(current_layers)} layers data...")
        data_dict = preload_all_layers(args.data_dir, seq_len, current_layers)
        sample_q = data_dict[current_layers[0]][0]
        print(f"  Loaded: q.shape={sample_q.shape}, layers={current_layers}")


        length_results = {}

        # ============ 校准（如果指定了 target_sparsities）============
        calibrated_vals = {}  # {sparsity: best_val}
        if args.target_sparsities is not None and args.use_cor and check_availability("myattn"):
            print(f"\n🔧 开始校准 {length_k}K ...")
            for sp in args.target_sparsities:
                print(f"\n  --- target_sparsity={sp} ---")
                best_val, corr_ret, final_ret = calibrate_from_qk(
                    data_dict, current_layers, args, sp
                )
                calibrated_vals[sp] = best_val
            print(f"✅ 校准完成\n")


        # ============ Full Attention (Baseline) ============
        print("Benchmarking full (flash_attn)...")
        avg_time_full, times_full, success = benchmark_multilayer(
            "full", data_dict, current_layers, args,
            num_warmups=args.num_warmups, num_iterations=args.num_iterations,
        )
        if success:
            times_ms = [round(t, 2) for t in times_full]
            print(f"  full: each={times_ms}, avg={avg_time_full:.2f} ms")
            length_results["full"] = {"time_ms": round(avg_time_full, 4)}
        else:
            print("  full: FAILED")
            length_results["full"] = None
            avg_time_full = float('inf')

        # ============ MyAttn ============
        if check_availability("myattn"):
            if args.use_cor:
                # 如果有校准结果，用校准值；否则用原来的参数扫描
                if args.target_sparsities is not None and calibrated_vals:
                    sweep_items = [(f"sp{sp}", v) for sp, v in calibrated_vals.items()]
                else:
                    if args.corr_selection_mode == "threshold":
                        sweep_items = [(f"myattn_thres{v}", v) for v in args.corr_thres]
                    else:
                        sweep_items = [(f"myattn_topk{v}", v) for v in args.corr_topk_ratios]

                for label, val in sweep_items:
                    print(f"\nBenchmarking {label}...")
                    avg_time, times, success = benchmark_multilayer(
                        "myattn", data_dict, current_layers, args,
                        corr_val=val,
                        num_warmups=args.num_warmups, num_iterations=args.num_iterations,
                    )
                    if success:
                        times_ms = [round(t, 2) for t in times]
                        print(f"  {label}: each={times_ms}, avg={avg_time:.2f} ms")
                        result_entry = {
                            "time_ms": round(avg_time, 4),
                            "selection_mode": args.corr_selection_mode,
                            "corr_val": val,
                        }
                        length_results[label] = result_entry
                    else:
                        print(f"  {label}: FAILED")
                        length_results[label] = None

                    #pdb.set_trace()

        # ============ XAttn ============
        if check_availability("xattn"):
            for stride in args.xattn_stride:
                label = f"xattn_s{stride}"
                print(f"\nBenchmarking {label}...")
                avg_time, times, success = benchmark_multilayer(
                    "xattn", data_dict, current_layers, args,
                    xattn_stride=stride,
                    num_warmups=args.num_warmups, num_iterations=args.num_iterations,
                )
                if success:
                    times_ms = [round(t, 2) for t in times]
                    print(f"  {label}: each={times_ms}, avg={avg_time:.2f} ms")
                    length_results[label] = {"time_ms": round(avg_time, 4), "stride": stride}
                else:
                    print(f"  {label}: FAILED")
                    length_results[label] = None

        # ============ FlexPrefill ============
        if check_availability("flex"):
            print(f"\nBenchmarking flex...")
            avg_time, times, success = benchmark_multilayer(
                "flex", data_dict, current_layers, args,
                num_warmups=args.num_warmups, num_iterations=args.num_iterations,
            )
            if success:
                times_ms = [round(t, 2) for t in times]
                print(f"  flex: each={times_ms}, avg={avg_time:.2f} ms")
                length_results["flex"] = {"time_ms": round(avg_time, 4)}
            else:
                print(f"  flex: FAILED")
                length_results["flex"] = None

        # ============ MInference ============
        if check_availability("minference"):
            print(f"\nBenchmarking minference...")
            avg_time, times, success = benchmark_multilayer(
                "minference", data_dict, current_layers, args,
                num_warmups=args.num_warmups, num_iterations=args.num_iterations,
            )
            if success:
                times_ms = [round(t, 2) for t in times]
                print(f"  minference: each={times_ms}, avg={avg_time:.2f} ms")
                length_results["minference"] = {"time_ms": round(avg_time, 4)}
            else:
                print(f"  minference: FAILED")
                length_results["minference"] = None

        # ============ Sparge ============
        if check_availability("sparge"):
            print(f"\nBenchmarking sparge...")
            avg_time, times, success = benchmark_multilayer(
                "sparge", data_dict, current_layers, args,
                num_warmups=args.num_warmups, num_iterations=args.num_iterations,
            )
            if success:
                times_ms = [round(t, 2) for t in times]
                print(f"  sparge: each={times_ms}, avg={avg_time:.2f} ms")
                length_results["sparge"] = {"time_ms": round(avg_time, 4)}
            else:
                print(f"  sparge: FAILED")
                length_results["sparge"] = None

        # ============ 打印当前长度的加速比 ============
        full_ms = length_results.get("full", {}).get("time_ms") if length_results.get("full") else None
        if full_ms:
            print(f"\n  Speedups over full ({full_ms:.2f} ms):")
            for label, val in length_results.items():
                if label == "full":
                    continue
                if val and isinstance(val, dict) and "time_ms" in val:
                    speedup = full_ms / val["time_ms"]
                    print(f"    {label:<25} {val['time_ms']:>10.2f} ms    {speedup:.2f}x")
                else:
                    print(f"    {label:<25} {'FAIL':>10}")

        all_results[f"{length_k}K"] = length_results

        # 释放数据
        del data_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ============ 打印汇总 ============
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # 收集所有方法名
    all_method_labels = []
    for length_key, lr in all_results.items():
        for m in lr.keys():
            if m not in all_method_labels:
                all_method_labels.append(m)

    # header
    header = f"{'Length':<10}"
    for m in all_method_labels:
        header += f"{m + '(ms)':<20}"
    for m in all_method_labels:
        if m != "full":
            header += f"{m + '(↑)':<20}"
    print(header)
    print("-" * len(header))

    # rows
    for length_k in args.lens:
        key = f"{length_k}K"
        if key not in all_results:
            continue
        lr = all_results[key]

        full_time = lr.get("full", {}).get("time_ms") if lr.get("full") else None

        row = f"{key:<10}"
        for m in all_method_labels:
            val = lr.get(m)
            if val and isinstance(val, dict) and "time_ms" in val:
                row += f"{val['time_ms']:<20.2f}"
            else:
                row += f"{'FAIL':<20}"

        for m in all_method_labels:
            if m == "full":
                continue
            val = lr.get(m)
            if val and isinstance(val, dict) and "time_ms" in val and full_time:
                speedup = full_time / val["time_ms"]
                row += f"{speedup:<20.2f}x"
            else:
                row += f"{'-':<20}"

        print(row)

    print("=" * len(header))

    # ============ 保存结果 ============
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    output = {
        "config": {
            "data_dir": args.data_dir,
            "lens": args.lens,
            "layers": args.layers,
            "num_warmups": args.num_warmups,
            "num_iterations": args.num_iterations,
            "block_size": args.block_size,
            "sink_ratio": args.sink_ratio,
            "recent_ratio": args.recent_ratio,
            "local_ratio": args.local_ratio,
            "use_cor": args.use_cor,
            "corr_selection_mode": args.corr_selection_mode,
            "corr_thres": args.corr_thres,
            "corr_topk_ratios": args.corr_topk_ratios,
            "xattn_stride": args.xattn_stride,
            "flex_gamma": args.flex_gamma,
            "flex_tau": args.flex_tau,
        },
        "results": all_results,
    }

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.save_path}")

    # ============ 保存为 CSV(便于画图)============
    import csv

    csv_path = args.save_path.replace(".json", ".csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # 表头:Length, method1_ms, method2_ms, ..., method1_speedup, method2_speedup, ...
        header = ["Length"]
        for m in all_method_labels:
            header.append(f"{m}_ms")
        for m in all_method_labels:
            if m != "full":
                header.append(f"{m}_speedup")
        writer.writerow(header)

        # 数据行
        for length_k in args.lens:
            key = f"{length_k}K"
            if key not in all_results:
                continue
            lr = all_results[key]
            full_time = lr.get("full", {}).get("time_ms") if lr.get("full") else None

            row = [key]
            for m in all_method_labels:
                val = lr.get(m)
                if val and isinstance(val, dict) and "time_ms" in val:
                    row.append(val["time_ms"])
                else:
                    row.append("")
            for m in all_method_labels:
                if m == "full":
                    continue
                val = lr.get(m)
                if val and isinstance(val, dict) and "time_ms" in val and full_time:
                    row.append(round(full_time / val["time_ms"], 4))
                else:
                    row.append("")
            writer.writerow(row)

    print(f"CSV saved to {csv_path}")
# my_attn_v5_timed.py

import sys
import torch
from flash_attn import flash_attn_func
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# =============================================================================
# 计时器
# =============================================================================
# 结构: {layer_idx: {stage_name: [time_ms, ...]}}
_layer_times = {}
_LAST_LAYER = 30  # llama layer 31

def reset_timing():
    global _layer_times
    _layer_times = {}

def get_layer_times():
    return _layer_times

def _record(layer_idx, stage, ms):
    if layer_idx not in _layer_times:
        _layer_times[layer_idx] = {}
    if stage not in _layer_times[layer_idx]:
        _layer_times[layer_idx][stage] = []
    _layer_times[layer_idx][stage].append(ms)

def _cuda_time(func):
    """测量一个函数的 CUDA 耗时（ms）"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = func()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def _print_summary():
    print("\n" + "=" * 70)
    print("📊 MyAttn 各阶段耗时汇总（所有层均值，单位 ms）")
    print("=" * 70)

    all_stages = []
    for layer_idx in sorted(_layer_times.keys()):
        for stage in _layer_times[layer_idx]:
            if stage not in all_stages:
                all_stages.append(stage)

    layers_sorted = sorted(_layer_times.keys())

    # 先算每层总耗时和grand_total
    layer_totals = {}
    for layer_idx in layers_sorted:
        layer_totals[layer_idx] = sum(
            sum(v) / len(v) if v else 0.0
            for v in _layer_times[layer_idx].values()
        )
    grand_total = sum(layer_totals.values())

    # 打印表头：横向是各阶段 + total
    header = "  {:<8} |".format("Layer")
    for stage in all_stages:
        header += " {:>12} |".format(stage[:12])
    header += " {:>8} |".format("total")
    print(header)
    print("-" * len(header))

    # 每层一行
    for layer_idx in layers_sorted:
        row = "  {:<8} |".format("L" + str(layer_idx))
        layer_total = 0.0
        for stage in all_stages:
            times = _layer_times[layer_idx].get(stage, [])
            avg = sum(times) / len(times) if times else 0.0
            row += " {:>10.1f}ms |".format(avg)
            layer_total += avg
        row += " {:>6.1f}ms |".format(layer_total)
        print(row)

    print("-" * len(header))

    # 汇总行：每阶段sum
    sum_row = "  {:<8} |".format("sum")
    stage_sums = []
    for stage in all_stages:
        s = sum(
            sum(_layer_times[l].get(stage, [])) / len(_layer_times[l].get(stage, [1]))
            if _layer_times[l].get(stage) else 0.0
            for l in layers_sorted
        )
        stage_sums.append(s)
        sum_row += " {:>10.1f}ms |".format(s)
    sum_row += " {:>6.1f}ms |".format(grand_total)
    print(sum_row)

    # 占比行：每阶段pct
    pct_row = "  {:<8} |".format("pct")
    for s in stage_sums:
        pct = s / grand_total * 100 if grand_total > 0 else 0.0
        pct_row += " {:>10.1f}%  |".format(pct)
    pct_row += " {:>6} |".format("100.0%")
    print(pct_row)

    # 不计L0的汇总
    layers_no_l0 = [l for l in layers_sorted if l != 0]
    if layers_no_l0:
        total_no_l0 = sum(layer_totals[l] for l in layers_no_l0)
        print("\n  [不计L0]")
        no_l0_row = "  {:<8} |".format("sum(no L0)")
        stage_sums_no_l0 = []
        for stage in all_stages:
            s = sum(
                sum(_layer_times[l].get(stage, [])) / len(_layer_times[l].get(stage, [1]))
                if _layer_times[l].get(stage) else 0.0
                for l in layers_no_l0
            )
            stage_sums_no_l0.append(s)
            no_l0_row += " {:>10.1f}ms |".format(s)
        no_l0_row += " {:>6.1f}ms |".format(total_no_l0)
        print(no_l0_row)

        pct_no_l0_row = "  {:<8} |".format("pct(no L0)")
        for s in stage_sums_no_l0:
            pct = s / total_no_l0 * 100 if total_no_l0 > 0 else 0.0
            pct_no_l0_row += " {:>10.1f}%  |".format(pct)
        pct_no_l0_row += " {:>6} |".format("100.0%")
        print(pct_no_l0_row)


    print("=" * 70 + "\n")


# =============================================================================
# AttentionCache
# =============================================================================
class AttentionCache:
    def __init__(self):
        self._cu_seqlens = {}
        self._head_mask_type = {}

    def get_cu_seqlens(self, seq_len, device):
        key = (seq_len, str(device))
        if key not in self._cu_seqlens:
            self._cu_seqlens[key] = torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            )
        return self._cu_seqlens[key]

    def get_head_mask_type(self, num_heads, device):
        key = (num_heads, str(device))
        if key not in self._head_mask_type:
            self._head_mask_type[key] = torch.tensor(
                [1] * num_heads, dtype=torch.int32, device=device
            )
        return self._head_mask_type[key]

    def reset(self):
        self._cu_seqlens.clear()
        self._head_mask_type.clear()

_attention_cache = AttentionCache()


# =============================================================================
# 运行时自适应状态
# =============================================================================
_adaptive_state = {
    'retrieval_triggered': False,
    'trigger_layer': -1,
    'corr_topk_ratio_boost': 0.15,
    'stripe_threshold_scale': 0.5,
    'qk_topk_ratio_boost': 2,
}

def reset_adaptive_state():
    _adaptive_state['retrieval_triggered'] = False
    _adaptive_state['trigger_layer'] = -1

def get_adaptive_state():
    return _adaptive_state


# =============================================================================
# 先验 Mask 生成函数
# =============================================================================

def create_last_block_query_mask(Q_b, K_b, threshold=0.01, max_blocks=128):
    N, H, D = Q_b.shape
    Q_last = Q_b[-1]
    K_all = K_b.permute(1, 0, 2)
    scores = torch.bmm(Q_last.unsqueeze(1), K_all.transpose(1, 2)).squeeze(1)
    scores = scores / (D ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    mask = attn_weights >= threshold
    num_selected = mask.sum(dim=-1)
    if (num_selected > max_blocks).any():
        k = min(max_blocks, N)
        _, topk_idx = torch.topk(attn_weights, k=k, dim=-1)
        topk_mask = torch.zeros_like(mask)
        topk_mask.scatter_(1, topk_idx, True)
        exceed = (num_selected > max_blocks).unsqueeze(1)
        mask = torch.where(exceed, topk_mask, mask)
    return mask.unsqueeze(0).unsqueeze(2)


def create_deterministic_block_mask(q_block_num, k_block_num, H, device,
                                     keep_sink=1, keep_recent=1, local_window=1):
    idx_q = torch.arange(q_block_num, device=device)[:, None]
    idx_k = torch.arange(k_block_num, device=device)[None, :]
    causal_mask = idx_q >= idx_k
    sink_mask = idx_k < keep_sink
    recent_mask = idx_k >= (k_block_num - keep_recent)
    if local_window > 0:
        local_mask = (idx_q - idx_k).abs() <= local_window
    else:
        local_mask = torch.zeros(q_block_num, k_block_num, dtype=torch.bool, device=device)
    combined_mask = (sink_mask | recent_mask | local_mask) & causal_mask
    final_mask = combined_mask.unsqueeze(0).expand(H, q_block_num, k_block_num).clone()
    return final_mask.unsqueeze(0)


# =============================================================================
# FFT 相关
# =============================================================================

def block_mean_pool(x, block_size):
    L, H, D = x.shape
    N = (L + block_size - 1) // block_size
    pad_len = N * block_size - L
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
    x = x.view(N, block_size, H, D)
    return x.mean(dim=1)


def batch_fft_cross_correlation(Q_b, K_b, local_window=0, keep_sink=0):
    N, H, D = Q_b.shape
    device = Q_b.device
    Q_t = Q_b.permute(1, 2, 0).contiguous().float()
    K_t = K_b.permute(1, 2, 0).contiguous().float()
    fft_size = 2 * N
    Q_fft = torch.fft.rfft(Q_t, n=fft_size, dim=-1)
    K_fft = torch.fft.rfft(K_t, n=fft_size, dim=-1)
    cross_spectrum = Q_fft * torch.conj(K_fft)
    corr_per_dim = torch.fft.irfft(cross_spectrum, n=fft_size, dim=-1)
    corr_total = corr_per_dim.sum(dim=1)
    corr_causal_sum = corr_total[:, :N]
    tau = torch.arange(N, device=device).float()
    num_pairs = (N - tau).clamp(min=1)
    corr_causal = corr_causal_sum / num_pairs.unsqueeze(0)
    exclude_tau = local_window + 1
    sink_tau_start = (N - keep_sink - 5)
    corr_causal[:, :exclude_tau] = 0
    corr_causal[:, sink_tau_start:] = 0
    if exclude_tau < sink_tau_start:
        valid = corr_causal[:, exclude_tau:sink_tau_start]
        corr_min = valid.min(dim=-1, keepdim=True).values
        corr_causal[:, exclude_tau:sink_tau_start] = valid - corr_min
    if exclude_tau < sink_tau_start:
        valid_tau = tau[exclude_tau:sink_tau_start]
        valid_num_pairs = (N - valid_tau).clamp(min=1)
        pair_weight = torch.log10(valid_num_pairs) / torch.log10(valid_num_pairs.max()).clamp(min=1e-6)
        corr_causal[:, exclude_tau:sink_tau_start] = corr_causal[:, exclude_tau:sink_tau_start] * pair_weight.unsqueeze(0)
    return corr_causal


def create_correlation_block_mask(Q_b, K_b, layer_idx, selection_mode="threshold",
                                   corr_threshold=0.1, topk_ratio=0.2,
                                   local_window=0, keep_sink=0, keep_recent=0):
    N, H, D = Q_b.shape
    device = Q_b.device
    corr = batch_fft_cross_correlation(Q_b, K_b, local_window=local_window, keep_sink=keep_sink)
    exclude_tau = max(local_window, keep_recent - 1)
    remote_start = exclude_tau + 1
    if remote_start >= N:
        return torch.zeros(1, H, N, N, dtype=torch.bool, device=device)
    corr_remote = corr[:, remote_start:]
    num_remote_taus = corr_remote.shape[1]
    if selection_mode == "topk":
        if num_remote_taus == 0:
            final_tau_mask = torch.zeros(H, 0, dtype=torch.bool, device=device)
        else:
            quantile_val = 1.0 - topk_ratio
            threshold_val = torch.quantile(corr_remote.float(), quantile_val, dim=-1, keepdim=True)
            final_tau_mask = corr_remote >= threshold_val
    elif selection_mode == "threshold":
        corr_max = corr_remote.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        corr_normalized = corr_remote / corr_max
        final_tau_mask = corr_normalized >= corr_threshold
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    idx_i = torch.arange(N, device=device).view(N, 1)
    idx_j = torch.arange(N, device=device).view(1, N)
    tau_matrix = idx_i - idx_j
    causal_mask = tau_matrix >= 0
    non_sink_mask = idx_j >= keep_sink
    tau_to_remote_idx = tau_matrix - remote_start
    valid_remote = (tau_to_remote_idx >= 0) & (tau_to_remote_idx < num_remote_taus)
    safe_idx = tau_to_remote_idx.clamp(0, num_remote_taus - 1)
    corr_mask = final_tau_mask[:, safe_idx.view(-1)].view(H, N, N)
    corr_mask = corr_mask & valid_remote.unsqueeze(0)
    corr_mask = corr_mask & causal_mask.unsqueeze(0) & non_sink_mask.unsqueeze(0)
    return corr_mask.unsqueeze(0)


def compute_block_scores(Q_b, K_b):
    N, H, D = Q_b.shape
    scores = torch.bmm(
        Q_b.permute(1, 0, 2).contiguous().float(),
        K_b.permute(1, 0, 2).contiguous().float().transpose(1, 2),
    ) / (D ** 0.5)
    return scores


def create_qk_topk_block_mask(scores, qk_topk_ratio=0.2):
    H, N, _ = scores.shape
    device = scores.device
    idx_i = torch.arange(N, device=device).view(N, 1)
    idx_j = torch.arange(N, device=device).view(1, N)
    causal_mask = idx_i >= idx_j
    scores_masked = scores.clone()
    scores_masked.masked_fill_(~causal_mask.unsqueeze(0), float('-inf'))
    visible_per_row = torch.arange(1, N + 1, device=device)
    k_per_row = (visible_per_row.float() * qk_topk_ratio).long().clamp(min=1)
    k_max = int(k_per_row.max().item())
    _, topk_idx = torch.topk(scores_masked, k=k_max, dim=-1)
    rank = torch.arange(k_max, device=device).view(1, 1, k_max)
    row_k = k_per_row.view(1, N, 1)
    valid_topk = rank < row_k
    topk_idx_safe = topk_idx.masked_fill(~valid_topk, N)
    qk_mask = torch.zeros(H, N, N + 1, dtype=torch.bool, device=device)
    qk_mask.scatter_(2, topk_idx_safe, True)
    qk_mask = qk_mask[:, :, :N]
    qk_mask = qk_mask & causal_mask.unsqueeze(0)
    return qk_mask.unsqueeze(0)


def compute_stripe_variance(scores, remote_start, num_sample_diags=5):
    H, N, _ = scores.shape
    device = scores.device
    if remote_start >= N - 1:
        return torch.full((H,), float('inf'), device=device), torch.zeros(H, 1, device=device)
    max_tau = (N + remote_start) // 2
    num_available = max(1, max_tau - remote_start + 1)
    actual_num = min(num_sample_diags, num_available)
    if actual_num <= 1:
        sample_taus = [remote_start]
    else:
        indices = torch.linspace(0, num_available - 1, actual_num).long()
        sample_taus = (indices + remote_start).tolist()
    variances = []
    for tau in sample_taus:
        diag_vals = torch.diagonal(scores, offset=-tau, dim1=1, dim2=2)[:, :-1]
        diff = diag_vals[:, 1:] - diag_vals[:, :-1]
        diff_var = diff.var(dim=-1)
        diag_mean = diag_vals.abs().mean(dim=-1).clamp(min=1e-6)
        variances.append(diff_var / diag_mean)
    per_tau_var = torch.stack(variances, dim=-1)
    mean_var = per_tau_var.mean(dim=-1)
    return mean_var, per_tau_var


def create_column_block_mask(Q_b, K_b, column_topk_ratio=0.1,
                              col_start_offset=4, col_end_offset=5):
    N, H, D = Q_b.shape
    device = Q_b.device
    Q_t = Q_b.permute(1, 0, 2).contiguous().float()
    K_t = K_b.permute(1, 0, 2).contiguous().float()
    Q_cumsum = Q_t.cumsum(dim=1)
    Q_total = Q_cumsum[:, -2, :]
    prev_cumsum = torch.cat([torch.zeros(H, 1, D, device=device), Q_cumsum[:, :-1, :]], dim=1)
    partial_sum = Q_total.unsqueeze(1) - prev_cumsum
    partial_sum[:, -1, :] = 0
    col_dot = (partial_sum * K_t).sum(dim=-1) / (D ** 0.5)
    idx_j = torch.arange(N, device=device).float()
    col_valid_count = (N - idx_j - 1).clamp(min=1)
    col_mean = col_dot / col_valid_count.unsqueeze(0)
    col_start = col_start_offset
    col_end = N - col_end_offset
    if col_end <= col_start:
        return torch.zeros(1, H, N, N, dtype=torch.bool, device=device)
    col_mean_valid = col_mean[:, col_start:col_end]
    num_valid_cols = col_mean_valid.shape[1]
    topk_count = max(1, int(num_valid_cols * column_topk_ratio))
    topk_count = min(topk_count, num_valid_cols)
    _, topk_indices = torch.topk(col_mean_valid, k=topk_count, dim=-1)
    topk_global_indices = topk_indices + col_start
    col_selected = torch.zeros(H, N, dtype=torch.bool, device=device)
    col_selected.scatter_(1, topk_global_indices, True)
    idx_i = torch.arange(N, device=device).view(N, 1)
    idx_j_int = torch.arange(N, device=device).view(1, N)
    causal_mask = idx_i >= idx_j_int
    col_mask = col_selected.unsqueeze(1).expand(H, N, N)
    col_mask = col_mask & causal_mask.unsqueeze(0)
    return col_mask.unsqueeze(0)


# =============================================================================
# 主函数（计时版）
# =============================================================================

def myattn_prefill(
        query_states,
        key_states,
        value_states,
        layer_idx,
        block_size=128,
        is_causal=True,
        is_visual=False,                    # 新增
        sink_ratio=0.1,
        recent_ratio=0.1,
        local_span_ratio=0.1,
        enable_correlation_mask=True,
        correlation_selection_mode="threshold",
        correlation_topk_ratio=0.2,
        corr_threshold=0.1,
        collect_corr_stats=False,           # 新增，保持接口兼容，内部忽略
        enable_column_mask=True,
        column_topk_ratio=0.1,
        column_start_exclude_ratio=0.1,
        column_end_exclude_ratio=0.2,
        enable_last_block_mask=True,
        last_block_threshold=0.01,
        diag_sample_ratio=0.15,
        min_diag_samples=5,
        max_diag_samples=64,
        stripe_threshold=0.3,
        qk_topk_ratio=0.2,
        attention_vis_heads=[0],            # 新增，保持接口兼容，内部忽略
        attention_vis_dir=ROOT_DIR / "vis_attn",  # 新增，保持接口兼容，内部忽略
):
    assert query_states.is_cuda
    L_q = query_states.shape[2]
    L_k = key_states.shape[2]

    # 短序列回退
    if L_k < block_size * 10:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            causal=True,
        ).transpose(1, 2)
        bsz = query_states.shape[0]
        return attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

    bsz = 1
    H = query_states.shape[1]
    device = query_states.device

    # ===== 阶段1: 重塑 + padding =====
    (q_unpad, k_unpad, v_unpad, L_q_padded, L_k_padded), t_reshape = _cuda_time(lambda: _reshape_and_pad(
        query_states, key_states, value_states, L_q, L_k, block_size
    ))
    _record(layer_idx, '1_reshape_pad', t_reshape)

    q_block_num = L_q_padded // block_size
    k_block_num = L_k_padded // block_size

    # ===== 阶段2: 先验 Mask =====
    keep_sink = max(1, int(k_block_num * sink_ratio))
    keep_recent = max(1, int(k_block_num * recent_ratio))
    local_span = max(2, int(k_block_num * local_span_ratio))
    local_window = local_span // 2

    prior_mask, t_prior = _cuda_time(lambda: create_deterministic_block_mask(
        q_block_num, k_block_num, H, device, keep_sink, keep_recent, local_window
    ))
    _record(layer_idx, '2_prior_mask', t_prior)

    # ===== 阶段3: Block Mean Pool =====
    (Q_b, K_b), t_pool = _cuda_time(lambda: (
        block_mean_pool(q_unpad, block_size),
        block_mean_pool(k_unpad, block_size),
    ))
    _record(layer_idx, '3_mean_pool', t_pool)

    # ===== 阶段4: Last Block Mask =====
    if enable_last_block_mask:
        last_block_mask, t_last = _cuda_time(lambda: create_last_block_query_mask(
            Q_b, K_b, threshold=last_block_threshold
        ))
        prior_mask[:, :, -1:, :] = prior_mask[:, :, -1:, :] | last_block_mask
        _record(layer_idx, '4_last_block', t_last)
    else:
        _record(layer_idx, '4_last_block', 0.0)

    # ===== effective 参数 =====
    correlation_top_k = max(1, int(q_block_num * correlation_topk_ratio))
    if _adaptive_state['retrieval_triggered'] and layer_idx > _adaptive_state['trigger_layer']:
        effective_corr_topk_ratio = min(0.8, correlation_topk_ratio + _adaptive_state['corr_topk_ratio_boost'])
        effective_stripe_threshold = stripe_threshold * _adaptive_state['stripe_threshold_scale']
        effective_qk_topk_ratio = min(1.0, qk_topk_ratio * _adaptive_state['qk_topk_ratio_boost'])
    else:
        effective_corr_topk_ratio = correlation_topk_ratio
        effective_stripe_threshold = stripe_threshold
        effective_qk_topk_ratio = qk_topk_ratio

    if enable_correlation_mask and q_block_num > correlation_top_k:

        # ===== 阶段5: FFT 相关性 Mask =====
        corr_mask, t_corr = _cuda_time(lambda: create_correlation_block_mask(
            Q_b, K_b, layer_idx,
            selection_mode=correlation_selection_mode,
            corr_threshold=corr_threshold,
            topk_ratio=effective_corr_topk_ratio,
            local_window=local_window,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        ))
        _record(layer_idx, '5_fft_corr', t_corr)

        # ===== 阶段6: Block Scores =====
        exclude_tau = max(local_window, keep_recent - 1)
        remote_start = exclude_tau + 1
        block_scores, t_scores = _cuda_time(lambda: compute_block_scores(Q_b, K_b))
        _record(layer_idx, '6_block_scores', t_scores)

        # ===== 阶段7: Stripe Variance =====
        num_remote_diags = max(1, (q_block_num + remote_start) // 2 - remote_start + 1)
        num_diag_samples = int(num_remote_diags * diag_sample_ratio)
        num_diag_samples = max(min_diag_samples, min(num_diag_samples, max_diag_samples))

        (stripe_variance, per_tau_var), t_stripe = _cuda_time(lambda: compute_stripe_variance(
            block_scores, remote_start=remote_start, num_sample_diags=num_diag_samples
        ))
        _record(layer_idx, '7_stripe_var', t_stripe)

        # 检索检测
        if not _adaptive_state['retrieval_triggered'] and layer_idx <= 10:
            high_var_per_head = (per_tau_var > 0.3).any(dim=-1)
            high_mean_per_head = stripe_variance >= 0.2
            combined_ratio = (high_var_per_head & high_mean_per_head).float().mean().item()
            if combined_ratio >= 0.3:
                _adaptive_state['retrieval_triggered'] = True
                _adaptive_state['trigger_layer'] = layer_idx

        is_stripe = stripe_variance <= effective_stripe_threshold
        num_stripe = is_stripe.sum().item()
        num_diffuse = H - num_stripe

        # ===== 阶段8: QK TopK Mask（弥散head） =====
        if num_diffuse > 0:
            qk_mask, t_qk = _cuda_time(lambda: create_qk_topk_block_mask(
                block_scores, qk_topk_ratio=effective_qk_topk_ratio
            ))
            _record(layer_idx, '8_qk_topk', t_qk)
            head_selector = is_stripe.view(1, H, 1, 1)
            dynamic_mask = torch.where(head_selector, corr_mask, qk_mask)
        else:
            _record(layer_idx, '8_qk_topk', 0.0)
            dynamic_mask = corr_mask

        final_mask = prior_mask | dynamic_mask
        is_stripe_ref = is_stripe
        num_stripe_ref = num_stripe

    else:
        _record(layer_idx, '5_fft_corr', 0.0)
        _record(layer_idx, '6_block_scores', 0.0)
        _record(layer_idx, '7_stripe_var', 0.0)
        _record(layer_idx, '8_qk_topk', 0.0)
        final_mask = prior_mask
        is_stripe_ref = None
        num_stripe_ref = 0

    # ===== 阶段9: 列重要性 Mask =====
    if enable_column_mask and q_block_num > 1 and (is_stripe_ref is None or num_stripe_ref > 0):
        N_blocks = Q_b.shape[0]
        col_ratio_scale = min(1.0, (128.0 / N_blocks) ** 0.5)
        scaled_column_topk_ratio = column_topk_ratio * col_ratio_scale

        col_mask, t_col = _cuda_time(lambda: create_column_block_mask(
            Q_b, K_b,
            column_topk_ratio=scaled_column_topk_ratio,
            col_start_offset=max(1, int(q_block_num * column_start_exclude_ratio)),
            col_end_offset=max(1, int(q_block_num * column_end_exclude_ratio)),
        ))
        if is_stripe_ref is not None:
            stripe_selector = is_stripe_ref.view(1, H, 1, 1)
            col_mask = col_mask & stripe_selector
        final_mask = final_mask | col_mask
        _record(layer_idx, '9_col_mask', t_col)
    else:
        _record(layer_idx, '9_col_mask', 0.0)

    final_mask = final_mask.contiguous()

    # ===== 阶段10: Kernel 调用 =====
    q_cu_seqlens = _attention_cache.get_cu_seqlens(L_q_padded, device)
    k_cu_seqlens = _attention_cache.get_cu_seqlens(L_k_padded, device)
    head_mask_type = _attention_cache.get_head_mask_type(H, device)

    attn_output_unpad, t_kernel = _cuda_time(lambda: block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        q_cu_seqlens, k_cu_seqlens,
        head_mask_type=head_mask_type,
        streaming_info=None,
        base_blockmask=final_mask,
        max_seqlen_q_=L_q_padded,
        max_seqlen_k_=L_k_padded,
        p_dropout=0.0,
        deterministic=True,
        is_causal=is_causal,
    ))
    _record(layer_idx, '10_kernel', t_kernel)

    # 恢复输出格式
    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)
    attn_output = attn_output[:, :, :L_q, :]
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

    # ===== 当前层打印 =====
    stage_times = _layer_times[layer_idx]
    total = sum(v[-1] for v in stage_times.values())


    # ===== 最后一层打印汇总 =====
    if layer_idx == _LAST_LAYER:
        _print_summary()

    return attn_output


def _reshape_and_pad(query_states, key_states, value_states, L_q, L_k, block_size):
    q_unpad = query_states.squeeze(0).transpose(0, 1).contiguous()
    k_unpad = key_states.squeeze(0).transpose(0, 1).contiguous()
    v_unpad = value_states.squeeze(0).transpose(0, 1).contiguous()
    pad_len_q = (block_size - L_q % block_size) % block_size
    pad_len_k = (block_size - L_k % block_size) % block_size
    if pad_len_q > 0:
        q_unpad = F.pad(q_unpad, (0, 0, 0, 0, 0, pad_len_q))
    if pad_len_k > 0:
        k_unpad = F.pad(k_unpad, (0, 0, 0, 0, 0, pad_len_k))
        v_unpad = F.pad(v_unpad, (0, 0, 0, 0, 0, pad_len_k))
    return q_unpad, k_unpad, v_unpad, L_q + pad_len_q, L_k + pad_len_k
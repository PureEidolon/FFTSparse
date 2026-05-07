# my_attn_v3_timed.py

import pdb
import sys
import time
import torch
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func

from .vis_utils import *
from .corr_stats import get_corr_collector

# ==============================
# 加项目根目录到 sys.path
# ==============================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

IS_DEBUG = False
IS_TIMING = False


# =============================================================================
# 全局计时收集器
# =============================================================================
class TimingCollector:
    """收集所有层的计时数据，用于最终汇总统计"""

    def __init__(self):
        self.all_timings = {}  # {layer_idx: timings_dict}
        self.num_layers = 32   # 默认值，会在运行时更新

    def set_num_layers(self, n):
        self.num_layers = n

    def add(self, layer_idx, timings):
        self.all_timings[layer_idx] = timings

    def is_last_layer(self, layer_idx):
        return layer_idx == self.num_layers - 1

    def print_summary(self):
        if not self.all_timings:
            return

        num_layers = len(self.all_timings)
        # 收集所有步骤名称（保持顺序）
        step_names = list(next(iter(self.all_timings.values())).keys())

        # 计算每个步骤的平均值
        avg_timings = {}
        for step in step_names:
            vals = [self.all_timings[l][step] for l in self.all_timings if step in self.all_timings[l]]
            avg_timings[step] = sum(vals) / len(vals)

        total_avg = sum(avg_timings.values())
        mask_steps = ['3A_prior_mask', '3A2_last_block_mask', '3B_corr_mask', '3C_column_mask', '3D_mask_contiguous']
        mask_avg = sum(avg_timings.get(s, 0) for s in mask_steps)

        # 计算每层总耗时
        layer_totals = {l: sum(t.values()) for l, t in self.all_timings.items()}
        min_layer = min(layer_totals, key=layer_totals.get)
        max_layer = max(layer_totals, key=layer_totals.get)

        print(f"\n{'='*60}")
        print(f"📊 全部 {num_layers} 层平均计时统计")
        print(f"{'='*60}")
        print(f"  {'步骤':<25} {'平均耗时(ms)':>12} {'占比':>8}")
        print(f"  {'-'*47}")
        for step in step_names:
            val = avg_timings[step]
            pct = val / total_avg * 100 if total_avg > 0 else 0
            print(f"  {step:<25} {val*1000:>12.3f} {pct:>7.1f}%")
        print(f"  {'-'*47}")
        print(f"  {'mask总计':<25} {mask_avg*1000:>12.3f} {mask_avg/total_avg*100:>7.1f}%")
        print(f"  {'每层平均总计':<25} {total_avg*1000:>12.3f} {'100.0%':>8}")
        print(f"  {'所有层总计':<25} {total_avg*num_layers*1000:>12.3f}")
        print(f"  {'-'*47}")
        print(f"  最快层: Layer {min_layer} ({layer_totals[min_layer]*1000:.3f}ms)")
        print(f"  最慢层: Layer {max_layer} ({layer_totals[max_layer]*1000:.3f}ms)")
        print(f"{'='*60}\n")

    def reset(self):
        self.all_timings = {}


_timing_collector = TimingCollector()

def get_timing_collector():
    return _timing_collector


class AttentionCache:
    """Attention 相关 tensor 的缓存，避免重复创建"""

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


# 全局缓存实例
_attention_cache = AttentionCache()


# =============================================================================
# 原有的先验 Mask 生成函数
# =============================================================================
def create_last_block_query_mask(
        Q_b: torch.Tensor,
        K_b: torch.Tensor,
        threshold: float = 0.01,
        max_blocks: int = 128,
) -> torch.Tensor:
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


def create_deterministic_block_mask(
        q_block_num: int,
        k_block_num: int,
        H: int,
        device: torch.device,
        keep_sink: int = 1,
        keep_recent: int = 1,
        local_window: int = 1,
        last_block_mask: torch.Tensor = None,
) -> torch.Tensor:

    idx_q = torch.arange(q_block_num, device=device)[:, None]
    idx_k = torch.arange(k_block_num, device=device)[None, :]

    causal_mask = idx_q >= idx_k

    sink_mask = idx_k < keep_sink
    recent_mask = idx_k >= (k_block_num - keep_recent)

    if local_window > 0:
        local_mask = (idx_q - idx_k).abs() <= local_window
    else:
        local_mask = torch.zeros(q_block_num, k_block_num, dtype=torch.bool, device=device)

    combined_mask = sink_mask | recent_mask | local_mask
    combined_mask = combined_mask & causal_mask

    final_mask = combined_mask.unsqueeze(0).expand(H, q_block_num, k_block_num).clone()

    if last_block_mask is not None:
        final_mask[:, -1, :] = final_mask[:, -1, :] | last_block_mask

    return final_mask.unsqueeze(0)


# =============================================================================
# FFT延迟相关性 Mask 相关函数
# =============================================================================

def block_mean_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    L, H, D = x.shape
    N = (L + block_size - 1) // block_size

    pad_len = N * block_size - L
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))

    x = x.view(N, block_size, H, D)
    x_pooled = x.mean(dim=1)

    return x_pooled


def batch_fft_cross_correlation(Q_b: torch.Tensor, K_b: torch.Tensor,
                                 local_window: int = 0, keep_sink: int = 0) -> torch.Tensor:
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

    corr_before_shift = corr_causal.clone()
    valid = corr_causal[:, exclude_tau:sink_tau_start]
    corr_min = valid.min(dim=-1, keepdim=True).values
    corr_causal[:, exclude_tau:sink_tau_start] = valid - corr_min

    corr_before_weight = corr_causal.clone()
    valid_tau = tau[exclude_tau:sink_tau_start]
    valid_num_pairs = (N - valid_tau).clamp(min=1)
    pair_weight = torch.log10(valid_num_pairs) / torch.log10(valid_num_pairs.max()).clamp(min=1e-6)
    corr_causal[:, exclude_tau:sink_tau_start] = corr_causal[:, exclude_tau:sink_tau_start] * pair_weight.unsqueeze(0)

    return corr_causal, corr_causal_sum, corr_before_shift, corr_before_weight


def create_correlation_block_mask(
        Q_b: torch.Tensor,
        K_b: torch.Tensor,
        layer_idx: int,
        selection_mode: str = "threshold",
        corr_threshold: float = 0.1,
        topk_ratio: float = 0.2,
        local_window: int = 0,
        keep_sink: int = 0,
        keep_recent: int = 0,
        collect_stats: bool = False,
) -> torch.Tensor:
    N, H, D = Q_b.shape
    device = Q_b.device

    corr, corr_sum, corr_before_shift, corr_before_weight = batch_fft_cross_correlation(Q_b, K_b, local_window=local_window, keep_sink=keep_sink)

    exclude_tau = max(local_window, keep_recent - 1)
    remote_start = exclude_tau + 1

    if collect_stats:
        corr_remote_for_stats = corr[:, remote_start:] if remote_start < N else None
        get_corr_collector().add(layer_idx, corr, corr_remote_for_stats)

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
        corr_normalized = None

    elif selection_mode == "threshold":
        corr_max = corr_remote.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        corr_normalized = corr_remote / corr_max
        final_tau_mask = corr_normalized >= corr_threshold
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}, expected 'threshold' or 'topk'")

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

    return corr_mask.unsqueeze(0), corr, corr_sum, corr_before_shift, corr_normalized, corr_before_weight


# =============================================================================
# 列重要性 Mask 生成（Key Block 热点检测）
# =============================================================================

def create_column_block_mask(
        Q_b: torch.Tensor,
        K_b: torch.Tensor,
        column_topk_ratio: float = 0.1,
        col_start_offset: int = 4,
        col_end_offset: int = 5,
) -> torch.Tensor:
    """
    找出哪些 Key Block 是"热点"（被大多数 Query 高度关注），整列保留。
    用 cumsum 直接算每列的分数均值，避免构建 N×N 矩阵。
    """
    N, H, D = Q_b.shape
    device = Q_b.device

    Q_t = Q_b.permute(1, 0, 2).contiguous().float()  # [H, N, D]
    K_t = K_b.permute(1, 0, 2).contiguous().float()  # [H, N, D]

    # 前缀和，用于快速算任意区间的 Q 向量之和
    Q_cumsum = Q_t.cumsum(dim=1)  # [H, N, D]

    # Q[0] 到 Q[N-2] 的总和（排除最后一行 query）
    Q_total = Q_cumsum[:, -2, :]  # [H, D]

    # partial_sum[j] = Q[j] + Q[j+1] + ... + Q[N-2]，即列 j 的所有有效 query 之和
    prev_cumsum = torch.cat(
        [torch.zeros(H, 1, D, device=device), Q_cumsum[:, :-1, :]], dim=1
    )  # [H, N, D]
    partial_sum = Q_total.unsqueeze(1) - prev_cumsum  # [H, N, D]
    partial_sum[:, -1, :] = 0  # 最后一列没有有效 query

    # 每列分数 = partial_sum[j] · K[j] / √D
    col_dot = (partial_sum * K_t).sum(dim=-1) / (D ** 0.5)  # [H, N]

    # 每列有效 query 个数（排除最后一行）
    idx_j = torch.arange(N, device=device).float()
    col_valid_count = (N - idx_j - 1).clamp(min=1)
    col_mean = col_dot / col_valid_count.unsqueeze(0)  # [H, N]

    # 在有效范围内选分数最高的几列
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

    # 选中的列整列标记为 True（causal 范围内）
    col_selected = torch.zeros(H, N, dtype=torch.bool, device=device)
    col_selected.scatter_(1, topk_global_indices, True)

    idx_i = torch.arange(N, device=device).view(N, 1)
    idx_j_int = torch.arange(N, device=device).view(1, N)
    causal_mask = idx_i >= idx_j_int

    col_mask = col_selected.unsqueeze(1).expand(H, N, N)
    col_mask = col_mask & causal_mask.unsqueeze(0)

    return col_mask.unsqueeze(0)

# =============================================================================
# 增强版主函数（计时版本）
# =============================================================================

def myattn_prefill(
        query_states,
        key_states,
        value_states,
        layer_idx,
        block_size=128,
        is_causal=True,
        is_visual=False,
        # 先验 mask 参数
        sink_ratio=0.1,
        recent_ratio=0.1,
        local_span_ratio=0.1,
        # FFT延迟相关性 mask 参数
        enable_correlation_mask=True,
        correlation_selection_mode: str = "threshold",
        correlation_topk_ratio: float = 0.2,
        corr_threshold: float = 0.1,
        collect_corr_stats: bool = False,

        # 列重要性 mask 参数
        enable_column_mask: bool = True,
        column_topk_ratio: float = 0.1,
        column_start_exclude_ratio: float = 0.1,
        column_end_exclude_ratio: float = 0.2,

        # 最后一个 block 参数
        enable_last_block_mask=True,
        last_block_threshold=0.01,

        # 可视化参数
        attention_vis_heads=[0],
        attention_vis_dir = ROOT_DIR / "vis_attn",
):
    """
    执行增强版 Block-Sparse Attention（仅用于 prefilling 阶段）- 计时版本

    融合三种稀疏策略：
    1. 先验 Mask：sink + recent + local window（保证基础性能）
    2. FFT延迟相关性 Mask：基于 FFT 互相关的动态选择（捕获长程对角线依赖）
    3. 列重要性 Mask：基于 block-level 注意力分数的列均值（捕获全局热点 Key Block）

    最终 Mask = 先验 Mask OR FFT延迟相关性 Mask OR 列重要性 Mask
    """
    assert query_states.is_cuda, "输入张量必须位于 CUDA 上"
    bsz = 1
    L_q = query_states.shape[2]
    L_k = key_states.shape[2]
    H = query_states.shape[1]
    head_dim = query_states.shape[3]
    device = query_states.device

    # ===== 计时字典 =====
    timings = {}

    # ========== 阶段 1: 张量重塑 ==========
    torch.cuda.synchronize()
    t0 = time.time()

    q_unpad = query_states.squeeze(0).transpose(0, 1).contiguous()
    k_unpad = key_states.squeeze(0).transpose(0, 1).contiguous()
    v_unpad = value_states.squeeze(0).transpose(0, 1).contiguous()

    torch.cuda.synchronize()
    timings['1_reshape'] = time.time() - t0

    # ========== 阶段 2: 计算 block 数量 ==========
    q_block_num = (L_q + block_size - 1) // block_size
    k_block_num = (L_k + block_size - 1) // block_size

    # ========== 阶段 3A: 构建先验 Mask（sink + recent + local） ==========
    torch.cuda.synchronize()
    t0 = time.time()

    keep_sink = max(1, int(k_block_num * sink_ratio))
    keep_recent = max(1, int(k_block_num * recent_ratio))
    local_span = max(2, int(k_block_num * local_span_ratio))
    local_window = local_span // 2

    prior_mask = create_deterministic_block_mask(
        q_block_num=q_block_num,
        k_block_num=k_block_num,
        H=H,
        device=device,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        local_window=local_window,
    )

    torch.cuda.synchronize()
    timings['3A_prior_mask'] = time.time() - t0


    # ========== 统一计算 block mean pooling（只算一次） ==========
    torch.cuda.synchronize()
    t0 = time.time()

    if enable_last_block_mask or enable_correlation_mask or enable_column_mask:
        Q_b = block_mean_pool(q_unpad, block_size)
        K_b = block_mean_pool(k_unpad, block_size)

    torch.cuda.synchronize()
    timings['2_mean_pool'] = time.time() - t0

    # ========== 阶段 3A-2: 最后一个 block 的特殊 mask ==========
    torch.cuda.synchronize()
    t0 = time.time()

    if enable_last_block_mask:
        last_block_mask = create_last_block_query_mask(Q_b, K_b, threshold=last_block_threshold)
        prior_mask[:, :, -1:, :] = prior_mask[:, :, -1:, :] | last_block_mask

    torch.cuda.synchronize()
    timings['3A2_last_block_mask'] = time.time() - t0

    # ========== 阶段 3B: 构建 FFT延迟相关性 Mask（基于互相关） ==========
    torch.cuda.synchronize()
    t0 = time.time()

    correlation_top_k = max(1, int(q_block_num * correlation_topk_ratio))

    if enable_correlation_mask and q_block_num > correlation_top_k:

        corr_mask, corr_causal, corr_causal_sum, corr_before_shift, corr_normalized_remote, corr_before_weight = create_correlation_block_mask(
            Q_b=Q_b,
            K_b=K_b,
            layer_idx=layer_idx,
            selection_mode=correlation_selection_mode,
            corr_threshold=corr_threshold,
            topk_ratio=correlation_topk_ratio,
            local_window=local_window,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            collect_stats=collect_corr_stats,
        )

        final_mask = prior_mask | corr_mask
    else:
        final_mask = prior_mask
        corr_mask = None

    torch.cuda.synchronize()
    timings['3B_corr_mask'] = time.time() - t0

    # ========== 阶段 3C: 构建列重要性 Mask（Key Block 热点检测） ==========
    torch.cuda.synchronize()
    t0 = time.time()

    if enable_column_mask and q_block_num > 1:

        col_mask = create_column_block_mask(
            Q_b=Q_b,
            K_b=K_b,
            column_topk_ratio=column_topk_ratio,
            col_start_offset=max(1, int(q_block_num * column_start_exclude_ratio)),
            col_end_offset=max(1, int(q_block_num * column_end_exclude_ratio)),
        )

        final_mask = final_mask | col_mask
    else:
        col_mask = None

    torch.cuda.synchronize()
    timings['3C_column_mask'] = time.time() - t0

    # ========== mask 合并 & contiguous ==========
    torch.cuda.synchronize()
    t0 = time.time()

    final_mask = final_mask.contiguous()

    torch.cuda.synchronize()
    timings['3D_mask_contiguous'] = time.time() - t0

    # ========== 阶段 4: 准备 kernel 参数 ==========
    torch.cuda.synchronize()
    t0 = time.time()

    q_cu_seqlens = _attention_cache.get_cu_seqlens(L_q, device)
    k_cu_seqlens = _attention_cache.get_cu_seqlens(L_k, device)
    head_mask_type = _attention_cache.get_head_mask_type(H, device)

    torch.cuda.synchronize()
    timings['4_kernel_prep'] = time.time() - t0

    # ========== 阶段 5: 调用底层 kernel ==========
    torch.cuda.synchronize()
    t0 = time.time()

    attn_output_unpad = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        q_cu_seqlens, k_cu_seqlens,
        head_mask_type=head_mask_type,
        streaming_info=None,
        base_blockmask=final_mask,
        max_seqlen_q_=L_q,
        max_seqlen_k_=L_k,
        p_dropout=0.0,
        deterministic=True,
        is_causal=is_causal,
    )

    torch.cuda.synchronize()
    timings['5_kernel'] = time.time() - t0

    # ========== 阶段 6: 恢复输出格式 ==========
    torch.cuda.synchronize()
    t0 = time.time()

    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

    torch.cuda.synchronize()
    timings['6_output_reshape'] = time.time() - t0

    # ========== 逐层打印计时统计 ==========
    total_time = sum(timings.values())
    mask_time = timings['3A_prior_mask'] + timings['3A2_last_block_mask'] + timings['3B_corr_mask'] + timings['3C_column_mask'] + timings['3D_mask_contiguous']

    print(f"\n⏱️  [Layer {layer_idx}] 计时统计 (L_q={L_q}, blocks={q_block_num})")
    print(f"  {'步骤':<25} {'耗时(ms)':>10} {'占比':>8}")
    print(f"  {'-'*45}")
    for step_name, elapsed in timings.items():
        pct = elapsed / total_time * 100 if total_time > 0 else 0
        print(f"  {step_name:<25} {elapsed*1000:>10.3f} {pct:>7.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'mask总计':<25} {mask_time*1000:>10.3f} {mask_time/total_time*100:>7.1f}%")
    print(f"  {'总计':<25} {total_time*1000:>10.3f} {'100.0%':>8}")

    # ========== 收集到全局收集器 & 最后一层打印汇总 ==========
    _timing_collector.add(layer_idx, timings)

    if _timing_collector.is_last_layer(layer_idx):
        _timing_collector.print_summary()
        _timing_collector.reset()

    return attn_output
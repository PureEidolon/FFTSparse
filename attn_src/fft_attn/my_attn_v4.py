# my_attn_v4.py

import pdb
import sys
import torch
from flash_attn import flash_attn_func
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func

from .vis_utils import *
from .corr_stats import get_corr_collector

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

IS_DEBUG = False


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


_attention_cache = AttentionCache()


# =============================================================================
# 先验 Mask 生成函数
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

    combined_mask = (sink_mask | recent_mask | local_mask) & causal_mask
    final_mask = combined_mask.unsqueeze(0).expand(H, q_block_num, k_block_num).clone()

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
    return x.mean(dim=1)


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
    if exclude_tau < sink_tau_start:
        valid = corr_causal[:, exclude_tau:sink_tau_start]
        corr_min = valid.min(dim=-1, keepdim=True).values
        corr_causal[:, exclude_tau:sink_tau_start] = valid - corr_min

    corr_before_weight = corr_causal.clone()
    if exclude_tau < sink_tau_start:
        valid_tau = tau[exclude_tau:sink_tau_start]
        valid_num_pairs = (N - valid_tau).clamp(min=1)
        pair_weight = torch.log10(valid_num_pairs) / torch.log10(valid_num_pairs.max()).clamp(min=1e-6)
        corr_causal[:, exclude_tau:sink_tau_start] = corr_causal[:, exclude_tau:sink_tau_start] * pair_weight.unsqueeze(
            0)
    
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
    """
    基于 FFT延迟相关性 的 block-level 互相关分析，生成稀疏 mask。
    选择相关性较高的延迟对应的 block 对进行保留。
    """
    N, H, D = Q_b.shape
    device = Q_b.device

    corr, corr_sum, corr_before_shift, corr_before_weight = batch_fft_cross_correlation(
        Q_b, K_b, local_window=local_window, keep_sink=keep_sink
    )

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

    return corr_mask.unsqueeze(0), corr, corr_sum, corr_before_shift, corr_normalized, corr_before_weight


# =============================================================================
# Block-level QK 点积（公共计算，供 stripe_variance 和 qk_topk 复用）
# =============================================================================

def compute_block_scores(Q_b: torch.Tensor, K_b: torch.Tensor) -> torch.Tensor:
    """
    计算 block-level QK 点积矩阵，只算一次，供多个下游函数复用。

    Args:
        Q_b: [N, H, D] block-level query
        K_b: [N, H, D] block-level key
    Returns:
        scores: [H, N, N] block-level attention scores
    """
    N, H, D = Q_b.shape
    scores = torch.bmm(
        Q_b.permute(1, 0, 2).contiguous().float(),
        K_b.permute(1, 0, 2).contiguous().float().transpose(1, 2),
    ) / (D ** 0.5)
    return scores



# =============================================================================
# QK 点积 Top-K Mask（弥散模式专用）
# =============================================================================

def create_qk_topk_block_mask(
        scores: torch.Tensor,
        qk_topk_ratio: float = 0.2,
) -> torch.Tensor:
    H, N, _ = scores.shape
    device = scores.device

    idx_i = torch.arange(N, device=device).view(N, 1)
    idx_j = torch.arange(N, device=device).view(1, N)
    causal_mask = idx_i >= idx_j  # [N, N]

    scores_masked = scores.clone()
    scores_masked.masked_fill_(~causal_mask.unsqueeze(0), float('-inf'))

    # 每行动态 k
    visible_per_row = torch.arange(1, N + 1, device=device)        # [N]
    k_per_row = (visible_per_row.float() * qk_topk_ratio).long().clamp(min=1)  # [N]
    k_max = int(k_per_row.max().item())

    # topk_idx: [H, N, k_max]
    _, topk_idx = torch.topk(scores_masked, k=k_max, dim=-1)

    # 每行只有前 k_per_row[i] 个是有效的，其余置为无效 idx
    rank = torch.arange(k_max, device=device).view(1, 1, k_max)    # [1, 1, k_max]
    row_k = k_per_row.view(1, N, 1)                                 # [1, N, 1]
    valid_topk = rank < row_k                                        # [1, N, k_max] -> broadcast [H, N, k_max]

    # 超出动态 k 的位置填 N（越界哨兵，scatter 时不会命中任何有效列）
    topk_idx_safe = topk_idx.masked_fill(~valid_topk, N)

    # scatter 到 [H, N, N+1]，多一列作哨兵，最后丢掉
    qk_mask = torch.zeros(H, N, N + 1, dtype=torch.bool, device=device)
    qk_mask.scatter_(2, topk_idx_safe, True)
    qk_mask = qk_mask[:, :, :N]  # 丢掉哨兵列 [H, N, N]

    # 保险起见再 & 因果 mask
    qk_mask = qk_mask & causal_mask.unsqueeze(0)

    return qk_mask.unsqueeze(0)


# =============================================================================
# 能量集中度判断（条带 vs 弥散）
# =============================================================================

def compute_stripe_variance(
        scores: torch.Tensor,
        remote_start: int,
        num_sample_diags: int = 5,
        layer_idx: int = 0,
        stripe_threshold: float = 0.7,
        is_visual: bool = False,
        save_dir: str = "./vis_attn",
) -> tuple:
    """
    通过在远程区域**均匀采样**对角线的**归一化相邻差分方差**来判断条带/弥散模式。
    """
    H, N, _ = scores.shape
    device = scores.device

    if remote_start >= N - 1:
        mean_var = torch.full((H,), float('inf'), device=device)
        is_stripe = torch.zeros(H, dtype=torch.bool, device=device)
        return mean_var, is_stripe

    # 只在前半段远程区域采样，避免 τ 过大时元素太少导致方差不稳定
    max_tau = (N + remote_start) // 2

    # 在 [remote_start, max_tau] 范围内均匀采样
    num_available = max(1, max_tau - remote_start + 1)
    actual_num = min(num_sample_diags, num_available)  # 外部已按比例计算，此处仅防越界

    if actual_num <= 1:
        sample_taus = [remote_start]
    else:
        indices = torch.linspace(0, num_available - 1, actual_num).long()
        sample_taus = (indices + remote_start).tolist()

    variances = []
    per_tau_len = []
    # 可视化用的中间值
    diag_raw_list = []  # 每条对角线的原始值
    diag_diff_list = []  # 每条对角线的差分值
    diag_mean_list = []  #
    diff_var_list = []  # 每条对角线的差分方差

    for tau in sample_taus:
        diag_vals = torch.diagonal(scores, offset=-tau, dim1=1, dim2=2)  # (H, N-tau)
        # 去掉最后一个元素，排除 last query block 的异常值
        diag_vals = diag_vals[:, :-1]  # (H, N-tau-1)
        diff = diag_vals[:, 1:] - diag_vals[:, :-1]  # (H, N-tau-2)
        diff_var = diff.var(dim=-1)
        diag_mean = diag_vals.abs().mean(dim=-1).clamp(min=1e-6)
        normalized_var = diff_var / diag_mean

        variances.append(normalized_var)
        per_tau_len.append(diag_vals.shape[1])

        diag_raw_list.append(diag_vals)
        diag_diff_list.append(diff)
        diag_mean_list.append(diag_mean)
        diff_var_list.append(diff_var)

    per_tau_var = torch.stack(variances, dim=-1)  # (H, actual_num)
    mean_var = per_tau_var.mean(dim=-1)  # (H,)

    # 判断条带/弥散
    is_stripe = mean_var <= stripe_threshold

    # 可视化
    if is_visual:
        visualize_stripe_variance(
            scores=scores,
            sample_taus=sample_taus,
            per_tau_var=per_tau_var,
            per_tau_len=per_tau_len,
            mean_var=mean_var,
            is_stripe=is_stripe,
            stripe_threshold=stripe_threshold,
            layer_idx=layer_idx,
            save_dir=save_dir,
            diag_raw=diag_raw_list,
            diag_diff=diag_diff_list,
            diag_mean_list=diag_mean_list,
            diff_var_list=diff_var_list,
        )

    return mean_var, is_stripe


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

    Q_t = Q_b.permute(1, 0, 2).contiguous().float()
    K_t = K_b.permute(1, 0, 2).contiguous().float()

    Q_cumsum = Q_t.cumsum(dim=1)
    Q_total = Q_cumsum[:, -2, :]

    prev_cumsum = torch.cat(
        [torch.zeros(H, 1, D, device=device), Q_cumsum[:, :-1, :]], dim=1
    )
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
# 主函数
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
        # 条带/弥散自适应参数
        diag_sample_ratio: float = 0.15,
        min_diag_samples: int = 5,
        max_diag_samples: int = 64,
        stripe_threshold: float = 0.3,
        qk_topk_ratio: float = 0.2,
        # 可视化参数
        attention_vis_heads=[0],
        attention_vis_dir=ROOT_DIR / "vis_attn",
):
    """
    执行增强版 Block-Sparse Attention（仅用于 prefilling 阶段）

    融合多种稀疏策略：
    1. 先验 Mask：sink + recent + local window
    2. 自适应动态 Mask：
       - 条带模式 head → FFT延迟相关性 Mask
       - 弥散模式 head → QK 点积 per-row top-k Mask
    3. 列重要性 Mask：基于 block-level 注意力分数的列均值

    最终 Mask = 先验 Mask OR 动态 Mask OR 列重要性 Mask
    """
    assert query_states.is_cuda, "输入张量必须位于 CUDA 上"
    L_q = query_states.shape[2]
    L_k = key_states.shape[2]



    # === 短序列回退：block 数不足时直接用 full attention ===
    L_q = query_states.shape[2]
    L_k = key_states.shape[2]
    min_blocks_needed = 10  # 至少需要 10 个 block 才有稀疏的意义
    if L_k < block_size * min_blocks_needed:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            causal=True,
        ).transpose(1, 2)
        bsz = query_states.shape[0]
        if layer_idx == 0:print("序列过短，采用flash_attn_func...")
        return attn_output.transpose(1, 2).reshape(bsz, L_q, -1)


    bsz = 1

    H = query_states.shape[1]
    head_dim = query_states.shape[3]
    device = query_states.device

    # ========== 阶段 1: 张量重塑 ==========
    q_unpad = query_states.squeeze(0).transpose(0, 1).contiguous()
    k_unpad = key_states.squeeze(0).transpose(0, 1).contiguous()
    v_unpad = value_states.squeeze(0).transpose(0, 1).contiguous()

    # ========== 阶段 2: 计算 block 数量 ==========
    q_block_num = (L_q + block_size - 1) // block_size
    k_block_num = (L_k + block_size - 1) // block_size

    # ========== 阶段 3A: 构建先验 Mask（sink + recent + local） ==========
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

    # ========== 统一计算 block mean pooling（只算一次） ==========
    if enable_last_block_mask or enable_correlation_mask or enable_column_mask:
        Q_b = block_mean_pool(q_unpad, block_size)
        K_b = block_mean_pool(k_unpad, block_size)

    # ========== 阶段 3A-2: 最后一个 block 的特殊 mask ==========
    if enable_last_block_mask:
        last_block_mask = create_last_block_query_mask(Q_b, K_b, threshold=last_block_threshold)
        prior_mask[:, :, -1:, :] = prior_mask[:, :, -1:, :] | last_block_mask

    # ========== 阶段 3B: 自适应动态 Mask（条带→FFT / 弥散→QK top-k） ==========
    correlation_top_k = max(1, int(q_block_num * correlation_topk_ratio))

    if enable_correlation_mask and q_block_num > correlation_top_k:
        corr_mask, corr_causal, corr_causal_sum, corr_before_shift, corr_normalized_remote, corr_before_weight = create_correlation_block_mask(
            Q_b=Q_b, K_b=K_b, layer_idx=layer_idx,
            selection_mode=correlation_selection_mode,
            corr_threshold=corr_threshold,
            topk_ratio=correlation_topk_ratio,
            local_window=local_window,
            keep_sink=keep_sink, keep_recent=keep_recent,
            collect_stats=collect_corr_stats,
        )

        # 计算 block-level 点积（只算一次，后续 stripe_variance 和 qk_topk 复用）
        exclude_tau = max(local_window, keep_recent - 1)
        remote_start = exclude_tau + 1
        block_scores = compute_block_scores(Q_b, K_b)

        num_remote_diags = max(1, (q_block_num + remote_start) // 2 - remote_start + 1)
        num_diag_samples = int(num_remote_diags * diag_sample_ratio)
        num_diag_samples = max(min_diag_samples, min(num_diag_samples, max_diag_samples))

        # 用差分方差判断每个 head 的模式
        stripe_variance, is_stripe = compute_stripe_variance(
            block_scores,
            remote_start=remote_start,
            num_sample_diags=num_diag_samples,
            layer_idx=layer_idx,
            stripe_threshold=stripe_threshold,
            is_visual=is_visual,
            save_dir=attention_vis_dir,
        )
        num_stripe = is_stripe.sum().item()
        num_diffuse = H - num_stripe

        if num_diffuse > 0:
            # 弥散模式的 head 需要 QK top-k mask（复用 block_scores，避免重复计算点积）
            qk_mask = create_qk_topk_block_mask(block_scores, qk_topk_ratio=qk_topk_ratio)

            # 按 head 合并：条带 head 用 corr_mask，弥散 head 用 qk_mask
            head_selector = is_stripe.view(1, H, 1, 1)
            dynamic_mask = torch.where(head_selector, corr_mask, qk_mask)
        else:
            dynamic_mask = corr_mask

        # print(f"  => 条带/弥散自适应: {num_stripe} stripe heads, {num_diffuse} diffuse heads "f"(threshold={stripe_threshold:.2f})")

        final_mask = prior_mask | dynamic_mask
    else:
        final_mask = prior_mask
        corr_mask = None
        corr_causal = None
        corr_normalized_remote = None
        stripe_variance = None
        is_stripe = None
        num_stripe = 0

    # ========== 阶段 3C: 列重要性 Mask（仅条带 head 需要） ==========
    if enable_column_mask and q_block_num > 1 and (is_stripe is None or num_stripe > 0):

        N_blocks = Q_b.shape[0]
        col_ratio_scale = min(1.0, (128.0 / N_blocks) ** 0.5)
        scaled_column_topk_ratio = column_topk_ratio * col_ratio_scale

        col_mask = create_column_block_mask(
            Q_b=Q_b,
            K_b=K_b,
            column_topk_ratio=scaled_column_topk_ratio,
            col_start_offset=max(1, int(q_block_num * column_start_exclude_ratio)),
            col_end_offset=max(1, int(q_block_num * column_end_exclude_ratio)),
        )
        if is_stripe is not None:
            # 只给条带 head 加列重要性 mask
            stripe_selector = is_stripe.view(1, H, 1, 1)
            col_mask = col_mask & stripe_selector

        final_mask = final_mask | col_mask
    else:
        col_mask = None

    final_mask = final_mask.contiguous()

    # ========== 收集稀疏率（供校准使用） ==========
    if hasattr(sys.modules[__name__], '_sparsity_collector'):
        if corr_mask is not None:
            corr_total = corr_mask.numel()
            corr_kept = corr_mask.sum().item()
            _sparsity_collector['corr'].append(corr_kept / corr_total)
        total = final_mask.numel()
        kept = final_mask.sum().item()
        _sparsity_collector['final'].append(kept / total)



    #print(" === is_visual:",is_visual)
    # 可视化掩码 =====================================================================================================
    if q_unpad.shape[0] > 200 and is_visual and corr_mask is not None:
        total = final_mask.numel()
        kept = final_mask.sum().item()
        ratio = kept / total
        print(f"  => 第 {layer_idx} 层: 数据驱动 block 掩码保留率: {ratio:.4f} ({kept}/{total})")

        print("H:", H)
        vis_heads = [int(x) for x in attention_vis_heads.split(",")]
        for head_idx in vis_heads:
            if head_idx < H:
                attn_scores = compute_attention_scores(q_unpad, k_unpad, head_idx, apply_causal=True)

                '''
                # 可视化1：token-level attention分数热力图 + block mask绿框叠加 + mask稀疏率
                visualize_attention_with_block_mask(
                    attn_scores, final_mask, block_size, layer_idx, head_idx,
                    attention_vis_dir, apply_softmax=False
                )

                # 可视化2：完整attention vs 稀疏attention对比（raw scores + softmax dropped分析）
                visualize_attention_comparison(
                    attn_scores, final_mask, block_size, layer_idx, head_idx, attention_vis_dir
                )
                '''

                # 可视化3：block均值池化后的attention score热力图 + 选中block标记 + 选中/未选中分数分布
                remote_start_val = max(local_window, keep_recent - 1) + 1
                visualize_block_level_attention(
                    Q_b, K_b, final_mask, layer_idx, head_idx, attention_vis_dir,
                    corr_causal_sum=corr_causal_sum, corr_causal=corr_causal,
                    corr_before_shift=corr_before_shift,
                    corr_before_weight=corr_before_weight,
                    corr_normalized_remote=corr_normalized_remote,
                    remote_start=remote_start_val,
                    attn_scores=attn_scores,
                    block_size=block_size,
                    stripe_variance=stripe_variance ,
                    stripe_threshold=stripe_threshold,
                    is_stripe=is_stripe
                )
    # ==============================================================================================================




    # ========== 阶段 4: 准备 kernel 参数 ==========
    q_cu_seqlens = _attention_cache.get_cu_seqlens(L_q, device)
    k_cu_seqlens = _attention_cache.get_cu_seqlens(L_k, device)
    head_mask_type = _attention_cache.get_head_mask_type(H, device)

    # ========== 阶段 5: 调用底层 kernel ==========
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

    # ========== 阶段 6: 恢复输出格式 ==========
    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

    return attn_output
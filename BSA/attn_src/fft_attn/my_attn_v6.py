# my_attn_v6.py

import sys
import torch
from flash_attn import flash_attn_func
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func
from pathlib import Path
from .vis_utils import *

ROOT_DIR = Path(__file__).resolve().parents[2]


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
        per_tau_var = torch.zeros(H, 1, device=device)
        return mean_var, per_tau_var

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

    T = len(sample_taus)
    taus = torch.tensor(sample_taus, device=device)  # [T]

    # 各对角线长度（去掉最后一个元素后）
    diag_lens = [N - int(tau) - 1 for tau in sample_taus]  # [T]
    max_len = max(diag_lens)

    # 一次性 batch gather，padding 到 max_len，无效位置填 0
    # row_idx[t, k] = tau_t + k，col_idx[t, k] = k
    row_indices = torch.zeros(T, max_len, dtype=torch.long, device=device)
    col_indices = torch.zeros(T, max_len, dtype=torch.long, device=device)
    valid_mask = torch.zeros(T, max_len, dtype=torch.bool, device=device)

    for t, (tau, dlen) in enumerate(zip(sample_taus, diag_lens)):
        tau_int = int(tau)
        row_indices[t, :dlen] = torch.arange(tau_int, tau_int + dlen, device=device)
        col_indices[t, :dlen] = torch.arange(0, dlen, device=device)
        valid_mask[t, :dlen] = True

    # batch gather: [H, T, max_len]
    H_idx = torch.arange(H, device=device).view(H, 1, 1).expand(H, T, max_len)
    T_row = row_indices.unsqueeze(0).expand(H, T, max_len)
    T_col = col_indices.unsqueeze(0).expand(H, T, max_len)
    diag_batch = scores[H_idx, T_row, T_col]  # [H, T, max_len]
    diag_batch = diag_batch * valid_mask.unsqueeze(0)  # padding 位置置 0

    # 差分：[H, T, max_len-1]
    diff = diag_batch[:, :, 1:] - diag_batch[:, :, :-1]
    # padding 边界处的差分无意义，也 mask 掉
    diff_valid_mask = valid_mask[:, 1:] & valid_mask[:, :-1]  # [T, max_len-1]
    diff = diff * diff_valid_mask.unsqueeze(0)

    # 各对角线有效长度 [T]
    lens_tensor = torch.tensor(diag_lens, device=device).float()  # [T]
    diff_lens = (lens_tensor - 1).clamp(min=1)  # [T] 差分长度

    # 差分方差：只在有效位置计算
    # E[x] = sum(x) / n，Var[x] = E[x^2] - E[x]^2
    diff_sum = diff.sum(dim=-1)  # [H, T]
    diff_sum2 = (diff ** 2).sum(dim=-1)  # [H, T]
    diff_mean = diff_sum / diff_lens.unsqueeze(0)  # [H, T]
    diff_var = diff_sum2 / diff_lens.unsqueeze(0) - diff_mean ** 2  # [H, T]
    diff_var = diff_var.clamp(min=0)

    # 对角线元素绝对值均值（只在有效位置）
    diag_abs_sum = diag_batch.abs().sum(dim=-1)  # [H, T]
    diag_mean = diag_abs_sum / lens_tensor.unsqueeze(0)  # [H, T]
    diag_mean = diag_mean.clamp(min=1e-6)

    per_tau_var = diff_var / diag_mean  # [H, T]
    mean_var = per_tau_var.mean(dim=-1)  # [H]

    per_tau_len = diag_lens  # 保留真实长度，供可视化用








    # 可视化
    if is_visual:
        is_stripe_for_vis = mean_var <= stripe_threshold
        diag_raw_list = [diag_batch[:, t, :diag_lens[t]] for t in range(T)]
        diag_diff_list = [diff[:, t, :diag_lens[t] - 1] for t in range(T)]
        diag_mean_list = [diag_mean[:, t] for t in range(T)]
        diff_var_list = [diff_var[:, t] for t in range(T)]

        visualize_stripe_variance(
            scores=scores,
            sample_taus=sample_taus,
            per_tau_var=per_tau_var,
            per_tau_len=per_tau_len,
            mean_var=mean_var,
            is_stripe=is_stripe_for_vis,
            stripe_threshold=stripe_threshold,
            layer_idx=layer_idx,
            save_dir=save_dir,
            diag_raw=diag_raw_list,
            diag_diff=diag_diff_list,
            diag_mean_list=diag_mean_list,
            diff_var_list=diff_var_list,
        )

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
        sink_ratio=0.1,
        recent_ratio=0.1,
        local_span_ratio=0.1,
        enable_correlation_mask=True,
        correlation_selection_mode="threshold",
        correlation_topk_ratio=0.2,
        corr_threshold=0.1,
        collect_corr_stats=False,
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
        attention_vis_heads=[0],
        attention_vis_dir = "",
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
        if layer_idx == 0:
            print("序列过短，采用flash_attn_func...")
        return attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

    bsz = 1
    H = query_states.shape[1]
    device = query_states.device

    # 重塑 + padding
    q_unpad, k_unpad, v_unpad, L_q_padded, L_k_padded = _reshape_and_pad(
        query_states, key_states, value_states, L_q, L_k, block_size
    )

    q_block_num = L_q_padded // block_size
    k_block_num = L_k_padded // block_size

    # 先验 Mask
    keep_sink = max(1, int(k_block_num * sink_ratio))
    keep_recent = max(1, int(k_block_num * recent_ratio))
    local_span = max(2, int(k_block_num * local_span_ratio))
    local_window = local_span // 2

    prior_mask = create_deterministic_block_mask(
        q_block_num, k_block_num, H, device, keep_sink, keep_recent, local_window
    )

    # Block Mean Pool
    Q_b = block_mean_pool(q_unpad, block_size)
    K_b = block_mean_pool(k_unpad, block_size)

    # Last Block Mask
    if enable_last_block_mask:
        last_block_mask = create_last_block_query_mask(Q_b, K_b, threshold=last_block_threshold)
        prior_mask[:, :, -1:, :] = prior_mask[:, :, -1:, :] | last_block_mask

    # effective 参数
    correlation_top_k = max(1, int(q_block_num * correlation_topk_ratio))
    if _adaptive_state['retrieval_triggered'] and layer_idx > _adaptive_state['trigger_layer']:
        effective_corr_topk_ratio = min(0.8, correlation_topk_ratio + _adaptive_state['corr_topk_ratio_boost'])
        effective_stripe_threshold = stripe_threshold * _adaptive_state['stripe_threshold_scale']
        effective_qk_topk_ratio = min(1.0, qk_topk_ratio * _adaptive_state['qk_topk_ratio_boost'])
        if layer_idx == _adaptive_state['trigger_layer'] + 1:
            print(f"⚙️  [Layer {layer_idx}] 检索模式参数已生效:")
            print(f"    corr_topk_ratio : {correlation_topk_ratio:.3f} => {effective_corr_topk_ratio:.3f}")
            print(f"    stripe_threshold: {stripe_threshold:.3f} => {effective_stripe_threshold:.3f}")
            print(f"    qk_topk_ratio   : {qk_topk_ratio:.3f} => {effective_qk_topk_ratio:.3f}")
    else:
        effective_corr_topk_ratio = correlation_topk_ratio
        effective_stripe_threshold = stripe_threshold
        effective_qk_topk_ratio = qk_topk_ratio

    if enable_correlation_mask and q_block_num > correlation_top_k:

        corr_mask, corr_causal, corr_causal_sum, corr_before_shift, corr_normalized_remote, corr_before_weight = create_correlation_block_mask(
            Q_b=Q_b, K_b=K_b, layer_idx=layer_idx,
            selection_mode=correlation_selection_mode,
            corr_threshold=corr_threshold,
            topk_ratio=effective_corr_topk_ratio,
            local_window=local_window,
            keep_sink=keep_sink, keep_recent=keep_recent,
        )


        exclude_tau = max(local_window, keep_recent - 1)
        remote_start = exclude_tau + 1
        block_scores = compute_block_scores(Q_b, K_b)

        num_remote_diags = max(1, (q_block_num + remote_start) // 2 - remote_start + 1)
        num_diag_samples = int(num_remote_diags * diag_sample_ratio)
        num_diag_samples = max(min_diag_samples, min(num_diag_samples, max_diag_samples))

        stripe_variance, per_tau_var = compute_stripe_variance(
            block_scores,
            remote_start=remote_start,
            num_sample_diags=num_diag_samples,
            is_visual=is_visual,
            stripe_threshold=effective_stripe_threshold,
            layer_idx=layer_idx,
            save_dir=attention_vis_dir,
        )

        # 检索检测
        if not _adaptive_state['retrieval_triggered'] and layer_idx <= 10:
            high_var_per_head = (per_tau_var > 0.3).any(dim=-1)
            high_mean_per_head = stripe_variance >= 0.2
            combined_ratio = (high_var_per_head & high_mean_per_head).float().mean().item()
            if combined_ratio >= 0.3:
                _adaptive_state['retrieval_triggered'] = True
                _adaptive_state['trigger_layer'] = layer_idx
                print(f"🔍 [Layer {layer_idx}] 检测到检索模式: {combined_ratio * 100:.0f}% heads触发")

        is_stripe = stripe_variance <= effective_stripe_threshold
        num_stripe = is_stripe.sum().item()
        num_diffuse = H - num_stripe

        if num_diffuse > 0:
            qk_mask = create_qk_topk_block_mask(block_scores, qk_topk_ratio=effective_qk_topk_ratio)
            head_selector = is_stripe.view(1, H, 1, 1)
            dynamic_mask = torch.where(head_selector, corr_mask, qk_mask)
        else:
            dynamic_mask = corr_mask

        final_mask = prior_mask | dynamic_mask
        is_stripe_ref = is_stripe
        num_stripe_ref = num_stripe

    else:
        final_mask = prior_mask
        is_stripe_ref = None
        num_stripe_ref = 0
        corr_mask = None
        corr_causal = None
        corr_causal_sum = None
        corr_before_shift = None
        corr_before_weight = None
        corr_normalized_remote = None
        stripe_variance = None
        is_stripe = None  # 可视化块里用的是 is_stripe

        
    # 列重要性 Mask
    if enable_column_mask and q_block_num > 1 and (is_stripe_ref is None or num_stripe_ref > 0):
        N_blocks = Q_b.shape[0]
        col_ratio_scale = min(1.0, (128.0 / N_blocks) ** 0.5)
        scaled_column_topk_ratio = column_topk_ratio * col_ratio_scale

        col_mask = create_column_block_mask(
            Q_b, K_b,
            column_topk_ratio=scaled_column_topk_ratio,
            col_start_offset=max(1, int(q_block_num * column_start_exclude_ratio)),
            col_end_offset=max(1, int(q_block_num * column_end_exclude_ratio)),
        )
        if is_stripe_ref is not None:
            stripe_selector = is_stripe_ref.view(1, H, 1, 1)
            col_mask = col_mask & stripe_selector
        final_mask = final_mask | col_mask

    final_mask = final_mask.contiguous()












    # ========== 收集稀疏率（供校准使用） ==============================================================================
    if hasattr(sys.modules[__name__], '_sparsity_collector'):
        if 'corr_mask' in dir() and corr_mask is not None:  # 改这里
            corr_total = corr_mask.numel()
            corr_kept = corr_mask.sum().item()
            _sparsity_collector['corr'].append(corr_kept / corr_total)
        total = final_mask.numel()
        kept = final_mask.sum().item()
        _sparsity_collector['final'].append(kept / total)

    # print(" === is_visual:",is_visual)
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






















    # Kernel 调用
    q_cu_seqlens = _attention_cache.get_cu_seqlens(L_q_padded, device)
    k_cu_seqlens = _attention_cache.get_cu_seqlens(L_k_padded, device)
    head_mask_type = _attention_cache.get_head_mask_type(H, device)

    attn_output_unpad = block_sparse_attn_func(
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
    )

    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)
    attn_output = attn_output[:, :, :L_q, :]
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)

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
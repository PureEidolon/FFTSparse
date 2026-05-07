# my_attn_v2.py

import pdb
import sys
import torch
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func

from .vis_utils import *
from .corr_stats import get_corr_collector  # 新增导入

# ==============================
# 加项目根目录到 sys.path
# ==============================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]  # 0是父目录，1是爷目录...

IS_DEBUG = False
IS_TIMING = False


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

    Q_last = Q_b[-1]  # [H, D]
    K_all = K_b.permute(1, 0, 2)  # [H, N, D]

    scores = torch.bmm(Q_last.unsqueeze(1), K_all.transpose(1, 2)).squeeze(1)  # [H, N]
    scores = scores / (D ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)  # [H, N]

    # 先用阈值筛选
    mask = attn_weights >= threshold  # [H, N]

    # 检查是否有 head 超过 max_blocks
    num_selected = mask.sum(dim=-1)  # [H]

    if (num_selected > max_blocks).any():
        # 只对超限的 head 用 top-k 截断
        k = min(max_blocks, N)
        _, topk_idx = torch.topk(attn_weights, k=k, dim=-1)  # [H, k]
        topk_mask = torch.zeros_like(mask)
        topk_mask.scatter_(1, topk_idx, True)

        # 超限的 head 用 topk_mask，否则保持原 mask
        exceed = (num_selected > max_blocks).unsqueeze(1)  # [H, 1]
        mask = torch.where(exceed, topk_mask, mask)

    #num_selected = mask.sum(dim=-1)
    #print(f"[LastBlockMask] 保留: min={num_selected.min().item()}, max={num_selected.max().item()}, mean={num_selected.float().mean().item():.1f}/{N}")

    return mask.unsqueeze(0).unsqueeze(2)


def create_deterministic_block_mask(
        q_block_num: int,
        k_block_num: int,
        H: int,
        device: torch.device,
        keep_sink: int = 1,
        keep_recent: int = 1,
        local_window: int = 1,
        last_block_mask: torch.Tensor = None,  # 新增参数
) -> torch.Tensor:

    idx_q = torch.arange(q_block_num, device=device)[:, None]
    idx_k = torch.arange(k_block_num, device=device)[None, :]

    causal_mask = idx_q >= idx_k

    # ========== Sink mask ==========
    sink_mask = idx_k < keep_sink

    # ========== Recent mask ==========
    recent_mask = idx_k >= (k_block_num - keep_recent)

    # ========== Local window mask ==========
    if local_window > 0:
        local_mask = (idx_q - idx_k).abs() <= local_window
    else:
        local_mask = torch.zeros(q_block_num, k_block_num, dtype=torch.bool, device=device)

    # ========== 合并 ==========
    combined_mask = sink_mask | recent_mask | local_mask
    combined_mask = combined_mask & causal_mask

    # ========== 扩展到所有 heads ==========
    final_mask = combined_mask.unsqueeze(0).expand(H, q_block_num, k_block_num).clone()

    # ========== 应用最后一个 block 的特殊 mask ==========
    if last_block_mask is not None:
        # last_block_mask: [H, k_block_num]
        # 替换最后一行
        final_mask[:, -1, :] = final_mask[:, -1, :] | last_block_mask

    return final_mask.unsqueeze(0)


# =============================================================================
# 新增： Correlation-based Mask 生成
# =============================================================================

def block_mean_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    对输入张量进行 block-level mean pooling。

    Args:
        x: [L, H, D] 格式的张量
        block_size: 每个 block 的大小

    Returns:
        x_pooled: [N, H, D]，其中 N = ceil(L / block_size)
    """
    L, H, D = x.shape
    N = (L + block_size - 1) // block_size

    # 补零到 N * block_size
    pad_len = N * block_size - L
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))

    # reshape 并求均值
    x = x.view(N, block_size, H, D)
    x_pooled = x.mean(dim=1)  # [N, H, D]

    return x_pooled


# 用FFT计算不同τ下的斜线相关分数
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

    corr_causal_sum = corr_total[:, :N]  # [H, N]

    # === 归一化用于选择 τ ===
    tau = torch.arange(N, device=device).float()
    num_pairs = (N - tau).clamp(min=1)
    corr_causal = corr_causal_sum / num_pairs.unsqueeze(0)

    # === 剔除 local 和 sink 区域 ===
    exclude_tau = local_window + 1             # 从local_window之外开始计算
    sink_tau_start = (N - keep_sink - 5)       # 远离注意力分数矩阵的左下角，此处一条斜线只有几个元素，且统计了last-query block中的元素，很容易拉高平均分
    corr_causal[:, :exclude_tau] = 0
    corr_causal[:, sink_tau_start:] = 0

    # === 偏移：只基于有效区域的min ===
    corr_before_shift = corr_causal.clone()  # 保存偏移前的值（用于可视化）
    valid = corr_causal[:, exclude_tau:sink_tau_start]
    corr_min = valid.min(dim=-1, keepdim=True).values
    corr_causal[:, exclude_tau:sink_tau_start] = valid - corr_min  # 偏移后直接覆盖，用于后续计算

    # === 加权：τ 小的斜线（配对数多）分数更高 ===
    corr_before_weight = corr_causal.clone()  # 保存加权前的值（用于可视化）
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
    """
    基于 block-level 互相关分析。

    Args:
        Q_b: [N, H, D] block-level query 表征
        K_b: [N, H, D] block-level key 表征
        layer_idx: 当前层索引，用于统计收集
        selection_mode: 选择模式，"threshold" 或 "topk"
        corr_threshold: 阈值模式下的相关性阈值（归一化后），超过此值的延迟被选中
        topk_ratio: topk 模式下选择的比例
        local_window: 需要排除的 local window 大小
        keep_sink: 需要排除的 sink 列数
        keep_recent: 需要排除的 recent 范围
        collect_stats: 是否收集统计数据

    Returns:
        mask: [1, H, N, N] (bool)，表示哪些 block 对需要计算 attention
    """
    N, H, D = Q_b.shape
    device = Q_b.device

    # ==================== Step 1: 计算互相关 ====================
    corr, corr_sum, corr_before_shift, corr_before_weight = batch_fft_cross_correlation(Q_b, K_b, local_window=local_window, keep_sink=keep_sink)

    # ==================== Step 2: 确定远程区域范围 ====================
    # 远程区域 = 排除 local_window 和 keep_recent 覆盖的部分
    exclude_tau = max(local_window, keep_recent - 1)
    remote_start = exclude_tau + 1

    # 收集统计数据（可选）
    if collect_stats:
        corr_remote_for_stats = corr[:, remote_start:] if remote_start < N else None
        get_corr_collector().add(layer_idx, corr, corr_remote_for_stats)

    # 如果所有 block 都被 local 区域覆盖，返回空 mask
    if remote_start >= N:
        return torch.zeros(1, H, N, N, dtype=torch.bool, device=device)

    # 提取远程部分的相关性
    corr_remote = corr[:, remote_start:]  # [H, num_remote_taus]
    num_remote_taus = corr_remote.shape[1]

    # ==================== Step 3: 根据模式选择远程 block ====================
    if selection_mode == "topk":
        # TopK 模式：用分位数阈值近似 topk，避免排序和 scatter
        if num_remote_taus == 0:
            # 所有 block 都被 local/sink/recent 覆盖，没有远程延迟可选，返回空 mask
            final_tau_mask = torch.zeros(H, 0, dtype=torch.bool, device=device)
        else:
            quantile_val = 1.0 - topk_ratio
            threshold_val = torch.quantile(corr_remote.float(), quantile_val, dim=-1, keepdim=True)
            final_tau_mask = corr_remote >= threshold_val
        corr_normalized = None

    elif selection_mode == "threshold":
        # 阈值模式：选择归一化相关性超过阈值的延迟
        corr_max = corr_remote.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        corr_normalized = corr_remote / corr_max
        final_tau_mask = corr_normalized >= corr_threshold
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}, expected 'threshold' or 'topk'")

    # ==================== Step 4: 构建 block mask ====================
    # 创建位置索引
    idx_i = torch.arange(N, device=device).view(N, 1)  # query block 索引
    idx_j = torch.arange(N, device=device).view(1, N)  # key block 索引
    tau_matrix = idx_i - idx_j  # [N, N]，tau_matrix[i,j] = i - j 表示延迟

    # 因果约束：query 只能 attend 到之前的 key
    causal_mask = tau_matrix >= 0  # [N, N]

    # 排除 sink 区域（sink 由 prior mask 处理）
    non_sink_mask = idx_j >= keep_sink  # [1, N]

    # 将 tau 映射到远程索引空间
    tau_to_remote_idx = tau_matrix - remote_start  # [N, N]
    valid_remote = (tau_to_remote_idx >= 0) & (tau_to_remote_idx < num_remote_taus)  # [N, N]

    # 安全索引（无效位置用 0 填充，后续会被 valid_remote 过滤）
    safe_idx = tau_to_remote_idx.clamp(0, num_remote_taus - 1)  # [N, N]

    # 根据 final_tau_mask 构建每个 head 的 mask
    corr_mask = final_tau_mask[:, safe_idx.view(-1)].view(H, N, N)  # [H, N, N]
    corr_mask = corr_mask & valid_remote.unsqueeze(0)  # 过滤无效位置

    # 应用因果约束和排除 sink 区域
    corr_mask = corr_mask & causal_mask.unsqueeze(0) & non_sink_mask.unsqueeze(0)  # [H, N, N]

    return corr_mask.unsqueeze(0), corr, corr_sum, corr_before_shift, corr_normalized, corr_before_weight





# =============================================================================
# 增强版主函数
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
        # 数据驱动 mask 参数
        enable_correlation_mask=True,
        correlation_selection_mode: str = "threshold",  # "threshold" 或 "topk"
        correlation_topk_ratio: float = 0.2,
        corr_threshold: float = 0.1,
        collect_corr_stats: bool = False,

        # 最后一个 block 参数
        enable_last_block_mask=True,
        last_block_threshold=0.01,

        # 可视化参数
        attention_vis_heads=[0],
        attention_vis_dir = ROOT_DIR / "vis_attn",
):
    """
    执行增强版 Block-Sparse Attention（仅用于 prefilling 阶段）

    融合两种稀疏策略：
    1. 先验 Mask：sink + recent + local window（保证基础性能）
    2. 数据驱动 Mask：基于 FFT 互相关的动态选择（捕获长程依赖）

    最终 Mask = 先验 Mask OR 数据驱动 Mask
    """
    assert query_states.is_cuda, "输入张量必须位于 CUDA 上"
    bsz = 1
    L_q = query_states.shape[2]
    L_k = key_states.shape[2]
    H = query_states.shape[1]
    head_dim = query_states.shape[3]
    device = query_states.device

    # ========== 阶段 1: 张量重塑 ==========
    q_unpad = query_states.squeeze(0).transpose(0, 1).contiguous()  # [L_q, H, D]
    k_unpad = key_states.squeeze(0).transpose(0, 1).contiguous()  # [L_k, H, D]
    v_unpad = value_states.squeeze(0).transpose(0, 1).contiguous()  # [L_k, H, D]

    # ========== 阶段 2: 计算 block 数量 ==========
    q_block_num = (L_q + block_size - 1) // block_size
    k_block_num = (L_k + block_size - 1) // block_size

    if q_block_num > 1 and IS_DEBUG:
        print(f"查询长度 L_q: {L_q}, 键长度 L_k: {L_k}, 头数 H: {H}, "
              f"查询 block 数: {q_block_num}, 键 block 数: {k_block_num}")

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
    )  # [1, H, q_block_num, k_block_num]



    # ========== 阶段 3A-2: 最后一个 block 的特殊 mask ==========
    if enable_last_block_mask:
        Q_b = block_mean_pool(q_unpad, block_size)
        K_b = block_mean_pool(k_unpad, block_size)
        last_block_mask = create_last_block_query_mask(Q_b, K_b, threshold=last_block_threshold)  # [1, H, 1, N]

        # 合并到 prior_mask 的最后一行
        prior_mask[:, :, -1:, :] = prior_mask[:, :, -1:, :] | last_block_mask




    # ========== 阶段 3B: 构建数据驱动 Mask（基于互相关） ==========
    correlation_top_k = max(1, int(q_block_num * correlation_topk_ratio))

    if enable_correlation_mask and q_block_num > correlation_top_k:
        # Block-level mean pooling
        Q_b = block_mean_pool(q_unpad, block_size)  # [N_q, H, D]
        K_b = block_mean_pool(k_unpad, block_size)  # [N_k, H, D]

        # 创建 correlation-based mask（排除所有 prior_mask 已覆盖的区域）
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

        # 融合：先验 OR 数据驱动
        final_mask = prior_mask | corr_mask

        ''' 
        if layer_idx == 0:
            if correlation_selection_mode == "topk":
                print(f"使用融合稀疏 mask（先验 + 数据驱动 TopK，ratio={correlation_topk_ratio:.0%}）...")
            else:
                print(f"使用融合稀疏 mask（先验 + 数据驱动阈值，threshold={corr_threshold}）...")
        '''

        
    else:
        final_mask = prior_mask
        corr_mask = None
        if layer_idx == 0:
            print("使用先验稀疏 mask（sink + recent + 局部窗口）...")

    final_mask = final_mask.contiguous()





    # 收集稀疏率（供校准使用）--------------------------------------------------------------------------------------
    if hasattr(sys.modules[__name__], '_sparsity_collector'):
        if corr_mask is not None:
            # 校准阶段：只看 corr_mask 的保留率
            corr_total = corr_mask.numel()
            corr_kept = corr_mask.sum().item()
            corr_retention = corr_kept / corr_total
            _sparsity_collector['corr'].append(corr_retention)
        # 始终记录 final_mask 的真实保留率
        total = final_mask.numel()
        kept = final_mask.sum().item()
        final_retention = kept / total
        _sparsity_collector['final'].append(final_retention)




    # 打印保留率 --------------------------------------------------------------------------------------
    '''
    if layer_idx == 15:
        total = final_mask.numel()
        kept = final_mask.sum().item()
        retention = kept / total
        print(f"  [Layer {layer_idx}] 保留率: {retention:.4f} | 保留块: {kept}/{total}")
    '''




    # 可视化掩码 =====================================================================================================
    if q_unpad.shape[0] > 200 and is_visual and corr_mask is not None:
        total = final_mask.numel()
        kept = final_mask.sum().item()
        ratio = kept / total
        print(f"  => 第 {layer_idx} 层: 数据驱动 block 掩码保留率: {ratio:.4f} ({kept}/{total})")


        print("H:",H)
        vis_heads = [int(x) for x in attention_vis_heads.split(",")]
        for head_idx in vis_heads:
            if head_idx < H:
                attn_scores = compute_attention_scores(q_unpad, k_unpad, head_idx, apply_causal=True)

                # 可视化1：token-level attention分数热力图 + block mask绿框叠加 + mask稀疏率
                visualize_attention_with_block_mask(
                    attn_scores, final_mask, block_size, layer_idx, head_idx,
                    attention_vis_dir, apply_softmax=False
                )

                # 可视化2：完整attention vs 稀疏attention对比（raw scores + softmax dropped分析）
                visualize_attention_comparison(
                    attn_scores, final_mask, block_size, layer_idx, head_idx, attention_vis_dir
                )

                # 可视化3：block均值池化后的attention score热力图 + 选中block标记 + 选中/未选中分数分布
                remote_start_val = max(local_window, keep_recent - 1) + 1
                visualize_block_level_attention(
                    Q_b, K_b, final_mask, layer_idx, head_idx, attention_vis_dir,
                    corr_causal_sum=corr_causal_sum, corr_causal=corr_causal,
                    corr_before_shift=corr_before_shift,
                    corr_before_weight=corr_before_weight,
                    corr_normalized_remote=corr_normalized_remote, remote_start=remote_start_val
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
    )  # [L_q, H, D]

    # ========== 阶段 6: 恢复输出格式 ==========
    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)  # [1, H, L_q, D]
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)  # [1, L_q, H*D]

    return attn_output
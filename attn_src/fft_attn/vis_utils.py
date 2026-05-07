# vis_utils.py

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def visualize_stripe_variance(scores, sample_taus, per_tau_var, per_tau_len,
                              mean_var, is_stripe, stripe_threshold,
                              layer_idx, save_dir="./vis_attn",
                              diag_raw=None, diag_diff=None,
                              diag_mean_list=None, diff_var_list=None,
                              display_head=0,
                              actual_row_start=None,
                              actual_diag_len=None
                              ):
    """
    可视化条带/弥散判断的详细信息。

    布局 (1 + num_taus) 行：
      第1行 3 列:
        第1列: block-level QK 点积矩阵（取 display_head），标注采样的对角线位置
        第2列: 每条采样对角线的归一化差分方差热力图 (H x num_taus)
        第3列: 每个 head 的平均归一化差分方差柱状图
      第2行 ~ 第(1+num_taus)行:
        每行对应一条采样对角线，跨整行，展示 display_head 的：
          raw:  该对角线上所有块的原始值
          diff: 所有块的差分值
          diag_var / diff_var / normalized 汇总统计
    """

    os.makedirs(save_dir, exist_ok=True)

    H, N, _ = scores.shape
    num_taus = len(sample_taus)
    per_tau_var_np = per_tau_var.detach().cpu().numpy()
    mean_var_np = mean_var.detach().cpu().numpy()
    is_stripe_np = is_stripe.detach().cpu().numpy()

    # 布局: 第1行高度=1, 后续每行高度=0.4
    num_rows = 1 + num_taus
    height_ratios = [1.0] + [0.1] * num_taus
    fig = plt.figure(figsize=(30, 10 + num_taus * 1))

    gs = fig.add_gridspec(num_rows, 3, hspace=0.4, wspace=0.3, height_ratios=height_ratios)

    # ===================== 第1行: 原有 3 列 =====================

    # === 第1行第1列：block-level 点积矩阵 + 采样对角线标注 ===
    ax_scores = fig.add_subplot(gs[0, 0])
    scores_h = scores[display_head].detach().cpu().numpy()
    scores_display = np.where(np.isinf(scores_h), np.nan, scores_h)

    im0 = ax_scores.imshow(scores_display, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_scores.set_title(f'Block-Level Scores (Head {display_head})\n+ Sampled Diagonals', fontsize=11)
    ax_scores.set_xlabel('Key Block Index')
    ax_scores.set_ylabel('Query Block Index')
    plt.colorbar(im0, ax=ax_scores, fraction=0.046, pad=0.04)

    colors = plt.cm.Set1(np.linspace(0, 1, max(num_taus, 2)))

    for idx, tau in enumerate(sample_taus):
        tau_int = int(tau)
        dlen = per_tau_len[idx]  # 真实长度 = N - tau - 1
        i_coords = np.arange(tau_int, tau_int + dlen)
        j_coords = i_coords - tau_int
        ax_scores.plot(j_coords, i_coords, color=colors[idx], linewidth=2.0, alpha=0.9,
                       label=f'tau={tau_int} (n={dlen})')



    ax_scores.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 0.0), ncol=1, framealpha=0.8)

    # === 第1行第2列：归一化差分方差热力图 ===
    ax_heatmap = fig.add_subplot(gs[0, 1])
    im1 = ax_heatmap.imshow(per_tau_var_np, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_heatmap.set_title(f'Normalized Diff Variance per Head per tau\n(Yellow=Stripe, Red=Diffuse)', fontsize=11)
    ax_heatmap.set_xlabel('Sampled Diagonal tau')
    ax_heatmap.set_ylabel('Head Index')
    ax_heatmap.set_xticks(range(num_taus))
    ax_heatmap.set_xticklabels([f'tau={t}\n(n={l})' for t, l in zip(sample_taus, per_tau_len)], fontsize=8)
    ax_heatmap.set_yticks(range(H))
    ax_heatmap.set_yticklabels([f'H{h}' for h in range(H)], fontsize=8)
    plt.colorbar(im1, ax=ax_heatmap, fraction=0.046, pad=0.04)

    for h in range(H):
        for t in range(num_taus):
            val = per_tau_var_np[h, t]
            text_color = 'white' if val > per_tau_var_np.max() * 0.6 else 'black'
            axes_font = max(5, min(8, 200 // (H * num_taus)))
            ax_heatmap.text(t, h, f'{val:.3f}', ha='center', va='center',
                            fontsize=axes_font, color=text_color, fontfamily='monospace')

    for h in range(H):
        mode = 'S' if is_stripe_np[h] else 'D'
        color = 'blue' if is_stripe_np[h] else 'red'
        ax_heatmap.text(num_taus + 0.3, h, mode, ha='left', va='center',
                        fontsize=9, fontweight='bold', color=color)

    # === 第1行第3列：柱状图 ===
    ax_bar = fig.add_subplot(gs[0, 2])
    head_indices = np.arange(H)
    bar_colors = ['steelblue' if s else 'indianred' for s in is_stripe_np]

    ax_bar.bar(head_indices, mean_var_np, color=bar_colors, edgecolor='gray', linewidth=0.5)
    ax_bar.axhline(y=stripe_threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold={stripe_threshold:.4f}')
    ax_bar.set_title(f'Mean Normalized Diff Variance per Head\nLayer {layer_idx}', fontsize=11)
    ax_bar.set_xlabel('Head Index')
    ax_bar.set_ylabel('Mean Normalized Diff Variance')
    ax_bar.set_xticks(head_indices)
    ax_bar.set_xticklabels([f'H{h}' for h in range(H)], fontsize=7, rotation=45)

    for h in range(H):
        ax_bar.text(h, mean_var_np[h] + mean_var_np.max() * 0.02,
                    f'{mean_var_np[h]:.3f}', ha='center', va='bottom',
                    fontsize=max(5, min(7, 300 // H)), fontfamily='monospace')

    stripe_patch = patches.Patch(color='steelblue', label=f'Stripe ({int(is_stripe_np.sum())} heads)')
    diffuse_patch = patches.Patch(color='indianred', label=f'Diffuse ({int((~is_stripe_np).sum())} heads)')
    ax_bar.legend(handles=[stripe_patch, diffuse_patch, ax_bar.lines[0]], fontsize=9)

    # ===================== 第2行起：每条对角线的详细中间值 =====================
    if diag_raw is not None:
        for idx, tau in enumerate(sample_taus):
            ax = fig.add_subplot(gs[1 + idx, :])  # 跨整行
            ax.axis('off')

            raw_vals = diag_raw[idx][display_head].detach().cpu().numpy()
            diff_vals = diag_diff[idx][display_head].detach().cpu().numpy()
            d_mean_sq = diag_mean_list[idx][display_head].item()
            df_var = diff_var_list[idx][display_head].item()
            n_var = per_tau_var_np[display_head, idx]

            # 每个值保留2位小数
            raw_str = " ".join([f"{v:>6.2f}" for v in raw_vals])
            diff_str = " ".join([f"{v:>6.2f}" for v in diff_vals])
            # diff 比 raw 少一个元素，前面补空格对齐
            diff_str_aligned = "       " + diff_str

            lines = []
            lines.append(f"--- tau={tau}  n={per_tau_len[idx]}  Head {display_head} ---")
            lines.append(f"raw:  [{raw_str}]")
            lines.append(f"diff: [{diff_str_aligned}]")
            lines.append(f"diag_mean={d_mean_sq:.4f}  diff_var={df_var:.4f}  normalized={n_var:.4f}")

            text = "\n".join(lines)
            # 字体大小根据元素数量自适应
            fontsize = max(4, min(8, 600 // per_tau_len[idx]))
            ax.text(0.01, 0.95, text, transform=ax.transAxes,
                    fontsize=fontsize, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.15))
            # 左侧用对应颜色标记
            ax.axvline(x=0.005, ymin=0.05, ymax=0.95, color=colors[idx],
                       linewidth=4)

    # 底部统计信息
    num_stripe = int(is_stripe_np.sum())
    num_diffuse = H - num_stripe
    fig.text(0.5, 0.005,
             f'Layer {layer_idx} | Head {display_head} | Threshold={stripe_threshold:.4f} | '
             f'{num_stripe} Stripe / {num_diffuse} Diffuse | '
             f'Sampled tau={sample_taus}',
             ha='center', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.subplots_adjust(bottom=0.03)
    save_path = os.path.join(save_dir, f'layer{layer_idx}_stripe_variance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  => stripe/diffuse variance saved: {save_path}")
    return save_path


def compute_attention_scores(query_states, key_states, head_idx=0, apply_causal=True):
    """计算未分块的 QK^T 注意力分数矩阵。"""
    if query_states.dim() == 4:
        q = query_states.squeeze(0).transpose(0, 1)
        k = key_states.squeeze(0).transpose(0, 1)
    else:
        q = query_states
        k = key_states

    q_h = q[:, head_idx, :]
    k_h = k[:, head_idx, :]
    d = q_h.shape[-1]
    attn_scores = torch.matmul(q_h.float(), k_h.float().T) / (d ** 0.5)

    if apply_causal:
        L_q, L_k = attn_scores.shape
        causal_mask = torch.triu(torch.ones(L_q, L_k, device=attn_scores.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    return attn_scores


def visualize_attention_with_block_mask(attn_scores, block_mask, block_size, layer_idx,
                                        head_idx=0, save_dir="./vis_attn", apply_softmax=True,
                                        figsize=(18, 6), downsample_threshold=1024):
    """可视化注意力分数矩阵，并用框标记 block mask 的位置。"""
    os.makedirs(save_dir, exist_ok=True)

    L_q, L_k = attn_scores.shape
    attn_np = attn_scores.detach().cpu().numpy()
    mask_h = block_mask[0, head_idx].detach().cpu().numpy()
    N_q, N_k = mask_h.shape

    if apply_softmax:
        attn_finite = np.where(np.isinf(attn_np), -1e9, attn_np)
        attn_exp = np.exp(attn_finite - attn_finite.max(axis=1, keepdims=True))
        attn_softmax = attn_exp / (attn_exp.sum(axis=1, keepdims=True) + 1e-9)
        attn_display = attn_softmax
        cmap_attn = 'viridis'
        title_suffix = "(Softmax)"
    else:
        attn_display = np.where(np.isinf(attn_np), np.nan, attn_np)
        cmap_attn = 'RdBu_r'
        title_suffix = "(Raw Scores)"

    downsample_factor = 1
    if L_q > downsample_threshold or L_k > downsample_threshold:
        downsample_factor = max(L_q, L_k) // downsample_threshold + 1
        new_L_q = L_q // downsample_factor
        new_L_k = L_k // downsample_factor
        attn_display = attn_display[:new_L_q * downsample_factor, :new_L_k * downsample_factor]
        attn_display = attn_display.reshape(new_L_q, downsample_factor, new_L_k, downsample_factor)
        attn_display = np.nanmean(attn_display, axis=(1, 3))
        block_size_display = block_size / downsample_factor
    else:
        block_size_display = block_size

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax1 = axes[0]
    im1 = ax1.imshow(attn_display, aspect='auto', cmap=cmap_attn, interpolation='nearest')
    ax1.set_title(f'Layer {layer_idx} Head {head_idx}\nAttention Scores {title_suffix}', fontsize=10)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(attn_display, aspect='auto', cmap=cmap_attn, interpolation='nearest')
    ax2.set_title(f'Attention + Block Mask Overlay\n(Green: Active Blocks)', fontsize=10)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')

    for i in range(N_q):
        for j in range(N_k):
            if mask_h[i, j]:
                x_start = j * block_size_display
                y_start = i * block_size_display
                rect = patches.Rectangle(
                    (x_start - 0.5, y_start - 0.5),
                    block_size_display, block_size_display,
                    linewidth=0.5, edgecolor='lime', facecolor='none', alpha=0.8
                )
                ax2.add_patch(rect)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = axes[2]
    im3 = ax3.imshow(mask_h.astype(float), aspect='auto', cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
    ax3.set_title(f'Block Mask\n(Blocks: {N_q}x{N_k}, Size: {block_size})', fontsize=10)
    ax3.set_xlabel('Key Block Index')
    ax3.set_ylabel('Query Block Index')

    total_causal = N_q * (N_q + 1) // 2
    active_blocks = mask_h.sum()
    sparsity = active_blocks / total_causal if total_causal > 0 else 0
    ax3.text(0.02, 0.98, f'Sparsity: {sparsity:.2%}', transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}_attn_with_mask.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  => 注意力可视化已保存: {save_path}")
    return save_path







def visualize_block_level_attention(Q_b, K_b, block_mask, layer_idx, head_idx=0,
                                    save_dir="./vis_attn", figsize=(48, 14),
                                    corr_causal_sum=None, corr_causal=None,
                                    corr_before_shift=None,
                                    corr_before_weight=None,
                                    corr_normalized_remote=None,
                                    remote_start=0,
                                    attn_scores=None,
                                    block_size=128,
                                    stripe_variance=None,
                                    stripe_threshold=None,
                                    is_stripe=None):
    """
    可视化 block-level（均值池化后）的 attention score，并标记被选中的 block。

    固定 4 列布局：
      第1列: token-level 原始注意力分数热力图
      第2列: block-level 均值池化 QK 点积热力图
      第3列: block scores + mask overlay + 底部 corr 值
      第4列: 被选中 vs 未选中 block 的分数分布直方图

    Args:
        Q_b: [N, H, D] block-level query
        K_b: [N, H, D] block-level key
        block_mask: [1, H, N, N] bool mask
        layer_idx: 层索引
        head_idx: 头索引
        save_dir: 保存目录
        corr_causal_sum: [H, N] FFT 互相关原始求和
        corr_causal: [H, N] FFT 互相关归一化后（均值）
        corr_before_shift: [H, N] 偏移前（减去最小值之前）
        corr_before_weight: [H, N] 加权前
        corr_normalized_remote: [H, num_remote] 远程归一化相关性
        remote_start: 远程区域起始 τ
        attn_scores: [L_q, L_k] token-level 原始注意力分数矩阵
        block_size: block 大小，用于 token-level 图的下采样
        stripe_variance: [H] 每个 head 的条带方差
        stripe_threshold: 条带/弥散判断阈值
        is_stripe: [H] bool，每个 head 是否为条带模式
    """
    os.makedirs(save_dir, exist_ok=True)

    N, H, D = Q_b.shape

    # 计算 block-level attention score
    q_h = Q_b[:, head_idx, :].float()  # [N, D]
    k_h = K_b[:, head_idx, :].float()  # [N, D]
    block_scores = torch.matmul(q_h, k_h.T) / (D ** 0.5)  # [N, N]

    # causal mask
    causal = torch.tril(torch.ones(N, N, device=Q_b.device))
    block_scores = block_scores.masked_fill(causal == 0, float('-inf'))

    block_scores_np = block_scores.detach().cpu().numpy()
    block_scores_display = np.where(np.isinf(block_scores_np), np.nan, block_scores_np)

    mask_h = block_mask[0, head_idx].detach().cpu().numpy()  # [N, N]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # === 第1列：token-level 原始注意力分数 ===
    attn_np = attn_scores.detach().cpu().numpy()
    attn_display = np.where(np.isinf(attn_np), np.nan, attn_np)

    # 下采样，避免图太大
    L_q, L_k = attn_display.shape
    downsample_threshold = 1024
    if L_q > downsample_threshold or L_k > downsample_threshold:
        factor = max(L_q, L_k) // downsample_threshold + 1
        new_L_q = L_q // factor
        new_L_k = L_k // factor
        attn_display = attn_display[:new_L_q * factor, :new_L_k * factor]
        attn_display = attn_display.reshape(new_L_q, factor, new_L_k, factor)
        attn_display = np.nanmean(attn_display, axis=(1, 3))

    im0 = axes[0].imshow(attn_display, aspect='equal', cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title(f'Token-Level Attention Scores\nLayer {layer_idx} Head {head_idx}\n(Raw QK^T/√d)',
                      fontsize=10)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # === 第2列：block-level raw scores ===
    im1 = axes[1].imshow(block_scores_display, aspect='equal', cmap='RdBu_r', interpolation='nearest')
    axes[1].set_title(f'Block-Level Attention Scores\nLayer {layer_idx} Head {head_idx}\n(Mean Pooled Q·K/√d)',
                      fontsize=10)
    axes[1].set_xlabel('Key Block Index')
    axes[1].set_ylabel('Query Block Index')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 在第2列图上打印条带/弥散判断信息
    if stripe_variance is not None:
        info_lines = []
        conc_val = stripe_variance[head_idx].item()
        info_lines.append(f"stripe_variance: {conc_val:.4f}")
        if stripe_threshold is not None:
            info_lines.append(f"threshold: {stripe_threshold:.4f}")
        if is_stripe is not None:
            mode = "STRIPE" if is_stripe[head_idx].item() else "DIFFUSE"
            info_lines.append(f"mode: {mode}")
        info_text = "\n".join(info_lines)
        axes[1].text(0.02, 0.98, info_text,
                     transform=axes[1].transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # === 第3列：scores + mask overlay ===
    im2 = axes[2].imshow(block_scores_display, aspect='equal', cmap='RdBu_r', interpolation='nearest')
    axes[2].set_title('Block Scores + Mask Overlay\n(Green: Selected Blocks)', fontsize=10)
    axes[2].set_xlabel('Key Block Index', labelpad=40)
    axes[2].set_ylabel('Query Block Index')

    for i in range(N):
        for j in range(N):
            if mask_h[i, j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1.0, edgecolor='lime', facecolor='none', alpha=0.9
                )
                axes[2].add_patch(rect)

    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # === 在第3列图下方，对齐每个 block 打印 corr 值 ===
    if corr_causal_sum is not None and corr_causal is not None:
        corr_sum_h = corr_causal_sum[head_idx].detach().cpu().numpy()
        corr_norm_h = corr_causal[head_idx].detach().cpu().numpy()

        corr_before_shift_h = None
        if corr_before_shift is not None:
            corr_before_shift_h = corr_before_shift[head_idx].detach().cpu().numpy()

        corr_before_weight_h = None
        if corr_before_weight is not None:
            corr_before_weight_h = corr_before_weight[head_idx].detach().cpu().numpy()

        fig.subplots_adjust(bottom=0.28)

        ax = axes[2]
        for t in range(N):
            x_pos = N - 1 - t
            font_size = max(3, min(6, 180 // N))

            # 第1行：corr_sum（蓝色）
            ax.text(x_pos, -0.01, f"{corr_sum_h[t]:.3f}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=font_size, fontfamily='monospace',
                    ha='center', va='top', rotation=90, color='blue')

            # 第2行：corr_before_shift（橙色）
            if corr_before_shift_h is not None:
                ax.text(x_pos, -0.04, f"{corr_before_shift_h[t]:.3f}",
                        transform=ax.get_xaxis_transform(),
                        fontsize=font_size, fontfamily='monospace',
                        ha='center', va='top', rotation=90, color='orange')

            # 第3行：corr_before_weight（橙色）
            if corr_before_weight_h is not None:
                ax.text(x_pos, -0.07, f"{corr_before_weight_h[t]:.3f}",
                        transform=ax.get_xaxis_transform(),
                        fontsize=font_size, fontfamily='monospace',
                        ha='center', va='top', rotation=90, color='orange')

            # 第4行：corr_avg（红色）
            ax.text(x_pos, -0.10, f"{corr_norm_h[t]:.3f}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=font_size, fontfamily='monospace',
                    ha='center', va='top', rotation=90, color='red')

            # 第5行：corr_normalized_remote（绿色）
            if corr_normalized_remote is not None:
                remote_idx = t - remote_start
                if 0 <= remote_idx < corr_normalized_remote.shape[1]:
                    val = f"{corr_normalized_remote[head_idx, remote_idx].item():.3f}"
                else:
                    val = "  -  "
            else:
                val = "  -  "
            ax.text(x_pos, -0.13, val,
                    transform=ax.get_xaxis_transform(),
                    fontsize=font_size, fontfamily='monospace',
                    ha='center', va='top', rotation=90, color='green')

        # 标签
        ax.text(-0.02, -0.01, "sum:", transform=ax.get_xaxis_transform(),
                fontsize=7, fontfamily='monospace', ha='right', va='top', color='blue')
        ax.text(-0.02, -0.04, "shift:", transform=ax.get_xaxis_transform(),
                fontsize=7, fontfamily='monospace', ha='right', va='top', color='orange')
        ax.text(-0.02, -0.07, "weight:", transform=ax.get_xaxis_transform(),
                fontsize=7, fontfamily='monospace', ha='right', va='top', color='orange')
        ax.text(-0.02, -0.10, "avg:", transform=ax.get_xaxis_transform(),
                fontsize=7, fontfamily='monospace', ha='right', va='top', color='red')
        ax.text(-0.02, -0.13, "norm:", transform=ax.get_xaxis_transform(),
                fontsize=7, fontfamily='monospace', ha='right', va='top', color='green')



    # === 第4列：被选中 vs 未选中的分数分布 ===
    causal_mask = np.tril(np.ones((N, N), dtype=bool))
    selected_scores = block_scores_np[mask_h & causal_mask]
    unselected_scores = block_scores_np[~mask_h & causal_mask]

    selected_scores = selected_scores[np.isfinite(selected_scores)]
    unselected_scores = unselected_scores[np.isfinite(unselected_scores)]

    if len(selected_scores) > 0:
        axes[3].hist(selected_scores, bins=40, alpha=0.7, label=f'Selected ({len(selected_scores)})', color='green')
    if len(unselected_scores) > 0:
        axes[3].hist(unselected_scores, bins=40, alpha=0.7, label=f'Unselected ({len(unselected_scores)})', color='red')

    axes[3].set_xlabel('Block Attention Score')
    axes[3].set_ylabel('Count')
    axes[3].set_title('Score Distribution\nSelected vs Unselected Blocks', fontsize=10)
    axes[3].legend(fontsize=9)

    if len(selected_scores) > 0 and len(unselected_scores) > 0:
        axes[3].text(0.02, 0.98,
                     f'Selected mean: {selected_scores.mean():.2f}\nUnselected mean: {unselected_scores.mean():.2f}',
                     transform=axes[3].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if corr_causal_sum is not None:
        bottom_margin = 0.35 if corr_before_shift is not None else 0.30
        fig.subplots_adjust(bottom=bottom_margin)
    save_path = os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}_block_level_attn.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  => Block-level 注意力可视化已保存: {save_path}")
    return save_path

























def visualize_attention_comparison(attn_scores, block_mask, block_size, layer_idx,
                                   head_idx=0, save_dir="./vis_attn/output", figsize=(20, 5)):
    """可视化对比：完整注意力 vs 稀疏注意力"""
    os.makedirs(save_dir, exist_ok=True)

    L_q, L_k = attn_scores.shape
    attn_np = attn_scores.detach().cpu().numpy()
    mask_h = block_mask[0, head_idx].detach().cpu()
    N_q, N_k = mask_h.shape

    token_mask = torch.zeros(L_q, L_k, dtype=torch.bool)
    for i in range(N_q):
        for j in range(N_k):
            if mask_h[i, j]:
                q_start, q_end = i * block_size, min((i + 1) * block_size, L_q)
                k_start, k_end = j * block_size, min((j + 1) * block_size, L_k)
                token_mask[q_start:q_end, k_start:k_end] = True
    token_mask_np = token_mask.numpy()

    # ===== 前两列：raw scores =====
    attn_raw = np.where(np.isinf(attn_np), np.nan, attn_np)
    attn_raw_sparse = attn_raw.copy()
    attn_raw_sparse[~token_mask_np] = np.nan

    # ===== 后两列：softmax =====
    attn_finite = np.where(np.isinf(attn_np), -1e9, attn_np)
    attn_exp = np.exp(attn_finite - attn_finite.max(axis=1, keepdims=True))
    attn_softmax = attn_exp / (attn_exp.sum(axis=1, keepdims=True) + 1e-9)
    attn_dropped = attn_softmax.copy()
    attn_dropped[token_mask_np] = 0

    max_display = 512
    if L_q > max_display or L_k > max_display:
        factor = max(L_q, L_k) // max_display + 1

        def downsample(arr):
            new_h, new_w = L_q // factor, L_k // factor
            arr = arr[:new_h * factor, :new_w * factor]
            return arr.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))

        attn_raw_display = downsample(np.nan_to_num(attn_raw, nan=0))
        attn_raw_sparse_display = downsample(np.nan_to_num(attn_raw_sparse, nan=0))
        attn_dropped_display = downsample(attn_dropped)
    else:
        attn_raw_display = attn_raw
        attn_raw_sparse_display = attn_raw_sparse
        attn_dropped_display = attn_dropped

    # 把上三角设为 nan，不参与颜色映射
    causal_raw = np.tril(np.ones(attn_raw_display.shape, dtype=bool))
    attn_raw_display[~causal_raw] = np.nan
    causal_sparse = np.tril(np.ones(attn_raw_sparse_display.shape, dtype=bool))
    attn_raw_sparse_display[~causal_sparse] = np.nan

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 第1列：raw scores 完整
    im1 = axes[0].imshow(attn_raw_display, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title(f'Full Attention (Raw Scores)\nLayer {layer_idx} Head {head_idx}', fontsize=10)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 第2列：raw scores 稀疏
    im2 = axes[1].imshow(attn_raw_sparse_display, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[1].set_title('Sparse Attention (Raw Scores)\n(Kept by Mask)', fontsize=10)
    axes[1].set_xlabel('Key Position')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 第3列：softmax dropped
    im3 = axes[2].imshow(attn_dropped_display, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[2].set_title('Dropped Attention (Softmax)\n(Masked Out)', fontsize=10)
    axes[2].set_xlabel('Key Position')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 第4列：softmax dropped per query
    dropped_per_row = attn_dropped.sum(axis=1)
    axes[3].plot(dropped_per_row, linewidth=0.5)
    axes[3].fill_between(range(len(dropped_per_row)), dropped_per_row, alpha=0.3)
    axes[3].set_title('Dropped Softmax Weight per Query\n(Lower is Better)', fontsize=10)
    axes[3].set_xlabel('Query Position')
    axes[3].set_ylabel('Sum of Dropped Weights')
    axes[3].set_xlim(0, len(dropped_per_row))

    mean_drop, max_drop = dropped_per_row.mean(), dropped_per_row.max()
    axes[3].axhline(y=mean_drop, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_drop:.4f}')
    axes[3].legend(fontsize=8)
    axes[3].text(0.98, 0.98, f'Max: {max_drop:.4f}', transform=axes[3].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}_attn_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  => 注意力对比可视化已保存: {save_path}")
    return save_path


# =============================================================================
# 新增：Correlation Mask 准确性评估
# =============================================================================

def expand_block_mask_to_token(block_mask, block_size, L_q, L_k):
    """将 block-level mask 扩展到 token-level。"""
    N_q, N_k = block_mask.shape
    token_mask = np.zeros((L_q, L_k), dtype=bool)
    for i in range(N_q):
        for j in range(N_k):
            if block_mask[i, j]:
                q_start, q_end = i * block_size, min((i + 1) * block_size, L_q)
                k_start, k_end = j * block_size, min((j + 1) * block_size, L_k)
                token_mask[q_start:q_end, k_start:k_end] = True
    return token_mask


def compute_diagonal_attention_sum(attn_softmax, N, block_size, exclude_tau=0, keep_sink=0):
    """计算每条对角线（延迟 τ）上的真实注意力权重总和。"""
    L_q, L_k = attn_softmax.shape
    diag_attn = {}

    for tau in range(N):
        if tau <= exclude_tau:
            continue
        total = 0.0
        count = 0
        for i in range(N):
            j = i - tau
            if j < keep_sink or j < 0:
                continue
            q_start, q_end = i * block_size, min((i + 1) * block_size, L_q)
            k_start, k_end = j * block_size, min((j + 1) * block_size, L_k)
            if q_start < L_q and k_start < L_k:
                total += attn_softmax[q_start:q_end, k_start:k_end].sum()
                count += 1
        if count > 0:
            diag_attn[tau] = total
    return diag_attn


def compute_oracle_taus(attn_softmax, N, block_size, num_taus, exclude_tau=0, keep_sink=0):
    """基于真实注意力，计算最优的对角线选择（Oracle）。"""
    diag_attn = compute_diagonal_attention_sum(attn_softmax, N, block_size, exclude_tau, keep_sink)
    if not diag_attn:
        return [], {}
    sorted_taus = sorted(diag_attn.keys(), key=lambda t: diag_attn[t], reverse=True)
    oracle_taus = sorted_taus[:num_taus]
    return oracle_taus, diag_attn


def evaluate_correlation_mask(attn_scores, corr_mask, prior_mask, fft_corr, top_taus,
                              block_size, head_idx, keep_sink, keep_recent, local_window,
                              layer_idx, save_dir="./vis_attn"):
    """
    全面评估 Correlation Mask 的准确性。

    评估指标：
    1. τ 选择重叠度：FFT 选择的 τ 与 Oracle 最优 τ 的重叠比例
    2. 远程注意力召回率：捕获了多少远程高注意力
    3. 远程注意力覆盖率：覆盖了多少远程注意力权重
    4. FFT 相关性 vs 真实注意力的 Pearson 相关系数
    """
    os.makedirs(save_dir, exist_ok=True)

    L_q, L_k = attn_scores.shape
    N_q = (L_q + block_size - 1) // block_size
    N = N_q

    # 计算 softmax 注意力
    attn_np = attn_scores.detach().cpu().numpy()
    attn_finite = np.where(np.isinf(attn_np), -1e9, attn_np)
    attn_exp = np.exp(attn_finite - attn_finite.max(axis=1, keepdims=True))
    attn_softmax = attn_exp / (attn_exp.sum(axis=1, keepdims=True) + 1e-9)

    # 获取 mask
    corr_mask_h = corr_mask[0, head_idx].detach().cpu().numpy()
    prior_mask_h = prior_mask[0, head_idx].detach().cpu().numpy()

    # 获取 FFT 选择的 τ
    fft_taus = top_taus[head_idx].detach().cpu().tolist()
    num_selected_taus = len(fft_taus)
    exclude_tau = max(local_window, keep_recent - 1)

    # ========== 指标 1: τ 选择重叠度 ==========
    oracle_taus, diag_attn = compute_oracle_taus(
        attn_softmax, N, block_size, num_selected_taus, exclude_tau, keep_sink
    )
    tau_overlap = len(set(fft_taus) & set(oracle_taus)) / len(oracle_taus) if oracle_taus else 0.0

    # ========== 指标 2 & 3: 远程区域的召回率和覆盖率 ==========
    corr_token = expand_block_mask_to_token(corr_mask_h, block_size, L_q, L_k)
    prior_token = expand_block_mask_to_token(prior_mask_h, block_size, L_q, L_k)

    causal = np.tril(np.ones((L_q, L_k), dtype=bool))
    remote_region = causal & ~prior_token

    recall, coverage = 0.0, 0.0
    threshold = 0.0
    if remote_region.sum() > 0:
        remote_values = attn_softmax[remote_region]
        if len(remote_values) > 0:
            threshold = np.percentile(remote_values, 90)
            important_remote = (attn_softmax >= threshold) & remote_region
            captured = important_remote & corr_token
            recall = captured.sum() / important_remote.sum() if important_remote.sum() > 0 else 0
            remote_total_attn = attn_softmax[remote_region].sum()
            corr_covered_attn = attn_softmax[corr_token & remote_region].sum()
            coverage = corr_covered_attn / remote_total_attn if remote_total_attn > 0 else 0

    # ========== 指标 4: FFT 相关性 vs 真实注意力的相关系数 ==========
    fft_corr_h = fft_corr[head_idx].detach().cpu().numpy()
    remote_start = exclude_tau + 1
    correlation_coef = 0.0
    fft_values, true_values, tau_indices = [], [], []

    if remote_start < N and diag_attn:
        for tau in range(remote_start, N):
            if tau in diag_attn:
                fft_values.append(fft_corr_h[tau])
                true_values.append(diag_attn[tau])
                tau_indices.append(tau)
        if len(fft_values) > 1:
            correlation_coef = np.corrcoef(fft_values, true_values)[0, 1]
            if np.isnan(correlation_coef):
                correlation_coef = 0.0

    # ========== 可视化评估结果 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 子图 1: FFT 相关性 vs 真实注意力对比
    ax1 = axes[0, 0]
    if tau_indices:
        x = np.arange(len(tau_indices))
        width = 0.35
        fft_norm = np.array(fft_values)
        fft_norm = (fft_norm - fft_norm.min()) / (fft_norm.max() - fft_norm.min() + 1e-9)
        true_norm = np.array(true_values)
        true_norm = (true_norm - true_norm.min()) / (true_norm.max() - true_norm.min() + 1e-9)

        ax1.bar(x - width / 2, fft_norm, width, label='FFT Correlation (norm)', alpha=0.8)
        ax1.bar(x + width / 2, true_norm, width, label='True Attention (norm)', alpha=0.8)

        for i, tau in enumerate(tau_indices):
            if tau in fft_taus:
                ax1.axvline(x=i, color='green', linestyle='--', alpha=0.5, linewidth=1)
            if tau in oracle_taus:
                ax1.scatter(i, 1.05, marker='*', color='red', s=100, zorder=5)

        ax1.set_xlabel('τ index (in remote region)')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title(f'FFT Correlation vs True Attention\nPearson r = {correlation_coef:.4f}')
        ax1.legend()
        step = max(1, len(x) // 10)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([str(tau_indices[i]) for i in range(0, len(tau_indices), step)])
    else:
        ax1.text(0.5, 0.5, 'No remote τ data', ha='center', va='center', transform=ax1.transAxes)

    # 子图 2: τ 选择对比
    ax2 = axes[0, 1]
    all_taus = sorted(set(fft_taus) | set(oracle_taus))
    if all_taus:
        fft_set, oracle_set = set(fft_taus), set(oracle_taus)
        overlap_set = fft_set & oracle_set
        colors = ['green' if t in overlap_set else 'blue' if t in fft_set else 'red' for t in all_taus]

        ax2.bar(range(len(all_taus)), [1] * len(all_taus), color=colors)
        ax2.set_xticks(range(len(all_taus)))
        ax2.set_xticklabels([str(t) for t in all_taus], rotation=45)
        ax2.set_xlabel('τ value')
        ax2.set_title(f'τ Selection Comparison\nOverlap: {tau_overlap:.1%}')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='FFT Only'),
            Patch(facecolor='red', label='Oracle Only'),
            Patch(facecolor='green', label='Overlap'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No τ data', ha='center', va='center', transform=ax2.transAxes)

    # 子图 3: 远程区域注意力分布
    ax3 = axes[1, 0]
    if remote_region.sum() > 0:
        remote_attn_flat = attn_softmax[remote_region]
        ax3.hist(remote_attn_flat, bins=50, alpha=0.7, label='Remote Attention Distribution')
        if threshold > 0:
            ax3.axvline(x=threshold, color='r', linestyle='--', label=f'Top 10% Threshold')
        ax3.set_xlabel('Attention Weight')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Remote Attention Distribution\nRecall: {recall:.1%}, Coverage: {coverage:.1%}')
        ax3.legend()
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'No remote region', ha='center', va='center', transform=ax3.transAxes)

    # 子图 4: 评估指标汇总
    ax4 = axes[1, 1]
    ax4.axis('off')

    report_text = f"""
        ============ Correlation Mask 评估报告 ============
        Layer {layer_idx} Head {head_idx}

        【τ 选择准确性】
          FFT 选择的 τ:     {fft_taus[:8]}{'...' if len(fft_taus) > 8 else ''}
          Oracle 最优 τ:    {oracle_taus[:8]}{'...' if len(oracle_taus) > 8 else ''}
          重叠度:           {tau_overlap:.1%}

        【远程注意力捕获】
          Top-10% 召回率:   {recall:.1%}
          注意力覆盖率:     {coverage:.1%}

        【FFT 与真实注意力相关性】
          Pearson 相关系数: {correlation_coef:.4f}

        【诊断】"""

    diagnostics = []
    if tau_overlap < 0.3:
        diagnostics.append("  ⚠️ τ 选择重叠度低，FFT 互相关可能不准确")
    if recall < 0.3:
        diagnostics.append("  ⚠️ 远程高注意力召回率低，可能遗漏重要依赖")
    if correlation_coef < 0.5:
        diagnostics.append("  ⚠️ FFT 相关性与真实注意力相关性弱")
    if tau_overlap >= 0.5 and recall >= 0.5 and correlation_coef >= 0.5:
        diagnostics.append("  ✓ Correlation Mask 表现良好")
    if not diagnostics:
        diagnostics.append("  - 需要更多数据进行诊断")

    report_text += "\n" + "\n".join(diagnostics)
    report_text += "\n================================================"

    ax4.text(0.05, 0.95, report_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}_corr_mask_eval.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  => Correlation Mask 评估报告已保存: {save_path}")
    print(f"\n  ====== Layer {layer_idx} Head {head_idx} Correlation Mask 评估 ======")
    print(
        f"    τ 重叠度: {tau_overlap:.1%} | 召回率: {recall:.1%} | 覆盖率: {coverage:.1%} | 相关系数: {correlation_coef:.4f}")
    print(f"    FFT τ: {fft_taus[:5]}{'...' if len(fft_taus) > 5 else ''}")
    print(f"    Oracle τ: {oracle_taus[:5]}{'...' if len(oracle_taus) > 5 else ''}")

    return {
        'tau_overlap': tau_overlap,
        'recall': recall,
        'coverage': coverage,
        'correlation_coef': correlation_coef,
        'fft_taus': fft_taus,
        'oracle_taus': oracle_taus,
    }
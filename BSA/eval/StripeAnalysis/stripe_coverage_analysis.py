#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stripe_coverage_analysis.py

独立脚本：加载模型 & 数据集，提取 QK attention logit 矩阵，
分析平行于主对角线的条纹（stripe）对高分值的覆盖率。

输出：
  1) 四张对比图：原图 / 条纹mask / 保留图 / 残差图
  2) 覆盖率曲线：横轴=条纹比例，纵轴=覆盖率，多条线对应不同 top-p 阈值

使用方式（LVEval）：
  python stripe_coverage_analysis.py \
      --model llama-3.1-8b-instruct \
      --task dureader_mixup_32k \
      --layer 2 --head 0 \
      --downsample 99999 \
      --out_dir ./stripe_analysis \
      --model2path ../../config/model2path.json \
      --data_root /root/cjh/pro/resources/datasets/LVEval/dureader_mixup
"""

import sys
from pathlib import Path

# 加项目根目录到 sys.path（与 pred_vis.py 保持一致）
ROOT_DIR = Path(__file__).resolve().parents[0]
# 如果脚本放在 eval/VisAttn/ 下，改为 parents[2]
# ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import os
import json
import math
import types
import argparse
import gc
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# 参数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stripe coverage analysis on QK attention logits")
    p.add_argument("--model", type=str, required=True, help="model key in model2path.json")
    p.add_argument("--task", type=str, required=True,
                   help="LVEval task name, e.g. dureader_mixup_32k")
    p.add_argument("--layer", type=int, default=0, help="which layer to analyze")
    p.add_argument("--head", type=int, default=0, help="which head to analyze")
    p.add_argument("--downsample", type=int, default=512,
                   help="downsample size for plotting only (analysis uses full resolution)")
    p.add_argument("--max_length", type=int, default=9000, help="tokenizer max_length")
    p.add_argument("--out_dir", type=str, default="./stripe_analysis")
    p.add_argument("--sample_idx", type=int, default=0, help="which sample in dataset")
    p.add_argument("--model2path", type=str, default=None,
                   help="path to model2path.json")
    p.add_argument("--data_root", type=str,
                   default="/root/cjh/pro/resources/datasets/LVEval/dureader_mixup",
                   help="directory containing {task}.jsonl files")
    return p.parse_args()


# =============================================================================
# Hook：在指定层截获 Q、K
# =============================================================================

class QKCaptured(Exception):
    """截获到 Q、K 后抛出，用于提前终止 forward 以节省显存。"""
    pass


class QKCatcher:
    """用 forward hook 截获指定层的 query_states 和 key_states。"""

    def __init__(self, target_layer: int, target_head: int):
        self.target_layer = target_layer
        self.target_head = target_head
        self.q = None
        self.k = None

    def hook_fn(self, module, args, output):
        pass  # 实际截获在 patched forward 里做


def make_capture_forward(original_forward, catcher, target_layer, target_head):
    """
    包装原始 attention forward：
      - 目标层：只做 Q/K 投影 + RoPE，截获后抛异常终止
      - 非目标层：跳过 attention 计算，返回零向量，节省显存
    因为我们只需要可视化 QK logit，不关心模型输出是否正确。
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, nn

    @torch.no_grad()
    def patched_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
    ):
        bsz, q_len, hidden_dim = hidden_states.size()

        if self.layer_idx == target_layer:
            # ★ 目标层：做 Q/K 投影 + RoPE，截获后终止
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                # 只需要一个 dummy tensor 来获取 cos/sin
                cos, sin = self.rotary_emb(hidden_states[:, :, :self.head_dim], position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            catcher.q = query_states[0, target_head].float().cpu()  # [L, D]
            catcher.k = key_states[0, target_head].float().cpu()
            del query_states, key_states
            torch.cuda.empty_cache()
            raise QKCaptured(f"Q/K captured at layer {target_layer}, head {target_head}")
        else:
            # ★ 非目标层：跳过 attention，返回零向量通过 o_proj
            # 这样 hidden_states 形状正确，模型可以继续 forward 到目标层
            dummy = torch.zeros(bsz, q_len, hidden_dim, dtype=hidden_states.dtype,
                                device=hidden_states.device)
            return dummy, None, past_key_value

    return patched_forward


# =============================================================================
# QK logit 矩阵计算 & 降采样
# =============================================================================

def compute_logit_matrix(q, k):
    """
    q, k: [L, D]  float32 cpu tensors
    返回全尺寸 logit 矩阵 (numpy)，causal mask 外的位置为 nan。
    """
    L, D = q.shape
    print(f"  [logit matrix] L={L}, D={D}, 计算 Q·K^T / sqrt(D) ...")
    t0 = time.time()
    scores = torch.matmul(q, k.T) / math.sqrt(D)  # [L, L]
    print(f"  [logit matrix] Q·K^T 完成, 耗时 {time.time() - t0:.1f}s, 矩阵大小 {L}x{L}")

    # causal mask: 上三角 (q < k) 设为 nan
    print(f"  [logit matrix] 应用 causal mask ...")
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("nan"))
    vis = scores.numpy().astype(np.float32)
    del scores, causal
    gc.collect()
    print(f"  [logit matrix] 完成, 总耗时 {time.time() - t0:.1f}s")
    return vis


def downsample_matrix(vis, downsample_size=512):
    """
    将全尺寸矩阵降采样到 downsample_size x downsample_size，仅用于画图。
    返回 (vis_small, factor)。
    """
    L = vis.shape[0]
    factor = max(1, L // downsample_size)
    if factor > 1:
        s = L // factor
        vis_small = np.nanmean(
            vis[:s * factor, :s * factor].reshape(s, factor, s, factor),
            axis=(1, 3),
        )
        return vis_small, factor
    else:
        return vis.copy(), 1


# =============================================================================
# 条纹分析核心
# =============================================================================

def analyze_stripes(vis, s):
    """
    按对角线偏移 d = q_idx - k_idx 聚合。
    d=0 是主对角线，d>0 表示 query 在 key 后面 d 个位置。

    返回：
      diag_mean: shape [s], diag_mean[d] = 偏移为 d 的对角线上所有非nan值的均值
      diag_rank: 按 diag_mean 降序排列的 d 值数组
    """
    t0 = time.time()
    diag_mean = np.full(s, np.nan, dtype=np.float32)
    log_interval = max(1, s // 10)

    for d in range(s):
        diag_vals = np.array([vis[i, i - d] for i in range(d, s)], dtype=np.float32)
        valid = diag_vals[~np.isnan(diag_vals)]
        if len(valid) > 0:
            diag_mean[d] = np.mean(valid)
        if (d + 1) % log_interval == 0 or d == s - 1:
            print(f"  [analyze_stripes] 对角线聚合: {d + 1}/{s} ({(d + 1) / s * 100:.0f}%), "
                  f"耗时 {time.time() - t0:.1f}s", flush=True)

    # 按均值降序排列
    valid_mask = ~np.isnan(diag_mean)
    ranked_indices = np.argsort(-np.where(valid_mask, diag_mean, -np.inf))
    print(f"  [analyze_stripes] 完成, 总耗时 {time.time() - t0:.1f}s")
    return diag_mean, ranked_indices


def compute_coverage(vis, s, diag_rank, stripe_fracs, top_p_list):
    """
    逐行 top-p 覆盖率统计。

    对每一行（query position），找出该行 logit 值最大的 top_p 比例的位置，
    然后统计这些位置中有多少被条纹覆盖。

    stripe_fracs: list of float, e.g. [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    top_p_list:   list of float, e.g. [0.02, 0.05, 0.10]

    返回：
      coverage_global: dict[top_p] -> list of float (整体覆盖率, one per stripe_frac)
      coverage_per_row: dict[top_p] -> dict[stripe_frac] -> np.array shape [s] (每行覆盖率)
    """
    # 预计算每行的 top-p 高分位置 mask
    # high_mask[top_p] 是 bool 矩阵 [s, s]，标记每行的 top-p 位置
    high_mask = {}
    row_top_count = {}  # 每行有多少个 top-p 位置
    t0 = time.time()
    log_interval = max(1, s // 10)

    for tp_idx, top_p in enumerate(top_p_list):
        print(f"  [coverage] 预计算逐行 top-{int(top_p * 100)}% mask ({tp_idx + 1}/{len(top_p_list)}) ...",
              flush=True)
        mask = np.zeros((s, s), dtype=bool)
        counts = np.zeros(s, dtype=np.int32)
        for i in range(s):
            row = vis[i, :]
            valid_idx = np.where(~np.isnan(row))[0]
            if len(valid_idx) == 0:
                continue
            valid_vals = row[valid_idx]
            k = max(1, int(top_p * len(valid_idx)))
            # 找出该行最大的 k 个位置
            top_k_pos = valid_idx[np.argpartition(valid_vals, -k)[-k:]]
            mask[i, top_k_pos] = True
            counts[i] = k
            if (i + 1) % log_interval == 0 or i == s - 1:
                print(f"    行进度: {i + 1}/{s} ({(i + 1) / s * 100:.0f}%), "
                      f"耗时 {time.time() - t0:.1f}s", flush=True)
        high_mask[top_p] = mask
        row_top_count[top_p] = counts

    # 对每个 stripe_frac，构建条纹 mask，计算逐行和整体覆盖率
    coverage_global = {tp: [] for tp in top_p_list}
    coverage_per_row = {tp: {} for tp in top_p_list}

    n_fracs = len(stripe_fracs)
    for frac_idx, frac in enumerate(stripe_fracs):
        print(f"  [coverage] stripe_frac={frac * 100:.0f}% ({frac_idx + 1}/{n_fracs}), "
              f"耗时 {time.time() - t0:.1f}s", flush=True)
        stripe_mask = build_stripe_mask(s, diag_rank, frac)

        for top_p in top_p_list:
            # 逐行覆盖率
            row_coverage = np.zeros(s, dtype=np.float32)
            for i in range(s):
                n_high = row_top_count[top_p][i]
                if n_high == 0:
                    row_coverage[i] = np.nan
                    continue
                covered = (stripe_mask[i, :] & high_mask[top_p][i, :]).sum()
                row_coverage[i] = covered / n_high
            coverage_per_row[top_p][frac] = row_coverage

            # 整体覆盖率：所有行的高分位置汇总
            total_high = high_mask[top_p].sum()
            total_covered = (stripe_mask & high_mask[top_p]).sum()
            rate = total_covered / total_high if total_high > 0 else 0.0
            coverage_global[top_p].append(rate)

    print(f"  [coverage] 全部完成, 总耗时 {time.time() - t0:.1f}s")
    return coverage_global, coverage_per_row


def build_stripe_mask(s, diag_rank, frac):
    """构建条纹 mask：选 top frac 比例的对角线。"""
    K = max(1, int(frac * s))
    selected = set(diag_rank[:K].tolist())
    mask = np.zeros((s, s), dtype=bool)
    for d in selected:
        for i in range(d, s):
            mask[i, i - d] = True
    return mask


# =============================================================================
# 可视化
# =============================================================================

VIS_MAX_SIZE = 1024  # 热力图渲染的最大分辨率


def downsample_for_plot(mat, max_size=VIS_MAX_SIZE):
    """将矩阵降采样到 max_size x max_size 以加速 imshow 渲染。
    返回 (降采样后矩阵, 缩放因子)。如果不需要降采样则原样返回。"""
    s = mat.shape[0]
    if s <= max_size:
        return mat, 1
    f = s // max_size
    new_s = s // f
    small = np.nanmean(
        mat[:new_s * f, :new_s * f].reshape(new_s, f, new_s, f),
        axis=(1, 3),
    )
    return small, f


def plot_three_panels(vis, stripe_mask, layer, head, factor, out_dir, frac_label="10%"):
    s = vis.shape[0]

    kept = vis.copy()
    kept[~stripe_mask] = np.nan

    residual = vis.copy()
    residual[stripe_mask] = np.nan

    valid = vis[~np.isnan(vis)]
    vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99)

    mid = (vmin + vmax) / 2
    residual_pos = residual.copy()
    residual_pos[residual_pos < mid] = np.nan

    # 降采样以加速渲染
    data_list_full = [vis, kept, residual, residual_pos]
    data_list = []
    vis_factor = 1
    for d in data_list_full:
        d_small, vis_factor = downsample_for_plot(d)
        data_list.append(d_small)
    total_factor = factor * vis_factor
    plot_s = data_list[0].shape[0]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    titles = [
        "Original",
        f"Kept (stripes only, top {frac_label})",
        "Residual (non-stripe)",
        "Residual positive only",
    ]

    for ax, title, data in zip(axes, titles, data_list):
        im = ax.imshow(data, aspect="equal", cmap="RdBu_r", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xlabel("Key position", fontsize=20, fontweight="bold")
        ax.set_ylabel("Query position", fontsize=20, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n_ticks = 4
        tick_pos = np.linspace(0, plot_s - 1, n_ticks)
        tick_labels = [str(int(t * total_factor)) for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=9)

    plt.suptitle(f"Layer {layer}, Head {head} — Stripe Coverage (top {frac_label} diagonals)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"three_panels_L{layer}_H{head}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 对比图已保存: {path}")


def plot_coverage_curve(coverage_global, stripe_fracs, top_p_list, layer, head, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    for idx, top_p in enumerate(top_p_list):
        label = f"Top {int(top_p * 100)}% values (per-row)"
        ax.plot(np.array(stripe_fracs) * 100, np.array(coverage_global[top_p]) * 100,
                marker="o", linewidth=2, color=colors[idx % len(colors)],
                label=label, markersize=5)

    ax.plot(np.array(stripe_fracs) * 100, np.array(stripe_fracs) * 100,
            linestyle="--", color="gray", linewidth=1.5, label="Random baseline")

    ax.set_xlabel("Selected stripes (% of total diagonals)", fontsize=16)
    ax.set_ylabel("Coverage (%)", fontsize=16)
    ax.set_title(f"Stripe Coverage — Layer {layer}, Head {head}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=15, loc="lower right")
    ax.set_xlim(0, max(stripe_fracs) * 100 + 2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, f"coverage_curve_L{layer}_H{head}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 覆盖率曲线已保存: {path}")


def plot_per_row_coverage(coverage_per_row, top_p_list, stripe_frac, factor, layer, head, out_dir):
    """
    逐行覆盖率图。

    横轴：query position（行号）
    纵轴：该行的 top-p 高分位置中被条纹覆盖的比例
    多条线对应不同的 top_p。
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for idx, top_p in enumerate(top_p_list):
        row_cov = coverage_per_row[top_p][stripe_frac]
        s = len(row_cov)
        x = np.arange(s) * factor
        ax.plot(x, row_cov * 100, linewidth=0.6, color=colors[idx % len(colors)],
                alpha=0.7, label=f"Top {int(top_p * 100)}% per row")

    ax.set_xlabel("Query position (token index)", fontsize=14)
    ax.set_ylabel("Per-row coverage (%)", fontsize=14)
    ax.set_title(f"Per-Row Stripe Coverage — Layer {layer}, Head {head} "
                 f"(top {int(stripe_frac * 100)}% stripes)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, f"per_row_coverage_L{layer}_H{head}_sf{int(stripe_frac * 100)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 逐行覆盖率图已保存: {path}")


def plot_per_row_coverage_summary(coverage_per_row, top_p_list, stripe_fracs, layer, head, out_dir):
    """
    逐行覆盖率的统计汇总图：box plot / violin plot。

    每个 stripe_frac 一组，每个 top_p 一个 box，展示所有行覆盖率的分布。
    """
    n_fracs = len(stripe_fracs)
    fig, axes = plt.subplots(1, n_fracs, figsize=(5 * n_fracs, 5), squeeze=False)
    axes = axes[0]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for ax_idx, frac in enumerate(stripe_fracs):
        ax = axes[ax_idx]
        data = []
        labels = []
        for tp in top_p_list:
            row_cov = coverage_per_row[tp][frac]
            valid = row_cov[~np.isnan(row_cov)] * 100
            data.append(valid)
            labels.append(f"Top {int(tp * 100)}%")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                        showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors[:len(top_p_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # 标注 mean
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.plot(i + 1, mean_val, 'D', color='black', markersize=5)
            ax.annotate(f"{mean_val:.1f}%", (i + 1, mean_val),
                        textcoords="offset points", xytext=(12, 0), fontsize=9)

        ax.set_title(f"Stripes = {int(frac * 100)}%", fontsize=13, fontweight="bold")
        ax.set_ylabel("Per-row coverage (%)", fontsize=12)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Per-Row Coverage Distribution — Layer {layer}, Head {head}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"per_row_coverage_boxplot_L{layer}_H{head}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 逐行覆盖率箱线图已保存: {path}")


def plot_diag_mean_profile(diag_mean, diag_rank, factor, layer, head, out_dir, frac=0.10):
    s = len(diag_mean)
    K = max(1, int(frac * s))
    top_k_d = set(diag_rank[:K].tolist())

    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(s) * factor

    ax.plot(x, diag_mean, linewidth=0.5, color="steelblue", alpha=0.4)

    selected_mask = np.array([d in top_k_d for d in range(s)])
    sel_mean = np.where(selected_mask, diag_mean, np.nan)
    ax.plot(x, sel_mean, linewidth=0.8, color="red", alpha=0.7, label=f"Selected top {int(frac * 100)}%")

    ax.set_xlabel("Diagonal offset d (token distance)", fontsize=16)
    ax.set_ylabel("Mean logit on diagonal", fontsize=16)
    ax.set_title(f"Diagonal Mean Profile — Layer {layer}, Head {head} (red = top {int(frac * 100)}% diagonals)",
                 fontsize=19, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, f"diag_profile_L{layer}_H{head}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 对角线均值分布已保存: {path}")


def plot_combined_figure(vis, stripe_mask, coverage_global, stripe_fracs, top_p_list,
                         layer, head, factor, out_dir, frac_label="20%"):
    s = vis.shape[0]

    kept = vis.copy()
    kept[~stripe_mask] = np.nan
    residual = vis.copy()
    residual[stripe_mask] = np.nan

    valid = vis[~np.isnan(vis)]
    vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99)

    # 降采样以加速渲染
    data_list_full = [vis, kept, residual]
    data_list = []
    vis_factor = 1
    for d in data_list_full:
        d_small, vis_factor = downsample_for_plot(d)
        data_list.append(d_small)
    total_factor = factor * vis_factor
    plot_s = data_list[0].shape[0]

    fig = plt.figure(figsize=(24, 5.5))
    gs_left = GridSpec(1, 1, figure=fig, left=0.03, right=0.30, bottom=0.15, top=0.85)
    gs_right = GridSpec(1, 3, figure=fig, left=0.37, right=0.98, wspace=0.40)

    ax_left = fig.add_subplot(gs_left[0, 0])
    axes_right = [fig.add_subplot(gs_right[0, i]) for i in range(3)]

    # ---- (a) 左1张覆盖率曲线 ----
    ax = ax_left
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for idx, top_p in enumerate(top_p_list):
        label = f"Top {int(top_p * 100)}% QK values"
        ax.plot(np.array(stripe_fracs) * 100, np.array(coverage_global[top_p]) * 100,
                marker="o", linewidth=2.5, color=colors[idx], label=label, markersize=6)

    ax.plot(np.array(stripe_fracs) * 100, np.array(stripe_fracs) * 100,
            linestyle="--", color="gray", linewidth=1.5, label="Random baseline")

    ax.set_xlabel("Selected Stripes (% of Total Diagonals)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Coverage of High-score QK Values (%)", fontsize=15, fontweight="bold")
    ax.set_title("Stripe Coverage Curve", fontsize=16, fontweight="bold", pad=8)
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, max(stripe_fracs) * 100 + 2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # ---- 中间分隔竖线 ----
    line_x = (gs_left.right + gs_right.left) / 2 - 0.01
    fig.add_artist(plt.Line2D(
        [line_x, line_x], [0.08, 0.90],
        transform=fig.transFigure, color="gray",
        linewidth=1.0, linestyle="-", alpha=0.6,
    ))

    # ---- (b) 右3张热力图 ----
    titles = ["Original Attention Logits",
              f"Retained by Top {frac_label} Stripes",
              "Residual (Non-stripe)"]

    for ax, title, data in zip(axes_right, titles, data_list):
        im = ax.imshow(data, aspect="equal", cmap="RdBu_r", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
        ax.set_xlabel("Key Position", fontsize=16, fontweight="bold")
        ax.set_ylabel("Query Position", fontsize=16, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=13)

        n_ticks = 4
        tick_pos = np.linspace(0, plot_s - 1, n_ticks)
        tick_labels = [str(int(t * total_factor)) for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=12)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=12)

    # ---- (a)(b) 标注在正下方居中 ----
    left_center_x = (gs_left.left + gs_left.right) / 2
    fig.text(left_center_x, 0.01, "(a)", fontsize=16, fontweight="bold",
             ha="center", va="bottom")

    right_center_x = (gs_right.left + gs_right.right) / 2
    fig.text(right_center_x, 0.01, "(b)", fontsize=16, fontweight="bold",
             ha="center", va="bottom")

    path = os.path.join(out_dir, f"combined_L{layer}_H{head}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  => 论文合并图已保存: {path}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 60)
    print("Stripe Coverage Analysis")
    print(f"  model:      {args.model}")
    print(f"  task:       {args.task}")
    print(f"  layer:      {args.layer}")
    print(f"  head:       {args.head}")
    print(f"  downsample: {args.downsample}")
    print(f"  max_length: {args.max_length}")
    print(f"  sample_idx: {args.sample_idx}")
    print(f"  out_dir:    {args.out_dir}")
    print("=" * 60)

    # ---- 路径 ----
    if args.model2path:
        model2path_file = args.model2path
    else:
        candidates = [
            str(ROOT_DIR) + "/eval/LongBench/config/model2path.json",
            "./eval/LongBench/config/model2path.json",
        ]
        model2path_file = next((c for c in candidates if os.path.exists(c)), candidates[0])

    print(f"model2path:      {model2path_file}")
    model2path = json.load(open(model2path_file))

    # ---- 加载模型 ----
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = model2path[args.model]
    print(f"\n[1/6] 加载模型: {model_path}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    print(f"  => 模型加载完成, 耗时 {time.time() - t0:.1f}s")

    # ---- Hook: 仅 patch 目标层的 attention 为 eager（截获 Q/K）----
    # 其他层保持 flash attention 2，节省显存
    catcher = QKCatcher(args.layer, args.head)

    target_attn_name = f"model.layers.{args.layer}.self_attn"
    for name, module in model.named_modules():
        if name == target_attn_name:
            patched = make_capture_forward(None, catcher, args.layer, args.head)
            module.forward = types.MethodType(patched, module)
            print(f"  => 已 patch 层 {args.layer} 的 attention (eager mode)")
            break

    # ---- 加载数据（LVEval 格式）----
    print(f"\n[2/6] 加载数据...")
    data_path = os.path.join(args.data_root, f"{args.task}.jsonl")
    print(f"数据集: {data_path}")
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == args.sample_idx:
                sample = json.loads(line)
                break

    # LVEval: 直接用 context + input 拼接 prompt
    prompt = sample["context"] + "\n\n" + sample["input"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=args.max_length).to("cuda")
    seq_len = inputs.input_ids.shape[-1]
    print(f"seq_len = {seq_len}")

    # ---- Prefilling ----
    print(f"\n[3/6] Running prefilling (seq_len={seq_len}) ...")
    t0 = time.time()
    with torch.no_grad():
        try:
            model(**inputs, use_cache=False)
        except QKCaptured as e:
            print(f"  => 提前终止: {e}")
    print(f"  => Prefilling 耗时 {time.time() - t0:.1f}s")

    # 释放模型显存
    del model, inputs
    torch.cuda.empty_cache()
    gc.collect()

    assert catcher.q is not None, "未能截获 Q、K，请检查 layer 参数是否正确"
    print(f"截获 Q shape: {catcher.q.shape}, K shape: {catcher.k.shape}")

    # ---- 计算 QK logit 矩阵（全尺寸，用于精确分析）----
    print(f"\n[4/6] 计算 QK logit 矩阵...")
    vis = compute_logit_matrix(catcher.q, catcher.k)
    del catcher.q, catcher.k
    gc.collect()
    s = vis.shape[0]
    print(f"全尺寸矩阵大小: {s}x{s}")

    # ---- 条纹分析（全尺寸）----
    print(f"\n[5/6] 分析对角线条纹 (s={s}) ...")
    diag_mean, diag_rank = analyze_stripes(vis, s)

    # ---- 覆盖率计算（全尺寸）----
    stripe_fracs = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    top_p_list = [0.02, 0.05, 0.10]

    print(f"\n[6/6] 计算逐行覆盖率 (top_p={[f'{t * 100:.0f}%' for t in top_p_list]}) ...")
    coverage_global, coverage_per_row = compute_coverage(vis, s, diag_rank, stripe_fracs, top_p_list)

    # 打印整体覆盖率表格
    print(f"\n{'Stripe%':>10}", end="")
    for tp in top_p_list:
        print(f"  Top{int(tp * 100):>2}%", end="")
    print()
    print("-" * (10 + 8 * len(top_p_list)))
    for i, frac in enumerate(stripe_fracs):
        print(f"{frac * 100:>9.0f}%", end="")
        for tp in top_p_list:
            print(f"  {coverage_global[tp][i] * 100:>5.1f}%", end="")
        print()

    # ---- 画图用降采样 ----
    plot_ds = args.downsample
    print(f"\n画图降采样: {s}x{s} -> ~{plot_ds}x{plot_ds} ...")
    vis_plot, plot_factor = downsample_matrix(vis, plot_ds)
    s_plot = vis_plot.shape[0]
    print(f"  => 画图矩阵大小: {s_plot}x{s_plot}, factor={plot_factor}")

    # 对角线排序结果也需要映射到降采样尺度（重新在小矩阵上分析）
    diag_mean_plot, diag_rank_plot = analyze_stripes(vis_plot, s_plot)

    # ---- 可视化 ----
    print(f"\n生成可视化图表...")
    t0 = time.time()
    demo_frac = 0.20
    stripe_mask_plot = build_stripe_mask(s_plot, diag_rank_plot, demo_frac)

    print(f"  [图 1/6] 四面板对比图...")
    plot_three_panels(vis_plot, stripe_mask_plot, args.layer, args.head, plot_factor, args.out_dir,
                      frac_label=f"{int(demo_frac * 100)}%")

    print(f"  [图 2/6] 覆盖率曲线...")
    plot_coverage_curve(coverage_global, stripe_fracs, top_p_list, args.layer, args.head, args.out_dir)

    print(f"  [图 3/6] 对角线均值分布...")
    plot_diag_mean_profile(diag_mean_plot, diag_rank_plot, plot_factor, args.layer, args.head, args.out_dir,
                           frac=demo_frac)

    print(f"  [图 4/6] 论文合并图...")
    plot_combined_figure(vis_plot, stripe_mask_plot, coverage_global, stripe_fracs, top_p_list,
                         args.layer, args.head, plot_factor, args.out_dir,
                         frac_label=f"{int(demo_frac * 100)}%")

    print(f"  [图 5/6] 逐行覆盖率折线图...")
    plot_per_row_coverage(coverage_per_row, top_p_list, demo_frac, 1,
                          args.layer, args.head, args.out_dir)

    print(f"  [图 6/6] 逐行覆盖率箱线图...")
    summary_fracs = [0.05, 0.10, 0.20, 0.50]
    plot_per_row_coverage_summary(coverage_per_row, top_p_list, summary_fracs,
                                  args.layer, args.head, args.out_dir)
    print(f"  => 全部图表生成完成, 耗时 {time.time() - t0:.1f}s")

    print(f"\n✅ 分析完成，结果保存在 {args.out_dir}/")
    print(f"   总耗时: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
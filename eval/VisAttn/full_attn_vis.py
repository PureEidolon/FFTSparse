# eval/VisAttn/full_attn_vis.py

import os
import math
import numpy as np
import torch

_DEFAULT_CFG = dict(
    vis_layers    = {0},
    vis_heads     = [0],
    vis_dir       = "./vis_attn",
    vis_seq_min   = 200,
    vis_downsample= 512,
)

def _merge_cfg(user_cfg: dict) -> dict:
    cfg = dict(_DEFAULT_CFG)
    cfg.update(user_cfg or {})
    return cfg

def maybe_vis_full_attn(query_states, key_states, layer_idx: int, cfg: dict = None):
    cfg = _merge_cfg(cfg)

    L = query_states.shape[2]
    if L < cfg["vis_seq_min"]:
        return

    os.makedirs(cfg["vis_dir"], exist_ok=True)

    for head_idx in cfg["vis_heads"]:
        if head_idx >= query_states.shape[1]:
            continue
        _save_one_head(query_states, key_states, layer_idx, head_idx,
                       cfg["vis_dir"], cfg["vis_downsample"])

def _save_one_head(query_states, key_states, layer_idx, head_idx, save_dir, downsample_size=512):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = query_states[0, head_idx].float().cpu()   # [L, D]
    k = key_states[0, head_idx].float().cpu()
    L, D = q.shape

    scores = torch.matmul(q, k.T) / math.sqrt(D)  # [L, L]
    causal = torch.triu(torch.ones(L, L), diagonal=1).bool()
    scores = scores.masked_fill(causal, float("nan"))

    scores_np = scores.numpy().astype(np.float32)
    del scores, q, k

    # 降采样
    factor = max(1, L // downsample_size)
    if factor > 1:
        s   = L // factor
        vis = np.nanmean(
            scores_np[:s * factor, :s * factor].reshape(s, factor, s, factor),
            axis=(1, 3)
        )
    else:
        s   = L
        vis = scores_np
    del scores_np

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    valid = vis[~np.isnan(vis)]
    vmin, vmax = np.percentile(valid, 0.1), np.percentile(valid, 99.9)
    im = ax.imshow(vis, aspect="equal", cmap="RdBu_r", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Key position", fontsize=18, fontweight="bold")
    ax.set_ylabel("Query position", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)

    # 坐标轴显示真实 token 位置
    n_ticks = 4
    tick_pos = np.linspace(0, s - 1, n_ticks)
    tick_labels = [str(int(t * factor)) for t in tick_pos]
    ax.set_xticks(tick_pos);
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticks(tick_pos);
    ax.set_yticklabels(tick_labels, fontsize=18)

    plt.subplots_adjust(left=0.2)
    path = os.path.join(save_dir, f"full_attn_layer{layer_idx}_head{head_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    del vis
    import gc; gc.collect()
    print(f"  => [VisAttn] 已保存: {path}")
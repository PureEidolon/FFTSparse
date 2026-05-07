# src/vis_utils.py

import os
import torch
from matplotlib import pyplot as plt


def visualize_block_mask(
    base_blockmask: torch.Tensor,
    layer_idx: int,
    seq_len: int,
    keep_ratio: float,
    threshold: float,
    keep_sink: float,
    keep_recent: float,
    local_window: float,
    head_idx: int = 0,
    save_dir: str = "./vis_masks",
    prefix: str = "mask"
):
    """
    可视化 block sparse attention mask。

    Args:
        base_blockmask: [1, H, q_blocks, k_blocks] (bool)
        layer_idx: 当前层索引
        seq_len: 输入序列长度（用于命名）
        keep_ratio: 保留比例（用于标题）
        head_idx: 要可视化的 head 索引（默认 0）
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    if base_blockmask.dim() != 4:
        raise ValueError("base_blockmask must be [1, H, q_b, k_b]")

    # 提取指定 head 的 mask
    mask_np = base_blockmask[0, head_idx].cpu().numpy()  # [q_b, k_b]

    plt.figure(figsize=(8, 6))
    plt.imshow(mask_np, cmap='gray_r', aspect='auto', interpolation='none')
    threshold_str = f"{threshold:.3f}" if threshold is not None else "N/A"
    plt.title(f"Block Sparse Mask (Layer {layer_idx}, Head {head_idx})\n"
              f"Keep Ratio: {keep_ratio:.3f} | Q blocks: {mask_np.shape[0]}, K blocks: {mask_np.shape[1]}\n"
              f"threshold: {threshold_str} | keep_sink: {keep_sink} | keep_recent: {keep_recent} | local_window: {local_window}")
    plt.xlabel("Key Blocks")
    plt.ylabel("Query Blocks")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{prefix}_layer{layer_idx}_seqlen{seq_len}_head{head_idx}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  => Saved mask visualization to {filename}")

# src/vis_utils.py （追加到文件末尾）

def visualize_block_scores(
    block_scores: torch.Tensor,
    layer_idx: int,
    seq_len: int,
    threshold: float,
    keep_sink: float,
    keep_recent: float,
    local_window: float,
    head_idx: int = 0,
    save_dir: str = "./vis_masks",
    prefix: str = "scores"
):
    """
    可视化 block 相关性得分（热力图）。

    Args:
        block_scores: [H, q_blocks, k_blocks] (float)
        layer_idx: 当前层索引
        seq_len: 输入序列长度（用于命名）
        head_idx: 要可视化的 head 索引（默认 0）
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    if block_scores.dim() != 3:
        raise ValueError("block_scores must be [H, q_b, k_b]")

    # 提取指定 head 的 scores
    scores_tensor = block_scores[head_idx]
    if scores_tensor.dtype == torch.bfloat16 or scores_tensor.dtype == torch.float16:
        scores_tensor = scores_tensor.float()
    scores_np = scores_tensor.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(scores_np, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label="Score")
    plt.title(f"Block Attention Scores (Layer {layer_idx}, Head {head_idx})\n"
              f"Q blocks: {scores_np.shape[0]}, K blocks: {scores_np.shape[1]}\n"
              f"threshold : {threshold:.3f}  | keep_sink: {keep_sink} | keep_recent: {keep_recent} | local_window: {local_window}\n")
    plt.xlabel("Key Blocks")
    plt.ylabel("Query Blocks")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{prefix}_layer{layer_idx}_seqlen{seq_len}_head{head_idx}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  => Saved block scores visualization to {filename}")
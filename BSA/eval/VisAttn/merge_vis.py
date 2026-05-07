# eval/VisAttn/merge_vis.py

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# =============================================================================
# 配置
# =============================================================================

VIS_ROOT  = "./vis_attn"
SAVE_PATH = "./vis_attn/merged.png"

DATASETS = ["2wikimqa", "gov_report", "lcc", "passage_count"]

DATASET_LABELS = {
    "2wikimqa":          "Multi-doc QA\n(2wikimqa)",
    "gov_report":        "Summarization\n(gov_report)",
    "lcc":               "Code\n(lcc)",
    "passage_count":     "Retrival \n(passage_count)",
}

LAYERS = list(range(32))
#LAYERS = [0,1,2,3,10,11,12,13,20,21,22,23,28,29,30,31]
HEAD   = 0

# =============================================================================
# 拼图：3行(层) x 4列(数据集)
# =============================================================================

n_rows = len(LAYERS)
n_cols = len(DATASETS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

for row, layer in enumerate(LAYERS):
    for col, dataset in enumerate(DATASETS):
        ax = axes[row, col]
        path = os.path.join(VIS_ROOT, f"FULL-{dataset}",
                            f"full_attn_layer{layer}_head{HEAD}.png")
        TARGET_H = 500
        if os.path.exists(path):
            img = Image.open(path)
            w, h = img.size
            new_w = int(w * TARGET_H / h)
            img = np.array(img.resize((new_w, TARGET_H)))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"missing\n{path}",
                    ha="center", va="center", fontsize=7, color="red",
                    transform=ax.transAxes)

        ax.axis("off")

        # 列标题：数据集名（第一行显示）
        if row == 0:
            ax.set_title(DATASET_LABELS[dataset], fontsize=18, pad=8, fontweight="bold")

        # 行标签：层号（第一列显示）
        if col == 0:
            ax.text(-0.1, 0.5, f" Layer {layer}",
                    fontsize=18, fontweight="bold",
                    ha="center", va="center", rotation=90,
                    transform=ax.transAxes)

plt.suptitle("Full Attention Score (pre-softmax)  |  Head 0", fontsize=20, y=1.01)
plt.subplots_adjust(hspace=-0.1, wspace=0, left=0.1)
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"=> 已保存: {SAVE_PATH}")
"""
读取 real_all.csv 和 niah_all.csv,合并求平均,保存 real+niah_all.csv,并画一张加速比柱状图
直接运行: python plot_benchmark.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ============ 固定路径 ============
EVAL_DIR = "eval_result"
REAL_CSV   = os.path.join(EVAL_DIR, "real_all.csv")
NIAH_CSV   = os.path.join(EVAL_DIR, "niah_all.csv")
MERGED_CSV = os.path.join(EVAL_DIR, "real+niah_all.csv")
OUT_PNG    = os.path.join(EVAL_DIR, "plots", "real+niah_speedup.png")


# ============ 配色============
# 参考 ColorBrewer 和 Nature/IEEE 论文常用配色
FIXED_COLORS = {
    "minference": "#767671",   #
    "flex":       "#CCA300",   #
    "xattn_s8":   "#8497B0",   #
    "xattn_s16":  "#004178",   #
}

def get_display_name(method):
    """把内部方法名转换成图表显示用的名字"""
    if method.startswith("myattn_thres"):
        thres = method.replace("myattn_thres", "")
        return f"FFTSparse T={thres}"
    if method.startswith("myattn_topk"):
        topk = method.replace("myattn_topk", "")
        return f"FFTSparse K={topk}"
    if method == "minference":
        return "MInference"
    if method == "minference":
        return "MInference"
    if method == "flex":
        return "FlexPrefill"
    if method == "xattn_s8":
        return "XAttn(S=8)"
    return method   # 其他方法名不变

def get_color(method, all_methods):
    if method in FIXED_COLORS:
        return FIXED_COLORS[method]
    if method.startswith("myattn"):
        myattn_methods = sorted([m for m in all_methods if m.startswith("myattn")])
        idx = myattn_methods.index(method)
        n = len(myattn_methods)
        # myattn 用酒红色系(从浅到深),区别于 xattn 蓝色
        myattn_palette = ["#B03E00", "#6A1900", "#7D1D3F"]   # 浅 → 中 → 深
        if n <= len(myattn_palette):
            return myattn_palette[idx]
        # 超过 3 个 myattn 时退化为 colormap
        cmap = cm.get_cmap("Reds")
        return cmap(0.45 + 0.45 * idx / max(n - 1, 1))
    return "#7F7F7F"


# ============ 1. 读取 + 纵向拼接(新增 Source 列) ============
# ============ 1. 读取 ============
print(f"读取 {REAL_CSV}")
real_df = pd.read_csv(REAL_CSV)
print(f"读取 {NIAH_CSV}")
niah_df = pd.read_csv(NIAH_CSV)

# ---- 1a. 拼接版(带 Source 列),把 speedup 合并进 _ms 列后保存 ----
# ---- 1a. 拼接版 ----
real_tagged = real_df.copy()
niah_tagged = niah_df.copy()
real_tagged.insert(0, "Source", "real")
niah_tagged.insert(0, "Source", "niah")
concat_df = pd.concat([real_tagged, niah_tagged], ignore_index=True)

# 合并 speedup 进 _ms 列
ms_cols = [c for c in concat_df.columns if c.endswith("_ms")]
for ms_col in ms_cols:
    sp_col = ms_col.replace("_ms", "_speedup")
    if sp_col in concat_df.columns:
        concat_df[ms_col] = [
            f"{ms:.2f}({sp:.2f}×)" if pd.notna(ms) and pd.notna(sp) else (
                f"{ms:.2f}" if pd.notna(ms) else ms
            )
            for ms, sp in zip(concat_df[ms_col], concat_df[sp_col])
        ]
        concat_df = concat_df.drop(columns=[sp_col])

# 重排列顺序
COLUMN_ORDER = [
    "full",
    "minference",
    "flex",
    "xattn_s8",
    #"myattn_thres0.75",
    "myattn_thres0.85",
    "myattn_thres0.9",
]
fixed_cols = [c for c in concat_df.columns if not c.endswith("_ms")]
ordered_ms_cols = [f"{m}_ms" for m in COLUMN_ORDER if f"{m}_ms" in concat_df.columns]
extra_ms_cols = [c for c in concat_df.columns if c.endswith("_ms") and c not in ordered_ms_cols]
concat_df = concat_df[fixed_cols + ordered_ms_cols + extra_ms_cols]

# 重命名列(去掉 _ms,用 get_display_name)
rename_map = {}
for c in concat_df.columns:
    if c.endswith("_ms"):
        method = c.replace("_ms", "")
        rename_map[c] = "Full" if method == "full" else get_display_name(method)
concat_df = concat_df.rename(columns=rename_map)

os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
concat_df.to_csv(MERGED_CSV, index=False, encoding="utf-8")


# ---- 1b. 平均版,用于画图(保持原逻辑不变) ----
real_indexed = real_df.set_index("Length")
niah_indexed = niah_df.set_index("Length")

common_cols = real_indexed.columns.intersection(niah_indexed.columns)
common_lens = real_indexed.index.intersection(niah_indexed.index)

real_aligned = real_indexed.loc[common_lens, common_cols]
niah_aligned = niah_indexed.loc[common_lens, common_cols]

merged = (real_aligned + niah_aligned) / 2.0
merged = merged.round(4).reset_index()

def length_to_int(s):
    return int(s.replace("K", "")) if isinstance(s, str) else s
merged = merged.sort_values("Length", key=lambda col: col.map(length_to_int)).reset_index(drop=True)



# ============ 2. 画加速比柱状图 ============
# 想要的方法显示顺序(从左到右),改这里就能调整柱子顺序
METHOD_ORDER = [
    "minference",
    "flex",
    "xattn_s8",
    #"xattn_s16",
    #"myattn_thres0.75",
    "myattn_thres0.85",
    "myattn_thres0.9",
]

# 自动取出 CSV 里所有 speedup 列对应的方法
sp_cols_raw = [c for c in merged.columns if c.endswith("_speedup")]
methods_all = [c.replace("_speedup", "") for c in sp_cols_raw]

# 按 METHOD_ORDER 重排,没在列表里的方法自动放到最后
methods = [m for m in METHOD_ORDER if m in methods_all]
#methods += [m for m in methods_all if m not in METHOD_ORDER]
print("methods:",methods)

sp_cols = [f"{m}_speedup" for m in methods]
lengths = merged["Length"].tolist()

n_methods = len(methods)
n_lengths = len(lengths)
group_width = 0.8           # 每组的总宽度(占 X 轴 1 个单位的 80%)
gap_ratio = 0.15            # 组内柱子间隙占单根柱宽的比例,越大越疏
bar_width = group_width / n_methods * (1 - gap_ratio)
slot_width = group_width / n_methods   # 每根柱子的"槽位宽度"
x = np.arange(n_lengths)




plt.rcParams.update({
    "font.family":      "DejaVu Sans",   # 论文常用,系统都有
    "font.weight":      "bold",          # 全局加粗
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth":   1.5,             # 坐标轴加粗
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
})

fig, ax = plt.subplots(figsize=(max(10, n_lengths * 4), 5))
fig.patch.set_facecolor("white")

for i, (col, method) in enumerate(zip(sp_cols, methods)):
    values = merged[col].values.astype(float)
    offsets = x + (i - n_methods / 2 + 0.5) * slot_width
    color = get_color(method, methods)
    bars = ax.bar(
        offsets, values, bar_width,
        label=get_display_name(method),
        color=color,
        edgecolor="black",
        linewidth=1.2,
    )

    # 柱顶标数值(加粗)
    for bar, v in zip(bars, values):
        if pd.notna(v) and v > 0:
            ypos = v + max(values) * 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.2f}" if v >= 0.01 else f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=15, fontweight="bold",
            )

# baseline 红线
ax.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1.5,
           alpha=0.85, label="baseline (1.0x)", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(lengths, fontsize=18, fontweight="bold")
ax.set_xlabel("Sequence Length", fontsize=18, fontweight="bold")
ax.set_ylabel("Attention Speedup", fontsize=24, fontweight="bold")
ax.set_title("Speedup per Length (Real + Synthetic averaged)",
             fontsize=14, fontweight="bold", pad=18)

# Y 轴刻度加粗
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontsize(18)

# 图例加粗
legend = ax.legend(loc="upper left",
                   fontsize=16, ncol=7,
                   framealpha=0.95, edgecolor="black",
                   handlelength=1.2, handleheight=1.2, handletextpad=0.5)
legend.get_frame().set_linewidth(1.2)
for text in legend.get_texts():
    text.set_fontweight("bold")

ax.grid(True, alpha=0.35, axis="y", linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)   # 网格放到柱子后面

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)


ax.set_ylim(0, max(merged[sp_cols].max()) * 1.15)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")   # dpi 调到 200 更清晰
plt.close()
print(f"\n✅ 图片保存到 {OUT_PNG}")
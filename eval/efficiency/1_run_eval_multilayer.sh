#!/bin/bash
# ============================================================
# Benchmark: 一次性跑完所有长度,每个长度自动选层数
# ============================================================

# 想测的长度列表(空格分隔)
#LENS="8 16 32 64 128"

LENS="128"

# 每个长度对应的层数(LEN:L1,L2,...)
# - A6000 48GB 安全配置
# - 注意:层之间用逗号,LEN和层用冒号,不同长度之间用空格
LAYERS_PER_LEN=(
    "16:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    "32:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    "64:0,3,7,11,15,19,23,27,30,31"
    "128:0,7,15,23,31"
    "256:0,15,31"
)

NUM_WARMUPS=3
NUM_ITERATIONS=10
COLUMN_TOPK_RATIO=0.1

# ============================================================
# MODE 控制
# ============================================================
MODE="threshold"   # topk / threshold / target_sparsity

TOPK_RATIOS="0.05"
THRESHOLDS="0.7 0.75 0.85 0.9"

if [ "$MODE" = "topk" ]; then
    MODE_ARGS="--corr_selection_mode topk --corr_topk_ratios $TOPK_RATIOS"
elif [ "$MODE" = "threshold" ]; then
    MODE_ARGS="--corr_selection_mode threshold --corr_thres $THRESHOLDS"
elif [ "$MODE" = "target_sparsity" ]; then
    MODE_ARGS="--corr_selection_mode target_sparsity --target_sparsity 0.9"
else
    echo "未知 MODE: $MODE"
    exit 1
fi

# 注意:这里的 --layers 是 fallback,只在某长度没出现在 layers_per_len 里时才用
COMMON_ARGS="\
    --use_cor \
    --sink_ratio 0.01 \
    --recent_ratio 0.0 \
    --local_ratio 0.02 \
    --enable_last_block \
    --enable_column_mask \
    --last_block_thres 0.001 \
    --diag_sample_ratio 0.15 \
    --min_diag_samples 5 \
    --max_diag_samples 64 \
    --qk_topk_ratio 0.3 \
    --stripe_threshold 0.3 \
    --num_warmups $NUM_WARMUPS \
    --num_iterations $NUM_ITERATIONS \
    --lens $LENS \
    --layers 0 7 15 23 31"

# ============================================================
# 真实数据
# ============================================================
echo "======================================================================"
echo "===== Benchmark: 真实数据 (MODE=$MODE) ====="
echo "======================================================================"
python benchmark_attention_multilayer.py \
    --data_dir output/real_qk_data \
    --column_topk_ratio 0.2 \
    --save_path eval_result/real_all.json \
    --layers_per_len "${LAYERS_PER_LEN[@]}" \
    $COMMON_ARGS \
    $MODE_ARGS

# ============================================================
# NIAH 数据
# ============================================================
echo "======================================================================"
echo "===== Benchmark: NIAH 数据 (MODE=$MODE) ====="
echo "======================================================================"
python benchmark_attention_multilayer.py \
    --data_dir output/needle_qk_data \
    --column_topk_ratio 0.2 \
    --save_path eval_result/niah_all.json \
    --layers_per_len "${LAYERS_PER_LEN[@]}" \
    $COMMON_ARGS \
    $MODE_ARGS

echo ""
echo "======================================================================"
echo "===== 全部完成 ====="
echo "======================================================================"
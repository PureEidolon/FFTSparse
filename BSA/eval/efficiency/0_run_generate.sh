#!/bin/bash
# ============================================================
# 生成 Real QK + Needle QK 数据
# 每个长度按 benchmark 需要的层数生成,与 benchmark 脚本对齐
# ============================================================

GPU_NAME=$(nvidia-smi --query-gpu=name --id=0 --format=csv,noheader 2>/dev/null)
if [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "检测到 RTX A6000,使用 /backup01 路径"
    MODEL_PATH="/backup01/cjh/projects/resources/models/llama-3.1-8b-instruct"
    DATA_PATH="/backup01/cjh/projects/resources/datasets/InfiniteBench/longbook_sum_eng.jsonl"
    OUTPUT_DIR="/backup01/cjh/projects/BSA_outputs/efficiency"
    export CUDA_VISIBLE_DEVICES=1
else
    echo "检测到 GPU: $GPU_NAME,使用默认 (output) 路径"
    MODEL_PATH="/root/cjh/pro/resources/models/llama-3.1-8b-instruct"
    DATA_PATH="/root/cjh/pro/resources/datasets/InfiniteBench/longbook_sum_eng.jsonl"
    OUTPUT_DIR="output"
fi

echo "GPU:        $GPU_NAME"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"

TEXT_DIR="real_texts"
NEEDLE_POSITION="random"

# ============================================================
# 想跑哪些长度
# ============================================================
LENS_LIST="64"

# ============================================================
# 每个长度对应的层数(必须和 benchmark 脚本完全一致!)
# - 注意:层之间用空格(因为要传给 Python 的 nargs='+')
# ============================================================
get_layers_for_len() {
    case $1 in
        8|16)  echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" ;;
        32)    echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" ;;
        64)    echo "0 3 7 11 15 19 23 27 30 31" ;;
        128)   echo "0 7 15 23 31" ;;
        256)   echo "0 7 15 23 31" ;;
        *)     echo "0 7 15 23 31" ;;
    esac
}

# ============================================================
# Step 1:准备真实文本(一次准备所有长度)
# ============================================================
echo ""
echo "===== Step 1: 准备真实文本 ====="
python prepare_real_texts.py \
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$TEXT_DIR" \
    --target_lens $LENS_LIST

# ============================================================
# Step 2 & 3:每个长度单独生成 QK 数据,自动选层
# ============================================================
for LEN in $LENS_LIST; do
    LAYERS=$(get_layers_for_len $LEN)
    NUM_LAYERS=$(echo $LAYERS | wc -w)

    REAL_OUT="$OUTPUT_DIR/real_qk_data/${LEN}K"
    NEEDLE_OUT="$OUTPUT_DIR/needle_qk_data/${LEN}K"
    mkdir -p "$REAL_OUT" "$NEEDLE_OUT"

    echo ""
    echo "######################################################################"
    echo "###  Length=${LEN}K  |  Layers=${NUM_LAYERS}"
    echo "###  LAYERS: $LAYERS"
    echo "######################################################################"

    echo ""
    echo "===== Step 2: Real QKV ($LEN K) -> $REAL_OUT ====="
    python capture_qk_real.py \
        --model_path "$MODEL_PATH" \
        --text_dir "$TEXT_DIR" \
        --output_dir "$REAL_OUT" \
        --lens $LEN \
        --layers_to_save $LAYERS

    echo ""
    echo "===== Step 3: Needle QK ($LEN K) -> $NEEDLE_OUT ====="
    python capture_qk_synthetic.py \
        --model_path "$MODEL_PATH" \
        --layers $LAYERS \
        --lens $LEN \
        --needle_position $NEEDLE_POSITION \
        --out_dir "$NEEDLE_OUT"
done

echo ""
echo "======================================================================"
echo "===== 全部完成 ====="
echo "Real QKV:    $OUTPUT_DIR/real_qk_data/<XK>"
echo "Synthetic:   $OUTPUT_DIR/needle_qk_data/<XK>"
echo "======================================================================"
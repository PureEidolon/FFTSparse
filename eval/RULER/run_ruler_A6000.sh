#!/bin/bash
# RULER 评测脚本
# 删除Windows 行尾符     sed -i 's/\r$//' run_ruler_A6000.sh

export CUDA_VISIBLE_DEVICES=1,4   # ← 控制用哪块 GPU,多卡用 "0,1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 拼接 RULER 路径：假设 RULER-main 在脚本同级目录下
RULER_DIR="${SCRIPT_DIR}/RULER-main/scripts"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/backup01/cjh/projects/resources/models/llama-3.1-8b-instruct"
MODEL_NAME="llama-3.1-8b-instruct"
TOKENIZER_PATH=${MODEL_PATH}
TOKENIZER_TYPE="hf"
MODEL_TEMPLATE_TYPE="meta-llama3"
NUM_SAMPLES=100

# ============================================================
# 选择方法、序列长度、任务
# ============================================================
METHODS=(
    "full"
    "myattn"
    "xattn"
    #"flex"
    "minference"
)

SEQ_LENS=(
    #8192
    16384
    #32768
    #65536
    #131072
)

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_multikey_1"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

# ============================================================
# myattn 参数网格(同 LongBench 风格)
# ============================================================
SINK_RATIO=0.01
RECENT_RATIO=0.00
LOCAL_RATIO=0.02
BLOCK_SIZE=128

CORR_MODE="threshold"   # threshold 或 topk
if [ "$CORR_MODE" == "threshold" ]; then
    CORR_VALUES=(0.75 0.8 0.85)
else
    CORR_VALUES=(0.05)
fi

QK_TOPK_RATIOS=(0.2)
STRIPE_THRESHOLDS=(0.4)
DIAG_SAMPLE_RATIOS=(0.15)
COLUMN_TOPK_RATIOS=(0.2)
LAST_BLOCK_THRES=0.001

# ============================================================
# 实验名(用于目录隔离)
# ============================================================
get_exp_name() {
    local m=$1
    if [ "$m" == "myattn" ]; then
        echo "myattn-${CORR_MODE}${CORR_VAL}-qk${QK_TOPK}-st${STRIPE}-col${COL}-lb${LAST_BLOCK_THRES}"
    else
        echo "$m"
    fi
}

# ============================================================
# 主循环
# ============================================================
for METHOD in "${METHODS[@]}"; do
  for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for CORR_VAL in "${CORR_VALUES[@]}"; do
      for QK_TOPK in "${QK_TOPK_RATIOS[@]}"; do
        for STRIPE in "${STRIPE_THRESHOLDS[@]}"; do
          for DIAG in "${DIAG_SAMPLE_RATIOS[@]}"; do
            for COL in "${COLUMN_TOPK_RATIOS[@]}"; do

              # 非 myattn 时只跑一次,跳过参数网格的多轮
              if [ "$METHOD" != "myattn" ]; then
                if [ "$CORR_VAL" != "${CORR_VALUES[0]}" ] || \
                   [ "$QK_TOPK" != "${QK_TOPK_RATIOS[0]}" ] || \
                   [ "$STRIPE" != "${STRIPE_THRESHOLDS[0]}" ] || \
                   [ "$DIAG" != "${DIAG_SAMPLE_RATIOS[0]}" ] || \
                   [ "$COL" != "${COLUMN_TOPK_RATIOS[0]}" ]; then
                    continue
                fi
              fi

              EXP_NAME=$(get_exp_name $METHOD)
              RESULTS_DIR="${EVAL_DIR}/results/${MODEL_NAME}/${EXP_NAME}/${SEQ_LEN}"
              DATA_DIR="${RESULTS_DIR}/data"
              PRED_DIR="${RESULTS_DIR}/pred"
              mkdir -p ${DATA_DIR} ${PRED_DIR}

              echo ""
              echo "############################################################"
              echo "  EXP=${EXP_NAME}  SEQ_LEN=${SEQ_LEN}"
              if [ "$METHOD" == "myattn" ]; then
                echo "  corr=${CORR_MODE}-${CORR_VAL} qk=${QK_TOPK} stripe=${STRIPE} diag=${DIAG} col=${COL}"
              fi
              echo "############################################################"

              # ---- 数据生成(行数不足才补,顺便清空旧 pred)----
              for TASK in "${TASKS[@]}"; do
                JSONL="${DATA_DIR}/${TASK}/validation.jsonl"
                CUR=0
                [ -f "${JSONL}" ] && CUR=$(wc -l < "${JSONL}")

                if [ ${CUR} -lt ${NUM_SAMPLES} ]; then
                  echo "▶ 生成数据: ${TASK} (现有 ${CUR}/${NUM_SAMPLES})"
                  cd ${RULER_DIR}
                  python data/prepare.py \
                    --save_dir ${DATA_DIR} \
                    --benchmark synthetic \
                    --task ${TASK} \
                    --tokenizer_path ${TOKENIZER_PATH} \
                    --tokenizer_type ${TOKENIZER_TYPE} \
                    --max_seq_length ${SEQ_LEN} \
                    --model_template_type ${MODEL_TEMPLATE_TYPE} \
                    --num_samples ${NUM_SAMPLES}
                  # 数据重新生成,旧 pred 已失效,删掉
                  rm -f "${PRED_DIR}/${TASK}.jsonl"
                  echo "  ↳ 已清空旧 pred(index 集合变了)"
                else
                  echo "✓ 数据已足够: ${TASK} (${CUR} 条)"
                fi
              done

              # ---- 推理(每个 task 单独调用,容错) ----
              for TASK in "${TASKS[@]}"; do
                echo "▶ 推理: ${TASK}"
                python -u ${EVAL_DIR}/pred.py \
                  --ruler_scripts ${RULER_DIR} \
                  --model_path ${MODEL_PATH} \
                  --data_dir ${DATA_DIR} \
                  --save_dir ${PRED_DIR} \
                  --task ${TASK} \
                  --num_samples ${NUM_SAMPLES} \
                  --method ${METHOD} \
                  --sink_ratio ${SINK_RATIO} \
                  --recent_ratio ${RECENT_RATIO} \
                  --local_ratio ${LOCAL_RATIO} \
                  --block_size ${BLOCK_SIZE} \
                  --corr_selection_mode ${CORR_MODE} \
                  --corr_thres ${CORR_VAL} \
                  --corr_topk_ratio ${CORR_VAL} \
                  --column_topk_ratio ${COL} \
                  --diag_sample_ratio ${DIAG} \
                  --stripe_threshold ${STRIPE} \
                  --qk_topk_ratio ${QK_TOPK} \
                  --last_block_thres ${LAST_BLOCK_THRES} \
                  --use_cor \
                  --enable_last_block \
                  --enable_column_mask
              done


              echo "✅ 推理完成: ${PRED_DIR}"

            done
          done
        done
      done
    done
  done
done

echo ""
echo "============================================================"
echo "📊 评估并汇总所有结果"
echo "============================================================"
python ${EVAL_DIR}/summarize.py --results_dir ${EVAL_DIR}/results

echo ""
echo "============================================================"
echo "全部完成: ${EVAL_DIR}/results/${MODEL_NAME}/"
echo "============================================================"
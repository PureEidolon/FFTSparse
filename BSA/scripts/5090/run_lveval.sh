#!/bin/bash
models=("llama-3.1-8b-instruct")

# -1表示全部
NUM_SAMPLES=-1

DATA_DIR="/root/cjh/pro/resources/datasets/LVEval"

# 任务 = 数据集名_长度，如 hotpotwikiqa_mixup_32k  factrecall_en  loogle_CR_mixup  multifieldqa_en_mixup
tasks=("dureader_mixup_16k")


# =============================================================================================================
# 可选数据集   LV-Eval
# --------------------------------------------------------------------------------------------------------------
# 数据集_长度组合，长度: 16k / 32k / 64k / 128k / 256k
# Multi-doc QA   : hotpotwikiqa_mixup   2wikimqa_mixup   musique_mixup   dureader_mixup
# Single-doc QA  : multifieldqa_en_mixup   multifieldqa_zh_mixup   loogle_SD_mixup
# Long-context   : loogle_CR_mixup   loogle_MIR_mixup   factrecall_en   factrecall_zh
# Code           : lic_mixup
# CMRC           : cmrc_mixup
# =============================================================================================================



# ------  段注释开始  -----------------------------------------------------------------------------------
<<'COMMENT'

#===== "minference"  "flex"  "full"  "xattn" "sparge"==============

methods=("full" "xattn")

# 使用对比模型
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
      for method in "${methods[@]}"; do
          echo "======================================================================"
          echo "Running: model=$model, task=$task, method=$method"
          echo "======================================================================"
          python -u eval/LVEval/pred.py \
              --model "$model" \
              --task "$task" \
              --method "$method" \
              --num_samples $NUM_SAMPLES \
              --timing \
              --data_dir "$DATA_DIR" \
              --load_4bit
      done
  done
done

COMMENT
# ------  段注释结束  -----------------------------------------------------------------------------------



# ------  段注释开始  ------
#<<'COMMENT'
echo "===== 使用myattn进行推理 ======="
echo "${tasks[@]}"
sink_ratio=0.01
recent_ratio=0.00
local_span_ratio=0.02

# ====== 模式切换 "threshold" 或 "topk"======
corr_selection_mode="threshold"

if [ "$corr_selection_mode" == "threshold" ]; then
    corr_values=(0.7  0.85)
else
    corr_values=(0.05 0.08 0.1)
fi

qk_topk_ratios=(0.2)
stripe_thresholds=(0.4)
diag_sample_ratios=(0.15)
column_topk_ratios=(0.2)

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    for val in "${corr_values[@]}"; do
      for qk_topk in "${qk_topk_ratios[@]}"; do
        for stripe_thresh in "${stripe_thresholds[@]}"; do
          for diag_ratio in "${diag_sample_ratios[@]}"; do
            for col_ratio in "${column_topk_ratios[@]}"; do

              if [ "$corr_selection_mode" == "threshold" ]; then
                  corr_flag="--corr_thres $val"
                  echo_info="corr_thres=$val"
              else
                  corr_flag="--corr_topk_ratio $val"
                  echo_info="corr_topk_ratio=$val"
              fi

              echo "=========================================================================="
              echo "Running: task=$task, mode=$corr_selection_mode, $echo_info, qk_topk=$qk_topk, stripe=$stripe_thresh, diag_ratio=$diag_ratio, col=$col_ratio"
              echo "=========================================================================="

              python -u eval/LVEval/pred.py \
                  --model "$model" \
                  --task "$task" \
                  --method "myattn" \
                  --num_samples $NUM_SAMPLES \
                  --sink_ratio "$sink_ratio" \
                  --recent_ratio "$recent_ratio" \
                  --local_ratio "$local_span_ratio" \
                  --corr_selection_mode "$corr_selection_mode" \
                  $corr_flag \
                  --block_size 128 \
                  --use_cor \
                  --enable_last_block \
                  --last_block_thres 0.001 \
                  --enable_column_mask \
                  --column_topk_ratio "$col_ratio" \
                  --diag_sample_ratio "$diag_ratio" \
                  --min_diag_samples 5 \
                  --max_diag_samples 64 \
                  --stripe_threshold "$stripe_thresh" \
                  --qk_topk_ratio "$qk_topk" \
                  --timing \
                  --data_dir "$DATA_DIR" \
                  --load_4bit

            done
          done
        done
      done
    done
  done
done
#COMMENT
# ------  段注释结束  ------
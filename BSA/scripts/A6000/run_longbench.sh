#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

models=("llama-3.1-8b-instruct-a6000")
#models=("qwen2.5-7b-instruct")


# -1表示全部
NUM_SAMPLES=-1


#tasks=("hotpotqa" "2wikimqa" "musique" "dureader" "multifieldqa" "narrativeqa" "qasper" "gov_report" "qmsum" "multi_news" "vcsum" "passage_retrieval_en" "passage_count" "lcc" "repobench-p" "triviaqa" "trec" "samsum")



#tasks=("passage_retrieval_en"  "2wikimqa"  "hotpotqa"  "passage_count"   "qasper" "triviaqa" "musique"  "trec" "lsht" "qasper" "multifieldqa_en" "multifieldqa_zh" "lcc" "repobench-p" "multi_news" "samsum" "vcsum" "qmsum" "dureader" "narrativeqa")




tasks=("gov_report" "narrativeqa")







# =============================================================================================================
# 可选数据集   LongBench                                                          max_gen
# --------------------------------------------------------------------------------------------------------------
# Multi-doc QA   : hotpotqa(9151,32)              2wikimqa(4887,32)         musique(11214,32)        dureader(15768,128)
# Single-doc QA  : multifieldqa_en(4559,64)       multifieldqa_zh(6701,64)  narrativeqa(18409,128)   qasper(3619,128)
# Summarization  : gov_report(8734,512)           qmsum(10614,512)          multi_news(2113,512)     vcsum(15380,512)
# Few shot       : triviaqa(8209,32)              samsum(6258,128)          trec(5177,64)            lsht(22337,64)
# Synthetic      : passage_retrieval_en(9289,32)  passage_retrieval_zh(9289,32)                      passage_count(11141,32) 
# Code           : lcc(1235,64)                   repobench-p(4206,64)
# =============================================================================================================







# ------  段注释开始  -----------------------------------------------------------------------------------
<<'COMMENT'

#===== "minference"  "flex"  "full"  "xattn" "sparge"==============


methods=("full"  "xattn")

# 使用对比模型
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
      for method in "${methods[@]}"; do
          echo "======================================================================"
          echo "Running: model=$model, task=$task, method=$method"
          echo "======================================================================"
          python -u eval/LongBench/pred.py \
              --data_dir "/backup01/cjh/projects/resources/datasets/LongBench_data/data" \
              --model "$model" \
              --tasks "$task" \
              --method "$method" \
              --num_samples $NUM_SAMPLES
              #--timing \
              #--load_4bit
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
    corr_values=(0.85 0.8 0.7)
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

              python -u eval/LongBench/pred.py \
	          --data_dir "/backup01/cjh/projects/resources/datasets/LongBench_data/data" \
                  --model "$model" \
                  --tasks "$task" \
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
                  --qk_topk_ratio "$qk_topk"

            done
          done
        done
      done
    done
  done
done
#COMMENT
# ------  段注释结束  ------


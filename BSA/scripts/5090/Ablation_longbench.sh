#!/bin/bash
models=("llama-3.1-8b-instruct")
#models=("qwen2.5-7b-instruct")


# -1表示全部
NUM_SAMPLES=-1


#tasks=("passage_retrieval_en"  "2wikimqa"  "hotpotqa"  "passage_count"   "qasper" "triviaqa" "musique"  "trec" "lsht" "qasper" "multifieldqa_en" "multifieldqa_zh" "lcc" "repobench-p" "multi_news" "samsum" "vcsum" "qmsum" "dureader" "gov_report")


tasks=("hotpotqa" "2wikimqa" "qasper" "multifieldqa_en" "triviaqa" "trec" "passage_retrieval_en" "passage_count" "lcc" "repobench-p" "qmsum" "multi_news")


# =============================================================================================================
# 可选数据集   LongBench                                                                               max_gen
# --------------------------------------------------------------------------------------------------------------
# Multi-doc QA   : hotpotqa(9151,32)              2wikimqa(4887,32)         musique(11214,32)        dureader(15768,128)
# Single-doc QA  : multifieldqa_en(4559,64)       multifieldqa_zh(6701,64)  narrativeqa(18409,128)   qasper(3619,128)
# Summarization  : gov_report(8734,512)           qmsum(10614,512)          multi_news(2113,512)     vcsum(15380,512)
# Few shot       : triviaqa(8209,32)              samsum(6258,128)          trec(5177,64)            lsht(22337,64)
# Synthetic      : passage_retrieval_en(9289,32)  passage_retrieval_zh(9289,32)                      passage_count(11141,32) 
# Code           : lcc(1235,64)                   repobench-p(4206,64)
# =============================================================================================================


echo "===== 开始执行 myattn 组件消融实验 (Component Ablation) ======="

# 1. 在这里新增 "no_dynamic" 选项
#ablations=("all_stripe" "all_diffuse" "no_col" "no_prior" "no_dynamic")


ablations=("no_dynamic")

# 共享的最优基础参数 (以你实际调优的最佳参数为准)
corr_selection_mode="threshold"
val=0.85
qk_topk=0.2
diag_ratio=0.15
col_ratio=0.2


# 遍历所有任务
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for ablation in "${ablations[@]}"; do
      
      # 1. 每次循环重置为最佳默认基础参数
      sink_ratio=0.01
      recent_ratio=0.00
      local_span_ratio=0.02
      last_block_thres=0.001  # 显式定义
      stripe_thresh=0.4
      col_ratio=0.2
      
      # 2. 默认开启所有标志位
      enable_col_flag="--enable_column_mask"
      enable_last_flag="--enable_last_block"
      enable_cor_flag="--use_cor"
      
      # 3. 根据消融模式，覆盖特定参数或关闭特定标志位
      case $ablation in
        "all_stripe")
          stripe_thresh=999.0
          echo_info="All Stripe"
          ;;
        "all_diffuse")
          stripe_thresh=-1.0
          echo_info="All Diffuse"
          ;;
        "no_col")
          enable_col_flag="" # Python 会在文件名显示 ColR-OFF
          echo_info="w/o Column Mask"
          ;;
        "no_prior")
          sink_ratio=0.0
          recent_ratio=0.0
          local_span_ratio=0.0
          enable_last_flag="" # Python 会在文件名显示 OFF
          echo_info="w/o Prior Mask"
          ;;
        "no_dynamic")
          enable_cor_flag="" # 清空标志位，跳过 FFT 和 QK TopK 掩码
          echo_info="w/o Dynamic Mask (Prior + Col ONLY)"
          ;;
      esac

      # 设置 corr_flag
      if [ "$corr_selection_mode" == "threshold" ]; then
          corr_flag="--corr_thres $val"
      else
          corr_flag="--corr_topk_ratio $val"
      fi

      echo "Running $echo_info on $task..."

      # 运行脚本（注意这里的 $enable_cor_flag 替换了原本写死的 --use_cor）
      python -u eval/LongBench/pred.py \
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
          $enable_cor_flag \
          $enable_last_flag \
          --last_block_thres "$last_block_thres" \
          $enable_col_flag \
          --column_topk_ratio "$col_ratio" \
          --diag_sample_ratio "$diag_ratio" \
          --min_diag_samples 5 \
          --max_diag_samples 64 \
          --stripe_threshold "$stripe_thresh" \
          --qk_topk_ratio "$qk_topk" 2>&1 | tee "$log_file"
    done
  done
done
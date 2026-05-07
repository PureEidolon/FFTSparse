#!/bin/bash
models=("llama-3.1-8b-instruct")

# -1表示全部
NUM_SAMPLES=-1
MAX_LENGTH=100000
KV_CACHE_QUANT=4    # 0=不量化, 2=2bit, 4=4bit, 8=8bit


# =============================================================================================================
# InfiniteBench 任务列表
# --------------------------------------------------------------------------------------------------------------
# Retrieval      : passkey(122k,2)            number_string(122k,4)      kv_retrieval(90k,23)
# Book           : longbook_sum_eng(172k,1.1k) longbook_qa_eng(193k,5)   longbook_choice_eng(184k,5)  longbook_qa_chn(2069k,6)
# Dialogue       : longdialogue_qa_eng(104k,3)
# Math           : math_find(88k,1)           math_calc(44k,44k)
# Code           : code_run(75k,1)            code_debug(115k,5)
# =============================================================================================================

#tasks=("passkey" "number_string" "kv_retrieval" "longbook_qa_eng" "longbook_choice_eng" "longdialogue_qa_eng" "math_find" "code_run" "code_debug")

tasks=("longbook_qa_eng" "longdialogue_qa_eng")


# ------  段注释开始  -----------------------------------------------------------------------------------
#<<'COMMENT'

#===== "minference"  "flex"  "full"  "xattn" ==============

methods=("full" "xattn")

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for method in "${methods[@]}"; do
            echo "======================================================================"
            echo "Running: model=$model, task=$task, method=$method"
            echo "======================================================================"
            python -u eval/InfiniteBench/src/pred_infinitebench.py \
                --model "$model" \
                --task "$task" \
                --method "$method" \
                --max_length $MAX_LENGTH \
                --num_samples $NUM_SAMPLES \
                --load_4bit \
                --kv_cache_quant $KV_CACHE_QUANT
                #--timing \
	
                
        done
    done
done

#COMMENT
# ------  段注释结束  -----------------------------------------------------------------------------------






# ------  段注释开始  -----------------------------------------------------------------------------------
#<<'COMMENT'


echo "===== 使用myattn进行推理 ======="
echo "${tasks[@]}"

sink_ratios=(0.01)
recent_ratios=(0.01)
local_span_ratios=(0.02)



#  topk   threshold
corr_selection_mode="topk"

corr_topk_ratios=(0.05 0.1 0.2)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for sink in "${sink_ratios[@]}"; do
            for recent in "${recent_ratios[@]}"; do
                for local_span in "${local_span_ratios[@]}"; do
                    if [ "$corr_selection_mode" == "threshold" ]; then
                        sweep_values=("${corr_thresholds[@]}")
                    else
                        sweep_values=("${corr_topk_ratios[@]}")
                    fi

                    for val in "${sweep_values[@]}"; do
                        if [ "$corr_selection_mode" == "threshold" ]; then
                            corr_flag="--corr_thres $val"
                            echo_info="corr_thres=$val"
                        else
                            corr_flag="--corr_topk_ratio $val"
                            echo_info="topk_ratio=$val"
                        fi

                        echo "=========================================================================="
                        echo "Running: sink=$sink, recent=$recent, local_span=$local_span, mode=$corr_selection_mode, $echo_info"
                        echo "==========================================================================="
                        python -u eval/InfiniteBench/src/pred_infinitebench.py \
                            --model "$model" \
                            --task "$task" \
                            --method "myattn" \
                            --num_samples $NUM_SAMPLES \
                            --max_length $MAX_LENGTH \
                            --sink_ratio "$sink" \
                            --recent_ratio "$recent" \
                            --local_ratio "$local_span" \
                            --corr_selection_mode "$corr_selection_mode" \
                            $corr_flag \
                            --block_size 128 \
                            --use_cor \
                            --enable_last_block \
                            --last_block_thres 0.001 \
                            --enable_column_mask \
                            --column_topk_ratio 0.1 \
                            --load_4bit \
                            --kv_cache_quant $KV_CACHE_QUANT
                    done
                done
            done
        done
    done
done


#COMMENT
# ------  段注释结束  -----------------------------------------------------------------------------------
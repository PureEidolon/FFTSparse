#!/bin/bash
# InfiniteBench 评估运行脚本示例






# ============================================================================	
# 配置（请修改为你的实际路径）
# ============================================================================
MODEL="llama-3.1-8b-instruct"
MODEL_PATH="/backup01/cjh/projects/resources/models/llama-3.1-8b-instruct"
DATA_DIR="/backup01/cjh/projects/resources/datasets/InfiniteBench"


#"full"  "flex" "minference"


METHODS=("full"  "xattn")  
NUM_SAMPLES=-1




# ============================================================================
# 所有任务列表
# ============================================================================
TASKS=(
    "passkey"              # 122.4k tokens |    2 tokens | 密钥检索
    #"number_string"        # 122.4k tokens |    4 tokens | 数字序列检索
    #"kv_retrieval"         #  89.9k tokens |   23 tokens | KV 键值检索
    #"longbook_sum_eng"     # 171.5k tokens | 1.1k tokens | 英文书籍摘要
    #"longbook_qa_eng"      # 192.6k tokens |    5 tokens | 英文书籍问答
    #"longbook_choice_eng"  # 184.4k tokens |    5 tokens | 英文书籍选择题
    #"longbook_qa_chn"      # 2068.6k tokens|    6 tokens | 中文书籍问答
    #"longdialogue_qa_eng"  # 103.6k tokens |    3 tokens | 对话角色识别
    "math_find"            #  87.9k tokens |    1 tokens | 数学查找
    #"math_calc"            #  43.9k tokens |  44k tokens | 数学计算（输出超长）
    #"code_run"             #  75.2k tokens |    1 tokens | 代码执行
    #"code_debug"           # 114.7k tokens |    5 tokens | 代码调试
)






# ------  段注释开始  -------------------------------------------------------
#<<'COMMENT'




# ============================================================================
# 循环运行所有任务和方法
# ============================================================================
for TASK in "${TASKS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "====== Running $TASK with method $METHOD ======"
        if [ "$METHOD" == "xattn" ]; then
            CUDA_VISIBLE_DEVICES=1 python -u eval/InfiniteBench/src/pred_infinitebench.py \
                --model "$MODEL" \
                --max_length 110000 \
                --method "$METHOD" \
                --task "$TASK" \
                --num_samples $NUM_SAMPLES \
                --data_dir "$DATA_DIR" \
                --load_4bit
        else
            python -u eval/InfiniteBench/src/pred_infinitebench.py \
                --model "$MODEL" \
                --max_length 110000 \
                --method "$METHOD" \
                --task "$TASK" \
                --num_samples $NUM_SAMPLES \
                --data_dir "$DATA_DIR" \
                --load_4bit
        fi
    done
done




#COMMENT
# ------  段注释结束  -------------------------------------------------------













# ------  段注释开始  -------------------------------------------------------
#<<'COMMENT'
echo "===== 使用myattn进行推理 ======="
echo "${TASKS[@]}"

# 固定参数
sink_ratio=0.01
recent_ratio=0.00
local_span_ratio=0.02

# 可变参数
corr_selection_mode="topk"
fft_topk_ratios=(0.1)
qk_topk_ratios=(0.2)
stripe_thresholds=(0.4)
diag_sample_ratios=(0.15)
column_topk_ratios=(0.2)

for task in "${TASKS[@]}"; do
    for val in "${fft_topk_ratios[@]}"; do
        for qk_topk in "${qk_topk_ratios[@]}"; do
            for stripe_thresh in "${stripe_thresholds[@]}"; do
                for diag_ratio in "${diag_sample_ratios[@]}"; do
                    for col_ratio in "${column_topk_ratios[@]}"; do
                        if [ "$corr_selection_mode" == "threshold" ]; then
                            corr_flag="--corr_thres $val"
                            echo_info="fft-corr_thres=$val"
                        else
                            corr_flag="--corr_topk_ratio $val"
                            echo_info="fft-topk_ratio=$val"
                        fi
                        echo "=========================================================================="
                        echo "Running: task=$task, mode=$corr_selection_mode, $echo_info, qk_topk=$qk_topk, stripe=$stripe_thresh, diag_ratio=$diag_ratio, col=$col_ratio"
                        echo "=========================================================================="
                        CUDA_VISIBLE_DEVICES=1,4,5 python -u eval/InfiniteBench/src/pred_infinitebench.py \
                            --model "$MODEL" \
                            --task "$task" \
                            --method "myattn" \
                            --max_length 110000 \
                            --num_samples $NUM_SAMPLES \
                            --data_dir "$DATA_DIR" \
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
                            --load_4bit
                    done
                done
            done
        done
    done
done
#COMMENT
# ------  段注释结束  -------------------------------------------------------
echo "====== Done! ======"
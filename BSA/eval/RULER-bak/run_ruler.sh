#!/bin/bash
# ============================================================
# RULER 评测脚本 - 多方法 x 多长度
# ============================================================
RULER_DIR="/root/cjh/pro/resources/datasets/RULER/scripts"
EVAL_DIR="/root/cjh/pro/BSA/eval/RULER"
MODEL_PATH="/root/cjh/pro/resources/models/llama-3.1-8b-instruct"
MODEL_NAME="llama-3.1-8b-instruct"
TOKENIZER_PATH=${MODEL_PATH}
TOKENIZER_TYPE="hf"
MODEL_TEMPLATE_TYPE="meta-llama3"
NUM_SAMPLES=500      

# ============================================================
# ★ myattn 参数配置
# ============================================================
SINK_RATIO=0.01
RECENT_RATIO=0.01
LOCAL_RATIO=0.02
BLOCK_SIZE=128
CORR_THRES=0.8
USE_COR="--use_cor"                          # 填 "--use_cor" 开启相关性，留空关闭
ENABLE_LAST_BLOCK="--enable_last_block"      # 填 "--enable_last_block" 开启，留空关闭
LAST_BLOCK_THRES=0.01

# ============================================================
# ★ 在这里自选方法和序列长度
# ============================================================
METHODS=(
    "full"
    #"myattn"
    "xattn"
    #"flex"
    #"minference"
    # "sparge"
)

SEQ_LENS=(
    # 4096
    8192
    # 16384
    # 32768
    # 65536
    # 131072
)

# ============================================================
# ★ 在这里自选任务（注释掉不想跑的）
# ============================================================
TASKS=(
    "niah_single_1"
    #"niah_single_2"
    #"niah_single_3"
    #"niah_multikey_1"
    #"niah_multikey_2"
    #"niah_multikey_3"
    #"niah_multivalue"
    #"niah_multiquery"
    #"vt"
    #"cwe"
    #"fwe"
    "qa_1"
    #"qa_2"
)

# ============================================================
# 根据方法生成唯一实验名称
# ============================================================
get_exp_name() {
    local method=$1
    if [ "${method}" == "myattn" ]; then
        local cor_tag="nocor"
        local lb_tag="nolb"
        [ -n "${USE_COR}" ]           && cor_tag="cor${CORR_THRES}"
        [ -n "${ENABLE_LAST_BLOCK}" ] && lb_tag="lb${LAST_BLOCK_THRES}"
        echo "myattn-s${SINK_RATIO}-r${RECENT_RATIO}-l${LOCAL_RATIO}-bs${BLOCK_SIZE}-${cor_tag}-${lb_tag}"
    else
        echo "${method}"
    fi
}

# ============================================================
# 主循环
# ============================================================
for METHOD in "${METHODS[@]}"; do
    for SEQ_LEN in "${SEQ_LENS[@]}"; do

        EXP_NAME=$(get_exp_name ${METHOD})
        RESULTS_DIR="${EVAL_DIR}/results/${MODEL_NAME}/${EXP_NAME}/${SEQ_LEN}"
        DATA_DIR="${RESULTS_DIR}/data"
        PRED_DIR="${RESULTS_DIR}/pred"
        mkdir -p ${DATA_DIR} ${PRED_DIR}

        echo ""
        echo "############################################################"
        echo "  EXP_NAME=${EXP_NAME}  SEQ_LEN=${SEQ_LEN}"
        echo "############################################################"
        echo "  模型路径     : ${MODEL_PATH}"
        echo "  样本数量     : ${NUM_SAMPLES}"
        if [ "${METHOD}" == "myattn" ]; then
            echo "  ── myattn 参数 ──────────────────────────"
            echo "  sink_ratio        : ${SINK_RATIO}"
            echo "  recent_ratio      : ${RECENT_RATIO}"
            echo "  local_ratio       : ${LOCAL_RATIO}"
            echo "  block_size        : ${BLOCK_SIZE}"
            echo "  use_cor           : ${USE_COR:-关闭}"
            echo "  corr_thres        : ${CORR_THRES}"
            echo "  enable_last_block : ${ENABLE_LAST_BLOCK:-关闭}"
            echo "  last_block_thres  : ${LAST_BLOCK_THRES}"
            echo "  ─────────────────────────────────────────"
        fi
        echo "############################################################"

        cd ${RULER_DIR}

        for TASK in "${TASKS[@]}"; do
            echo "------------------------------------------------------"
            echo "▶ 生成数据: ${TASK}"
            python data/prepare.py \
                --save_dir ${DATA_DIR} \
                --benchmark synthetic \
                --task ${TASK} \
                --tokenizer_path ${TOKENIZER_PATH} \
                --tokenizer_type ${TOKENIZER_TYPE} \
                --max_seq_length ${SEQ_LEN} \
                --model_template_type ${MODEL_TEMPLATE_TYPE} \
                --num_samples ${NUM_SAMPLES}

            echo "▶ 模型推理: ${TASK}"
            python ${EVAL_DIR}/call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir ${PRED_DIR} \
                --benchmark synthetic \
                --task ${TASK} \
                --server_type my_model \
                --model_name_or_path ${MODEL_PATH} \
                --method ${METHOD} \
                --sink_ratio ${SINK_RATIO} \
                --recent_ratio ${RECENT_RATIO} \
                --local_ratio ${LOCAL_RATIO} \
                --block_size ${BLOCK_SIZE} \
                --corr_thres ${CORR_THRES} \
                --last_block_thres ${LAST_BLOCK_THRES} \
                ${USE_COR} ${ENABLE_LAST_BLOCK} \
                --batch_size 1 \
                --threads 1
        done

        echo "------------------------------------------------------"
        echo "▶ 评估: EXP_NAME=${EXP_NAME} SEQ_LEN=${SEQ_LEN}"
        python ${EVAL_DIR}/evaluate.py \
            --data_dir ${PRED_DIR} \
            --benchmark synthetic

        echo "✅ 完成: results saved to ${PRED_DIR}"
    done
done

echo ""
echo "============================================================"
echo "全部实验完成！结果目录: ${EVAL_DIR}/results/${MODEL_NAME}/"
echo "============================================================"
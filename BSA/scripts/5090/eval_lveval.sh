#!/bin/bash
export LD_LIBRARY_PATH=/backup01/cjh/miniconda3/envs/xattn/lib:$LD_LIBRARY_PATH

models=("llama-3.1-8b-instruct")

cd eval/LVEval || { echo "Failed to enter eval/LVEval"; exit 1; }

for model in "${models[@]}"; do
    echo "======================================================================"
    echo "Evaluating model: $model"
    echo "======================================================================"
    python -u evaluation.py --input-dir pred/$model


    # 进入 eval_result 目录执行 parse_result.py
    pushd pred/$model/eval_result > /dev/null || { echo "Failed to enter pred/$model/eval_result"; continue; }
    python parse_result.py
    popd > /dev/null
done



echo "All done!"
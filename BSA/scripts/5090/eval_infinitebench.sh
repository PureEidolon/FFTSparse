#!/bin/bash

MODEL="llama-3.1-8b-instruct"
PRED_DIR="eval/InfiniteBench/pred/$MODEL"

# 所有合法任务（横排）
VALID_TASKS=("passkey" "number_string" "kv_retrieval" "longbook_sum_eng" "longbook_qa_eng" "longbook_choice_eng" "longbook_qa_chn" "longdialogue_qa_eng" "math_find" "math_calc" "code_run" "code_debug")

for f in "$PRED_DIR"/*.jsonl; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    
    for task in "${VALID_TASKS[@]}"; do
        if [[ "$name" == *"$task"* ]]; then
            echo "--- $task ---"
            python eval/InfiniteBench/src/compute_scores.py --pred_path "$f" --task "$task"
            break
        fi
    done
done

echo "====== Done! ======"
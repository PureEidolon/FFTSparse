# ============================================================
# 评估配置
# ============================================================

models=("llama-3.1-8b-instruct")
#models=("qwen2.5-7b-instruct")

#datasets为空时评估所有文件

#datasets=("passage_retrieval_en"  "2wikimqa"  "hotpotqa"  "passage_count"   "qasper" "triviaqa" "musique"  "trec" "lsht" "qasper" "multifieldqa_en" "multifieldqa_zh" "lcc" "repobench-p" "multi_news" "samsum" "vcsum" "qmsum" "dureader" "narrativeqa" "gov_report")


datasets=()

#进入评估目录并并行运行 eval.py
cd eval/LongBench || { echo "Failed to enter eval/LongBench"; exit 1; }

for model in "${models[@]}"; do
    echo "Evaluating model: $model"
    if [ ${#datasets[@]} -eq 0 ]; then
        python -u eval.py --model "$model" &
    else
        python -u eval.py --model "$model" --datasets "${datasets[@]}" &
    fi
done

wait
echo "All done!"
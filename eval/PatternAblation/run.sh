#!/bin/bash
# 推理
python pred_stripe_attn.py \
    --model llama-3.1-8b-instruct \
    --task 2wikimqa_0-5000 \
    --num_samples 20 \
    --out_dir ./stripe_exp_results \
    --model2path ./config/model2path.json \
    --dataset2prompt ./config/dataset2prompt.json \
    --dataset2maxlen ./config/dataset2maxlen.json \
    --data_root ./filtered_data


# 评估
python eval.py --results_path ./stripe_exp_results
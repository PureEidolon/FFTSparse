#!/bin/bash
python stripe_coverage_analysis.py \
    --model llama-3.1-8b-instruct \
    --task dureader_mixup_64k \
    --layer 2 --head 0 \
    --downsample 99999 \
    --max_length 65000 \
    --out_dir ./stripe_analysis \
    --model2path ../../config/model2path.json \
    --data_root /root/cjh/pro/resources/datasets/LVEval/dureader_mixup

#sed -i 's/\r$//' run_stripe.sh
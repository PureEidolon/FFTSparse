#!/bin/bash

python eval/InfiniteBench/src/compare_timing.py \
    --pred_dir eval/InfiniteBench/pred/ \
    --model llama-3.1-8b-instruct \
    --output eval/InfiniteBench/output/timing_comparison.csv
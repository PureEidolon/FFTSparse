#!/bin/bash
# eval/VisAttn/run_vis.sh

model="llama-3.1-8b-instruct"
num_samples=1
#vis_layers="0,1,2,3,10,11,12,13,20,21,22,23,28,29,30,31"
vis_layers="-1"
vis_heads="0"
vis_downsample=99999

tasks=("2wikimqa" "gov_report" "lcc" "qasper")

for task in "${tasks[@]}"; do
    echo "=============================="
    echo "Running: task=$task"
    echo "=============================="
    python -u ./pred_vis.py \
        --model          "$model" \
        --task           "$task" \
        --num_samples    $num_samples \
        --vis_layers     "$vis_layers" \
        --vis_heads      "$vis_heads" \
        --vis_dir        "./vis_attn/FULL-${task}" \
        --vis_downsample $vis_downsample
done

echo "=============================="
echo "Merging plots..."
echo "=============================="
python ./merge_vis.py
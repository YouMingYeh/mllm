#!/bin/bash

# ===========================================
# MLLMs Know Where to Look - Benchmark Runner
# ===========================================

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllms_know

# GPU device
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
MODEL="llava"
TASKS=("textvqa" "aokvqa")
CROP_MODES=("no_crop" "single_crop" "smart_multi_crop")
METHODS=("rel_att" "grad_att")
SAVE_PATH="./data/results"

mkdir -p $SAVE_PATH

echo "=========================================="
echo "MLLMs Know Where to Look - Benchmark"
echo "=========================================="
echo "Model: $MODEL"
echo "Tasks: ${TASKS[*]}"
echo "Crop Modes: ${CROP_MODES[*]}"
echo "Methods: ${METHODS[*]}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Total runs: $((${#TASKS[@]} * ${#CROP_MODES[@]} * ${#METHODS[@]}))"
echo "=========================================="

for task in "${TASKS[@]}"; do
    for crop_mode in "${CROP_MODES[@]}"; do
        for method in "${METHODS[@]}"; do
            echo ""
            echo ">>> $task | $crop_mode | $method"
            python run.py \
                --model $MODEL \
                --task $task \
                --method $method \
                --crop_mode $crop_mode \
                --save_path $SAVE_PATH
        done
    done
done

echo ""
echo "=========================================="
echo "Generating evaluation report..."
echo "=========================================="
python get_score.py --data_dir $SAVE_PATH --save_path ./

echo ""
echo "Done! Results saved to ./evaluation_report.csv"

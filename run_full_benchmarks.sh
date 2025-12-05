#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllms_know

# Device to use
export CUDA_VISIBLE_DEVICES=4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
MODEL="llava"
TASKS=("textvqa" "aokvqa")
CROP_MODES=("no_crop" "single_crop" "smart_multi_crop")
METHODS=("rel_att" "grad_att")
SAVE_PATH="./data/results"

mkdir -p $SAVE_PATH

echo "=========================================="
echo "Running Full Benchmark Suite"
echo "Model: $MODEL"
echo "Tasks: ${TASKS[*]}"
echo "Crop Modes: ${CROP_MODES[*]}"
echo "Methods: ${METHODS[*]}"
echo "Total runs: $((${#TASKS[@]} * ${#CROP_MODES[@]} * ${#METHODS[@]}))"
echo "=========================================="

for task in "${TASKS[@]}"; do
    for crop_mode in "${CROP_MODES[@]}"; do
        for method in "${METHODS[@]}"; do
            echo ""
            echo ">>> Running: $MODEL - $task - $method - $crop_mode"
            python run.py \
                --model $MODEL \
                --task $task \
                --method $method \
                --crop_mode $crop_mode \
                --save_path $SAVE_PATH
            echo "<<< Finished: $MODEL - $task - $method - $crop_mode"
        done
    done
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="

# Generate comparison table
echo ""
echo "Generating evaluation report..."
python get_score.py --data_dir $SAVE_PATH --save_path ./

echo ""
echo "Results saved to:"
echo "  - ./evaluation_report.json"
echo "  - ./evaluation_report.csv"

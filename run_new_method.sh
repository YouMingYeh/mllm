#!/bin/bash

task=$1
model=$2
method=$3
crop_mode="smart_multi_crop"

declare -a gpus=(5)

num_gpus=${#gpus[@]}

declare -a parts=()

for ((i=0; i<num_gpus; i++)); do
    parts+=($i)
done

for i in "${!gpus[@]}"; do
  command="CUDA_VISIBLE_DEVICES=${gpus[$i]} python run.py --chunk_id ${parts[$i]} --total_chunks $num_gpus --model $model --task $task --method $method --crop_mode $crop_mode --num_samples 50"
  echo "Executing: $command"
  eval $command &
  sleep 10
done

wait

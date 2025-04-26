#!/bin/bash

name="Qwen2.5-VL-7B-Instruct"
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}

# debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "Evaluating task in $CHUNKS GPUs"

BASE_ARGS=(
    "./evaluate/configs/eval_configs.py"
    "evaluate" "True"
    "model" "qwenvl"
    "batch_size" "2"
    "num_workers" "0"
    "model_path" "./model_outputs/${name}"
    "save_interval" "2"
)

tasks=("mira_hallu")
echo "Starting task: $task"
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching chunk: $IDX for task: $task"
    ARGS=(
        "${BASE_ARGS[@]}"
        "val_tag" "$task"
        "output_dir" "eval_outputs"
        "num_chunks" "$CHUNKS"
        "chunk_idx" "$IDX"
        "device" "cuda:${GPULIST[$IDX]}"
    )
    echo "ARGS: ${ARGS[@]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 evaluate/infer.py "${ARGS[@]}"
done
wait
echo "Finish task: $task!"
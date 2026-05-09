#!/bin/bash
# Evaluate the model on the test set using a pretrained checkpoint without training.
# Usage: bash test.sh [GPU_ID] [path_to_checkpoint] [additional python args]
# Example override: bash test.sh 0 ckpt/best_model.pt --encoder_v resnet101

GPU_ID=${1:-0}
shift
MODEL_PATH=${1:-ckpt/best_model.pt}
if [ $# -gt 0 ]; then
  shift
fi
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --encoder_v resnet152 --gate --load_model "$MODEL_PATH" "$@"

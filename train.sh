#!/bin/bash
# Train the model with the default ResNet-152 visual encoder and gating, saving checkpoints every epoch.
# Usage: bash train.sh [GPU_ID] [additional python args]
# Example override: bash train.sh 0 --encoder_v resnet101

GPU_ID=${1:-0}
shift
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --encoder_v resnet152 --gate --save_interval 1 "$@"

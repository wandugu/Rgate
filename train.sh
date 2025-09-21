#!/bin/bash
# Train the model with ResNet101 visual encoder and gating, saving checkpoints every epoch.
# Usage: bash train.sh [GPU_ID] [additional python args]

GPU_ID=${1:-0}
shift
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --encoder_v resnet101 --gate --save_interval 1 "$@"


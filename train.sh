#!/bin/bash
# Train the model with ResNet101 visual encoder and gating, saving checkpoints every epoch.

python main.py --encoder_v resnet101 --gate --save_interval 1 "$@"

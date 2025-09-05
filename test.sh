#!/bin/bash
# Evaluate the model on the test set using a pretrained checkpoint without training.
# Usage: bash test.sh [path_to_checkpoint]

MODEL_PATH=${1:-ckpt/best_model.pt}
python main.py --encoder_v resnet101 --gate --load_model "$MODEL_PATH"

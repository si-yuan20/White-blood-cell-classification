#!/bin/bash
# scripts/train_davit.sh
# Train DaViT baseline.

DATA_DIR="${1:?Usage: $0 <data_dir>}"
SEED="${2:-42}"

python main.py \
  --data_dir "$DATA_DIR" \
  --model_name davit \
  --epochs 200 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 0.003 \
  --seed "$SEED"

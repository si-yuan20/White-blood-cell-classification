#!/bin/bash
# scripts/train_ablation.sh
# Train all ablation variants for a given dataset.

DATA_DIR="${1:?Usage: $0 <data_dir>}"
SEED="${2:-42}"

for variant in convnext mamba dual_naive dual_saf dual_rgbf dual_full; do
  echo "=== Training $variant ==="
  python main.py \
    --data_dir "$DATA_DIR" \
    --model_name "$variant" \
    --epochs 200 \
    --batch_size 64 \
    --lr 3e-4 \
    --weight_decay 0.003 \
    --seed "$SEED"
done

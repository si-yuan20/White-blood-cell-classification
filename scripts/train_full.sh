#!/bin/bash
# scripts/train_full.sh
# Train the full proposed framework (ConvNeXt + Mamba + SAF + RGBF)
python main.py \
  --data_dir "$DATA_DIR" \
  --model_name dual_full \
  --epochs 200 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 0.003 \
  --seed 42

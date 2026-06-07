#!/bin/bash
# scripts/eval_robustness.sh
# Run salt-and-pepper noise robustness evaluation.
# Usage: bash scripts/eval_robustness.sh <data_dir> <checkpoint_path>

DATA_DIR="${1:?Usage: $0 <data_dir> <checkpoint_path>}"
CHECKPOINT="${2:?Usage: $0 <data_dir> <checkpoint_path>}"

python evaluation/robustness_eval.py \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --model_name dual_full \
  --noise_levels 0.0 0.01 0.05 0.10 \
  --output_dir results/robustness

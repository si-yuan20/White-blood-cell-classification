#!/bin/bash
# scripts/eval_cross_dataset.sh
# Run all 6 cross-dataset transfer directions.
# Usage: bash scripts/eval_cross_dataset.sh <checkpoint_dir>

CKPT_DIR="${1:-models}"
PBC_DATA="${2:-/path/to/PBC}"
LDWBC_DATA="${3:-/path/to/LDWBC}"
RAABIN_DATA="${4:-/path/to/Raabin-WBC}"

DIRECTIONS=(
  "PBC LDWBC $LDWBC_DATA"
  "PBC Raabin-WBC $RAABIN_DATA"
  "LDWBC PBC $PBC_DATA"
  "LDWBC Raabin-WBC $RAABIN_DATA"
  "Raabin-WBC PBC $PBC_DATA"
  "Raabin-WBC LDWBC $LDWBC_DATA"
)

for dir in "${DIRECTIONS[@]}"; do
  read -r SRC TGT TGT_DIR <<< "$dir"
  CKPT="$CKPT_DIR/${SRC}_best.pth"
  echo "=== $SRC -> $TGT ==="
  python evaluation/cross_dataset_eval.py \
    --source_name "$SRC" \
    --target_name "$TGT" \
    --target_dir "$TGT_DIR" \
    --checkpoint "$CKPT" \
    --model_name dual_full \
    --output_dir "results/cross_dataset/${SRC}_to_${TGT}"
done

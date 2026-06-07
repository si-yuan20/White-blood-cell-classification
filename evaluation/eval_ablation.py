# evaluation/eval_ablation.py
# -*- coding: utf-8 -*-
"""
Ablation evaluation: evaluate all model variants on a dataset and produce
results compatible with Table 7.

Variant definitions (matching paper):
  - convnext     : ConvNeXt-only baseline
  - mamba        : Mamba-only baseline
  - dual_naive   : ConvNeXt+Mamba naive concat (SAF off, RGBF off)
  - dual_saf     : ConvNeXt+Mamba+SAF (RGBF off)
  - dual_rgbf    : ConvNeXt+Mamba+RGBF (SAF off)
  - dual_full    : ConvNeXt+Mamba+SAF+RGBF (full framework)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prepare import MedicalDataset
from dual_model import create_classifier
from config import make_default_cfg
from utils import set_global_seed, calculate_auc
import albumentations as A
from albumentations.pytorch import ToTensorV2

ABLATION_VARIANTS = ["convnext", "mamba", "dual_naive", "dual_saf", "dual_rgbf", "dual_full"]

VARIANT_META = {
    "convnext":   {"use_convnext": True,  "use_mamba": False, "use_saf": False, "use_rgbf": False},
    "mamba":      {"use_convnext": False, "use_mamba": True,  "use_saf": False, "use_rgbf": False},
    "dual_naive": {"use_convnext": True,  "use_mamba": True,  "use_saf": False, "use_rgbf": False},
    "dual_saf":   {"use_convnext": True,  "use_mamba": True,  "use_saf": True,  "use_rgbf": False},
    "dual_rgbf":  {"use_convnext": True,  "use_mamba": True,  "use_saf": False, "use_rgbf": True},
    "dual_full":  {"use_convnext": True,  "use_mamba": True,  "use_saf": True,  "use_rgbf": True},
}


def build_val_transform(img_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def evaluate_variant(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            elif isinstance(batch, dict):
                inputs, labels = batch["image"], batch["label"]
            else:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            probs = F.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix)
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    auc_val = calculate_auc(targets, probs, num_classes=probs.shape[1])
    cm = confusion_matrix(targets, preds)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1),
        "auc": float(auc_val),
        "params_M": float(n_params),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for Table 7")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., PBC, LDWBC, Raabin-WBC)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--checkpoints_json", type=str, default=None,
                        help="JSON mapping variant -> checkpoint path. "
                             "If not provided, looks in --checkpoint_dir.")
    parser.add_argument("--checkpoint_dir", type=str, default="models",
                        help="Directory with {variant}_best.pth files")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=ABLATION_VARIANTS,
                        help="Variants to evaluate")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve checkpoints
    checkpoints = {}
    if args.checkpoints_json and os.path.isfile(args.checkpoints_json):
        with open(args.checkpoints_json, "r") as f:
            checkpoints = json.load(f)
    else:
        # Auto-discover from checkpoint_dir
        for variant in args.variants:
            candidate = os.path.join(args.checkpoint_dir, f"{variant}_best.pth")
            if os.path.isfile(candidate):
                checkpoints[variant] = candidate

    # Build target dataset
    transform = build_val_transform(args.img_size)
    dataset = MedicalDataset(args.data_dir, transform=transform, two_view=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    class_names = dataset.class_names
    num_classes = len(class_names)

    print(f"Dataset: {args.dataset_name} ({len(dataset)} samples, {num_classes} classes)")
    print(f"Classes: {class_names}")
    print(f"Variants to evaluate: {args.variants}")

    rows = []
    for variant in args.variants:
        meta = VARIANT_META.get(variant, {})
        print(f"\n--- {variant} ---")

        ckpt_path = checkpoints.get(variant)
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"  [MISSING] Checkpoint not found for {variant}. Skipping.")
            rows.append({
                "dataset": args.dataset_name,
                "variant": variant,
                **meta,
                "accuracy": "MISSING",
                "precision": "MISSING",
                "recall": "MISSING",
                "macro_f1": "MISSING",
                "auc": "MISSING",
                "params_M": "MISSING",
            })
            continue

        print(f"  Checkpoint: {ckpt_path}")

        try:
            model = create_classifier(variant, num_classes=num_classes, pretrained=False)
            state = torch.load(ckpt_path, map_location=device)
            if hasattr(model, "model"):
                model.model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            rows.append({
                "dataset": args.dataset_name,
                "variant": variant,
                **meta,
                "accuracy": f"ERROR:{type(e).__name__}",
                "precision": f"ERROR:{type(e).__name__}",
                "recall": f"ERROR:{type(e).__name__}",
                "macro_f1": f"ERROR:{type(e).__name__}",
                "auc": f"ERROR:{type(e).__name__}",
                "params_M": f"ERROR:{type(e).__name__}",
            })
            continue

        results = evaluate_variant(model, loader, device)
        row = {
            "dataset": args.dataset_name,
            "variant": variant,
            **meta,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "macro_f1": results["macro_f1"],
            "auc": results["auc"],
            "params_M": results["params_M"],
        }
        rows.append(row)
        print(f"  Acc={results['accuracy']:.4f}  F1={results['macro_f1']:.4f}  AUC={results['auc']:.4f}")

    # Save results
    out_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(out_dir, f"{args.dataset_name}_ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # CSV (Table 7 format)
    csv_path = os.path.join(out_dir, f"{args.dataset_name}_ablation_results.csv")
    fieldnames = ["dataset", "variant", "use_convnext", "use_mamba", "use_saf",
                  "use_rgbf", "accuracy", "precision", "recall", "macro_f1", "auc", "params_M"]
    with open(csv_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in fieldnames) + "\n")

    # Pretty-print table
    print(f"\n{'='*100}")
    print(f"Ablation Results — {args.dataset_name}")
    print(f"{'='*100}")
    header = f"{'Variant':<14} {'ConvNeXt':<10} {'Mamba':<8} {'SAF':<6} {'RGBF':<6} {'Acc':<10} {'F1':<10} {'AUC':<10}"
    print(header)
    print("-" * 100)
    for r in rows:
        print(f"{r['variant']:<14} {str(r['use_convnext']):<10} {str(r['use_mamba']):<8} "
              f"{str(r['use_saf']):<6} {str(r['use_rgbf']):<6} "
              f"{str(r['accuracy']):<10} {str(r['macro_f1']):<10} {str(r['auc']):<10}")
    print(f"{'='*100}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()

# evaluation/cross_dataset_eval.py
# -*- coding: utf-8 -*-
"""
Cross-dataset transfer evaluation (without target-domain fine-tuning).

Evaluates a model trained on a source dataset directly on a target dataset.
Supports six transfer directions among PBC, LDWBC, Raabin-WBC.
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


def build_val_transform(img_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def evaluate_on_target(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            elif isinstance(batch, dict):
                inputs, labels = batch["image"], batch["label"]
            else:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

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

    # Per-class F1
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1),
        "auc": float(auc_val),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": [float(x) for x in per_class_f1],
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer evaluation")
    parser.add_argument("--source_name", type=str, required=True,
                        choices=["PBC", "LDWBC", "Raabin-WBC"],
                        help="Source dataset name")
    parser.add_argument("--target_name", type=str, required=True,
                        choices=["PBC", "LDWBC", "Raabin-WBC"],
                        help="Target dataset name")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Path to target dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to source-trained checkpoint")
    parser.add_argument("--model_name", type=str, default="dual",
                        help="Model variant")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    model = create_classifier(args.model_name, num_classes=args.num_classes,
                              pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    # Handle wrapped models
    if hasattr(model, "model"):
        model.model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Build target dataset
    transform = build_val_transform(args.img_size)
    target_dataset = MedicalDataset(args.target_dir, transform=transform, two_view=False)
    target_loader = DataLoader(
        target_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Transfer: {args.source_name} -> {args.target_name}")
    print(f"Target samples: {len(target_dataset)}")
    print(f"Target classes: {target_dataset.class_names}")

    results = evaluate_on_target(model, target_loader, device)

    # Print results
    print(f"\n{'='*50}")
    print(f"Cross-dataset: {args.source_name} -> {args.target_name}")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"Macro-F1:    {results['macro_f1']:.4f}")
    print(f"AUC:         {results['auc']:.4f}")
    print(f"Per-class F1: {results['per_class_f1']}")
    print(f"{'='*50}")

    # Save results
    out_dir = os.path.join(args.output_dir, f"{args.source_name}_to_{args.target_name}")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    out_csv = os.path.join(out_dir, "results.csv")
    with open(out_csv, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            if k not in ("confusion_matrix", "per_class_f1"):
                f.write(f"{k},{v}\n")
        for i, pf1 in enumerate(results["per_class_f1"]):
            f.write(f"per_class_f1_{i},{pf1}\n")

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

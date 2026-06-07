# evaluate.py
# -*- coding: utf-8 -*-
"""
Unified evaluation entry point.

Supports single-dataset evaluation, ablation comparison,
cross-dataset transfer, and robustness evaluation.
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

from data_prepare import MedicalDataset, create_loaders
from dual_model import create_classifier
from config import make_default_cfg
from utils import (
    set_global_seed, MetricTracker, calculate_auc,
    plot_confusion_matrix, plot_roc_curve, plot_learning_curves,
    build_model_name, prepare_result_dirs,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_val_transform(img_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def evaluate_single(args, device):
    """Single-dataset evaluation."""
    transform = build_val_transform(args.img_size)
    dataset = MedicalDataset(args.data_dir, transform=transform, two_view=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    class_names = dataset.class_names

    model = create_classifier(args.model_name, num_classes=len(class_names),
                              pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    if hasattr(model, "model"):
        model.model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    tracker = MetricTracker()
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

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss_val = F.cross_entropy(outputs, labels)
            preds = outputs.argmax(dim=1)
            probs = F.softmax(outputs, dim=1)
            tracker.update(loss_val, preds, probs, labels)

    results = tracker.compute(with_raw=True)
    print(f"\n{'='*50}")
    print(f"Model: {args.model_name}")
    print(f"Accuracy:  {results['acc']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Macro-F1:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"{'='*50}")

    # Per-class metrics
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    targets = np.concatenate(tracker.all_targets)
    preds = np.concatenate(tracker.all_preds)
    per_class_prec, per_class_rec, per_class_f1, per_class_support = \
        precision_recall_fscore_support(targets, preds, labels=range(len(class_names)),
                                        zero_division=0)

    print(f"\n{'='*70}")
    print("Per-Class Metrics")
    print(f"{'Class':<16} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name:<16} {per_class_prec[i]:<12.4f} {per_class_rec[i]:<12.4f} "
              f"{per_class_f1[i]:<12.4f} {per_class_support[i]:<10}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        out = {k: float(v) if not isinstance(v, (np.ndarray, list)) else v
               for k, v in results.items()
               if k not in ("raw", "targets", "preds", "probs")}
        out["per_class"] = {
            cls_name: {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i, cls_name in enumerate(class_names)
        }
        json.dump(out, f, indent=2)

    # Save per-class CSV
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    classwise_dir = os.path.join("results", "classwise")
    os.makedirs(classwise_dir, exist_ok=True)
    with open(os.path.join(classwise_dir, f"{dataset_name}_classwise_metrics.csv"), "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for i, cls_name in enumerate(class_names):
            f.write(f"{cls_name},{per_class_prec[i]:.6f},{per_class_rec[i]:.6f},"
                    f"{per_class_f1[i]:.6f},{per_class_support[i]}\n")

    # Plots
    if args.save_plots:
        dataset_name = os.path.basename(os.path.normpath(args.data_dir))
        plot_confusion_matrix(
            tracker.all_targets, tracker.all_preds,
            class_names=class_names,
            dataset_name=dataset_name,
            model_name=args.model_name,
            save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
        )
        plot_roc_curve(
            np.concatenate(tracker.all_targets),
            np.concatenate(tracker.all_probs),
            class_names=class_names,
            dataset_name=dataset_name,
            model_name=args.model_name,
            save_path=os.path.join(args.output_dir, "roc_curve.png"),
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset (single-dataset eval)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dual",
                        help="Model variant")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_plots", action="store_true", default=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.data_dir:
        evaluate_single(args, device)
    else:
        print("Please specify --data_dir for single-dataset evaluation.")
        print("For cross-dataset: python evaluation/cross_dataset_eval.py ...")
        print("For robustness:    python evaluation/robustness_eval.py ...")
        print("For Grad-CAM:      python visualization/grad_cam.py ...")


if __name__ == "__main__":
    main()

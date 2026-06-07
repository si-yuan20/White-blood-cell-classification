# evaluation/robustness_eval.py
# -*- coding: utf-8 -*-
"""
Salt-and-pepper noise robustness evaluation.

Evaluates model performance under increasing noise levels (p=0.00, 0.01, 0.05, 0.10).
Generates robustness bar plots and saves noisy example images.
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
from tools.noise import add_salt_pepper_noise_batch, save_noisy_examples
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(img_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def evaluate_with_noise(model, loader, device, noise_p: float = 0.0):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Noise p={noise_p}"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            elif isinstance(batch, dict):
                inputs, labels = batch["image"], batch["label"]
            else:
                continue

            if noise_p > 0:
                inputs = add_salt_pepper_noise_batch(inputs, noise_p)

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

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    auc_val = calculate_auc(targets, probs, num_classes=probs.shape[1])
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)

    return {
        "noise_p": noise_p,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "auc": float(auc_val),
        "per_class_f1": [float(x) for x in per_class_f1],
    }


def plot_robustness(results: list, output_dir: str, dataset_name: str):
    """Generate robustness bar plot (accuracy and Macro-F1 drop)."""
    import matplotlib.pyplot as plt

    noise_levels = [r["noise_p"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["macro_f1"] * 100 for r in results]

    # Clean baselines
    clean_acc = accs[0] if accs else 100.0
    clean_f1 = f1s[0] if f1s else 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy bar plot
    bars = axes[0].bar([str(p) for p in noise_levels], accs,
                       color=["green"] + ["orange"] * (len(accs) - 1))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title(f"Accuracy under Salt-and-Pepper Noise ({dataset_name})")
    axes[0].set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    # Macro-F1 drop (corrected: uses Macro-F1, not Accuracy)
    f1_drops = [(clean_f1 - f1) for f1 in f1s]
    bars2 = axes[1].bar([str(p) for p in noise_levels], f1_drops,
                        color=["green"] + ["red"] * (len(f1_drops) - 1))
    axes[1].set_ylabel("Macro-F1 Drop (%)")
    axes[1].set_title(f"Macro-F1 Drop under Noise ({dataset_name})")
    for bar, drop in zip(bars2, f1_drops):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{drop:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_bar.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Salt-and-pepper noise robustness evaluation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dual")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.10])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/robustness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_noisy_samples", action="store_true", default=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    out_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    model = create_classifier(args.model_name, num_classes=args.num_classes,
                              pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    if hasattr(model, "model"):
        model.model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Build dataset
    transform = build_transform(args.img_size)
    dataset = MedicalDataset(args.data_dir, transform=transform, two_view=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Dataset: {dataset_name} ({len(dataset)} samples)")
    print(f"Noise levels: {args.noise_levels}")

    all_results = []
    for p in args.noise_levels:
        res = evaluate_with_noise(model, loader, device, noise_p=p)
        all_results.append(res)
        print(f"  p={p:.2f}: Acc={res['accuracy']:.4f}, F1={res['macro_f1']:.4f}, AUC={res['auc']:.4f}")

    # Compute F1 drop from clean
    clean_f1 = all_results[0]["macro_f1"] if all_results else 0.0
    clean_acc = all_results[0]["accuracy"] if all_results else 0.0
    for r in all_results:
        r["macro_f1_drop"] = clean_f1 - r["macro_f1"]
        r["accuracy_drop"] = clean_acc - r["accuracy"]

    # Save results
    with open(os.path.join(out_dir, "robustness_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    with open(os.path.join(out_dir, "robustness_results.csv"), "w") as f:
        f.write("noise_p,accuracy,macro_f1,auc,macro_f1_drop,accuracy_drop\n")
        for r in all_results:
            f.write(f"{r['noise_p']},{r['accuracy']},{r['macro_f1']},{r['auc']},"
                    f"{r['macro_f1_drop']},{r['accuracy_drop']}\n")

    # Plot
    plot_robustness(all_results, out_dir, dataset_name)
    print(f"Results saved to {out_dir}")

    # Save noisy examples
    if args.save_noisy_samples:
        sample_batch = next(iter(loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_imgs = sample_batch[0][:4]
        else:
            sample_imgs = sample_batch["image"][:4]
        save_noisy_examples(sample_imgs, os.path.join(out_dir, "noisy_examples.png"), p=0.05)


if __name__ == "__main__":
    main()

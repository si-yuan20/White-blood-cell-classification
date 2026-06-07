# visualization/quantitative_interpretability.py
# -*- coding: utf-8 -*-
"""
Quantitative interpretability analysis for Grad-CAM heatmaps.

Implements the metrics reported in Table 14:
  - foreground activation ratio
  - background activation ratio
  - pointing accuracy
  - deletion score

The foreground mask is generated automatically via Otsu thresholding and
morphological post-processing. It is intended for computational assessment
only — NOT as expert pathological annotation.
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
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prepare import MedicalDataset
from dual_model import create_classifier
from utils import set_global_seed
from visualization.grad_cam import GradCAM, find_target_layer
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")


# ============================================================
# Foreground mask generation (automatic, NOT expert annotation)
# ============================================================
def generate_foreground_mask(
    image: np.ndarray,
    otsu_scale: float = 1.0,
    min_area: int = 100,
    close_kernel: int = 5,
) -> np.ndarray:
    """
    Generate a binary cell foreground mask from a raw image.

    Pipeline:
      1. Convert RGB to grayscale.
      2. Apply Otsu thresholding (scikit-image or fallback).
      3. Morphological closing to fill holes.
      4. Remove small connected components.

    The mask is automatically computed and is intended for
    computational interpretability assessment only — it is
    NOT an expert pathological annotation.

    Args:
        image: (H, W, 3) numpy array in [0, 1] or [0, 255].
        otsu_scale: Scale factor for Otsu threshold (1.0 = default).
        min_area: Minimum area for connected components to retain.
        close_kernel: Kernel size for morphological closing.

    Returns:
        Binary mask (H, W) as uint8 numpy array (0 or 1).
    """
    # Normalize to [0, 255]
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    # Grayscale
    if img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        gray = np.dot(img_u8[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_u8.squeeze()

    # Otsu threshold
    try:
        from skimage.filters import threshold_otsu
        thresh_val = threshold_otsu(gray) * otsu_scale
    except ImportError:
        # Fallback: simple percentile-based threshold
        thresh_val = np.percentile(gray, 50) * otsu_scale

    mask = (gray > thresh_val).astype(np.uint8)

    # Morphological closing
    try:
        from skimage.morphology import binary_closing, remove_small_objects, disk
        selem = disk(close_kernel)
        mask = binary_closing(mask.astype(bool), selem).astype(np.uint8)
        mask = remove_small_objects(mask.astype(bool), min_size=min_area).astype(np.uint8)
    except ImportError:
        # OpenCV fallback
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Simple connected component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 1
        mask = clean

    return mask.astype(np.uint8)


# ============================================================
# Metrics
# ============================================================
def foreground_activation_ratio(heatmap: np.ndarray, fg_mask: np.ndarray) -> float:
    """
    Foreground Activation Ratio:
      sum(heatmap * fg_mask) / sum(heatmap)

    Higher = model focuses more on the cell foreground.
    """
    hm = heatmap.astype(np.float64)
    fg = fg_mask.astype(np.float64)
    total = hm.sum()
    if total == 0:
        return 0.0
    return float((hm * fg).sum() / total)


def background_activation_ratio(heatmap: np.ndarray, fg_mask: np.ndarray) -> float:
    """
    Background Activation Ratio:
      sum(heatmap * (1 - fg_mask)) / sum(heatmap)

    Lower = model focuses less on background.
    """
    hm = heatmap.astype(np.float64)
    bg = 1.0 - fg_mask.astype(np.float64)
    total = hm.sum()
    if total == 0:
        return 0.0
    return float((hm * bg).sum() / total)


def pointing_accuracy(heatmap: np.ndarray, fg_mask: np.ndarray) -> int:
    """
    Pointing Accuracy: whether the maximum response point of the
    heatmap falls within the foreground mask.

    Returns 0 or 1.
    """
    max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(fg_mask[max_loc] > 0)


def deletion_score_single(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    device: torch.device,
    delete_ratio: float = 0.15,
    fill_value: float = 0.0,
) -> float:
    """
    Compute deletion score for a single image.

    1. Find top delete_ratio fraction of heatmap pixels.
    2. Mask (zero-fill) those pixels in the input.
    3. Compare predicted class confidence before and after deletion.

    deletion_score = confidence_original - confidence_masked

    Args:
        model: Trained model in eval mode.
        image_tensor: (1, C, H, W) original input to model.
        heatmap: (H_h, W_h) Grad-CAM heatmap.
        device: Torch device.
        delete_ratio: Fraction of high-activation pixels to remove.
        fill_value: Value to fill deleted pixels with.

    Returns:
        Deletion score (float).
    """
    was_training = model.training
    model.eval()

    # Original confidence
    img = image_tensor.to(device)
    with torch.no_grad():
        out = model(img)
        probs = F.softmax(out, dim=-1)
        pred_class = out.argmax(dim=1).item()
        orig_conf = probs[0, pred_class].item()

    # Resize heatmap to image size
    hm_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hm_resized = F.interpolate(
        hm_tensor, size=img.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # Flatten and find top-k pixels
    flat_hm = hm_resized.flatten()
    k = max(1, int(len(flat_hm) * delete_ratio))
    threshold = np.partition(flat_hm, -k)[-k]
    mask_2d = (hm_resized >= threshold).astype(np.float32)
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Mask image
    masked_img = img.clone()
    # Broadcast mask across channels
    masked_img = masked_img * (1.0 - mask_tensor) + fill_value * mask_tensor

    with torch.no_grad():
        out_masked = model(masked_img)
        probs_masked = F.softmax(out_masked, dim=-1)
        masked_conf = probs_masked[0, pred_class].item()

    if was_training:
        model.train()

    return max(0.0, orig_conf - masked_conf)


# ============================================================
# Main evaluation loop
# ============================================================
def evaluate_quantitative(
    model: torch.nn.Module,
    dataloader: DataLoader,
    gradcam: GradCAM,
    raw_images: np.ndarray,
    device: torch.device,
    delete_ratio: float = 0.15,
    max_samples: int = 100,
):
    """
    Run quantitative interpretability evaluation over a dataset.

    Returns per-sample list and summary dict.
    """
    results = []
    far_list, bg_list, pa_list, ds_list = [], [], [], []

    for idx in tqdm(range(min(max_samples, len(dataloader.dataset))), desc="Quantitative interp"):
        try:
            img_tensor, label = dataloader.dataset[idx]
            if isinstance(img_tensor, list):
                img_tensor = img_tensor[0]
            img_batch = img_tensor.unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(img_batch)
                pred = output.argmax(dim=1).item()
                prob = F.softmax(output, dim=1)[0, pred].item()

            # Grad-CAM heatmap
            heatmap = gradcam(img_batch, class_idx=pred)

            # Raw image for mask generation
            raw_img = raw_images[idx]
            if raw_img.max() <= 1.0:
                raw_disp = raw_img
            else:
                raw_disp = raw_img / 255.0

            # Foreground mask
            fg_mask = generate_foreground_mask(raw_disp)

            # Resize heatmap to mask size
            hm_resized = F.interpolate(
                torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
                size=fg_mask.shape, mode="bilinear", align_corners=False,
            ).squeeze().numpy()
            if hm_resized.max() > 0:
                hm_resized /= hm_resized.max()

            # Metrics
            far = foreground_activation_ratio(hm_resized, fg_mask)
            bg = background_activation_ratio(hm_resized, fg_mask)
            pa = pointing_accuracy(hm_resized, fg_mask)
            ds = deletion_score_single(model, img_batch, heatmap, device, delete_ratio=delete_ratio)

            far_list.append(far)
            bg_list.append(bg)
            pa_list.append(pa)
            ds_list.append(ds)

            results.append({
                "sample_idx": idx,
                "true_label": int(label) if not isinstance(label, torch.Tensor) else label.item(),
                "pred_label": int(pred),
                "confidence": float(prob),
                "foreground_activation_ratio": float(far),
                "background_activation_ratio": float(bg),
                "pointing_accuracy": int(pa),
                "deletion_score": float(ds),
            })

        except Exception as e:
            print(f"  [WARN] Sample {idx} failed: {e}")
            continue

    n = max(len(far_list), 1)
    summary = {
        "foreground_activation_ratio_mean": float(np.mean(far_list)) if far_list else 0.0,
        "foreground_activation_ratio_std": float(np.std(far_list)) if far_list else 0.0,
        "background_activation_ratio_mean": float(np.mean(bg_list)) if bg_list else 0.0,
        "background_activation_ratio_std": float(np.std(bg_list)) if bg_list else 0.0,
        "pointing_accuracy_mean": float(np.mean(pa_list)) if pa_list else 0.0,
        "pointing_accuracy_std": float(np.std(pa_list)) if pa_list else 0.0,
        "deletion_score_mean": float(np.mean(ds_list)) if ds_list else 0.0,
        "deletion_score_std": float(np.std(ds_list)) if ds_list else 0.0,
        "num_samples": n,
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative interpretability analysis (Table 14)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="dual_full",
                        help="Model variant name")
    parser.add_argument("--target_layer", type=str, default="auto",
                        choices=["auto", "convnext_last", "saf_out"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--delete_ratio", type=float, default=0.15,
                        help="Fraction of high-activation pixels to delete")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/interpretability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true",
                        help="Test with random model and minimal samples")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    out_dir = os.path.join(args.output_dir, f"{dataset_name}_{args.model_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    model = create_classifier(args.model_name, num_classes=args.num_classes,
                              pretrained=False)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        if hasattr(model, "model"):
            model.model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    elif args.dry_run:
        print("[DRY_RUN] Using randomly initialized model.")
    else:
        print("[WARN] No checkpoint provided. Using randomly initialized model.")

    model = model.to(device)
    model.eval()

    # Build dataset (normalized for model input)
    transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = MedicalDataset(args.data_dir, transform=transform, two_view=False)

    # Also load raw images (un-normalized) for mask generation
    raw_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=args.img_size, width=args.img_size),
        ToTensorV2(),
    ])
    raw_dataset = MedicalDataset(args.data_dir, transform=raw_transform, two_view=False)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Pre-load raw images for mask generation
    n_load = min(args.max_samples, len(raw_dataset))
    raw_images = []
    for i in range(n_load):
        raw_img, _ = raw_dataset[i]
        raw_np = raw_img.cpu().numpy().transpose(1, 2, 0)
        raw_images.append(raw_np)

    raw_images = np.array(raw_images) if raw_images else np.array([])

    # Build GradCAM
    target_layer = find_target_layer(model, args.target_layer)
    print(f"Target layer: {target_layer.__class__.__name__}")
    gradcam = GradCAM(model, target_layer)

    # Run evaluation
    print(f"Dataset: {dataset_name} ({len(dataset)} samples)")
    print(f"Model: {args.model_name}")
    print(f"Max samples: {args.max_samples}, Delete ratio: {args.delete_ratio}")

    per_sample, summary = evaluate_quantitative(
        model, loader, gradcam, raw_images, device,
        delete_ratio=args.delete_ratio, max_samples=args.max_samples,
    )

    gradcam.remove_hooks()

    # Save per-sample results
    with open(os.path.join(out_dir, "per_sample_results.json"), "w") as f:
        json.dump(per_sample, f, indent=2)

    # Per-sample CSV
    import csv
    csv_fields = ["sample_idx", "true_label", "pred_label", "confidence",
                  "foreground_activation_ratio", "background_activation_ratio",
                  "pointing_accuracy", "deletion_score"]
    with open(os.path.join(out_dir, "per_sample_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_sample)

    # Save summary (Table 14 format)
    summary_row = {
        "dataset": dataset_name,
        "method": args.model_name,
        **summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary_row, f, indent=2)

    # Table 14 CSV
    tbl_fields = ["dataset", "method",
                  "foreground_activation_ratio_mean", "foreground_activation_ratio_std",
                  "pointing_accuracy_mean", "pointing_accuracy_std",
                  "deletion_score_mean", "deletion_score_std",
                  "background_activation_ratio_mean", "background_activation_ratio_std"]
    with open(os.path.join(out_dir, "table14_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tbl_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(summary_row)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Quantitative Interpretability — {dataset_name} / {args.model_name}")
    print(f"{'='*70}")
    print(f"Foreground Activation Ratio:  {summary['foreground_activation_ratio_mean']:.4f} ± {summary['foreground_activation_ratio_std']:.4f}")
    print(f"Background Activation Ratio:  {summary['background_activation_ratio_mean']:.4f} ± {summary['background_activation_ratio_std']:.4f}")
    print(f"Pointing Accuracy:            {summary['pointing_accuracy_mean']:.4f} ± {summary['pointing_accuracy_std']:.4f}")
    print(f"Deletion Score:               {summary['deletion_score_mean']:.4f} ± {summary['deletion_score_std']:.4f}")
    print(f"Samples: {summary['num_samples']}")
    print(f"{'='*70}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

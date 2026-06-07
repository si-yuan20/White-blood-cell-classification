# visualization/grad_cam.py
# -*- coding: utf-8 -*-
"""
Grad-CAM heatmap visualization for interpretability analysis.

Uses registered forward/backward hooks to compute Grad-CAM from a target layer.
Supports ConvNeXt branch, SAF output, and Mamba restored spatial features.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prepare import MedicalDataset
from dual_model import create_classifier
from utils import set_global_seed
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GradCAM:
    """
    Grad-CAM via forward/backward hooks.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(image_tensor, class_idx=None)
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._hook_handle_fwd = target_layer.register_forward_hook(self._save_activation)
        self._hook_handle_bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Args:
            x: Input tensor (B, C, H, W) — batch_size >= 2 recommended if model has BatchNorm1d.
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            Heatmap as numpy array (H, W) in [0, 1].
        """
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            output = self.model(x)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            score = output[0, class_idx]
            score.backward()

        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer.")

        # Global average pool gradients — use first sample
        pooled_grads = gradients[0:1].mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted = (pooled_grads * activations[0:1]).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        heatmap = F.relu(weighted).squeeze().cpu().numpy()  # (H', W')

        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    def remove_hooks(self):
        self._hook_handle_fwd.remove()
        self._hook_handle_bwd.remove()


def find_target_layer(model: torch.nn.Module, layer_name: str = "convnext_last") -> torch.nn.Module:
    """
    Find a target layer for Grad-CAM by heuristic matching.

    Options:
      - "convnext_last": Last Conv2d in ConvNeXt backbone
      - "saf_out": SAF refinement output (last Conv2d in SAF)
      - "auto": Auto-select first available
    """
    candidates = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            candidates.append((name, module))

    if not candidates:
        raise ValueError("No Conv2d layers found in model for Grad-CAM.")

    if layer_name == "auto":
        return candidates[-1][1]  # Last conv layer

    for cname, cmod in candidates:
        if "convnext" in cname.lower() and "backbone" in cname.lower():
            if layer_name == "convnext_last":
                # Find last convnext conv
                last_convnext = [(n, m) for n, m in candidates if "convnext" in n.lower()]
                if last_convnext:
                    return last_convnext[-1][1]
        if "saf" in cname.lower() and ("refine" in cname.lower() or "dwsep" in cname.lower()):
            if layer_name == "saf_out":
                return cmod

    # Fallback: last Conv2d in model
    return candidates[-1][1]


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        img: (H, W, 3) image in [0, 1].
        heatmap: (H, W) heatmap in [0, 1].
        alpha: Overlay transparency.

    Returns:
        (H, W, 3) overlay image.
    """
    import matplotlib.cm as cm
    heatmap_resized = np.array(
        torch.nn.functional.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=img.shape[:2], mode="bilinear", align_corners=False,
        ).squeeze().numpy()
    )
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # (H, W, 3)
    overlay = (1 - alpha) * img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM interpretability analysis")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dual")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--target_layer", type=str, default="auto",
                        choices=["auto", "convnext_last", "saf_out"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/grad_cam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
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

    # Find target layer and create GradCAM
    target_layer = find_target_layer(model, args.target_layer)
    print(f"Target layer for Grad-CAM: {target_layer.__class__.__name__}")
    gradcam = GradCAM(model, target_layer)

    # Build dataset
    transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = MedicalDataset(args.data_dir, transform=transform, two_view=False)

    # Also need un-normalized images for overlay
    raw_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=args.img_size, width=args.img_size),
        ToTensorV2(),
    ])
    raw_dataset = MedicalDataset(args.data_dir, transform=raw_transform, two_view=False)

    n_samples = min(args.max_samples, len(dataset))

    for idx in tqdm(range(n_samples), desc="Generating Grad-CAM"):
        img_tensor, label = dataset[idx]
        raw_img, _ = raw_dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_batch)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred].item()

        # Grad-CAM
        try:
            heatmap = gradcam(img_batch, class_idx=pred)
        except Exception as e:
            print(f"  [WARN] Grad-CAM failed for sample {idx}: {e}")
            continue

        # Prepare raw image for overlay
        raw_np = raw_img.cpu().numpy().transpose(1, 2, 0)
        raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)

        overlay = overlay_heatmap(raw_np, heatmap, alpha=0.5)

        # Save
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_np)
        axes[0].set_title(f"Original (True:{dataset.class_names[label]})")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Pred:{dataset.class_names[pred]} {prob:.2f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gradcam_{idx:03d}.png"), dpi=150)
        plt.close()

    gradcam.remove_hooks()
    print(f"Saved {n_samples} Grad-CAM images to {out_dir}")


if __name__ == "__main__":
    main()

# tools/noise.py
# -*- coding: utf-8 -*-
"""Salt-and-pepper noise generation for robustness evaluation."""

import numpy as np
import torch


def add_salt_pepper_noise(image: torch.Tensor, p: float = 0.01) -> torch.Tensor:
    """
    Add salt-and-pepper noise to a single image tensor.

    Args:
        image: Tensor of shape (C, H, W) with values in [0, 1] or normalized range.
        p: Noise probability (fraction of pixels to corrupt).
           Half become salt (max value), half become pepper (min value).

    Returns:
        Noisy image tensor of same shape.
    """
    if p <= 0:
        return image.clone()

    c, h, w = image.shape
    noisy = image.clone()

    # Determine value range for salt/pepper
    vmin = image.min().item()
    vmax = image.max().item()

    # Generate random mask
    mask = torch.rand(h, w) < p
    salt_mask = mask & (torch.rand(h, w) < 0.5)
    pepper_mask = mask & (~salt_mask)

    for ch in range(c):
        noisy[ch][salt_mask] = vmax
        noisy[ch][pepper_mask] = vmin

    return noisy


def add_salt_pepper_noise_batch(images: torch.Tensor, p: float = 0.01) -> torch.Tensor:
    """
    Add salt-and-pepper noise to a batch of images.

    Args:
        images: Tensor of shape (B, C, H, W).
        p: Noise probability.

    Returns:
        Noisy batch tensor of same shape.
    """
    if p <= 0:
        return images.clone()

    noisy = []
    for i in range(images.shape[0]):
        noisy.append(add_salt_pepper_noise(images[i], p))
    return torch.stack(noisy, dim=0)


def save_noisy_examples(images: torch.Tensor, save_path: str, p: float):
    """
    Save representative noisy images for qualitative inspection.

    Args:
        images: (B, C, H, W) batch of images.
        save_path: Output image path.
        p: Noise level used.
    """
    import matplotlib.pyplot as plt

    n = min(4, images.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].set_title(f"SP noise p={p}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

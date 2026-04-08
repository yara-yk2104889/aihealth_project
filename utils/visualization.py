"""
utils/visualization.py — Visualisation helpers.

All functions are standalone (no trainer dependency) so they can be called
from a notebook or from train.py equally.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    title: str = "",
):
    """
    Display up to n_samples slices as a grid:
        Row 0 — DWI channel
        Row 1 — ADC channel
        Row 2 — Ground-truth mask
        Row 3 — Predicted mask (thresholded)

    Parameters
    ----------
    images   : [B, 2, H, W]
    masks    : [B, 1, H, W]
    preds    : [B, 1, H, W]  (raw logits)
    """
    n = min(n_samples, images.size(0))
    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12))

    if n == 1:
        axes = axes[:, None]   # ensure 2-D indexing

    pred_bin = (torch.sigmoid(preds) > 0.5).float()
    row_labels = ["DWI", "ADC", "GT Mask", "Prediction"]

    for i in range(n):
        dwi  = _to_numpy(images[i, 0])
        adc  = _to_numpy(images[i, 1])
        gt   = _to_numpy(masks[i, 0])
        pred = _to_numpy(pred_bin[i, 0])

        for row, (arr, cmap) in enumerate(zip(
            [dwi, adc, gt, pred],
            ["gray", "gray", "hot", "hot"]
        )):
            ax = axes[row, i]
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1 if row >= 2 else None)
            ax.axis("off")
            if i == 0:
                ax.set_title(row_labels[row], fontsize=10, loc="left")

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"[VIS] Saved → {save_path}")

    plt.show()
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot loss + dice curves for train and val.

    Parameters
    ----------
    history : dict returned by Trainer.fit()
    """
    metrics_to_plot = ["loss", "dice", "iou"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))

    for ax, metric in zip(axes, metrics_to_plot):
        train_key = f"train_{metric}"
        val_key   = f"val_{metric}"

        if train_key in history:
            ax.plot(history[train_key], label="train", linewidth=2)
        if val_key in history:
            ax.plot(history[val_key],   label="val",   linewidth=2, linestyle="--")

        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"[VIS] Saved → {save_path}")

    plt.show()
    plt.close(fig)

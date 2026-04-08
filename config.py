"""
config.py — All hyperparameters and paths in one place.
Edit this file before running train.py.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    data_root: str = r"C:\Users\yarah\Desktop\capstone\ISLES-2022\ISLES-2022"
    checkpoint_dir: str = "checkpoints"
    vis_dir: str = "visualizations"

    # ── Data ──────────────────────────────────────────────────────────────
    target_size: Tuple[int, int] = (128, 128)   # (H, W) for every 2-D slice
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0                         # set >0 on Linux/Mac

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 5              # ReduceLROnPlateau patience

    # ── Model ─────────────────────────────────────────────────────────────
    in_channels: int = 2                     # DWI + ADC
    out_channels: int = 1                    # binary mask
    features: Tuple[int, ...] = (32, 64, 128, 256)

    # ── Loss ──────────────────────────────────────────────────────────────
    # Loss = bce_weight * BCE  +  dice_weight * Dice
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # ── Misc ──────────────────────────────────────────────────────────────
    device: str = "cuda"                     # "cuda" or "cpu"
    log_interval: int = 10                   # print every N batches
    vis_n_samples: int = 4                   # images to visualise after each epoch

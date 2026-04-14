"""
train.py — Main training script for stroke lesion segmentation.

Usage
-----
    python train.py                     # use defaults in config.py
    python train.py --epochs 100        # override any Config field

Quick experiment flags
----------------------
    --loss  bce_dice | focal            (default: bce_dice)
    --lr    1e-3
    --features  32 64 128 256
"""

import argparse
import os
import sys
import torch

# Make sub-packages importable without installing
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data import build_loaders
from models import UNet
from losses import BCEDiceLoss, FocalLoss
from training import Trainer
from utils import visualize_batch, plot_training_curves


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on ISLES 2022")
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--loss",        choices=["bce_dice", "focal"], default="bce_dice")
    parser.add_argument("--device",      default=None, help="cuda or cpu")
    parser.add_argument("--features",    type=int, nargs="+", default=None)
    parser.add_argument("--no_vis",      action="store_true", help="Skip visualisation")
    return parser.parse_args()


# ── Setup helpers ─────────────────────────────────────────────────────────────

def get_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cfg.device)


def build_loss(loss_name: str, cfg: Config) -> torch.nn.Module:
    if loss_name == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    # default
    return BCEDiceLoss(bce_weight=cfg.bce_weight)


# ── Visualisation callback ────────────────────────────────────────────────────

def visualise_val_samples(model, val_loader, device, cfg, epoch):
    model.eval()
    images, masks, _ = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        preds = model(images)

    save_path = os.path.join(cfg.vis_dir, f"epoch_{epoch:03d}.png")
    visualize_batch(
        images, masks, preds,
        n_samples=cfg.vis_n_samples,
        save_path=save_path,
        title=f"Epoch {epoch}",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = Config()

    # Apply CLI overrides
    if args.data_root:  cfg.data_root  = args.data_root
    if args.epochs:     cfg.num_epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.learning_rate = args.lr
    if args.device:     cfg.device     = args.device
    if args.features:   cfg.features   = tuple(args.features)

    device = get_device(cfg)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Config: {cfg}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        data_root      = cfg.data_root,
        target_size    = cfg.target_size,
        val_fraction   = cfg.val_fraction,
        batch_size     = cfg.batch_size,
        num_workers    = cfg.num_workers,
        seed           = cfg.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        in_channels  = cfg.in_channels,
        out_channels = cfg.out_channels,
        features     = cfg.features,
    )
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.scheduler_patience, factor=0.5, verbose=True,
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = build_loss(args.loss, cfg)
    print(f"[INFO] Loss function: {criterion.__class__.__name__}")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        optimizer      = optimizer,
        criterion      = criterion,
        device         = device,
        checkpoint_dir = cfg.checkpoint_dir,
        scheduler      = scheduler,
        log_interval   = cfg.log_interval,
        # refiner = SAMRefiner(...)   ← plug in SAM here later
    )

    # ── Train ─────────────────────────────────────────────────────────────
    history = trainer.fit(train_loader, val_loader, cfg.num_epochs)

    # ── Post-training visualisation ────────────────────────────────────────
    if not args.no_vis:
        os.makedirs(cfg.vis_dir, exist_ok=True)
        visualise_val_samples(model, val_loader, device, cfg, epoch=cfg.num_epochs)
        plot_training_curves(
            history,
            save_path=os.path.join(cfg.vis_dir, "training_curves.png"),
        )

    return history


if __name__ == "__main__":
    main()

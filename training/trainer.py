"""
training/trainer.py — Training and validation loops.

The Trainer is intentionally thin so that experimenting with different models,
losses, and refiners is frictionless.

SAM refinement hook
-------------------
Pass refiner=<callable> to Trainer.  The refiner receives the raw model output
and the input image and returns a refined output.  Default is None (no-op).

    class SAMRefiner(nn.Module):
        def __call__(self, logits, image): ...

    trainer = Trainer(..., refiner=SAMRefiner())
"""

import os
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics import SegmentationMetrics


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        scheduler: Optional[ReduceLROnPlateau] = None,
        refiner: Optional[Callable] = None,      # SAM refinement hook
        log_interval: int = 10,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.device         = device
        self.scheduler      = scheduler
        self.refiner        = refiner
        self.log_interval   = log_interval
        self.checkpoint_dir = checkpoint_dir
        self._best_val_loss = float("inf")

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Single epoch loops ────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics   = SegmentationMetrics()
        total_loss = 0.0

        for batch_idx, (images, masks, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)

            # Optional SAM refinement (no-op by default)
            if self.refiner is not None:
                logits = self.refiner(logits, images)

            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            metrics.update(logits.detach(), masks.detach())

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]"
                    f"  loss={loss.item():.4f}"
                )

        results = metrics.compute()
        results["loss"] = total_loss / len(loader)
        return results

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics    = SegmentationMetrics()
        total_loss = 0.0

        for images, masks, _ in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            if self.refiner is not None:
                logits = self.refiner(logits, images)

            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            metrics.update(logits, masks)

        results = metrics.compute()
        results["loss"] = total_loss / len(loader)
        return results

    # ── Full training run ─────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, list]:
        """
        Train for num_epochs, tracking history.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'train_dice',
                  'val_dice', 'train_iou', 'val_iou', etc.
        """
        history: Dict[str, list] = {}

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*55}")
            print(f"  Epoch {epoch}/{num_epochs}")
            print(f"{'='*55}")

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics   = self.val_epoch(val_loader)

            # Step LR scheduler on val loss
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Accumulate history
            for k, v in train_metrics.items():
                history.setdefault(f"train_{k}", []).append(v)
            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(v)

            # Checkpoint on improved val loss
            if val_metrics["loss"] < self._best_val_loss:
                self._best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics)

        print("\n[INFO] Training complete.")
        print(f"[INFO] Best val loss: {self._best_val_loss:.4f}")
        return history

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, path)
        print(f"  [CKPT] Saved best model → {path}  (val_loss={val_metrics['loss']:.4f})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[INFO] Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
        return ckpt

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_epoch(epoch, train, val):
        print(
            f"\n  Train | loss={train['loss']:.4f}  dice={train['dice']:.4f}"
            f"  iou={train['iou']:.4f}  prec={train['precision']:.4f}  rec={train['recall']:.4f}"
        )
        print(
            f"  Val   | loss={val['loss']:.4f}  dice={val['dice']:.4f}"
            f"  iou={val['iou']:.4f}  prec={val['precision']:.4f}  rec={val['recall']:.4f}"
        )

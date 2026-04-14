"""
losses/losses.py — Loss functions for binary segmentation.

All losses expect:
    pred   : FloatTensor [B, 1, H, W]  (raw logits, before sigmoid)
    target : FloatTensor [B, 1, H, W]  (binary, 0 or 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss.
        L = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat   = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted combination: α * BCE + (1-α) * Dice.
    Default α = 0.5 gives equal weight.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + \
               self.dice_weight * self.dice(pred, target)


class FocalLoss(nn.Module):
    """
    Binary Focal Loss — down-weights easy negatives.
        FL = -α_t * (1 - p_t)^γ * log(p_t)
    Useful when lesion pixels are rare (heavy class imbalance).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t      = torch.exp(-bce_loss)
        alpha_t  = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal    = alpha_t * (1.0 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

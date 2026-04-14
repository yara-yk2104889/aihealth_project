"""
metrics/metrics.py — Segmentation evaluation metrics.

All functions accept:
    pred   : FloatTensor [B, 1, H, W]  (raw logits or probabilities)
    target : FloatTensor [B, 1, H, W]  (binary, 0 or 1)
    threshold : float — binarisation threshold on sigmoid(pred)
"""

from typing import Dict, Tuple
import torch


def _binarise(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(pred) > threshold).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin    = _binarise(pred, threshold).view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_bin * target_flat).sum(dim=1)
    score = (2.0 * intersection + smooth) / (
        pred_bin.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return score.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin    = _binarise(pred, threshold).view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_bin * target_flat).sum(dim=1)
    union        = pred_bin.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision_recall(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1.0) -> Tuple[float, float]:
    pred_bin    = _binarise(pred, threshold).view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_bin * target_flat).sum(dim=1)
    fp = (pred_bin * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_bin) * target_flat).sum(dim=1)

    precision = ((tp + smooth) / (tp + fp + smooth)).mean().item()
    recall    = ((tp + smooth) / (tp + fn + smooth)).mean().item()
    return precision, recall


# ── Aggregator ────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Running accumulator for a whole epoch.

    Usage
    -----
    m = SegmentationMetrics()
    for pred, target in loader:
        m.update(pred, target)
    results = m.compute()   # dict with dice, iou, precision, recall
    m.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._totals: Dict[str, float] = {}
        self._count = 0

    def reset(self):
        self._totals = {}
        self._count  = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            dice       = dice_score(pred, target, self.threshold)
            iou        = iou_score(pred, target, self.threshold)
            prec, rec  = precision_recall(pred, target, self.threshold)

        batch = pred.size(0)
        for key, val in [("dice", dice), ("iou", iou),
                         ("precision", prec), ("recall", rec)]:
            self._totals[key] = self._totals.get(key, 0.0) + val * batch
        self._count += batch

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            return {k: 0.0 for k in ("dice", "iou", "precision", "recall")}
        return {k: v / self._count for k, v in self._totals.items()}

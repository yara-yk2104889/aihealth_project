"""
utils/sam_refiner.py — SAM-based refinement of U-Net predictions.

Pipeline
--------
1. Run U-Net → coarse logit
2. Threshold → binary mask → extract bounding box
3. Feed image + bounding box as prompt to SAM
4. SAM refines boundaries → final mask

No retraining required. SAM runs as pure post-processing.

Input images are converted from 2-channel (DWI+ADC) to 3-channel RGB-like
for SAM by stacking: [DWI, ADC, DWI].
"""

import numpy as np
import torch
import torch.nn.functional as F


class SAMRefiner:
    """
    Wraps a SAM predictor to refine U-Net predictions batch by batch.

    Parameters
    ----------
    predictor    : SamPredictor from segment_anything
    threshold    : binarisation threshold on U-Net sigmoid output
    min_pixels   : skip SAM if U-Net mask has fewer pixels (no lesion detected)
    """

    def __init__(self, predictor, threshold: float = 0.5, min_pixels: int = 10):
        self.predictor  = predictor
        self.threshold  = threshold
        self.min_pixels = min_pixels

    def refine_batch(
        self,
        images: torch.Tensor,     # [B, 2, H, W]  DWI + ADC
        unet_logits: torch.Tensor, # [B, 1, H, W]  raw U-Net logits
    ) -> torch.Tensor:
        """Returns refined binary masks [B, 1, H, W] as float tensors."""
        refined = []
        for img, logit in zip(images, unet_logits):
            refined.append(self._refine_single(img, logit))
        return torch.stack(refined, dim=0).to(unet_logits.device)

    def _refine_single(
        self,
        image: torch.Tensor,   # [2, H, W]
        logit: torch.Tensor,   # [1, H, W]
    ) -> torch.Tensor:
        probs    = torch.sigmoid(logit).squeeze().cpu().numpy()  # [H, W]
        unet_bin = (probs > self.threshold).astype(np.uint8)

        # If U-Net found nothing, return zeros (SAM has nothing to refine)
        if unet_bin.sum() < self.min_pixels:
            return torch.zeros_like(logit)

        # Convert [2,H,W] → [H,W,3] uint8 for SAM
        img_rgb = self._to_rgb(image)
        self.predictor.set_image(img_rgb)

        # Bounding box prompt from U-Net mask
        bbox = self._get_bbox(unet_bin)

        sam_masks, scores, _ = self.predictor.predict(
            box               = bbox,
            multimask_output  = True,
        )
        # Pick highest-scoring SAM mask
        best = sam_masks[np.argmax(scores)]   # [H, W] bool

        return torch.from_numpy(best.astype(np.float32)).unsqueeze(0)

    @staticmethod
    def _to_rgb(image: torch.Tensor) -> np.ndarray:
        """[2, H, W] float → [H, W, 3] uint8.  Stack DWI, ADC, DWI."""
        dwi = image[0].cpu().numpy()
        adc = image[1].cpu().numpy()
        rgb = np.stack([dwi, adc, dwi], axis=-1)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _get_bbox(mask: np.ndarray) -> np.ndarray:
        """Binary mask [H,W] → [x_min, y_min, x_max, y_max] bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Add small padding
        H, W = mask.shape
        rmin = max(0, rmin - 2)
        rmax = min(H - 1, rmax + 2)
        cmin = max(0, cmin - 2)
        cmax = min(W - 1, cmax + 2)
        return np.array([cmin, rmin, cmax, rmax])

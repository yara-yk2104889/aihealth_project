"""
data/dataset_3d.py — 3-D volumetric data loading for ISLES 2022.

Each item is a full 3-D volume resampled to a fixed (D, H, W) shape,
rather than individual 2-D axial slices.

Returns
-------
image : FloatTensor [2, D, H, W]   — channel 0 = DWI, channel 1 = ADC
mask  : FloatTensor [1, D, H, W]   — binary lesion mask {0, 1}
meta  : dict                       — {'subject': str}
"""

import random
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dataset import find_isles_cases, train_val_split, percentile_normalise


class ISLES3DDataset(Dataset):
    """
    Loads full 3-D DWI+ADC volumes and masks, resampled to `target_size`.

    Parameters
    ----------
    cases       : list of (dwi_path, adc_path, mask_path)
    target_size : (D, H, W) to resample every volume to; default (32, 128, 128)
    augment     : random axis flips during training
    """

    def __init__(
        self,
        cases: List[Tuple[str, str, str]],
        target_size: Tuple[int, int, int] = (32, 128, 128),
        augment: bool = False,
    ):
        self.cases       = cases
        self.target_size = target_size
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        dwi_path, adc_path, mask_path = self.cases[idx]
        subject = Path(dwi_path).parts[-4]   # sub-strokecaseXXXX

        dwi_t  = self._load_resize(dwi_path,  is_mask=False)  # [1, D, H, W]
        adc_t  = self._load_resize(adc_path,  is_mask=False)
        mask_t = self._load_resize(mask_path, is_mask=True)

        image = torch.cat([dwi_t, adc_t], dim=0)  # [2, D, H, W]

        if self.augment:
            image, mask_t = self._augment(image, mask_t)

        return image, mask_t, {"subject": subject}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_resize(self, path: str, is_mask: bool) -> torch.Tensor:
        """Load a NIfTI volume, normalise/binarise, resize to target_size."""
        vol = nib.load(path).get_fdata().astype(np.float32)
        if vol.ndim == 4:           # DWI sometimes has an extra b-value dim
            vol = vol[..., -1]

        if is_mask:
            vol = (vol > 0).astype(np.float32)
        else:
            vol = percentile_normalise(vol)

        # NiBabel loads as (H, W, D) → transpose to (D, H, W)
        vol = vol.transpose(2, 0, 1)

        t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        mode = "nearest" if is_mask else "trilinear"
        align = False if mode == "trilinear" else None
        t = F.interpolate(t, size=self.target_size, mode=mode, align_corners=align)
        return t.squeeze(0)  # [1, D, H, W]

    @staticmethod
    def _augment(
        image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for dim in [-1, -2, -3]:
            if random.random() > 0.5:
                image = torch.flip(image, dims=[dim])
                mask  = torch.flip(mask,  dims=[dim])
        return image, mask


# ── Convenience builder ───────────────────────────────────────────────────────

def build_3d_loaders(
    data_root: str,
    target_size: Tuple[int, int, int] = (32, 128, 128),
    val_fraction: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Discover ISLES 2022 cases, split by patient, return (train_loader, val_loader)
    yielding full 3-D volumes.
    """
    cases = find_isles_cases(data_root)
    train_cases, val_cases = train_val_split(cases, val_fraction, seed)

    print(f"[INFO] 3D — Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

    train_ds = ISLES3DDataset(train_cases, target_size, augment=True)
    val_ds   = ISLES3DDataset(val_cases,   target_size, augment=False)

    print(f"[INFO] 3D — Train volumes: {len(train_ds)} | Val volumes: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader

"""
data/dataset.py — ISLES 2022 data loading pipeline.

Structure expected on disk:
    <data_root>/
        sub-strokecaseXXXX/
            ses-0001/
                dwi/
                    *_dwi.nii.gz
                    *_adc.nii.gz
        derivatives/
            sub-strokecaseXXXX/
                ses-0001/
                    *_msk.nii.gz

Each 3-D volume is split into 2-D axial slices.

Lazy loading strategy
---------------------
__init__ scans only mask headers/data to build the slice index — fast.
__getitem__ loads DWI/ADC on demand with a small in-memory cache (default
cache_size=15 volumes, ~3 volumes per case * 5 cases in flight at once).
This keeps startup fast and memory bounded regardless of dataset size.
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Path discovery ────────────────────────────────────────────────────────────

def find_isles_cases(data_root: str) -> List[Tuple[str, str, str]]:
    """
    Walk the ISLES-2022 BIDS tree and return a list of
    (dwi_path, adc_path, mask_path) for every subject with all three files.
    """
    data_root = Path(data_root)
    cases = []

    for subject_dir in sorted(data_root.glob("sub-strokecase*")):
        if not subject_dir.is_dir():
            continue

        dwi_files  = list(subject_dir.glob("ses-*/dwi/*_dwi.nii.gz"))
        adc_files  = list(subject_dir.glob("ses-*/dwi/*_adc.nii.gz"))
        mask_files = list(
            (data_root / "derivatives" / subject_dir.name).glob("ses-*/*_msk.nii.gz")
        )

        if dwi_files and adc_files and mask_files:
            cases.append((str(dwi_files[0]), str(adc_files[0]), str(mask_files[0])))
        else:
            missing = [m for m, f in [("DWI", dwi_files), ("ADC", adc_files), ("mask", mask_files)] if not f]
            print(f"[WARN] Skipping {subject_dir.name}: missing {', '.join(missing)}")

    print(f"[INFO] Found {len(cases)} complete cases in {data_root.name}")
    return cases


def train_val_split(
    cases: List[Tuple[str, str, str]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List, List]:
    """Patient-level split — no slice leakage between train and val."""
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


# ── Normalisation ─────────────────────────────────────────────────────────────

def percentile_normalise(vol: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Clip to [p_lo, p_hi] percentiles, then scale to [0, 1]."""
    p_lo, p_hi = np.percentile(vol, lo), np.percentile(vol, hi)
    vol = np.clip(vol, p_lo, p_hi)
    if p_hi > p_lo:
        vol = (vol - p_lo) / (p_hi - p_lo)
    return vol.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ISLESDataset(Dataset):
    """
    Yields 2-D axial slices as tensors.

    Returns
    -------
    image : FloatTensor [2, H, W]   — channel 0 = DWI, channel 1 = ADC
    mask  : FloatTensor [1, H, W]   — binary lesion mask {0, 1}
    meta  : dict                    — {'subject': str, 'slice': int}

    Lazy loading
    ------------
    Only mask volumes are pre-scanned at construction (to build the index).
    DWI/ADC volumes are loaded on first access and cached up to `cache_size`
    volumes (oldest evicted first).  Masks are cached separately.
    """

    def __init__(
        self,
        cases: List[Tuple[str, str, str]],
        target_size: Tuple[int, int] = (128, 128),
        skip_empty: bool = False,
        augment: bool = False,
        cache_size: int = 20,
    ):
        self.cases       = cases         # list of (dwi_path, adc_path, mask_path)
        self.target_size = target_size
        self.skip_empty  = skip_empty
        self.augment     = augment

        # Separate LRU caches for image volumes and mask volumes
        self._img_cache:  dict = {}      # path  -> np.ndarray (normalised)
        self._mask_cache: dict = {}      # path  -> np.ndarray (binary float32)
        self._cache_size = cache_size

        # Flat index: list of (case_idx, slice_idx, subject_name)
        self._index: List[Tuple[int, int, str]] = []
        self._build_index()

    # ── Index building (fast — loads only masks) ──────────────────────────

    def _build_index(self):
        for case_idx, (dwi_path, adc_path, mask_path) in enumerate(self.cases):
            subject  = Path(dwi_path).parts[-4]   # sub-strokecaseXXXX
            mask_vol = self._load_mask(mask_path)  # [H, W, D]
            n_slices = mask_vol.shape[2]

            for s in range(n_slices):
                if self.skip_empty and mask_vol[:, :, s].sum() == 0:
                    continue
                self._index.append((case_idx, s, subject))

    # ── Caching loaders ───────────────────────────────────────────────────

    def _evict(self, cache: dict):
        """Evict the oldest entry when cache is full."""
        if len(cache) >= self._cache_size:
            cache.pop(next(iter(cache)))

    def _load_img(self, path: str) -> np.ndarray:
        """Load + normalise an image volume; cache the result."""
        if path not in self._img_cache:
            self._evict(self._img_cache)
            vol = nib.load(path).get_fdata().astype(np.float32)
            if vol.ndim == 4:       # DWI may have b-value dimension
                vol = vol[..., -1]
            self._img_cache[path] = percentile_normalise(vol)
        return self._img_cache[path]

    def _load_mask(self, path: str) -> np.ndarray:
        """Load + binarise a mask volume; cache the result."""
        if path not in self._mask_cache:
            self._evict(self._mask_cache)
            vol = nib.load(path).get_fdata().astype(np.float32)
            self._mask_cache[path] = (vol > 0).astype(np.float32)
        return self._mask_cache[path]

    # ── Slice → tensor helpers ────────────────────────────────────────────

    def _resize(self, arr: np.ndarray, mode: str = "bilinear") -> torch.Tensor:
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        t = F.interpolate(t, size=self.target_size, mode=mode,
                          align_corners=False if mode == "bilinear" else None)
        return t.squeeze(0)   # [1,H,W]

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
            mask  = torch.flip(mask,  dims=[-1])
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])
            mask  = torch.flip(mask,  dims=[-2])
        return image, mask

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        case_idx, slice_idx, subject = self._index[idx]
        dwi_path, adc_path, mask_path = self.cases[case_idx]

        # Load volumes (cached after first access)
        dwi_vol  = self._load_img(dwi_path)
        adc_vol  = self._load_img(adc_path)
        mask_vol = self._load_mask(mask_path)

        # Extract 2D axial slices
        dwi_t  = self._resize(dwi_vol[:, :, slice_idx],  mode="bilinear")  # [1,H,W]
        adc_t  = self._resize(adc_vol[:, :, slice_idx],  mode="bilinear")  # [1,H,W]
        mask_t = self._resize(mask_vol[:, :, slice_idx], mode="nearest")   # [1,H,W]

        image = torch.cat([dwi_t, adc_t], dim=0)   # [2,H,W]

        if self.augment:
            image, mask_t = self._augment(image, mask_t)

        return image, mask_t, {"subject": subject, "slice": slice_idx}


# ── Convenience builder ───────────────────────────────────────────────────────

def build_loaders(
    data_root: str,
    target_size: Tuple[int, int] = (128, 128),
    val_fraction: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    skip_empty_train: bool = False,
    cache_size: int = 20,
) -> Tuple[DataLoader, DataLoader]:
    """
    Discover cases, split by patient, return (train_loader, val_loader).
    """
    cases = find_isles_cases(data_root)
    train_cases, val_cases = train_val_split(cases, val_fraction, seed)

    print(f"[INFO] Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

    train_ds = ISLESDataset(train_cases, target_size,
                            skip_empty=skip_empty_train, augment=True, cache_size=cache_size)
    val_ds   = ISLESDataset(val_cases,   target_size,
                            skip_empty=False,            augment=False, cache_size=cache_size)

    print(f"[INFO] Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader

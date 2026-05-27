"""
dataset.py — WESAD data loading, preprocessing, and windowing.

WESAD sensor rates used here
-----------------------------
  Wrist EDA  :  4 Hz   ← keep as-is
  Wrist BVP  : 64 Hz   ← downsample to 4 Hz  (÷16)
  Labels     :700 Hz   ← nearest-neighbour resample to 4 Hz

Label mapping (binary task)
----------------------------
  WESAD code 1  →  class 0  (baseline)
  WESAD code 2  →  class 1  (stress)
  All other codes (0=undef, 3=amusement, 4=meditation) are discarded.

Sliding window
--------------
  Window length : 60 s × 4 Hz = 240 samples
  Training stride:  10 s × 4 Hz =  40 samples  (overlapping, more training data)
  Inference stride:  1 s × 4 Hz =   4 samples  (nearly continuous in live system)

Label purity filter
-------------------
  A window is accepted only if ≥80% of its label samples belong to a single
  valid class.  Windows that straddle a label boundary are discarded.

Normalisation
-------------
  Per-subject z-score before windowing.  Physiological signals like EDA have
  large inter-subject amplitude variation that would dominate the features;
  subject-level normalisation removes the DC offset and scale, leaving the
  task-relevant temporal dynamics.

  For inference, the live system performs session-level normalisation using
  the first 60 s of the session as a baseline (see inference_service/).
  Global fallback statistics are also saved in the model checkpoint by
  train.py in case a session baseline is unavailable.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WRIST_EDA_HZ   = 4
WRIST_BVP_HZ   = 64
LABEL_HZ       = 700
TARGET_HZ      = 4                    # common rate after resampling

WINDOW_SEC     = 60
STRIDE_TRAIN_SEC = 10
STRIDE_INFER_SEC =  1

WINDOW_SAMPLES    = WINDOW_SEC * TARGET_HZ          # 240
STRIDE_TRAIN      = STRIDE_TRAIN_SEC * TARGET_HZ    #  40
STRIDE_INFER      = STRIDE_INFER_SEC * TARGET_HZ    #   4

LABEL_MAP         = {1: 0, 2: 1}     # baseline → 0, stress → 1
MIN_LABEL_PURITY  = 0.80             # fraction of window that must be one class


# ---------------------------------------------------------------------------
# Subject-level loading
# ---------------------------------------------------------------------------

def _load_subject(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one WESAD subject .pkl file.

    Returns
    -------
    eda    : (N,) float32  at TARGET_HZ
    bvp    : (N,) float32  at TARGET_HZ (downsampled from 64 Hz)
    labels : (N,) int32    at TARGET_HZ (nearest-neighbour from 700 Hz)
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    eda = raw["signal"]["wrist"]["EDA"].squeeze().astype(np.float32)   # 4 Hz
    bvp = raw["signal"]["wrist"]["BVP"].squeeze().astype(np.float32)   # 64 Hz
    lbl = raw["label"].squeeze().astype(np.int32)                       # 700 Hz

    # --- Downsample BVP 64 Hz → 4 Hz ----------------------------------------
    downsample_factor = WRIST_BVP_HZ // TARGET_HZ  # = 16
    bvp_ds = resample_poly(bvp, up=1, down=downsample_factor).astype(np.float32)

    # --- Align EDA and BVP lengths -------------------------------------------
    # Floating-point rounding in resample_poly can produce ±1 sample difference
    n = min(len(eda), len(bvp_ds))
    eda    = eda[:n]
    bvp_ds = bvp_ds[:n]

    # --- Resample labels 700 Hz → 4 Hz (nearest-neighbour) ------------------
    # Never interpolate categorical labels — map each target index to the
    # nearest source index directly.
    src_indices = (np.arange(n) * (LABEL_HZ / TARGET_HZ)).astype(int)
    src_indices = np.clip(src_indices, 0, len(lbl) - 1)
    lbl_ds = lbl[src_indices].astype(np.int32)

    return eda, bvp_ds, lbl_ds


def _zscore_norm(
    arr: np.ndarray,
    mean: float = None,
    std: float = None,
) -> tuple[np.ndarray, float, float]:
    """Z-score normalise an array, optionally using pre-computed statistics."""
    m = float(arr.mean()) if mean is None else mean
    s = float(arr.std())  if std  is None else std
    s = max(s, 1e-8)      # guard against flat signal
    return (arr - m) / s, m, s


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_windows(
    wesad_root: str,
    stride: int = STRIDE_TRAIN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Walk through all subject directories, load signals, apply per-subject
    z-score, extract overlapping windows, and filter by label purity.

    Parameters
    ----------
    wesad_root : str
        Path to the WESAD/ directory that contains S2/, S3/, ... subdirectories.
    stride : int
        Stride in samples between consecutive windows.

    Returns
    -------
    X            : (N, 2, WINDOW_SAMPLES) float32  — [EDA, BVP] windows
    y            : (N,) int64              — class labels (0=baseline, 1=stress)
    global_stats : dict                    — {"mean": (2,), "std": (2,)} arrays
                   Per-channel statistics over the entire training corpus;
                   saved in the model checkpoint for inference fallback.
    """
    root = Path(wesad_root)
    subject_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("S")
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories (S2/, S3/, ...) found under '{wesad_root}'"
        )

    all_X: list[np.ndarray] = []
    all_y: list[int]        = []
    all_eda_norm: list[np.ndarray] = []   # for computing global stats later
    all_bvp_norm: list[np.ndarray] = []

    for subj_dir in subject_dirs:
        pkl_path = subj_dir / f"{subj_dir.name}.pkl"
        if not pkl_path.exists():
            print(f"  [skip] {pkl_path} not found")
            continue

        print(f"  Loading {subj_dir.name} ...", end=" ")
        eda, bvp, lbl = _load_subject(pkl_path)

        # Per-subject normalisation
        eda_n, _, _ = _zscore_norm(eda)
        bvp_n, _, _ = _zscore_norm(bvp)
        all_eda_norm.append(eda_n)
        all_bvp_norm.append(bvp_n)

        subj_windows = 0
        for start in range(0, len(eda_n) - WINDOW_SAMPLES + 1, stride):
            end     = start + WINDOW_SAMPLES
            w_lbl   = lbl[start:end]

            # Accept window only if ≥80% of samples belong to one valid class
            accepted = False
            for orig_lbl, cls_idx in LABEL_MAP.items():
                if np.mean(w_lbl == orig_lbl) >= MIN_LABEL_PURITY:
                    window = np.stack(
                        [eda_n[start:end], bvp_n[start:end]], axis=0
                    )  # shape (2, WINDOW_SAMPLES)
                    all_X.append(window)
                    all_y.append(cls_idx)
                    subj_windows += 1
                    accepted = True
                    break

        print(f"{subj_windows} windows")

    if not all_X:
        raise RuntimeError("No valid windows were extracted. Check WESAD path and label codes.")

    X = np.array(all_X, dtype=np.float32)   # (N, 2, WINDOW_SAMPLES)
    y = np.array(all_y, dtype=np.int64)

    # Compute global per-channel statistics across all normalised samples
    # shape (2,) each — used as fallback normalisation stats at inference
    all_eda_cat = np.concatenate(all_eda_norm)
    all_bvp_cat = np.concatenate(all_bvp_norm)
    global_stats = {
        "mean": np.array([all_eda_cat.mean(), all_bvp_cat.mean()], dtype=np.float32),
        "std":  np.array([all_eda_cat.std(),  all_bvp_cat.std()],  dtype=np.float32),
    }

    print(f"\nTotal windows : {len(y)}")
    print(f"  Baseline (0): {int((y == 0).sum())}")
    print(f"  Stress   (1): {int((y == 1).sum())}")
    print(f"  Shape X      : {X.shape}")

    return X, y, global_stats


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class WESADDataset(Dataset):
    """
    Minimal wrapper around pre-built (X, y) arrays for use with DataLoader.

    Parameters
    ----------
    X : (N, 2, T) float32 numpy array
    y : (N,) int64 numpy array
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
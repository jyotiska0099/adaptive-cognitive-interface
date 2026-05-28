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
  Training stride:  10 s × 4 Hz =  40 samples
  Inference stride:  1 s × 4 Hz =   4 samples

Label purity filter
-------------------
  A window is accepted only if ≥80% of its label samples belong to a single
  valid class.  Windows that straddle a label boundary are discarded.

Normalisation
-------------
  Per-subject z-score before windowing.  EDA amplitude varies greatly between
  subjects; subject-level normalisation removes inter-subject scale differences,
  leaving the task-relevant temporal dynamics intact.

  For inference, the live system performs session-level normalisation using
  the first 60 s of the session as a baseline (see inference_service/).

Subject tracking
----------------
  build_windows() now returns a subject_ids array alongside X and y.
  Each entry is the integer subject index (0-indexed) corresponding to
  the subject directory sort order.  This is required for LOSO evaluation
  in train.py.
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

WRIST_EDA_HZ     = 4
WRIST_BVP_HZ     = 64
LABEL_HZ         = 700
TARGET_HZ        = 4

WINDOW_SEC       = 60
STRIDE_TRAIN_SEC = 10
STRIDE_INFER_SEC =  1

WINDOW_SAMPLES   = WINDOW_SEC * TARGET_HZ           # 240
STRIDE_TRAIN     = STRIDE_TRAIN_SEC * TARGET_HZ     #  40
STRIDE_INFER     = STRIDE_INFER_SEC * TARGET_HZ     #   4

LABEL_MAP        = {1: 0, 2: 1}    # baseline → 0, stress → 1
MIN_LABEL_PURITY = 0.80


# ---------------------------------------------------------------------------
# Subject-level loading
# ---------------------------------------------------------------------------

def _load_subject(pkl_path: Path) -> tuple:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    eda = raw["signal"]["wrist"]["EDA"].squeeze().astype(np.float32)
    bvp = raw["signal"]["wrist"]["BVP"].squeeze().astype(np.float32)
    lbl = raw["label"].squeeze().astype(np.int32)

    bvp_ds = resample_poly(bvp, up=1, down=WRIST_BVP_HZ // TARGET_HZ).astype(np.float32)

    n = min(len(eda), len(bvp_ds))
    eda    = eda[:n]
    bvp_ds = bvp_ds[:n]

    src_idx = (np.arange(n) * (LABEL_HZ / TARGET_HZ)).astype(int)
    src_idx = np.clip(src_idx, 0, len(lbl) - 1)
    lbl_ds  = lbl[src_idx].astype(np.int32)

    return eda, bvp_ds, lbl_ds


def _zscore(arr: np.ndarray) -> np.ndarray:
    m = arr.mean()
    s = max(arr.std(), 1e-8)
    return (arr - m) / s


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_windows(
    wesad_root: str,
    stride: int = STRIDE_TRAIN,
) -> tuple:
    """
    Walk through all subject directories, load signals, apply per-subject
    z-score, extract overlapping windows, and filter by label purity.

    Parameters
    ----------
    wesad_root : str
        Path to the WESAD/ directory containing S2/, S3/, ... subdirectories.
    stride : int
        Stride in samples between consecutive windows.

    Returns
    -------
    X            : (N, 2, WINDOW_SAMPLES) float32
    y            : (N,) int64
    subject_ids  : (N,) int64  — subject index (0-based) for each window
                   Required for LOSO cross-validation in train.py.
    global_stats : dict {"mean": (2,), "std": (2,)}
                   Per-channel statistics saved in the model checkpoint
                   as a normalisation fallback for the inference service.
    """
    root = Path(wesad_root)
    subject_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("S")
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories found under '{wesad_root}'"
        )

    all_X:    list[np.ndarray] = []
    all_y:    list[int]        = []
    all_sids: list[int]        = []   # ← NEW: subject index per window
    all_eda_norm: list[np.ndarray] = []
    all_bvp_norm: list[np.ndarray] = []

    for subj_idx, subj_dir in enumerate(subject_dirs):
        pkl_path = subj_dir / f"{subj_dir.name}.pkl"
        if not pkl_path.exists():
            print(f"  [skip] {pkl_path} not found")
            continue

        print(f"  Loading {subj_dir.name} ...", end=" ")
        eda, bvp, lbl = _load_subject(pkl_path)

        eda_n = _zscore(eda)
        bvp_n = _zscore(bvp)
        all_eda_norm.append(eda_n)
        all_bvp_norm.append(bvp_n)

        subj_windows = 0
        for start in range(0, len(eda_n) - WINDOW_SAMPLES + 1, stride):
            end   = start + WINDOW_SAMPLES
            w_lbl = lbl[start:end]

            for orig_lbl, cls_idx in LABEL_MAP.items():
                if np.mean(w_lbl == orig_lbl) >= MIN_LABEL_PURITY:
                    window = np.stack(
                        [eda_n[start:end], bvp_n[start:end]], axis=0
                    )
                    all_X.append(window)
                    all_y.append(cls_idx)
                    all_sids.append(subj_idx)   # ← tag with subject index
                    subj_windows += 1
                    break

        print(f"{subj_windows} windows")

    if not all_X:
        raise RuntimeError("No valid windows extracted. Check WESAD path and label codes.")

    X           = np.array(all_X,    dtype=np.float32)
    y           = np.array(all_y,    dtype=np.int64)
    subject_ids = np.array(all_sids, dtype=np.int64)   # ← NEW

    all_eda_cat = np.concatenate(all_eda_norm)
    all_bvp_cat = np.concatenate(all_bvp_norm)
    global_stats = {
        "mean": np.array([all_eda_cat.mean(), all_bvp_cat.mean()], dtype=np.float32),
        "std":  np.array([all_eda_cat.std(),  all_bvp_cat.std()],  dtype=np.float32),
    }

    print(f"\nTotal windows : {len(y)}")
    print(f"  Baseline (0): {int((y == 0).sum())}")
    print(f"  Stress   (1): {int((y == 1).sum())}")
    print(f"  Subjects     : {len(np.unique(subject_ids))}")
    print(f"  Shape X      : {X.shape}")

    return X, y, subject_ids, global_stats   # ← returns 4 values now


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class WESADDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
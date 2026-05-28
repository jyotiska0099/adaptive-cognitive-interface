"""
train.py — Train the TCN cognitive-load classifier on WESAD with LOSO evaluation.

Usage
-----
  python train.py --wesad_root /path/to/WESAD

  # Optional overrides:
  python train.py --wesad_root /path/to/WESAD --epochs 80 --lr 5e-4

Evaluation strategy
-------------------
  Leave-One-Subject-Out (LOSO) cross-validation:
    For each of the N subjects, train on the remaining N-1 subjects and
    evaluate on the held-out one.  This is the correct evaluation for
    physiological signal models because EDA and BVP are highly person-specific.
    A random window-level split leaks subject identity into the test set,
    making the reported numbers optimistic.

  After LOSO, a final model is trained on ALL subjects and saved as the
  deployment checkpoint.  LOSO gives the honest generalisation estimate;
  the final model benefits from the full dataset.

  Within each fold, 15% of the training windows are held out as a validation
  set for early stopping.  This validation set is sampled from the training
  subjects only — never from the held-out subject.

Output
------
  tcn_cognitive_load.pth  — final model trained on all subjects

  Checkpoint contents:
    model_state_dict : weights, loadable into TCN(**model_config)
    model_config     : dict of TCN constructor kwargs
    global_stats     : {"mean": (2,), "std": (2,)} normalisation fallback
    loso_summary     : {"mean_f1": float, "std_f1": float, "per_fold": list}
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import WESADDataset, build_windows
from tcn_arch import TCN


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    epochs       = 60,
    batch_size   = 64,
    lr           = 1e-3,
    weight_decay = 1e-4,
    dropout      = 0.2,
    patience     = 10,
    seed         = 42,
    out          = "tcn_cognitive_load.pth",
)

MODEL_CFG = dict(
    input_channels = 2,
    num_classes    = 2,
    channel_sizes  = [32, 64, 64, 128],
    kernel_size    = 3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_criterion(y_train: np.ndarray, device: torch.device) -> nn.CrossEntropyLoss:
    counts  = np.bincount(y_train)
    weights = 1.0 / counts.astype(float)
    weights /= weights.sum()
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(
        WESADDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Single epoch train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds, labels = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total += criterion(logits, y).item() * len(y)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    loss = total / len(loader.dataset)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="binary", pos_label=1)
    return loss, acc, f1


# ---------------------------------------------------------------------------
# Core training loop (reused by both LOSO folds and the final model)
# ---------------------------------------------------------------------------

def train_model(
    X_tr, y_tr,
    X_vl, y_vl,
    args,
    device,
    verbose: bool = False,
) -> tuple:
    """
    Train a fresh TCN on (X_tr, y_tr), validate on (X_vl, y_vl).
    Returns (best_model_state_dict, best_val_f1).
    """
    model_cfg = {**MODEL_CFG, "dropout": args.dropout}
    model     = TCN(**model_cfg).to(device)
    criterion = make_criterion(y_tr, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    vl_loader = make_loader(X_vl, y_vl, args.batch_size, shuffle=False)

    best_f1, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss             = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1 = evaluate(model, vl_loader, criterion, device)
        scheduler.step(vl_f1)

        if verbose:
            marker = " ✓" if vl_f1 > best_f1 else ""
            print(
                f"  epoch {epoch:3d} | "
                f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                f"acc={vl_acc:.3f}  f1={vl_f1:.3f}{marker}"
            )

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    return best_state, best_f1


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(X, y, subject_ids, args, device) -> list:
    """
    Leave-One-Subject-Out cross-validation.

    For each unique subject:
      - Hold out all windows belonging to that subject as the test set.
      - Train on the remaining subjects (with an internal 85/15 val split
        for early stopping).
      - Record acc and F1 on the held-out subject.

    Returns a list of per-fold dicts.
    """
    unique_subjects = np.unique(subject_ids)
    n_folds         = len(unique_subjects)
    fold_results    = []

    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation  ({n_folds} folds)")
    print(f"{'='*60}")

    for fold_i, held_out in enumerate(unique_subjects):
        tr_pool_idx = np.where(subject_ids != held_out)[0]
        ts_idx      = np.where(subject_ids == held_out)[0]

        # Internal train/val split (85/15) within the training subjects
        tr_idx, vl_idx = train_test_split(
            tr_pool_idx,
            test_size=0.15,
            stratify=y[tr_pool_idx],
            random_state=args.seed,
        )

        print(
            f"\nFold {fold_i+1:2d}/{n_folds} | "
            f"held-out subject idx={held_out} | "
            f"train={len(tr_idx)}  val={len(vl_idx)}  test={len(ts_idx)}"
        )

        best_state, best_vl_f1 = train_model(
            X[tr_idx], y[tr_idx],
            X[vl_idx], y[vl_idx],
            args, device,
            verbose=False,   # keep LOSO output compact
        )

        # Evaluate on held-out subject
        model_cfg = {**MODEL_CFG, "dropout": args.dropout}
        model     = TCN(**model_cfg).to(device)
        model.load_state_dict(best_state)
        criterion = make_criterion(y[tr_idx], device)
        ts_loader = make_loader(X[ts_idx], y[ts_idx], args.batch_size, shuffle=False)
        _, ts_acc, ts_f1 = evaluate(model, ts_loader, criterion, device)

        print(
            f"         result     | val_f1={best_vl_f1:.3f}  "
            f"test_acc={ts_acc:.3f}  test_f1={ts_f1:.3f}"
        )
        fold_results.append({
            "subject_idx": int(held_out),
            "test_acc":    round(ts_acc, 4),
            "test_f1":     round(ts_f1,  4),
            "val_f1":      round(best_vl_f1, 4),
        })

    # Summary
    f1_scores = [r["test_f1"] for r in fold_results]
    ac_scores = [r["test_acc"] for r in fold_results]
    mean_f1   = float(np.mean(f1_scores))
    std_f1    = float(np.std(f1_scores))
    mean_acc  = float(np.mean(ac_scores))

    print(f"\n{'='*60}")
    print(f"LOSO Summary")
    print(f"  mean F1  : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  mean acc : {mean_acc:.3f}")
    print(f"  worst F1 : {min(f1_scores):.3f}  (subject idx={f1_scores.index(min(f1_scores))})")
    print(f"  best  F1 : {max(f1_scores):.3f}  (subject idx={f1_scores.index(max(f1_scores))})")
    print(f"{'='*60}\n")

    return fold_results


# ---------------------------------------------------------------------------
# Final model — trained on ALL subjects
# ---------------------------------------------------------------------------

def train_final_model(X, y, global_stats, fold_results, args, device) -> None:
    """
    Train the deployment model on all subjects.
    Uses a small internal val split (10%) for early stopping only.
    Saves to args.out.
    """
    print("Training final model on all subjects ...")

    tr_idx, vl_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.10,
        stratify=y,
        random_state=args.seed,
    )

    best_state, best_vl_f1 = train_model(
        X[tr_idx], y[tr_idx],
        X[vl_idx], y[vl_idx],
        args, device,
        verbose=True,
    )

    loso_summary = {
        "mean_f1":  round(float(np.mean([r["test_f1"]  for r in fold_results])), 4),
        "std_f1":   round(float(np.std( [r["test_f1"]  for r in fold_results])), 4),
        "mean_acc": round(float(np.mean([r["test_acc"] for r in fold_results])), 4),
        "per_fold": fold_results,
    }

    model_cfg = {**MODEL_CFG, "dropout": args.dropout}
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config":     model_cfg,
            "global_stats":     global_stats,
            "loso_summary":     loso_summary,
        },
        args.out,
    )

    print(f"\nCheckpoint saved → {Path(args.out).resolve()}")
    print(f"  val_f1 (final model) : {best_vl_f1:.3f}")
    print(f"  LOSO mean_f1         : {loso_summary['mean_f1']:.3f} ± {loso_summary['std_f1']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Device     : {device}")
    print(f"WESAD root : {args.wesad_root}")
    print(f"Output     : {args.out}")
    print(f"{'='*60}")

    print("\nBuilding windows from WESAD dataset ...")
    X, y, subject_ids, global_stats = build_windows(args.wesad_root)

    # 1. LOSO: honest generalisation estimate
    fold_results = run_loso(X, y, subject_ids, args, device)

    # 2. Final model: trained on all subjects for deployment
    train_final_model(X, y, global_stats, fold_results, args, device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TCN on WESAD with LOSO cross-validation."
    )
    parser.add_argument(
        "--wesad_root", required=True,
        help="Path to WESAD/ directory (must contain S2/, S3/, ... subdirs)."
    )
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout",      type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--patience",     type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--out",          type=str,   default=DEFAULTS["out"])
    main(parser.parse_args())
"""
train.py — Train the TCN cognitive-load classifier on WESAD.

Usage
-----
  # From the model/ directory, with your virtual env active:
  python train.py --wesad_root /path/to/WESAD

  # Optional overrides:
  python train.py --wesad_root /path/to/WESAD --epochs 80 --lr 5e-4 --batch_size 32

Output
------
  tcn_cognitive_load.pth  — best checkpoint (saved whenever val F1 improves)

  Checkpoint contents:
    model_state_dict  : weights, loadable into TCN(**model_config)
    model_config      : dict of TCN constructor kwargs (no need to hard-code)
    global_stats      : {"mean": (2,), "std": (2,)} — per-channel normalisation
                        stats computed over the full corpus.
                        Used at inference when no session baseline is available.

Normalisation note
------------------
  Windows are per-subject z-scored in dataset.py before they reach this script.
  'global_stats' are statistics OF those already-normalised windows — they
  describe the distribution of the normalised data and serve as a consistent
  reference frame for the inference service.

Split strategy
--------------
  Stratified random split (75/15/10 train/val/test) at the window level.
  A true leave-one-subject-out (LOSO) split would give a better estimate of
  generalisation to new users, but requires more subjects than the demo setup.
  The split here is sufficient to validate convergence and avoid overfitting.
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
# Hyper-parameters (override via CLI args)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    epochs      = 60,
    batch_size  = 64,
    lr          = 1e-3,
    weight_decay= 1e-4,
    dropout     = 0.2,
    patience    = 10,           # early-stopping patience (epochs)
    seed        = 42,
    out         = "tcn_cognitive_load.pth",
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


def class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights to handle baseline/stress imbalance."""
    counts = np.bincount(y_train)
    weights = 1.0 / counts.astype(float)
    weights /= weights.sum()          # normalise so they sum to 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(y)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    return loss, acc, f1


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
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load and window the dataset
    # ------------------------------------------------------------------
    print("Building windows from WESAD dataset...")
    X, y, global_stats = build_windows(args.wesad_root)

    # ------------------------------------------------------------------
    # 2. Stratified split (window level)
    # ------------------------------------------------------------------
    idx = np.arange(len(y))

    # 75% train, 15% val, 10% test
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.25, stratify=y, random_state=args.seed
    )
    vl_idx, ts_idx = train_test_split(
        tmp_idx, test_size=0.40, stratify=y[tmp_idx], random_state=args.seed
    )

    print(f"\nSplit  →  train: {len(tr_idx)}  val: {len(vl_idx)}  test: {len(ts_idx)}")

    tr_loader = DataLoader(
        WESADDataset(X[tr_idx], y[tr_idx]),
        batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    vl_loader = DataLoader(
        WESADDataset(X[vl_idx], y[vl_idx]),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    ts_loader = DataLoader(
        WESADDataset(X[ts_idx], y[ts_idx]),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # 3. Model, loss, optimiser
    # ------------------------------------------------------------------
    model_cfg = dict(
        input_channels = 2,
        num_classes    = 2,
        channel_sizes  = [32, 64, 64, 128],
        kernel_size    = 3,
        dropout        = args.dropout,
    )
    model     = TCN(**model_cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y[tr_idx], device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {param_count:,}\n")

    # ------------------------------------------------------------------
    # 4. Training loop with early stopping on val F1
    # ------------------------------------------------------------------
    best_val_f1   = 0.0
    patience_ctr  = 0
    header_printed = False

    for epoch in range(1, args.epochs + 1):
        tr_loss             = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1 = evaluate(model, vl_loader, criterion, device)
        scheduler.step(vl_f1)

        if not header_printed:
            print(f"{'Epoch':>6}  {'tr_loss':>9}  {'vl_loss':>9}  {'vl_acc':>7}  {'vl_f1':>7}")
            print("-" * 52)
            header_printed = True

        marker = " ✓" if vl_f1 > best_val_f1 else ""
        print(
            f"{epoch:6d}  {tr_loss:9.4f}  {vl_loss:9.4f}  "
            f"{vl_acc:7.3f}  {vl_f1:7.3f}{marker}"
        )

        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict" : model.state_dict(),
                    "model_config"     : model_cfg,
                    "global_stats"     : global_stats,   # for inference fallback
                },
                args.out,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val F1 improvement for {args.patience} epochs).")
                break

    # ------------------------------------------------------------------
    # 5. Final test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint from '{args.out}' ...")
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ts_loss, ts_acc, ts_f1 = evaluate(model, ts_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"Test  →  loss={ts_loss:.4f}  acc={ts_acc:.3f}  f1={ts_f1:.3f}")
    print(f"{'='*60}\n")

    # Detailed per-class report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y_batch in ts_loader:
            all_preds.extend(model(X.to(device)).argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    print(classification_report(all_labels, all_preds,
                                 target_names=["baseline", "stress"]))
    print(f"Checkpoint saved to  : {Path(args.out).resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TCN cognitive-load classifier on WESAD dataset."
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
    parser.add_argument("--out",          type=str,   default=DEFAULTS["out"],
                        help="Output path for the model checkpoint.")
    main(parser.parse_args())
"""
neural/train_hapticnet.py — Train HapticNet from recorded sensor data.

Loads:
    data/tactile_recordings.csv  — material-labeled sensor data
    data/slip_recordings.csv     — slip-labeled sensor data

Creates sliding windows of 20 frames, computes target outputs:
    material class → target freq/wave from preset table
    pressure magnitude → target duty (Weber-Fechner normalized)
    slip label → target slip_prob

Trains with AdamW + MSE, 300 epochs.
Saves to models/hapticnet.pt.

Usage:
    python neural/train_hapticnet.py                     # uses recorded data
    python neural/train_hapticnet.py --synthetic 5000    # synthetic data (no CSV)
    python neural/train_hapticnet.py --epochs 500        # more epochs
"""

import os
import sys
import csv
import math
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.hapticnet import (
    HapticNet, MockSensorStream, save_model,
    WINDOW_LEN, N_PRESSURE, N_TORQUE,
)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
RUNS_DIR   = os.path.join(DATA_DIR, "runs")
MODEL_PATH = os.path.join(ROOT, "models", "hapticnet.pt")

MATERIAL_CSV = os.path.join(DATA_DIR, "tactile_recordings.csv")
SLIP_CSV     = os.path.join(DATA_DIR, "slip_recordings.csv")

# Pressure normalization: recorded data is in [0,1] (divided by TACTILE_MAX=100).
# Synthetic data (MockSensorStream) is in [0,~5].
# We normalize all inputs to [0,1] for consistency.
PRESSURE_NORM_SYNTHETIC = 5.0  # divide synthetic pressure by this to get [0,1]

# ── Material class → target haptic params ────────────────────────────────────
# freq ∈ [0,7], duty ∈ [0,31], wave ∈ {0,1}
# We store as normalized [0,1] targets to match sigmoid output.

MATERIAL_TARGETS = {
    1: {"freq": 6/7, "wave": 0.0},  # smooth/hard: high freq, square
    2: {"freq": 3/7, "wave": 1.0},  # rough: mid freq, sine
    3: {"freq": 1/7, "wave": 1.0},  # soft: low freq, sine
    4: {"freq": 0.0, "wave": 0.0},  # no contact: zero
}


def weber_fechner(force: float) -> float:
    """Perceptually linear force → duty mapping, returns [0,1]."""
    return math.log(1.0 + 9.0 * max(0.0, min(1.0, force))) / math.log(10.0)


# ── Dataset: from CSV ────────────────────────────────────────────────────────

def load_csv(path: str):
    """Load recording CSV → (timestamps, pressure, torque, labels)."""
    timestamps, pressure, torque, labels = [], [], [], []

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            timestamps.append(float(row[0]))
            p = [float(row[1 + i]) for i in range(N_PRESSURE)]
            t = [float(row[1 + N_PRESSURE + i]) for i in range(N_TORQUE)]
            label = int(row[-1])
            pressure.append(p)
            torque.append(t)
            labels.append(label)

    return (
        np.array(timestamps, dtype=np.float64),
        np.array(pressure,   dtype=np.float32),
        np.array(torque,     dtype=np.float32),
        np.array(labels,     dtype=np.int64),
    )


class RecordedDataset(Dataset):
    """
    Sliding-window dataset from recorded CSVs.

    Each sample: sequence of SEQ_LEN consecutive windows.
    Each window: 20 frames of pressure + torque.
    Target: per-window haptic params.
    """

    def __init__(self, material_csv: str = None, slip_csv: str = None,
                 seq_len: int = 10, fragility: float = 0.5):
        self.seq_len = seq_len
        self.fragility = fragility
        self.windows = []   # list of (pressure_win, torque_win, target)

        # load legacy single files
        if material_csv and os.path.exists(material_csv):
            self._load_material(material_csv)

        if slip_csv and os.path.exists(slip_csv):
            self._load_slip(slip_csv)

        # also load individual run files from data/runs/
        if os.path.isdir(RUNS_DIR):
            import glob
            for f in sorted(glob.glob(os.path.join(RUNS_DIR, "material_*.csv"))):
                self._load_material(f)
            for f in sorted(glob.glob(os.path.join(RUNS_DIR, "slip_*.csv"))):
                self._load_slip(f)

        if not self.windows:
            raise FileNotFoundError(
                f"No data found. Expected:\n  {material_csv}\n  {slip_csv}\n"
                f"  or CSVs in {RUNS_DIR}/\n"
                "Run record_data.py first, or use --synthetic."
            )

        print(f"[Dataset] {len(self.windows)} windows, "
              f"{len(self)} sequences of {seq_len}")

    def _load_material(self, path: str):
        print(f"[Dataset] Loading material data: {path}")
        _, pressure, torque, labels = load_csv(path)
        N = len(pressure)

        for i in range(WINDOW_LEN, N):
            p_win = pressure[i - WINDOW_LEN: i]   # (20, 6)
            t_win = torque[i - WINDOW_LEN: i]     # (20, 12)
            label = int(labels[i])

            # compute targets
            mt = MATERIAL_TARGETS.get(label, MATERIAL_TARGETS[4])
            # recorded pressure is already in [0,1] (normalized by record_data.py)
            p_mag = float(np.mean(np.abs(p_win[-1])))
            p_norm = min(1.0, p_mag)

            target = np.array([
                mt["freq"],                 # freq_raw
                weber_fechner(p_norm),      # duty_raw
                mt["wave"],                 # wave_raw
                0.0,                        # slip_prob (from material session = 0)
            ], dtype=np.float32)

            self.windows.append((p_win, t_win, target))

        counts = {}
        for l in labels:
            counts[int(l)] = counts.get(int(l), 0) + 1
        print(f"  Loaded {N} frames, label distribution: {counts}")

    def _load_slip(self, path: str):
        print(f"[Dataset] Loading slip data: {path}")
        _, pressure, torque, labels = load_csv(path)
        N = len(pressure)

        n_slip = 0
        for i in range(WINDOW_LEN, N):
            p_win = pressure[i - WINDOW_LEN: i]
            t_win = torque[i - WINDOW_LEN: i]
            is_slip = float(labels[i])

            p_mag = float(np.mean(np.abs(p_win[-1])))
            p_norm = min(1.0, p_mag)  # already [0,1] from recording

            target = np.array([
                3/7,                       # neutral freq
                weber_fechner(p_norm),     # duty from pressure
                1.0,                       # sine wave
                float(is_slip),            # slip label
            ], dtype=np.float32)

            self.windows.append((p_win, t_win, target))
            if is_slip:
                n_slip += 1

        print(f"  Loaded {N} frames, {n_slip} slip events ({100*n_slip/max(1,N):.1f}%)")

    def __len__(self):
        return max(0, len(self.windows) - self.seq_len)

    def __getitem__(self, idx):
        # extract a sequence of consecutive windows
        seq_p, seq_t, seq_target = [], [], []
        for i in range(self.seq_len):
            p, t, tgt = self.windows[idx + i]
            seq_p.append(p)
            seq_t.append(t)
            seq_target.append(tgt)

        return (
            np.array(seq_p, dtype=np.float32),       # (S, 20, 6)
            np.array(seq_t, dtype=np.float32),       # (S, 20, 12)
            np.array([self.fragility], dtype=np.float32),  # (1,)
            np.array(seq_target, dtype=np.float32),  # (S, 4)
        )


# ── Dataset: synthetic (no CSV needed) ──────────────────────────────────────

class SyntheticDataset(Dataset):
    """
    Generates training data from MockSensorStream.
    No CSV files needed — useful for initial testing.
    """

    def __init__(self, n_samples: int = 5000, seq_len: int = 10):
        self.seq_len = seq_len
        self.data = []

        mock = MockSensorStream(hz=50.0)
        print(f"[SyntheticDataset] Generating {n_samples} sequences...")

        # generate a long stream, then slice into sequences
        all_windows = []
        buf_p = np.zeros((WINDOW_LEN, N_PRESSURE), dtype=np.float32)
        buf_t = np.zeros((WINDOW_LEN, N_TORQUE), dtype=np.float32)

        # we need n_samples * seq_len + WINDOW_LEN frames total
        total_frames = n_samples * seq_len + WINDOW_LEN + 100
        for frame_i in range(total_frames):
            d = mock.step()
            buf_p = np.roll(buf_p, -1, axis=0)
            buf_t = np.roll(buf_t, -1, axis=0)
            # normalize synthetic pressure to [0,1] to match recorded data scale
            buf_p[-1] = d["pressure"] / PRESSURE_NORM_SYNTHETIC
            buf_t[-1] = d["torque"]

            if frame_i >= WINDOW_LEN:
                mat_label = mock.get_material_label()
                mt = MATERIAL_TARGETS.get(mat_label, MATERIAL_TARGETS[4])
                p_mag = float(np.mean(np.abs(d["pressure"])))
                # synthetic pressure is [0,~5], normalize to [0,1] like recorded data
                p_norm = min(1.0, p_mag / PRESSURE_NORM_SYNTHETIC)

                target = np.array([
                    mt["freq"],
                    weber_fechner(p_norm),
                    mt["wave"],
                    float(d.get("slip_label", False)),
                ], dtype=np.float32)

                all_windows.append((buf_p.copy(), buf_t.copy(), target))

        # slice into sequences
        for i in range(0, len(all_windows) - seq_len, seq_len // 2):
            seq = all_windows[i: i + seq_len]
            if len(seq) == seq_len:
                self.data.append(seq)
            if len(self.data) >= n_samples:
                break

        print(f"[SyntheticDataset] {len(self.data)} sequences ready")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        p = np.array([s[0] for s in seq], dtype=np.float32)
        t = np.array([s[1] for s in seq], dtype=np.float32)
        tgt = np.array([s[2] for s in seq], dtype=np.float32)
        frag = np.array([0.5], dtype=np.float32)
        return p, t, frag, tgt


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[train] Device: {device}")
    print(f"[train] Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # --- dataset ---
    if args.synthetic > 0:
        dataset = SyntheticDataset(n_samples=args.synthetic, seq_len=args.seq_len)
    else:
        dataset = RecordedDataset(
            material_csv=MATERIAL_CSV,
            slip_csv=SLIP_CSV,
            seq_len=args.seq_len,
        )

    # train/val split (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    print(f"[train] Train: {n_train}  Val: {n_val}")

    # --- model ---
    model = HapticNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] HapticNet: {n_params:,} parameters")

    # --- optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- loss: weighted MSE ---
    # [freq, duty, wave, slip] — weight slip higher (safety-critical)
    loss_weights = torch.tensor([1.0, 1.0, 1.0, 2.0], device=device)

    def weighted_mse(pred, target):
        diff2 = (pred - target) ** 2            # (B, T, 4)
        weighted = diff2 * loss_weights         # broadcast
        return weighted.mean()

    # --- training ---
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for pressure, torque, frag, target in train_loader:
            pressure = pressure.to(device)
            torque = torque.to(device)
            frag = frag.to(device)
            target = target.to(device)

            pred, _ = model(pressure, torque, frag)
            loss = weighted_mse(pred, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(1, n_batches)

        # validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for pressure, torque, frag, target in val_loader:
                pressure = pressure.to(device)
                torque = torque.to(device)
                frag = frag.to(device)
                target = target.to(device)

                pred, _ = model(pressure, torque, frag)
                loss = weighted_mse(pred, target)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val = val_loss / max(1, n_val_batches)

        # save best
        if avg_val < best_val:
            best_val = avg_val
            save_model(model, MODEL_PATH, epoch=epoch, loss=avg_val)

        # print progress
        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train={avg_train:.5f}  val={avg_val:.5f}  "
                  f"best={best_val:.5f}  lr={lr:.2e}  "
                  f"t={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n[train] Done in {elapsed:.1f}s")
    print(f"[train] Best val loss: {best_val:.5f}")
    print(f"[train] Model saved: {MODEL_PATH}")

    # --- validation metrics ---
    print("\n" + "=" * 50)
    print("  Validation Metrics")
    print("=" * 50)

    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for pressure, torque, frag, target in val_loader:
            pressure = pressure.to(device)
            torque = torque.to(device)
            frag = frag.to(device)
            pred, _ = model(pressure, torque, frag)
            all_pred.append(pred.cpu())
            all_tgt.append(target)

    pred = torch.cat(all_pred).numpy().reshape(-1, 4)
    tgt  = torch.cat(all_tgt).numpy().reshape(-1, 4)

    names = ["freq_raw", "duty_raw", "wave_raw", "slip_prob"]
    for i, name in enumerate(names):
        mae = float(np.mean(np.abs(pred[:, i] - tgt[:, i])))
        rmse = float(np.sqrt(np.mean((pred[:, i] - tgt[:, i]) ** 2)))
        print(f"  {name:12s}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # wave accuracy (threshold at 0.5)
    wave_pred = (pred[:, 2] >= 0.5).astype(int)
    wave_tgt  = (tgt[:, 2] >= 0.5).astype(int)
    wave_acc  = float(np.mean(wave_pred == wave_tgt))
    print(f"  {'wave_acc':12s}  {wave_acc:.4f}")

    # slip detection (threshold at 0.5)
    slip_pred = (pred[:, 3] >= 0.5).astype(int)
    slip_tgt  = (tgt[:, 3] >= 0.5).astype(int)
    n_slip_tgt = int(slip_tgt.sum())
    if n_slip_tgt > 0:
        tp = int(((slip_pred == 1) & (slip_tgt == 1)).sum())
        fp = int(((slip_pred == 1) & (slip_tgt == 0)).sum())
        fn = int(((slip_pred == 0) & (slip_tgt == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-8, prec + rec)
        print(f"  {'slip_prec':12s}  {prec:.4f}")
        print(f"  {'slip_recall':12s}  {rec:.4f}")
        print(f"  {'slip_f1':12s}  {f1:.4f}")
    else:
        print(f"  (no slip events in validation set)")

    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HapticNet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length (consecutive windows per sample)")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Generate N synthetic samples instead of loading CSV")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Train TactileNet LSTM on recorded tactile CSV data.

Reads CSVs from data/runs/ (individual run files) and/or data/tactile_recordings.csv + data/slip_recordings.csv.

Tasks:
  1. Material classification: label 1=smooth, 2=rough, 3=soft → 3 classes
  2. Slip detection: label 0=normal, 1=slip → binary
  3. Haptic rendering: (freq_raw, duty_raw, wave_raw) from material presets

Usage:
    python haptics/training/train_tactile_lstm.py
    python haptics/training/train_tactile_lstm.py --epochs 100 --seq-len 40
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "models")

N_PRESSURE = 6
N_TORQUE   = 12
N_INPUT    = N_PRESSURE + N_TORQUE  # 18

# Material label → class index (0-based for CrossEntropyLoss)
# CSV labels: 0=baseline/no-contact, 1=smooth, 2=rough, 3=soft
# We map: 1→0(smooth), 2→1(rough), 3→2(soft), 0→excluded
MATERIAL_MAP = {1: 0, 2: 1, 3: 2}
MATERIAL_NAMES = ["smooth", "rough", "soft"]
N_MATERIALS = len(MATERIAL_NAMES)

# Material → target haptic params [freq_raw, duty_raw, wave_raw] ∈ [0, 1]
# These are the "ground truth" haptic outputs the LSTM should learn
MATERIAL_HAPTIC = {
    0: [6/7, 0.7, 0.0],   # smooth → high freq, square
    1: [3/7, 0.7, 1.0],   # rough  → mid freq, sine
    2: [1/7, 0.5, 1.0],   # soft   → low freq, sine
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class TactileDataset(Dataset):
    """
    Loads tactile CSVs into windowed sequences for LSTM training.

    Each sample: (seq_len, 18) input → material class + haptic params + slip label
    """

    def __init__(self, csv_paths: list, seq_len: int = 20, task: str = "material"):
        """
        Args:
            csv_paths: list of CSV file paths
            seq_len: LSTM sequence length (frames)
            task: "material" or "slip"
        """
        self.seq_len = seq_len
        self.task = task

        frames = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                if "label" not in df.columns:
                    continue
                frames.append(df)
            except Exception as e:
                print(f"  skip {path}: {e}")

        if not frames:
            self.X = np.zeros((0, seq_len, N_INPUT), dtype=np.float32)
            self.y_material = np.zeros(0, dtype=np.int64)
            self.y_haptic = np.zeros((0, 3), dtype=np.float32)
            self.y_slip = np.zeros(0, dtype=np.float32)
            return

        df = pd.concat(frames, ignore_index=True)

        # extract features
        p_cols = [f"p{i}" for i in range(N_PRESSURE)]
        t_cols = [f"t{i}" for i in range(N_TORQUE)]
        features = df[p_cols + t_cols].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)

        # normalize torques to ~[-1, 1]
        torque_std = features[:, N_PRESSURE:].std(axis=0).clip(min=1.0)
        features[:, N_PRESSURE:] /= torque_std

        # build sequences
        X, y_mat, y_hap, y_slip = [], [], [], []

        if task == "material":
            # only use rows with material labels (1, 2, 3)
            for i in range(len(features) - seq_len):
                window_labels = labels[i:i+seq_len]
                last_label = int(window_labels[-1])
                if last_label not in MATERIAL_MAP:
                    continue
                X.append(features[i:i+seq_len])
                mat_idx = MATERIAL_MAP[last_label]
                y_mat.append(mat_idx)
                y_hap.append(MATERIAL_HAPTIC[mat_idx])
                y_slip.append(0.0)

        elif task == "slip":
            # slip CSVs: label 0=normal, 1=slip
            for i in range(len(features) - seq_len):
                X.append(features[i:i+seq_len])
                y_mat.append(0)
                y_hap.append(MATERIAL_HAPTIC[0])
                y_slip.append(float(labels[i + seq_len - 1]))

        self.X = np.array(X, dtype=np.float32) if X else np.zeros((0, seq_len, N_INPUT), dtype=np.float32)
        self.y_material = np.array(y_mat, dtype=np.int64)
        self.y_haptic = np.array(y_hap, dtype=np.float32)
        self.y_slip = np.array(y_slip, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y_material[idx]),
            torch.tensor(self.y_haptic[idx]),
            torch.tensor(self.y_slip[idx]),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class TactileLSTM(nn.Module):
    """
    Simple LSTM for tactile perception.
    Input: (B, T, 18) → pressure(6) + torque(12)
    Output: material class, haptic params, slip probability
    """

    def __init__(self, input_dim=N_INPUT, hidden=64, n_layers=2, n_materials=N_MATERIALS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.material_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, n_materials),
        )
        self.haptic_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 3), nn.Sigmoid(),
        )
        self.slip_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, T, 18) → material (B, N), haptic (B, 3), slip (B,)"""
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last timestep
        material = self.material_head(h)
        haptic = self.haptic_head(h)
        slip = self.slip_head(h).squeeze(-1)
        return material, haptic, slip


# ── Training ──────────────────────────────────────────────────────────────────

def find_csvs():
    """Find all tactile CSV files."""
    paths = {"material": [], "slip": []}

    # individual run files
    runs_dir = os.path.join(DATA_DIR, "runs")
    if os.path.isdir(runs_dir):
        for f in sorted(glob.glob(os.path.join(runs_dir, "material_*.csv"))):
            paths["material"].append(f)
        for f in sorted(glob.glob(os.path.join(runs_dir, "slip_*.csv"))):
            paths["slip"].append(f)
        for f in sorted(glob.glob(os.path.join(runs_dir, "baseline_*.csv"))):
            paths["material"].append(f)  # baseline = label 0, will be filtered

    # legacy single files
    legacy_mat = os.path.join(DATA_DIR, "tactile_recordings.csv")
    legacy_slip = os.path.join(DATA_DIR, "slip_recordings.csv")
    if os.path.exists(legacy_mat):
        paths["material"].append(legacy_mat)
    if os.path.exists(legacy_slip):
        paths["slip"].append(legacy_slip)

    return paths


def train(epochs=100, seq_len=20, batch_size=64, lr=1e-3, val_split=0.15):
    paths = find_csvs()

    print(f"[train] Material CSVs: {len(paths['material'])}")
    for p in paths["material"]:
        print(f"  {os.path.basename(p)}")
    print(f"[train] Slip CSVs: {len(paths['slip'])}")
    for p in paths["slip"]:
        print(f"  {os.path.basename(p)}")

    # build datasets
    mat_ds = TactileDataset(paths["material"], seq_len=seq_len, task="material")
    slip_ds = TactileDataset(paths["slip"], seq_len=seq_len, task="slip")

    print(f"\n[train] Material samples: {len(mat_ds)}")
    if len(mat_ds) > 0:
        counts = np.bincount(mat_ds.y_material, minlength=N_MATERIALS)
        for i, name in enumerate(MATERIAL_NAMES):
            print(f"  {name}: {counts[i]}")

    print(f"[train] Slip samples: {len(slip_ds)}")
    if len(slip_ds) > 0:
        n_slip = int(slip_ds.y_slip.sum())
        print(f"  normal: {len(slip_ds) - n_slip}  slip: {n_slip}")

    # combine both datasets
    if len(mat_ds) == 0 and len(slip_ds) == 0:
        print("[train] ERROR: no training data found")
        return

    # merge: material + slip samples
    all_X = []
    all_mat = []
    all_hap = []
    all_slip = []

    if len(mat_ds) > 0:
        all_X.append(mat_ds.X)
        all_mat.append(mat_ds.y_material)
        all_hap.append(mat_ds.y_haptic)
        all_slip.append(mat_ds.y_slip)

    if len(slip_ds) > 0:
        all_X.append(slip_ds.X)
        all_mat.append(slip_ds.y_material)
        all_hap.append(slip_ds.y_haptic)
        all_slip.append(slip_ds.y_slip)

    X = np.concatenate(all_X)
    y_mat = np.concatenate(all_mat)
    y_hap = np.concatenate(all_hap)
    y_slip = np.concatenate(all_slip)

    n = len(X)
    perm = np.random.permutation(n)
    X, y_mat, y_hap, y_slip = X[perm], y_mat[perm], y_hap[perm], y_slip[perm]

    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X[:n_train]), torch.tensor(y_mat[:n_train]),
        torch.tensor(y_hap[:n_train]), torch.tensor(y_slip[:n_train]),
    )
    val_X = torch.tensor(X[n_train:])
    val_mat = torch.tensor(y_mat[n_train:])
    val_hap = torch.tensor(y_hap[n_train:])
    val_slip = torch.tensor(y_slip[n_train:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TactileLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # losses
    # class weights for imbalanced material data
    mat_counts = np.bincount(y_mat[:n_train], minlength=N_MATERIALS).astype(np.float32)
    mat_counts = np.maximum(mat_counts, 1.0)
    mat_weights = torch.tensor(1.0 / mat_counts, device=device)
    mat_weights = mat_weights / mat_weights.sum() * N_MATERIALS

    ce_loss = nn.CrossEntropyLoss(weight=mat_weights)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # slip weight (slip is rare, upweight it)
    slip_pos = y_slip[:n_train].sum()
    slip_total = len(y_slip[:n_train])
    slip_w = max(1.0, (slip_total - slip_pos) / max(slip_pos, 1.0))
    slip_w = min(slip_w, 20.0)  # cap at 20x

    best_val_loss = float("inf")
    best_state = None

    print(f"\n[train] Training TactileLSTM on {device}")
    print(f"  {n_train} train / {n_val} val | seq_len={seq_len} | {epochs} epochs")
    print(f"  slip_weight={slip_w:.1f}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        mat_correct = 0
        mat_total = 0

        for xb, mb, hb, sb in train_dl:
            xb = xb.to(device)
            mb = mb.to(device)
            hb = hb.to(device)
            sb = sb.to(device)

            mat_pred, hap_pred, slip_pred = model(xb)

            loss_mat = ce_loss(mat_pred, mb)
            loss_hap = mse_loss(hap_pred, hb)
            loss_slip = bce_loss(slip_pred, sb) * slip_w

            loss = loss_mat + loss_hap + 0.5 * loss_slip

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)
            mat_correct += (mat_pred.argmax(1) == mb).sum().item()
            mat_total += len(mb)

        train_acc = mat_correct / max(1, mat_total)

        # validation
        model.eval()
        with torch.no_grad():
            vx = val_X.to(device)
            vm_pred, vh_pred, vs_pred = model(vx)
            v_mat_loss = ce_loss(vm_pred, val_mat.to(device)).item()
            v_hap_loss = mse_loss(vh_pred, val_hap.to(device)).item()
            v_slip_loss = bce_loss(vs_pred, val_slip.to(device)).item()
            v_loss = v_mat_loss + v_hap_loss + 0.5 * v_slip_loss
            v_acc = (vm_pred.argmax(1) == val_mat.to(device)).float().mean().item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | loss={total_loss/max(1,mat_total):.4f} "
                  f"| train_acc={train_acc:.2%} | val_acc={v_acc:.2%} "
                  f"| val_loss={v_loss:.4f}")

    # save
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "tactile_lstm.pt")
    torch.save({
        "model": best_state,
        "input_dim": N_INPUT,
        "hidden": 64,
        "n_layers": 2,
        "n_materials": N_MATERIALS,
        "material_names": MATERIAL_NAMES,
        "seq_len": seq_len,
        "val_loss": best_val_loss,
    }, save_path)
    print(f"\n[train] Saved → {save_path} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size, lr=args.lr)

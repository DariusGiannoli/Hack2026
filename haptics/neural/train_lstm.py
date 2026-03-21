"""
Train NeuralHapticRenderer (LSTM) via behavior cloning on synthetic grasp sequences.

Strategy:
  1. Use frozen MLP backbone (pretrained on DTD) to extract 64-dim material features
     from synthetic per-class DINO-like embeddings.
  2. Generate realistic synthetic grasp sequences (approach → contact → hold → release).
  3. Label each timestep with the 3-layer analytical rules (ground truth for cloning).
  4. Train LSTM to reproduce those labels — it learns temporal dynamics implicitly.

The LSTM generalizes beyond the rules because it sees noisy force sequences and must
learn smooth, temporally consistent predictions rather than per-frame lookups.

Run:
    python -m haptics.neural.train_lstm
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from haptics.neural.haptic_mlp import HapticMLP
from haptics.neural.haptic_lstm import NeuralHapticRenderer, save_renderer

# ── Paths ─────────────────────────────────────────────────────────────────────
MLP_PATH  = "models/haptic_mlp.pt"
LSTM_PATH = "models/haptic_lstm.pt"

# ── Training hyper-params ─────────────────────────────────────────────────────
EPOCHS     = 200
BATCH_SIZE = 64
LR         = 3e-4
SEQ_LEN    = 60     # timesteps per sequence  (~1.2 s at 50 Hz)
N_SEQ      = 300    # sequences per material class
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Material anchors ──────────────────────────────────────────────────────────
# Each class: freq_base (0–7), wave (0=square/1=sine), cap (max duty 0–31)
MATERIALS = {
    "metal":   {"freq_base": 6, "wave": 0, "cap": 31},
    "stone":   {"freq_base": 5, "wave": 0, "cap": 28},
    "wood":    {"freq_base": 4, "wave": 1, "cap": 25},
    "plastic": {"freq_base": 4, "wave": 0, "cap": 24},
    "rubber":  {"freq_base": 5, "wave": 1, "cap": 20},
    "leather": {"freq_base": 3, "wave": 1, "cap": 22},
    "fabric":  {"freq_base": 3, "wave": 1, "cap": 20},
    "foam":    {"freq_base": 2, "wave": 1, "cap": 15},
    "glass":   {"freq_base": 2, "wave": 1, "cap":  8},
}


# ── Analytical 3-layer labels ─────────────────────────────────────────────────

def _weber_fechner(force: float) -> float:
    """Psychophysically correct force → duty_raw mapping in [0,1]."""
    return math.log(1.0 + 9.0 * force) / math.log(10.0)


def _label(force_seq: np.ndarray, freq_base: int, wave: int) -> np.ndarray:
    """
    Apply 3-layer rules to a force sequence.
    Returns (T, 3): [freq_raw, duty_raw, wave_raw]  all in [0,1]

    Layer 2 — texture + pressure coupling:
        freq increases slightly with force (harder press → feel more texture detail)
    Layer 3 — Weber-Fechner duty curve:
        perceptually linear intensity scaling
    """
    T = len(force_seq)
    out = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        f = float(force_seq[t, 0])
        freq = min(7, freq_base + int(f * 2))
        out[t, 0] = freq / 7.0
        out[t, 1] = float(np.clip(_weber_fechner(f), 0, 1))
        out[t, 2] = float(wave)
    return out


# ── Synthetic force sequence generation ──────────────────────────────────────

def _make_force_seq(T: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Realistic grasp: approach → contact onset → hold → release.
    Returns (T, 3): [torque_norm, d_torque_norm, contact_flag]
    """
    torque    = np.zeros(T, dtype=np.float32)
    peak      = rng.uniform(0.3, 1.0)
    t_contact = rng.randint(6, 12)
    t_hold    = rng.randint(28, 42)
    t_release = min(t_hold + rng.randint(8, 16), T)

    for t in range(T):
        if t < t_contact:
            torque[t] = 0.0
        elif t < t_contact + 3:
            torque[t] = peak * (t - t_contact) / 3.0
        elif t < t_hold:
            torque[t] = peak + rng.normal(0, 0.04)
        elif t < t_release:
            frac = (t - t_hold) / max(1, t_release - t_hold)
            torque[t] = peak * (1 - frac) + rng.normal(0, 0.02)
        else:
            torque[t] = 0.0

    torque   = np.clip(torque, 0.0, 1.0)
    d_torque = np.clip(np.diff(torque, prepend=torque[0]) / 0.3, -1.0, 1.0)
    contact  = (torque > 0.05).astype(np.float32)

    return np.stack([torque, d_torque, contact], axis=1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticGraspDataset(Dataset):
    """
    N_SEQ x len(MATERIALS) grasp sequences with 3-layer analytical labels.
    Material features extracted from the frozen MLP backbone using
    class-clustered synthetic DINO embeddings (Gaussian noise around centroid).
    """

    def __init__(self, backbone: nn.Module, n_seq: int = N_SEQ,
                 T: int = SEQ_LEN, device: str = "cpu"):
        backbone = backbone.to(device).eval()
        rng = np.random.RandomState(0)
        self._data = []

        for mat_name, props in MATERIALS.items():
            centroid = rng.randn(384).astype(np.float32)
            centroid /= np.linalg.norm(centroid)

            for i in range(n_seq):
                emb = centroid + rng.randn(384).astype(np.float32) * 0.15
                emb /= np.linalg.norm(emb)

                with torch.no_grad():
                    t = torch.tensor(emb).unsqueeze(0).to(device)
                    feat = backbone(t).squeeze(0).cpu().numpy()   # (64,)

                force_seq = _make_force_seq(T, rng)
                labels    = _label(force_seq, props["freq_base"], props["wave"])
                self._data.append((feat, force_seq, labels))

        print(f"[SyntheticGraspDataset] {len(self._data)} sequences "
              f"({len(MATERIALS)} classes x {n_seq})")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        feat, force, labels = self._data[idx]
        return (
            torch.tensor(feat,   dtype=torch.float32),
            torch.tensor(force,  dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print(f"[train_lstm] Device: {DEVICE}")

    if not os.path.exists(MLP_PATH):
        raise FileNotFoundError(
            f"MLP checkpoint not found at {MLP_PATH}. "
            "Run haptics/neural/train_mlp.py first."
        )

    mlp = HapticMLP()
    mlp.load_state_dict(torch.load(MLP_PATH, map_location=DEVICE))
    backbone = mlp.backbone.to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("[train_lstm] MLP backbone loaded (frozen).")

    dataset = SyntheticGraspDataset(backbone, n_seq=N_SEQ, T=SEQ_LEN, device=DEVICE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model   = NeuralHapticRenderer(hidden_size=64, num_layers=1).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for feat, force_seq, labels in loader:
            feat      = feat.to(DEVICE)
            force_seq = force_seq.to(DEVICE)
            labels    = labels.to(DEVICE)

            freq_raw, duty_raw, wave_raw, _ = model(feat, force_seq)
            pred = torch.stack([freq_raw, duty_raw, wave_raw], dim=-1)
            loss = loss_fn(pred, labels)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        sched.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total/len(loader):.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

    save_renderer(model, LSTM_PATH)
    print(f"[train_lstm] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    train()

"""
Train a small MLP classifier on top of frozen DINOv2 embeddings.

Input:  photos organized as data/photos/{smooth,rough,soft}/
Output: models/dino_mlp.pt

Usage:
    python haptics/training/train_dino_mlp.py
    python haptics/training/train_dino_mlp.py --epochs 30 --lr 1e-3
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()  # enables PIL to open .HEIC files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from haptics.perception.dino_encoder import DinoEncoder

# ── Config ────────────────────────────────────────────────────────────────────

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "photos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
CACHE_PATH = os.path.join(MODEL_DIR, "dino_photo_embeddings.npz")

CLASS_MAP = {"smooth": 0, "rough": 1, "soft": 2}
CLASS_NAMES = ["smooth", "rough", "soft"]
N_CLASSES = len(CLASS_NAMES)
EMBED_DIM = 384


# ── MLP ───────────────────────────────────────────────────────────────────────

class MaterialMLP(nn.Module):
    """384-dim DINO embedding → material class."""

    def __init__(self, in_dim=EMBED_DIM, hidden=128, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Extract DINO features ────────────────────────────────────────────────────

def extract_embeddings(use_cache=True):
    """Load photos, run through DINO, return (embeddings, labels)."""
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"[train] Loading cached embeddings from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data["embeddings"], data["labels"]

    encoder = DinoEncoder()
    encoder.load()

    embeddings = []
    labels = []

    for class_name, class_idx in CLASS_MAP.items():
        class_dir = os.path.join(PHOTO_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"[train] WARNING: {class_dir} not found — skipping")
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic"))]
        print(f"[train] {class_name}: {len(files)} photos")

        for fname in files:
            path = os.path.join(class_dir, fname)
            try:
                pil_img = Image.open(path).convert("RGB")
                frame = np.array(pil_img)[:, :, ::-1].copy()  # RGB → BGR for encoder
                emb = encoder.encode(frame)
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(class_idx)
            except Exception as e:
                print(f"  skip ({e}): {fname}")

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(CACHE_PATH, embeddings=embeddings, labels=labels)
    print(f"[train] Cached {len(embeddings)} embeddings → {CACHE_PATH}")

    return embeddings, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs=50, lr=1e-3, batch_size=16, val_split=0.2):
    embeddings, labels = extract_embeddings()

    n = len(embeddings)
    print(f"\n[train] Total samples: {n}")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels == i).sum()
        print(f"  {name}: {count}")

    if n < 6:
        print("[train] ERROR: not enough data to train")
        return

    # shuffle and split
    perm = np.random.permutation(n)
    embeddings = embeddings[perm]
    labels = labels[perm]

    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    X_train = torch.tensor(embeddings[:n_train])
    y_train = torch.tensor(labels[:n_train])
    X_val = torch.tensor(embeddings[n_train:])
    y_val = torch.tensor(labels[n_train:])

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaterialMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = torch.tensor(1.0 / class_counts, device=device)
    weights = weights / weights.sum() * N_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state = None

    print(f"\n[train] Training MaterialMLP on {device} — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)

        train_acc = correct / max(1, total)

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_pred = val_logits.argmax(1)
            val_acc = (val_pred == y_val.to(device)).float().mean().item()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | loss={total_loss/max(1,total):.4f} "
                  f"| train_acc={train_acc:.2%} | val_acc={val_acc:.2%}")

    # save best
    save_path = os.path.join(MODEL_DIR, "dino_mlp.pt")
    torch.save({
        "model": best_state,
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "embed_dim": EMBED_DIM,
        "val_acc": best_val_acc,
    }, save_path)
    print(f"\n[train] Saved → {save_path} (val_acc={best_val_acc:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.no_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

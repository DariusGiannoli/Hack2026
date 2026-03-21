# neural/train_mlp.py
# Run: python3 -m neural.train_mlp

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM  = 384
EPOCHS     = 300
LR         = 3e-4
BATCH_SIZE = 128

BASE = os.path.join(os.path.dirname(__file__), '..')
SAVE_PATH  = os.path.join(BASE, 'models', 'haptic_mlp.pt')
CACHE_PATH = os.path.join(BASE, 'models', 'embeddings_cache.npz')
DTD_ROOT   = os.path.join(BASE, 'data', 'dtd', 'images')
OWN_ROOT   = os.path.join(BASE, 'data', 'own')

ANCHORS = {
    "smooth metal":  {"freq": 6, "duty": 22, "wave": 0},
    "rigid plastic": {"freq": 5, "duty": 20, "wave": 0},
    "glass":         {"freq": 7, "duty": 18, "wave": 0},
    "rough fabric":  {"freq": 3, "duty": 24, "wave": 1},
    "soft foam":     {"freq": 1, "duty": 16, "wave": 1},
    "human skin":    {"freq": 2, "duty": 14, "wave": 1},
    "wood":          {"freq": 4, "duty": 20, "wave": 0},
    "rubber":        {"freq": 3, "duty": 18, "wave": 1},
    "cardboard":     {"freq": 4, "duty": 16, "wave": 0},
}

DTD_MAPPING = {
    "smooth metal":  ["waffled", "striped", "grid", "crystalline", "crosshatched"],
    "rigid plastic": ["smooth", "glossy", "sprinkled", "studded", "polka-dotted"],
    "glass":         ["gauzy", "crystalline", "perforated"],
    "rough fabric":  ["woven", "knitted", "braided", "fibrous", "interlaced", "lacelike"],
    "soft foam":     ["bubbly", "porous", "spongy", "potholed", "pitted"],
    "human skin":    ["freckled", "wrinkled", "flecked", "blotchy", "stained"],
    "wood":          ["grooved", "lined", "stratified", "cracked", "veined", "marbled"],
    "rubber":        ["bumpy", "banded", "cobwebbed", "scaly", "smeared"],
    "cardboard":     ["matted", "fibrous", "meshed", "pleated", "frilly"],
}


def load_dino():
    print(f"[train_mlp] Loading DINOv2 ViT-S/14 on {DEVICE}...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[train_mlp] DINOv2 ready")
    return model


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ])


def encode_folder(folder, dino, preprocess, label):
    exts  = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    if not paths:
        return []
    results, batch_tensors = [], []
    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert('RGB')
            batch_tensors.append(preprocess(img))
        except:
            continue
        if len(batch_tensors) == 32 or i == len(paths) - 1:
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(DEVICE)
                with torch.no_grad():
                    embs = dino(batch).cpu().numpy()
                for emb in embs:
                    results.append((emb.astype(np.float32), label))
                batch_tensors = []
    return results


def load_dtd(dino, preprocess):
    dtd_root = Path(DTD_ROOT)
    if not dtd_root.exists():
        print(f"[train_mlp] DTD not found at {DTD_ROOT}")
        return []
    print("\n[train_mlp] Encoding DTD images...")
    all_data = []
    for anchor, categories in DTD_MAPPING.items():
        samples = []
        for cat in categories:
            cat_path = dtd_root / cat
            if cat_path.exists():
                samples.extend(encode_folder(cat_path, dino, preprocess, anchor))
        print(f"  {anchor:20s} ← {len(samples):4d} images")
        all_data.extend(samples)
    return all_data


def load_own(dino, preprocess):
    own_root = Path(OWN_ROOT)
    if not own_root.exists():
        return []
    print("\n[train_mlp] Encoding own photos...")
    all_data = []
    for anchor in ANCHORS:
        folder = own_root / anchor.replace(' ', '_')
        if not folder.exists():
            continue
        items = encode_folder(folder, dino, preprocess, anchor)
        weighted = items * 4
        print(f"  {anchor:20s} ← {len(items)} photos × 4 = {len(weighted)} samples")
        all_data.extend(weighted)
    return all_data


def build_tensors(all_data):
    X_list, Y_list, counts = [], [], {}
    for emb, label in all_data:
        if label not in ANCHORS:
            continue
        p = ANCHORS[label]
        target = np.array([p['freq']/7.0, p['duty']/31.0, float(p['wave'])],
                          dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        X_list.append(emb)
        Y_list.append(target)
        counts[label] = counts.get(label, 0) + 1
    return (torch.tensor(np.stack(X_list), dtype=torch.float32),
            torch.tensor(np.stack(Y_list), dtype=torch.float32),
            counts)


def train():
    from neural.haptic_mlp import HapticMLP, save_model

    print(f"[train_mlp] Device: {DEVICE}")
    os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"\n[train_mlp] Loading cached embeddings...")
        cache        = np.load(CACHE_PATH, allow_pickle=True)
        X            = torch.tensor(cache['X'], dtype=torch.float32)
        Y            = torch.tensor(cache['Y'], dtype=torch.float32)
        label_counts = dict(cache['label_counts'].item())
    else:
        dino       = load_dino()
        preprocess = get_preprocess()
        all_data   = load_dtd(dino, preprocess) + load_own(dino, preprocess)
        if not all_data:
            print("\n[!] No data found.")
            return
        X, Y, label_counts = build_tensors(all_data)
        np.savez(CACHE_PATH,
                 X=X.numpy(), Y=Y.numpy(),
                 label_counts=np.array(label_counts, dtype=object))
        print(f"[train_mlp] Cached → {CACHE_PATH}")
        del dino
        torch.cuda.empty_cache()

    print(f"\n[train_mlp] Dataset:")
    for label, count in sorted(label_counts.items()):
        p = ANCHORS.get(label, {})
        print(f"  {label:20s} {count:5d} → freq={p.get('freq','?')} duty={p.get('duty','?')} wave={p.get('wave','?')}")
    print(f"  TOTAL {X.shape[0]}\n")

    loader    = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model     = HapticMLP(input_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    loss_fn   = nn.MSELoss()

    print(f"[train_mlp] Training {EPOCHS} epochs...")
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            raw  = model(xb)
            pred = torch.stack([raw['freq_raw'], raw['duty_raw'], raw['wave_raw']], dim=1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | loss={total_loss/len(loader):.6f} | elapsed={time.time()-t0:.0f}s")

    save_model(model, SAVE_PATH)
    print(f"\n[train_mlp] Done in {time.time()-t0:.1f}s")

    model.eval()
    with torch.no_grad():
        idx  = torch.randperm(X.shape[0])[:64]
        raw  = model(X[idx].to(DEVICE))
        pred = torch.stack([raw['freq_raw'], raw['duty_raw'], raw['wave_raw']], dim=1).cpu()
        mse  = ((pred - Y[idx]) ** 2).mean().item()
    print(f"[train_mlp] Val MSE: {mse:.6f} | freq±{mse*7:.4f} duty±{mse*31:.4f}")


if __name__ == '__main__':
    train()
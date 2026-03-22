"""Quick test: run DINO MLP on one photo from each class."""
import sys, os, torch, numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

sys.path.insert(0, os.path.dirname(__file__))
from haptics.perception.dino_encoder import DinoEncoder
from haptics.training.train_dino_mlp import MaterialMLP

# load models
enc = DinoEncoder()
enc.load()

ckpt = torch.load("models/dino_mlp.pt", weights_only=False)
mlp = MaterialMLP()
mlp.load_state_dict(ckpt["model"])
mlp.eval()
names = ckpt["class_names"]

# test one from each class
tests = [
    ("smooth", "data/photos/smooth/IMG_5906.HEIC"),
    ("rough",  "data/photos/rough/IMG_5929.HEIC"),
    ("soft",   "data/photos/soft/IMG_5919.HEIC"),
]

# also test CLI arg if provided
if len(sys.argv) > 1:
    tests = [("unknown", sys.argv[1])]

for expected, path in tests:
    img = Image.open(path).convert("RGB")
    frame = np.array(img)[:, :, ::-1].copy()
    emb = enc.encode(frame)
    with torch.no_grad():
        logits = mlp(torch.tensor(emb).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    pred = names[probs.argmax()]
    ok = "OK" if pred == expected or expected == "unknown" else "WRONG"
    print(f"  {os.path.basename(path):30s} expected={expected:8s} pred={pred:8s} "
          f"[{' | '.join(f'{n}={probs[i]:.0%}' for i, n in enumerate(names))}] {ok}")

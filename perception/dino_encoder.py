# perception/dino_encoder.py
import torch
import numpy as np
import threading
from torchvision import transforms
from PIL import Image


class DinoEncoder:
    def __init__(self, device=None):
        self.device   = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model    = None
        self._lock    = threading.Lock()
        self._ready   = False
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self):
        try:
            print(f"[DinoEncoder] Loading dinov2_vits14 on {self.device}...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
            self.model = self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._ready = True
            print("[DinoEncoder] Ready — 384-dim")
            return True
        except Exception as e:
            print(f"[DinoEncoder] Failed: {e}")
            return False

    def encode(self, frame_bgr, bbox=None):
        if not self._ready:
            return None
        with self._lock:
            try:
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        crop = frame_bgr
                else:
                    crop = frame_bgr
                pil    = Image.fromarray(crop[..., ::-1])
                tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model(tensor).squeeze(0).cpu().numpy()
                norm = np.linalg.norm(emb)
                return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)
            except Exception as e:
                print(f"[DinoEncoder] error: {e}")
                return None

    @property
    def ready(self):
        return self._ready

    @property
    def embedding_dim(self):
        return 384
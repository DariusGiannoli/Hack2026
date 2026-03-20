# neural/haptic_mlp.py

import torch
import torch.nn as nn
import numpy as np
import os


class HapticMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.head_freq = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.head_duty = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.head_wave = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.backbone(x)
        return {
            'freq_raw': self.head_freq(h).squeeze(-1),
            'duty_raw': self.head_duty(h).squeeze(-1),
            'wave_raw': self.head_wave(h).squeeze(-1),
        }

    def predict(self, embedding, device='cuda'):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
            raw = self.forward(x)
            freq_cont = raw['freq_raw'].item() * 7.0
            duty_cont = raw['duty_raw'].item() * 31.0
            wave_cont = raw['wave_raw'].item()
        return {
            'freq':      int(round(freq_cont)),
            'duty':      int(round(duty_cont)),
            'wave':      int(wave_cont >= 0.5),
            'freq_cont': freq_cont,
            'duty_cont': duty_cont,
            'wave_cont': wave_cont,
        }


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[HapticMLP] Saved to {path}")


def load_model(path, device='cuda'):
    model = HapticMLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    print(f"[HapticMLP] Loaded from {path}")
    return model
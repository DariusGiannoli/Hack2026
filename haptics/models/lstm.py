"""
NeuralHapticRenderer — LSTM-based real-time haptic signal renderer.

Architecture:
  - material_feat (64-dim) from frozen MLP backbone (DINO → MLP.backbone)
  - force_features (3-dim): [torque_norm, d_torque_norm, contact_flag]
  - LSTM (67 → hidden_size=64, 1 layer) — learns grasp dynamics
  - Three heads: freq_raw, duty_raw, wave_raw  ∈ [0,1]

Inference:
  freq = round(freq_raw × 7)          → 0–7
  duty = round(duty_raw × fragility_cap)  → VLM cap applied externally
  wave = int(wave_raw ≥ 0.5)          → 0 or 1
"""

import os
import numpy as np
import torch
import torch.nn as nn


MATERIAL_DIM = 64   # MLP backbone output
FORCE_DIM    = 3    # torque_norm, d_torque_norm, contact_flag
LSTM_INPUT   = MATERIAL_DIM + FORCE_DIM   # 67


class NeuralHapticRenderer(nn.Module):
    """
    Pure LSTM haptic renderer.
    Takes pre-extracted material features (64-dim) + force features (3-dim).
    No vision backbone inside — handled externally by HapticInference.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head_freq = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.head_duty = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.head_wave = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, material_feat, force_seq, hidden=None):
        """
        material_feat : (B, 64)     — constant per grasp
        force_seq     : (B, T, 3)   — [torque_norm, d_torque_norm, contact]
        hidden        : optional LSTM state for online inference

        Returns:
            freq_raw  (B, T) ∈ [0,1]  → × 7    = freq
            duty_raw  (B, T) ∈ [0,1]  → × cap  = duty  (VLM cap applied outside)
            wave_raw  (B, T) ∈ [0,1]  → ≥ 0.5  = sine
            hidden    updated LSTM state
        """
        B, T, _ = force_seq.shape
        mat = material_feat.unsqueeze(1).expand(B, T, -1)   # (B, T, 64)
        x   = torch.cat([mat, force_seq], dim=-1)            # (B, T, 67)

        out, hidden = self.lstm(x, hidden)                   # (B, T, 64)
        return (
            self.head_freq(out).squeeze(-1),   # (B, T)
            self.head_duty(out).squeeze(-1),
            self.head_wave(out).squeeze(-1),
            hidden,
        )


class HapticInference:
    """
    Stateful online inference.  Maintains LSTM hidden state across timesteps.

    Usage:
        inf = HapticInference.from_checkpoints("models/haptic_mlp.pt",
                                                "models/haptic_lstm.pt")
        inf.set_material(dino_embedding_384)   # call every 2–5 frames

        # at every 50 Hz tick:
        cmd = inf.step(torque_norm, d_torque_norm, contact_flag)
        # → {"freq": int, "duty_raw": float, "wave": int}
        # apply VLM cap outside: duty = round(cmd["duty_raw"] × fragility_cap)
    """

    def __init__(self, backbone: nn.Module, renderer: "NeuralHapticRenderer",
                 device: str = "cpu"):
        self.device   = device
        self.backbone = backbone.to(device).eval()
        self.renderer = renderer.to(device).eval()

        self._hidden       = None
        self._material_feat = None   # (1, 1, 64) cached

        for p in self.backbone.parameters():
            p.requires_grad = False

    # ── material update (2–5 Hz) ──────────────────────────────────────────────
    def set_material(self, dino_emb: np.ndarray):
        """Feed a new 384-dim DINO embedding. Safe to call at any time."""
        with torch.no_grad():
            t = torch.tensor(dino_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            feat = self.backbone(t)                       # (1, 64)
            self._material_feat = feat.unsqueeze(1)       # (1, 1, 64)

    # ── force step (50–100 Hz) ────────────────────────────────────────────────
    def step(self, torque_norm: float, d_torque_norm: float,
             contact: float) -> dict:
        """
        Single timestep.  Returns {"freq", "duty_raw", "wave"}.
        duty_raw ∈ [0,1] — multiply by fragility_cap to get final duty.
        """
        if self._material_feat is None:
            return {"freq": 3, "duty_raw": 0.0, "wave": 1}

        force = torch.tensor(
            [[[torque_norm, d_torque_norm, contact]]],
            dtype=torch.float32,
        ).to(self.device)                                  # (1, 1, 3)

        x = torch.cat([self._material_feat, force], dim=-1)  # (1, 1, 67)

        with torch.no_grad():
            out, self._hidden = self.renderer.lstm(x, self._hidden)
            freq_raw = self.renderer.head_freq(out).item()
            duty_raw = self.renderer.head_duty(out).item()
            wave_raw = self.renderer.head_wave(out).item()

        return {
            "freq":     int(round(freq_raw * 7)),
            "duty_raw": float(duty_raw),
            "wave":     int(wave_raw >= 0.5),
        }

    def reset(self):
        """Reset LSTM state — call when new object or grasp ends."""
        self._hidden = None

    # ── factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_checkpoints(cls, mlp_path: str, lstm_path: str,
                         device: str = "cpu") -> "HapticInference":
        from haptics.models.mlp import HapticMLP
        mlp = HapticMLP()
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))

        renderer = NeuralHapticRenderer()
        renderer.load_state_dict(torch.load(lstm_path, map_location=device))

        return cls(backbone=mlp.backbone, renderer=renderer, device=device)


def save_renderer(model: NeuralHapticRenderer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[NeuralHapticRenderer] Saved → {path}")

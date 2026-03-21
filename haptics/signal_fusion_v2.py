"""
SignalFusionV2 — 3-layer neural haptic fusion.

Layer 1 — Contact transient  (event-driven, instant)
    Detects torque onset spike → fires a short high-intensity pulse.
    Mimics FA-I mechanoreceptor response to contact events.

Layer 2 — Texture rendering  (DINO + LSTM, 2–5 Hz update)
    NeuralHapticRenderer predicts freq + wave from material features.
    Pressure-texture coupling baked into LSTM (learned from 3-layer rules).

Layer 3 — Force magnitude    (arm torques, 50–100 Hz)
    Weber-Fechner log curve: perceptually linear intensity scaling.
    Capped by VLM fragility_cap (safety ceiling per object).

Usage:
    fusion = SignalFusionV2()
    fusion.load_model("models/haptic_mlp.pt", "models/haptic_lstm.pt")

    fusion.update_from_scene(scene_dict)       # once, after VLM
    fusion.update_from_dino(dino_embedding)    # every 2–5 frames

    cmd = fusion.step(torque_sum)              # 50–100 Hz
    # → {"freq": int, "duty": int, "wave": int}
"""

import math
import sys
import os
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── torque normalization ───────────────────────────────────────────────────────
TORQUE_MIN = 2.0    # Nm — below this is noise / no contact
TORQUE_MAX = 25.0   # Nm — clamp ceiling

# ── contact event detection ───────────────────────────────────────────────────
CONTACT_DELTA_THRESH = 0.15   # normalized torque spike to trigger transient
CONTACT_DUTY         = 28     # transient intensity (high, brief)
CONTACT_FREQ         = 6      # transient frequency (sharp feel)
CONTACT_WAVE         = 0      # square wave (more impactful)
CONTACT_MS           = 25     # transient duration in ms


def _normalize(torque_sum: float) -> float:
    v = (torque_sum - TORQUE_MIN) / (TORQUE_MAX - TORQUE_MIN)
    return float(np.clip(v, 0.0, 1.0))


def _weber_fechner(force: float, cap: int) -> int:
    """Perceptually linear duty: log curve scaled to fragility cap."""
    raw = math.log(1.0 + 9.0 * force) / math.log(10.0)
    return int(np.clip(raw * cap, 0, cap))


class SignalFusionV2:

    def __init__(self):
        self._lock = threading.Lock()

        # VLM state
        self._fragility_cap = 31
        self._object_name   = "unknown"
        self._active        = False

        # LSTM inference state
        self._inference     = None   # HapticInference, loaded lazily
        self._lstm_freq     = 3
        self._lstm_wave     = 1

        # contact event state
        self._prev_torque_norm = 0.0
        self._haptic_ctrl      = None   # set via attach_controller()

    # ── model loading ──────────────────────────────────────────────────────────
    def load_model(self, mlp_path: str, lstm_path: str, device: str = "cpu"):
        """Load pretrained MLP backbone + LSTM renderer."""
        from haptics.neural.haptic_lstm import HapticInference
        self._inference = HapticInference.from_checkpoints(mlp_path, lstm_path, device)
        print(f"[FusionV2] LSTM renderer loaded ({device})")

    def attach_controller(self, haptic_ctrl):
        """Attach HapticController for contact transient pulses (Layer 1)."""
        self._haptic_ctrl = haptic_ctrl

    # ── slow update: VLM ──────────────────────────────────────────────────────
    def update_from_scene(self, scene: dict, target_object: str = None):
        """Call once after VLM identifies the scene."""
        objects = scene.get("objects", [])
        if not objects:
            return
        obj = objects[0]
        if target_object:
            obj = next(
                (o for o in objects if target_object.lower() in o["name"].lower()),
                objects[0],
            )
        name = obj.get("name", "unknown")
        cap  = scene.get("fragility_caps", {}).get(name, 31)

        with self._lock:
            self._object_name   = name
            self._fragility_cap = cap
            self._active        = True
            if self._inference:
                self._inference.reset()

        print(f"[FusionV2] VLM  → {name}  fragility_cap={cap}")

    # ── medium update: DINO embedding (2–5 Hz) ────────────────────────────────
    def update_from_dino(self, dino_emb: np.ndarray):
        """Feed a fresh DINO embedding. LSTM caches material features."""
        if self._inference is not None:
            self._inference.set_material(dino_emb)

    # ── fast step: 50–100 Hz ─────────────────────────────────────────────────
    def step(self, torque_sum: float) -> dict:
        """
        Main control tick. Returns {"freq": int, "duty": int, "wave": int}.
        Call at 50–100 Hz with current arm torque sum (Nm).
        """
        with self._lock:
            if not self._active:
                return {"freq": 0, "duty": 0, "wave": 0}

            force      = _normalize(torque_sum)
            d_force    = force - self._prev_torque_norm
            contact    = float(force > 0.05)

            # ── Layer 1: contact transient ────────────────────────────────────
            if d_force > CONTACT_DELTA_THRESH and self._haptic_ctrl is not None:
                from haptics.preset_library import FINGER_ADDRS
                for addr in FINGER_ADDRS.values():
                    self._haptic_ctrl.pulse(
                        addr, CONTACT_DUTY, CONTACT_FREQ, CONTACT_WAVE, CONTACT_MS
                    )

            # ── Layer 2+3: LSTM → freq/wave, then Weber-Fechner duty ──────────
            if self._inference is not None:
                lstm_out = self._inference.step(force, d_force, contact)
                freq     = lstm_out["freq"]
                wave     = lstm_out["wave"]
                # LSTM duty_raw already pressure-coupled; apply VLM cap
                duty = int(np.clip(
                    lstm_out["duty_raw"] * self._fragility_cap,
                    0, self._fragility_cap
                ))
            else:
                # fallback if no model loaded
                freq = self._lstm_freq
                wave = self._lstm_wave
                duty = _weber_fechner(force, self._fragility_cap)

            self._prev_torque_norm = force

        return {"freq": freq, "duty": duty, "wave": wave}

    # ── convenience ───────────────────────────────────────────────────────────
    @property
    def object_name(self):
        return self._object_name

    @property
    def fragility_cap(self):
        return self._fragility_cap

    @property
    def is_active(self):
        return self._active

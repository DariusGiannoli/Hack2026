"""
neural/hapticnet.py — HapticNet: real-time haptic prediction from tactile sensors.

Architecture
------------
                                          ┌──────────────┐
  pressure (B,T,20,6) ──► Conv1D encoder ─►  pool ──► 32 ─┐
                                                           │
  torque   (B,T,20,12)──► Conv1D encoder ─►  pool ──► 32 ─┼──► cat(72)
                                                           │      │
  fragility (B,1)      ──► Linear(1→8)─► ReLU ───────► 8 ─┘      │
                                                              Linear(72→64)
                                                                  │
                                                            LSTM(64,64,1)
                                                                  │
                                                            Linear(64→4)
                                                                  │
                                                              Sigmoid
                                                                  │
                                                   ┌──────────────┼───────────┐
                                                   ▼              ▼           ▼
                                              freq_raw       duty_raw    wave_raw
                                               [0,1]          [0,1]      [0,1]
                                              ×7→int        ×cap→int    ≥.5→int
                                                                         slip_prob
                                                                          [0,1]

Input shapes (training — sequences of windows):
    pressure_windows : (B, T, 20, 6)   T consecutive windows of 20 frames × 6 sensors
    torque_windows   : (B, T, 20, 12)  T consecutive windows of 20 frames × 12 joints
    fragility        : (B, 1)          scalar from GPT-4V, constant per sequence

Output:
    (B, T, 4) — [freq_raw, duty_raw, wave_raw, slip_prob] ∈ [0,1]

Inference (single-step, 50 Hz):
    Call .step() with one 20-frame window, carries LSTM hidden state.

Parameter count: ~46K (runs comfortably at 50 Hz on CPU).
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_LEN   = 20   # frames per observation window
N_PRESSURE   = 6    # pressure sensor channels
N_TORQUE     = 12   # joint torque channels (6 per arm, excluding wrist yaw)
CONV_CH      = 32   # conv hidden channels
FRAG_DIM     = 8    # fragility embedding dim
PROJECT_DIM  = 64   # projection / LSTM hidden dim


# ── Model ────────────────────────────────────────────────────────────────────

class HapticNet(nn.Module):

    def __init__(self):
        super().__init__()

        # --- Pressure encoder: (B, 6, 20) → (B, 32) ---
        self.pressure_enc = nn.Sequential(
            nn.Conv1d(N_PRESSURE, CONV_CH, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(CONV_CH, CONV_CH, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # (B, 32, 1)
            nn.Flatten(),              # (B, 32)
        )

        # --- Torque encoder: (B, 12, 20) → (B, 32) ---
        self.torque_enc = nn.Sequential(
            nn.Conv1d(N_TORQUE, CONV_CH, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(CONV_CH, CONV_CH, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # --- Fragility embedding: (B, 1) → (B, 8) ---
        self.frag_enc = nn.Sequential(
            nn.Linear(1, FRAG_DIM),
            nn.ReLU(),
        )

        # --- Projection: cat(32+32+8)=72 → 64 ---
        cat_dim = CONV_CH + CONV_CH + FRAG_DIM  # 72
        self.project = nn.Sequential(
            nn.Linear(cat_dim, PROJECT_DIM),
            nn.ReLU(),
        )

        # --- Temporal: LSTM(64, 64, 1 layer) ---
        self.lstm = nn.LSTM(
            input_size=PROJECT_DIM,
            hidden_size=PROJECT_DIM,
            num_layers=1,
            batch_first=True,
        )

        # --- Output head: 64 → 4, all in [0,1] ---
        self.head = nn.Sequential(
            nn.Linear(PROJECT_DIM, 4),
            nn.Sigmoid(),
        )

    # ── encode one window ────────────────────────────────────────────────

    def _encode_window(self, pressure: torch.Tensor, torque: torch.Tensor,
                       fragility: torch.Tensor) -> torch.Tensor:
        """
        Encode a single observation window to a 64-dim feature vector.

        Args:
            pressure:  (B, 20, 6)
            torque:    (B, 20, 12)
            fragility: (B, 1)

        Returns:
            (B, 64)
        """
        # Conv1d expects (B, C, L) — permute from (B, L, C)
        p = self.pressure_enc(pressure.permute(0, 2, 1))  # (B, 32)
        t = self.torque_enc(torque.permute(0, 2, 1))      # (B, 32)
        f = self.frag_enc(fragility)                       # (B, 8)
        cat = torch.cat([p, t, f], dim=1)                  # (B, 72)
        return self.project(cat)                           # (B, 64)

    # ── training forward: full sequence ──────────────────────────────────

    def forward(self, pressure: torch.Tensor, torque: torch.Tensor,
                fragility: torch.Tensor, hidden=None):
        """
        Process a sequence of observation windows.

        Args:
            pressure:  (B, T, 20, 6)   — T windows of 20 pressure frames
            torque:    (B, T, 20, 12)   — T windows of 20 torque frames
            fragility: (B, 1)           — constant per sequence
            hidden:    optional LSTM state

        Returns:
            output: (B, T, 4) — [freq_raw, duty_raw, wave_raw, slip_prob]
            hidden: LSTM hidden state
        """
        B, T = pressure.shape[:2]

        # encode each window
        feats = []
        for t in range(T):
            feats.append(self._encode_window(
                pressure[:, t],   # (B, 20, 6)
                torque[:, t],     # (B, 20, 12)
                fragility,        # (B, 1)
            ))
        feat_seq = torch.stack(feats, dim=1)  # (B, T, 64)

        # LSTM
        lstm_out, hidden = self.lstm(feat_seq, hidden)  # (B, T, 64)

        # output head
        output = self.head(lstm_out)  # (B, T, 4)

        return output, hidden

    # ── inference: single step ───────────────────────────────────────────

    def step(self, pressure: torch.Tensor, torque: torch.Tensor,
             fragility: torch.Tensor, hidden):
        """
        Process one observation window at inference time.

        Args:
            pressure:  (1, 20, 6)
            torque:    (1, 20, 12)
            fragility: (1, 1)
            hidden:    LSTM hidden state from previous step (or None)

        Returns:
            output: (4,)  — [freq_raw, duty_raw, wave_raw, slip_prob]
            hidden: updated LSTM state
        """
        feat = self._encode_window(pressure, torque, fragility)  # (1, 64)
        feat = feat.unsqueeze(1)  # (1, 1, 64)
        lstm_out, hidden = self.lstm(feat, hidden)
        output = self.head(lstm_out[:, 0, :])  # (1, 4)
        return output.squeeze(0), hidden


# ── Inference wrapper ────────────────────────────────────────────────────────

class HapticNetInference:
    """
    Real-time inference at 50 Hz with rolling window and LSTM state.

    Usage:
        inf = HapticNetInference("models/hapticnet.pt")
        inf.set_fragility(0.3)

        # every 20ms — feed raw sensor readings
        result = inf.step(pressure_6, torque_12)
        # result = {"freq": 5, "duty_raw": 0.72, "wave": 0, "slip": 0.03}
    """

    def __init__(self, model_or_path, device: str = "cpu"):
        if isinstance(model_or_path, str):
            self.model = load_model(model_or_path, device)
        else:
            self.model = model_or_path.to(device)
        self.model.eval()
        self.device = device

        # rolling sensor buffers (last 20 frames)
        self._pressure_buf = np.zeros((WINDOW_LEN, N_PRESSURE), dtype=np.float32)
        self._torque_buf   = np.zeros((WINDOW_LEN, N_TORQUE),   dtype=np.float32)
        self._fragility    = 0.5
        self._hidden       = None
        self._frame_count  = 0

    def set_fragility(self, value: float):
        """Set fragility scalar from GPT-4V (0.0 = indestructible, 1.0 = glass)."""
        self._fragility = float(np.clip(value, 0.0, 1.0))

    def reset(self):
        """Reset LSTM state and buffers (call on new object / scene change)."""
        self._pressure_buf[:] = 0
        self._torque_buf[:] = 0
        self._hidden = None
        self._frame_count = 0

    @torch.no_grad()
    def step(self, pressure_6: list, torque_12: list) -> dict:
        """
        Feed one frame of raw sensor data, get haptic command.

        Args:
            pressure_6:  list/array of 6 floats — finger pressure sensors
            torque_12:   list/array of 12 floats — arm joint torques

        Returns:
            dict with keys: freq (int 0-7), duty_raw (float 0-1),
                            wave (int 0 or 1), slip (float 0-1)
        """
        # shift buffer left, append new frame
        self._pressure_buf = np.roll(self._pressure_buf, -1, axis=0)
        self._torque_buf   = np.roll(self._torque_buf,   -1, axis=0)
        self._pressure_buf[-1] = np.array(pressure_6, dtype=np.float32)[:N_PRESSURE]
        self._torque_buf[-1]   = np.array(torque_12,  dtype=np.float32)[:N_TORQUE]
        self._frame_count += 1

        # build tensors
        p = torch.tensor(self._pressure_buf, dtype=torch.float32).unsqueeze(0).to(self.device)
        t = torch.tensor(self._torque_buf,   dtype=torch.float32).unsqueeze(0).to(self.device)
        f = torch.tensor([[self._fragility]], dtype=torch.float32).to(self.device)

        # inference
        out, self._hidden = self.model.step(p, t, f, self._hidden)
        out = out.cpu().numpy()

        return {
            "freq":     int(round(out[0] * 7.0)),
            "duty_raw": float(out[1]),
            "wave":     int(out[2] >= 0.5),
            "slip":     float(out[3]),
        }

    @property
    def warmed_up(self) -> bool:
        """True once we've received at least WINDOW_LEN frames."""
        return self._frame_count >= WINDOW_LEN


# ── Mock sensor stream ───────────────────────────────────────────────────────

# Material-specific texture profiles for synthetic data generation
# Each has: texture_freq (Hz), texture_amp, base_pressure, noise_std, grip_force_range
MATERIAL_PROFILES = {
    1: {  # smooth/hard (metal, glass, plastic)
        "texture_freq": [8.0, 10.0],   # high-freq micro-vibrations
        "texture_amp": 0.05,            # low amplitude — smooth surface
        "base_pressure": [1.5, 4.0],    # medium-high grip force needed
        "noise_std": 0.01,              # very clean signal
        "torque_base": [2.0, 5.0],      # medium arm effort
        "slip_rate": 0.15,              # slippery — slips more often
    },
    2: {  # rough (fabric, cardboard, wood)
        "texture_freq": [2.0, 5.0],    # mid-freq texture bumps
        "texture_amp": 0.25,            # strong texture signal
        "base_pressure": [1.0, 3.0],    # moderate grip
        "noise_std": 0.04,              # noisy from texture
        "torque_base": [1.5, 4.0],
        "slip_rate": 0.05,              # rough surfaces grip well
    },
    3: {  # soft (foam, sponge, rubber)
        "texture_freq": [0.5, 2.0],    # low-freq deformation
        "texture_amp": 0.12,
        "base_pressure": [0.3, 2.0],    # low pressure — soft object
        "noise_std": 0.02,
        "torque_base": [0.5, 2.0],      # light objects
        "slip_rate": 0.03,              # rubber grips well
    },
}

# Per-finger contact patterns for different grasp types
GRASP_PROFILES = {
    "power": {
        # all fingers engaged, thumb opposes
        "finger_weights": [0.6, 0.8, 1.0, 0.9, 1.0, 0.7],  # pinky→thumb_rot
        "contact_order": [0.0, 0.02, 0.04, 0.06, 0.10, 0.10],  # stagger (seconds)
    },
    "precision": {
        # mostly index + thumb, others light
        "finger_weights": [0.1, 0.2, 0.3, 1.0, 1.0, 0.8],
        "contact_order": [0.15, 0.10, 0.05, 0.0, 0.02, 0.02],
    },
    "lateral": {
        # thumb + index side pinch
        "finger_weights": [0.05, 0.1, 0.15, 0.8, 1.0, 0.3],
        "contact_order": [0.2, 0.15, 0.10, 0.0, 0.03, 0.03],
    },
    "tripod": {
        # thumb + index + middle
        "finger_weights": [0.05, 0.1, 0.9, 1.0, 1.0, 0.6],
        "contact_order": [0.18, 0.12, 0.0, 0.02, 0.04, 0.04],
    },
}
GRASP_NAMES = list(GRASP_PROFILES.keys())


class MockSensorStream:
    """
    Generates rich synthetic sensor data for training without the robot.

    Features:
      - Material-specific texture signatures (smooth/rough/soft)
      - Per-finger contact patterns for different grasp types (power/precision/lateral/tripod)
      - Staggered finger contact onset and release
      - Weber-Fechner-aware pressure profiles
      - Realistic slip events: torque spikes + pressure drops + finger-specific patterns
      - Varied grasp dynamics: random cycle lengths, grip strengths, approach speeds
    """

    def __init__(self, hz: float = 50.0, seed: int = 42):
        self.hz = hz
        self._t = 0.0
        self._dt = 1.0 / hz
        self._rng = np.random.RandomState(seed)

        # current grasp cycle state
        self._new_cycle()

        # slip event scheduling
        self._next_slip = self._rng.uniform(2.0, 6.0)
        self._slip_active = False
        self._slip_end = 0.0
        self._slip_finger_mask = np.ones(6)  # which fingers lose pressure during slip

    def _new_cycle(self):
        """Initialize a new grasp cycle with randomized parameters."""
        self._cycle_start = self._t

        # random cycle timing
        self._approach_dur = self._rng.uniform(0.5, 1.5)
        self._hold_dur = self._rng.uniform(4.0, 10.0)
        self._release_dur = self._rng.uniform(0.5, 1.5)
        self._pause_dur = self._rng.uniform(0.5, 2.0)
        self._cycle_len = (self._approach_dur + self._hold_dur
                           + self._release_dur + self._pause_dur)

        # random material and grasp type
        self._material = self._rng.choice([1, 2, 3])
        self._grasp = GRASP_NAMES[self._rng.randint(len(GRASP_NAMES))]
        self._profile = MATERIAL_PROFILES[self._material]
        self._grasp_profile = GRASP_PROFILES[self._grasp]

        # random grip strength within material range
        bp = self._profile["base_pressure"]
        self._grip_strength = self._rng.uniform(bp[0], bp[1])

        # random texture frequency within range
        tf = self._profile["texture_freq"]
        self._tex_freq = self._rng.uniform(tf[0], tf[1])

        # random per-finger phase offsets
        self._finger_phases = self._rng.uniform(0, 2 * np.pi, 6)

    def _contact_envelope(self, cycle_t: float, finger_idx: int) -> float:
        """Per-finger contact envelope with staggered onset/release."""
        delay = self._grasp_profile["contact_order"][finger_idx]
        weight = self._grasp_profile["finger_weights"][finger_idx]

        # adjust timing for this finger's delay
        t = cycle_t - delay
        if t < 0:
            return 0.0

        a = self._approach_dur - delay
        if a <= 0:
            a = 0.1  # minimum ramp

        if t < a:
            return weight * (t / a)  # ramp up
        elif t < a + self._hold_dur:
            return weight  # hold
        elif t < a + self._hold_dur + self._release_dur:
            return weight * (1.0 - (t - a - self._hold_dur) / self._release_dur)
        return 0.0

    def step(self) -> dict:
        """
        Returns one frame of simulated sensor data.

        Returns:
            dict with "pressure" (6,), "torque" (12,), "slip_label" (bool)
        """
        t = self._t
        cycle_t = t - self._cycle_start

        # check if we need a new cycle
        if cycle_t >= self._cycle_len:
            self._new_cycle()
            cycle_t = 0.0

        prof = self._profile
        grip = self._grip_strength

        # --- per-finger pressure ---
        pressure = np.zeros(6, dtype=np.float32)
        any_contact = False
        for i in range(6):
            envelope = self._contact_envelope(cycle_t, i)
            if envelope < 0.01:
                pressure[i] = max(0.0, self._rng.normal(0, 0.005))
                continue

            any_contact = True

            # base force from grip strength × envelope
            base = grip * envelope

            # material-specific texture vibration
            tex = prof["texture_amp"] * np.sin(
                2 * np.pi * self._tex_freq * t + self._finger_phases[i]
            )
            # add harmonics for rough materials
            if self._material == 2:
                tex += 0.12 * np.sin(
                    2 * np.pi * self._tex_freq * 2.3 * t + self._finger_phases[i] * 1.7
                )
                tex += 0.06 * np.sin(
                    2 * np.pi * self._tex_freq * 4.1 * t + self._finger_phases[i] * 0.5
                )

            # sensor noise
            noise = self._rng.normal(0, prof["noise_std"])

            pressure[i] = max(0.0, base + base * tex + noise)

        # --- torque: arm joint effort ---
        torque = np.zeros(12, dtype=np.float32)
        tb = prof["torque_base"]
        torque_base = self._rng.uniform(tb[0], tb[1]) if any_contact else 0.0

        # compute overall contact level for torque scaling
        contact_level = np.mean([self._contact_envelope(cycle_t, i) for i in range(6)])

        for i in range(12):
            # base effort scales with contact
            effort = torque_base * contact_level
            # slow postural drift
            drift = 0.3 * np.sin(0.3 * t + i * 0.5)
            # high-freq tremor (natural hand tremor ~8-12 Hz)
            tremor = 0.05 * np.sin(2 * np.pi * 9.5 * t + i * 0.7)
            torque[i] = effort * (1.0 + drift * 0.1) + tremor + self._rng.normal(0, 0.08)

        # --- slip events ---
        slip_label = False

        # schedule next slip based on material slip rate
        if (t >= self._next_slip and not self._slip_active
                and contact_level > 0.4):
            self._slip_active = True
            self._slip_end = t + self._rng.uniform(0.08, 0.35)
            # next slip interval depends on material
            interval = 1.0 / max(0.01, prof["slip_rate"])
            self._next_slip = t + self._rng.uniform(interval * 0.5, interval * 1.5)
            # random subset of fingers affected by slip
            n_affected = self._rng.randint(2, 6)
            affected = self._rng.choice(6, n_affected, replace=False)
            self._slip_finger_mask = np.ones(6)
            self._slip_finger_mask[affected] = self._rng.uniform(0.3, 0.7, n_affected)

        if self._slip_active:
            slip_label = True
            slip_progress = (t - (self._slip_end - 0.15)) / 0.15

            # torque spike: sudden arm compensation (gravity fighting)
            spike_mag = self._rng.uniform(5.0, 12.0)
            spike = spike_mag * np.exp(-15.0 * max(0, slip_progress))
            # primarily shoulder and elbow joints compensate
            torque[0:4] += spike          # left shoulder + elbow
            torque[6:10] += spike * 0.5   # right arm sympathetic

            # pressure drop on affected fingers (object sliding)
            for i in range(6):
                pressure[i] *= self._slip_finger_mask[i]

            if t >= self._slip_end:
                self._slip_active = False

        self._t += self._dt

        return {
            "pressure": pressure,
            "torque": torque,
            "slip_label": slip_label,
        }

    def get_material_label(self) -> int:
        """Returns the current simulated material class (1=smooth, 2=rough, 3=soft)."""
        return self._material


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_model(path: str, device: str = "cpu") -> HapticNet:
    """Load a trained HapticNet from checkpoint."""
    model = HapticNet()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[HapticNet] Loaded from {path} ({sum(p.numel() for p in model.parameters())} params)")
    return model


def save_model(model: HapticNet, path: str, epoch: int = 0, loss: float = 0.0):
    """Save HapticNet checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    print(f"[HapticNet] Saved to {path}")


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = HapticNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HapticNet: {n_params:,} parameters")

    # test training forward
    B, T = 4, 10
    p = torch.randn(B, T, WINDOW_LEN, N_PRESSURE)
    t = torch.randn(B, T, WINDOW_LEN, N_TORQUE)
    f = torch.rand(B, 1)
    out, h = model(p, t, f)
    print(f"Training forward:  input=({B},{T},20,6)+({B},{T},20,12)+({B},1)  output={tuple(out.shape)}")
    assert out.shape == (B, T, 4)
    assert (out >= 0).all() and (out <= 1).all(), "outputs must be in [0,1]"

    # test single-step inference
    p1 = torch.randn(1, WINDOW_LEN, N_PRESSURE)
    t1 = torch.randn(1, WINDOW_LEN, N_TORQUE)
    f1 = torch.tensor([[0.5]])
    out1, h1 = model.step(p1, t1, f1, None)
    print(f"Inference step:    output={tuple(out1.shape)}  values={out1.detach().numpy()}")

    # test inference wrapper
    inf = HapticNetInference(model)
    inf.set_fragility(0.3)
    for i in range(25):
        r = inf.step([0.1]*6, [1.0]*12)
    print(f"Wrapper output:    {r}")
    print(f"Warmed up:         {inf.warmed_up}")

    # test mock stream
    mock = MockSensorStream()
    for i in range(100):
        d = mock.step()
    print(f"Mock stream:       pressure={d['pressure'][:3]}...  slip={d['slip_label']}")

    # benchmark inference speed
    import time
    inf.reset()
    t0 = time.time()
    N = 2000
    for _ in range(N):
        inf.step([0.1]*6, [1.0]*12)
    elapsed = time.time() - t0
    hz = N / elapsed
    print(f"Benchmark:         {hz:.0f} Hz ({elapsed/N*1000:.2f} ms/step)")

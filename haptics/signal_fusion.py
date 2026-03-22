"""
haptics/signal_fusion.py — Per-finger haptic fusion with HapticNet + GPT-4V context.

Pipeline (50 Hz):
    1. HapticNet predicts shared: freq, wave, slip_prob, duty_raw (overall intensity)
    2. Per-finger duty from Weber-Fechner scaling of individual pressure sensors
    3. Apply fragility cap from GPT-4V scene analysis
    4. Slip override: if slip_prob > threshold → urgent buzz on all fingers
    5. Send per-finger (freq, duty_i, wave) to HapticController

Finger mapping (pressure index → LRA):
    pressure[0] (pinky)      → "pinky"  (addr=4)
    pressure[1] (ring)       → "ring"   (addr=3)
    pressure[2] (middle)     → "middle" (addr=2)
    pressure[3] (index)      → "index"  (addr=1)
    pressure[4] (thumb_bend) → "thumb"  (addr=0)  — combined with pressure[5]
    pressure[5] (thumb_rot)  → merged into thumb

Usage:
    fusion = SignalFusion(haptic_controller, hapticnet_inference)
    fusion.set_scene(scene_dict)              # from SceneSeeder
    cmd = fusion.step(pressure_6, torque_12)  # every 20ms
    # cmd = {"freq": 5, "duties": [12, 18, 15, 10, 8], "wave": 0, "slip": False, ...}
"""

import sys
import os
import threading
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from haptics.presets import FINGER_ADDRS, ANCHORS, CONTACT_EVENTS

# ── Config ───────────────────────────────────────────────────────────────────

SLIP_THRESHOLD  = 0.6    # slip_prob above this triggers warning
SLIP_DUTY_BOOST = 10     # extra duty during slip
SLIP_FREQ       = 7      # maximum urgency
SLIP_WAVE       = 0      # square = more alarming

DEFAULT_FRAG_CAP = 22    # duty cap when no VLM result
CONTACT_THRESH   = 0.05  # minimum pressure to consider "in contact"

# Weber-Fechner: perceived intensity = k * ln(1 + pressure / p0)
# Mapped so that pressure ~5.0 → duty_raw ~1.0
WEBER_K  = 0.55          # gain
WEBER_P0 = 0.3           # reference threshold (below this, near-zero output)

# Pressure index → finger name (Inspire DOF order)
PRESSURE_TO_FINGER = ["pinky", "ring", "middle", "index", "thumb"]
N_FINGERS = 5


def _weber_fechner(pressure: float) -> float:
    """Weber-Fechner psychophysical force-to-intensity mapping."""
    if pressure < 0.01:
        return 0.0
    raw = WEBER_K * np.log1p(pressure / WEBER_P0)
    return float(np.clip(raw, 0.0, 1.0))


class SignalFusion:
    """
    Per-finger haptic fusion: HapticNet provides shared material properties
    (freq, wave, slip), while each finger gets independent duty based on
    its own pressure via Weber-Fechner scaling.

    Parameters
    ----------
    controller : HapticController
        Connected haptic output device.
    inference : HapticNetInference
        Loaded model wrapper with rolling window.
    """

    def __init__(self, controller, inference):
        self.controller = controller
        self.inference = inference
        self._lock = threading.Lock()

        # scene state (from GPT-4V)
        self._frag_cap    = DEFAULT_FRAG_CAP
        self._object_name = "unknown"
        self._preset_name = "default"

        # per-finger contact transient detection
        self._prev_pressure = np.zeros(N_FINGERS, dtype=np.float32)
        self._contact_active = np.zeros(N_FINGERS, dtype=bool)

    # ── scene context ────────────────────────────────────────────────────

    def set_scene(self, scene: dict, target_object: str = None):
        """
        Apply GPT-4V scene analysis.

        Args:
            scene: output from SceneSeeder.seed()
            target_object: which object to focus on (default: first)
        """
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
        frag = int(obj.get("fragility", 3))
        preset = obj.get("haptic_preset", "default")
        cap = scene.get("fragility_caps", {}).get(name, DEFAULT_FRAG_CAP)

        # convert fragility 1-5 to [0,1] for HapticNet
        frag_scalar = (frag - 1) / 4.0

        with self._lock:
            self._frag_cap = cap
            self._object_name = name
            self._preset_name = preset
            self.inference.set_fragility(frag_scalar)
            self.inference.reset()  # fresh LSTM for new object

        print(f"[Fusion] Scene → {name}  cap={cap}  preset={preset}  frag={frag_scalar:.2f}")

    # ── main step (50 Hz) ────────────────────────────────────────────────

    def step(self, pressure_6: list, torque_12: list) -> dict:
        """
        Process one sensor frame and send per-finger haptic commands to ESP32.

        Args:
            pressure_6:  list of 6 floats — finger pressure sensors
                         [pinky, ring, middle, index, thumb_bend, thumb_rot]
            torque_12:   list of 12 floats — arm joint torques

        Returns:
            dict with: freq, duties (5 ints), wave, slip, object, pressure, etc.
        """
        # --- HapticNet inference (shared material properties) ---
        net_out = self.inference.step(pressure_6, torque_12)
        freq     = net_out["freq"]       # int 0-7
        duty_raw = net_out["duty_raw"]   # float 0-1 (overall intensity hint)
        wave     = net_out["wave"]       # int 0 or 1
        slip     = net_out["slip"]       # float 0-1

        with self._lock:
            cap = self._frag_cap
            obj_name = self._object_name

        # --- per-finger pressure → duty via Weber-Fechner ---
        # Collapse 6 pressure channels to 5 fingers (thumb = max of bend + rot)
        finger_pressure = np.zeros(N_FINGERS, dtype=np.float32)
        finger_pressure[0] = float(pressure_6[0])  # pinky
        finger_pressure[1] = float(pressure_6[1])  # ring
        finger_pressure[2] = float(pressure_6[2])  # middle
        finger_pressure[3] = float(pressure_6[3])  # index
        finger_pressure[4] = max(float(pressure_6[4]), float(pressure_6[5]))  # thumb

        duties = np.zeros(N_FINGERS, dtype=np.int32)
        for i in range(N_FINGERS):
            weber = _weber_fechner(finger_pressure[i])
            # blend: 70% per-finger Weber, 30% HapticNet global duty_raw
            blended = 0.7 * weber + 0.3 * duty_raw
            duties[i] = int(round(blended * cap))
            duties[i] = max(0, min(31, duties[i]))

        # --- per-finger contact transient detection ---
        for i in range(N_FINGERS):
            d_p = finger_pressure[i] - self._prev_pressure[i]
            is_onset = (d_p > 0.8 and finger_pressure[i] > 0.5
                        and not self._contact_active[i])
            if is_onset:
                self._contact_active[i] = True
                addr = FINGER_ADDRS[PRESSURE_TO_FINGER[i]]
                self.controller.pulse(addr, 28, 6, 0, 25)

            if finger_pressure[i] < CONTACT_THRESH:
                self._contact_active[i] = False

        self._prev_pressure = finger_pressure.copy()

        # --- slip override: all fingers get urgent buzz ---
        slip_triggered = slip > SLIP_THRESHOLD
        if slip_triggered:
            freq = SLIP_FREQ
            wave = SLIP_WAVE
            for i in range(N_FINGERS):
                duties[i] = min(31, duties[i] + SLIP_DUTY_BOOST)

        # --- send per-finger commands to ESP32 ---
        any_active = False
        for i, finger_name in enumerate(PRESSURE_TO_FINGER):
            if duties[i] > 0:
                self.controller.send_finger(finger_name, duties[i], freq, wave)
                any_active = True
            else:
                # silence this finger
                self.controller.send_finger(finger_name, 0, 0, wave)

        if not any_active:
            self.controller.stop_all()

        # max duty for backward-compatible reporting
        duty_max = int(duties.max())

        return {
            "freq":      freq,
            "duty":      duty_max,
            "duties":    duties.tolist(),
            "wave":      wave,
            "slip":      slip_triggered,
            "object":    obj_name,
            "sent":      True,
            "duty_raw":  duty_raw,
            "slip_prob": slip,
            "pressure":  finger_pressure.tolist(),
        }

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def object_name(self) -> str:
        with self._lock:
            return self._object_name

    @property
    def fragility_cap(self) -> int:
        with self._lock:
            return self._frag_cap

    def stop(self):
        """Stop all haptic output."""
        self.controller.stop_all()


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from haptics.controller import HapticController
    from neural.hapticnet import HapticNetInference, HapticNet, MockSensorStream

    # mock everything
    ctrl = HapticController(mock=True)
    ctrl.connect()

    model = HapticNet()
    inf = HapticNetInference(model)

    fusion = SignalFusion(ctrl, inf)

    # simulate a scene
    fusion.set_scene({
        "objects": [{"name": "coffee mug", "fragility": 2, "haptic_preset": "rigid plastic"}],
        "fragility_caps": {"coffee mug": 28},
    })

    # run 50 frames
    mock = MockSensorStream()
    for i in range(50):
        d = mock.step()
        cmd = fusion.step(d["pressure"].tolist(), d["torque"].tolist())
        if i % 10 == 0:
            print(f"  t={i:3d}  freq={cmd['freq']}  duties={cmd['duties']}  "
                  f"wave={cmd['wave']}  slip={cmd['slip']}  "
                  f"duty_raw={cmd['duty_raw']:.3f}  slip_p={cmd['slip_prob']:.3f}")

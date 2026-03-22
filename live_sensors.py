"""
live_sensors.py — Live view of all haptic-relevant sensor data.

Subscribes directly to:
  rt/inspire_hand/state/{l,r}  — inspire_hand_state (force_act motor torques)
  rt/inspire_hand/touch/{l,r}  — inspire_hand_touch (tactile arrays → tip means)
  rt/lowstate                  — G1 arm joint torques

Usage:
    conda activate haptic
    python live_sensors.py
    python live_sensors.py enp131s0
"""

import sys, os, time, threading
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

_SDK = os.path.expanduser("~/GR00T-WholeBodyControl/inspire_hand_ws/inspire_hand_sdk")
_DDS = os.path.join(_SDK, "inspire_sdkpy")
for _p in [_SDK, _DDS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inspire_dds import inspire_hand_state, inspire_hand_touch
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# ── finger / joint config ─────────────────────────────────────────────────────
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
FINGER_SHORT  = ["Pk",   "Rg",  "Md",     "Ix",    "Tb",         "Tr"]

# tactile tip field names per finger (None = no tip sensor)
TIP_FIELDS = [
    "fingerone_tip_touch",   # pinky
    "fingertwo_tip_touch",   # ring
    "fingerthree_tip_touch", # middle
    "fingerfour_tip_touch",  # index
    "fingerfive_tip_touch",  # thumb_bend
    None,                    # thumb_rot
]

ARM_JOINTS = {
    "L_ShoPit": 15, "L_ShoRol": 16, "L_ShoYaw": 17,
    "L_Elbow":  18, "L_WriRol": 19, "L_WriPit": 20,
    "R_ShoPit": 22, "R_ShoRol": 23, "R_ShoYaw": 24,
    "R_Elbow":  25, "R_WriRol": 26, "R_WriPit": 27,
}


class LiveSensors:
    def __init__(self):
        self._lock = threading.Lock()
        self._contact  = {"l": [0.0]*6, "r": [0.0]*6}  # tactile tip means
        self._force_act = {"l": [0]*6,  "r": [0]*6}     # motor torques (int)
        self._torques  = {name: 0.0 for name in ARM_JOINTS}
        self._g1_fresh      = False
        self._inspire_fresh = {"l": False, "r": False}
        self._subs = []

    def connect(self, interface=None):
        print(f"DDS init ({interface or 'auto'})...", flush=True)
        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            ChannelFactoryInitialize(0)

        for side in ["l", "r"]:
            def make_state_cb(s):
                def cb(msg):
                    with self._lock:
                        self._force_act[s] = list(msg.force_act)
                        self._inspire_fresh[s] = True
                return cb

            def make_touch_cb(s):
                def cb(msg):
                    tips = []
                    for field in TIP_FIELDS:
                        if field:
                            tips.append(float(np.mean(getattr(msg, field))))
                        else:
                            tips.append(0.0)
                    with self._lock:
                        self._contact[s] = tips
                return cb

            sub_state = ChannelSubscriber(f"rt/inspire_hand/state/{side}", inspire_hand_state)
            sub_state.Init(make_state_cb(side), 10)
            self._subs.append(sub_state)

            sub_touch = ChannelSubscriber(f"rt/inspire_hand/touch/{side}", inspire_hand_touch)
            sub_touch.Init(make_touch_cb(side), 10)
            self._subs.append(sub_touch)

        sub_low = ChannelSubscriber("rt/lowstate", LowState_)
        sub_low.Init(self._on_lowstate, 10)
        self._subs.append(sub_low)

        print("Subscribed to inspire state/touch + g1 lowstate", flush=True)

    def _on_lowstate(self, msg: LowState_):
        with self._lock:
            for name, idx in ARM_JOINTS.items():
                self._torques[name] = msg.motor_state[idx].tau_est
            self._g1_fresh = True

    def snapshot(self):
        with self._lock:
            return {
                "contact":    {s: list(v) for s, v in self._contact.items()},
                "force_act":  {s: list(v) for s, v in self._force_act.items()},
                "torques":    dict(self._torques),
                "g1_fresh":   self._g1_fresh,
                "inspire_l":  self._inspire_fresh["l"],
                "inspire_r":  self._inspire_fresh["r"],
            }


def bar(val, max_val=5.0, width=10):
    filled = int(min(1.0, abs(float(val)) / max_val) * width)
    return "█" * filled + "·" * (width - filled)


def render(snap):
    lines = []
    lines.append(f"══ Live Sensors  {time.strftime('%H:%M:%S')} ══════════════════════")

    for side, label in [("l", "LEFT "), ("r", "RIGHT")]:
        fresh   = "✓" if snap[f"inspire_{side}"] else "✗ waiting..."
        contact = snap["contact"][side]
        force   = snap["force_act"][side]
        lines.append(f"\n  Inspire {label} [{fresh}]   peak={max(contact):.2f}")
        c_row = "  ".join(f"{n}={v:5.2f} {bar(v)}" for n, v in zip(FINGER_SHORT, contact))
        f_row = "  ".join(f"{n}={v:+5d}" for n, v in zip(FINGER_SHORT, force))
        lines.append(f"    contact : {c_row}")
        lines.append(f"    force_act: {f_row}")

    torques = snap["torques"]
    fresh   = "✓" if snap["g1_fresh"] else "✗ waiting..."
    lines.append(f"\n  G1 lowstate [{fresh}]")
    joints = list(ARM_JOINTS.keys())
    l_str = "  ".join(f"{j[2:]}={torques[j]:+5.2f}" for j in joints[:6])
    r_str = "  ".join(f"{j[2:]}={torques[j]:+5.2f}" for j in joints[6:])
    lines.append(f"    L arm: {l_str}")
    lines.append(f"    R arm: {r_str}")

    # HapticNet input summary (normalized)
    TACTILE_MAX = 100.0
    pressure_6_raw  = snap["contact"]["r"]
    pressure_6_norm = [min(1.0, v / TACTILE_MAX) for v in pressure_6_raw]
    torque_12  = [torques[j] for j in joints]
    lines.append(f"\n  ── HapticNet inputs (normalized) ──")
    lines.append(f"    pressure[6] raw : " + "  ".join(f"{v:6.2f}" for v in pressure_6_raw))
    lines.append(f"    pressure[6] /100: " + "  ".join(f"{v:6.3f}" for v in pressure_6_norm))
    lines.append(f"    torque[12]      : " + "  ".join(f"{v:+.2f}" for v in torque_12))

    return "\n".join(lines)


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else None
    sensors = LiveSensors()
    sensors.connect(interface)
    time.sleep(0.3)
    print("\nStreaming at 10 Hz (Ctrl+C to stop)\n")

    try:
        while True:
            t0 = time.time()
            print("\033[H\033[J", end="")
            print(render(sensors.snapshot()))
            elapsed = time.time() - t0
            time.sleep(max(0, 0.1 - elapsed))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()

"""
robot/inspire_hand_sensors.py
------------------------------
Thread-safe DDS subscriber for Inspire RH56DFTP hand sensors.

Subscribes to:
    rt/inspire_hand/state/l   →  inspire_hand_state  (pos, angle, force, current)
    rt/inspire_hand/state/r   →  inspire_hand_state
    rt/inspire_hand/touch/l   →  inspire_hand_touch  (tactile arrays, optional)
    rt/inspire_hand/touch/r   →  inspire_hand_touch

DOF order (6):  pinky | ring | middle | index | thumb_bend | thumb_rot

Usage:
    sensors = InspireHandSensors()
    sensors.connect()                        # blocks until first data or timeout
    lf = sensors.left.force_act             # list[6] of int16 motor forces
    rc = sensors.right.contact_pct         # list[6] floats in [0, 100]
    both = sensors.both_max_force()         # float — peak force across all fingers
"""

import sys, os, time, threading
import numpy as np

# ── locate inspire_dds IDL types ─────────────────────────────────────────────
# We import inspire_dds directly (bypassing inspire_sdkpy/__init__.py) to avoid
# pulling in inspire_sdk.py which requires pymodbus (Modbus TCP, not needed here).
_GROOT    = os.path.expanduser("~/GR00T-WholeBodyControl/inspire_hand_ws/inspire_hand_sdk")
_SDK_PKG  = os.path.join(_GROOT, "inspire_sdkpy")   # contains inspire_dds/
for _p in [_GROOT, _SDK_PKG]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inspire_dds import inspire_hand_state, inspire_hand_touch
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# ── constants ─────────────────────────────────────────────────────────────────
DOF_NAMES   = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
NUM_DOF     = 6
ANGLE_SCALE = 10.0   # Inspire angle units → pct (1000 full-open → 100.0 pct)


class _HandState:
    """State snapshot for one hand (left or right)."""

    def __init__(self, side: str):
        self.side = side
        self._lock = threading.Lock()
        self._pos_act:   list = [0] * NUM_DOF
        self._angle_act: list = [0] * NUM_DOF
        self._force_act: list = [0] * NUM_DOF
        self._current:   list = [0] * NUM_DOF
        self._touch_raw: dict = {}    # tactile tip arrays if subscribed
        self._palm_raw:  list = []    # tactile palm arrays if subscribed
        self._stamp: float = 0.0

    # ── DDS callbacks ──────────────────────────────────────────────────────
    def on_state(self, msg: inspire_hand_state):
        with self._lock:
            self._pos_act   = list(msg.pos_act)
            self._angle_act = list(msg.angle_act)
            self._force_act = list(msg.force_act)
            self._current   = list(msg.current)
            self._stamp     = time.time()

    # tip field names in DOF order: pinky, ring, middle, index, thumb, (thumb_rot=None)
    _TIP_FIELDS = [
        "fingerone_tip_touch",   # pinky   — int16[9]
        "fingertwo_tip_touch",   # ring    — int16[9]
        "fingerthree_tip_touch", # middle  — int16[9]
        "fingerfour_tip_touch",  # index   — int16[9]
        "fingerfive_tip_touch",  # thumb   — int16[9]
        None,                    # thumb_rot — no tip sensor
    ]

    # palm (inner finger surface) — fires on ANY grip, not just fingertip
    _PALM_FIELDS = [
        "fingerone_palm_touch",   # pinky   — int16[80]
        "fingertwo_palm_touch",   # ring    — int16[80]
        "fingerthree_palm_touch", # middle  — int16[80]
        "fingerfour_palm_touch",  # index   — int16[80]
        "fingerfive_palm_touch",  # thumb   — int16[96]
        None,                     # thumb_rot
    ]

    def on_touch(self, msg: inspire_hand_touch):
        tip_means = []
        palm_means = []
        for tip_f, palm_f in zip(self._TIP_FIELDS, self._PALM_FIELDS):
            tip_means.append(float(np.mean(getattr(msg, tip_f))) if tip_f else 0.0)
            palm_means.append(float(np.mean(getattr(msg, palm_f))) if palm_f else 0.0)
        with self._lock:
            self._touch_raw = tip_means    # list[6] tip means
            self._palm_raw  = palm_means   # list[6] palm means

    # ── public API ─────────────────────────────────────────────────────────
    @property
    def force_act(self) -> list:
        """Raw motor force for each DOF (int16, signed)."""
        with self._lock:
            return list(self._force_act)

    @property
    def angle_act(self) -> list:
        """Actual finger angle in Inspire units [0, 1000]."""
        with self._lock:
            return list(self._angle_act)

    @property
    def tip_pressure(self) -> list:
        """Per-finger tactile tip mean [pinky,ring,middle,index,thumb_bend,thumb_rot].
        This is the REAL contact signal. Requires subscribe_touch=True."""
        with self._lock:
            return list(self._touch_raw) if self._touch_raw else [0.0] * NUM_DOF

    @property
    def palm_pressure(self) -> list:
        """Per-finger palm (inner surface) tactile mean. Fires on ANY grip."""
        with self._lock:
            return list(self._palm_raw) if self._palm_raw else [0.0] * NUM_DOF

    @property
    def pressure(self) -> list:
        """Best available pressure per finger: max(tip, palm).
        This gives signal from ALL fingers involved in a grasp."""
        with self._lock:
            tips = list(self._touch_raw) if self._touch_raw else [0.0] * NUM_DOF
            palms = list(self._palm_raw) if self._palm_raw else [0.0] * NUM_DOF
        return [max(t, p) for t, p in zip(tips, palms)]

    @property
    def contact_pct(self) -> list:
        """Finger angle as open% [0=closed, 100=fully open]. NOT a contact signal."""
        with self._lock:
            return [round(a / ANGLE_SCALE, 1) for a in self._angle_act]

    @property
    def max_force(self) -> float:
        """Peak absolute motor torque across all 6 DOFs (not pressure)."""
        with self._lock:
            return float(max(abs(f) for f in self._force_act)) if self._force_act else 0.0

    @property
    def total_force(self) -> float:
        """Sum of absolute motor torques across all 6 DOFs."""
        with self._lock:
            return float(sum(abs(f) for f in self._force_act))

    @property
    def touch(self) -> list:
        """Per-finger tactile tip means (alias for tip_pressure)."""
        with self._lock:
            return list(self._touch_raw) if self._touch_raw else [0.0] * NUM_DOF

    @property
    def is_fresh(self) -> bool:
        """True if data arrived within the last 0.5s."""
        return (time.time() - self._stamp) < 0.5

    def as_dict(self) -> dict:
        """Full snapshot as a plain dict (for logging/fusion)."""
        with self._lock:
            return {
                "side":        self.side,
                "contact_pct": [round(a / ANGLE_SCALE, 1) for a in self._angle_act],
                "force_act":   list(self._force_act),
                "pos_act":     list(self._pos_act),
                "current":     list(self._current),
            }

    def print_state(self):
        p = self.tip_pressure
        f = self.force_act
        side = self.side[0].upper()
        print(f"{side} tip_pres: " +
              "  ".join(f"{n}={v:5.2f}" for n, v in zip(DOF_NAMES, p)))
        print(f"{side} torque:   " +
              "  ".join(f"{n}={v:4d}" for n, v in zip(DOF_NAMES, f)))


class InspireHandSensors:
    """
    Dual-hand sensor reader for Inspire RH56DFTP via DDS.

    Parameters
    ----------
    network_interface : str or None
        Network interface for DDS (e.g. 'enp131s0').  None = auto.
    subscribe_touch : bool
        Whether to also subscribe to high-density tactile arrays.
        These are large messages (~2 kB each) at ~10 Hz per hand.
    """

    def __init__(self, network_interface=None, subscribe_touch: bool = False):
        self.interface      = network_interface
        self._sub_touch     = subscribe_touch
        self.left           = _HandState("left")
        self.right          = _HandState("right")
        self._subs: list    = []

    # ── connection ────────────────────────────────────────────────────────
    def connect(self, timeout: float = 10.0, init_dds: bool = True) -> bool:
        """Subscribe to DDS topics.  Blocks until first data or *timeout* seconds."""
        if init_dds:
            print("[InspireHand] DDS init...", flush=True)
            if self.interface:
                ChannelFactoryInitialize(0, self.interface)
            else:
                ChannelFactoryInitialize(0)

        for side, hand in [("l", self.left), ("r", self.right)]:
            sub = ChannelSubscriber(f"rt/inspire_hand/state/{side}", inspire_hand_state)
            sub.Init(hand.on_state, 10)
            self._subs.append(sub)

            if self._sub_touch:
                tsub = ChannelSubscriber(f"rt/inspire_hand/touch/{side}", inspire_hand_touch)
                tsub.Init(hand.on_touch, 10)
                self._subs.append(tsub)

        print("[InspireHand] Subscribed — waiting for data...", flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.left.is_fresh or self.right.is_fresh:
                sides = []
                if self.left.is_fresh:  sides.append("left")
                if self.right.is_fresh: sides.append("right")
                print(f"[InspireHand] Connected ({', '.join(sides)})")
                return True
            time.sleep(0.1)

        print("[InspireHand] Timeout — no data received")
        return False

    # ── convenience accessors ─────────────────────────────────────────────
    def both_max_force(self) -> float:
        """Peak absolute force across both hands."""
        return max(self.left.max_force, self.right.max_force)

    def both_total_force(self) -> float:
        """Sum of absolute forces across both hands."""
        return self.left.total_force + self.right.total_force

    def per_finger_force(self) -> dict:
        """Dict with 'left' and 'right' force lists, each length 6."""
        return {
            "left":  self.left.force_act,
            "right": self.right.force_act,
        }

    def snapshot(self) -> dict:
        """Full snapshot of both hands."""
        return {
            "left":  self.left.as_dict(),
            "right": self.right.as_dict(),
        }

    def print_state(self):
        self.left.print_state()
        self.right.print_state()


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    sensors = InspireHandSensors(network_interface=iface, subscribe_touch=False)
    if not sensors.connect():
        print("No data — is the Inspire hand driver running?")
        sys.exit(1)

    print(f"\nStreaming Inspire hand sensor data (Ctrl+C to stop):\n")
    try:
        while True:
            sensors.print_state()
            print(f"  peak_force={sensors.both_max_force():.0f}  "
                  f"total={sensors.both_total_force():.0f}")
            print()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped")

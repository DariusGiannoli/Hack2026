"""
haptic_direct.py — Direct pressure-to-vibration mapping.

Reads right hand tactile tip pressure via DDS, maps each finger
to its LRA actuator, and sends haptic commands at 50 Hz.

No neural net — just Weber-Fechner psychophysical scaling.

Usage:
    python haptic_direct.py --iface enp131s0
    python haptic_direct.py --mock-esp              # no glove
    python haptic_direct.py --mock-esp --mock-hand  # no robot either
"""

import sys, os, time, math, argparse, threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Inspire DDS imports ──────────────────────────────────────────────────────
_SDK = os.path.expanduser("~/GR00T-WholeBodyControl/inspire_hand_ws/inspire_hand_sdk")
_DDS = os.path.join(_SDK, "inspire_sdkpy")
for _p in [_SDK, _DDS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from haptics.controller import HapticController

# ── Config ───────────────────────────────────────────────────────────────────

SAMPLE_HZ    = 50
TACTILE_MAX  = 100.0

# Inspire DOF order: [pinky, ring, middle, index, thumb_bend, thumb_rot]
# LRA finger order:  [thumb, index, middle, ring, pinky]
TIP_FIELDS = [
    "fingerone_tip_touch",   # pinky    — int16[9]
    "fingertwo_tip_touch",   # ring     — int16[9]
    "fingerthree_tip_touch", # middle   — int16[9]
    "fingerfour_tip_touch",  # index    — int16[9]
    "fingerfive_tip_touch",  # thumb    — int16[9]
    None,                    # thumb_rot → merged into thumb
]

# Palm (inner finger surface) — fires on ANY grip, much broader coverage
PALM_FIELDS = [
    "fingerone_palm_touch",   # pinky    — int16[80]
    "fingertwo_palm_touch",   # ring     — int16[80]
    "fingerthree_palm_touch", # middle   — int16[80]
    "fingerfour_palm_touch",  # index    — int16[80]
    "fingerfive_palm_touch",  # thumb    — int16[96]
    None,                     # thumb_rot
]

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb"]

# Weber-Fechner parameters
WF_K  = 0.55   # scaling constant
WF_P0 = 0.3    # reference pressure (avoids log(0))
MAX_DUTY = 31
BASE_FREQ = 3
BASE_WAVE = 1   # sine

# Dead zone: ignore pressure below this (filters sensor noise at rest)
DEAD_ZONE = 0.08  # normalized [0,1] — tune up if fingers still buzz at rest


def pressure_to_duty(p: float) -> int:
    """Weber-Fechner: duty = k * ln(1 + p/p0), scaled to [0, 31]."""
    if p < DEAD_ZONE:
        return 0
    # subtract dead zone so vibration starts from zero at threshold
    p_adj = p - DEAD_ZONE
    raw = WF_K * math.log(1.0 + p_adj / WF_P0)
    return max(0, min(MAX_DUTY, int(raw * MAX_DUTY)))


# ── Mock hand (for testing without robot) ────────────────────────────────────

class MockHand:
    """Generates fake tactile data with slow sine waves per finger."""
    def __init__(self):
        self._t0 = time.time()

    def tip_pressure(self) -> list:
        t = time.time() - self._t0
        return [
            max(0, 40 * math.sin(0.3 * t + i * 0.7) + 20 + np.random.normal(0, 2))
            for i in range(6)
        ]


# ── Real hand reader ─────────────────────────────────────────────────────────

FORCE_ACT_MAX = 600.0  # Inspire motor force_act range → normalized raw value


class RealHandReader:
    """Subscribes to rt/inspire_hand/touch/r + state/r for right hand."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tips = [0.0] * 6
        self._palms = [0.0] * 6
        self._force = [0] * 6
        self._fresh = False

    def connect(self, interface=None):
        from inspire_dds import inspire_hand_touch, inspire_hand_state
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            ChannelFactoryInitialize(0)

        sub_touch = ChannelSubscriber("rt/inspire_hand/touch/r", inspire_hand_touch)
        sub_touch.Init(self._on_touch, 10)
        self._sub_touch = sub_touch

        sub_state = ChannelSubscriber("rt/inspire_hand/state/r", inspire_hand_state)
        sub_state.Init(self._on_state, 10)
        self._sub_state = sub_state

        print("[direct] Subscribed to rt/inspire_hand/{touch,state}/r — waiting...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._fresh:
                print("[direct] Receiving hand data")
                return True
            time.sleep(0.1)
        print("[direct] Timeout — no data (is Headless_driver running?)")
        return False

    def _on_touch(self, msg):
        tips = []
        palms = []
        for tip_f, palm_f in zip(TIP_FIELDS, PALM_FIELDS):
            tips.append(float(np.mean(getattr(msg, tip_f))) if tip_f else 0.0)
            palms.append(float(np.mean(getattr(msg, palm_f))) if palm_f else 0.0)
        with self._lock:
            self._tips = tips
            self._palms = palms
            self._fresh = True

    def _on_state(self, msg):
        with self._lock:
            self._force = list(msg.force_act)
            self._fresh = True

    def tip_pressure(self) -> list:
        """Best pressure per finger: max(tip, palm), fallback to force_act."""
        with self._lock:
            tips = list(self._tips)
            palms = list(self._palms)
            force = list(self._force)

        # blend: use whichever tactile surface reads higher
        blended = [max(t, p) for t, p in zip(tips, palms)]

        # last resort: where tactile reads ~0 but motor is working, use force
        for i in range(min(len(blended), len(force))):
            if blended[i] < 2.0 and abs(force[i]) > 10:
                blended[i] = min(TACTILE_MAX, abs(float(force[i])) / FORCE_ACT_MAX * TACTILE_MAX)

        return blended


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Direct pressure → haptic mapping")
    parser.add_argument("--iface", default="enp131s0", help="DDS network interface")
    parser.add_argument("--mock-esp", action="store_true", help="Mock the haptic glove")
    parser.add_argument("--mock-hand", action="store_true", help="Mock the Inspire hand")
    parser.add_argument("--port", default=None, help="Serial port for ESP32")
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE,
                        help=f"Pressure dead zone 0-1 (default: {DEAD_ZONE})")
    args = parser.parse_args()

    global DEAD_ZONE
    DEAD_ZONE = args.dead_zone

    # sensor source
    if args.mock_hand:
        print("[direct] Using MOCK hand")
        hand = MockHand()
    else:
        hand = RealHandReader()
        if not hand.connect(args.iface):
            print("[direct] Falling back to mock hand")
            hand = MockHand()

    # haptic output
    hc = HapticController(mock=args.mock_esp)
    if not hc.connect(port=args.port):
        print("[direct] ESP32 not found — switching to mock")
        hc = HapticController(mock=True)
        hc.connect()

    print(f"\n{'='*60}")
    print("  Direct Haptic Mapping — Right Hand → Glove")
    print(f"  {SAMPLE_HZ} Hz | Weber-Fechner scaling | dead_zone={DEAD_ZONE:.2f}")
    print(f"  Ctrl+C to stop")
    print(f"{'='*60}\n")

    interval = 1.0 / SAMPLE_HZ
    n = 0

    try:
        while True:
            t0 = time.time()

            raw_tips = hand.tip_pressure()  # 6 values

            # normalize to [0, 1]
            norm = [max(0.0, min(1.0, v / TACTILE_MAX)) for v in raw_tips]

            # collapse 6 DOF → 5 fingers (thumb = max of thumb_bend, thumb_rot)
            pressure_5 = [
                norm[0],                      # pinky
                norm[1],                      # ring
                norm[2],                      # middle
                norm[3],                      # index
                max(norm[4], norm[5]),         # thumb
            ]

            # compute duty per finger
            duties = [pressure_to_duty(p) for p in pressure_5]

            # send to each LRA
            for finger, duty in zip(FINGER_NAMES, duties):
                hc.send_finger(finger, duty, BASE_FREQ, BASE_WAVE)

            # status every 1s
            n += 1
            if n % SAMPLE_HZ == 0:
                bars = "  ".join(
                    f"{name[:2]}={duties[i]:2d} {'█' * (duties[i] // 3)}{'·' * (10 - duties[i] // 3)}"
                    for i, name in enumerate(FINGER_NAMES)
                )
                raw_str = "  ".join(f"{v:5.1f}" for v in raw_tips[:5])
                print(f"  raw=[{raw_str}]  {bars}")

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    except KeyboardInterrupt:
        print("\nStopping...")
        hc.stop_all()
        hc.disconnect()
        print("Done")


if __name__ == "__main__":
    main()

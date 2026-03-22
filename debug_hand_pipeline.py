#!/usr/bin/env python3
"""
debug_hand_pipeline.py — Debug the full hand command pipeline, level by level.

Pipeline:
  Level 1: rt/inspire_hand/{finger}/{l,r}  FloatMsg [0..1]     ← teleop sends these
  Level 2: hand_teleop_bridge.py converts → inspire_hand_ctrl  ← bridge publishes these
  Level 3: rt/inspire_hand/ctrl/{l,r}      inspire_hand_ctrl   ← driver receives these
  Level 4: Headless_driver / ModbusDataHandler → Modbus → hand  ← physical hand
  Level 5: rt/inspire_hand/state/{l,r}     inspire_hand_state  ← hand feedback

This script subscribes to ALL levels at once so you can see where data stops flowing.

Usage:
    python3 debug_hand_pipeline.py                  # auto interface
    python3 debug_hand_pipeline.py eth0             # specify interface
    python3 debug_hand_pipeline.py --raw            # one-line per message
"""

import sys
import os
import time
import threading
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
_GROOT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "darius_PC", "GR00T-WholeBodyControl"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "darius_PC", "inspire_hand_ws"),
    "/root/Projects/GR00T-WholeBodyControl",
    os.path.expanduser("~/GR00T-WholeBodyControl"),
]
for _c in _GROOT_CANDIDATES:
    _sdk = os.path.join(_c, "external_dependencies", "unitree_sdk2_python")
    if os.path.isdir(_sdk) and _sdk not in sys.path:
        sys.path.insert(0, _sdk)
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

# Also add inspire_hand_sdk
_inspire_sdk_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "darius_PC", "inspire_hand_ws", "inspire_hand_sdk"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "darius_PC", "GR00T-WholeBodyControl", "inspire_hand_ws", "inspire_hand_sdk"),
]
for _c in _inspire_sdk_candidates:
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

import cyclonedds.idl as idl
from dataclasses import dataclass

@dataclass
class FloatMsg(idl.IdlStruct, typename="FloatMsg"):
    value: float

# ── Constants ──────────────────────────────────────────────────────────────────
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
SAFE_MIN = 200
SAFE_MAX = 800

class C:
    RESET = "\033[0m";   BOLD = "\033[1m"
    CYAN = "\033[96m";   GREEN = "\033[92m"
    YELLOW = "\033[93m"; RED = "\033[91m"
    MAGENTA = "\033[95m"; GREY = "\033[90m"
    WHITE = "\033[97m";  BLUE = "\033[94m"


def _bar(val, vmin, vmax, width=12, color=C.GREEN):
    ratio = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))
    filled = int(ratio * width)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


class HandPipelineDebugger:
    def __init__(self, interface=None, raw_mode=False):
        self._iface = interface
        self._raw = raw_mode
        self._lock = threading.Lock()

        # ── Level 1: per-finger FloatMsg (teleop → bridge) ──
        # {(finger_name, side): (value, timestamp)}
        self._l1_fingers = {}
        self._l1_count = 0
        self._l1_times = []

        # ── Level 2: grasp FloatMsg (pico_streamer grasp topics) ──
        self._l2_grasp = {}  # {side: (value, timestamp)}
        self._l2_count = 0
        self._l2_times = []

        # ── Level 3: inspire_hand_ctrl (bridge → driver) ──
        self._l3_ctrl = {}  # {side: (angle_set_list, mode, timestamp)}
        self._l3_count = 0
        self._l3_times = []

        # ── Level 4: inspire_hand_state (driver feedback) ──
        self._l4_state = {}  # {side: (state_dict, timestamp)}
        self._l4_count = 0
        self._l4_times = []

        # ── Level 5: inspire_hand_touch (tactile) ──
        self._l5_touch = {}
        self._l5_count = 0
        self._l5_times = []

    def connect(self):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelSubscriber,
                ChannelFactoryInitialize,
            )
        except ImportError as e:
            print(f"{C.RED}ERROR: unitree_sdk2py not found: {e}{C.RESET}")
            sys.exit(1)

        # Try importing inspire types
        try:
            from inspire_sdkpy.inspire_dds import inspire_hand_ctrl, inspire_hand_state, inspire_hand_touch
            has_inspire = True
        except ImportError:
            print(f"{C.YELLOW}WARNING: inspire_sdkpy not found — levels 3-5 disabled{C.RESET}")
            has_inspire = False
            inspire_hand_ctrl = None
            inspire_hand_state = None
            inspire_hand_touch = None

        print(f"{C.CYAN}[DDS] Initializing on interface: {self._iface or 'auto'}{C.RESET}")
        if self._iface:
            ChannelFactoryInitialize(0, self._iface)
        else:
            ChannelFactoryInitialize(0)
        print(f"{C.GREEN}[DDS] Init OK{C.RESET}")

        # ── Level 1: per-finger FloatMsg topics ──
        for name in FINGER_NAMES:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/{name}/{side}"
                sub = ChannelSubscriber(topic, FloatMsg)
                cb = self._make_l1_cb(name, side)
                sub.Init(cb, 10)
                # prevent GC
                setattr(self, f"_sub_l1_{name}_{side}", sub)
        print(f"{C.GREEN}[L1] Subscribed to rt/inspire_hand/{{finger}}/{{l,r}} — 12 FloatMsg topics{C.RESET}")

        # ── Level 2: grasp FloatMsg topics ──
        for side in ("l", "r"):
            topic = f"rt/inspire_hand/grasp/{side}"
            sub = ChannelSubscriber(topic, FloatMsg)
            cb = self._make_l2_cb(side)
            sub.Init(cb, 10)
            setattr(self, f"_sub_l2_grasp_{side}", sub)
        print(f"{C.GREEN}[L2] Subscribed to rt/inspire_hand/grasp/{{l,r}} — 2 grasp topics{C.RESET}")

        # ── Level 3: inspire_hand_ctrl (output of hand_teleop_bridge) ──
        if has_inspire and inspire_hand_ctrl is not None:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/ctrl/{side}"
                sub = ChannelSubscriber(topic, inspire_hand_ctrl)
                cb = self._make_l3_cb(side)
                sub.Init(cb, 10)
                setattr(self, f"_sub_l3_ctrl_{side}", sub)
            print(f"{C.GREEN}[L3] Subscribed to rt/inspire_hand/ctrl/{{l,r}} — inspire_hand_ctrl{C.RESET}")

        # ── Level 4: inspire_hand_state (feedback from Headless_driver) ──
        if has_inspire and inspire_hand_state is not None:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/state/{side}"
                sub = ChannelSubscriber(topic, inspire_hand_state)
                cb = self._make_l4_cb(side)
                sub.Init(cb, 10)
                setattr(self, f"_sub_l4_state_{side}", sub)
            print(f"{C.GREEN}[L4] Subscribed to rt/inspire_hand/state/{{l,r}} — hand feedback{C.RESET}")

        # ── Level 5: inspire_hand_touch (tactile) ──
        if has_inspire and inspire_hand_touch is not None:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/touch/{side}"
                sub = ChannelSubscriber(topic, inspire_hand_touch)
                cb = self._make_l5_cb(side)
                sub.Init(cb, 10)
                setattr(self, f"_sub_l5_touch_{side}", sub)
            print(f"{C.GREEN}[L5] Subscribed to rt/inspire_hand/touch/{{l,r}} — tactile{C.RESET}")

        print(f"\n{C.YELLOW}Waiting for data... Ctrl+C to quit{C.RESET}\n")

    # ── Callback factories ────────────────────────────────────────────────────

    def _update_hz(self, times_list):
        now = time.time()
        times_list.append(now)
        if len(times_list) > 100:
            del times_list[:50]
        if len(times_list) >= 2:
            dt = times_list[-1] - times_list[0]
            return (len(times_list) - 1) / dt if dt > 0 else 0.0
        return 0.0

    def _make_l1_cb(self, name, side):
        def cb(msg):
            with self._lock:
                self._l1_fingers[(name, side)] = (msg.value, time.time())
                self._l1_count += 1
                self._update_hz(self._l1_times)
            if self._raw:
                print(f"[L1 FloatMsg] {side}/{name} = {msg.value:.4f}")
        return cb

    def _make_l2_cb(self, side):
        def cb(msg):
            with self._lock:
                self._l2_grasp[side] = (msg.value, time.time())
                self._l2_count += 1
                self._update_hz(self._l2_times)
            if self._raw:
                print(f"[L2 Grasp] {side} = {msg.value:.4f}")
        return cb

    def _make_l3_cb(self, side):
        def cb(msg):
            with self._lock:
                self._l3_ctrl[side] = (list(msg.angle_set), msg.mode, time.time())
                self._l3_count += 1
                self._update_hz(self._l3_times)
            if self._raw:
                print(f"[L3 ctrl] {side} mode={msg.mode:#06b} angles={list(msg.angle_set)}")
        return cb

    def _make_l4_cb(self, side):
        def cb(msg):
            with self._lock:
                self._l4_state[side] = ({
                    "angle_act": list(msg.angle_act) if hasattr(msg, 'angle_act') else None,
                    "pos_act": list(msg.pos_act) if hasattr(msg, 'pos_act') else None,
                    "force_act": list(msg.force_act) if hasattr(msg, 'force_act') else None,
                    "current": list(msg.current) if hasattr(msg, 'current') else None,
                }, time.time())
                self._l4_count += 1
                self._update_hz(self._l4_times)
            if self._raw:
                a = list(msg.angle_act) if hasattr(msg, 'angle_act') else "?"
                print(f"[L4 state] {side} angle_act={a}")
        return cb

    def _make_l5_cb(self, side):
        def cb(msg):
            with self._lock:
                self._l5_touch[side] = (True, time.time())
                self._l5_count += 1
                self._update_hz(self._l5_times)
        return cb

    # ── Hz helper ─────────────────────────────────────────────────────────────

    def _get_hz(self, times_list):
        if len(times_list) >= 2:
            dt = times_list[-1] - times_list[0]
            return (len(times_list) - 1) / dt if dt > 0 else 0.0
        return 0.0

    # ── TUI ───────────────────────────────────────────────────────────────────

    def print_tui(self):
        with self._lock:
            l1_fingers = dict(self._l1_fingers)
            l1_count = self._l1_count
            l1_hz = self._get_hz(self._l1_times)

            l2_grasp = dict(self._l2_grasp)
            l2_count = self._l2_count
            l2_hz = self._get_hz(self._l2_times)

            l3_ctrl = dict(self._l3_ctrl)
            l3_count = self._l3_count
            l3_hz = self._get_hz(self._l3_times)

            l4_state = dict(self._l4_state)
            l4_count = self._l4_count
            l4_hz = self._get_hz(self._l4_times)

            l5_count = self._l5_count
            l5_hz = self._get_hz(self._l5_times)

        ts = time.strftime("%H:%M:%S")
        now = time.time()
        print("\033[2J\033[H", end="")

        # ── Header ────────────────────────────────────────────────────────
        print(f"{C.BOLD}{C.CYAN}╔══ HAND PIPELINE DEBUG — {ts} ═══════════════════════════════╗{C.RESET}")
        print()

        # ── Status summary ────────────────────────────────────────────────
        def status_line(label, count, hz, color_ok=C.GREEN):
            ok = count > 0
            s = f"{color_ok}RECEIVING{C.RESET}" if ok else f"{C.RED}NO DATA{C.RESET}"
            return f"  {label:42s} {s}  msgs={C.WHITE}{count:>7}{C.RESET}  hz={C.YELLOW}{hz:6.1f}{C.RESET}"

        print(status_line("L1  FloatMsg  (teleop → bridge)        ", l1_count, l1_hz))
        print(status_line("L2  Grasp     (pico grasp topics)      ", l2_count, l2_hz))
        print(status_line("L3  ctrl      (bridge → driver)        ", l3_count, l3_hz, C.GREEN))
        print(status_line("L4  state     (driver → feedback)      ", l4_count, l4_hz, C.BLUE))
        print(status_line("L5  touch     (tactile sensor)         ", l5_count, l5_hz, C.MAGENTA))

        # ── Diagnostic ────────────────────────────────────────────────────
        print()
        if l1_count > 0 and l3_count == 0:
            print(f"  {C.RED}{C.BOLD}⚠  L1 has data but L3 is empty!{C.RESET}")
            print(f"     → {C.YELLOW}hand_teleop_bridge.py is NOT running (or not publishing){C.RESET}")
            print(f"     → Run: python hand_teleop_bridge.py [interface]")
            print(f"     → File: darius_PC/inspire_hand_ws/inspire_hand_sdk/example/hand_teleop_bridge.py")
            print()
        elif l3_count > 0 and l4_count == 0:
            print(f"  {C.RED}{C.BOLD}⚠  L3 has ctrl commands but L4 has no feedback!{C.RESET}")
            print(f"     → {C.YELLOW}Headless_driver_double.py is NOT running{C.RESET}")
            print(f"     → Or hands are not connected (Modbus TCP to 192.168.123.210/211)")
            print(f"     → Run: python Headless_driver_double.py [interface]")
            print()
        elif l1_count > 0 and l3_count > 0 and l4_count > 0:
            print(f"  {C.GREEN}{C.BOLD}✓  Full pipeline is flowing: teleop → bridge → driver → hand{C.RESET}")
            print()
        elif l1_count == 0 and l2_count == 0:
            print(f"  {C.RED}{C.BOLD}⚠  No finger data at all{C.RESET}")
            print(f"     → Edgard's teleop is not sending finger commands")
            print(f"     → Or DDS interface mismatch (current: {self._iface or 'auto'})")
            print()

        # ── L1: Per-finger FloatMsg detail ────────────────────────────────
        print(f"{C.BOLD}{C.GREEN}── L1: Per-Finger FloatMsg (teleop output) ──────────────────{C.RESET}")
        if not l1_fingers:
            print(f"  {C.GREY}No data yet...{C.RESET}")
        else:
            for side, side_label, color in [("l", "LEFT ", C.CYAN), ("r", "RIGHT", C.MAGENTA)]:
                parts = []
                for name in FINGER_NAMES:
                    key = (name, side)
                    if key in l1_fingers:
                        v, t = l1_fingers[key]
                        age = now - t
                        stale = " " if age < 0.5 else f"{C.GREY}!"
                        inspire = int(SAFE_MAX + v * (SAFE_MIN - SAFE_MAX))
                        bar = _bar(v, 0, 1, width=6)
                        parts.append(f"{name[:5]}={v:.2f}→{inspire:3d}{stale}{C.RESET}")
                    else:
                        parts.append(f"{name[:5]}={C.GREY}---{C.RESET}")
                print(f"  {color}{side_label}{C.RESET} {' '.join(parts)}")

        # ── L2: Grasp ────────────────────────────────────────────────────
        print()
        print(f"{C.BOLD}{C.YELLOW}── L2: Grasp Topics (pico_streamer) ─────────────────────────{C.RESET}")
        if not l2_grasp:
            print(f"  {C.GREY}No grasp data (not using pico or grasp topics){C.RESET}")
        else:
            for side in ("l", "r"):
                if side in l2_grasp:
                    v, t = l2_grasp[side]
                    age = now - t
                    stale = "" if age < 0.5 else f" {C.GREY}(stale {age:.1f}s)"
                    print(f"  {side.upper()}: {v:.3f} → inspire={int(SAFE_MAX + v * (SAFE_MIN - SAFE_MAX))}{stale}{C.RESET}")

        # ── L3: inspire_hand_ctrl (bridge output → driver input) ──────────
        print()
        print(f"{C.BOLD}{C.BLUE}── L3: inspire_hand_ctrl (bridge → driver) ─────────────────{C.RESET}")
        if not l3_ctrl:
            print(f"  {C.GREY}No ctrl messages — hand_teleop_bridge.py not running?{C.RESET}")
        else:
            for side in ("l", "r"):
                if side in l3_ctrl:
                    angles, mode, t = l3_ctrl[side]
                    age = now - t
                    stale = "" if age < 0.5 else f" {C.GREY}(stale {age:.1f}s)"
                    color = C.CYAN if side == "l" else C.MAGENTA
                    label = "LEFT " if side == "l" else "RIGHT"
                    angle_str = " ".join(f"{a:4d}" for a in angles[:6])
                    print(f"  {color}{label}{C.RESET} mode={mode:#06b} angles=[{angle_str}]{stale}{C.RESET}")
                    # Visual bar for each finger
                    for i, name in enumerate(FINGER_NAMES):
                        if i < len(angles):
                            a = angles[i]
                            bar = _bar(a, SAFE_MIN, SAFE_MAX, width=15, color=color)
                            pct = (SAFE_MAX - a) / (SAFE_MAX - SAFE_MIN) * 100  # 0%=open, 100%=closed
                            print(f"    {name:<12} {a:4d}  {bar} {pct:5.1f}% closed")

        # ── L4: inspire_hand_state (driver feedback from physical hand) ───
        print()
        print(f"{C.BOLD}{C.MAGENTA}── L4: Hand State (feedback from physical hand) ─────────────{C.RESET}")
        if not l4_state:
            print(f"  {C.GREY}No state feedback — Headless_driver not running or hand offline{C.RESET}")
        else:
            for side in ("l", "r"):
                if side in l4_state:
                    state, t = l4_state[side]
                    age = now - t
                    stale = "" if age < 0.5 else f" {C.GREY}(stale {age:.1f}s)"
                    color = C.CYAN if side == "l" else C.MAGENTA
                    label = "LEFT " if side == "l" else "RIGHT"

                    angle_act = state.get("angle_act")
                    force_act = state.get("force_act")

                    if angle_act:
                        angle_str = " ".join(f"{a:4d}" if a is not None else " ???" for a in angle_act[:6])
                        print(f"  {color}{label}{C.RESET} angle_act=[{angle_str}]{stale}{C.RESET}")

                        # Compare with L3 command if available
                        if side in l3_ctrl:
                            cmd_angles = l3_ctrl[side][0]
                            errors = []
                            for i, name in enumerate(FINGER_NAMES):
                                if i < len(angle_act) and i < len(cmd_angles) and angle_act[i] is not None:
                                    err = abs(cmd_angles[i] - angle_act[i])
                                    err_color = C.RED if err > 100 else (C.YELLOW if err > 30 else C.GREEN)
                                    errors.append(f"{name[:5]}:{err_color}{err:3d}{C.RESET}")
                            if errors:
                                print(f"    cmd-actual error: {' '.join(errors)}")

                    if force_act:
                        force_str = " ".join(f"{f:4d}" if f is not None else " ???" for f in force_act[:6])
                        print(f"    force_act=[{force_str}]")

        print(f"\n{C.GREY}[Ctrl+C to quit | refresh 5 Hz]{C.RESET}")

    def run(self, hz=5.0):
        if self._raw:
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}Stopped.{C.RESET}")
        else:
            try:
                while True:
                    self.print_tui()
                    time.sleep(1.0 / hz)
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}Stopped.{C.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug the full hand pipeline: teleop → bridge → driver → hand"
    )
    parser.add_argument("interface", nargs="?", default=None,
                        help="Network interface (eth0, enp3s0, etc.)")
    parser.add_argument("--raw", action="store_true",
                        help="Raw mode: one line per message")
    parser.add_argument("--hz", type=float, default=5.0,
                        help="TUI refresh rate (default: 5)")
    args = parser.parse_args()

    dbg = HandPipelineDebugger(interface=args.interface, raw_mode=args.raw)
    dbg.connect()
    dbg.run(hz=args.hz)


if __name__ == "__main__":
    main()

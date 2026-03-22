#!/usr/bin/env python3
"""
debug_teleop_receiver.py — See what teleop commands ARRIVE on the robot via DDS.

This subscribes to the topics that precision_bridge.py PUBLISHES:
  rt/lowcmd                       → full 29-DOF joint commands (arms + legs)
  rt/inspire_hand/{finger}/{l,r}  → per-finger FloatMsg
  rt/inspire_hand/ctrl/{l,r}      → inspire_hand_ctrl (if available)

Run this ON THE ROBOT (or on any PC on the same DDS domain) to verify that
Edgard's teleop data actually arrives.

Usage:
    python3 debug_teleop_receiver.py                    # auto-detect interface
    python3 debug_teleop_receiver.py eth0               # specify interface
    python3 debug_teleop_receiver.py --arms-only        # show only arm joints 15-28
    python3 debug_teleop_receiver.py --fingers-only     # show only finger topics
    python3 debug_teleop_receiver.py --raw              # one-line-per-message (no TUI)
"""

import sys
import os
import time
import threading
import argparse
import math

# ── Try to add GR00T paths for unitree_sdk2py ─────────────────────────────────
_GROOT_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "darius_PC", "GR00T-WholeBodyControl"),
    "/root/Projects/GR00T-WholeBodyControl",
    os.path.expanduser("~/GR00T-WholeBodyControl"),
]
for _c in _GROOT_CANDIDATES:
    _sdk = os.path.join(_c, "external_dependencies", "unitree_sdk2_python")
    if os.path.isdir(_sdk) and _sdk not in sys.path:
        sys.path.insert(0, _sdk)
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

import numpy as np

try:
    import cyclonedds.idl as idl
    from dataclasses import dataclass

    @dataclass
    class FloatMsg(idl.IdlStruct, typename="FloatMsg"):
        value: float
except ImportError:
    FloatMsg = None

# ── Constants ──────────────────────────────────────────────────────────────────
ARM_JOINTS = {
    15: "L_ShPitch",   16: "L_ShRoll",   17: "L_ShYaw",
    18: "L_Elbow",     19: "L_WrRoll",   20: "L_WrPitch",  21: "L_WrYaw",
    22: "R_ShPitch",   23: "R_ShRoll",   24: "R_ShYaw",
    25: "R_Elbow",     26: "R_WrRoll",   27: "R_WrPitch",  28: "R_WrYaw",
}

LEG_JOINTS = {
    0: "L_HipYaw",    1: "L_HipRoll",   2: "L_HipPitch",
    3: "L_Knee",      4: "L_AnkPitch",  5: "L_AnkRoll",
    6: "R_HipYaw",    7: "R_HipRoll",   8: "R_HipPitch",
    9: "R_Knee",      10: "R_AnkPitch", 11: "R_AnkRoll",
    12: "WaistYaw",   13: "WaistRoll",  14: "WaistPitch",
}

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

# ── ANSI ───────────────────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m";   BOLD = "\033[1m"
    CYAN = "\033[96m";   GREEN = "\033[92m"
    YELLOW = "\033[93m"; RED = "\033[91m"
    MAGENTA = "\033[95m"; GREY = "\033[90m"
    WHITE = "\033[97m";  BLUE = "\033[94m"


class TeleopReceiver:
    def __init__(self, interface=None, raw_mode=False, arms_only=False, fingers_only=False):
        self._iface = interface
        self._raw = raw_mode
        self._arms_only = arms_only
        self._fingers_only = fingers_only
        self._lock = threading.Lock()

        # ── State ──
        self._lowcmd = None
        self._lowcmd_count = 0
        self._lowcmd_hz = 0.0
        self._lowcmd_times = []
        self._lowcmd_arm_q = np.zeros(14)  # joints 15-28

        self._lowstate = None
        self._lowstate_count = 0
        self._lowstate_hz = 0.0
        self._lowstate_times = []

        # Finger values: {(finger_name, side): (value, timestamp)}
        self._fingers = {}
        self._finger_count = 0
        self._finger_hz = 0.0
        self._finger_times = []

    def connect(self):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelSubscriber,
                ChannelFactoryInitialize,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as LowCmd,
                LowState_ as LowState,
            )
        except ImportError as e:
            print(f"{C.RED}ERROR: unitree_sdk2py not found: {e}{C.RESET}")
            print("  Install it or set the path to GR00T-WholeBodyControl")
            sys.exit(1)

        print(f"{C.CYAN}[DDS] Initializing on interface: {self._iface or 'auto'}{C.RESET}")
        if self._iface:
            ChannelFactoryInitialize(0, self._iface)
        else:
            ChannelFactoryInitialize(0)
        print(f"{C.GREEN}[DDS] Init OK{C.RESET}")

        # ── Subscribe to rt/lowcmd (what teleop SENDS) ──
        if not self._fingers_only:
            self._cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd)
            self._cmd_sub.Init(self._on_lowcmd, 10)
            print(f"{C.GREEN}[DDS] Subscribed to rt/lowcmd (teleop commands){C.RESET}")

            # Also subscribe to rt/lowstate (robot feedback) for comparison
            self._state_sub = ChannelSubscriber("rt/lowstate", LowState)
            self._state_sub.Init(self._on_lowstate, 10)
            print(f"{C.GREEN}[DDS] Subscribed to rt/lowstate (robot feedback){C.RESET}")

        # ── Subscribe to finger topics ──
        if not self._arms_only and FloatMsg is not None:
            for name in FINGER_NAMES:
                for side in ("l", "r"):
                    topic = f"rt/inspire_hand/{name}/{side}"
                    sub = ChannelSubscriber(topic, FloatMsg)
                    # Use a factory to capture name/side in closure
                    cb = self._make_finger_cb(name, side)
                    sub.Init(cb, 10)
            print(f"{C.GREEN}[DDS] Subscribed to rt/inspire_hand/{{finger}}/{{l,r}} (12 topics){C.RESET}")

        print(f"\n{C.YELLOW}Waiting for teleop data... (Ctrl+C to quit){C.RESET}\n")

    def _make_finger_cb(self, name, side):
        def cb(msg):
            now = time.time()
            with self._lock:
                self._fingers[(name, side)] = (msg.value, now)
                self._finger_count += 1
                self._finger_times.append(now)
                if len(self._finger_times) > 100:
                    self._finger_times = self._finger_times[-50:]
                if len(self._finger_times) >= 2:
                    dt = self._finger_times[-1] - self._finger_times[0]
                    self._finger_hz = (len(self._finger_times) - 1) / dt if dt > 0 else 0.0
            if self._raw:
                print(f"[FINGER] {side}/{name} = {msg.value:.4f}")
        return cb

    def _on_lowcmd(self, msg):
        now = time.time()
        with self._lock:
            self._lowcmd = msg
            self._lowcmd_count += 1
            self._lowcmd_times.append(now)
            if len(self._lowcmd_times) > 100:
                self._lowcmd_times = self._lowcmd_times[-50:]
            if len(self._lowcmd_times) >= 2:
                dt = self._lowcmd_times[-1] - self._lowcmd_times[0]
                self._lowcmd_hz = (len(self._lowcmd_times) - 1) / dt if dt > 0 else 0.0
            # Cache arm positions
            for i in range(14):
                self._lowcmd_arm_q[i] = msg.motor_cmd[15 + i].q
        if self._raw:
            arms = " ".join(f"{msg.motor_cmd[15+i].q:+.3f}" for i in range(14))
            print(f"[LOWCMD] #{self._lowcmd_count} arms=[{arms}]")

    def _on_lowstate(self, msg):
        now = time.time()
        with self._lock:
            self._lowstate = msg
            self._lowstate_count += 1
            self._lowstate_times.append(now)
            if len(self._lowstate_times) > 100:
                self._lowstate_times = self._lowstate_times[-50:]
            if len(self._lowstate_times) >= 2:
                dt = self._lowstate_times[-1] - self._lowstate_times[0]
                self._lowstate_hz = (len(self._lowstate_times) - 1) / dt if dt > 0 else 0.0

    def _snapshot(self):
        with self._lock:
            return {
                "lowcmd": self._lowcmd,
                "lowcmd_count": self._lowcmd_count,
                "lowcmd_hz": self._lowcmd_hz,
                "lowstate": self._lowstate,
                "lowstate_count": self._lowstate_count,
                "lowstate_hz": self._lowstate_hz,
                "fingers": dict(self._fingers),
                "finger_count": self._finger_count,
                "finger_hz": self._finger_hz,
            }

    def print_tui(self):
        s = self._snapshot()
        ts = time.strftime("%H:%M:%S")
        print("\033[2J\033[H", end="")  # clear

        # ── Header ─────────────────────────────────────────────────────────
        print(f"{C.BOLD}{C.CYAN}╔══ TELEOP RECEIVER DEBUG — {ts} ══╗{C.RESET}")
        print()

        # ── Connection status ──────────────────────────────────────────────
        cmd_ok = s["lowcmd_count"] > 0
        state_ok = s["lowstate_count"] > 0
        fing_ok = s["finger_count"] > 0

        print(f"  rt/lowcmd   (teleop→robot): "
              f"{C.GREEN + 'RECEIVING' if cmd_ok else C.RED + 'NO DATA'}{C.RESET}  "
              f"msgs={C.WHITE}{s['lowcmd_count']:>6}{C.RESET}  "
              f"hz={C.YELLOW}{s['lowcmd_hz']:5.1f}{C.RESET}")
        print(f"  rt/lowstate (robot→teleop): "
              f"{C.GREEN + 'RECEIVING' if state_ok else C.RED + 'NO DATA'}{C.RESET}  "
              f"msgs={C.WHITE}{s['lowstate_count']:>6}{C.RESET}  "
              f"hz={C.YELLOW}{s['lowstate_hz']:5.1f}{C.RESET}")
        print(f"  rt/inspire_hand (fingers) : "
              f"{C.GREEN + 'RECEIVING' if fing_ok else C.RED + 'NO DATA'}{C.RESET}  "
              f"msgs={C.WHITE}{s['finger_count']:>6}{C.RESET}  "
              f"hz={C.YELLOW}{s['finger_hz']:5.1f}{C.RESET}")

        # ── Diagnostic ────────────────────────────────────────────────────
        print()
        if not cmd_ok and not state_ok:
            print(f"  {C.RED}{C.BOLD}⚠  NO DDS DATA AT ALL — check:{C.RESET}")
            print(f"     1. Is Edgard's teleop running? (teleop_edgard_new_setup.py)")
            print(f"     2. Are both PCs on the same network/subnet?")
            print(f"     3. Is the DDS interface correct? (current: {self._iface or 'auto'})")
            print(f"     4. Firewall blocking UDP multicast?")
            print()
        elif cmd_ok and not state_ok:
            print(f"  {C.YELLOW}⚠  Getting commands but NO robot feedback{C.RESET}")
            print(f"     → Robot might be off or DDS domain mismatch")
            print()
        elif not cmd_ok and state_ok:
            print(f"  {C.YELLOW}⚠  Getting robot state but NO teleop commands{C.RESET}")
            print(f"     → Edgard's precision_bridge.py is not publishing?")
            print()

        # ── Arm commands (rt/lowcmd) vs actual (rt/lowstate) ──────────────
        if not self._fingers_only:
            print(f"{C.BOLD}{C.BLUE}── ARM JOINTS: Command vs Actual ─────────────────────────────{C.RESET}")
            cmd = s["lowcmd"]
            state = s["lowstate"]
            if cmd is None:
                print(f"  {C.GREY}Waiting for rt/lowcmd...{C.RESET}")
            else:
                header = f"  {'Joint':<14} {'CMD q':>9} {'CMD kp':>8} {'CMD kd':>8}"
                if state is not None:
                    header += f" {'ACTUAL q':>10} {'ERROR':>8}"
                print(f"{C.GREY}{header}{C.RESET}")
                print(f"  {C.GREY}{'─' * 65}{C.RESET}")

                for idx, name in sorted(ARM_JOINTS.items()):
                    mc = cmd.motor_cmd[idx]
                    q_cmd = mc.q
                    kp = mc.kp
                    kd = mc.kd

                    color = C.CYAN if name.startswith("L_") else C.MAGENTA
                    line = f"  {color}{name:<14}{C.RESET} {q_cmd:>+9.4f} {kp:>8.1f} {kd:>8.2f}"

                    if state is not None:
                        q_act = state.motor_state[idx].q
                        err = abs(q_cmd - q_act)
                        err_color = C.RED if err > 0.1 else (C.YELLOW if err > 0.03 else C.GREEN)
                        line += f" {q_act:>+10.4f} {err_color}{err:>8.4f}{C.RESET}"

                    print(line)

                # Also show leg/waist summary (are they moving?)
                if not self._arms_only:
                    print()
                    print(f"  {C.GREY}Legs/Waist (0-14): ", end="")
                    leg_parts = []
                    for idx in range(15):
                        q = cmd.motor_cmd[idx].q
                        if abs(q) > 0.01:
                            n = LEG_JOINTS.get(idx, f"j{idx}")
                            leg_parts.append(f"{n}={q:+.2f}")
                    if leg_parts:
                        print(" ".join(leg_parts[:8]))
                        if len(leg_parts) > 8:
                            print(f"           {' '.join(leg_parts[8:])}")
                    else:
                        print("all ~0 (standing)")
                    print(C.RESET, end="")

        # ── Finger values ─────────────────────────────────────────────────
        if not self._arms_only:
            print()
            print(f"{C.BOLD}{C.GREEN}── FINGER COMMANDS (rt/inspire_hand) ────────────────────────{C.RESET}")
            fingers = s["fingers"]
            if not fingers:
                print(f"  {C.GREY}Waiting for finger data...{C.RESET}")
            else:
                for side, side_label, color in [("l", "LEFT", C.CYAN), ("r", "RIGHT", C.MAGENTA)]:
                    vals = []
                    for name in FINGER_NAMES:
                        key = (name, side)
                        if key in fingers:
                            v, t = fingers[key]
                            age = time.time() - t
                            stale = age > 1.0
                            val_str = f"{v:.3f}"
                            if stale:
                                val_str = f"{C.GREY}{val_str}(stale){C.RESET}"
                            vals.append(f"{name}={val_str}")
                        else:
                            vals.append(f"{name}={C.GREY}---{C.RESET}")
                    print(f"  {color}{side_label}:{C.RESET} {' '.join(vals)}")

        print(f"\n{C.GREY}[Ctrl+C to quit | refresh 5 Hz]{C.RESET}")

    def run(self, hz=5.0):
        if self._raw:
            # Raw mode: callbacks print directly, just block
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
        description="Debug: see what teleop data arrives on robot via DDS"
    )
    parser.add_argument(
        "interface", nargs="?", default=None,
        help="Network interface (e.g. eth0, enp3s0). Omit = auto."
    )
    parser.add_argument("--raw", action="store_true",
                        help="Raw mode: print one line per message (no TUI)")
    parser.add_argument("--arms-only", action="store_true",
                        help="Only show arm joint data (skip fingers)")
    parser.add_argument("--fingers-only", action="store_true",
                        help="Only show finger data (skip joints)")
    parser.add_argument("--hz", type=float, default=5.0,
                        help="TUI refresh rate (default: 5)")
    args = parser.parse_args()

    recv = TeleopReceiver(
        interface=args.interface,
        raw_mode=args.raw,
        arms_only=args.arms_only,
        fingers_only=args.fingers_only,
    )
    recv.connect()
    recv.run(hz=args.hz)


if __name__ == "__main__":
    main()

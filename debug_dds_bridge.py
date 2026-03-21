#!/usr/bin/env python3
"""
debug_dds_bridge.py — Moniteur en temps réel des topics DDS Unitree (ROS2 bridge).

Topics surveillés :
  rt/lowstate     → positions, vitesses, couples des articulations du bras + IMU
  rt/hand_state   → capteurs de pression des doigts

Usage :
    python3 debug_dds_bridge.py [interface_réseau]
    python3 debug_dds_bridge.py eth0
    python3 debug_dds_bridge.py enp3s0

Appuyer sur Ctrl+C pour arrêter.
"""

import sys
import time
import threading
import argparse
import json
import struct

# ─── Constantes indices articulations bras G1 ────────────────────────────────
LEFT_ARM_JOINTS = {
    "L_ShoulderPitch": 15,
    "L_ShoulderRoll":  16,
    "L_ShoulderYaw":   17,
    "L_Elbow":         18,
    "L_WristRoll":     19,
    "L_WristPitch":    20,
    "L_WristYaw":      21,
}
RIGHT_ARM_JOINTS = {
    "R_ShoulderPitch": 22,
    "R_ShoulderRoll":  23,
    "R_ShoulderYaw":   24,
    "R_Elbow":         25,
    "R_WristRoll":     26,
    "R_WristPitch":    27,
    "R_WristYaw":      28,
}
ALL_ARM_JOINTS = {**LEFT_ARM_JOINTS, **RIGHT_ARM_JOINTS}

# ─── Noms des 17 joints upper-body en ordre IsaacLab ─────────────────────────
# Source : upper_body_joint_isaaclab_order_in_mujoco_index (policy_parameters.hpp)
UPPER_BODY_NAMES = [
    "Waist_Yaw",
    "Waist_Roll",
    "Waist_Pitch",
    "L_ShouldPitch",
    "R_ShouldPitch",
    "L_ShouldRoll",
    "R_ShouldRoll",
    "L_ShouldYaw",
    "R_ShouldYaw",
    "L_Elbow",
    "R_Elbow",
    "L_WristRoll",
    "R_WristRoll",
    "L_WristPitch",
    "R_WristPitch",
    "L_WristYaw",
    "R_WristYaw",
]

# ─── Couleurs ANSI ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"


def _bar(val, vmin, vmax, width=20, color=C.GREEN):
    """Barre de progression ASCII colorée."""
    ratio = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{C.RESET}"


class DDSMonitor:
    def __init__(self, interface: str | None = None, zmq_port: int = 5556, simple: bool = False):
        self._iface    = interface
        self._zmq_port = zmq_port
        self._simple   = simple
        self._lock     = threading.Lock()

        # ── état partagé ─────────────────────────────────────────────────────
        self._low_state  = None
        self._hand_state = None
        self._low_count  = 0
        self._hand_count = 0
        self._low_hz     = 0.0
        self._hand_hz    = 0.0
        self._last_low_t  = time.time()
        self._last_hand_t = time.time()

        # ── état ZMQ (bras téléop) ────────────────────────────────────────────
        self._zmq_upper_body = None   # np.ndarray (17,) en ordre IsaacLab
        self._zmq_count      = 0
        self._zmq_hz         = 0.0
        self._zmq_times      = []

        # historique pour mesurer Hz
        self._low_times  = []
        self._hand_times = []

    # ─── Thread ZMQ subscriber ───────────────────────────────────────────────
    def _zmq_listener(self):
        """Reçoit les messages 'planner' du téléop et extrait upper_body_position."""
        try:
            import zmq
            import numpy as np
        except ImportError:
            print(f"{C.YELLOW}[ZMQ] pyzmq/numpy introuvable — section bras téléop désactivée{C.RESET}")
            return

        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://localhost:{self._zmq_port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"planner")
        sub.setsockopt(zmq.RCVTIMEO, 200)   # timeout 200 ms
        print(f"{C.GREEN}[ZMQ] Abonné à tcp://localhost:{self._zmq_port} (topic=planner){C.RESET}")


        while True:
            try:
                raw = sub.recv()
            except zmq.Again:
                continue
            except Exception:
                break

            try:
                # topic + header (taille variable : 1024 ou 1280) + data
                topic_end = raw.index(b'{')
                # fin du JSON = premier byte nul
                try:
                    null_end = raw.index(b'\x00', topic_end)
                except ValueError:
                    null_end = len(raw)
                header = json.loads(raw[topic_end:null_end].decode('utf-8'))
                # détecte la taille de header (padding tout-nul)
                header_end = null_end + 1
                for _hs in (1024, 1280, 512, 2048):
                    _end = topic_end + _hs
                    if _end <= len(raw) and all(b == 0 for b in raw[null_end:_end]):
                        header_end = _end
                        break
                payload = raw[header_end:]

                # calcule l'offset du champ upper_body_position
                offset = 0
                upper_body = None
                for field in header.get("fields", []):
                    fname = field["name"]
                    shape = field["shape"]
                    n     = 1
                    for s in shape:
                        n *= s
                    if field["dtype"] == "f32":
                        nbytes = n * 4
                    elif field["dtype"] == "i32":
                        nbytes = n * 4
                    else:
                        nbytes = n

                    if fname == "upper_body_position":
                        import numpy as np
                        upper_body = np.frombuffer(
                            payload[offset: offset + nbytes], dtype="<f4"
                        ).astype(float).copy()
                        break
                    offset += nbytes

                if upper_body is not None and len(upper_body) == 17:
                    now = time.time()
                    with self._lock:
                        self._zmq_upper_body = upper_body
                        self._zmq_count += 1
                        self._zmq_times.append(now)
                        if len(self._zmq_times) > 50:
                            self._zmq_times.pop(0)
                        if len(self._zmq_times) >= 2:
                            dt = self._zmq_times[-1] - self._zmq_times[0]
                            self._zmq_hz = (len(self._zmq_times) - 1) / dt if dt > 0 else 0.0
                    if self._simple:
                        import math
                        parts = []
                        for i, name in enumerate(UPPER_BODY_NAMES):
                            if name.startswith("Waist"):
                                continue
                            parts.append(f"{name}={math.degrees(float(upper_body[i])):+7.2f}°")
                        print("  ".join(parts))
            except Exception:
                pass

        sub.close()
        ctx.term()

    # ─── Callbacks DDS ───────────────────────────────────────────────────────
    def _on_low_state(self, msg):
        now = time.time()
        with self._lock:
            self._low_state = msg
            self._low_count += 1
            self._low_times.append(now)
            if len(self._low_times) > 50:
                self._low_times.pop(0)
            if len(self._low_times) >= 2:
                dt = self._low_times[-1] - self._low_times[0]
                self._low_hz = (len(self._low_times) - 1) / dt if dt > 0 else 0.0

    def _on_hand_state(self, msg):
        now = time.time()
        with self._lock:
            self._hand_state = msg
            self._hand_count += 1
            self._hand_times.append(now)
            if len(self._hand_times) > 50:
                self._hand_times.pop(0)
            if len(self._hand_times) >= 2:
                dt = self._hand_times[-1] - self._hand_times[0]
                self._hand_hz = (len(self._hand_times) - 1) / dt if dt > 0 else 0.0

    # ─── Connexion DDS ───────────────────────────────────────────────────────
    def connect(self):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelSubscriber,
                ChannelFactoryInitialize,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_
        except ImportError as e:
            print(f"{C.RED}[ERREUR] unitree_sdk2py introuvable : {e}{C.RESET}")
            print("  → Installer avec : pip install unitree_sdk2py")
            sys.exit(1)

        print(f"{C.CYAN}[DDS] Initialisation sur interface : {self._iface or 'auto'}{C.RESET}")
        if self._iface:
            ChannelFactoryInitialize(0, self._iface)
        else:
            ChannelFactoryInitialize(0)
        print(f"{C.GREEN}[DDS] Init OK{C.RESET}")

        # abonnement rt/lowstate
        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 10)
        print(f"{C.GREEN}[DDS] Abonné à rt/lowstate{C.RESET}")

        # abonnement rt/hand_state
        self._hand_sub = ChannelSubscriber("rt/hand_state", HandState_)
        self._hand_sub.Init(self._on_hand_state, 10)
        print(f"{C.GREEN}[DDS] Abonné à rt/hand_state{C.RESET}")

        print(f"\n{C.YELLOW}En attente des premiers messages...{C.RESET}")

        # thread ZMQ (non-bloquant)
        t = threading.Thread(target=self._zmq_listener, daemon=True)
        t.start()

    # ─── Affichage ───────────────────────────────────────────────────────────
    def _snapshot(self):
        with self._lock:
            ls   = self._low_state
            hs   = self._hand_state
            lc   = self._low_count
            hc   = self._hand_count
            lhz  = self._low_hz
            hhz  = self._hand_hz
            zub  = self._zmq_upper_body.copy() if self._zmq_upper_body is not None else None
            zc   = self._zmq_count
            zhz  = self._zmq_hz
        return ls, hs, lc, hc, lhz, hhz, zub, zc, zhz

    def print_once(self):
        ls, hs, lc, hc, lhz, hhz, zub, zc, zhz = self._snapshot()
        ts = time.strftime("%H:%M:%S")

        print("\033[2J\033[H", end="")  # clear screen

        # ── Header ─────────────────────────────────────────────────────────
        print(f"{C.BOLD}{C.CYAN}╔══ DDS Bridge Monitor — {ts} ══╗{C.RESET}")
        print(f"  rt/lowstate   : {C.GREEN if lc > 0 else C.RED}"
              f"{'OK' if lc > 0 else 'AUCUN MSG'}{C.RESET}  "
              f"msgs={C.WHITE}{lc:>6}{C.RESET}  "
              f"hz={C.YELLOW}{lhz:5.1f}{C.RESET}")
        print(f"  rt/hand_state : {C.GREEN if hc > 0 else C.RED}"
              f"{'OK' if hc > 0 else 'AUCUN MSG'}{C.RESET}  "
              f"msgs={C.WHITE}{hc:>6}{C.RESET}  "
              f"hz={C.YELLOW}{hhz:5.1f}{C.RESET}")
        print(f"  ZMQ planner   : {C.GREEN if zc > 0 else C.RED}"
              f"{'OK' if zc > 0 else 'AUCUN MSG'}{C.RESET}  "
              f"msgs={C.WHITE}{zc:>6}{C.RESET}  "
              f"hz={C.YELLOW}{zhz:5.1f}{C.RESET}")
        print()

        # ── LowState : bras ────────────────────────────────────────────────
        print(f"{C.BOLD}{C.BLUE}── Articulations Bras (rt/lowstate) ─────────────────────────{C.RESET}")
        if ls is None:
            print(f"  {C.GREY}En attente de données...{C.RESET}")
        else:
            header = f"  {'Joint':<20} {'q (rad)':>10} {'dq (rad/s)':>12} {'tau_est (Nm)':>13}"
            print(f"{C.GREY}{header}{C.RESET}")
            print(f"  {C.GREY}{'─'*58}{C.RESET}")
            for name, idx in ALL_ARM_JOINTS.items():
                ms  = ls.motor_state[idx]
                q   = ms.q
                dq  = ms.dq
                tau = ms.tau_est
                sep = "│" if name.startswith("L_") else "│"
                color = C.CYAN if name.startswith("L_") else C.MAGENTA
                tau_color = C.RED if abs(tau) > 30 else C.WHITE
                bar = _bar(abs(tau), 0, 50, width=12,
                           color=C.RED if abs(tau) > 30 else C.GREEN)
                print(f"  {color}{name:<20}{C.RESET} "
                      f"{q:>+10.4f} "
                      f"{dq:>+12.4f} "
                      f"{tau_color}{tau:>+12.3f}{C.RESET}  {bar}")

            # poids estimé
            w_joints = [15, 16, 18, 22, 23, 25]
            weight = sum(abs(ls.motor_state[j].tau_est) for j in w_joints)
            print(f"\n  {C.BOLD}Proxy poids (épaules+coudes) :{C.RESET} "
                  f"{C.YELLOW}{weight:.2f} Nm{C.RESET}  "
                  f"{_bar(weight, 0, 100, width=20, color=C.YELLOW)}")

            # IMU
            print()
            print(f"{C.BOLD}{C.BLUE}── IMU (rt/lowstate) ─────────────────────────────────────────{C.RESET}")
            try:
                imu = ls.imu_state
                rpy = imu.rpy
                gyr = imu.gyroscope
                acc = imu.accelerometer
                print(f"  RPY  : R={C.WHITE}{rpy[0]:+.4f}{C.RESET}  "
                      f"P={C.WHITE}{rpy[1]:+.4f}{C.RESET}  "
                      f"Y={C.WHITE}{rpy[2]:+.4f}{C.RESET}  rad")
                print(f"  Gyro : x={C.WHITE}{gyr[0]:+.4f}{C.RESET}  "
                      f"y={C.WHITE}{gyr[1]:+.4f}{C.RESET}  "
                      f"z={C.WHITE}{gyr[2]:+.4f}{C.RESET}  rad/s")
                print(f"  Accel: x={C.WHITE}{acc[0]:+.4f}{C.RESET}  "
                      f"y={C.WHITE}{acc[1]:+.4f}{C.RESET}  "
                      f"z={C.WHITE}{acc[2]:+.4f}{C.RESET}  m/s²")
            except Exception as ex:
                print(f"  {C.GREY}IMU indisponible : {ex}{C.RESET}")

        # ── ZMQ : bras commandés par le téléop (upper_body_position) ──────────
        print()
        print(f"{C.BOLD}{C.GREEN}── Bras Commandés — ZMQ planner (ordre IsaacLab, 17 DOF) ────{C.RESET}")
        if zub is None:
            print(f"  {C.GREY}En attente du téléop (tcp://localhost:{self._zmq_port})...{C.RESET}")
        else:
            header = f"  {'Joint':<16} {'cmd (rad)':>10}  {'deg':>8}"
            print(f"{C.GREY}{header}{C.RESET}")
            print(f"  {C.GREY}{'─'*38}{C.RESET}")
            for i, name in enumerate(UPPER_BODY_NAMES):
                val_rad = float(zub[i])
                val_deg = val_rad * 57.2958
                # sépare visuellement waist / bras gauche / bras droit
                if i == 0:
                    print(f"  {C.GREY}  ·· Taille ··{C.RESET}")
                elif i == 3:
                    print(f"  {C.GREY}  ·· Bras (L=cyan  R=magenta) ··{C.RESET}")
                is_left  = name.startswith("L_")
                is_waist = name.startswith("Waist")
                color = C.CYAN if is_left else (C.MAGENTA if not is_waist else C.WHITE)
                bar = _bar(abs(val_rad), 0, 1.5, width=14,
                           color=C.RED if abs(val_rad) > 1.2 else color)
                print(f"  {color}{name:<16}{C.RESET} "
                      f"{val_rad:>+10.4f}  "
                      f"{val_deg:>+7.1f}°  {bar}")

        print(f"\n{C.GREY}[Ctrl+C pour quitter]{C.RESET}")

    def run(self, hz: float = 10.0):
        """Boucle d'affichage principale."""
        interval = 1.0 / hz
        try:
            while True:
                self.print_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Arrêt du moniteur.{C.RESET}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Moniteur DDS bridge Unitree G1 (rt/lowstate + rt/hand_state)"
    )
    parser.add_argument(
        "interface", nargs="?", default=None,
        help="Interface réseau (ex: eth0, enp3s0). Omis = auto-détection."
    )
    parser.add_argument(
        "--hz", type=float, default=5.0,
        help="Fréquence de rafraîchissement de l'affichage (défaut: 5 Hz)"
    )
    parser.add_argument(
        "--zmq-port", type=int, default=5556,
        help="Port ZMQ du publisher téléop (défaut: 5556)"
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Mode simple : affiche uniquement les positions bras envoyées au bridge (pas de TUI)"
    )
    args = parser.parse_args()

    monitor = DDSMonitor(interface=args.interface, zmq_port=args.zmq_port, simple=args.simple)

    if args.simple:
        # Mode simple : pas besoin de DDS, juste ZMQ
        print(f"[simple] Écoute ZMQ tcp://localhost:{args.zmq_port}  (Ctrl+C pour quitter)\n")
        import math
        header = "  ".join(f"{n:<20}" for n in UPPER_BODY_NAMES if not n.startswith("Waist"))
        print(header)
        print("-" * len(header))
        monitor._zmq_listener()   # bloque jusqu'à Ctrl+C
        return

    monitor.connect()

    # attendre un peu avant de commencer l'affichage
    time.sleep(1.0)
    monitor.run(hz=args.hz)


if __name__ == "__main__":
    main()

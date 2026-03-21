#!/usr/bin/env python3
"""
listen_bridge.py — Listener ZMQ pour PrecisionBridge.

Affiche en temps réel les messages envoyés par precision_bridge.py :
  - topic 'command'  : start/stop/planner mode
  - topic 'planner'  : upper_body_position 17 DOF (bras + taille)
  - topic 'pose'     : joint_pos 29 DOF (si utilisé)

Usage :
    python3 bridge/listen_bridge.py             # port 5556 (défaut)
    python3 bridge/listen_bridge.py --port 5557
    python3 bridge/listen_bridge.py --topic planner
    python3 bridge/listen_bridge.py --topic all
"""

import argparse
import json
import struct
import sys
import time

import numpy as np

# Tailles de header connues : gear_sonic=1280, test_zmq_manager=1024, _zmq_deploy_publisher=1024
_KNOWN_HEADER_SIZES = (1024, 1280, 512, 2048)

# ── ANSI ──────────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m";  BOLD    = "\033[1m"
    CYAN    = "\033[96m"; GREEN   = "\033[92m"
    YELLOW  = "\033[93m"; RED     = "\033[91m"
    MAGENTA = "\033[95m"; GREY    = "\033[90m"
    WHITE   = "\033[97m"; BLUE    = "\033[94m"

# ── Noms des 17 joints upper-body (ordre IsaacLab) ────────────────────────────
UPPER_BODY_NAMES = [
    "Waist_Yaw",    "Waist_Roll",   "Waist_Pitch",
    "L_ShPitch",    "R_ShPitch",
    "L_ShRoll",     "R_ShRoll",
    "L_ShYaw",      "R_ShYaw",
    "L_Elbow",      "R_Elbow",
    "L_WrRoll",     "R_WrRoll",
    "L_WrPitch",    "R_WrPitch",
    "L_WrYaw",      "R_WrYaw",
]

# ── Parsing du packed-message ZMQ ─────────────────────────────────────────────
def _parse_header(raw: bytes) -> tuple[str, dict, bytes]:
    """Retourne (topic_str, header_dict, payload_bytes).

    Auto-détecte la taille du header (1024 ou 1280 selon l'émetteur) en
    cherchant la fin du JSON null-terminé puis en vérifiant que le padding
    jusqu'au prochain candidat est bien composé de zéros.
    """
    # topic = préfixe ASCII avant le '{'
    brace = raw.index(b"{")
    topic = raw[:brace].decode("ascii", errors="replace").strip()

    # Fin du JSON = premier byte nul après l'accolade ouvrante
    try:
        null_end = raw.index(b"\x00", brace)
    except ValueError:
        null_end = len(raw)

    header_json_bytes = raw[brace:null_end]
    header = json.loads(header_json_bytes.decode("utf-8"))

    # Cherche la taille de header qui correspond à un padding tout-nul
    header_size = None
    for hsize in _KNOWN_HEADER_SIZES:
        end = brace + hsize
        if end > len(raw):
            continue
        padding = raw[null_end:end]
        if all(b == 0 for b in padding):
            header_size = hsize
            break

    if header_size is None:
        # Dernier recours : payload commence juste après le JSON nul
        payload = raw[null_end + 1:]
    else:
        payload = raw[brace + header_size:]

    return topic, header, payload


def _extract_fields(header: dict, payload: bytes) -> dict[str, np.ndarray]:
    """Décode les champs binaires en tableaux numpy."""
    fields = {}
    offset = 0
    dtype_map = {
        "f32": (np.float32, 4),
        "f64": (np.float64, 8),
        "i32": (np.int32,   4),
        "i64": (np.int64,   8),
        "u8":  (np.uint8,   1),
        "bool":(np.bool_,   1),
    }
    for f in header.get("fields", []):
        name  = f["name"]
        dtype_str = f["dtype"]
        shape = f["shape"]
        n = 1
        for s in shape:
            n *= s
        np_dtype, item_size = dtype_map.get(dtype_str, (np.float32, 4))
        nbytes = n * item_size
        chunk = payload[offset: offset + nbytes]
        arr = np.frombuffer(chunk, dtype=np.dtype(f"<{np_dtype().dtype.str[1:]}")).reshape(shape)
        fields[name] = arr
        offset += nbytes
    return fields


# ── Afficheurs par topic ──────────────────────────────────────────────────────
_msg_counts: dict[str, int] = {}
_last_print_t = 0.0
_hz_times: dict[str, list] = {}


def _hz(topic: str) -> float:
    times = _hz_times.setdefault(topic, [])
    times.append(time.time())
    if len(times) > 50:
        times.pop(0)
    if len(times) < 2:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


def _bar(val: float, vmin: float, vmax: float, width: int = 12) -> str:
    ratio = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))
    filled = int(ratio * width)
    color = C.RED if abs(val) > 1.2 else (C.CYAN if val >= 0 else C.MAGENTA)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def print_command(fields: dict, hz: float) -> None:
    start   = int(fields.get("start",   [0])[0])
    stop    = int(fields.get("stop",    [0])[0])
    planner = int(fields.get("planner", [0])[0])
    dh      = fields.get("delta_heading")
    mode_s  = "PLANNER" if planner else "STREAMED"
    status  = f"{C.GREEN}START{C.RESET}" if start else (f"{C.RED}STOP{C.RESET}" if stop else f"{C.GREY}NOP{C.RESET}")
    dh_s    = f"  Δheading={np.degrees(dh[0]):+.1f}°" if dh is not None else ""
    print(f"{C.BOLD}{C.YELLOW}[command]{C.RESET}  {status}  mode={C.CYAN}{mode_s}{C.RESET}{dh_s}"
          f"  {C.GREY}{hz:.1f} Hz{C.RESET}")


def print_planner(fields: dict, hz: float) -> None:
    print(f"\n{C.BOLD}{C.GREEN}[planner]{C.RESET}  {C.GREY}{hz:.1f} Hz{C.RESET}")

    ub = fields.get("upper_body_position")
    if ub is not None:
        ub = ub.flatten()
        print(f"  {C.BOLD}upper_body_position (17 DOF — IsaacLab order):{C.RESET}")
        # Ligne 1 : taille
        print(f"  {C.GREY}·· Taille ··{C.RESET}")
        for i in range(3):
            v = float(ub[i])
            print(f"    {C.WHITE}{UPPER_BODY_NAMES[i]:<14}{C.RESET}"
                  f"  {v:+.4f} rad  {np.degrees(v):+7.2f}°  {_bar(v, -1.5, 1.5)}")
        # Ligne 2 : bras (L/R entrelacés)
        print(f"  {C.GREY}·· Bras gauche (cyan) / droit (magenta) ··{C.RESET}")
        for i in range(3, 17):
            v = float(ub[i])
            is_l = (i % 2 == 1)   # indices 3,5,7,9,11,13,15 = L
            col  = C.CYAN if is_l else C.MAGENTA
            print(f"    {col}{UPPER_BODY_NAMES[i]:<14}{C.RESET}"
                  f"  {v:+.4f} rad  {np.degrees(v):+7.2f}°  {_bar(v, -1.5, 1.5)}")

    for key in ("left_hand_joints", "right_hand_joints"):
        arr = fields.get(key)
        if arr is not None:
            vals = " ".join(f"{v:+.3f}" for v in arr.flatten())
            label = "L_hand" if "left" in key else "R_hand"
            print(f"  {C.YELLOW}{label}{C.RESET}: [{vals}]")

    mode = fields.get("mode")
    if mode is not None:
        print(f"  mode={int(mode.flat[0])}  "
              f"movement={fields.get('movement', '?')}  "
              f"facing={fields.get('facing', '?')}")


def print_pose(fields: dict, hz: float) -> None:
    jp = fields.get("joint_pos")
    if jp is None:
        return
    jp = jp.flatten()
    print(f"\n{C.BOLD}{C.BLUE}[pose]{C.RESET}  {C.GREY}{hz:.1f} Hz{C.RESET}"
          f"  frame={fields.get('frame_index', ['?'])[0]}")
    arm_il = {
        "L_ShPitch": 15, "R_ShPitch": 16,
        "L_ShRoll":  17, "R_ShRoll":  18,
        "L_ShYaw":   19, "R_ShYaw":   20,
        "L_Elbow":   21, "R_Elbow":   22,
    }
    if len(jp) >= 29:
        parts = [f"{n}={np.degrees(jp[i]):+.1f}°" for n, i in arm_il.items()]
        print("  " + "  ".join(parts))


# ── Boucle principale ─────────────────────────────────────────────────────────
def run(port: int, topics: set[str]) -> None:
    try:
        import zmq
    except ImportError:
        print(f"{C.RED}pyzmq non installé : pip install pyzmq{C.RESET}")
        sys.exit(1)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{port}")

    for t in topics:
        sub.setsockopt(zmq.SUBSCRIBE, t.encode())
        print(f"{C.GREEN}[ZMQ] Abonné → tcp://localhost:{port}  topic={t!r}{C.RESET}")

    sub.setsockopt(zmq.RCVTIMEO, 500)
    print(f"{C.GREY}En attente de messages... (Ctrl+C pour quitter){C.RESET}\n")

    total = 0
    try:
        while True:
            try:
                raw = sub.recv()
            except zmq.Again:
                print(f"\r{C.GREY}[attente...]{C.RESET}", end="", flush=True)
                continue

            try:
                topic, header, payload = _parse_header(raw)
            except Exception as e:
                print(f"{C.RED}[parse error] {e}{C.RESET}")
                continue

            fields = _extract_fields(header, payload)
            hz     = _hz(topic)
            total += 1
            _msg_counts[topic] = _msg_counts.get(topic, 0) + 1

            ts = time.strftime("%H:%M:%S")
            print(f"\r{C.GREY}{ts}  #{total}{C.RESET}  ", end="")

            if topic == "command":
                print_command(fields, hz)
            elif topic == "planner":
                print_planner(fields, hz)
            elif topic == "pose":
                print_pose(fields, hz)
            else:
                print(f"{C.GREY}[{topic}] {list(fields.keys())}{C.RESET}")

    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}Arrêt — {total} messages reçus{C.RESET}")
        for t, n in _msg_counts.items():
            print(f"  {t:<12} : {n}")
    finally:
        sub.close()
        ctx.term()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Listener ZMQ PrecisionBridge")
    p.add_argument("--port",  type=int, default=5556,
                   help="Port ZMQ du publisher (défaut: 5556)")
    p.add_argument("--topic", type=str, default="all",
                   help="Topic à écouter : command | planner | pose | all (défaut: all)")
    args = p.parse_args()

    if args.topic == "all":
        topics = {"command", "planner", "pose"}
    else:
        topics = {args.topic}

    run(args.port, topics)


if __name__ == "__main__":
    main()

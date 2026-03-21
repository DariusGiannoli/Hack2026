"""
demo_teleop.py — Full haptic teleoperation pipeline.

Threads:
  T1 (main, 50 Hz)   : G1 torques → SignalFusionV2 → ESP32
  T2 (3 Hz)          : camera frame → DinoEncoder → MLP → fusion.update_from_dino()
  T3 (one-shot)      : camera frame → VLM → fusion.update_from_scene()
  T4 (background)    : camera UDP receiver (already running separately)

Run:
    # on laptop:
    python demo_teleop.py [--iface enp131s0] [--mock] [--no-vlm]

Args:
    --iface     network interface for G1 DDS  (default: enp131s0)
    --mock      mock ESP32 (no serial needed)
    --no-vlm    skip VLM scene seeding (use default caps)
    --port      ESP32 serial port
"""

import sys, os, time, threading, argparse
import numpy as np
import cv2

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── imports ───────────────────────────────────────────────────────────────────
from haptics.signal_fusion_v2      import SignalFusionV2
from haptics.haptic_controller     import HapticController
from haptics.perception.scene_seeder import SceneSeeder
from haptics.perception.dino_encoder import DinoEncoder
from haptics.neural.haptic_mlp      import load_model as load_mlp
from robot.g1_sensors               import G1Sensors

# ── config ────────────────────────────────────────────────────────────────────
CONTROL_HZ  = 50      # haptic output rate
DINO_HZ     = 3       # visual update rate
SCENE_JPG   = os.path.join(ROOT, "data", "scene.jpg")   # snapshot from receiver
MLP_PATH    = os.path.join(ROOT, "models", "haptic_mlp.pt")
LSTM_PATH   = os.path.join(ROOT, "models", "haptic_lstm.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iface",  default="enp131s0")
    p.add_argument("--mock",   action="store_true", help="mock ESP32")
    p.add_argument("--no-vlm", action="store_true", help="skip VLM")
    p.add_argument("--port",   default=None,        help="ESP32 serial port")
    p.add_argument("--vlm",    default="moondream",  help="vlm backend: gpt4v|moondream|gemma")
    return p.parse_args()


def load_frame(path: str):
    """Load latest camera snapshot. Returns None if not available."""
    if not os.path.exists(path):
        return None
    frame = cv2.imread(path)
    return frame


def main():
    args = parse_args()

    print("=" * 60)
    print("  Haptic Teleoperation — full pipeline")
    print(f"  iface={args.iface}  mock={args.mock}  vlm={args.vlm}")
    print("=" * 60)

    # ── 1. Haptic controller ──────────────────────────────────────────────────
    haptics = HapticController(mock=args.mock)
    if not haptics.connect(port=args.port):
        print("[main] ERROR: Could not connect to ESP32. Use --mock to test.")
        sys.exit(1)
    print("[main] ESP32 connected")

    # ── 2. G1 sensors ─────────────────────────────────────────────────────────
    try:
        sensors = G1Sensors(network_interface=args.iface)
        if not sensors.connect(timeout=10.0):
            print("[main] WARNING: G1 sensors not available — using zero torques")
            sensors = None
        else:
            print("[main] G1 sensors connected")
    except Exception as e:
        print(f"[main] WARNING: G1 sensors failed ({e}) — using zero torques")
        sensors = None

    # ── 3. Signal fusion ──────────────────────────────────────────────────────
    fusion = SignalFusionV2()
    fusion.attach_controller(haptics)

    # load LSTM if available, fallback to analytical rules
    if os.path.exists(LSTM_PATH) and os.path.exists(MLP_PATH):
        fusion.load_model(MLP_PATH, LSTM_PATH)
        print("[main] LSTM renderer loaded")
    else:
        print("[main] No LSTM checkpoint — using analytical rules")

    # ── 4. DINO + MLP ─────────────────────────────────────────────────────────
    dino = DinoEncoder()
    mlp  = None
    if dino.load() and os.path.exists(MLP_PATH):
        mlp = load_mlp(MLP_PATH, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        print("[main] DINO + MLP ready")
    else:
        print("[main] WARNING: DINO/MLP not available — texture features disabled")

    # ── 5. VLM scene seeding (one-shot, background) ───────────────────────────
    def vlm_thread():
        if args.no_vlm:
            print("[VLM] Skipped (--no-vlm)")
            # activate fusion with default caps
            fusion._active = True
            return

        print("[VLM] Waiting for camera snapshot...")
        for _ in range(30):          # wait up to 30s for snapshot
            frame = load_frame(SCENE_JPG)
            if frame is not None:
                break
            time.sleep(1.0)
        else:
            print("[VLM] No snapshot found — using default caps")
            fusion._active = True
            return

        try:
            seeder = SceneSeeder(backend=args.vlm)
            print(f"[VLM] Seeding with {args.vlm}...")
            scene  = seeder.seed(frame)
            fusion.update_from_scene(scene)
            objs = [o["name"] for o in scene.get("objects", [])]
            print(f"[VLM] Objects: {objs}")
        except Exception as e:
            print(f"[VLM] Error: {e} — using default caps")
            fusion._active = True

    threading.Thread(target=vlm_thread, daemon=True).start()

    # ── 6. DINO update thread (3 Hz) ─────────────────────────────────────────
    def dino_thread():
        interval = 1.0 / DINO_HZ
        while True:
            t0 = time.time()
            if dino.ready and mlp is not None:
                frame = load_frame(SCENE_JPG)
                if frame is not None:
                    emb = dino.encode(frame)
                    if emb is not None:
                        fusion.update_from_dino(emb)
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    threading.Thread(target=dino_thread, daemon=True).start()

    # ── 7. Main control loop (50 Hz) ─────────────────────────────────────────
    print("\n[main] Starting haptic control loop at 50 Hz — Ctrl+C to stop\n")
    interval = 1.0 / CONTROL_HZ
    t_last_print = 0

    try:
        while True:
            t0 = time.time()

            # get torque
            if sensors is not None:
                torque_sum = sensors.estimated_weight
            else:
                torque_sum = 0.0

            # compute haptic command
            cmd = fusion.step(torque_sum)

            # send to ESP32
            if cmd["duty"] > 0:
                haptics.send_all_fingers(
                    duty=cmd["duty"],
                    freq=cmd["freq"],
                    wave=cmd["wave"],
                )
            else:
                haptics.stop_all()

            # print status every 0.5s
            if time.time() - t_last_print > 0.5:
                obj  = fusion.object_name
                cap  = fusion.fragility_cap
                print(f"  [{obj:15s}] cap={cap:2d} | "
                      f"torque={torque_sum:6.1f}Nm | "
                      f"freq={cmd['freq']} duty={cmd['duty']:2d} wave={cmd['wave']}")
                t_last_print = time.time()

            # sleep to hit 50 Hz
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n[main] Stopping...")
    finally:
        haptics.stop_all()
        haptics.disconnect()
        print("[main] Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# demo_haptic.py
# V1 pipeline: GPT-4V scene seed + simulated force → haptic feedback
#
# Usage:
#   python3 demo_haptic.py                    # mock mode, no ESP32
#   python3 demo_haptic.py --real             # real ESP32
#   python3 demo_haptic.py --image path.jpg   # use image instead of webcam

import sys, os, time, argparse, threading
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from haptics.perception.scene_seeder import SceneSeeder
from haptics.signal_fusion_v1 import SignalFusionV1
from haptics.controller import HapticController
from haptics.presets import FINGER_ADDRS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real',   action='store_true', help='Use real ESP32')
    p.add_argument('--image',  type=str, default=None)
    p.add_argument('--cam',    type=int, default=0)
    p.add_argument('--object', type=str, default=None,
                   help='Target object name from scene')
    return p.parse_args()


def get_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read: {args.image}")
            sys.exit(1)
        return frame
    # webcam with warmup
    cap = cv2.VideoCapture(args.cam)
    time.sleep(1.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot open webcam")
        sys.exit(1)
    return frame


def simulate_force(t: float) -> float:
    """
    Simulate fingertip force for testing without real sensors.
    Returns 0.0-1.0.
    Ramps up → holds → releases → repeat
    """
    cycle = t % 4.0
    if cycle < 1.0:
        return cycle          # ramp up 0→1
    elif cycle < 2.5:
        return 1.0            # hold
    elif cycle < 3.0:
        return (3.0 - cycle) * 2.0  # ramp down
    else:
        return 0.0            # release


def force_loop(fusion: SignalFusionV1,
               ctrl: HapticController,
               stop_event: threading.Event,
               hz: int = 50):
    """
    Fast loop: reads force → computes haptic params → sends to LRAs.
    Replace simulate_force() with real sensor read when hardware available.
    """
    interval = 1.0 / hz
    t_start  = time.time()

    while not stop_event.is_set():
        t = time.time() - t_start

        # ── swap this line for real sensor ──────────────────────────────
        force = simulate_force(t)
        # force = real_fingertip_sensor.read()  # when hardware available
        # ────────────────────────────────────────────────────────────────

        params = fusion.compute(force)

        if params["duty"] > 0:
            ctrl.send_all_fingers(
                duty=params["duty"],
                freq=params["freq"],
                wave=params["wave"]
            )
        else:
            ctrl.stop_all()

        time.sleep(interval)


def draw_ui(frame, fusion, force, params):
    """Simple OpenCV overlay for judges."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(text, y, color=(0, 255, 180)):
        cv2.putText(frame, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    txt(f"Object:    {fusion.object_name}", 30)
    txt(f"Force:     {force:.2f}  ({'touching' if force > 0.05 else 'none'})", 58)
    txt(f"Duty:      {params['duty']:2d} / {fusion.fragility_cap}  (cap)", 86)
    txt(f"Freq:      {params['freq']}   Wave: {'sine' if params['wave'] else 'square'}", 114)
    txt("Q to quit", 142, (180, 180, 180))
    return frame


def main():
    args  = parse_args()
    mock  = not args.real

    print("\n=== Haptic Teleoperation V1 ===")
    print(f"Mode: {'MOCK' if mock else 'REAL ESP32'}\n")

    # 1 — grab frame
    print("[main] Capturing scene frame...")
    frame = get_frame(args)
    cv2.imwrite("/tmp/scene_frame.jpg", frame)
    print("[main] Frame captured")

    # 2 — GPT-4V scene seed
    print("[main] Seeding scene with GPT-4V...")
    seeder = SceneSeeder()
    scene  = seeder.seed(frame)

    if not scene["objects"]:
        print("[main] No objects detected — using default preset")

    # 3 — setup fusion
    fusion = SignalFusionV1()
    fusion.update_from_scene(scene, target_object=args.object)

    # 4 — setup haptic controller
    ctrl = HapticController(mock=mock)
    if not ctrl.connect():
        print("[main] Failed to connect haptic controller")
        sys.exit(1)

    # 5 — start force loop in background thread
    stop_event = threading.Event()
    force_thread = threading.Thread(
        target=force_loop,
        args=(fusion, ctrl, stop_event),
        daemon=True
    )
    force_thread.start()
    print("\n[main] Running — press Q to quit\n")

    # 6 — UI loop
    t_start = time.time()
    display = frame.copy()

    while True:
        t     = time.time() - t_start
        force = simulate_force(t)
        params = fusion.compute(force)

        ui_frame = draw_ui(display.copy(), fusion, force, params)
        cv2.imshow("Haptic V1", ui_frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break

    # cleanup
    stop_event.set()
    ctrl.stop_all()
    ctrl.disconnect()
    cv2.destroyAllWindows()
    print("[main] Done")


if __name__ == '__main__':
    main()
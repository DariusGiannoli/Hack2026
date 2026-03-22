"""
demo_haptic.py — HapticNet real-time teleoperation demo.

Threads:
    T1 (50 Hz)  : Read sensors → rolling window buffer
    T2 (50 Hz)  : HapticNet inference → SignalFusion → ESP32
    T3 (async)  : GPT-4V scene seed at startup
    T4 (10 Hz)  : OpenCV UI — camera feed, pressure bars, slip gauge, params

Usage:
    # mock mode (no robot, no ESP32)
    python demo_haptic.py

    # real mode
    python demo_haptic.py --real --iface enp131s0 --port /dev/ttyUSB0

    # real sensors, mock ESP32
    python demo_haptic.py --real --iface enp131s0 --mock-esp

    # skip VLM
    python demo_haptic.py --no-vlm

    # train first with synthetic data, then demo
    python neural/train_hapticnet.py --synthetic 5000
    python demo_haptic.py
"""

import sys
import os
import time
import threading
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import cv2

from haptics.controller import HapticController
from haptics.signal_fusion import SignalFusion
from neural.hapticnet import (
    HapticNet, HapticNetInference, MockSensorStream,
    load_model, N_PRESSURE, N_TORQUE, WINDOW_LEN,
)

# ── Config ───────────────────────────────────────────────────────────────────

CONTROL_HZ = 50
UI_HZ      = 10
MODEL_PATH = os.path.join(ROOT, "models", "hapticnet.pt")
SCENE_JPG  = os.path.join(ROOT, "data", "scene.jpg")


def parse_args():
    p = argparse.ArgumentParser(description="HapticNet real-time demo")
    p.add_argument("--real",      action="store_true", help="Use real G1 sensors")
    p.add_argument("--iface",     default="enp131s0",  help="G1 network interface")
    p.add_argument("--port",      default=None,        help="ESP32 serial port")
    p.add_argument("--mock-esp",  action="store_true", help="Mock ESP32 even in real mode")
    p.add_argument("--no-vlm",    action="store_true", help="Skip GPT-4V scene seeding")
    p.add_argument("--vlm",       default="gpt4v",     help="VLM backend: gpt4v|moondream|gemma")
    p.add_argument("--no-ui",     action="store_true", help="Headless mode (no OpenCV window)")
    return p.parse_args()


# ── Sensor adapter ───────────────────────────────────────────────────────────

TACTILE_MAX = 100.0   # raw tactile tip mean range — divide to get [0, 1]

TORQUE_ORDER = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw",
    "L_Elbow", "L_WristRoll", "L_WristPitch",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw",
    "R_Elbow", "R_WristRoll", "R_WristPitch",
]


class RealSensorAdapter:
    """
    Provides pressure(6) + torque(12) per step.

    pressure_6 : Inspire right-hand tactile tip means / TACTILE_MAX → [0, 1]
                 Falls back to Inspire force_act or G1 hand_state if unavailable.
    torque_12  : G1 arm joint tau_est (6 left + 6 right).
    """

    def __init__(self, g1, inspire=None):
        self.g1      = g1        # G1Sensors (always present)
        self.inspire = inspire   # InspireHandSensors (optional)

    def step(self):
        # ── pressure: prefer Inspire tactile tip means ──────────────────────
        pressure = np.zeros(N_PRESSURE, dtype=np.float32)

        if self.inspire is not None and self.inspire.right.is_fresh:
            tips = self.inspire.right.tip_pressure   # list[6] raw tactile means
            for i in range(min(N_PRESSURE, len(tips))):
                pressure[i] = float(tips[i]) / TACTILE_MAX
        elif self.inspire is not None and self.inspire.left.is_fresh:
            tips = self.inspire.left.tip_pressure
            for i in range(min(N_PRESSURE, len(tips))):
                pressure[i] = float(tips[i]) / TACTILE_MAX
        else:
            # fallback: G1 hand_state finger pressures
            fp = self.g1.finger_pressures
            for idx in range(min(N_PRESSURE, len(fp))):
                if idx in fp:
                    pressure[idx] = float(np.sum(fp[idx])) / TACTILE_MAX

        pressure = np.clip(pressure, 0.0, 1.0)

        # ── torque: G1 arm joints ───────────────────────────────────────────
        at = self.g1.arm_torques
        torque = np.array(
            [at.get(name, 0.0) for name in TORQUE_ORDER],
            dtype=np.float32,
        )
        return {"pressure": pressure, "torque": torque}


# ── UI rendering ─────────────────────────────────────────────────────────────

class DemoUI:
    """OpenCV window showing live haptic state."""

    def __init__(self, width: int = 800, height: int = 500):
        self.w = width
        self.h = height
        self._lock = threading.Lock()
        self._state = {}
        self._camera_frame = None

    def update(self, state: dict, camera_frame=None):
        with self._lock:
            self._state = dict(state)
            if camera_frame is not None:
                self._camera_frame = camera_frame.copy()

    def render(self) -> np.ndarray:
        with self._lock:
            state = dict(self._state)
            cam = self._camera_frame

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # --- camera feed (left half) ---
        cam_w, cam_h = self.w // 2 - 20, self.h - 20
        if cam is not None:
            frame = cv2.resize(cam, (cam_w, cam_h))
            canvas[10:10+cam_h, 10:10+cam_w] = frame
        else:
            cv2.putText(canvas, "No camera feed", (60, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

        # --- info panel (right half) ---
        x0 = self.w // 2 + 10
        y = 30

        def text(label, value, color=(200, 200, 200)):
            nonlocal y
            cv2.putText(canvas, f"{label}: {value}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 25

        obj = state.get("object", "?")
        text("Object", obj, (100, 255, 100))
        text("Frag cap", str(state.get("frag_cap", "?")))
        y += 5

        freq = state.get("freq", 0)
        duties = state.get("duties", [state.get("duty", 0)] * 5)
        wave = "sine" if state.get("wave", 1) else "square"
        text("Freq", str(freq))
        text("Wave", wave)
        y += 5

        slip_p = state.get("slip_prob", 0.0)
        slip_color = (0, 0, 255) if state.get("slip", False) else (200, 200, 200)
        text("Slip", f"{slip_p:.2f}", slip_color)

        if state.get("slip", False):
            cv2.putText(canvas, "!! SLIP WARNING !!", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y += 35
        y += 10

        # --- per-finger duty + pressure bars (side by side) ---
        text("Per-finger feedback", "", (150, 150, 255))
        pressures = state.get("pressure", [0]*6)
        bar_w = 14
        pair_gap = 3   # gap between pressure/duty bars in a pair
        finger_gap = 10  # gap between finger groups
        bar_max_h = 80
        fingers = ["Pk", "Rg", "Md", "Ix", "Tb"]

        for i, (fname, duty_val) in enumerate(zip(fingers, duties)):
            # pressure bar (blue)
            bx = x0 + i * (bar_w * 2 + pair_gap + finger_gap)
            by = y + bar_max_h
            p_val = pressures[i] if i < len(pressures) else 0
            bh = int(min(1.0, abs(p_val) / 5.0) * bar_max_h)
            cv2.rectangle(canvas, (bx, by - bh), (bx + bar_w, by), (150, 150, 255), -1)
            cv2.rectangle(canvas, (bx, y), (bx + bar_w, by), (80, 80, 80), 1)

            # duty bar (green/yellow)
            dx = bx + bar_w + pair_gap
            dh = int(min(1.0, duty_val / 31.0) * bar_max_h)
            d_color = (0, 200, 0) if duty_val < 20 else (0, 200, 200)
            cv2.rectangle(canvas, (dx, by - dh), (dx + bar_w, by), d_color, -1)
            cv2.rectangle(canvas, (dx, y), (dx + bar_w, by), (80, 80, 80), 1)

            # finger label + duty value
            cv2.putText(canvas, fname, (bx, by + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
            cv2.putText(canvas, str(duty_val), (bx + 2, by + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, d_color, 1)

        y += bar_max_h + 35

        # --- slip probability gauge ---
        gauge_w = self.w // 2 - 40
        gauge_h = 15
        gx, gy = x0, y
        cv2.rectangle(canvas, (gx, gy), (gx + gauge_w, gy + gauge_h), (80, 80, 80), 1)
        fill_w = int(max(0, min(1, slip_p)) * gauge_w)
        bar_color = (0, 255, 0) if slip_p < 0.3 else (0, 255, 255) if slip_p < 0.6 else (0, 0, 255)
        if fill_w > 0:
            cv2.rectangle(canvas, (gx, gy), (gx + fill_w, gy + gauge_h), bar_color, -1)
        cv2.putText(canvas, "Slip gauge", (gx, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- hz counter ---
        hz = state.get("hz", 0)
        cv2.putText(canvas, f"{hz:.0f} Hz", (x0, self.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return canvas


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  HapticNet — Real-time Haptic Teleoperation Demo")
    mode = "REAL" if args.real else "MOCK"
    print(f"  mode={mode}  vlm={args.vlm}  ui={'ON' if not args.no_ui else 'OFF'}")
    print("=" * 60)

    # ── 1. ESP32 controller ──────────────────────────────────────────────
    mock_esp = not args.real or args.mock_esp
    haptics = HapticController(mock=mock_esp)
    if not haptics.connect(port=args.port):
        print("[demo] ERROR: Could not connect to ESP32")
        sys.exit(1)
    print(f"[demo] ESP32 {'MOCK' if mock_esp else 'connected'}")

    # ── 2. Sensor source ────────────────────────────────────────────────
    if args.real:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, args.iface)

            from robot.g1_sensors import G1Sensors
            g1 = G1Sensors(network_interface=args.iface)
            g1.connect(init_dds=False)

            inspire = None
            if not args.no_inspire:
                try:
                    from robot.inspire_hand_sensors import InspireHandSensors
                    inspire = InspireHandSensors(
                        network_interface=args.iface,
                        subscribe_touch=True,
                    )
                    inspire.connect(timeout=5.0, init_dds=False)
                    print("[demo] Inspire hand sensors connected (tactile tip)")
                except Exception as e:
                    print(f"[demo] Inspire not available ({e}) — pressure from G1")

            sensor_stream = RealSensorAdapter(g1, inspire)
            print("[demo] Sensor adapter ready")
        except Exception as e:
            print(f"[demo] Sensor connect failed ({e}) — using mock")
            sensor_stream = MockSensorStream(hz=CONTROL_HZ)
    else:
        sensor_stream = MockSensorStream(hz=CONTROL_HZ)
        print("[demo] Using MOCK sensor stream")

    # ── 3. HapticNet model ──────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, device="cpu")
    else:
        print(f"[demo] No checkpoint at {MODEL_PATH} — using untrained model")
        print("       Run: python neural/train_hapticnet.py --synthetic 5000")
        model = HapticNet()

    inference = HapticNetInference(model, device="cpu")
    print("[demo] HapticNet ready")

    # ── 4. Signal fusion ────────────────────────────────────────────────
    fusion = SignalFusion(haptics, inference)

    # ── 5. VLM scene seeding (async) ────────────────────────────────────
    def vlm_thread():
        if args.no_vlm:
            print("[VLM] Skipped (--no-vlm)")
            return

        # wait for camera frame
        for _ in range(30):
            if os.path.exists(SCENE_JPG):
                break
            time.sleep(1.0)
        else:
            print("[VLM] No camera snapshot — using defaults")
            return

        try:
            from haptics.perception.scene_seeder import SceneSeeder
            frame = cv2.imread(SCENE_JPG)
            if frame is None:
                print("[VLM] Failed to read scene.jpg")
                return
            seeder = SceneSeeder(backend=args.vlm)
            scene = seeder.seed(frame)
            fusion.set_scene(scene)
        except Exception as e:
            print(f"[VLM] Error: {e}")

    threading.Thread(target=vlm_thread, daemon=True).start()

    # ── 6. Camera frame loader ──────────────────────────────────────────
    camera_frame = None
    camera_lock = threading.Lock()

    def camera_thread():
        nonlocal camera_frame
        while True:
            if os.path.exists(SCENE_JPG):
                frame = cv2.imread(SCENE_JPG)
                if frame is not None:
                    with camera_lock:
                        camera_frame = frame
            time.sleep(0.2)

    threading.Thread(target=camera_thread, daemon=True).start()

    # ── 7. UI ───────────────────────────────────────────────────────────
    ui = DemoUI() if not args.no_ui else None

    # ── 8. Main control loop ────────────────────────────────────────────
    print(f"\n[demo] Running at {CONTROL_HZ} Hz — press Q or Ctrl+C to stop\n")

    interval = 1.0 / CONTROL_HZ
    ui_interval = 1.0 / UI_HZ
    t_last_print = 0.0
    t_last_ui = 0.0
    hz_counter = 0
    hz_t0 = time.time()
    measured_hz = 0.0

    try:
        while True:
            t0 = time.time()

            # --- read sensors ---
            data = sensor_stream.step()
            pressure = data["pressure"].tolist()
            torque = data["torque"].tolist()

            # --- hapticnet + fusion → ESP32 ---
            cmd = fusion.step(pressure, torque)

            # --- hz measurement ---
            hz_counter += 1
            if time.time() - hz_t0 >= 1.0:
                measured_hz = hz_counter / (time.time() - hz_t0)
                hz_counter = 0
                hz_t0 = time.time()

            # --- terminal status (2 Hz) ---
            now = time.time()
            if now - t_last_print > 0.5:
                slip_str = " SLIP!" if cmd["slip"] else ""
                duties = cmd.get("duties", [cmd["duty"]] * 5)
                duty_str = ",".join(f"{d:2d}" for d in duties)
                print(f"  [{cmd['object']:15s}] "
                      f"freq={cmd['freq']} duties=[{duty_str}] "
                      f"wave={'sin' if cmd['wave'] else 'sq ':3s} | "
                      f"slip={cmd['slip_prob']:.2f}{slip_str} | "
                      f"{measured_hz:.0f} Hz")
                t_last_print = now

            # --- UI update (10 Hz) ---
            if ui and now - t_last_ui > ui_interval:
                with camera_lock:
                    cam = camera_frame.copy() if camera_frame is not None else None

                ui.update({
                    "object":    cmd["object"],
                    "frag_cap":  fusion.fragility_cap,
                    "freq":      cmd["freq"],
                    "duty":      cmd["duty"],
                    "duties":    cmd.get("duties", [cmd["duty"]] * 5),
                    "wave":      cmd["wave"],
                    "slip":      cmd["slip"],
                    "slip_prob": cmd["slip_prob"],
                    "pressure":  pressure,
                    "hz":        measured_hz,
                }, cam)

                frame = ui.render()
                cv2.imshow("HapticNet Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                t_last_ui = now

            # --- maintain rate ---
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n[demo] Stopping...")
    finally:
        fusion.stop()
        haptics.disconnect()
        if ui:
            cv2.destroyAllWindows()
        print("[demo] Done")


if __name__ == "__main__":
    main()

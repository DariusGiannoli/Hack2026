"""
neural/record_data.py — Record training data for HapticNet.

Two recording modes:

  Material recording (one material per run, clean labels):
    python neural/record_data.py --session material --label 1 --real --iface enp131s0 --camera
    python neural/record_data.py --session material --label 2 --real --iface enp131s0 --camera
    python neural/record_data.py --session material --label 3 --real --iface enp131s0 --camera

    Labels:
      1 = smooth/hard  (metal, glass, plastic)
      2 = rough        (fabric, cardboard, wood)
      3 = soft         (foam, sponge, rubber)

  Slip recording (auto-labeled):
    python neural/record_data.py --session slip --real --iface enp131s0 --camera

Each run APPENDS to the CSV (never overwrites).
Camera frames saved to data/frames/<session>_<timestamp>/ with matching timestamps.
Sensor warmup validates data is flowing before recording starts.
"""

import os
import sys
import csv
import time
import threading
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.hapticnet import MockSensorStream, N_PRESSURE, N_TORQUE

# ── Config ───────────────────────────────────────────────────────────────────

SAMPLE_HZ    = 50
CAMERA_HZ    = 3
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MATERIAL_CSV = os.path.join(DATA_DIR, "tactile_recordings.csv")
SLIP_CSV     = os.path.join(DATA_DIR, "slip_recordings.csv")

MATERIAL_NAMES = {1: "smooth/hard", 2: "rough", 3: "soft"}

CSV_HEADER = (
    ["timestamp"]
    + [f"p{i}" for i in range(N_PRESSURE)]
    + [f"t{i}" for i in range(N_TORQUE)]
    + ["label"]
)

# ── Slip auto-detector ──────────────────────────────────────────────────────
# Tuned for blended pressure (tip+palm), values in [0, 1] after /TACTILE_MAX.
# Slip = arm torque derivative spikes while pressure drops or stays flat.

TORQUE_SPIKE_THRESH  = 1.5    # Nm — arm torque derivative to flag spike
PRESSURE_DROP_THRESH = -0.02  # normalized — pressure must be dropping (negative)
SLIP_WINDOW          = 8      # frames lookback (~160ms at 50Hz)


class SlipDetector:
    """Auto-labels slip events from torque/pressure time series."""

    def __init__(self):
        self._torque_hist = []
        self._pressure_hist = []

    def step(self, pressure: np.ndarray, torque: np.ndarray) -> bool:
        """
        Returns True if a slip event is detected at this frame.

        Slip = arm torque derivative spike (arm fighting gravity as object slides)
               + pressure derivative negative (contact force dropping)
        """
        self._torque_hist.append(torque.copy())
        self._pressure_hist.append(pressure.copy())

        if len(self._torque_hist) < SLIP_WINDOW + 1:
            return False

        self._torque_hist = self._torque_hist[-(SLIP_WINDOW + 1):]
        self._pressure_hist = self._pressure_hist[-(SLIP_WINDOW + 1):]

        # arm torque derivative: mean absolute change over window
        t_now = self._torque_hist[-1]
        t_prev = self._torque_hist[-SLIP_WINDOW]
        d_torque = np.mean(np.abs(t_now - t_prev)) / SLIP_WINDOW

        # pressure derivative: mean change (negative = dropping)
        p_now = self._pressure_hist[-1]
        p_prev = self._pressure_hist[-SLIP_WINDOW]
        d_pressure = np.mean(p_now - p_prev) / SLIP_WINDOW

        return d_torque > TORQUE_SPIKE_THRESH and d_pressure < PRESSURE_DROP_THRESH


# ── Real sensor adapter ─────────────────────────────────────────────────────

TACTILE_MAX   = 100.0   # raw tactile mean → normalized [0, 1]
FORCE_ACT_MAX = 600.0   # Inspire motor force_act → normalized [0, 1]

TORQUE_ORDER = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw",
    "L_Elbow", "L_WristRoll", "L_WristPitch",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw",
    "R_Elbow", "R_WristRoll", "R_WristPitch",
]


class RealSensorAdapter:
    """
    Provides pressure(6) + torque(12) per step.

    Pressure signal priority per finger:
      1. max(tip_touch, palm_touch)  — tactile arrays (best signal)
      2. force_act                   — motor torque (always active when gripping)
    """

    def __init__(self, g1, inspire=None):
        self.g1 = g1
        self.inspire = inspire

    def _read_inspire_hand(self, hand) -> np.ndarray:
        pressure = np.zeros(N_PRESSURE, dtype=np.float32)

        # primary: blended tip + palm tactile
        blended = hand.pressure  # max(tip, palm) per finger
        for i in range(min(N_PRESSURE, len(blended))):
            pressure[i] = float(blended[i]) / TACTILE_MAX

        # fallback: motor force_act where tactile is near-zero
        force = hand.force_act
        for i in range(min(N_PRESSURE, len(force))):
            if pressure[i] < 0.02 and abs(force[i]) > 10:
                pressure[i] = min(1.0, abs(float(force[i])) / FORCE_ACT_MAX)

        return pressure

    def step(self) -> dict:
        pressure = np.zeros(N_PRESSURE, dtype=np.float32)

        if self.inspire is not None and self.inspire.right.is_fresh:
            pressure = self._read_inspire_hand(self.inspire.right)
        elif self.inspire is not None and self.inspire.left.is_fresh:
            pressure = self._read_inspire_hand(self.inspire.left)
        else:
            fp = self.g1.finger_pressures
            for idx in range(min(N_PRESSURE, len(fp))):
                if idx in fp:
                    pressure[idx] = float(np.sum(fp[idx])) / TACTILE_MAX

        pressure = np.clip(pressure, 0.0, 1.0)

        at = self.g1.arm_torques
        torque = np.array(
            [at.get(name, 0.0) for name in TORQUE_ORDER],
            dtype=np.float32,
        )
        return {"pressure": pressure, "torque": torque}


# ── Camera frame saver ─────────────────────────────────────────────────────

class FrameSaver:
    """Saves camera frames at CAMERA_HZ alongside tactile recording."""

    def __init__(self, camera, session_name: str):
        self._camera = camera
        # unique folder per run so recordings don't clobber each other
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self._dir = os.path.join(DATA_DIR, "frames", f"{session_name}_{run_id}")
        os.makedirs(self._dir, exist_ok=True)
        self._interval = 1.0 / CAMERA_HZ
        self._last_save = 0.0
        self._count = 0

    def maybe_save(self, timestamp: float):
        if self._camera is None:
            return
        if (timestamp - self._last_save) < self._interval:
            return
        frame = self._camera.get_frame()
        if frame is None:
            return
        import cv2
        fname = os.path.join(self._dir, f"{timestamp:.6f}.jpg")
        cv2.imwrite(fname, frame)
        self._count += 1
        self._last_save = timestamp

    @property
    def count(self) -> int:
        return self._count

    @property
    def directory(self) -> str:
        return self._dir


# ── Sensor warmup ──────────────────────────────────────────────────────────

def warmup_sensors(stream, n_frames: int = 50) -> bool:
    """
    Read n_frames and check that sensors are producing real data.
    Returns True if at least one pressure channel is non-zero.
    """
    print("\n  Warming up sensors...", end="", flush=True)
    max_p = 0.0
    for _ in range(n_frames):
        data = stream.step()
        p_max = float(np.max(data["pressure"]))
        max_p = max(max_p, p_max)
        time.sleep(1.0 / SAMPLE_HZ)

    if max_p < 0.001:
        print(f" WARNING: all pressure channels read 0!")
        print(f"  Check: is the hand grasping anything? Are touch topics publishing?")
        return False
    else:
        print(f" OK (peak pressure={max_p:.3f})")
        return True


# ── Recording: material ────────────────────────────────────────────────────

def record_material(stream, output_path: str, label: int,
                    duration: float = 120.0, camera=None):
    """
    Record one material class. Fixed label for the entire run.

    Workflow per run:
      1. Pick up the object
      2. Start recording (this function)
      3. Grasp, squeeze, slide fingers, vary pressure
      4. Press Enter to stop
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    label_name = MATERIAL_NAMES.get(label, "unknown")

    print("\n" + "=" * 60)
    print(f"  RECORDING: label={label} ({label_name})")
    print("=" * 60)
    print(f"  Hold the object and vary your grip.")
    print(f"  Slide fingers across the surface for texture.")
    print(f"  Press Enter to stop.")
    print(f"  Recording at {SAMPLE_HZ} Hz → {output_path} (append)")
    if camera:
        print(f"  Camera frames → {DATA_DIR}/frames/")
    print("=" * 60 + "\n")

    running = True

    def wait_for_stop():
        nonlocal running
        try:
            input()
        except EOFError:
            pass
        running = False

    threading.Thread(target=wait_for_stop, daemon=True).start()

    # append mode — never overwrite previous data
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    n_frames = 0
    t0 = time.time()
    interval = 1.0 / SAMPLE_HZ
    fsaver = FrameSaver(camera, f"material_{label}")

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        while running and (time.time() - t0) < duration:
            t_frame = time.time()
            data = stream.step()
            ts = time.time()

            row = [f"{ts:.6f}"]
            row += [f"{v:.6f}" for v in data["pressure"]]
            row += [f"{v:.6f}" for v in data["torque"]]
            row += [str(label)]
            writer.writerow(row)
            n_frames += 1

            fsaver.maybe_save(ts)

            if n_frames % (SAMPLE_HZ * 2) == 0:
                elapsed = time.time() - t0
                p = data["pressure"]
                active = sum(1 for v in p if v > 0.02)
                p_str = " ".join(f"{v:.2f}" for v in p)
                print(f"  [{elapsed:5.1f}s] {n_frames:5d} frames | "
                      f"{active}/6 fingers active | p=[{p_str}]"
                      + (f" | imgs={fsaver.count}" if camera else ""))

            dt = time.time() - t_frame
            if dt < interval:
                time.sleep(interval - dt)

    running = False
    elapsed = time.time() - t0
    print(f"\n  Done: {n_frames} frames ({elapsed:.1f}s), label={label} ({label_name})")
    if fsaver.count:
        print(f"  Images: {fsaver.count} → {fsaver.directory}")
    print(f"  Appended to: {output_path}\n")
    return output_path


# ── Recording: no contact baseline ────────────────────────────────────────

def record_baseline(stream, output_path: str, duration: float = 15.0,
                    camera=None):
    """Record no-contact baseline (label=0). Short automatic run."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n  Recording 15s of NO CONTACT baseline (hands open, not touching)...")

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    n_frames = 0
    t0 = time.time()
    interval = 1.0 / SAMPLE_HZ
    fsaver = FrameSaver(camera, "baseline")

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        while (time.time() - t0) < duration:
            t_frame = time.time()
            data = stream.step()
            ts = time.time()

            row = [f"{ts:.6f}"]
            row += [f"{v:.6f}" for v in data["pressure"]]
            row += [f"{v:.6f}" for v in data["torque"]]
            row += ["0"]  # label 0 = no contact
            writer.writerow(row)
            n_frames += 1
            fsaver.maybe_save(ts)

            dt = time.time() - t_frame
            if dt < interval:
                time.sleep(interval - dt)

    print(f"  Baseline done: {n_frames} frames ({duration:.0f}s)\n")
    return output_path


# ── Recording: slip ────────────────────────────────────────────────────────

def record_slip(stream, output_path: str, duration: float = 300.0,
                camera=None):
    """
    Slip auto-labeling. Operator grasps objects and lets them slide.

    The auto-detector watches for: arm torque spike + pressure drop = slip.
    Press Enter to stop.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "=" * 60)
    print("  RECORDING: Slip Detection")
    print("=" * 60)
    print("  1. Grip an object firmly")
    print("  2. Slowly loosen your grip until it slides 2-3cm")
    print("  3. Re-grip and repeat")
    print("  4. Use different objects (heavy > light)")
    print("  Aim for ~10-15% slip frames in the status line.")
    print("  Press Enter to stop.")
    print(f"  Recording at {SAMPLE_HZ} Hz → {output_path} (append)")
    print("=" * 60 + "\n")

    detector = SlipDetector()
    running = True
    n_slips = 0

    def wait_for_stop():
        nonlocal running
        try:
            input()
        except EOFError:
            pass
        running = False

    threading.Thread(target=wait_for_stop, daemon=True).start()

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    n_frames = 0
    t0 = time.time()
    interval = 1.0 / SAMPLE_HZ
    fsaver = FrameSaver(camera, "slip")

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        while running and (time.time() - t0) < duration:
            t_frame = time.time()
            data = stream.step()
            is_slip = detector.step(data["pressure"], data["torque"])
            label = 1 if is_slip else 0
            if is_slip:
                n_slips += 1

            ts = time.time()
            row = [f"{ts:.6f}"]
            row += [f"{v:.6f}" for v in data["pressure"]]
            row += [f"{v:.6f}" for v in data["torque"]]
            row += [str(label)]
            writer.writerow(row)
            n_frames += 1
            fsaver.maybe_save(ts)

            if n_frames % (SAMPLE_HZ * 2) == 0:
                elapsed = time.time() - t0
                pct = 100 * n_slips / max(1, n_frames)
                p = data["pressure"]
                active = sum(1 for v in p if v > 0.02)
                status = "OK" if 5 < pct < 25 else ("grip tighter!" if pct > 25 else "loosen grip!")
                print(f"  [{elapsed:5.1f}s] {n_frames:5d} frames | "
                      f"slip={n_slips} ({pct:.1f}% — {status}) | {active}/6 fingers"
                      + (f" | imgs={fsaver.count}" if camera else ""))

            dt = time.time() - t_frame
            if dt < interval:
                time.sleep(interval - dt)

    running = False
    elapsed = time.time() - t0
    pct = 100 * n_slips / max(1, n_frames)
    print(f"\n  Done: {n_frames} frames, {n_slips} slip ({pct:.1f}%) in {elapsed:.1f}s")
    if fsaver.count:
        print(f"  Images: {fsaver.count} → {fsaver.directory}")
    print(f"  Appended to: {output_path}\n")
    return output_path


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record HapticNet training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record smooth/hard material (metal, glass, plastic)
  python neural/record_data.py --session material --label 1 --real --iface enp131s0 --camera

  # Record rough material (fabric, cardboard, wood)
  python neural/record_data.py --session material --label 2 --real --iface enp131s0 --camera

  # Record soft material (foam, sponge, rubber)
  python neural/record_data.py --session material --label 3 --real --iface enp131s0 --camera

  # Record slip events
  python neural/record_data.py --session slip --real --iface enp131s0 --camera

  # Record no-contact baseline (15s automatic)
  python neural/record_data.py --session baseline --real --iface enp131s0
""")
    parser.add_argument("--session", choices=["material", "slip", "baseline"], required=True)
    parser.add_argument("--label", type=int, choices=[1, 2, 3],
                        help="Material label (required for --session material)")
    parser.add_argument("--real", action="store_true",
                        help="Use real robot sensors (default: mock)")
    parser.add_argument("--iface", default="enp131s0")
    parser.add_argument("--no-inspire", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration (default: 120s material, 300s slip, 15s baseline)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--camera", action="store_true",
                        help="Record camera frames for DINO/VLM training")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    if args.session == "material" and args.label is None:
        parser.error("--label is required for material session (1=smooth, 2=rough, 3=soft)")

    # connect sensors
    if args.real:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, args.iface)

            from robot.g1_sensors import G1Sensors
            g1 = G1Sensors(network_interface=args.iface)
            g1.connect(init_dds=False)
            print("[record] G1 sensors connected")

            inspire = None
            if not args.no_inspire:
                try:
                    from robot.inspire_hand_sensors import InspireHandSensors
                    inspire = InspireHandSensors(
                        network_interface=args.iface,
                        subscribe_touch=True,
                    )
                    inspire.connect(timeout=5.0, init_dds=False)
                    print("[record] Inspire hand connected (tip + palm pressure)")
                except Exception as e:
                    print(f"[record] Inspire not available ({e}) — using G1 fallback")

            stream = RealSensorAdapter(g1, inspire)
        except Exception as e:
            print(f"[record] Sensor connect failed ({e}) — using mock")
            stream = MockSensorStream(hz=SAMPLE_HZ)
    else:
        print("[record] Using MOCK sensor stream")
        stream = MockSensorStream(hz=SAMPLE_HZ)

    # warmup: validate sensors produce data
    if not args.skip_warmup:
        warmup_sensors(stream)

    # optional camera (G1 head camera via UDP from Orin sender)
    camera = None
    if args.camera:
        try:
            from robot.g1_camera import G1Camera
            camera = G1Camera(port=9000)
            if camera.connect():
                camera.start()
                print(f"[record] G1 camera connected — saving frames at {CAMERA_HZ} Hz")
            else:
                print("[record] Camera failed — is g1_stream_sender.py running on Orin?")
                camera = None
        except Exception as e:
            print(f"[record] Camera not available ({e}) — tactile only")
            camera = None

    # each run gets its own timestamped CSV in data/runs/
    runs_dir = os.path.join(DATA_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    if args.session == "material":
        path = args.output or os.path.join(runs_dir, f"material_{args.label}_{run_ts}.csv")
        dur = args.duration or 120.0
        record_material(stream, path, args.label, dur, camera=camera)

    elif args.session == "slip":
        path = args.output or os.path.join(runs_dir, f"slip_{run_ts}.csv")
        dur = args.duration or 300.0
        record_slip(stream, path, dur, camera=camera)

    elif args.session == "baseline":
        path = args.output or os.path.join(runs_dir, f"baseline_{run_ts}.csv")
        dur = args.duration or 15.0
        record_baseline(stream, path, dur, camera=camera)

    if camera:
        camera.stop()


if __name__ == "__main__":
    main()

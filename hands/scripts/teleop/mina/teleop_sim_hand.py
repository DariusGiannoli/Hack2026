"""LEAP hand teleoperation — ZED stereo + MediaPipe Hands → HandSim.

Architecture
------------
[Camera thread ~30 Hz]                   [Sim main thread 200 Hz]
  ZEDCamera.get_frames()                   HandSim run loop
  → StereoHandDetector.process_raw()         → teleop_callback(sim)
  → stereo_hand_3d()  (if STEREO_DEPTH)        → IKRetargeter.retarget(lm)
  → draw skeleton on preview                   → palm_quat(lm)
  → send JPEG to _camera_viewer subprocess     → sim.set_finger_targets()
  → write to _SharedBuf (Lock)                 → sim.set_wrist_pos/quat()

Wrist orientation calibration
------------------------------
Press **A** in the MuJoCo viewer to snapshot the current palm orientation
as the neutral reference.  All subsequent frames express orientation as a
rotation relative to that snapshot.

Usage
-----
    mjpython scripts/teleop/teleop_sim_hand.py
    mjpython scripts/teleop/teleop_sim_hand.py --camera 1 --no-preview

Run with mjpython (required on macOS for GLFW thread safety).

Port of Binocular-Teleop/teleop_leap.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

import argparse
import contextlib
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows running from repo root without install
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "source" / "mina_teleop"))
sys.path.insert(0, str(_REPO_ROOT / "source" / "mina_assets"))

from mina_teleop.environments.mujoco_hand import HandSim, HandSimConfig
from mina_teleop.inputs.vision.hand_detector import StereoHandDetector
from mina_teleop.inputs.vision.stereo_depth import ZED2I, stereo_hand_3d
from mina_teleop.inputs.vision.zed_engine import ZEDCamera
from mina_teleop.processing.one_euro_filter import OneEuroFilter
from mina_teleop.processing.velocity_limiter import VelocityLimiter
from mina_teleop.retargeters.hand import IKRetargeter, palm_quat

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Toggle to False for desk testing without a ZED: wrist stays at _FIXED_WRIST_POS
STEREO_DEPTH: bool = True

# Camera-frame → world-frame offset applied after axis remapping.
# Tune so the simulated hand appears at a comfortable height/depth.
_CAM_TO_WORLD_OFFSET = np.array([0.0, 0.0, 0.30])   # metres

# Used when STEREO_DEPTH is False
_FIXED_WRIST_POS = np.array([0.0, 0.30, 0.45])

# One Euro Filter tuning for 16 finger joint outputs
_JOINT_CUTOFF = 1.5   # Hz — lower = smoother but laggier
_JOINT_BETA   = 0.3   # higher = more responsive on fast motion

# VelocityLimiter for finger joints
_FINGER_MAX_RAD_PER_SEC = 6.0  # generous — fingers are fast, but prevents snap on tracking loss

# GLFW keycodes
_GLFW_KEY_A = 65   # calibrate wrist orientation
_GLFW_KEY_R = 82   # reset hand to start pose

_PRINT_DT = 0.20   # seconds between terminal status lines

_CAMERA_VIEWER = Path(__file__).resolve().parent / "_camera_viewer.py"

_JOINT_NAMES = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention — MuJoCo)
# ---------------------------------------------------------------------------

def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q


# ---------------------------------------------------------------------------
# Camera-frame → world-frame position mapping
# ---------------------------------------------------------------------------

def _cam_to_world(pos3d: np.ndarray) -> np.ndarray:
    """Map ``stereo_hand_3d`` output to MuJoCo world coordinates.

    ``stereo_hand_3d`` returns [X, Y, Z] where:
        X > 0  → right of camera
        Y > 0  → below  camera  (camera Y points down)
        Z > 0  → in front of camera

    MuJoCo world frame:
        +X → right,  +Y → forward,  +Z → up
    """
    world_x = -pos3d[0]   # mirror left-right for right-hand view
    world_y =  pos3d[2]   # camera forward  → world forward
    world_z = -pos3d[1]   # camera downward → invert for world up
    return np.array([world_x, world_y, world_z]) + _CAM_TO_WORLD_OFFSET


# ---------------------------------------------------------------------------
# Camera viewer subprocess helpers
# ---------------------------------------------------------------------------

def _send_frame(proc: subprocess.Popen, frame_bgr: np.ndarray) -> None:
    """Encode ``frame_bgr`` as JPEG and write to viewer subprocess stdin."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    data = buf.tobytes()
    with contextlib.suppress(OSError):
        proc.stdin.write(struct.pack(">I", len(data)))
        proc.stdin.write(data)
        proc.stdin.flush()


def _shutdown_viewer(proc: subprocess.Popen) -> None:
    """Send the zero-length shutdown frame and wait for the process to exit."""
    with contextlib.suppress(OSError):
        proc.stdin.write(struct.pack(">I", 0))
        proc.stdin.flush()
        proc.stdin.close()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Thread-safe shared buffer
# ---------------------------------------------------------------------------

class _SharedBuf:
    """Passes latest detection results from camera thread to sim callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lm = None                              # raw MediaPipe landmark list
        self._wrist_cam: np.ndarray | None = None   # [X,Y,Z] camera frame

    def write(self, lm, wrist_cam: np.ndarray | None) -> None:
        with self._lock:
            self._lm       = lm
            self._wrist_cam = wrist_cam

    def read(self) -> tuple:
        with self._lock:
            return self._lm, (
                None if self._wrist_cam is None else self._wrist_cam.copy()
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LEAP hand ZED+MediaPipe teleoperation → HandSim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera",     type=int,   default=0,
                        help="OpenCV device index (ZED shows as a wide-format UVC device)")
    parser.add_argument("--y-offset",   type=int,   default=0,
                        help="Vertical pixel shift for right ZED frame")
    parser.add_argument("--physics-hz", type=float, default=200.0,
                        help="Physics simulation rate in Hz")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the _camera_viewer subprocess")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------

    camera     = ZEDCamera(camera_id=args.camera, y_offset=args.y_offset)
    detector   = StereoHandDetector(max_hands=1)
    retargeter = IKRetargeter()
    sim        = HandSim(HandSimConfig(physics_hz=args.physics_hz))
    buf        = _SharedBuf()

    # Per-DOF One Euro Filters for smooth finger output
    joint_filters = [
        OneEuroFilter(min_cutoff=_JOINT_CUTOFF, beta=_JOINT_BETA)
        for _ in range(16)
    ]
    finger_limiter = VelocityLimiter(max_rad_per_sec=_FINGER_MAX_RAD_PER_SEC, n_joints=16)

    # ------------------------------------------------------------------
    # Camera viewer subprocess (separate process — avoids cv2/Cocoa crash)
    # ------------------------------------------------------------------

    viewer_proc: subprocess.Popen | None = None
    if not args.no_preview:
        viewer_proc = subprocess.Popen(
            [sys.executable, str(_CAMERA_VIEWER)],
            stdin=subprocess.PIPE,
        )

    # ------------------------------------------------------------------
    # Calibration state
    # ------------------------------------------------------------------

    _cal: dict = {
        "pending":       False,
        "reset_pending": False,
        "cal_quat":      None,                        # palm_quat at calibration
        "start_q":       sim.cfg.start_quat.copy(),   # target at calibration pose
    }

    # ------------------------------------------------------------------
    # Camera thread (~30 Hz)
    # ------------------------------------------------------------------

    active = {"flag": True}

    def _camera_loop() -> None:
        while active["flag"]:
            try:
                left, right = camera.get_frames()
            except RuntimeError as exc:
                print(f"[camera] {exc}")
                time.sleep(0.1)
                continue

            res_l, res_r = detector.process_raw(left, right)

            # Raw landmark list (protobuf) — prefer left camera
            lm = None
            if res_l and res_l.multi_hand_landmarks:
                lm = res_l.multi_hand_landmarks[0].landmark

            # Stereo wrist depth
            wrist_cam: np.ndarray | None = None
            if STEREO_DEPTH and lm is not None:
                lm_r = (
                    res_r.multi_hand_landmarks[0].landmark
                    if res_r and res_r.multi_hand_landmarks
                    else None
                )
                if lm_r is not None:
                    h, w = left.shape[:2]
                    wrist_cam = stereo_hand_3d(lm, lm_r, frame_w=w, frame_h=h, cam=ZED2I)

            buf.write(lm, wrist_cam)

            # Preview frame
            if viewer_proc is not None and viewer_proc.poll() is None:
                # Draw skeleton on left frame before hstack (slice is non-contiguous)
                if lm is not None:
                    detector.draw_landmarks(left, res_l)
                preview = np.hstack([left, right])
                # Status overlay
                detected = lm is not None
                cv2.putText(
                    preview,
                    "Hand detected" if detected else "No hand",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 220, 0) if detected else (0, 60, 220), 2,
                )
                if wrist_cam is not None:
                    wp = _cam_to_world(wrist_cam)
                    cv2.putText(
                        preview,
                        f"wrist [{wp[0]:+.2f}  {wp[1]:+.2f}  {wp[2]:+.2f}]",
                        (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2,
                    )
                _send_frame(viewer_proc, preview)

    threading.Thread(target=_camera_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # GLFW key callback — called from the MuJoCo viewer main loop
    # ------------------------------------------------------------------

    def _key_callback(keycode: int) -> None:
        if keycode == _GLFW_KEY_A:
            _cal["pending"] = True
            print("[teleop] Calibration queued — hold hand in neutral pose.")
        elif keycode == _GLFW_KEY_R:
            _cal["reset_pending"] = True
            print("[teleop] Reset queued — returning to start pose.")

    # ------------------------------------------------------------------
    # Teleoperation callback — runs on sim main thread at physics_hz
    # ------------------------------------------------------------------

    _last_print  = [0.0]
    _last_q      = [np.zeros(16)]
    _last_cb_t   = [0.0]

    def teleop_callback(s: HandSim) -> None:
        if _cal["reset_pending"]:
            _cal["reset_pending"] = False
            _cal["cal_quat"]      = None
            s.reset()
            finger_limiter.reset()
            for f in joint_filters:
                f.reset()
            _last_cb_t[0] = 0.0
            print("[teleop] Reset complete.")
            return

        lm, wrist_cam = buf.read()

        if lm is None:
            return

        now = time.perf_counter()

        # 1. Finger joint angles (retarget + smooth + velocity clamp)
        dt_cb = now - _last_cb_t[0] if _last_cb_t[0] > 0.0 else 1.0 / args.physics_hz
        _last_cb_t[0] = now
        raw_q = retargeter.retarget(lm)
        for i in range(16):
            raw_q[i] = joint_filters[i].apply(raw_q[i], now)
        raw_q = finger_limiter.apply(raw_q, dt_cb)
        _last_q[0] = raw_q
        s.set_finger_targets(raw_q)

        # 2. Wrist orientation
        curr_q = _qnorm(palm_quat(lm))

        if _cal["pending"]:
            _cal["cal_quat"] = curr_q.copy()
            _cal["pending"]  = False
            print(f"[teleop] Calibrated  cal_q = {curr_q.round(3)}")

        if _cal["cal_quat"] is not None:
            # Relative rotation from calibration pose → apply to start orientation
            q_rel = _qmul(_qconj(_cal["cal_quat"]), curr_q)
            wrist_q = _qnorm(_qmul(_cal["start_q"], q_rel))
        else:
            wrist_q = curr_q

        s.set_wrist_quat(wrist_q)

        # 3. Wrist position
        if STEREO_DEPTH and wrist_cam is not None:
            s.set_wrist_pos(_cam_to_world(wrist_cam))
        else:
            s.set_wrist_pos(_FIXED_WRIST_POS)

        # 4. Terminal readout
        if now - _last_print[0] >= _PRINT_DT:
            _last_print[0] = now
            q   = _last_q[0]
            r1  = "  ".join(f"{n}:{v:+.2f}" for n, v in zip(_JOINT_NAMES[:8],  q[:8]))
            r2  = "  ".join(f"{n}:{v:+.2f}" for n, v in zip(_JOINT_NAMES[8:], q[8:]))
            cal = " [CAL]" if _cal["cal_quat"] is not None else "      "
            print(f"\r[hand ✓{cal}]  {r1}", flush=True)
            print(f"              {r2}", end="\033[1A", flush=True)

    # ------------------------------------------------------------------
    # Launch — blocks on main thread (GLFW / macOS requirement)
    # ------------------------------------------------------------------

    print(
        "\n[teleop] LEAP Hand simulation starting…\n"
        f"  Camera      : {args.camera}  (ZED SBS → split left/right)\n"
        f"  Stereo depth: {'on' if STEREO_DEPTH else 'off — fixed wrist pos'}\n"
        f"  Preview     : {'off' if args.no_preview else 'subprocess (_camera_viewer.py)'}\n"
        "\n  Press A in the MuJoCo viewer to calibrate wrist orientation.\n"
        "  Press R in the MuJoCo viewer to reset hand to start pose.\n"
    )

    try:
        sim.run(teleop_callback=teleop_callback, key_callback=_key_callback)
    finally:
        active["flag"] = False
        detector.close()
        camera.close()
        if viewer_proc is not None:
            _shutdown_viewer(viewer_proc)
        print("\n[teleop] Done.")


if __name__ == "__main__":
    main()

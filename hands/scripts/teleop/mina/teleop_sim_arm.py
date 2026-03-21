"""Bimanual Pink-IK teleoperation — MediaPipe Pose → ArmSim.

Uses MediaPipe Pose world-space 3D landmarks to track shoulder / elbow / wrist
positions in metric metres, then feeds them as end-effector targets to a Pink
(Pinocchio) IK solver that drives the Berkeley Humanoid Lite arm joints.

Improvements over naive retargeting
-------------------------------------
- Body-relative frame   : targets expressed relative to shoulder midpoint, so
                          leaning forward/sideways does not shift the robot arm.
- One Euro Filter       : smooths 3D EE target positions before IK to remove
                          MediaPipe jitter.
- VelocityLimiter       : clamps per-joint velocity to prevent sudden jumps on
                          tracking loss / re-engagement.
- Forearm roll (--track-hands) : runs MediaPipe Hands alongside Pose; derives
                          wrist rotation from palm normal + forearm axis.
- Delta + clutch        : maps human workspace delta to robot workspace;
                          Space / 'e' key toggles engagement.

Controls (terminal)
-------------------
  e + Enter   — engage / disengage tracking
  c + Enter   — recalibrate neutral pose
  q + Enter   — stop teleop

Usage
-----
    mjpython scripts/teleop/mina/teleop_sim_arm.py
    mjpython scripts/teleop/mina/teleop_sim_arm.py --arm both --track-hands
    mjpython scripts/teleop/mina/teleop_sim_arm.py --countdown 5 --no-preview
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
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "source" / "mina_teleop"))
sys.path.insert(0, str(_REPO_ROOT / "source" / "mina_assets"))

from mina_assets import ARM_URDF
from mina_teleop.environments.mujoco_arm import ArmSim, ArmSimConfig
from mina_teleop.ik.arm_ik import BimanualPinkIK
from mina_teleop.inputs.vision.arm_detector import (
    BimanualArmEstimator,
    BimanualArmLandmarks,
)
from mina_teleop.processing.one_euro_filter import OneEuroFilter
from mina_teleop.processing.velocity_limiter import VelocityLimiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Workspace scale — human wrist reach (~70 cm) → robot reach (~42 cm)
_SCALE: float = 0.60

# Camera-frame → robot-frame rotation.
# MediaPipe world: x = subject-right, y = up, z = AWAY from camera.
# Robot frame (Berkeley Humanoid Lite): x = forward, y = left, z = up.
_R_CAM_TO_ROBOT = np.array([
    [ 0,  0, -1],   # robot_x ← -cam_z
    [-1,  0,  0],   # robot_y ← -cam_x
    [ 0,  1,  0],   # robot_z ←  cam_y
], dtype=np.float64)

# Workspace clamp — max displacement from neutral EE [m]
_MAX_DELTA_M: float = 0.35

# One Euro Filter tuning for EE target positions
_EE_MIN_CUTOFF: float = 3.0   # Hz — lower = smoother, more lag
_EE_BETA:       float = 0.1   # higher = more responsive on fast motion

# Velocity limiter — max joint speed [rad/s]
_MAX_RAD_PER_SEC: float = 3.0

# Terminal refresh
_PRINT_DT: float = 1.0 / 5.0

_LABELS = [
    "L sh_pitch", "L sh_roll ", "L sh_yaw  ", "L el_pitch", "L el_roll ",
    "R sh_pitch", "R sh_roll ", "R sh_yaw  ", "R el_pitch", "R el_roll ",
]

_CAMERA_VIEWER = Path(__file__).resolve().parent / "_camera_viewer.py"


# ---------------------------------------------------------------------------
# Camera viewer subprocess helpers
# ---------------------------------------------------------------------------

def _send_frame(proc: subprocess.Popen, frame_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    data = buf.tobytes()
    with contextlib.suppress(OSError):
        proc.stdin.write(struct.pack(">I", len(data)))
        proc.stdin.write(data)
        proc.stdin.flush()


def _shutdown_viewer(proc: subprocess.Popen) -> None:
    with contextlib.suppress(OSError):
        proc.stdin.write(struct.pack(">I", 0))
        proc.stdin.flush()
        proc.stdin.close()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _palm_to_wrist_rot(
    palm_normal_cam: np.ndarray,
    elbow_cam: np.ndarray,
    wrist_cam: np.ndarray,
) -> np.ndarray:
    """Build a 3×3 wrist rotation in robot frame from palm normal + forearm vector.

    Frame convention:
      z-axis = forearm direction (elbow → wrist) in robot frame
      y-axis = palm normal orthogonalised w.r.t. forearm
      x-axis = cross(y, z)
    """
    forearm = (wrist_cam - elbow_cam).astype(np.float64)
    z = _R_CAM_TO_ROBOT @ forearm
    z_n = np.linalg.norm(z)
    if z_n < 1e-6:
        return np.eye(3)
    z /= z_n

    palm = _R_CAM_TO_ROBOT @ palm_normal_cam.astype(np.float64)
    y = palm - np.dot(palm, z) * z
    y_n = np.linalg.norm(y)
    if y_n < 1e-6:
        return np.eye(3)
    y /= y_n

    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _filter3(filters: list[OneEuroFilter], pos: np.ndarray, t: float) -> np.ndarray:
    """Apply three scalar One Euro Filters to a (3,) position vector."""
    return np.array([f.apply(float(v), t) for f, v in zip(filters, pos)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bimanual Pink-IK teleoperation → ArmSim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera",      type=int,   default=0)
    parser.add_argument("--kp",          type=float, default=20.0)
    parser.add_argument("--kd",          type=float, default=2.0)
    parser.add_argument("--countdown",   type=int,   default=3,
                        help="Seconds before auto-calibration at launch")
    parser.add_argument("--arm",         choices=["left", "right", "both"],
                        default="right",
                        help="Which arm(s) to track")
    parser.add_argument("--track-hands", action="store_true",
                        help="Enable MediaPipe Hands for forearm-roll tracking")
    parser.add_argument("--no-preview",  action="store_true",
                        help="Disable the camera debug viewer subprocess")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    orientation_cost = 0.1 if args.track_hands else 0.0

    estimator = BimanualArmEstimator(
        camera_id=args.camera,
        track_hands=args.track_hands,
    )
    ik  = BimanualPinkIK(ARM_URDF, orientation_cost=orientation_cost)
    sim = ArmSim(ArmSimConfig(kp=args.kp, kd=args.kd))

    estimator.start()

    # ------------------------------------------------------------------
    # Camera viewer subprocess
    # ------------------------------------------------------------------

    viewer_proc: subprocess.Popen | None = None
    if not args.no_preview:
        viewer_proc = subprocess.Popen(
            [sys.executable, str(_CAMERA_VIEWER)],
            stdin=subprocess.PIPE,
        )

    # ------------------------------------------------------------------
    # Filters + velocity limiter
    # ------------------------------------------------------------------

    def _make_filters() -> list[OneEuroFilter]:
        return [OneEuroFilter(min_cutoff=_EE_MIN_CUTOFF, beta=_EE_BETA) for _ in range(3)]

    filters = {
        "wrist_L":  _make_filters(),
        "wrist_R":  _make_filters(),
        "elbow_L":  _make_filters(),
        "elbow_R":  _make_filters(),
    }
    vel_limiter = VelocityLimiter(max_rad_per_sec=_MAX_RAD_PER_SEC)

    def _reset_filters() -> None:
        for fs in filters.values():
            for f in fs:
                f.reset()

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------

    state: dict = {
        # Calibration
        "calibrated":          False,
        "calibrate_requested": False,
        "shoulder_cal":        None,   # (3,) shoulder midpoint at calibration
        "wrist_cal_L":         None,
        "wrist_cal_R":         None,
        "elbow_cal_L":         None,
        "elbow_cal_R":         None,

        # Delta/clutch
        "engaged":             False,
        "offset_L":            np.zeros(3),
        "offset_R":            np.zeros(3),
        "elbow_offset_L":      np.zeros(3),
        "elbow_offset_R":      np.zeros(3),
        "space_requested":     False,

        # Display
        "latest_targets":      np.zeros(10),
        "last_print":          0.0,
        "active":              True,
    }

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _auto_calibrate() -> None:
        print("\n[teleop] Hold BOTH arms relaxed at your sides.")
        for i in range(args.countdown, 0, -1):
            print(f"[teleop] Calibrating in {i}…")
            time.sleep(1.0)
        while state["active"]:
            lm = estimator.get_landmarks()
            if lm is not None:
                _do_calibrate(lm)
                print("[teleop] ✓ Calibrated — press e + Enter to engage tracking.\n")
                return
            time.sleep(0.05)

    def _do_calibrate(lm: BimanualArmLandmarks) -> None:
        shoulder_mid = (
            lm.left.shoulder.astype(np.float64)
            + lm.right.shoulder.astype(np.float64)
        ) / 2.0
        state["shoulder_cal"] = shoulder_mid.copy()
        state["wrist_cal_L"]  = lm.left.wrist.copy().astype(np.float64)
        state["wrist_cal_R"]  = lm.right.wrist.copy().astype(np.float64)
        state["elbow_cal_L"]  = lm.left.elbow.copy().astype(np.float64)
        state["elbow_cal_R"]  = lm.right.elbow.copy().astype(np.float64)
        state["calibrated"]   = True
        _reset_filters()

    def _keyboard_listener() -> None:
        print("[teleop] Commands:  e + Enter = engage/disengage  |  c + Enter = recalibrate  |  q + Enter = stop\n")
        while state["active"]:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == "e":
                state["space_requested"] = True
            elif cmd == "c":
                state["calibrate_requested"] = True
            elif cmd == "q":
                state["active"] = False
                print("[teleop] Teleop stopped. Close the MuJoCo window to exit.")

    def _frame_sender() -> None:
        while state["active"]:
            if viewer_proc is not None and viewer_proc.poll() is None:
                frame = estimator.get_debug_frame()
                if frame is not None:
                    _send_frame(viewer_proc, frame)
            time.sleep(0.10)

    threading.Thread(target=_auto_calibrate,    daemon=True).start()
    threading.Thread(target=_keyboard_listener, daemon=True).start()
    if viewer_proc is not None:
        threading.Thread(target=_frame_sender,  daemon=True).start()

    # ------------------------------------------------------------------
    # Teleoperation callback — runs on main thread at ~500 Hz
    # ------------------------------------------------------------------

    def teleop_callback(s: ArmSim) -> None:
        if not state["active"]:
            return

        lm = estimator.get_landmarks()

        # ---- Recalibration ----
        if state["calibrate_requested"] and lm is not None:
            _do_calibrate(lm)
            state["calibrate_requested"] = False
            state["engaged"] = False
            ik.reset()
            vel_limiter.reset()
            print("[teleop] ✓ Recalibrated. Press e + Enter to engage.")

        if not state["calibrated"] or lm is None:
            now = time.perf_counter()
            if not state["calibrated"] and now - state["last_print"] >= _PRINT_DT:
                state["last_print"] = now
                print("[teleop] Waiting for landmarks…", flush=True)
            return

        # ---- Space / 'e' key: toggle engagement ----
        if state["space_requested"]:
            state["space_requested"] = False
            if not state["engaged"]:
                if args.arm in ("right", "both"):
                    state["offset_R"]       = lm.right.wrist.astype(np.float64) - state["wrist_cal_R"]
                    state["elbow_offset_R"] = lm.right.elbow.astype(np.float64) - state["elbow_cal_R"]
                if args.arm in ("left", "both"):
                    state["offset_L"]       = lm.left.wrist.astype(np.float64) - state["wrist_cal_L"]
                    state["elbow_offset_L"] = lm.left.elbow.astype(np.float64) - state["elbow_cal_L"]
                vel_limiter.reset(state["latest_targets"])
                state["engaged"] = True
                print(f"[teleop] ▶ Engaged — {args.arm} arm(s) tracking.")
            else:
                state["engaged"] = False
                print("[teleop] ◼ Disengaged — arms holding position.")

        # ---- Hold when disengaged ----
        if not state["engaged"]:
            s.set_joint_targets(state["latest_targets"])
            return

        # ---- Body-relative correction ----
        # Subtract how much the shoulder midpoint has moved since calibration.
        # This makes targets invariant to the person translating their body.
        shoulder_now = (
            lm.left.shoulder.astype(np.float64)
            + lm.right.shoulder.astype(np.float64)
        ) / 2.0
        body_shift = shoulder_now - state["shoulder_cal"]

        now = time.perf_counter()
        _arm = args.arm

        # ---- Right arm ----
        if _arm in ("right", "both"):
            delta_R       = (lm.right.wrist.astype(np.float64) - state["wrist_cal_R"] - body_shift) - state["offset_R"]
            delta_robot_R = np.clip(_R_CAM_TO_ROBOT @ delta_R * _SCALE, -_MAX_DELTA_M, _MAX_DELTA_M)
            target_R      = _filter3(filters["wrist_R"], ik.neutral_ee_right + delta_robot_R, now)

            elbow_delta_R       = (lm.right.elbow.astype(np.float64) - state["elbow_cal_R"] - body_shift) - state["elbow_offset_R"]
            elbow_delta_robot_R = np.clip(_R_CAM_TO_ROBOT @ elbow_delta_R * _SCALE, -_MAX_DELTA_M, _MAX_DELTA_M)
            elbow_target_R      = _filter3(filters["elbow_R"], ik.neutral_elbow_right + elbow_delta_robot_R, now)

            rot_R = (
                _palm_to_wrist_rot(lm.palm_normal_R, lm.right.elbow, lm.right.wrist)
                if args.track_hands and lm.palm_normal_R is not None
                else None
            )
        else:
            delta_robot_R  = np.zeros(3)
            target_R       = ik.neutral_ee_right
            elbow_target_R = None
            rot_R          = None

        # ---- Left arm ----
        if _arm in ("left", "both"):
            delta_L       = (lm.left.wrist.astype(np.float64) - state["wrist_cal_L"] - body_shift) - state["offset_L"]
            delta_robot_L = np.clip(_R_CAM_TO_ROBOT @ delta_L * _SCALE, -_MAX_DELTA_M, _MAX_DELTA_M)
            target_L      = _filter3(filters["wrist_L"], ik.neutral_ee_left + delta_robot_L, now)

            elbow_delta_L       = (lm.left.elbow.astype(np.float64) - state["elbow_cal_L"] - body_shift) - state["elbow_offset_L"]
            elbow_delta_robot_L = np.clip(_R_CAM_TO_ROBOT @ elbow_delta_L * _SCALE, -_MAX_DELTA_M, _MAX_DELTA_M)
            elbow_target_L      = _filter3(filters["elbow_L"], ik.neutral_elbow_left + elbow_delta_robot_L, now)

            rot_L = (
                _palm_to_wrist_rot(lm.palm_normal_L, lm.left.elbow, lm.left.wrist)
                if args.track_hands and lm.palm_normal_L is not None
                else None
            )
        else:
            delta_robot_L  = np.zeros(3)
            target_L       = ik.neutral_ee_left
            elbow_target_L = None
            rot_L          = None

        # ---- Solve IK ----
        dt = s.model.opt.timestep
        arm_q = ik.solve(
            target_L, target_R, dt,
            elbow_target_L, elbow_target_R,
            rot_L, rot_R,
        )

        # ---- Velocity limiter ----
        arm_q = vel_limiter.apply(arm_q, dt)

        state["latest_targets"] = arm_q
        s.set_joint_targets(arm_q)

        # ---- Terminal display at 5 Hz ----
        if now - state["last_print"] >= _PRINT_DT:
            state["last_print"] = now
            t = arm_q
            if _arm in ("left", "both"):
                left = "  ".join(f"{l}:{v:+.2f}" for l, v in zip(_LABELS[:5], t[:5]))
                print(f"[L] {left}  Δ:{np.round(delta_robot_L, 3)}", flush=True)
            if _arm in ("right", "both"):
                right = "  ".join(f"{l}:{v:+.2f}" for l, v in zip(_LABELS[5:], t[5:]))
                print(f"[R] {right}  Δ:{np.round(delta_robot_R, 3)}", flush=True)

    # ------------------------------------------------------------------
    # Space key callback (GLFW)
    # ------------------------------------------------------------------

    def _on_key(keycode: int) -> None:
        if keycode == 32:   # Space
            state["space_requested"] = True

    # ------------------------------------------------------------------
    # Run — blocks on main thread (required by GLFW / macOS Cocoa)
    # ------------------------------------------------------------------

    try:
        sim.run(teleop_callback=teleop_callback, key_callback=_on_key)
    finally:
        state["active"] = False
        estimator.stop()
        if viewer_proc is not None:
            _shutdown_viewer(viewer_proc)
        print("\n")


if __name__ == "__main__":
    main()

"""
teleop_leap.py — Hand teleoperation with direct angle retargeting.

Pipeline (30 Hz vision loop):
  ZED left camera frame (monocular mode)
    → MediaPipe hand tracking (single camera)
    → direct angle retargeting  (MediaPipe 3-D joint angles → 16 LEAP joints)
    → One Euro filtered mocap position  (hand proxy follows wrist, Y fixed)
    → One Euro filtered joint targets
    → MuJoCo position actuators

  Stereo depth is intentionally disabled (STEREO_DEPTH = False).
  Once IK is fully tuned, flip that flag to re-enable epipolar + triangulation.

Run with mjpython (NOT plain python — cv2.imshow conflicts with Cocoa on macOS):
    mjpython teleop_leap.py

Tuning guide (constants block below):
    JOINT_MC / JOINT_BETA  — joint filter: lower MC = smoother, higher beta = less lag
    POS_MC   / POS_BETA    — position filter
    X/Z_SCALE              — workspace size in simulation metres
    DEPTH_SCALE            — how much stereo depth maps to sim-Y movement (only when STEREO_DEPTH=True)
"""

import os
import multiprocessing as _mp
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from vision.camera                       import ZEDCamera
from vision.detectors                    import StereoHandTracker, ArmTracker
import vision.geometry                   as geo
from vision.smoother                     import OneEuroFilter
from robots.leap_hand.ik_retargeting     import IKRetargeter, palm_quat
from robots.leap_hand.arm_ik             import ArmIKSolver

# ── Tunable constants ─────────────────────────────────────────────────────────
CAMERA_ID    = 4       # 0 = webcam / seule caméra détectée. Mettre 1 quand la ZED est branchée.
N_SUBSTEPS   = 16       # lighter physics load for better FPS/thermals

# ── Mode toggle ───────────────────────────────────────────────────────────────
# False = monocular (left frame only, Y fixed) → tune IK first.
# True  = stereo    (epipolar + triangulated depth) → requires ZED camera.
STEREO_DEPTH = False  # mettre True une fois le tracking validé

# One Euro Filter — joints (16-dim)
JOINT_FREQ   = 30.0    # Hz: expected vision loop rate
JOINT_MC     = 1.0     # min_cutoff: lower → smoother at rest
JOINT_BETA   = 0.03    # beta:       higher → less lag during fast motion

# One Euro Filter — wrist position (3-dim)
POS_FREQ     = 30.0
POS_MC       = 0.8
POS_BETA     = 0.005

# Workspace mapping
# In the new pinhole back-projection model the workspace scales automatically
# with depth (back_project returns real metres).  X_SCALE / Z_SCALE are kept
# here as legacy references but are no longer applied to the position output.
X_SCALE      = 0.3     # legacy — no longer used
Z_SCALE      = 0.3     # legacy — no longer used

# Depth (stereo Z) → sim Y — only used when STEREO_DEPTH = True
DEPTH_MIN_M  = 0.20
DEPTH_MAX_M  = 0.90
DEPTH_MID_M  = 0.45    # neutral depth → hand sits at START_Y
DEPTH_SCALE  = 2.0     # m of sim-Y movement per m of depth change
TRANS_SCALE  = 6.0     # global translation gain (higher = more movement)
START_Y      = 0.30    # initial sim Y (forward) of the hand proxy — also used as fixed Y
START_Z      = 0.45    # initial sim Z (height) of the hand proxy

# Epipolar constraint — only checked when STEREO_DEPTH = True
EPIPOLAR_TOL = 40      # px (relaxed — tighten once Y_OFFSET_PX is tuned for your unit)

# ── Wrist rotation (via mocap_quat) ──────────────────────────────────────────
# Rz = roll  from knuckle line (INDEX_MCP → RING_MCP)
# Rx = pitch from wrist→middle-MCP tilt (uses MediaPipe .z depth)
# Ry = yaw   from index↔pinky depth skew (uses MediaPipe .z depth)
WRIST_SCALE    = 2.0    # gain on detected angle delta (shared X/Y/Z)  [-20%]
WRIST_DZ_RX    = 0.03   # rad (~7°) — deadzone pitch
WRIST_DZ_RY    = 0.12   # rad — larger yaw deadzone to reduce twitch
WRIST_DZ_RZ    = 0.12   # rad (~6°) — deadzone roll
WRIST_MAX_RAD  = 2.0    # max clamp (~45°, reduced to avoid vibration at limits)
MOCAP_MAX_STEP = 0.015  # max position change per frame (m) — prevents teleportation
RY_POS_BOOST   = 1.2    # reduced yaw sensitivity (positive side)
RY_NEG_BOOST   = 2.0    # reduced yaw sensitivity (negative side)
RZ_RY_DECOUPLE = 0.6    # subtract this × Ry from Rz to cancel cross-talk

# Morphological calibration (human -> robot scale)
MORPH_SCALE_MIN = 0.60
MORPH_SCALE_MAX = 1.50
MORPH_PRINT_EVERY_SEC = 1.0  # terminal log period for live morphology
ARM_RIGHT_GAIN = 1.60        # tuned from teleop_edgard.py — 1.60 est plus stable que 1.80
ARM_LEFT_GAIN  = 1.60        # idem bras gauche

# ── Torso-relative arm control ─────────────────────────────────────────────────
# True  = bras pilotés en repère torse (épaule→poignet, MediaPipe Pose world)
# False = ancien comportement repère monde (pixel projection)
USE_TORSO_RELATIVE = True

# MediaPipe Pose world → robot sim axis mapping
# MP world (origin=hips): x=cam-right (person-left), y=up, z=toward-cam
# Robot sim:              x=lateral,                 z=up, y=forward
# Signs à inverser si le mouvement part dans la mauvaise direction.
_MP_SIGN_X = -1.0   # flip lateral (caméra non miroir)
_MP_SIGN_Y =  +1.0   # mp.y (up) → robot.z (up) : même sens (lever bras = robot.z monte)
_MP_SIGN_Z = -1.0   # mp.z (toward-cam) → robot.y (forward) : sens opposé

# One Euro Filters for wrist angles (1-dim each)
WRIST_FREQ     = 30.0
WRIST_MC       = 0.3
WRIST_BETA     = 0.01

# Hold last pose when hand disappears (avoids jerk on tracking loss)
HOLD_POSE_SEC  = 1.0   # seconds to hold last pose before resetting

# Rest orientation of the hand (unchanged from last push)
BASE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # Rx(180°): palm facing up (stable physics)

# ── Handedness filter ─────────────────────────────────────────────────────────────────────
# ZED is a non-mirrored camera: your RIGHT hand appears on the LEFT side of the
# image, so MediaPipe labels it "Left".  Flip to "Right" if using a mirrored cam.
TARGET_HAND   = "Left"   # tracks your physical right hand on a non-mirrored ZED
OTHER_HAND    = "Right"  # your physical left hand on a non-mirrored ZED

# cv2.imshow conflicts with mjpython's Cocoa event loop on macOS.
# Set to True only when running with plain `python` (not `mjpython`).
SHOW_CAMERA  = True
PRINT_RIGHT_ARM_DEBUG = True

# ── Quaternion helpers ────────────────────────────────────────────────────────
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_ensure_hemi(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Negate q if it is in the opposite hemisphere from ref (avoids filter flips)."""
    return -q if np.dot(q, ref) < 0 else q


def _mp_world_to_robot(delta_mp: np.ndarray) -> np.ndarray:
    """Convertit un vecteur delta MediaPipe Pose world → repère robot sim.

    MP world : x=cam-right(person-left), y=up, z=toward-cam
    Robot sim : x=lateral, y=forward(depth), z=up
    Les signes sont ajustables via _MP_SIGN_X/Y/Z.
    """
    return np.array([
        _MP_SIGN_X * delta_mp[0],   # latéral
        _MP_SIGN_Z * delta_mp[2],   # profondeur → y robot
        _MP_SIGN_Y * delta_mp[1],   # hauteur    → z robot
    ])


# ── Hand selection helper ──────────────────────────────────────────────────────
def _find_hand_by_label(result, label: str):
    """Return (landmarks, score) for the hand matching *label*, or (None, 0)."""
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None, 0.0
    for i, hand_class in enumerate(result.multi_handedness):
        cls = hand_class.classification[0]
        if cls.label == label:
            return result.multi_hand_landmarks[i], cls.score
    return None, 0.0


def _find_other_hand(result, excluded_label: str):
    """
    Return (landmarks, score) for any hand whose label != excluded_label.

    Useful when handedness is unstable: if MediaPipe/ZED flips labels for one hand,
    we still get a valid "other hand" candidate instead of dropping left-arm IK.
    """
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None, 0.0

    best_i = None
    best_score = -1.0
    for i, hand_class in enumerate(result.multi_handedness):
        cls = hand_class.classification[0]
        if cls.label == excluded_label:
            continue
        if float(cls.score) > best_score:
            best_i = i
            best_score = float(cls.score)

    if best_i is None:
        return None, 0.0
    return result.multi_hand_landmarks[best_i], best_score


def _dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pose_morphology(pose_result):
    """Extract arm lengths + shoulder width from MediaPipe pose world landmarks."""
    if pose_result is None or pose_result.pose_world_landmarks is None:
        return None

    lm = pose_result.pose_world_landmarks.landmark

    def p(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)

    # MediaPipe Pose indices: L(11,13,15), R(12,14,16)
    l_sh, l_el, l_wr = p(11), p(13), p(15)
    r_sh, r_el, r_wr = p(12), p(14), p(16)

    l_upper = _dist3(l_sh, l_el)
    l_fore = _dist3(l_el, l_wr)
    r_upper = _dist3(r_sh, r_el)
    r_fore = _dist3(r_el, r_wr)
    shoulder_w = _dist3(l_sh, r_sh)

    # Reject obviously bad frames.
    vals = np.array([l_upper, l_fore, r_upper, r_fore, shoulder_w], dtype=float)
    if np.any(vals < 0.05) or np.any(vals > 0.80):
        return None

    return {
        "left_reach_h": l_upper + l_fore,
        "right_reach_h": r_upper + r_fore,
        "shoulder_w_h": shoulder_w,
    }


# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "leap_hand", "scene_humanoid_only.xml")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_hand(model: mujoco.MjModel, data: mujoco.MjData,
               arm_ik: 'ArmIKSolver | None' = None,
               left_arm_ik: 'ArmIKSolver | None' = None) -> np.ndarray:
    """
    Teleport the LEAP palm and set both arms to bent rest poses.

    Returns the offset (hand_proxy_pos − right arm_ee_pos) at init time.
    """
    # Right arm: bent rest pose
    if arm_ik is not None:
        sp_jid = model.joint("arm_right_shoulder_pitch_joint").id
        ep_jid = model.joint("arm_right_elbow_pitch_joint").id
        data.qpos[model.jnt_qposadr[sp_jid]] = np.pi / 2
        data.qpos[model.jnt_qposadr[ep_jid]] = -np.pi / 4

    # Left arm: mirrored bent rest pose
    if left_arm_ik is not None:
        sp_jid = model.joint("arm_left_shoulder_pitch_joint").id
        ep_jid = model.joint("arm_left_elbow_pitch_joint").id
        data.qpos[model.jnt_qposadr[sp_jid]] = -np.pi / 2
        data.qpos[model.jnt_qposadr[ep_jid]] = np.pi / 4

    mid  = model.body("hand_proxy").mocapid[0]
    pos  = np.array([0.0, START_Y, START_Z])

    data.mocap_pos[mid]  = pos
    data.mocap_quat[mid] = BASE_QUAT.copy()

    RELPOSE_QUAT = np.array([0.5, -0.5, 0.5, 0.5])
    palm_quat_init = _quat_mul(BASE_QUAT, RELPOSE_QUAT)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_free")
    if jid != -1:
        addr = model.jnt_qposadr[jid]
        data.qpos[addr:addr+3] = pos
        data.qpos[addr+3:addr+7] = palm_quat_init

    # Lock arm actuators at their current qpos
    for ik_solver in (arm_ik, left_arm_ik):
        if ik_solver is not None:
            for i, act_idx in enumerate(ik_solver.act_indices):
                data.ctrl[act_idx] = data.qpos[ik_solver.qpos_adr[i]]

    mujoco.mj_forward(model, data)

    if arm_ik is not None:
        return pos - data.xpos[arm_ik.ee_body_id].copy()
    return np.zeros(3)


def _update(data:         mujoco.MjData,
            zed:          ZEDCamera,
            tracker:      StereoHandTracker,
            pose_tracker: 'ArmTracker | None',
            ik:           IKRetargeter,
            pos_f:        OneEuroFilter,
            joint_f:      OneEuroFilter,
            orient_f:     OneEuroFilter,
            pitch_f:      OneEuroFilter,
            yaw_f:        OneEuroFilter,
            mid:          int,
            arm_ik:       'ArmIKSolver | None' = None,
            arm_offset:   np.ndarray = np.zeros(3),
            left_arm_ik:  'ArmIKSolver | None' = None,
            left_pos_f:   'OneEuroFilter | None' = None,
            right_torso_f:'OneEuroFilter | None' = None) -> None:
    """
    Single-frame update: capture → detect → retarget → actuate.

    Right physical hand → LEAP fingers + right arm IK.
    Left physical hand  → left arm IK (position only).
    """
    global _wrist_ref_angle, _wrist_calib_count, _pitch_ref_angle, _pitch_calib_count, _yaw_ref_angle, _yaw_calib_count, _last_hand_time, _calibrate_flag, _left_calib_flag, _left_ref_pos, _left_ee_start, _last_left_target, _left_mono_ref_span
    global _right_ref_pos, _right_ee_start, _arm_scale_left, _arm_scale_right
    global _robot_reach_left, _robot_reach_right, _robot_shoulder_w
    global _last_morph_print_time
    global _mp_right_hand_rel_ref, _mp_left_hand_rel_ref, _mp_left_smoothed_rel
    import time as _time
    ik_info = None
    left_ik_info = None
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    # Continuous console debug for right arm IK state.
    if PRINT_RIGHT_ARM_DEBUG and arm_ik is not None:
        r_ee = data.xpos[arm_ik.ee_body_id]
        rq = data.qpos[arm_ik.qpos_adr]
        rdeg = np.degrees(rq)
        #print(
        #    "[R_ARM] "
        #    f"ee=({r_ee[0]:+.3f}, {r_ee[1]:+.3f}, {r_ee[2]:+.3f})  "
        #    f"sh_pitch={rdeg[0]:+6.1f}  sh_roll={rdeg[1]:+6.1f}  sh_yaw={rdeg[2]:+6.1f}  "
        #    f"el_pitch={rdeg[3]:+6.1f}  el_roll={rdeg[4]:+6.1f}"
        #)

    # ── Reset (touche R) ──────────────────────────────────────────────────
    if _reset_flag is not None and _reset_flag.value:
        _reset_flag.value = 0
        _wrist_ref_angle = None; _wrist_calib_count = 0
        _pitch_ref_angle = None; _pitch_calib_count = 0
        _yaw_ref_angle   = None; _yaw_calib_count  = 0
        orient_f.reset(); pitch_f.reset(); yaw_f.reset()
        joint_f.reset()
        data.ctrl[:] = 0.0
        data.qvel[:] = 0.0
        _left_ref_pos = None
        _last_left_target = None
        _left_mono_ref_span = None
        _right_ref_pos = None
        _right_ee_start = None
        _mp_right_hand_rel_ref = None
        _mp_left_hand_rel_ref  = None
        _mp_left_smoothed_rel  = None
        if left_pos_f is not None:
            left_pos_f.reset()
        if right_torso_f is not None:
            right_torso_f.reset()
        _init_hand(data.model, data, arm_ik, left_arm_ik)
        # print("[RESET] Hand position, fingers & calibration reset (R key)")

    h, w, _ = frame_l.shape
    res_l, res_r = tracker.process(frame_l, frame_r)
    pose_res = pose_tracker.process(frame_l) if pose_tracker is not None else None
    morph_live = _pose_morphology(pose_res)
    now = _time.monotonic()
    if now - _last_morph_print_time >= MORPH_PRINT_EVERY_SEC:
        if (pose_res is not None
                and pose_res.pose_world_landmarks is not None):
            wlm = pose_res.pose_world_landmarks.landmark
            # Torso centre = average of shoulders (11,12) + hips (23,24)
            tx = (wlm[11].x + wlm[12].x + wlm[23].x + wlm[24].x) / 4.0
            ty = (wlm[11].y + wlm[12].y + wlm[23].y + wlm[24].y) / 4.0
            tz = (wlm[11].z + wlm[12].z + wlm[23].z + wlm[24].z) / 4.0
            print(
                f"[TORSO] x={tx:+.3f}  y={ty:+.3f}  z={tz:+.3f}  (m, MediaPipe world frame)"
            )
        else:
            print("[TORSO] pose non détectée")
        _last_morph_print_time = now

    # ── Find each hand by handedness label ──────────────────────────────
    lm_target, target_score = _find_hand_by_label(res_l, TARGET_HAND)
    lm_other,  other_score  = _find_hand_by_label(res_l, OTHER_HAND)
    if lm_other is None:
        # Fallback robust to handedness flip: pick any non-target hand.
        lm_other, other_score = _find_other_hand(res_l, TARGET_HAND)

    # Also look for hands in right camera (for stereo)
    lm_target_r, _ = _find_hand_by_label(res_r, TARGET_HAND)
    lm_other_r,  _ = _find_hand_by_label(res_r, OTHER_HAND)
    if lm_other_r is None:
        lm_other_r, _ = _find_other_hand(res_r, TARGET_HAND)

    # ── Target hand (your physical right) must be visible to control ────
    elapsed_since_start = _time.monotonic() - _start_time if _start_time else 0

    if lm_target is None:
        elapsed = _time.monotonic() - _last_hand_time
        holding = elapsed < HOLD_POSE_SEC and _last_hand_time > 0

        if not holding:
            orient_f.reset()
            pitch_f.reset()
            yaw_f.reset()
            data.mocap_quat[mid] = BASE_QUAT.copy()

        hold_label = f"HOLD {HOLD_POSE_SEC - elapsed:.1f}s" if holding else "NO HAND"
        hold_col   = (0, 200, 255) if holding else (0, 0, 255)
        if SHOW_CAMERA:
            h_no, w_no, _ = frame_l.shape
            cv2.putText(frame_l, f"R.hand: {hold_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hold_col, 2)
            if lm_other is not None:
                cv2.putText(frame_l, f"L.hand: OK ({other_score:.0%})", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            else:
                cv2.putText(frame_l, "L.hand: ---", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            if _wrist_ref_angle is None:
                remaining = max(0, AUTO_CALIB_SEC - elapsed_since_start)
                cv2.putText(frame_l, f"CALIBRATION DANS {remaining:.1f}s", (20, h_no // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            tracker.draw_landmarks(frame_l, res_l)
            if frame_r is not None:
                tracker.draw_landmarks(frame_r, res_r)
                _show(np.hstack([frame_l, frame_r]))
            else:
                _show(frame_l)
        else:
            _show(frame_l)
        return

    _last_hand_time = _time.monotonic()

    lm_l = lm_target.landmark
    cam   = geo.ZED2I

    # ── Palm center position (avg of wrist + 4 MCP) ────────────────────
    _palm_ids = (0, 5, 9, 13, 17)  # wrist, index/middle/ring/pinky MCP
    u_w = sum(lm_l[i].x for i in _palm_ids) / len(_palm_ids) * w
    v_w = sum(lm_l[i].y for i in _palm_ids) / len(_palm_ids) * h
    sim_x = -(u_w - cam.cx) / cam.fx * START_Y * TRANS_SCALE
    sim_z =  START_Z + (-(v_w - cam.cy) / cam.fy * START_Y) * TRANS_SCALE

    # ── Depth axis (sim Y): ZED point cloud at palm center ────────────────
    sim_y      = START_Y
    depth_cm   = None
    hud_mode   = "ZED DEPTH"
    hud_col    = (0, 220, 0)   # green = ZED depth active
    hud_detail = ""

    pc = zed.get_point_cloud()
    if pc is not None:
        try:
            import pyzed.sl as sl
            px_pc = int(np.clip(u_w, 0, w - 1))
            py_pc = int(np.clip(v_w, 0, h - 1))
            err_pc, pc_val = pc.get_value(px_pc, py_pc)
            z_m = float(pc_val[2])
            if (err_pc == sl.ERROR_CODE.SUCCESS and
                    not np.isnan(z_m) and
                    DEPTH_MIN_M < z_m < DEPTH_MAX_M):
                # ZED RIGHT_HANDED_Y_UP: Z = toward camera (depth)
                # ZED X = right, Y = up → recompute sim_x/sim_z from 3D
                x_m = float(pc_val[0])
                y_m = float(pc_val[1])
                sim_x  = -x_m * TRANS_SCALE
                sim_y  = START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE * TRANS_SCALE
                sim_z  = START_Z + (-y_m) * TRANS_SCALE
                depth_cm   = z_m * 100
                hud_detail = f"z={z_m:.2f}m"
            else:
                hud_mode   = "ZED NO DEPTH"
                hud_col    = (0, 165, 255)
                hud_detail = f"z={z_m:.2f}m (hors plage)" if not np.isnan(z_m) else "NaN"
        except Exception as e:
            hud_mode   = "ZED ERR"
            hud_col    = (0, 0, 255)
            hud_detail = str(e)
    else:
        hud_mode = "MONO [L]"
        hud_col  = (0, 165, 255)

    # ── Compute raw wrist angles ────────────────────────────────────────
    idx_mcp  = lm_l[5]
    ring_mcp = lm_l[13]
    raw_angle = np.arctan2(ring_mcp.y - idx_mcp.y, ring_mcp.x - idx_mcp.x)

    mid_mcp  = lm_l[9]
    wrist_lm = lm_l[0]
    dy_p = mid_mcp.y - wrist_lm.y
    dz_p = mid_mcp.z - wrist_lm.z
    raw_pitch = np.arctan2(dz_p, dy_p)

    idx_mcp_y = lm_l[5]
    pky_mcp_y = lm_l[17]
    dx_y = pky_mcp_y.x - idx_mcp_y.x
    dz_y = pky_mcp_y.z - idx_mcp_y.z
    raw_yaw = dz_y / max(abs(dx_y), 0.01)

    # ── Auto-calibration: triggers after AUTO_CALIB_SEC or on A key ──
    elapsed_since_start = _time.monotonic() - _start_time if _start_time else 0
    auto_trigger = (_wrist_ref_angle is None and elapsed_since_start >= AUTO_CALIB_SEC)
    if _calibrate_flag or auto_trigger:
        _wrist_ref_angle = raw_angle
        _pitch_ref_angle = raw_pitch
        _yaw_ref_angle   = raw_yaw
        orient_f.reset()
        pitch_f.reset()
        yaw_f.reset()
        pos_f.reset()
        joint_f.reset()
        _calibrate_flag = False
        _left_calib_flag = True
        _right_ref_pos = np.array([sim_x, sim_y, sim_z], dtype=float)
        mujoco.mj_forward(data.model, data)
        _right_ee_start = data.xpos[arm_ik.ee_body_id].copy() if arm_ik is not None else None

        # Capture référence torso-relative (bras droit + gauche)
        _mp_right_hand_rel_ref = None   # sera capturé au premier frame valide
        _mp_left_hand_rel_ref  = None   # idem bras gauche
        _mp_left_smoothed_rel  = None   # reset du signal filtré sensor-space gauche
        if right_torso_f is not None:
            right_torso_f.reset()

        morph = _pose_morphology(pose_res)
        if (morph is not None and
                _robot_reach_left is not None and _robot_reach_right is not None):
            left_scale = _robot_reach_left / max(morph["left_reach_h"], 1e-6)
            right_scale = _robot_reach_right / max(morph["right_reach_h"], 1e-6)
            _arm_scale_left = float(np.clip(left_scale, MORPH_SCALE_MIN, MORPH_SCALE_MAX))
            _arm_scale_right = float(np.clip(right_scale, MORPH_SCALE_MIN, MORPH_SCALE_MAX))
            print(
                f"[MORPH] scales L={_arm_scale_left:.2f} R={_arm_scale_right:.2f} "
                f"(human shoulder={morph['shoulder_w_h']:.3f}m, "
                f"robot shoulder={_robot_shoulder_w:.3f}m)"
            )
        else:
            _arm_scale_left = 1.0
            _arm_scale_right = 1.0
            #print("[MORPH] Pose indisponible: scale=1.0")

        # print("[CALIB] Orientation de référence capturée (auto)." if auto_trigger else "[CALIB] Orientation de référence capturée.")

    # ── Before calibration: hand frozen at start pose ─────────────────
    if _wrist_ref_angle is None:
        wrist_x = wrist_y = wrist_z = wrist_z_raw = 0.0
    else:
        # ── Mocap position (only after calibration) ──────────────────
        raw_pos = np.array([sim_x, sim_y, sim_z])
        new_pos = pos_f(raw_pos)
        delta_pos = new_pos - data.mocap_pos[mid]
        dist = np.linalg.norm(delta_pos)
        if dist > MOCAP_MAX_STEP:
            new_pos = data.mocap_pos[mid] + delta_pos * (MOCAP_MAX_STEP / dist)
        data.mocap_pos[mid] = new_pos

        # Roll (Rz)
        delta = (raw_angle - _wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta = -delta
        delta = float(np.clip(delta, -0.9, 0.9))
        delta = float(orient_f(np.array([delta]))[0])
        if abs(delta) < WRIST_DZ_RZ:
            delta = 0.0
        else:
            delta = np.sign(delta) * (abs(delta) - WRIST_DZ_RZ)
        wrist_z = float(np.clip(delta * WRIST_SCALE * 0.9, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Pitch (Rx)
        delta_p = (raw_pitch - _pitch_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta_p = delta_p
        delta_p = float(np.clip(delta_p, -0.9, 0.9))
        delta_p = float(pitch_f(np.array([delta_p]))[0])
        if abs(delta_p) < WRIST_DZ_RX:
            delta_p = 0.0
        else:
            delta_p = np.sign(delta_p) * (abs(delta_p) - WRIST_DZ_RX)
        wrist_x = float(np.clip(-delta_p * WRIST_SCALE * 5.0, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Yaw (Ry) — boost positive side to compensate MediaPipe .z asymmetry
        delta_y = (raw_yaw - _yaw_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta_y = -delta_y
        if delta_y > 0:
            delta_y *= RY_POS_BOOST
        else:
            delta_y *= RY_NEG_BOOST
        delta_y = float(np.clip(delta_y, -0.9, 0.9))
        delta_y = float(yaw_f(np.array([delta_y]))[0])
        if abs(delta_y) < WRIST_DZ_RY:
            delta_y = 0.0
        else:
            delta_y = np.sign(delta_y) * (abs(delta_y) - WRIST_DZ_RY)
        wrist_y = float(np.clip(-delta_y * WRIST_SCALE, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Decouple: shrink Rz toward zero when Ry is active (kills cross-talk)
        wrist_z_raw = wrist_z
        reduction = RZ_RY_DECOUPLE * abs(wrist_y)
        if abs(wrist_z) > reduction:
            wrist_z = wrist_z - np.sign(wrist_z) * reduction
        else:
            wrist_z = 0.0

        # Incremental rotation: small per-axis quats applied to BASE_QUAT
        # in the body-local frame (avoids gimbal lock)
        half_x = wrist_x / 2.0
        dq_x = np.array([np.cos(half_x), np.sin(half_x), 0.0, 0.0])
        half_y = wrist_z / 2.0
        dq_y = np.array([np.cos(half_y), 0.0, np.sin(half_y), 0.0])
        half_z = wrist_y / 2.0
        dq_z = np.array([np.cos(half_z), 0.0, 0.0, np.sin(half_z)])
        # Apply in body-local frame: BASE * dq_x * dq_y * dq_z
        q = _quat_mul(BASE_QUAT, dq_x)
        q = _quat_mul(q, dq_y)
        q = _quat_mul(q, dq_z)
        q = q / np.linalg.norm(q)
        data.mocap_quat[mid] = q

        # ── Arm IK: right arm ────────────────────────────────────────────
        ik_info = None
        if arm_ik is not None:
            arm_target_pos = None
            if (USE_TORSO_RELATIVE
                    and pose_res is not None
                    and pose_res.pose_world_landmarks is not None
                    and _right_ee_start is not None):
                # Repère torse : vecteur épaule droite → poignet droit (MP Pose world)
                plm = pose_res.pose_world_landmarks.landmark
                rsh = np.array([plm[12].x, plm[12].y, plm[12].z])
                rwr = np.array([plm[16].x, plm[16].y, plm[16].z])
                hand_rel = _mp_world_to_robot(rwr - rsh)
                # Filtre en espace capteur (même pipeline que teleop_edgard.py)
                hand_rel_f = right_torso_f(hand_rel) if right_torso_f is not None else hand_rel
                if _mp_right_hand_rel_ref is None:
                    _mp_right_hand_rel_ref = hand_rel_f.copy()
                    # print("[CALIB] Référence torse bras droit capturée.")
                delta_torso = (hand_rel_f - _mp_right_hand_rel_ref) * _arm_scale_right * ARM_RIGHT_GAIN
                arm_target_pos = _right_ee_start + delta_torso
            elif _right_ref_pos is not None and _right_ee_start is not None:
                # Fallback : repère monde (pixel projection)
                right_delta = data.mocap_pos[mid] - _right_ref_pos
                arm_target_pos = _right_ee_start + right_delta * _arm_scale_right * ARM_RIGHT_GAIN
            else:
                arm_target_pos = data.mocap_pos[mid] - arm_offset
            ik_info = arm_ik.solve(data.model, data, arm_target_pos, q)

        # ── Direct angle retargeting (only after calibration) ────────
        if ik is not None:
            q_raw    = ik.retarget(None, lm_l)
            q_smooth = joint_f(q_raw)
            n_leap = len(q_smooth)
            for i in range(n_leap):
                lo, hi = data.model.actuator_ctrlrange[i]
                q_smooth[i] = np.clip(q_smooth[i], lo, hi)
            data.ctrl[:n_leap] = q_smooth

    # ── Left arm IK ──────────────────────────────────────────────────────
    if left_arm_ik is not None:
        # Try torso-relative path first (ZED BODY_38 left shoulder + wrist)
        _torso_left_ok = False
        if (USE_TORSO_RELATIVE
                and pose_res is not None
                and pose_res.pose_world_landmarks is not None):
            plm = pose_res.pose_world_landmarks.landmark
            lsh = np.array([plm[11].x, plm[11].y, plm[11].z])
            lwr = np.array([plm[15].x, plm[15].y, plm[15].z])

            # Only use this path if both shoulder and wrist are valid (not NaN)
            if not (np.any(np.isnan(lsh)) or np.any(np.isnan(lwr))):
                _torso_left_ok = True
                hand_rel_l = _mp_world_to_robot(lwr - lsh)

                # ── Pipeline identique à teleop_edgard.py ──────────────────
                # 1. Filtre en espace capteur (sensor-space first)
                if left_pos_f is not None:
                    hand_rel_l_f = left_pos_f(hand_rel_l)
                else:
                    hand_rel_l_f = hand_rel_l.copy()

                # 2. MOCAP_MAX_STEP sur le signal filtré (pas sur la cible finale)
                if _mp_left_smoothed_rel is not None:
                    d_sens = hand_rel_l_f - _mp_left_smoothed_rel
                    dist_sens = np.linalg.norm(d_sens)
                    if dist_sens > MOCAP_MAX_STEP:
                        hand_rel_l_f = _mp_left_smoothed_rel + d_sens * (MOCAP_MAX_STEP / dist_sens)
                _mp_left_smoothed_rel = hand_rel_l_f.copy()

                if _left_calib_flag:
                    # Capture la ref sur le signal filtré (delta=0 à t=0, comme edgard.py)
                    _mp_left_hand_rel_ref = hand_rel_l_f.copy()
                    _mp_left_smoothed_rel = hand_rel_l_f.copy()
                    mujoco.mj_forward(data.model, data)
                    _left_ee_start = data.xpos[left_arm_ik.ee_body_id].copy()
                    _last_left_target = None
                    if left_pos_f is not None:
                        left_pos_f.reset()
                    _left_calib_flag = False

                if _mp_left_hand_rel_ref is not None and _left_ee_start is not None:
                    # 3. Delta × gain → target IK (sans re-filtrage ni re-clamp)
                    delta_l = (_mp_left_smoothed_rel - _mp_left_hand_rel_ref) * _arm_scale_left * ARM_LEFT_GAIN
                    target_lh = _left_ee_start + delta_l
                    _last_left_target = target_lh.copy()
                    no_orient = np.array([1.0, 0.0, 0.0, 0.0])
                    left_ik_info = left_arm_ik.solve(data.model, data, target_lh, no_orient)

        # ── Fallback: MediaPipe left hand landmarks (pixel-based) ────────
        # Used when torso-relative is disabled or ZED left wrist is NaN.
        if not _torso_left_ok and lm_other is not None:
            lm_left = lm_other.landmark
            cam = geo.ZED2I

            _palm_ids = (0, 5, 9, 13, 17)
            u_lh = sum(lm_left[i].x for i in _palm_ids) / len(_palm_ids) * w
            v_lh = sum(lm_left[i].y for i in _palm_ids) / len(_palm_ids) * h

            lh_x = -(u_lh - cam.cx) / cam.fx * START_Y * TRANS_SCALE
            lh_z = START_Z + (-(v_lh - cam.cy) / cam.fy * START_Y) * TRANS_SCALE
            lh_y = START_Y

            lh_span = np.hypot(lm_left[9].x * w - lm_left[0].x * w,
                               lm_left[9].y * h - lm_left[0].y * h)

            # ZED point cloud depth for left hand
            depth_ok = False
            pc = zed.get_point_cloud()
            if pc is not None:
                try:
                    import pyzed.sl as sl
                    px_lh = int(np.clip(u_lh, 0, w - 1))
                    py_lh = int(np.clip(v_lh, 0, h - 1))
                    err_lh, pc_lh = pc.get_value(px_lh, py_lh)
                    z_lh = float(pc_lh[2])
                    if (err_lh == sl.ERROR_CODE.SUCCESS and
                            not np.isnan(z_lh) and
                            DEPTH_MIN_M < z_lh < DEPTH_MAX_M):
                        lh_x = -float(pc_lh[0]) * TRANS_SCALE
                        lh_y = START_Y + (DEPTH_MID_M - z_lh) * DEPTH_SCALE * TRANS_SCALE
                        lh_z = START_Z + (-float(pc_lh[1])) * TRANS_SCALE
                        depth_ok = True
                except Exception:
                    pass

            if not depth_ok and _left_mono_ref_span is not None and lh_span > 10:
                mono_depth_m = DEPTH_MID_M * _left_mono_ref_span / lh_span
                mono_depth_m = float(np.clip(mono_depth_m, DEPTH_MIN_M, DEPTH_MAX_M))
                lh_y = START_Y + (DEPTH_MID_M - mono_depth_m) * DEPTH_SCALE * TRANS_SCALE

            raw_lh_pos = np.array([lh_x, lh_y, lh_z])

            if _left_calib_flag:
                _left_ref_pos = raw_lh_pos.copy()
                _left_mono_ref_span = lh_span if lh_span > 10 else None
                mujoco.mj_forward(data.model, data)
                _left_ee_start = data.xpos[left_arm_ik.ee_body_id].copy()
                _last_left_target = None
                if left_pos_f is not None:
                    left_pos_f.reset()
                _left_calib_flag = False
                # print(f"[CALIB] Left arm reference captured (palm span={lh_span:.0f}px).")

            if _left_ref_pos is not None:
                delta_lh = raw_lh_pos - _left_ref_pos
                target_lh = _left_ee_start + delta_lh * _arm_scale_left * ARM_LEFT_GAIN
                if left_pos_f is not None:
                    target_lh = left_pos_f(target_lh)
                if _last_left_target is not None:
                    delta_lt = target_lh - _last_left_target
                    dist_lt = np.linalg.norm(delta_lt)
                    left_max_step = MOCAP_MAX_STEP * _arm_scale_left * ARM_LEFT_GAIN
                    if dist_lt > left_max_step:
                        target_lh = _last_left_target + delta_lt * (left_max_step / dist_lt)
                _last_left_target = target_lh.copy()
                no_orient = np.array([1.0, 0.0, 0.0, 0.0])
                left_ik_info = left_arm_ik.solve(data.model, data, target_lh, no_orient)

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, res_l)
        if pose_tracker is not None and pose_res is not None:
            pose_tracker.draw_landmarks(frame_l, pose_res)

        # Morph scale HUD
        morph_txt = f"MORPH L={_arm_scale_left:.2f} R={_arm_scale_right:.2f}"
        morph_col = (220, 180, 0) if _arm_scale_left != 1.0 else (100, 100, 100)
        cv2.putText(frame_l, morph_txt, (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, morph_col, 2)

        # Top-left: mode + hand detection status
        cv2.putText(frame_l, hud_mode, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2)
        if hud_detail:
            cv2.putText(frame_l, hud_detail, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        # Hand detection status (both hands)
        rh_col = (0, 220, 0) if lm_target is not None else (0, 0, 255)
        rh_txt = f"R.hand: OK ({target_score:.0%})" if lm_target is not None else "R.hand: ---"
        lh_col = (0, 220, 0) if lm_other is not None else (100, 100, 100)
        lh_txt = f"L.hand: OK ({other_score:.0%})" if lm_other is not None else "L.hand: ---"
        cv2.putText(frame_l, rh_txt, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rh_col, 2)
        cv2.putText(frame_l, lh_txt, (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lh_col, 2)

        # Calibration status banner with countdown
        if _wrist_ref_angle is None:
            remaining = max(0, AUTO_CALIB_SEC - elapsed_since_start)
            calib_txt = f"CALIBRATION DANS {remaining:.1f}s"
            cv2.putText(frame_l, calib_txt, (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Top-right: depth readout (large)
        if depth_cm is not None:
            depth_str = f"DEPTH: {depth_cm:.0f} cm"
            d_col = (0, 220, 0)
        else:
            depth_str = "DEPTH: ---"
            d_col = (0, 165, 255)
        txt_size = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame_l, depth_str, (w - txt_size[0] - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_col, 2)

        # Bottom: rotation debug per axis (large text)
        rx_deg = float(np.degrees(wrist_x))
        ry_deg = float(np.degrees(wrist_y))
        rz_deg = float(np.degrees(wrist_z))
        rz_raw_deg = float(np.degrees(wrist_z_raw))

        rz_col = (0, 220, 0) if abs(rz_deg) > 0.1 else (100, 100, 100)
        cv2.putText(frame_l, f"Rz(roll) : {rz_deg:+6.1f}  (raw {rz_raw_deg:+.1f})",
                    (15, h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rz_col, 2)

        rx_col = (255, 100, 0) if abs(rx_deg) > 0.1 else (100, 100, 100)
        wz0 = lm_l[0].z; mz9 = lm_l[9].z
        cv2.putText(frame_l, f"Rx(pitch): {rx_deg:+6.1f}  wrist.z={wz0:+.2f} mid.z={mz9:+.2f}",
                    (15, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rx_col, 2)

        ry_col = (0, 220, 220) if abs(ry_deg) > 0.1 else (100, 100, 100)
        iz5 = lm_l[5].z; pz17 = lm_l[17].z
        cv2.putText(frame_l, f"Ry(yaw)  : {ry_deg:+6.1f}  idx.z={iz5:+.2f} pky.z={pz17:+.2f}",
                    (15, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ry_col, 2)

        cv2.putText(frame_l, f"SENT  Rx:{rx_deg:+.0f}  Ry:{ry_deg:+.0f}  Rz:{rz_deg:+.0f}",
                    (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 220), 3)

        # Show both cameras side by side with detection status
        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            n_hands_r = len(res_r.multi_hand_landmarks) if res_r.multi_hand_landmarks else 0
            r_det = f"R.cam: {n_hands_r} hand(s)"
            r_col = (0, 220, 0) if n_hands_r > 0 else (0, 0, 255)
            cv2.putText(frame_r, r_det, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_col, 2)

            # ── RIGHT ARM IK HUD (bottom-right of right frame) ──
            if ik_info is not None:
                d = ik_info["deg"]
                err = ik_info["err_mm"]
                err_col = (0, 220, 0) if err < 30 else (0, 165, 255) if err < 80 else (0, 0, 255)
                _ik_lines = [
                    (f"R.ARM  err={err:.0f}mm", err_col),
                    (f"sh_p={d[0]:+5.0f}  sh_r={d[1]:+5.0f}  sh_y={d[2]:+5.0f}", (200, 200, 200)),
                    (f"el_p={d[3]:+5.0f}  el_r={d[4]:+5.0f}", (200, 200, 200)),
                ]
                for li, (txt, col) in enumerate(_ik_lines):
                    y_pos = h - 30 - (len(_ik_lines) - 1 - li) * 40
                    sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.putText(frame_r, txt, (w - sz[0] - 15, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

            # ── LEFT ARM IK HUD (bottom-left of right frame) ──
            if left_ik_info is not None:
                dl = left_ik_info["deg"]
                lerr = left_ik_info["err_mm"]
                lerr_col = (0, 220, 0) if lerr < 30 else (0, 165, 255) if lerr < 80 else (0, 0, 255)
                _lk_lines = [
                    (f"L.ARM  err={lerr:.0f}mm", lerr_col),
                    (f"sh_p={dl[0]:+5.0f}  sh_r={dl[1]:+5.0f}  sh_y={dl[2]:+5.0f}", (200, 200, 200)),
                    (f"el_p={dl[3]:+5.0f}  el_r={dl[4]:+5.0f}", (200, 200, 200)),
                ]
                for li, (txt, col) in enumerate(_lk_lines):
                    y_pos = h - 30 - (len(_lk_lines) - 1 - li) * 40
                    cv2.putText(frame_r, txt, (15, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

            display = np.hstack([frame_l, frame_r])
        else:
            display = frame_l

    _show(display if SHOW_CAMERA else frame_l)


_wrist_ref_angle = None
_pitch_ref_angle = None
_yaw_ref_angle = None
_last_hand_time = 0.0
_calibrate_flag = False
_reset_flag = None
_frame_q = None
_show_counter = 0
_SHOW_EVERY = 5        # send 1 frame out of 5 to the viewer
_VIEWER_SCALE = 0.35   # stronger downscale for lower CPU/GPU load

# Left arm calibration state
_left_calib_flag = False
_left_ref_pos = None
_left_ee_start = None
_last_left_target = None
_left_mono_ref_span = None   # palm length in pixels at calibration (for mono depth)
_right_ref_pos = None
_right_ee_start = None

# Torso-relative calibration references (pose world coords, shoulder→wrist)
_mp_right_hand_rel_ref = None   # vecteur de référence épaule droite → poignet droit
_mp_left_hand_rel_ref  = None   # vecteur de référence épaule gauche → poignet gauche
_mp_left_smoothed_rel  = None   # signal bras gauche filtré en espace capteur (MOCAP_MAX_STEP)

# Morphological calibration state (human -> robot scaling)
_arm_scale_left = 1.0
_arm_scale_right = 1.0
_robot_reach_left = None
_robot_reach_right = None
_robot_shoulder_w = None
_last_morph_print_time = 0.0

# Auto-calibration timer (seconds after viewer opens)
AUTO_CALIB_SEC = 10
_start_time = None      # set once in main()


def _show(frame):
    """Send a downscaled frame to the viewer subprocess every N calls."""
    global _show_counter
    if not SHOW_CAMERA or _frame_q is None:
        return
    _show_counter += 1
    if _show_counter % _SHOW_EVERY != 0:
        return
    small = cv2.resize(frame, None, fx=_VIEWER_SCALE, fy=_VIEWER_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    if _frame_q.full():
        try:
            _frame_q.get_nowait()
        except Exception:
            pass
    try:
        _frame_q.put_nowait(small)
    except Exception:
        pass


def _key_callback(keycode):
    """MuJoCo viewer key callback: press A to (re-)calibrate both hands."""
    global _calibrate_flag, _left_calib_flag
    if keycode == 65:  # GLFW_KEY_A
        _calibrate_flag = True
        _left_calib_flag = True
        # print("[CALIB] Touche A détectée — calibration des deux mains au prochain frame.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _frame_q, _start_time, _robot_reach_left, _robot_reach_right, _robot_shoulder_w

    # Hardware
    # Pass y_offset so vertical alignment is ready when STEREO_DEPTH is re-enabled.
    zed     = ZEDCamera(camera_id=CAMERA_ID,
                        y_offset=geo.Y_OFFSET_PX if STEREO_DEPTH else 0)
    tracker      = StereoHandTracker(zed)
    pose_tracker = ArmTracker(zed)

    # Physics
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data  = mujoco.MjData(model)

    # Mocap body index for hand_proxy
    mid = model.body("hand_proxy").mocapid[0]

    # Retargeter, arm IK (right + left) and filters
    # ik = None → pas de main LEAP dans cette scène, retargeting doigts désactivé
    ik          = None
    arm_ik      = ArmIKSolver(model, side="right")
    left_arm_ik = ArmIKSolver(model, side="left")
    pos_f         = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)
    joint_f       = OneEuroFilter(JOINT_FREQ, min_cutoff=JOINT_MC, beta=JOINT_BETA)
    orient_f      = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    pitch_f       = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    yaw_f         = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    left_pos_f    = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)
    # Filtre dédié pour le bras droit en repère torse (sensor-space, comme edgard.py)
    right_torso_f = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)

    # Spawn hand + arms at rest position
    arm_offset = _init_hand(model, data, arm_ik, left_arm_ik)

    # Robot morphology references at neutral pose (for human->robot scaling)
    mujoco.mj_forward(model, data)
    r_sh = data.xpos[model.body("arm_right_shoulder_pitch").id].copy()
    l_sh = data.xpos[model.body("arm_left_shoulder_pitch").id].copy()
    r_ee = data.xpos[arm_ik.ee_body_id].copy()
    l_ee = data.xpos[left_arm_ik.ee_body_id].copy()
    _robot_reach_right = float(np.linalg.norm(r_ee - r_sh))
    _robot_reach_left = float(np.linalg.norm(l_ee - l_sh))
    _robot_shoulder_w = float(np.linalg.norm(r_sh - l_sh))

    # Enable clamping from the very first frame (prevents gravity drift
    # during the pre-calibration period)
    arm_ik._last_q = data.qpos[arm_ik.qpos_adr].copy()
    left_arm_ik._last_q = data.qpos[left_arm_ik.qpos_adr].copy()

    # Camera viewer in a separate lightweight process (only imports cv2,
    # NOT mujoco — avoids the Cocoa / OpenGL conflict with mjpython on macOS).
    viewer_proc = None
    if SHOW_CAMERA:
        from _camera_viewer import viewer_loop
        ctx = _mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=2)
        _reset_flag = ctx.Value('i', 0)
        viewer_proc = ctx.Process(target=viewer_loop, args=(_frame_q, _reset_flag), daemon=True)
        viewer_proc.start()

    # print("─" * 60)
    # print("  Binocular Hand Teleoperation — Humanoid Only (bras uniquement)")
    # print("  Main droite → bras droit IK")
    # print("  Main gauche → bras gauche IK")
    # print("  (Pas de main LEAP — retargeting doigts désactivé)")
    # print(f"  Auto-calibration dans {AUTO_CALIB_SEC:.0f}s (ou touche A)")
    # print("  ESC pour quitter.")
    # print("─" * 60)

    import time as _time
    _start_time = _time.monotonic()

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as v:
        while v.is_running():
            _update(data, zed, tracker, pose_tracker, ik, pos_f, joint_f, orient_f, pitch_f, yaw_f,
                    mid, arm_ik, arm_offset, left_arm_ik, left_pos_f, right_torso_f)

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
                arm_ik.clamp_after_step(data)
                left_arm_ik.clamp_after_step(data)
            v.sync()

    # Clean shutdown
    if viewer_proc is not None and _frame_q is not None:
        _frame_q.put(None)
        viewer_proc.join(timeout=3)

    zed.close()
    pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
teleop_edgard_fused_first.py — Hand teleoperation, caméra ZED pointée vers le HAUT.

Différences par rapport au setup frontal (teleop_edgard_copy_copy.py) :

  CAMERA OVERHEAD (pointée vers le plafond) :
    - L'utilisateur se tient AU-DESSUS de la caméra.
    - Pendant la calibration, la paume est orientée vers le BAS (vers la caméra).
    - BASE_QUAT adapté : paume vers le bas dans le repère robot.
    - USE_TORSO_RELATIVE = False : la caméra ne voit plus le torse.

  CHANGEMENT DES AXES DE TRANSLATION (sauf gauche/droite) :
    Old (caméra frontale) :
      sim_x ← pixel U  (inchangé)
      sim_z ← pixel V  (hauteur)
      sim_y ← depth ZED (profondeur avant/arrière)
    New (caméra overhead) :
      sim_x ← pixel U  (inchangé)
      sim_y ← pixel V  (profondeur avant/arrière dans le plan horizontal)
      sim_z ← depth ZED (hauteur au-dessus de la caméra)

  ROTATIONS DU POIGNET :
    Les formules raw_angle / raw_pitch / raw_yaw sont conservées (delta relatif
    à la calibration).  Avec le nouveau BASE_QUAT les axes body-local sont
    réorientés : si un axe part dans le mauvais sens, inverser son signe dans la
    section "Apply in body-local frame" ou ajuster WRIST_SCALE.

Pipeline (30 Hz vision loop) :
  ZED left camera frame (overhead, monocular mode)
    → MediaPipe hand tracking
    → direct angle retargeting (MediaPipe 3-D joint angles → 12 Inspire joints)
    → One Euro filtered mocap position
    → One Euro filtered joint targets
    → MuJoCo position actuators

Run:
    python teleop_edgard_fused_first.py
"""

import os
import faulthandler; faulthandler.enable()
import multiprocessing as _mp
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from vision.camera                       import ZEDCamera
from vision.detectors                    import StereoHandTracker, ArmTracker
import vision.geometry                   as geo
from vision.smoother                     import OneEuroFilter
from robots.inspire_hand.ik_retargeting  import IKRetargeter
from robots.leap_hand.ik_retargeting     import palm_quat
from robots.leap_hand.arm_ik             import ArmIKSolver

# ── Tunable constants ─────────────────────────────────────────────────────────
CAMERA_ID    = 4
N_SUBSTEPS   = 16

# ── Mode toggle ───────────────────────────────────────────────────────────────
STEREO_DEPTH = True

# One Euro Filter — joints
JOINT_FREQ   = 30.0
JOINT_MC     = 1.0
JOINT_BETA   = 0.03

# One Euro Filter — wrist position
POS_FREQ     = 30.0
POS_MC       = 0.8
POS_BETA     = 0.005

# Workspace mapping (legacy, non utilisés directement)
X_SCALE      = 0.3
Z_SCALE      = 0.3

# Caméra overhead : la "profondeur" est maintenant la hauteur au-dessus de la caméra.
# DEPTH_MID_M = hauteur typique de la main pendant la calibration.
DEPTH_MIN_M  = 0.10
DEPTH_MAX_M  = 2.00
DEPTH_MID_M  = 0.60    # hauteur neutre typique au-dessus de la ZED
DEPTH_SCALE  = 0.2     # m de déplacement sim-Z par m de hauteur
TRANS_SCALE  = 6.0
START_Y      = 0.30    # profondeur initiale sim (avant/arrière)
START_Z      = 0.45    # hauteur initiale sim

# Epipolar tolerance
EPIPOLAR_TOL = 40

# ── Wrist rotation ────────────────────────────────────────────────────────────
WRIST_SCALE    = 2.0
WRIST_DZ_RX    = 0.03
WRIST_DZ_RY    = 0.12
WRIST_DZ_RZ    = 0.12
WRIST_MAX_RAD  = 2.0
MOCAP_MAX_STEP = 0.015
RY_POS_BOOST   = 1.2
RY_NEG_BOOST   = 2.0
RZ_RY_DECOUPLE = 0.6

# Morphological calibration
MORPH_SCALE_MIN = 0.60
MORPH_SCALE_MAX = 1.50
MORPH_PRINT_EVERY_SEC = 1.0
ARM_RIGHT_GAIN = 1.60
ARM_LEFT_GAIN  = 1.60

# ── Torso-relative arm control ────────────────────────────────────────────────
# Désactivé : la caméra overhead ne voit pas le torse.
USE_TORSO_RELATIVE = False

# MediaPipe Pose world → robot sim axis mapping (référence, mais non utilisé ici)
# MP world (origin=hips): x=cam-right(person-left), y=up, z=toward-cam(=downward overhead)
_MP_SIGN_X = -1.0
_MP_SIGN_Y =  +1.0
_MP_SIGN_Z = -1.0

# One Euro Filters for wrist angles
WRIST_FREQ     = 30.0
WRIST_MC       = 0.3
WRIST_BETA     = 0.01

HOLD_POSE_SEC  = 1.0

# ── Orientation initiale de la main ──────────────────────────────────────────
# Caméra overhead : pendant la calibration la paume est tournée vers le BAS
# (vers la caméra au sol). BASE_QUAT = paume vers le bas dans le repère robot.
BASE_QUAT = np.array([-0.5, 0.5, -0.5, -0.5])  # paume vers le bas (Ry -90° en body frame)

# ── Handedness filter ─────────────────────────────────────────────────────────
TARGET_HAND   = "Left"   # main physique droite, caméra non miroir
OTHER_HAND    = "Right"  # main physique gauche

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
    return -q if np.dot(q, ref) < 0 else q


def _mp_world_to_robot(delta_mp: np.ndarray) -> np.ndarray:
    """Convertit un vecteur delta MediaPipe Pose world → repère robot sim.
    Non utilisé (USE_TORSO_RELATIVE=False) mais conservé pour compatibilité.
    """
    return np.array([
        _MP_SIGN_X * delta_mp[0],
        _MP_SIGN_Z * delta_mp[2],
        _MP_SIGN_Y * delta_mp[1],
    ])


# ── Hand selection helpers ─────────────────────────────────────────────────────
def _find_hand_by_label(result, label: str):
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None, 0.0
    for i, hand_class in enumerate(result.multi_handedness):
        cls = hand_class.classification[0]
        if cls.label == label:
            return result.multi_hand_landmarks[i], cls.score
    return None, 0.0


def _find_other_hand(result, excluded_label: str):
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


def _find_closest_hand_to_ref(result, ref_u: float, ref_v: float, w: int, h: int):
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None, 0.0
    best_i = None
    best_d2 = float("inf")
    for i, lms in enumerate(result.multi_hand_landmarks):
        wr = lms.landmark[0]
        u = float(wr.x) * w
        v = float(wr.y) * h
        d2 = (u - ref_u) ** 2 + (v - ref_v) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    if best_i is None:
        return None, 0.0
    score = float(result.multi_handedness[best_i].classification[0].score)
    return result.multi_hand_landmarks[best_i], score


def _dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pose_morphology(pose_result):
    """Non utilisé avec caméra overhead (torse non visible), conservé pour compatibilité."""
    if pose_result is None or pose_result.pose_world_landmarks is None:
        return None
    lm = pose_result.pose_world_landmarks.landmark

    def p(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)

    l_sh, l_el, l_wr = p(11), p(13), p(15)
    r_sh, r_el, r_wr = p(12), p(14), p(16)
    l_upper = _dist3(l_sh, l_el)
    l_fore  = _dist3(l_el, l_wr)
    r_upper = _dist3(r_sh, r_el)
    r_fore  = _dist3(r_el, r_wr)
    shoulder_w = _dist3(l_sh, r_sh)
    vals = np.array([l_upper, l_fore, r_upper, r_fore, shoulder_w], dtype=float)
    if np.any(vals < 0.05) or np.any(vals > 0.80):
        return None
    return {
        "left_reach_h":  l_upper + l_fore,
        "right_reach_h": r_upper + r_fore,
        "shoulder_w_h":  shoulder_w,
    }


# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "leap_hand", "scene_inspire_hand_fused.xml")

_G1_XML    = "/home/edgard/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
_G1_POS    = [3.0, 0.0, -0.036]
_G1_QUAT   = [0.7071068, 0.0, 0.0, 0.7071068]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_hand(model: mujoco.MjModel, data: mujoco.MjData,
               arm_ik: 'ArmIKSolver | None' = None,
               left_arm_ik: 'ArmIKSolver | None' = None) -> np.ndarray:
    """Teleport les mains inspire et positionne les bras en pose de repos."""
    if arm_ik is not None:
        sp_jid = model.joint("arm_right_shoulder_pitch_joint").id
        ep_jid = model.joint("arm_right_elbow_pitch_joint").id
        data.qpos[model.jnt_qposadr[sp_jid]] = np.pi / 2
        data.qpos[model.jnt_qposadr[ep_jid]] = -np.pi / 4

    if left_arm_ik is not None:
        sp_jid = model.joint("arm_left_shoulder_pitch_joint").id
        ep_jid = model.joint("arm_left_elbow_pitch_joint").id
        data.qpos[model.jnt_qposadr[sp_jid]] = -np.pi / 2
        data.qpos[model.jnt_qposadr[ep_jid]] = np.pi / 4

    mid   = model.body("hand_proxy").mocapid[0]
    pos_r = np.array([ 0.15, START_Y, START_Z])
    pos_l = np.array([-0.15, START_Y, START_Z])

    data.mocap_pos[mid]  = pos_r
    data.mocap_quat[mid] = BASE_QUAT.copy()

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_free")
    if jid != -1:
        addr = model.jnt_qposadr[jid]
        data.qpos[addr:addr+3]   = pos_r
        data.qpos[addr+3:addr+7] = BASE_QUAT.copy()

    mid_l_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy_l")
    if mid_l_bid != -1:
        mid_l = model.body("hand_proxy_l").mocapid[0]
        data.mocap_pos[mid_l]  = pos_l
        data.mocap_quat[mid_l] = BASE_QUAT.copy()

    jid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_free_l")
    if jid_l != -1:
        addr = model.jnt_qposadr[jid_l]
        data.qpos[addr:addr+3]   = pos_l
        data.qpos[addr+3:addr+7] = BASE_QUAT.copy()

    for ik_solver in (arm_ik, left_arm_ik):
        if ik_solver is not None:
            for i, act_idx in enumerate(ik_solver.act_indices):
                data.ctrl[act_idx] = data.qpos[ik_solver.qpos_adr[i]]

    mujoco.mj_forward(model, data)

    if arm_ik is not None:
        return pos_r - data.xpos[arm_ik.ee_body_id].copy()
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
            right_torso_f:'OneEuroFilter | None' = None,
            mid_l:        'int | None' = None,
            left_joint_f: 'OneEuroFilter | None' = None,
            left_pos2_f:  'OneEuroFilter | None' = None,
            left_orient_f:'OneEuroFilter | None' = None,
            left_pitch_f2:'OneEuroFilter | None' = None,
            left_yaw_f2:  'OneEuroFilter | None' = None) -> None:
    """
    Single-frame update : capture → detect → retarget → actuate.

    Caméra overhead : axes de position remappés.
      sim_x ← pixel U  (gauche/droite — inchangé)
      sim_y ← pixel V  (profondeur avant/arrière dans le plan horizontal)
      sim_z ← depth ZED (hauteur au-dessus de la caméra)
    """
    global _wrist_ref_angle, _wrist_calib_count, _pitch_ref_angle, _pitch_calib_count, _yaw_ref_angle, _yaw_calib_count, _last_hand_time, _calibrate_flag, _left_calib_flag, _left_mocap_teleport, _left_ref_pos, _left_ee_start, _last_left_target, _left_mono_ref_span
    global _left_wrist_ref_angle, _left_pitch_ref_angle, _left_yaw_ref_angle
    global _right_ref_pos, _right_ee_start, _arm_scale_left, _arm_scale_right
    global _g1_right_ref_pos, _g1_left_ref_pos, _g1_right_ee_start, _g1_left_ee_start
    global _robot_reach_left, _robot_reach_right, _robot_shoulder_w
    global _last_morph_print_time, _last_depth_log_time
    global _mp_right_hand_rel_ref, _mp_left_hand_rel_ref, _mp_left_smoothed_rel
    global _last_target_wrist_uv
    import time as _time
    ik_info = None
    left_ik_info = None
    n_hand = 12
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    if PRINT_RIGHT_ARM_DEBUG and arm_ik is not None:
        r_ee = data.xpos[arm_ik.ee_body_id]
        rq   = data.qpos[arm_ik.qpos_adr]
        rdeg = np.degrees(rq)

    # ── Reset (touche R) ──────────────────────────────────────────────────
    if _reset_flag is not None and _reset_flag.value:
        _reset_flag.value = 0
        _wrist_ref_angle = None; _wrist_calib_count = 0
        _pitch_ref_angle = None; _pitch_calib_count = 0
        _yaw_ref_angle   = None; _yaw_calib_count  = 0
        _left_wrist_ref_angle = None
        _left_pitch_ref_angle = None
        _left_yaw_ref_angle   = None
        orient_f.reset(); pitch_f.reset(); yaw_f.reset()
        if left_orient_f is not None: left_orient_f.reset()
        if left_pitch_f2 is not None: left_pitch_f2.reset()
        if left_yaw_f2   is not None: left_yaw_f2.reset()
        joint_f.reset()
        data.ctrl[:] = 0.0
        data.qvel[:] = 0.0
        _left_ref_pos        = None
        _left_mocap_teleport = False
        _last_left_target    = None
        _left_mono_ref_span  = None
        _right_ref_pos = None
        _right_ee_start = None
        _last_target_wrist_uv = None
        _mp_right_hand_rel_ref = None
        _mp_left_hand_rel_ref  = None
        _g1_right_ref_pos  = None
        _g1_left_ref_pos   = None
        _g1_right_ee_start = None
        _g1_left_ee_start  = None
        _mp_left_smoothed_rel  = None
        if left_pos_f is not None:
            left_pos_f.reset()
        if right_torso_f is not None:
            right_torso_f.reset()
        _init_hand(data.model, data, arm_ik, left_arm_ik)

    h, w, _ = frame_l.shape
    res_l, res_r = tracker.process(frame_l, frame_r)
    pose_res = pose_tracker.process(frame_l) if pose_tracker is not None else None
    morph_live = _pose_morphology(pose_res)
    now = _time.monotonic()
    if now - _last_morph_print_time >= MORPH_PRINT_EVERY_SEC:
        _last_morph_print_time = now

    # ── Find each hand by handedness label ──────────────────────────────
    lm_target, target_score = _find_hand_by_label(res_l, TARGET_HAND)
    lm_other,  other_score  = _find_hand_by_label(res_l, OTHER_HAND)
    if lm_other is None:
        lm_other, other_score = _find_other_hand(res_l, TARGET_HAND)
    if lm_target is None and _last_target_wrist_uv is not None:
        ref_u, ref_v = _last_target_wrist_uv
        lm_target, target_score = _find_closest_hand_to_ref(res_l, ref_u, ref_v, w, h)

    lm_target_r, _ = _find_hand_by_label(res_r, TARGET_HAND)
    lm_other_r,  _ = _find_hand_by_label(res_r, OTHER_HAND)
    if lm_other_r is None:
        lm_other_r, _ = _find_other_hand(res_r, TARGET_HAND)

    elapsed_since_start = _time.monotonic() - _start_time if _start_time else 0

    if lm_target is None:
        elapsed = _time.monotonic() - _last_hand_time
        holding = elapsed < HOLD_POSE_SEC and _last_hand_time > 0

        if not holding:
            orient_f.reset(); pitch_f.reset(); yaw_f.reset()
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
    _last_target_wrist_uv = (float(lm_target.landmark[0].x) * w,
                             float(lm_target.landmark[0].y) * h)

    lm_l = lm_target.landmark
    cam   = geo.ZED2I

    # ── Palm center position (avg of wrist + 4 MCP) ────────────────────
    _palm_ids = (0, 5, 9, 13, 17)
    u_w = sum(lm_l[i].x for i in _palm_ids) / len(_palm_ids) * w
    v_w = sum(lm_l[i].y for i in _palm_ids) / len(_palm_ids) * h

    # OVERHEAD — axe X (gauche/droite) : inchangé.
    # Axe Y (profondeur) : vient du pixel V (dans l'ancien setup c'était sim_z).
    # Axe Z (hauteur)    : vient de la depth ZED (dans l'ancien setup c'était sim_y).
    # Référence de distance en mono = START_Z (hauteur typique main au-dessus caméra).
    sim_x = -(u_w - cam.cx) / cam.fx * START_Z * TRANS_SCALE
    sim_y =  START_Y + (-(v_w - cam.cy) / cam.fy * START_Z) * TRANS_SCALE
    sim_z =  START_Z   # hauteur fixe en mode mono (depth inconnue sans ZED)

    # ── Height axis (sim_Z) : ZED point cloud ────────────────────────────
    depth_cm   = None
    hud_mode   = "MONO [L]"
    hud_col    = (0, 165, 255)
    hud_detail = ""

    if STEREO_DEPTH:
        pc = zed.get_point_cloud()
        if pc is not None:
            try:
                import pyzed.sl as sl
                px_pc = int(np.clip(u_w, 0, w - 1))
                py_pc = int(np.clip(v_w, 0, h - 1))
                err_pc, pc_val = pc.get_value(px_pc, py_pc)
                z_m = float(pc_val[2])
                # Caméra overhead : ZED Z pointe vers le haut → hauteur dans le monde.
                depth_m = abs(z_m)   # hauteur de la main au-dessus de la caméra
                if (err_pc == sl.ERROR_CODE.SUCCESS and
                        not np.isnan(z_m) and not np.isinf(z_m) and
                        DEPTH_MIN_M < depth_m < DEPTH_MAX_M):
                    x_m = float(pc_val[0])
                    y_m = float(pc_val[1])
                    # ZED X → latéral (inchangé)
                    sim_x = -x_m * TRANS_SCALE
                    # ZED Y → profondeur avant/arrière (était hauteur dans ancien setup)
                    sim_y = START_Y + y_m * TRANS_SCALE
                    # ZED Z (depth_m) → hauteur (était profondeur dans ancien setup)
                    sim_z = START_Z + (depth_m - DEPTH_MID_M) * DEPTH_SCALE * TRANS_SCALE
                    depth_cm   = depth_m * 100
                    hud_mode   = "ZED DEPTH"
                    hud_col    = (0, 220, 0)
                    hud_detail = f"height={depth_m:.2f}m"
                else:
                    hud_mode   = "ZED NO DEPTH"
                    hud_col    = (0, 165, 255)
                    hud_detail = f"h={depth_m:.2f}m (hors plage)" if not np.isnan(z_m) else "NaN"
            except Exception as e:
                hud_mode   = "ZED ERR"
                hud_col    = (0, 0, 255)
                hud_detail = str(e)

    # ── Log console (1 Hz) ────────────────────────────────────────────────
    _depth_now = _time.monotonic()
    if _depth_now - _last_depth_log_time >= 1.0:
        _last_depth_log_time = _depth_now
        if not STEREO_DEPTH:
            print("[OVERHEAD] STEREO_DEPTH=False — sim_Z fixe")
        elif depth_cm is not None:
            _l_mocap_z = f"  L_mocap_Z={float(data.mocap_pos[mid_l][2]):+.3f} m" if mid_l is not None else ""
            print(f"[OVERHEAD R] height={depth_cm/100:.3f} m  sim_Z={sim_z:+.3f} m  "
                  f"mocap_Z={float(data.mocap_pos[mid][2]):+.3f} m{_l_mocap_z}")
        else:
            print(f"[OVERHEAD R] {hud_mode}: {hud_detail if hud_detail else 'pas de PC valide'}  "
                  f"sim_Z={sim_z:+.3f} m  mocap_Z={float(data.mocap_pos[mid][2]):+.3f} m")

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

    # ── Auto-calibration ──────────────────────────────────────────────────
    elapsed_since_start = _time.monotonic() - _start_time if _start_time else 0
    auto_trigger = (_wrist_ref_angle is None and elapsed_since_start >= AUTO_CALIB_SEC)
    if _calibrate_flag or auto_trigger:
        _wrist_ref_angle = raw_angle
        _pitch_ref_angle = raw_pitch
        _yaw_ref_angle   = raw_yaw
        orient_f.reset(); pitch_f.reset(); yaw_f.reset()
        pos_f.reset(); joint_f.reset()
        _calibrate_flag      = False
        _left_calib_flag     = True
        _left_mocap_teleport = True
        _right_ref_pos = np.array([sim_x, sim_y, sim_z], dtype=float)
        mujoco.mj_forward(data.model, data)
        _right_ee_start = data.xpos[arm_ik.ee_body_id].copy() if arm_ik is not None else None
        _mp_right_hand_rel_ref = None
        _mp_left_hand_rel_ref  = None
        _mp_left_smoothed_rel  = None
        if right_torso_f is not None:
            right_torso_f.reset()
        _arm_scale_left  = 1.0
        _arm_scale_right = 1.0

    # ── Before calibration: hand frozen ──────────────────────────────────
    if _wrist_ref_angle is None:
        wrist_x = wrist_y = wrist_z = wrist_z_raw = 0.0
    else:
        # ── Mocap position ─────────────────────────────────────────────
        raw_pos = np.array([sim_x, sim_y, sim_z])
        new_pos = pos_f(raw_pos)
        delta_pos = new_pos - data.mocap_pos[mid]
        dist = np.linalg.norm(delta_pos)
        if dist > MOCAP_MAX_STEP:
            new_pos = data.mocap_pos[mid] + delta_pos * (MOCAP_MAX_STEP / dist)
        data.mocap_pos[mid] = new_pos

        # Roll (Rz) — rotation dans le plan image = rotation autour de Z monde (inchangé)
        delta = (raw_angle - _wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta = float(np.clip(delta, -0.9, 0.9))
        delta = float(orient_f(np.array([delta]))[0])
        if abs(delta) < WRIST_DZ_RZ:
            delta = 0.0
        else:
            delta = np.sign(delta) * (abs(delta) - WRIST_DZ_RZ)
        wrist_z = float(np.clip(delta * WRIST_SCALE * 0.9, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Pitch (Rx)
        delta_p = (raw_pitch - _pitch_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta_p = float(np.clip(delta_p, -0.9, 0.9))
        delta_p = float(pitch_f(np.array([delta_p]))[0])
        if abs(delta_p) < WRIST_DZ_RX:
            delta_p = 0.0
        else:
            delta_p = np.sign(delta_p) * (abs(delta_p) - WRIST_DZ_RX)
        wrist_x = float(np.clip(-delta_p * WRIST_SCALE * 5.0, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Yaw (Ry)
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

        wrist_z_raw = wrist_z
        reduction = RZ_RY_DECOUPLE * abs(wrist_y)
        if abs(wrist_z) > reduction:
            wrist_z = wrist_z - np.sign(wrist_z) * reduction
        else:
            wrist_z = 0.0

        # Rotation incrémentale en repère body-local sur BASE_QUAT (paume bas).
        # NOTA : avec la caméra overhead les axes body-local sont réorientés.
        # Si un axe part dans le mauvais sens, inverser le signe de wrist_x/y/z
        # ci-dessus ou inverser half_x/y/z ici.
        half_x = wrist_z / 2.0
        dq_x = np.array([np.cos(half_x), np.sin(half_x), 0.0, 0.0])
        half_y = wrist_y / 2.0
        dq_y = np.array([np.cos(half_y), 0.0, np.sin(half_y), 0.0])
        half_z = wrist_x / 2.0
        dq_z = np.array([np.cos(half_z), 0.0, 0.0, np.sin(half_z)])
        q = _quat_mul(BASE_QUAT, dq_x)
        q = _quat_mul(q, dq_y)
        q = _quat_mul(q, dq_z)
        q = q / np.linalg.norm(q)
        data.mocap_quat[mid] = q

        # ── Arm IK : bras droit (pixel-based uniquement, torse non visible) ──
        ik_info = None
        if arm_ik is not None:
            arm_target_pos = None
            if _right_ref_pos is not None and _right_ee_start is not None:
                right_delta = data.mocap_pos[mid] - _right_ref_pos
                arm_target_pos = _right_ee_start + right_delta * _arm_scale_right * ARM_RIGHT_GAIN
            else:
                arm_target_pos = data.mocap_pos[mid] - arm_offset
            ik_info = arm_ik.solve(data.model, data, arm_target_pos, q)

        # ── Finger retargeting (main droite) ──────────────────────────────
        if ik is not None:
            q_raw    = ik.retarget(None, lm_l)
            q_smooth = joint_f(q_raw)
            for i in range(n_hand):
                jid_r = data.model.actuator_trnid[i, 0]
                lo  = data.model.jnt_range[jid_r, 0]
                hi  = data.model.jnt_range[jid_r, 1]
                q_smooth[i] = float(np.clip(q_smooth[i], lo, hi))
            data.ctrl[:n_hand] = q_smooth

    # ── Left hand Inspire : position + finger retargeting ─────────────────
    if mid_l is not None and lm_other is not None and _wrist_ref_angle is not None:
        lm_other_l = lm_other.landmark
        cam = geo.ZED2I

        _palm_ids = (0, 5, 9, 13, 17)
        u_lh2 = sum(lm_other_l[i].x for i in _palm_ids) / len(_palm_ids) * w
        v_lh2 = sum(lm_other_l[i].y for i in _palm_ids) / len(_palm_ids) * h

        # Même mapping overhead que la main droite
        lh2_x = -(u_lh2 - cam.cx) / cam.fx * START_Z * TRANS_SCALE
        lh2_y =  START_Y + (-(v_lh2 - cam.cy) / cam.fy * START_Z) * TRANS_SCALE
        lh2_z =  START_Z

        if STEREO_DEPTH:
            pc_l = zed.get_point_cloud()
            if pc_l is not None:
                try:
                    import pyzed.sl as sl
                    px_lh2 = int(np.clip(u_lh2, 0, w - 1))
                    py_lh2 = int(np.clip(v_lh2, 0, h - 1))
                    err_lh2, pc_lh2_val = pc_l.get_value(px_lh2, py_lh2)
                    z_lh2   = float(pc_lh2_val[2])
                    depth_lh2 = abs(z_lh2)
                    if (err_lh2 == sl.ERROR_CODE.SUCCESS and
                            not np.isnan(z_lh2) and not np.isinf(z_lh2) and
                            DEPTH_MIN_M < depth_lh2 < DEPTH_MAX_M):
                        lh2_x = -float(pc_lh2_val[0]) * TRANS_SCALE
                        lh2_y =  START_Y + float(pc_lh2_val[1]) * TRANS_SCALE
                        lh2_z =  START_Z + (depth_lh2 - DEPTH_MID_M) * DEPTH_SCALE * TRANS_SCALE
                except Exception:
                    pass

        raw_lh2 = np.array([lh2_x, lh2_y, lh2_z])
        if left_pos2_f is not None:
            new_lh2 = left_pos2_f(raw_lh2)
        else:
            new_lh2 = raw_lh2

        if _left_mocap_teleport:
            data.mocap_pos[mid_l] = new_lh2
            if left_pos2_f is not None:
                left_pos2_f.reset()
                left_pos2_f(new_lh2)
            _left_mocap_teleport = False
        else:
            d2 = new_lh2 - data.mocap_pos[mid_l]
            if np.linalg.norm(d2) > MOCAP_MAX_STEP:
                new_lh2 = data.mocap_pos[mid_l] + d2 * (MOCAP_MAX_STEP / np.linalg.norm(d2))
            data.mocap_pos[mid_l] = new_lh2

        # ── Left wrist rotation ───────────────────────────────────────────
        lm_ol = lm_other.landmark
        raw_angle_l = np.arctan2(lm_ol[13].y - lm_ol[5].y, lm_ol[13].x - lm_ol[5].x)
        dy_pl = lm_ol[9].y - lm_ol[0].y
        dz_pl = lm_ol[9].z - lm_ol[0].z
        raw_pitch_l = np.arctan2(dz_pl, dy_pl)
        dx_yl = lm_ol[17].x - lm_ol[5].x
        dz_yl = lm_ol[17].z - lm_ol[5].z
        raw_yaw_l = dz_yl / max(abs(dx_yl), 0.01)

        if _left_wrist_ref_angle is None and _wrist_ref_angle is not None:
            _left_wrist_ref_angle = raw_angle_l
            _left_pitch_ref_angle = raw_pitch_l
            _left_yaw_ref_angle   = raw_yaw_l
            if left_orient_f is not None: left_orient_f.reset()
            if left_pitch_f2 is not None: left_pitch_f2.reset()
            if left_yaw_f2   is not None: left_yaw_f2.reset()

        if _left_wrist_ref_angle is None:
            data.mocap_quat[mid_l] = BASE_QUAT.copy()
        else:
            dl = (raw_angle_l - _left_wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
            dl = float(np.clip(dl, -0.9, 0.9))
            if left_orient_f is not None:
                dl = float(left_orient_f(np.array([dl]))[0])
            dl = 0.0 if abs(dl) < WRIST_DZ_RZ else np.sign(dl) * (abs(dl) - WRIST_DZ_RZ)
            wl_z = float(np.clip(dl * WRIST_SCALE * 0.9, -WRIST_MAX_RAD, WRIST_MAX_RAD))

            dp_l = (raw_pitch_l - _left_pitch_ref_angle + np.pi) % (2 * np.pi) - np.pi
            dp_l = float(np.clip(dp_l, -0.9, 0.9))
            if left_pitch_f2 is not None:
                dp_l = float(left_pitch_f2(np.array([dp_l]))[0])
            dp_l = 0.0 if abs(dp_l) < WRIST_DZ_RX else np.sign(dp_l) * (abs(dp_l) - WRIST_DZ_RX)
            wl_x = float(np.clip(-dp_l * WRIST_SCALE * 5.0, -WRIST_MAX_RAD, WRIST_MAX_RAD))

            dy_l = (raw_yaw_l - _left_yaw_ref_angle + np.pi) % (2 * np.pi) - np.pi
            dy_l = -dy_l
            dy_l *= RY_POS_BOOST if dy_l > 0 else RY_NEG_BOOST
            dy_l = float(np.clip(dy_l, -0.9, 0.9))
            if left_yaw_f2 is not None:
                dy_l = float(left_yaw_f2(np.array([dy_l]))[0])
            dy_l = 0.0 if abs(dy_l) < WRIST_DZ_RY else np.sign(dy_l) * (abs(dy_l) - WRIST_DZ_RY)
            wl_y = float(np.clip(dy_l * WRIST_SCALE, -WRIST_MAX_RAD, WRIST_MAX_RAD))

            red_l = RZ_RY_DECOUPLE * abs(wl_y)
            wl_z = (wl_z - np.sign(wl_z) * red_l) if abs(wl_z) > red_l else 0.0

            hx_l = wl_z / 2.0
            dqx_l = np.array([np.cos(hx_l), np.sin(hx_l), 0.0, 0.0])
            hy_l = wl_y / 2.0
            dqy_l = np.array([np.cos(hy_l), 0.0, np.sin(hy_l), 0.0])
            hz_l = wl_x / 2.0
            dqz_l = np.array([np.cos(hz_l), 0.0, 0.0, np.sin(hz_l)])
            q_l = _quat_mul(BASE_QUAT, dqx_l)
            q_l = _quat_mul(q_l, dqy_l)
            q_l = _quat_mul(q_l, dqz_l)
            data.mocap_quat[mid_l] = q_l / np.linalg.norm(q_l)

        if ik is not None and left_joint_f is not None:
            q_raw_l    = ik.retarget(None, lm_other_l)
            q_smooth_l = left_joint_f(q_raw_l)
            for i in range(n_hand):
                jid_l2 = data.model.actuator_trnid[n_hand + i, 0]
                lo = data.model.jnt_range[jid_l2, 0]
                hi = data.model.jnt_range[jid_l2, 1]
                q_smooth_l[i] = float(np.clip(q_smooth_l[i], lo, hi))
            data.ctrl[n_hand:n_hand * 2] = q_smooth_l

    # ── Left arm IK (pixel-based only) ────────────────────────────────────
    if left_arm_ik is not None and lm_other is not None:
        lm_left = lm_other.landmark
        cam = geo.ZED2I

        _palm_ids = (0, 5, 9, 13, 17)
        u_lh = sum(lm_left[i].x for i in _palm_ids) / len(_palm_ids) * w
        v_lh = sum(lm_left[i].y for i in _palm_ids) / len(_palm_ids) * h

        lh_x = -(u_lh - cam.cx) / cam.fx * START_Z * TRANS_SCALE
        lh_y = START_Y + (-(v_lh - cam.cy) / cam.fy * START_Z) * TRANS_SCALE
        lh_z = START_Z

        lh_span = np.hypot(lm_left[9].x * w - lm_left[0].x * w,
                           lm_left[9].y * h - lm_left[0].y * h)

        depth_ok_l = False
        if STEREO_DEPTH:
            pc = zed.get_point_cloud()
            if pc is not None:
                try:
                    import pyzed.sl as sl
                    px_lh = int(np.clip(u_lh, 0, w - 1))
                    py_lh = int(np.clip(v_lh, 0, h - 1))
                    err_lh, pc_lh = pc.get_value(px_lh, py_lh)
                    z_lh = float(pc_lh[2])
                    depth_lh = abs(z_lh)
                    if (err_lh == sl.ERROR_CODE.SUCCESS and
                            not np.isnan(z_lh) and not np.isinf(z_lh) and
                            DEPTH_MIN_M < depth_lh < DEPTH_MAX_M):
                        lh_x = -float(pc_lh[0]) * TRANS_SCALE
                        lh_y =  START_Y + float(pc_lh[1]) * TRANS_SCALE
                        lh_z =  START_Z + (depth_lh - DEPTH_MID_M) * DEPTH_SCALE * TRANS_SCALE
                        depth_ok_l = True
                except Exception:
                    pass

        if not depth_ok_l and _left_mono_ref_span is not None and lh_span > 10:
            mono_depth_m = DEPTH_MID_M * _left_mono_ref_span / lh_span
            mono_depth_m = float(np.clip(mono_depth_m, DEPTH_MIN_M, DEPTH_MAX_M))
            lh_z = START_Z + (mono_depth_m - DEPTH_MID_M) * DEPTH_SCALE * TRANS_SCALE

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

        morph_txt = f"MORPH L={_arm_scale_left:.2f} R={_arm_scale_right:.2f}"
        morph_col = (220, 180, 0) if _arm_scale_left != 1.0 else (100, 100, 100)
        cv2.putText(frame_l, morph_txt, (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, morph_col, 2)

        cv2.putText(frame_l, hud_mode, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2)
        if hud_detail:
            cv2.putText(frame_l, hud_detail, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        rh_col = (0, 220, 0) if lm_target is not None else (0, 0, 255)
        rh_txt = f"R.hand: OK ({target_score:.0%})" if lm_target is not None else "R.hand: ---"
        lh_col = (0, 220, 0) if lm_other is not None else (100, 100, 100)
        lh_txt = f"L.hand: OK ({other_score:.0%})" if lm_other is not None else "L.hand: ---"
        cv2.putText(frame_l, rh_txt, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rh_col, 2)
        cv2.putText(frame_l, lh_txt, (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lh_col, 2)

        if _wrist_ref_angle is None:
            remaining = max(0, AUTO_CALIB_SEC - elapsed_since_start)
            calib_txt = f"CALIBRATION DANS {remaining:.1f}s"
            cv2.putText(frame_l, calib_txt, (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if depth_cm is not None:
            depth_str = f"HEIGHT: {depth_cm:.0f} cm"
            d_col = (0, 220, 0)
        else:
            depth_str = "HEIGHT: --- (hors plage ou MONO)"
            d_col = (0, 165, 255)
        txt_size = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame_l, depth_str, (w - txt_size[0] - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_col, 2)
        simz_str = f"sim_Z: {sim_z:.3f} m"
        simz_col = (0, 220, 0) if depth_cm is not None else (0, 165, 255)
        ts2 = cv2.getTextSize(simz_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame_l, simz_str, (w - ts2[0] - 20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, simz_col, 1)
        if _wrist_ref_angle is not None:
            mocz_str = f"mocap_Z: {float(data.mocap_pos[mid][2]):.3f} m"
            ts3 = cv2.getTextSize(mocz_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame_l, mocz_str, (w - ts3[0] - 20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        wrist_x_safe = wrist_x if _wrist_ref_angle is not None else 0.0
        wrist_y_safe = wrist_y if _wrist_ref_angle is not None else 0.0
        wrist_z_safe = wrist_z if _wrist_ref_angle is not None else 0.0
        wrist_z_raw_safe = wrist_z_raw if _wrist_ref_angle is not None else 0.0
        rx_deg = float(np.degrees(wrist_x_safe))
        ry_deg = float(np.degrees(wrist_y_safe))
        rz_deg = float(np.degrees(wrist_z_safe))
        rz_raw_deg = float(np.degrees(wrist_z_raw_safe))

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

        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            n_hands_r = len(res_r.multi_hand_landmarks) if res_r.multi_hand_landmarks else 0
            r_det = f"R.cam: {n_hands_r} hand(s)"
            r_col = (0, 220, 0) if n_hands_r > 0 else (0, 0, 255)
            cv2.putText(frame_r, r_det, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_col, 2)

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


# ── Global state ──────────────────────────────────────────────────────────────
_wrist_ref_angle = None
_pitch_ref_angle = None
_yaw_ref_angle   = None
_wrist_calib_count = 0
_pitch_calib_count = 0
_yaw_calib_count   = 0
_left_wrist_ref_angle = None
_left_pitch_ref_angle = None
_left_yaw_ref_angle   = None
_last_hand_time = 0.0
_calibrate_flag = False
_reset_flag = None
_frame_q = None
_show_counter = 0
_SHOW_EVERY = 5
_VIEWER_SCALE = 0.35

_left_calib_flag    = False
_left_mocap_teleport = False
_left_ref_pos = None
_left_ee_start = None
_last_left_target = None
_left_mono_ref_span = None
_right_ref_pos = None
_right_ee_start = None
_last_target_wrist_uv = None

_g1_right_ref_pos  = None
_g1_left_ref_pos   = None
_g1_right_ee_start = None
_g1_left_ee_start  = None

_mp_right_hand_rel_ref = None
_mp_left_hand_rel_ref  = None
_mp_left_smoothed_rel  = None

_arm_scale_left  = 1.0
_arm_scale_right = 1.0
_robot_reach_left  = None
_robot_reach_right = None
_robot_shoulder_w  = None
_last_morph_print_time = 0.0
_last_depth_log_time   = 0.0

AUTO_CALIB_SEC = 10
_start_time = None


def _show(frame):
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
    global _calibrate_flag, _left_calib_flag
    if keycode == 65:  # GLFW_KEY_A
        _calibrate_flag = True
        _left_calib_flag = True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _frame_q, _start_time, _robot_reach_left, _robot_reach_right, _robot_shoulder_w
    global _g1_right_ref_pos, _g1_left_ref_pos, _g1_right_ee_start, _g1_left_ee_start

    zed          = ZEDCamera(camera_id=CAMERA_ID,
                             y_offset=geo.Y_OFFSET_PX if STEREO_DEPTH else 0)
    tracker      = StereoHandTracker(zed)
    pose_tracker = ArmTracker(zed)

    if _G1_XML is not None:
        _scene_spec = mujoco.MjSpec.from_file(_SCENE_XML)
        _g1_spec    = mujoco.MjSpec.from_file(_G1_XML)
        _g1_frame      = _scene_spec.worldbody.add_frame()
        _g1_frame.pos  = _G1_POS
        _g1_frame.quat = _G1_QUAT
        _scene_spec.attach(_g1_spec, prefix="g1_", suffix="", frame=_g1_frame)
        model = _scene_spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data = mujoco.MjData(model)

    if _G1_XML is not None:
        for _gi in range(model.ngeom):
            _bid  = model.geom_bodyid[_gi]
            _bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _bid)
            if _bname and _bname.startswith("g1_"):
                model.geom_contype[_gi]    = 0
                model.geom_conaffinity[_gi] = 0

    _G1_ARM_KW = ("shoulder", "elbow", "wrist")
    _g1_qpos_slices: list[slice] = []
    _g1_dof_slices:  list[slice] = []
    _g1_qpos_init:   list        = []
    if _G1_XML is not None:
        mujoco.mj_forward(model, data)
        for _ji in range(model.njnt):
            _jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, _ji)
            if not (_jname and _jname.startswith("g1_")):
                continue
            if any(_kw in _jname for _kw in _G1_ARM_KW):
                continue
            _jtype  = model.jnt_type[_ji]
            _nq     = 7 if _jtype == mujoco.mjtJoint.mjJNT_FREE else (4 if _jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
            _nv     = 6 if _jtype == mujoco.mjtJoint.mjJNT_FREE else (3 if _jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
            _qa     = int(model.jnt_qposadr[_ji])
            _da     = int(model.jnt_dofadr[_ji])
            _g1_qpos_slices.append(slice(_qa, _qa + _nq))
            _g1_dof_slices.append(slice(_da, _da + _nv))
            _g1_qpos_init.append(data.qpos[_qa:_qa + _nq].copy())

    g1_right_arm_ik = None
    g1_left_arm_ik  = None
    if _G1_XML is not None:
        try:
            g1_right_arm_ik = ArmIKSolver(model, side="g1_right",
                                           damping=5e-3, ik_step=0.6,
                                           ik_max_iters=2, recovery_max_iters=6)
            g1_left_arm_ik  = ArmIKSolver(model, side="g1_left",
                                           damping=5e-3, ik_step=0.6,
                                           ik_max_iters=2, recovery_max_iters=6)
        except Exception as _e:
            print(f"[G1 IK] Impossible d'initialiser : {_e}")

    mid   = model.body("hand_proxy").mocapid[0]
    mid_l_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy_l")
    mid_l = model.body("hand_proxy_l").mocapid[0] if mid_l_bid != -1 else None

    ik          = IKRetargeter(model)
    arm_ik      = None
    left_arm_ik = None
    pos_f         = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)
    joint_f       = OneEuroFilter(JOINT_FREQ, min_cutoff=JOINT_MC, beta=JOINT_BETA)
    orient_f      = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    pitch_f       = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    yaw_f         = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    left_pos_f    = None
    right_torso_f = None
    left_joint_f  = OneEuroFilter(JOINT_FREQ, min_cutoff=JOINT_MC, beta=JOINT_BETA)
    left_pos2_f   = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)
    left_orient_f = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    left_pitch_f2 = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    left_yaw_f2   = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)

    arm_offset = _init_hand(model, data)

    viewer_proc = None
    if SHOW_CAMERA:
        from _hands_viewer_worker import viewer_loop
        ctx = _mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=2)
        _reset_flag = ctx.Value('i', 0)
        viewer_proc = ctx.Process(target=viewer_loop, args=(_frame_q, _reset_flag), daemon=True)
        viewer_proc.start()

    import time as _time
    _start_time = _time.monotonic()

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as v:
        while v.is_running():
            _update(data, zed, tracker, pose_tracker, ik, pos_f, joint_f, orient_f, pitch_f, yaw_f,
                    mid, arm_ik, arm_offset, left_arm_ik, left_pos_f, right_torso_f,
                    mid_l=mid_l, left_joint_f=left_joint_f, left_pos2_f=left_pos2_f,
                    left_orient_f=left_orient_f, left_pitch_f2=left_pitch_f2, left_yaw_f2=left_yaw_f2)

            if g1_right_arm_ik is not None:
                if _wrist_ref_angle is not None and _g1_right_ref_pos is None:
                    _g1_right_ref_pos  = data.mocap_pos[mid].copy()
                    mujoco.mj_forward(model, data)
                    _g1_right_ee_start = data.xpos[g1_right_arm_ik.ee_body_id].copy()
                if _g1_right_ref_pos is not None:
                    _g1_target_r = _g1_right_ee_start + (data.mocap_pos[mid] - _g1_right_ref_pos)
                    g1_right_arm_ik.solve(model, data, _g1_target_r, np.array([1., 0., 0., 0.]))

            if g1_left_arm_ik is not None and mid_l is not None:
                if _wrist_ref_angle is not None and _g1_left_ref_pos is None:
                    _g1_left_ref_pos  = data.mocap_pos[mid_l].copy()
                    mujoco.mj_forward(model, data)
                    _g1_left_ee_start = data.xpos[g1_left_arm_ik.ee_body_id].copy()
                if _g1_left_ref_pos is not None:
                    _g1_target_l = _g1_left_ee_start + (data.mocap_pos[mid_l] - _g1_left_ref_pos)
                    g1_left_arm_ik.solve(model, data, _g1_target_l, np.array([1., 0., 0., 0.]))

            for _ in range(N_SUBSTEPS):
                for _sl, _q0 in zip(_g1_qpos_slices, _g1_qpos_init):
                    data.qpos[_sl] = _q0
                for _sl in _g1_dof_slices:
                    data.qvel[_sl] = 0.0
                mujoco.mj_step(model, data)
                if arm_ik is not None:
                    arm_ik.clamp_after_step(data)
                if left_arm_ik is not None:
                    left_arm_ik.clamp_after_step(data)
                if g1_right_arm_ik is not None:
                    g1_right_arm_ik.clamp_after_step(data)
                if g1_left_arm_ik is not None:
                    g1_left_arm_ik.clamp_after_step(data)
            v.sync()

    if viewer_proc is not None and _frame_q is not None:
        _frame_q.put(None)
        viewer_proc.join(timeout=3)

    zed.close()
    pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

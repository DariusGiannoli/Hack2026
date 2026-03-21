"""
Direct angle-based retargeting: MediaPipe 3D landmarks → LEAP joint angles.

Why not Jacobian IK targets?
  Phase 0 has no wrist orientation tracking, so we can't correctly map a
  MediaPipe image-plane vector to a 3-D world-space fingertip target — the
  required "natural extension" direction of each finger in world space changes
  every time the hand moves or rotates.

  Direct angle mapping avoids the problem entirely:
    1. Compute the bend angle at each MediaPipe joint from its two adjacent
       bone vectors (using all three x/y/z coordinates, so depth is captured).
    2. Scale the angle into the corresponding LEAP joint range.
    3. Done — no plan_data, no IK solver, no world-frame math.

Interface is preserved so mujoco_teleop.py needs no changes.

MediaPipe landmark indices (21 total)
--------------------------------------
WRIST=0
THUMB : CMC=1  MCP=2  IP=3  TIP=4
INDEX : MCP=5  PIP=6  DIP=7  TIP=8
MIDDLE: MCP=9  PIP=10 DIP=11 TIP=12
RING  : MCP=13 PIP=14 DIP=15 TIP=16
"""

import numpy as np
import mujoco
# ── Debug flag ─────────────────────────────────────────────────────────────────────
# Set True to print the running-max bend angle seen at each joint.
# Run with a tight fist — the printed peak is your real _MP_MAX.
DEBUG_ANGLES = False
# ── MediaPipe landmark indices ─────────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP_LM = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP   = 5; INDEX_PIP    = 6; INDEX_DIP = 7; INDEX_TIP  = 8
MIDDLE_MCP  = 9; MIDDLE_PIP   = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP    = 14; RING_DIP   = 15; RING_TIP   = 16

# ── Tuning ─────────────────────────────────────────────────────────────────────
# Maximum inter-bone angle MediaPipe actually produces at a fully-curled joint.
# With model_complexity=1 the observable max is ~1.8 rad (≈103°).
# Too high → fingers never reach full closure in sim.
# Too low  → joints saturate before you fully curl.
_MP_MAX = 1.8   # radians  (was 2.3 — too generous, prevented full closure)

# Practical LEAP upper limits (slightly below XML max to stay away from limits)
_MCP_MAX  = 2.0   # LEAP xml max 2.23
_PIP_MAX  = 1.7   # LEAP xml max 1.885
_DIP_MAX  = 1.8   # LEAP xml max 2.042
_ROT_LIM  = 0.6   # abduction clamp (xml ±1.047)
_CMC_MAX  = 1.8   # th_cmc
_AXL_MAX  = 1.8   # th_axl
_TMCP_MAX = 2.0   # th_mcp
_IPL_MAX  = 1.5   # th_ipl


# ── Helpers ────────────────────────────────────────────────────────────────────
def _lm3(lm, i: int) -> np.ndarray:
    """Landmark → (x, y, z) array. Uses MediaPipe's pseudo-3D z (depth)."""
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between two bone vectors at a joint.

    0   → straight finger (v1 and v2 point the same direction)
    π/2 → 90° bend
    π   → fully curled (theoretical)

    All three MediaPipe coordinates (x, y, z) are used, so finger depth
    contributes to the angle correctly.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _scale(angle: float, leap_max: float) -> float:
    """Map MediaPipe bend ∈ [0, _MP_MAX] → LEAP angle ∈ [0, leap_max]."""
    return float(np.clip(angle / _MP_MAX * leap_max, 0.0, leap_max))


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix → quaternion (w, x, y, z) — MuJoCo convention.
    Numerically stable Shepperd method.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,                 (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def palm_quat(lm) -> np.ndarray:
    """
    Compute the LEAP palm orientation as a MuJoCo quaternion (w, x, y, z)
    from MediaPipe landmarks.

    Palm frame (in MediaPipe camera space):
      long_axis : WRIST → MIDDLE_MCP          (finger direction)
      lat_axis  : INDEX_MCP → RING_MCP         (across knuckles)
      normal    : cross(lat_axis, long_axis)   (palm-outward normal)

    Camera → MuJoCo frame conversion:
      cam X (right)   → muj X (right)      [ no change ]
      cam Z (forward) → muj Y (forward)    [ swap ]
      cam Y (down)    → -muj Z (up)        [ invert + swap ]
    """
    w_lm = _lm3(lm, WRIST)
    imcp = _lm3(lm, INDEX_MCP)
    mmcp = _lm3(lm, MIDDLE_MCP)
    rmcp = _lm3(lm, RING_MCP)

    # Palm basis vectors in MediaPipe (camera) space
    long_v = mmcp - w_lm                            # wrist → middle MCP
    lat_v  = rmcp - imcp                            # index MCP → ring MCP
    norm_v = np.cross(lat_v, long_v)               # palm outward normal

    # Orthonormalise (Gram-Schmidt)
    long_v = long_v / (np.linalg.norm(long_v) + 1e-6)
    norm_v = norm_v / (np.linalg.norm(norm_v) + 1e-6)
    lat_v  = np.cross(long_v, norm_v)
    lat_v  = lat_v  / (np.linalg.norm(lat_v)  + 1e-6)

    # Rotation matrix in camera frame: columns = [lat, long, normal]
    R_cam = np.column_stack([lat_v, long_v, norm_v])

    # Camera → MuJoCo frame change-of-basis:
    #   muj_x =  cam_x   (col 0 unchanged)
    #   muj_y =  cam_z   (col 2 → col 1)
    #   muj_z = -cam_y   (col 1 → col 2, negated)
    T = np.array([[1,  0,  0],
                  [0,  0,  1],
                  [0, -1,  0]], dtype=float)
    R_muj = T @ R_cam

    return _rot_to_quat(R_muj)


# ── Main class ─────────────────────────────────────────────────────────────────
class IKRetargeter:
    """
    Maps MediaPipe hand landmarks → 16 LEAP joint angles.

    plan_data and palm_xpos are accepted for API compatibility but are unused;
    retargeting is purely angle-based and requires no world-frame information.

    Usage
    -----
        ik = IKRetargeter(model)
        q  = ik.retarget(plan_data, landmarks, palm_xpos)  # 16 joint angles
    """

    def __init__(self, model: mujoco.MjModel,
                 n_iters: int = 15,   # kept for API compat (unused)
                 step: float  = 0.5): # kept for API compat (unused)
        self.model    = model
        # Running-max angles for each bend joint (used by DEBUG_ANGLES).
        # Order: [if_mcp, if_pip, if_dip, mf_mcp, mf_pip, mf_dip,
        #         rf_mcp, rf_pip, rf_dip, th_cmc, th_mcp, th_ipl]
        self._angle_max = np.zeros(12)

    # ------------------------------------------------------------------
    def retarget(self,
                 plan_data,           # unused — kept for API compat
                 lm,
                 palm_xpos=None       # unused — kept for API compat
                 ) -> np.ndarray:
        """
        Compute 16 LEAP joint angles from a MediaPipe landmark list.

        Args:
            plan_data:  ignored (API compat)
            lm:         result.multi_hand_landmarks[0].landmark  (21 entries)
            palm_xpos:  ignored (API compat)

        Returns:
            q (16,): desired joint angles in LEAP actuator order
                    [if_mcp, if_rot, if_pip, if_dip,
                    mf_mcp, mf_rot, mf_pip, mf_dip,
                    rf_mcp, rf_rot, rf_pip, rf_dip,
                    th_cmc, th_axl, th_mcp, th_ipl]
        """
        q = np.zeros(16)

        w  = _lm3(lm, WRIST)
        mm = _lm3(lm, MIDDLE_MCP)   # used as lateral reference for rot joints

        # ── Index finger (0-3: mcp, rot, pip, dip) ────────────────────────────
        im  = _lm3(lm, INDEX_MCP);  ip  = _lm3(lm, INDEX_PIP)
        idd = _lm3(lm, INDEX_DIP);  it  = _lm3(lm, INDEX_TIP)

        q[0] = _scale(_bend(im - w,   ip  - im), _MCP_MAX)   # if_mcp
        # rot: index spreads laterally away from middle.
        # Normalize by wrist→middle-MCP distance so the value is scale-invariant
        # (hand at different distances from the camera gives the same spread angle).
        hand_scale = float(np.linalg.norm((mm - w)[:2])) + 1e-6
        spread_i = float(np.clip((im[0] - mm[0]) / hand_scale * 1.5, -_ROT_LIM, _ROT_LIM))
        q[1] = spread_i                                        # if_rot
        q[2] = _scale(_bend(ip - im,  idd - ip),  _PIP_MAX)   # if_pip
        q[3] = _scale(_bend(idd - ip, it  - idd), _DIP_MAX)   # if_dip

        # ── Middle finger (4-7) ────────────────────────────────────────────────
        mmp = _lm3(lm, MIDDLE_MCP); mpi = _lm3(lm, MIDDLE_PIP)
        mdi = _lm3(lm, MIDDLE_DIP); mti = _lm3(lm, MIDDLE_TIP)

        q[4] = _scale(_bend(mmp - w,   mpi - mmp), _MCP_MAX)  # mf_mcp
        # mf_rot: middle deviates from the midpoint between index and ring MCPs.
        # If middle is to the right of that midpoint → positive (spans toward ring).
        mid_ref_x = (im[0] + _lm3(lm, RING_MCP)[0]) * 0.5
        spread_m = float(np.clip((mid_ref_x - mm[0]) / hand_scale * 1.0,
                                  -_ROT_LIM, _ROT_LIM))
        q[5] = spread_m                                        # mf_rot
        q[6] = _scale(_bend(mpi - mmp, mdi - mpi), _PIP_MAX)  # mf_pip
        q[7] = _scale(_bend(mdi - mpi, mti - mdi), _DIP_MAX)  # mf_dip

        # ── Ring finger (8-11) ─────────────────────────────────────────────────
        rm  = _lm3(lm, RING_MCP);  rp  = _lm3(lm, RING_PIP)
        rd  = _lm3(lm, RING_DIP);  rt  = _lm3(lm, RING_TIP)

        q[8]  = _scale(_bend(rm - w,  rp - rm), _MCP_MAX)     # rf_mcp
        spread_r = float(np.clip((mm[0] - rm[0]) / hand_scale * 1.5, -_ROT_LIM, _ROT_LIM))
        q[9]  = spread_r                                       # rf_rot
        q[10] = _scale(_bend(rp - rm, rd - rp), _PIP_MAX)     # rf_pip
        q[11] = _scale(_bend(rd - rp, rt - rd), _DIP_MAX)     # rf_dip

        # ── Thumb (12-15: th_cmc, th_axl, th_mcp, th_ipl) ────────────────────
        tc  = _lm3(lm, THUMB_CMC)
        tm  = _lm3(lm, THUMB_MCP_LM)
        tip = _lm3(lm, THUMB_IP)
        tt  = _lm3(lm, THUMB_TIP)

        # th_cmc: metacarpal curl — how far thumb rises from palm
        q[12] = _scale(_bend(tc - w,  tm - tc),  _CMC_MAX)    # th_cmc

        # th_axl: opposition — how much thumb tip crosses toward the index side.
        # Normalize by hand_scale so opposition doesn't vanish as hand moves closer.
        opposition = float(np.clip((im[0] - tt[0]) / hand_scale * 1.2, 0.0, 1.0))
        q[13] = opposition * _AXL_MAX                          # th_axl

        q[14] = _scale(_bend(tm - tc,  tip - tm), _TMCP_MAX)  # th_mcp
        q[15] = _scale(_bend(tip - tm, tt - tip), _IPL_MAX)   # th_ipl

        if DEBUG_ANGLES:
            raw = np.array([
                _bend(im - w,   ip  - im),   # if_mcp
                _bend(ip - im,  idd - ip),   # if_pip
                _bend(idd - ip, it  - idd),  # if_dip
                _bend(mmp - w,  mpi - mmp),  # mf_mcp
                _bend(mpi - mmp, mdi - mpi), # mf_pip
                _bend(mdi - mpi, mti - mdi), # mf_dip
                _bend(rm - w,   rp  - rm),   # rf_mcp
                _bend(rp - rm,  rd  - rp),   # rf_pip
                _bend(rd - rp,  rt  - rd),   # rf_dip
                _bend(tc - w,   tm  - tc),   # th_cmc
                _bend(tm - tc,  tip - tm),   # th_mcp
                _bend(tip - tm, tt  - tip),  # th_ipl
            ])
            self._angle_max = np.maximum(self._angle_max, raw)
            names = ['if_mcp','if_pip','if_dip',
                     'mf_mcp','mf_pip','mf_dip',
                     'rf_mcp','rf_pip','rf_dip',
                     'th_cmc','th_mcp','th_ipl']
            parts = [f"{n}={v:.2f}(max={m:.2f})"
                     for n, v, m in zip(names, raw, self._angle_max)]
            print("ANGLES  " + "  ".join(parts))

        return q

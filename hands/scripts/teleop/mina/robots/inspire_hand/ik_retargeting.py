"""
Direct angle retargeting: MediaPipe 21 landmarks → 12 Inspire Hand joints.

Inspire Hand joint order (matches actuator order in inspire_hand_right_body.xml):
  [0]  thumb_proximal_yaw_joint    — thumb abduction/adduction  (0 → 1.308 rad)
  [1]  thumb_proximal_pitch_joint  — thumb CMC bend              (0 → 0.6 rad)
  [2]  thumb_intermediate_joint    — thumb MCP bend              (0 → 0.8 rad)
  [3]  thumb_distal_joint          — thumb IP bend               (0 → 0.4 rad)
  [4]  index_proximal_joint        — index MCP bend              (0 → 1.47 rad)
  [5]  index_intermediate_joint    — index PIP+DIP combined      (-0.04545 → 1.56 rad)
  [6]  middle_proximal_joint       — middle MCP bend             (0 → 1.47 rad)
  [7]  middle_intermediate_joint   — middle PIP+DIP combined     (-0.04545 → 1.56 rad)
  [8]  ring_proximal_joint         — ring MCP bend               (0 → 1.47 rad)
  [9]  ring_intermediate_joint     — ring PIP+DIP combined       (-0.04545 → 1.56 rad)
  [10] pinky_proximal_joint        — pinky MCP bend              (0 → 1.47 rad)
  [11] pinky_intermediate_joint    — pinky PIP+DIP combined      (-0.04545 → 1.56 rad)

MediaPipe landmark indices (21 total):
  WRIST=0
  THUMB : CMC=1  MCP=2   IP=3   TIP=4
  INDEX : MCP=5  PIP=6   DIP=7  TIP=8
  MIDDLE: MCP=9  PIP=10  DIP=11 TIP=12
  RING  : MCP=13 PIP=14  DIP=15 TIP=16
  PINKY : MCP=17 PIP=18  DIP=19 TIP=20
"""

import numpy as np

# ── Tuning ─────────────────────────────────────────────────────────────────────
# Max bend MediaPipe produces for regular fingers at full curl
_MP_MAX = 1.8   # radians — used for index/middle/ring/pinky joints

# Inspire finger limits (slightly below XML max)
_MCP_MAX  = 1.40   # proximal joints (xml max 1.47)
_INT_MAX  = 1.50   # intermediate joints (xml max 1.56)

# ── Thumb-specific tuning ───────────────────────────────────────────────────────
# The thumb joints have small ranges AND MediaPipe thumb bends are smaller.
# We use per-joint reference maximums calibrated to the thumb's actual ROM.
_TH_YAW       = 1.20   # abduction target (xml max 1.308)
_TH_PITCH     = 0.55   # thumb CMC pitch limit (xml max 0.6)
_TH_INT       = 0.75   # thumb intermediate limit (xml max 0.8)
_TH_DIS       = 0.38   # thumb distal limit (xml max 0.4)
# Reference bends: how much each thumb joint actually bends in MediaPipe at full curl
# (much smaller than _MP_MAX=1.8 because the thumb has a different anatomy)
_TH_CMC_REF   = 0.45   # CMC joint bend at full curl (~25°)
_TH_MCP_REF   = 0.55   # MCP joint bend at full curl (~30°)
_TH_IP_REF    = 1.00   # IP  joint bend at full curl (~57°)

# MediaPipe landmark indices
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


def _lm3(lm, i: int) -> np.ndarray:
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (radians) between two bone vectors at a joint. 0 = straight."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _scale(angle: float, joint_max: float) -> float:
    """Map MediaPipe bend ∈ [0, _MP_MAX] → joint angle ∈ [0, joint_max]."""
    return float(np.clip(angle / _MP_MAX * joint_max, 0.0, joint_max))


class IKRetargeter:
    """
    Maps MediaPipe hand landmarks → 12 Inspire Hand joint angles.

    Usage:
        ik = IKRetargeter(model)
        q  = ik.retarget(None, landmarks)  # returns (12,) array

    The model argument is accepted for API compatibility with the LEAP retargeter
    but is not used (retargeting is purely angle-based).
    """

    def __init__(self, model=None, n_iters: int = 15, step: float = 0.5):
        self.model = model

    def retarget(self, plan_data, lm, palm_xpos=None) -> np.ndarray:
        """
        Compute 12 Inspire joint angles from MediaPipe landmarks.

        Args:
            plan_data:  ignored (API compat)
            lm:         result.multi_hand_landmarks[0].landmark  (21 entries)
            palm_xpos:  ignored (API compat)

        Returns:
            q (12,) in the actuator order above.
        """
        q = np.zeros(12)

        w   = _lm3(lm, WRIST)
        mm  = _lm3(lm, MIDDLE_MCP)
        hand_scale = float(np.linalg.norm((mm - w)[:2])) + 1e-6

        # ── Thumb ────────────────────────────────────────────────────────────
        tc  = _lm3(lm, THUMB_CMC)
        tm  = _lm3(lm, THUMB_MCP)
        tip = _lm3(lm, THUMB_IP)
        tt  = _lm3(lm, THUMB_TIP)
        im  = _lm3(lm, INDEX_MCP)
        pm  = _lm3(lm, PINKY_MCP)

        # [0] Yaw (abduction): angle between thumb and palm plane.
        # Palm normal = cross(index_MCP - wrist, pinky_MCP - wrist).
        # Project thumb CMC→TIP vector onto the palm normal; the larger this
        # component is relative to the thumb length, the more the thumb is spread.
        palm_u = im - w
        palm_v = pm - w
        palm_normal = np.cross(palm_u, palm_v)
        n_pn = np.linalg.norm(palm_normal)
        thumb_dir = tt - tc
        n_td = np.linalg.norm(thumb_dir)
        if n_pn > 1e-6 and n_td > 1e-6:
            sin_abduct = abs(float(np.dot(thumb_dir / n_td, palm_normal / n_pn)))
        else:
            sin_abduct = 0.0
        q[0] = float(np.clip(sin_abduct * 1.5, 0.0, 1.0)) * _TH_YAW

        # [1][2][3] Thumb curl — each joint has its own calibrated reference bend.
        # Using per-joint _TH_*_REF instead of _MP_MAX gives full-range motion
        # even though the thumb bends less than fingers in MediaPipe space.
        b_cmc = _bend(tc - w,   tm  - tc)
        b_mcp = _bend(tm - tc,  tip - tm)
        b_ip  = _bend(tip - tm, tt  - tip)

        q[1] = float(np.clip(b_cmc / _TH_CMC_REF * _TH_PITCH, 0.0, _TH_PITCH))
        q[2] = float(np.clip(b_mcp / _TH_MCP_REF * _TH_INT,   0.0, _TH_INT))
        q[3] = float(np.clip(b_ip  / _TH_IP_REF  * _TH_DIS,   0.0, _TH_DIS))

        # ── Index finger ─────────────────────────────────────────────────────
        ip  = _lm3(lm, INDEX_MCP);  ipp = _lm3(lm, INDEX_PIP)
        idd = _lm3(lm, INDEX_DIP);  it  = _lm3(lm, INDEX_TIP)

        # [4] Proximal (MCP bend)
        q[4] = _scale(_bend(ip - w,   ipp - ip), _MCP_MAX)

        # [5] Intermediate: average of PIP and DIP bends (one DOF drives both)
        pip_b = _bend(ipp - ip,  idd - ipp)
        dip_b = _bend(idd - ipp, it  - idd)
        q[5] = _scale((pip_b + dip_b) * 0.5, _INT_MAX)

        # ── Middle finger ─────────────────────────────────────────────────────
        mmp = _lm3(lm, MIDDLE_MCP); mpi = _lm3(lm, MIDDLE_PIP)
        mdi = _lm3(lm, MIDDLE_DIP); mti = _lm3(lm, MIDDLE_TIP)

        # [6] Proximal
        q[6] = _scale(_bend(mmp - w,   mpi - mmp), _MCP_MAX)

        # [7] Intermediate
        pip_b = _bend(mpi - mmp, mdi - mpi)
        dip_b = _bend(mdi - mpi, mti - mdi)
        q[7] = _scale((pip_b + dip_b) * 0.5, _INT_MAX)

        # ── Ring finger ───────────────────────────────────────────────────────
        rm  = _lm3(lm, RING_MCP);  rp  = _lm3(lm, RING_PIP)
        rd  = _lm3(lm, RING_DIP);  rt  = _lm3(lm, RING_TIP)

        # [8] Proximal
        q[8] = _scale(_bend(rm - w,  rp - rm), _MCP_MAX)

        # [9] Intermediate
        pip_b = _bend(rp - rm, rd - rp)
        dip_b = _bend(rd - rp, rt - rd)
        q[9] = _scale((pip_b + dip_b) * 0.5, _INT_MAX)

        # ── Pinky finger ──────────────────────────────────────────────────────
        pm  = _lm3(lm, PINKY_MCP); pp  = _lm3(lm, PINKY_PIP)
        pd  = _lm3(lm, PINKY_DIP); pt  = _lm3(lm, PINKY_TIP)

        # [10] Proximal
        q[10] = _scale(_bend(pm - w,  pp - pm), _MCP_MAX)

        # [11] Intermediate
        pip_b = _bend(pp - pm, pd - pp)
        dip_b = _bend(pd - pp, pt - pd)
        q[11] = _scale((pip_b + dip_b) * 0.5, _INT_MAX)

        return q

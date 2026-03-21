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

# ── Finger limits (slightly below XML max for safety) ─────────────────────────
_MCP_MAX = 1.40   # proximal joints   (xml max 1.47)
_INT_MAX = 1.50   # intermediate joints (xml max 1.56)

# ── Reference bends at full curl (calibrated to MediaPipe output range) ───────
# These are the bend angles MediaPipe actually reports at maximum curl.
# Lowering these values makes fingers reach their robot limit more easily.
_MCP_REF = 1.10   # MCP joint: ~63° max bend in MediaPipe world coords
_PIP_REF = 1.40   # PIP joint: ~80° at full curl
_DIP_REF = 1.20   # DIP joint: ~69° at full curl

# ── Thumb-specific tuning ─────────────────────────────────────────────────────
_TH_YAW   = 1.30   # abduction target   (xml max 1.308)
_TH_PITCH = 0.60   # thumb CMC pitch    (xml max 0.6)
_TH_INT   = 0.80   # thumb MCP bend     (xml max 0.8)
_TH_DIS   = 0.40   # thumb IP bend      (xml max 0.4)
# Reference bends: très petits pour que le moindre mouvement du pouce = plein range
_TH_CMC_REF = 0.18   # CMC barely bends in MediaPipe → fire early
_TH_MCP_REF = 0.22   # idem MCP
_TH_IP_REF  = 0.20   # IP — très sensible, réagit dès le moindre pli

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


def _lm3(lm, i: int) -> np.ndarray:
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (radians) between two consecutive bone vectors at a joint."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _map(angle: float, ref: float, limit: float, power: float = 1.0) -> float:
    """Map bend angle ∈ [0, ref] → joint angle ∈ [0, limit] with optional power curve."""
    r = float(np.clip(angle / ref, 0.0, 1.0))
    return float(r ** power * limit)


class IKRetargeter:
    """
    Maps MediaPipe hand landmarks → 12 Inspire Hand joint angles.

    Usage:
        ik = IKRetargeter(model)
        q  = ik.retarget(None, landmarks)  # returns (12,) array
    """

    def __init__(self, model=None, n_iters: int = 15, step: float = 0.5):
        self.model = model

    def retarget(self, plan_data, lm, palm_xpos=None) -> np.ndarray:
        q = np.zeros(12)

        w  = _lm3(lm, WRIST)
        im = _lm3(lm, INDEX_MCP)
        pm = _lm3(lm, PINKY_MCP)
        mm = _lm3(lm, MIDDLE_MCP)

        # Palm reference direction (wrist → middle MCP = main palm axis)
        palm_axis = mm - w
        palm_len  = np.linalg.norm(palm_axis)
        if palm_len < 1e-6:
            return q

        # ── Thumb ─────────────────────────────────────────────────────────────
        tc  = _lm3(lm, THUMB_CMC)
        tm  = _lm3(lm, THUMB_MCP)
        tip = _lm3(lm, THUMB_IP)
        tt  = _lm3(lm, THUMB_TIP)

        # [0] Yaw (abduction) — thumb tip distance from palm plane, normalised
        palm_u = im - w
        palm_v = pm - w
        palm_normal = np.cross(palm_u, palm_v)
        n_pn = np.linalg.norm(palm_normal)
        thumb_vec = tt - tc
        n_tv = np.linalg.norm(thumb_vec)
        if n_pn > 1e-6 and n_tv > 1e-6:
            sin_abduct = abs(float(np.dot(thumb_vec / n_tv, palm_normal / n_pn)))
        else:
            sin_abduct = 0.0
        q[0] = float(np.clip(sin_abduct * 2.5, 0.0, 1.0)) * _TH_YAW

        # [1][2][3] Thumb curl — power < 1 pour réagir dès le début du mouvement
        b_cmc = _bend(tc - w,   tm  - tc)
        b_mcp = _bend(tm - tc,  tip - tm)
        b_ip  = _bend(tip - tm, tt  - tip)

        q[1] = _map(b_cmc, _TH_CMC_REF, _TH_PITCH, power=0.6)
        q[2] = _map(b_mcp, _TH_MCP_REF, _TH_INT,   power=0.6)
        q[3] = _map(b_ip,  _TH_IP_REF,  _TH_DIS,   power=0.4)

        # ── Index ──────────────────────────────────────────────────────────────
        ip  = _lm3(lm, INDEX_MCP);  ipp = _lm3(lm, INDEX_PIP)
        idd = _lm3(lm, INDEX_DIP);  it  = _lm3(lm, INDEX_TIP)
        # MCP bend: angle between metacarpal (wrist→MCP) and proximal phalanx (MCP→PIP)
        q[4] = _map(_bend(ip - w,  ipp - ip),  _MCP_REF, _MCP_MAX, power=0.85)
        # Intermediate: PIP drives the joint, DIP follows — weight PIP more
        pip_b = _bend(ipp - ip,  idd - ipp)
        dip_b = _bend(idd - ipp, it  - idd)
        q[5] = _map(pip_b * 0.65 + dip_b * 0.35, _PIP_REF, _INT_MAX, power=0.85)

        # ── Middle ─────────────────────────────────────────────────────────────
        mmp = _lm3(lm, MIDDLE_MCP); mpi = _lm3(lm, MIDDLE_PIP)
        mdi = _lm3(lm, MIDDLE_DIP); mti = _lm3(lm, MIDDLE_TIP)
        q[6] = _map(_bend(mmp - w,   mpi - mmp), _MCP_REF, _MCP_MAX, power=0.85)
        pip_b = _bend(mpi - mmp, mdi - mpi)
        dip_b = _bend(mdi - mpi, mti - mdi)
        q[7] = _map(pip_b * 0.65 + dip_b * 0.35, _PIP_REF, _INT_MAX, power=0.85)

        # ── Ring ───────────────────────────────────────────────────────────────
        rm = _lm3(lm, RING_MCP);  rp = _lm3(lm, RING_PIP)
        rd = _lm3(lm, RING_DIP);  rt = _lm3(lm, RING_TIP)
        q[8] = _map(_bend(rm - w,  rp - rm), _MCP_REF, _MCP_MAX, power=0.85)
        pip_b = _bend(rp - rm, rd - rp)
        dip_b = _bend(rd - rp, rt - rd)
        q[9] = _map(pip_b * 0.65 + dip_b * 0.35, _PIP_REF, _INT_MAX, power=0.85)

        # ── Pinky ──────────────────────────────────────────────────────────────
        pm2 = _lm3(lm, PINKY_MCP); pp = _lm3(lm, PINKY_PIP)
        pd  = _lm3(lm, PINKY_DIP); pt = _lm3(lm, PINKY_TIP)
        q[10] = _map(_bend(pm2 - w, pp - pm2), _MCP_REF, _MCP_MAX, power=0.85)
        pip_b = _bend(pp - pm2, pd - pp)
        dip_b = _bend(pd - pp,  pt - pd)
        q[11] = _map(pip_b * 0.65 + dip_b * 0.35, _PIP_REF, _INT_MAX, power=0.85)

        return q

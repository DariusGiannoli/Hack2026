"""
Direct angle-based retargeting: MediaPipe Pose landmarks → 7 Franka FR3 joint angles.

Strategy
--------
MediaPipe Pose provides 3-D world landmarks (hip-centred, metric) for the full body.
We extract the right arm (shoulder → elbow → wrist) and compute geometric angles
that map directly onto the Franka's 7 revolute joints:

  joint1 — shoulder pan      : horizontal azimuth of the upper arm
  joint2 — shoulder lift     : elevation of the upper arm from horizontal
  joint3 — upper arm twist   : locked to 0 (not observable from landmarks alone)
  joint4 — elbow flex        : bend angle between upper arm and forearm (negated)
  joint5 — forearm twist     : locked to 0 (not observable from landmarks alone)
  joint6 — wrist flex        : bend angle between forearm and hand direction
  joint7 — wrist rotation    : locked to 0 (not observable without hand markers)

Joints 3, 5, 7 are locked at their home values; add hand orientation (palm_quat)
to drive them in a future pass.

Coordinate conventions
----------------------
MediaPipe world landmarks:  x = right, y = up, z = toward camera (right-hand)
MuJoCo / Franka base frame: x = forward, y = left, z = up

The same T matrix as in LEAP's palm_quat is used:
    muj_x =  cam_x
    muj_y =  cam_z
    muj_z = -cam_y

Franka joint 4 home value is -π/2 (≈ -1.5708 rad) — fully bent elbow at rest.
This retargeter offsets from that home value.
"""

import numpy as np
import mujoco

# ── MediaPipe Pose landmark indices ────────────────────────────────────────────
SHOULDER_R = 12
ELBOW_R    = 14
WRIST_R    = 16
INDEX_R    = 20   # index fingertip — used for hand direction

# ── Franka FR3 joint ranges (from fr3v2.xml) ───────────────────────────────────
# joint: [min, max] in radians
_J_RANGE = np.array([
    [-2.7437,  2.7437],   # joint1: shoulder pan
    [-1.7837,  1.7837],   # joint2: shoulder lift
    [-2.9007,  2.9007],   # joint3: upper arm twist (locked)
    [-3.0421, -0.1518],   # joint4: elbow flex (always negative = bent)
    [-2.8065,  2.8065],   # joint5: forearm twist (locked)
    [ 0.5445,  4.5169],   # joint6: wrist flex
    [-3.0159,  3.0159],   # joint7: wrist rotation (locked)
])

# ── Home configuration (from fr3v2.xml keyframe "home") ───────────────────────
HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

# ── Human arm range assumptions (used for scaling) ────────────────────────────
# Maximum shoulder pan/tilt the user is expected to produce in front of camera.
_SHOULDER_PAN_MAX  = np.radians(80)   # ±80° horizontal sweep
_SHOULDER_LIFT_MAX = np.radians(80)   # ±80° vertical sweep
_ELBOW_BEND_MAX    = np.radians(150)  # 0–150° bend (MediaPipe observable max)
_WRIST_BEND_MAX    = np.radians(90)   # 0–90° wrist flex

# ── Debug flag ─────────────────────────────────────────────────────────────────
DEBUG_ANGLES = False


# ── Helpers ────────────────────────────────────────────────────────────────────
def _lm3(lm, i: int) -> np.ndarray:
    """World landmark → (x, y, z) in MediaPipe frame."""
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _to_mujoco(v_cam: np.ndarray) -> np.ndarray:
    """
    Convert a vector from MediaPipe world frame to MuJoCo world frame.

    MediaPipe world: x=right, y=up, z=toward-camera
    MuJoCo:          x=right, y=forward(=cam_z), z=up(=-cam_y)
    """
    return np.array([v_cam[0], v_cam[2], -v_cam[1]])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors.  0 = straight, π = fully curled."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _clamp(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))


# ── Main class ─────────────────────────────────────────────────────────────────
class FrankaIKRetargeter:
    """
    Maps MediaPipe Pose world landmarks → 7 Franka FR3 joint angles.

    Usage
    -----
        ik = FrankaIKRetargeter(model)
        q  = ik.retarget(pose_result)   # shape (7,)

    If the pose is not detected, returns the home configuration so the arm
    stays in a safe rest pose rather than snapping to zero.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    def retarget(self, pose_result) -> np.ndarray:
        """
        Compute 7 Franka joint targets from a MediaPipe Pose result.

        Args:
            pose_result: return value of ArmTracker.process(frame)
                         Uses pose_world_landmarks (metric, hip-centred).

        Returns:
            q (7,): desired joint angles in Franka actuator order
                    [joint1..joint7]
        """
        # Fallback: no pose detected → hold home configuration
        if (pose_result is None
                or pose_result.pose_world_landmarks is None):
            return HOME.copy()

        lm = pose_result.pose_world_landmarks.landmark

        # Check visibility of key landmarks
        s_vis = lm[SHOULDER_R].visibility
        e_vis = lm[ELBOW_R].visibility
        w_vis = lm[WRIST_R].visibility
        if s_vis < 0.5 or e_vis < 0.5 or w_vis < 0.5:
            return HOME.copy()

        # ── Raw positions in MediaPipe world frame ────────────────────────
        shoulder = _lm3(lm, SHOULDER_R)
        elbow    = _lm3(lm, ELBOW_R)
        wrist    = _lm3(lm, WRIST_R)

        # Use index finger tip if visible, else estimate from wrist direction
        i_vis = lm[INDEX_R].visibility
        if i_vis > 0.5:
            index = _lm3(lm, INDEX_R)
            hand_dir_mp = index - wrist
        else:
            hand_dir_mp = wrist - elbow   # fallback: extend forearm direction

        # ── Convert vectors to MuJoCo frame ──────────────────────────────
        upper_arm = _to_mujoco(elbow - shoulder)
        forearm   = _to_mujoco(wrist - elbow)
        hand_dir  = _to_mujoco(hand_dir_mp)

        # ── joint1: shoulder pan (rotation around MuJoCo Z = up) ─────────
        # Project upper arm onto horizontal plane (X-Y in MuJoCo).
        ua_horiz = upper_arm.copy(); ua_horiz[2] = 0.0
        pan_raw  = np.arctan2(ua_horiz[1], ua_horiz[0])   # azimuth
        # Franka rests pointing forward (+X).  Offset so neutral arm → joint1=0.
        joint1 = _clamp(pan_raw, _J_RANGE[0, 0], _J_RANGE[0, 1])

        # ── joint2: shoulder lift (elevation from horizontal) ─────────────
        ua_norm  = np.linalg.norm(upper_arm) + 1e-6
        lift_raw = np.arcsin(np.clip(-upper_arm[2] / ua_norm, -1.0, 1.0))
        # Scale human range to Franka range:
        # human -80°…+80° → Franka -1.78…+1.78 rad
        joint2 = _clamp(
            lift_raw / _SHOULDER_LIFT_MAX * _J_RANGE[1, 1],
            _J_RANGE[1, 0], _J_RANGE[1, 1]
        )

        # ── joint3: upper arm twist — locked ─────────────────────────────
        joint3 = HOME[2]

        # ── joint4: elbow flex ────────────────────────────────────────────
        # Bend angle between upper arm and forearm.
        # Franka joint4 is negative when bent; 0 = fully straight (singular).
        # Map human [0, 150°] → Franka [-3.04, -0.15].
        elbow_bend = _bend(upper_arm, forearm)   # 0 = straight, π = fully bent
        # Invert: more bend → more negative joint4
        j4_range   = _J_RANGE[3, 1] - _J_RANGE[3, 0]  # positive span ~2.89 rad
        joint4 = _clamp(
            _J_RANGE[3, 1] - (elbow_bend / _ELBOW_BEND_MAX) * j4_range,
            _J_RANGE[3, 0], _J_RANGE[3, 1]
        )

        # ── joint5: forearm twist — locked ───────────────────────────────
        joint5 = HOME[4]

        # ── joint6: wrist flex ────────────────────────────────────────────
        # Bend angle between forearm and hand direction.
        # Franka joint6 home is ~1.57 rad; range [0.54, 4.52].
        # Map human [0, 90°] → offset from home value.
        wrist_bend = _bend(forearm, hand_dir)
        # Scale: 0 bend → home value; max bend → home + 1.5 rad
        joint6 = _clamp(
            HOME[5] + (wrist_bend / _WRIST_BEND_MAX) * 1.5,
            _J_RANGE[5, 0], _J_RANGE[5, 1]
        )

        # ── joint7: wrist rotation — locked ──────────────────────────────
        joint7 = HOME[6]

        q = np.array([joint1, joint2, joint3, joint4, joint5, joint6, joint7])

        if DEBUG_ANGLES:
            names = ['pan', 'lift', 'twist', 'elbow', 'f-twist', 'w-flex', 'w-rot']
            parts = [f"{n}={v:.2f}" for n, v in zip(names, q)]
            print("FRANKA  " + "  ".join(parts))

        return q

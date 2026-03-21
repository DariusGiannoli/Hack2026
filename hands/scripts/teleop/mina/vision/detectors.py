"""
detectors.py — Hybrid hand/body tracker.

StereoHandTracker  : MediaPipe Hands (21 points 2D) + ZED point-cloud depth lifting.
                     Inspired by github.com/dokkev/6D_hand_pose_tracking.
                     → x, y  : MediaPipe normalised image coords [0, 1]
                     → z     : real metric relative depth from ZED point cloud
                                (z_lm - z_wrist) / z_wrist  [dimensionless]
                               Falls back to MediaPipe estimated z when ZED depth
                               is unavailable (NaN / occluded pixel).

ArmTracker         : ZED SDK BODY_38 full-body skeleton for shoulder/elbow/wrist/hip.
                     Provides MediaPipe-Pose-compatible world landmarks (hip-centred).

ZED SDK BODY_38 keypoint indices (sl.BODY_38_PARTS):
  Body skeleton:
    12 LEFT_SHOULDER   13 RIGHT_SHOULDER
    14 LEFT_ELBOW      15 RIGHT_ELBOW
    16 LEFT_WRIST      17 RIGHT_WRIST
    18 LEFT_HIP        19 RIGHT_HIP

MediaPipe Pose indices used by the teleoperation code:
  11 LEFT_SHOULDER   12 RIGHT_SHOULDER
  13 LEFT_ELBOW      14 RIGHT_ELBOW
  15 LEFT_WRIST      16 RIGHT_WRIST
  23 LEFT_HIP        24 RIGHT_HIP

⚠  Run in the mina310 conda environment (Python 3.10) where both
   MediaPipe 0.10 and pyzed 5.2 are installed:
     conda run -n mina310 python teleop_edgard_copy.py
"""

import cv2
import numpy as np
from typing import Optional


# ── ZED BODY_38 index constants (used by ArmTracker only) ──────────────────
_ZED_LEFT_SHOULDER  = 12;  _ZED_RIGHT_SHOULDER = 13
_ZED_LEFT_ELBOW     = 14;  _ZED_RIGHT_ELBOW    = 15
_ZED_LEFT_WRIST     = 16;  _ZED_RIGHT_WRIST    = 17
_ZED_LEFT_HIP       = 18;  _ZED_RIGHT_HIP      = 19

# Minimum ZED depth (m) to consider a point cloud sample valid
_MIN_DEPTH_M = 0.10
_MAX_DEPTH_M = 5.00


# ── MediaPipe-compatible wrapper objects ────────────────────────────────────

class _Landmark:
    """Emulates a MediaPipe normalized landmark with x, y ∈ [0,1] and z (depth)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _HandLandmarks:
    """Emulates MediaPipe multi_hand_landmarks[i] — holds 21 _Landmark objects."""
    def __init__(self, landmarks):
        self.landmark = landmarks   # list[_Landmark] of length 21


class _Classification:
    __slots__ = ("label", "score")

    def __init__(self, label: str, score: float):
        self.label = label
        self.score = float(score)


class _Handedness:
    """Emulates MediaPipe multi_handedness[i].classification[0]."""
    def __init__(self, label: str, score: float):
        self.classification = [_Classification(label, score)]


class _HandResult:
    """Emulates a MediaPipe Hands result object."""
    def __init__(self, hands):
        """
        hands : list of (label, score, list[_Landmark]) tuples,
                one per detected hand.
        """
        if hands:
            self.multi_hand_landmarks = [_HandLandmarks(h[2]) for h in hands]
            self.multi_handedness = [_Handedness(h[0], h[1]) for h in hands]
        else:
            self.multi_hand_landmarks = None
            self.multi_handedness = None


class _PoseWorldLandmark:
    """Emulates MediaPipe pose_world_landmarks.landmark[i]."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _PoseLandmarks:
    """Container for 33 pose world landmarks (indexed by MediaPipe Pose index)."""
    def __init__(self, landmarks):
        self.landmark = landmarks   # list[_PoseWorldLandmark] length ≥ 25


class _PoseResult:
    """Emulates a MediaPipe Pose result object."""
    def __init__(self, world_lm=None, img_lm=None):
        if world_lm is not None:
            self.pose_world_landmarks = _PoseLandmarks(world_lm)
            self.pose_landmarks = _PoseLandmarks(img_lm) if img_lm is not None else None
        else:
            self.pose_world_landmarks = None
            self.pose_landmarks = None


# ── Public detector classes ─────────────────────────────────────────────────

class StereoHandTracker:
    """
    Hand tracker: MediaPipe Hands (21 pts 2D) + ZED point-cloud depth lifting.

    Approach from github.com/dokkev/6D_hand_pose_tracking :
      1. MediaPipe Hands detects 21 landmarks per hand in image space.
      2. ZED point cloud provides real metric depth at each landmark pixel.
      3. z of each _Landmark = (Z_zed_at_pixel - Z_zed_at_wrist) / Z_wrist
         → dimensionless relative depth, same scale as MediaPipe .z.
      4. Falls back to MediaPipe estimated z when ZED depth is NaN/occluded.

    result.multi_handedness keeps MediaPipe's "Left"/"Right" convention:
      on a non-mirrored ZED, physical right hand → label "Left" (TARGET_HAND).

    process(frame_left, frame_right) → (res_left, res_right)
    draw_landmarks(frame, result)
    """

    # Hand skeleton connections (same as MediaPipe HAND_CONNECTIONS)
    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (0,9),(9,10),(10,11),(11,12),      # middle
        (0,13),(13,14),(14,15),(15,16),    # ring
        (0,17),(17,18),(18,19),(19,20),    # pinky
        (5,9),(9,13),(13,17),              # palm knuckle bar
    ]
    # MCP joints: most stable for drawing (slightly larger circles)
    _MCP_JOINTS = {0, 5, 9, 13, 17}

    def __init__(self, zed_camera=None):
        self._zed_cam = zed_camera
        self._mp_tracker = None
        self._mp_draw = None
        self._mp_hands_cls = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_hands_cls = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            # model_complexity=1 gives better z depth estimates
            self._mp_tracker = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.9,
            )
            print("[StereoHandTracker] MediaPipe Hands initialisé (model_complexity=1).")
        except Exception as e:
            print(f"[StereoHandTracker] MediaPipe non disponible ({e}).")

    # ── ZED depth lifting (dokkev approach) ──────────────────────────────

    def _lift_z(self, mp_landmarks, w: int, h: int, point_cloud) -> list:
        """
        Replace MediaPipe estimated .z with ZED real metric depth.

        Formula (dokkev/6D_hand_pose_tracking):
          z_corrected = (Z_zed_at_landmark - Z_zed_at_wrist) / Z_wrist

        Falls back to MediaPipe z (scaled by wrist depth) when ZED depth is
        invalid at a pixel (NaN, out of range, occluded).
        """
        try:
            import pyzed.sl as sl
            _sl_ok = sl.ERROR_CODE.SUCCESS
        except ImportError:
            # No ZED SDK: just wrap MediaPipe landmarks unchanged
            return [_Landmark(l.x, l.y, l.z) for l in mp_landmarks]

        lm = mp_landmarks

        # Sample ZED depth at wrist pixel
        wx = int(np.clip(lm[0].x * w, 0, w - 1))
        wy = int(np.clip(lm[0].y * h, 0, h - 1))
        err, wrist_pc = point_cloud.get_value(wx, wy)
        z_wrist = float(wrist_pc[2]) if err == _sl_ok else None
        wrist_valid = (z_wrist is not None and
                       not np.isnan(z_wrist) and
                       _MIN_DEPTH_M < z_wrist < _MAX_DEPTH_M)

        result = []
        for l in lm:
            px = int(np.clip(l.x * w, 0, w - 1))
            py = int(np.clip(l.y * h, 0, h - 1))
            err2, pc_val = point_cloud.get_value(px, py)
            z_lm = float(pc_val[2]) if err2 == _sl_ok else None
            lm_valid = (z_lm is not None and
                        not np.isnan(z_lm) and
                        _MIN_DEPTH_M < z_lm < _MAX_DEPTH_M)

            if wrist_valid and lm_valid:
                z_norm = (z_lm - z_wrist) / z_wrist
            elif wrist_valid:
                # ZED depth missing at this joint → use MediaPipe relative z
                # scaled to metric depth (matches dokkev fallback)
                z_norm = (l.z - lm[0].z) * z_wrist
            else:
                z_norm = l.z    # pure MediaPipe estimate

            result.append(_Landmark(l.x, l.y, z_norm))

        return result

    # ── process ──────────────────────────────────────────────────────────

    def process(self, frame_left, frame_right):
        """
        Run MediaPipe on the left frame, lift .z with ZED depth.

        Returns (res_left, res_right).  res_right is always empty because
        depth is already encoded in z — the stereo triangulation path is
        not needed.
        """
        empty = _HandResult([])
        if self._mp_tracker is None:
            return empty, empty

        rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        mp_res = self._mp_tracker.process(rgb)

        if not mp_res.multi_hand_landmarks:
            return empty, empty

        h, w = frame_left.shape[:2]
        point_cloud = (self._zed_cam.get_point_cloud()
                       if self._zed_cam is not None else None)

        hands = []
        for i, lm_mp in enumerate(mp_res.multi_hand_landmarks):
            cls = mp_res.multi_handedness[i].classification[0]
            label = cls.label   # "Left" or "Right" (MediaPipe convention)
            score = float(cls.score)

            if point_cloud is not None:
                lm_list = self._lift_z(lm_mp.landmark, w, h, point_cloud)
            else:
                lm_list = [_Landmark(l.x, l.y, l.z) for l in lm_mp.landmark]

            hands.append((label, score, lm_list))

        return _HandResult(hands), empty

    # ── draw ─────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame, result):
        """Draw full hand skeleton with colour-coded hands."""
        if result.multi_hand_landmarks is None:
            return
        h, w = frame.shape[:2]
        col_line  = {"Left": (0, 200, 0),   "Right": (200, 80, 0)}
        col_joint = {"Left": (0, 255, 60),  "Right": (255, 120, 0)}
        col_mcp   = {"Left": (0, 255, 200), "Right": (255, 200, 0)}

        for i, lm_obj in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[i].classification[0].label
            cl = col_line.get(label, (200, 200, 0))
            cj = col_joint.get(label, (200, 200, 0))
            cm = col_mcp.get(label, (255, 255, 0))
            lms = lm_obj.landmark

            def pt(idx):
                return (int(lms[idx].x * w), int(lms[idx].y * h))

            for a, b in self._HAND_CONNECTIONS:
                pa, pb = pt(a), pt(b)
                if (0 <= pa[0] < w and 0 <= pa[1] < h and
                        0 <= pb[0] < w and 0 <= pb[1] < h):
                    cv2.line(frame, pa, pb, cl, 2)

            for idx, lm in enumerate(lms):
                u, v = int(lm.x * w), int(lm.y * h)
                if 0 <= u < w and 0 <= v < h:
                    if idx in self._MCP_JOINTS:
                        cv2.circle(frame, (u, v), 6, cm, -1)
                        cv2.circle(frame, (u, v), 7, (255, 255, 255), 1)
                    else:
                        cv2.circle(frame, (u, v), 3, cj, -1)


class ArmTracker:
    """
    Full-body pose tracker backed by ZED SDK BODY_38.

    Exposes the same interface as the old MediaPipe ArmTracker:
      process(frame) → pose_result  (with .pose_world_landmarks.landmark[i])
      draw_landmarks(frame, result)
      close()

    MediaPipe Pose landmark indices emulated (those used by the teleoperation code):
      11 LEFT_SHOULDER   12 RIGHT_SHOULDER
      13 LEFT_ELBOW      14 RIGHT_ELBOW
      15 LEFT_WRIST      16 RIGHT_WRIST
      23 LEFT_HIP        24 RIGHT_HIP

    ZED 3-D positions are in camera frame (RIGHT_HANDED_Y_UP: X=right, Y=up, Z=toward
    camera), which matches the MediaPipe Pose world-landmark convention.
    Coordinates are hip-centred (pelvis subtracted) to match MediaPipe's origin.

    Falls back to MediaPipe if no ZEDCamera is provided.
    """

    # Landmark index constants (same as MediaPipe Pose for easy reference)
    SHOULDER_L = 11; SHOULDER_R = 12
    ELBOW_L    = 13; ELBOW_R    = 14
    WRIST_L    = 15; WRIST_R    = 16
    PINKY_L    = 17; PINKY_R    = 18
    INDEX_L    = 19; INDEX_R    = 20
    THUMB_L    = 21; THUMB_R    = 22

    def __init__(self, zed_camera=None):
        self._zed_cam = zed_camera
        self._use_zed = zed_camera is not None and getattr(zed_camera, 'using_zed_sdk', False)

        if not self._use_zed:
            self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._draw    = mp.solutions.drawing_utils
            self._pose    = self._mp_pose.Pose(
                model_complexity=0, smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.7, min_tracking_confidence=0.7)
        except Exception as e:
            print(f"[ArmTracker] MediaPipe non disponible ({e}). Aucun tracking actif.")
            self._mp_pose = None
            self._draw    = None
            self._pose    = None

    # ── ZED path ─────────────────────────────────────────────────────────

    def _process_zed(self, frame):
        bodies = self._zed_cam.get_bodies()
        if bodies is None or not bodies.is_new or not bodies.body_list:
            return _PoseResult()

        # Use the body with the highest average keypoint confidence
        best_body = max(bodies.body_list,
                        key=lambda b: float(b.confidence))
        kp3d = np.array(best_body.keypoint, dtype=float)   # (38, 3)
        kp2d = np.array(best_body.keypoint_2d, dtype=float)  # (38, 2)

        # Hip-centre the coordinates (same as MediaPipe's world origin)
        l_hip = kp3d[_ZED_LEFT_HIP]
        r_hip = kp3d[_ZED_RIGHT_HIP]
        if np.any(np.isnan(l_hip)) or np.any(np.isnan(r_hip)):
            hip_centre = np.zeros(3)
        else:
            hip_centre = (l_hip + r_hip) / 2.0

        # Build a 33-element world landmark array indexed by MediaPipe Pose indices.
        # Indices not covered by ZED BODY_38 are set to NaN (they are not used by
        # the teleoperation code).
        _N_MP_POSE = 33
        world_lm = [_PoseWorldLandmark(np.nan, np.nan, np.nan)] * _N_MP_POSE

        # Mapping: (ZED_index, MediaPipe_Pose_index)
        _MAP = [
            (_ZED_LEFT_SHOULDER,  11), (_ZED_RIGHT_SHOULDER, 12),
            (_ZED_LEFT_ELBOW,     13), (_ZED_RIGHT_ELBOW,    14),
            (_ZED_LEFT_WRIST,     15), (_ZED_RIGHT_WRIST,    16),
            (_ZED_LEFT_HIP,       23), (_ZED_RIGHT_HIP,      24),
        ]
        for zed_idx, mp_idx in _MAP:
            pt = kp3d[zed_idx] - hip_centre
            world_lm[mp_idx] = _PoseWorldLandmark(
                float(pt[0]), float(pt[1]), float(pt[2]))

        # Build image-normalised landmarks for draw_landmarks (optional)
        h, w = frame.shape[:2]
        img_lm = [_PoseWorldLandmark(np.nan, np.nan, np.nan)] * _N_MP_POSE
        for zed_idx, mp_idx in _MAP:
            px2d = kp2d[zed_idx]
            if not np.any(np.isnan(px2d)):
                img_lm[mp_idx] = _PoseWorldLandmark(
                    float(px2d[0]) / w,
                    float(px2d[1]) / h,
                    0.0)

        return _PoseResult(world_lm, img_lm)

    # Body skeleton connections (MediaPipe Pose indices stored in result)
    _BODY_CONNECTIONS = [
        # Spine / neck (approximated from available joints)
        (23, 11), (24, 12),          # hips → shoulders
        (11, 12),                    # shoulder bar
        (23, 24),                    # hip bar
        # Arms
        (11, 13), (13, 15),          # left  shoulder→elbow→wrist
        (12, 14), (14, 16),          # right shoulder→elbow→wrist
    ]
    _BODY_COL = {
        "left":  (255, 140,   0),   # orange — left side
        "right": (  0, 200, 255),   # cyan   — right side
        "spine": (180, 180, 180),   # grey   — torso bar
    }

    def _draw_zed(self, frame, result):
        """Draw body skeleton with labeled joints and colour-coded sides."""
        if result.pose_landmarks is None:
            return
        h, w = frame.shape[:2]
        lms = result.pose_landmarks.landmark

        def valid(idx):
            lm = lms[idx]
            return not (np.isnan(lm.x) or np.isnan(lm.y))

        def pt(idx):
            return (int(lms[idx].x * w), int(lms[idx].y * h))

        # Connection colour lookup (by MediaPipe index)
        conn_col = {
            (23, 11): self._BODY_COL["left"],
            (24, 12): self._BODY_COL["right"],
            (11, 12): self._BODY_COL["spine"],
            (23, 24): self._BODY_COL["spine"],
            (11, 13): self._BODY_COL["left"],
            (13, 15): self._BODY_COL["left"],
            (12, 14): self._BODY_COL["right"],
            (14, 16): self._BODY_COL["right"],
        }

        for (a, b), col in conn_col.items():
            if valid(a) and valid(b):
                cv2.line(frame, pt(a), pt(b), col, 3)

        # Joint labels (MediaPipe index → short name)
        _LABELS = {
            11: "L.Sh", 12: "R.Sh",
            13: "L.El", 14: "R.El",
            15: "L.Wr", 16: "R.Wr",
            23: "L.Hp", 24: "R.Hp",
        }
        joint_col = {
            11: self._BODY_COL["left"],  12: self._BODY_COL["right"],
            13: self._BODY_COL["left"],  14: self._BODY_COL["right"],
            15: self._BODY_COL["left"],  16: self._BODY_COL["right"],
            23: self._BODY_COL["left"],  24: self._BODY_COL["right"],
        }
        for idx, name in _LABELS.items():
            if valid(idx):
                u, v = pt(idx)
                col = joint_col.get(idx, (200, 200, 200))
                cv2.circle(frame, (u, v), 7, col, -1)
                cv2.circle(frame, (u, v), 8, (255, 255, 255), 1)
                cv2.putText(frame, name, (u + 9, v + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # ── MediaPipe path ────────────────────────────────────────────────────

    def _process_mp(self, frame):
        if self._pose is None:
            return _PoseResult()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._pose.process(rgb)

    def _draw_mp(self, frame, result):
        if self._draw is None or not hasattr(result, 'pose_landmarks'):
            return
        if result.pose_landmarks:
            self._draw.draw_landmarks(
                frame, result.pose_landmarks, self._mp_pose.POSE_CONNECTIONS)

    # ── Public interface ──────────────────────────────────────────────────

    def process(self, frame):
        if self._use_zed:
            return self._process_zed(frame)
        return self._process_mp(frame)

    def draw_landmarks(self, frame, result):
        if self._use_zed:
            self._draw_zed(frame, result)
        else:
            self._draw_mp(frame, result)

    def close(self):
        if not self._use_zed and self._pose is not None:
            self._pose.close()

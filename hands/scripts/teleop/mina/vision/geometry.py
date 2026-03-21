# geometry.py  —  Pinhole camera model + stereo depth for ZED 2i
#
# ── Pinhole model recap ───────────────────────────────────────────────────────
#
#   Forward projection (3-D → pixel):
#     u = fx * X/Z + cx
#     v = fy * Y/Z + cy
#
#   Back-projection (pixel + depth → 3-D):
#     X = (u - cx) * Z / fx      [metres, left = negative]
#     Y = (v - cy) * Z / fy      [metres, up   = negative (camera Y points down)]
#
#   Stereo depth (disparity → Z):
#     disparity  d = x_L - x_R   [raw pixels, principal point cancels out]
#     Z = fx * B / d             [metres]
#
# ── How to get real calibration values from the ZED SDK ──────────────────────
#   import pyzed.sl as sl
#   zed = sl.Camera()
#   params = zed.get_camera_information().camera_configuration.calibration_parameters
#   left = params.left_cam
#   print(left.fx, left.fy, left.cx, left.cy)
#   print(params.get_camera_baseline())   # metres
#
# Until you run that, the factory defaults below are accurate to ~1-2 %.

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class PinholeCamera:
    """
    Intrinsic + stereo parameters for one ZED eye.

    All values assume the SBS frame has been split so each half is 1280×720.
    Replace with your unit's actual SDK values for the best accuracy.
    """
    fx: float          # horizontal focal length [px]
    fy: float          # vertical   focal length [px]
    cx: float          # principal point x       [px]   (≈ frame_width  / 2)
    cy: float          # principal point y       [px]   (≈ frame_height / 2)
    baseline_m: float  # left–right lens separation [m]


# ZED 2i factory defaults @ 720p SBS (each half = 1280×720).
# Swap these with your SDK export for sub-percent accuracy.
ZED2I = PinholeCamera(
    fx         = 700.0,   # [px]  ZED 2i @ 720p ≈ 699.7
    fy         = 700.0,   # [px]  typically equal to fx
    cx         = 638.0,   # [px]  ≈ half of 1280 (640) minus small offset
    cy         = 363.0,   # [px]  ≈ half of 720  (360) minus small offset
    baseline_m = 0.12,    # [m ]  ZED 2i physical baseline = 12 cm
)

# Vertical pixel shift applied to the right frame in ZEDCamera.get_frames().
# Positive → right frame shifted DOWN (right lens sits higher in the housing).
# Flip sign if debug_tracking.py still shows ~30 px error after applying it.
Y_OFFSET_PX = 30

# ── Anchor MediaPipe landmark indices used for depth averaging ────────────────
# Wrist + 3 MCP joints: stable, spread across the palm, all well-tracked.
_DEPTH_ANCHORS: List[int] = [0, 5, 9, 13]   # wrist, index-MCP, mid-MCP, ring-MCP


# ── Epipolar constraint ───────────────────────────────────────────────────────

def check_epipolar_constraint(y_left: float, y_right: float,
                            tolerance_px: float = 5) -> tuple:
    """
    Return (is_valid, error_px).

    Y_OFFSET_PX is already baked into the frames by ZEDCamera.get_frames(),
    so this checks only the residual misalignment.
    """
    error = abs(y_left - y_right)
    return error <= tolerance_px, error


# ── Back-projection ───────────────────────────────────────────────────────────

def back_project(u: float, v: float, z_m: float,
                cam: PinholeCamera = ZED2I) -> np.ndarray:
    """
    Convert a pixel (u, v) + depth z_m [metres] to a 3-D camera-frame point.

    Returns
    -------
    np.ndarray shape (3,): [X_m, Y_m, Z_m]
        X_m: positive = right of camera
        Y_m: positive = below  camera  (camera Y points down)
        Z_m: positive = in front of camera
    """
    x_m = (u - cam.cx) * z_m / cam.fx
    y_m = (v - cam.cy) * z_m / cam.fy
    return np.array([x_m, y_m, z_m])


# ── Stereo depth averaging ────────────────────────────────────────────────────

def stereo_depth_m(lm_l, lm_r, frame_w: int,
                    cam: PinholeCamera = ZED2I,
                   anchors: List[int] = _DEPTH_ANCHORS) -> float | None:
    """
    Compute a robust depth estimate [metres] by triangulating several stable
    landmarks and returning their median.

    Averaging multiple landmarks rejects single-landmark outliers from MediaPipe.
    The principal point cancels in the disparity formula (x_L - x_R) so cx
    does not need to be subtracted here.

    Parameters
    ----------
    lm_l, lm_r : MediaPipe landmark lists (21 entries each)
    frame_w    : pixel width of one half-frame (e.g. 1280)
    cam        : PinholeCamera instance
    anchors    : landmark indices to average over

    Returns
    -------
    Median depth in metres, or None if every disparity was ≤0 (bad geometry).
    """
    depths = []
    for i in anchors:
        # Raw pixel x — cx cancels in x_L - x_R, no subtraction needed for Z
        x_l_px = lm_l[i].x * frame_w
        x_r_px = lm_r[i].x * frame_w
        disparity = x_l_px - x_r_px       # positive when hand is in front
        if disparity > 0.5:               # 0.5 px minimum to avoid noise
            depths.append((cam.fx * cam.baseline_m) / disparity)
    return float(np.median(depths)) if depths else None


# ── Full stereo 3-D hand position ─────────────────────────────────────────────

def stereo_hand_3d(lm_l, lm_r, frame_w: int, frame_h: int,
                    cam: PinholeCamera = ZED2I,
                    depth_min_m: float = 0.15,
                    depth_max_m: float = 1.20) -> np.ndarray | None:
    """
    Return the 3-D wrist position [metres] in the left-camera frame.

    Steps
    -----
    1. Triangulate depth from multiple anchor landmarks (median → robust).
    2. Clamp depth to a valid range.
    3. Back-project the wrist pixel (lm_l[0]) using the pinhole model
       to get (X, Y, Z) in metres, with principal-point correction.

    Returns
    -------
    np.ndarray [X_m, Y_m, Z_m] or None if depth is unavailable.
        X_m: positive = right  (mirror to get sim_x)
        Y_m: positive = down   (invert to get sim_z)
        Z_m: positive = forward (map to sim_y)
    """
    z_m = stereo_depth_m(lm_l, lm_r, frame_w, cam)
    if z_m is None:
        return None
    z_m = float(np.clip(z_m, depth_min_m, depth_max_m))

    # Back-project wrist (landmark 0) with principal-point correction
    u_wrist = lm_l[0].x * frame_w
    v_wrist = lm_l[0].y * frame_h
    return back_project(u_wrist, v_wrist, z_m, cam)


# ── Legacy helper (kept for backward compatibility) ───────────────────────────

def triangulate_depth(x_left: float, x_right: float,
                      cam: PinholeCamera = ZED2I) -> float | None:
    """
    Single-landmark depth [cm] via Z = fx * B / disparity.
    Prefer stereo_depth_m() for real use — it averages multiple landmarks.
    """
    disparity = x_left - x_right
    if disparity <= 0:
        return None
    return (cam.fx * cam.baseline_m * 100.0) / disparity  # metres → cm for compat
"""
kalman_gui_fusion.py — Real IMU  +  Webcam MediaPipe  →  Kalman fusion GUI  (X / Y / Z)

  Reads real IMU data from shared memory (teleop) and captures a local
  webcam stream.  MediaPipe Hands runs on the webcam frames to extract
  real hand position.  No ZED SDK required.

  Top row   — Webcam stream (live + MediaPipe overlay)
  Row 1-3   — Real IMU pos | Real Camera Δ | Kalman fused   for X / Y / Z

  Sensor fusion:
    - IMU (LPMS-B2):  euler angles → position  +  accelerometer → velocity
    - Webcam:         MediaPipe palm detection → 3D hand position
    - Kalman filter:  IMU acceleration as process input, camera as measurement

Run from repo root (teleop must be running for IMU):
    Terminal 1:  cd hands/scripts/teleop/mina && python teleop_edgard_new_setup.py
    Terminal 2:  python ui/kalman_gui_fusion.py
"""

import os, sys, time
import numpy as np
import cv2
import mediapipe as mp
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ── Import shared-memory readers (teleop publishes data there) ────────────────
_TELEOP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "hands", "scripts", "teleop", "mina",
)
if os.path.isdir(_TELEOP_DIR) and _TELEOP_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_TELEOP_DIR))

from imu_shm import ImuShmReader       # IMU euler/acc/gyr/quat

# ── MediaPipe setup ───────────────────────────────────────────────────────────
_mp_hands = mp.solutions.hands
_mp_pose  = mp.solutions.pose
_mp_draw  = mp.solutions.drawing_utils
_mp_draw_styles = mp.solutions.drawing_styles

# Hand skeleton connections (same as teleop)
_HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_MCP_IDS = {5, 9, 13, 17}

# Colours for hand overlay (BGR for cv2, but we work on RGB frames)
_COL_LINE_L   = (0, 200, 0)      # left hand lines  (green)
_COL_LINE_R   = (200, 80, 0)     # right hand lines (orange)
_COL_JOINT_L  = (0, 255, 60)
_COL_JOINT_R  = (255, 120, 0)
_COL_MCP_L    = (0, 255, 200)
_COL_MCP_R    = (255, 200, 0)
_COL_POSE_LINE  = (50, 200, 255)   # pose skeleton (cyan-ish)
_COL_POSE_JOINT = (80, 255, 255)

# ── Parameters ─────────────────────────────────────────────────────────────────
ANIM_DT       = 1.0 / 30.0    # GUI refresh rate  → ~30 Hz
WINDOW_SECS   = 10.0
WINDOW_STEPS  = int(WINDOW_SECS / ANIM_DT)

# Real camera tracking (MediaPipe palm detection on webcam frames)
CAM_INDEX     = 0              # /dev/video index for webcam
CAM_W         = 640            # capture width
CAM_H         = 480            # capture height
CAM_SCALE     = 0.5            # scale normalised camera coords to metres
PALM_REF_SIZE = 0.15           # reference palm size (normalised) for depth est.

# IMU → position integration gain
EULER_TO_M    = 1.0 / 180.0   # 180° ≈ 1 m on the graph
ACC_DT        = ANIM_DT       # integration dt
ACC_GAIN      = 0.01           # scale raw accelerometer (prevent blow-up)

# Kalman tuning (real sensors)
KF_Q          = 0.5            # process noise  (IMU accel is noisy)
KF_R          = 0.04           # measurement noise (MediaPipe jitter)

# ── Colours ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#c9d1d9"

AXES = ["x", "y", "z"]

C_IMU = {"x": "#58a6ff", "y": "#3fb950", "z": "#bc8cff"}
C_CAM = {"x": "#ff7b72", "y": "#ffa657", "z": "#f0e68c"}
C_KAL = {"x": "#79c0ff", "y": "#56d364", "z": "#e3b341"}

YLIM_POS   = (-0.3, 0.3)
YLIM_DELTA = (-0.04, 0.04)


# ── Real camera tracker: MediaPipe palm detection on webcam frames ─────────────
_PALM_IDS = (0, 5, 9, 13, 17)   # wrist + 4 MCPs (same as teleop)

class RealCameraTracker:
    """Extract 3D hand position from webcam frames via MediaPipe palm detection."""
    def __init__(self):
        self._ref_pos  = None
        self._prev_pos = np.zeros(3)

    def reset(self):
        self._ref_pos  = None
        self._prev_pos = np.zeros(3)

    def observe(self, hand_result, frame_shape):
        """Return (abs_pos_from_ref, delta) or (None, None) when no hand."""
        if not hand_result or not hand_result.multi_hand_landmarks:
            return None, None

        lm = hand_result.multi_hand_landmarks[0].landmark

        # Palm centre = average of wrist + 4 MCPs (matches teleop)
        u = sum(lm[i].x for i in _PALM_IDS) / len(_PALM_IDS)
        v = sum(lm[i].y for i in _PALM_IDS) / len(_PALM_IDS)

        # X, Y from normalised pixel coords (delta from frame centre)
        pos_x = -(u - 0.5) * CAM_SCALE
        pos_y = -(v - 0.5) * CAM_SCALE

        # Z from apparent palm size (larger palm = closer)
        wrist = lm[0]
        mcp9  = lm[9]
        palm_sz = np.sqrt((wrist.x - mcp9.x)**2 + (wrist.y - mcp9.y)**2)
        pos_z = (palm_sz - PALM_REF_SIZE) * CAM_SCALE * 2.0

        pos = np.array([pos_x, pos_y, pos_z])

        # Auto-calibrate on first valid detection
        if self._ref_pos is None:
            self._ref_pos  = pos.copy()
            self._prev_pos = pos.copy()

        abs_pos = pos - self._ref_pos
        delta   = abs_pos - (self._prev_pos - self._ref_pos)
        self._prev_pos = pos.copy()
        return abs_pos, delta


# ── Kalman filter — 6-state [px,vx, py,vy, pz,vz] ────────────────────────────
class Kalman:
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.q = KF_Q
        self.r = KF_R

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1

    def predict(self, dt: float, imu_acc: np.ndarray = None):
        """Predict with constant-velocity model + IMU acceleration control."""
        F = np.eye(6)
        for i in range(3):
            F[2*i, 2*i+1] = dt
        Q = np.zeros((6, 6))
        for i in range(3):
            ii = 2 * i
            Q[ii,   ii  ] = self.q * dt**3 / 3.0
            Q[ii,   ii+1] = self.q * dt**2 / 2.0
            Q[ii+1, ii  ] = self.q * dt**2 / 2.0
            Q[ii+1, ii+1] = self.q * dt
        self.x = F @ self.x
        # Apply IMU acceleration as control input  (Bu term)
        if imu_acc is not None:
            for i in range(3):
                self.x[2*i]     += 0.5 * imu_acc[i] * dt**2   # position
                self.x[2*i + 1] += imu_acc[i] * dt             # velocity
        self.P = F @ self.P @ F.T + Q

    def update(self, cam_pos: np.ndarray):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 2] = H[2, 4] = 1.0
        R = np.eye(3) * self.r
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (cam_pos - H @ self.x)
        self.P  = (np.eye(6) - K @ H) @ self.P

    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[2], self.x[4]])

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum([self.P[0,0], self.P[2,2], self.P[4,4]], 0.0))

    def kalman_gain(self) -> np.ndarray:
        H = np.zeros((3, 6)); H[0,0]=H[1,2]=H[2,4]=1.0
        R = np.eye(3) * self.r
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        return np.array([K[0,0], K[2,1], K[4,2]])


# ── GUI ────────────────────────────────────────────────────────────────────────
class KalmanFusionGUI:

    def __init__(self):
        self.cam  = RealCameraTracker()
        self.kf   = Kalman()
        self._imu_shm   = ImuShmReader("right")    # IMU from teleop

        # Webcam capture (no ZED SDK needed)
        self._cap = cv2.VideoCapture(CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        # MediaPipe detectors
        self._hands_det = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8,
        )
        self._pose_det = _mp_pose.Pose(
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mp_frame_counter = 0

        # Real IMU state
        self._imu_ref_euler = None
        self._imu_pos       = np.zeros(3)
        self._imu_vel       = np.zeros(3)

        # Placeholder image for webcam (dark grey)
        self._blank_frame = np.full((CAM_H, CAM_W, 3), 30, dtype=np.uint8)

        self._reset_buffers()
        self._build_fig()
        self._build_artists()
        self.fill_kal = [None, None, None]

    def _reset_buffers(self):
        self.cam.reset()
        self.kf.reset()
        self._imu_ref_euler = None
        self._imu_pos = np.zeros(3)
        self._imu_vel = np.zeros(3)

        n = WINDOW_STEPS
        self.times    = np.full(n, np.nan)
        self.imu_buf  = np.full((n, 3), np.nan)
        self.cdelta   = np.full((n, 3), np.nan)
        self.cabs     = np.full((n, 3), np.nan)
        self.kal_buf  = np.full((n, 3), np.nan)
        self.std_buf  = np.full((n, 3), np.nan)
        self.step_idx = 0

    # ── Figure layout ──────────────────────────────────────────────────────────
    def _build_fig(self):
        plt.rcParams.update({
            "figure.facecolor":  BG,
            "axes.facecolor":    PANEL_BG,
            "axes.labelcolor":   TEXT_COL,
            "axes.edgecolor":    GRID_COL,
            "xtick.color":       TEXT_COL,
            "ytick.color":       TEXT_COL,
            "text.color":        TEXT_COL,
            "grid.color":        GRID_COL,
            "grid.linestyle":    "--",
            "grid.alpha":        0.35,
            "lines.linewidth":   1.6,
            "font.family":       "monospace",
        })

        self.fig = plt.figure(figsize=(19, 13), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                "Kalman Fusion  —  Webcam MediaPipe + Real IMU")
        except Exception:
            pass

        # Layout: 4 rows × 3 cols
        #   Row 0  : [Webcam stream (full width)]
        #   Row 1-3: [IMU]  [Camera Δ]  [Kalman]   for X / Y / Z
        gs = GridSpec(
            4, 6, figure=self.fig,
            left=0.05, right=0.97, top=0.94, bottom=0.04,
            hspace=0.55, wspace=0.35,
            height_ratios=[1.3, 1, 1, 1],
        )

        # ── Webcam panel (row 0, full width) ───────────────────────────────────
        self.ax_cam = self.fig.add_subplot(gs[0, :])
        self.ax_cam.set_title("Webcam  (live + MediaPipe)", fontsize=9, pad=4, color=TEXT_COL)
        self.ax_cam.set_xticks([]); self.ax_cam.set_yticks([])
        for spine in self.ax_cam.spines.values():
            spine.set_edgecolor(GRID_COL)
        self._img_cam = self.ax_cam.imshow(self._blank_frame, aspect="equal")

        # ── Kalman graph panels (rows 1-3) ─────────────────────────────────────
        col_titles = [
            "IMU  (LPMS-B2)  —  real",
            "Webcam (MediaPipe)  —  real Δ",
            "Kalman  —  fused estimate",
        ]
        row_labels = ["X", "Y", "Z"]

        self.axes = []
        for row in range(3):
            row_axes = []
            for col in range(3):
                ax = self.fig.add_subplot(gs[row + 1, col * 2:(col + 1) * 2])
                ax.set_xlabel("t  (s)", fontsize=7)
                ax.set_ylabel(f"{row_labels[row]}  (m)", fontsize=7,
                               color=[C_IMU, C_CAM, C_KAL][col][AXES[row]])
                ax.grid(True)
                ax.set_xlim(0, WINDOW_SECS)
                ax.set_ylim(*(YLIM_DELTA if col == 1 else YLIM_POS))
                ax.tick_params(labelsize=6)
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9, pad=5, color=TEXT_COL)
                for spine in ax.spines.values():
                    spine.set_edgecolor(GRID_COL)
                row_axes.append(ax)
            self.axes.append(row_axes)

        self.fig.text(
            0.5, 0.98,
            "Webcam MediaPipe  +  Real IMU  →  Kalman Fusion",
            ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT_COL,
        )

        for row, lbl in enumerate(["X", "Y", "Z"]):
            col = C_IMU[AXES[row]]
            y = 0.69 - row * 0.195
            self.fig.text(0.01, y, lbl,
                          fontsize=14, fontweight="bold", color=col, va="center")

        # Status text
        self._status_txt = self.fig.text(
            0.5, 0.005, "Waiting for webcam + IMU data ...",
            ha="center", fontsize=9, color="#f0883e",
        )

    # ── Artists ────────────────────────────────────────────────────────────────
    def _build_artists(self):
        t   = np.linspace(0, WINDOW_SECS, WINDOW_STEPS)
        nan = np.full(WINDOW_STEPS, np.nan)

        self.ln_imu      = []
        self.ln_cdelta   = []
        self.ln_kal      = []
        self.ln_imu_over = []
        self.ln_cam_over = []
        self.kgain_txt   = []

        for row in range(3):
            k = AXES[row]
            ax0, ax1, ax2 = self.axes[row]

            # Column 0 — IMU position (real)
            li, = ax0.plot(t, nan.copy(), c=C_IMU[k], lw=1.9,
                            label="IMU", zorder=3)
            self.ln_imu.append(li)
            ax0.legend(loc="upper left", fontsize=6, facecolor=PANEL_BG,
                        edgecolor=GRID_COL, labelcolor=TEXT_COL)

            # Column 1 — Camera Δ (real MediaPipe, dots)
            ax1.axhline(0, color=GRID_COL, lw=0.8, alpha=0.6)
            ld, = ax1.plot(t, nan.copy(), ".", c=C_CAM[k], alpha=0.75,
                            ms=3.5, label="cam Δ", zorder=3)
            self.ln_cdelta.append(ld)
            ax1.legend(loc="upper left", fontsize=6, facecolor=PANEL_BG,
                        edgecolor=GRID_COL, labelcolor=TEXT_COL)

            # Column 2 — Kalman fused
            li2, = ax2.plot(t, nan.copy(), c=C_IMU[k], alpha=0.25,
                             lw=1.0, label="IMU", zorder=2)
            lc2, = ax2.plot(t, nan.copy(), ".", c=C_CAM[k], alpha=0.22,
                             ms=2.5, label="cam", zorder=2)
            lk, = ax2.plot(t, nan.copy(), c=C_KAL[k], lw=2.3,
                            label="Kalman", zorder=4)
            self.ln_imu_over.append(li2)
            self.ln_cam_over.append(lc2)
            self.ln_kal.append(lk)

            kg = ax2.text(0.98, 0.94, "K = ?", transform=ax2.transAxes,
                           ha="right", va="top", fontsize=7, color=C_KAL[k])
            self.kgain_txt.append(kg)

            ax2.legend(loc="upper left", fontsize=6, facecolor=PANEL_BG,
                        edgecolor=GRID_COL, labelcolor=TEXT_COL, markerscale=2)

    # ── MediaPipe overlay drawing ─────────────────────────────────────────────
    def _draw_hand_landmarks(self, frame_rgb, hand_result):
        """Draw hand skeleton on an RGB frame (modifies in-place)."""
        if not hand_result.multi_hand_landmarks:
            return
        h, w, _ = frame_rgb.shape
        for i, lm_set in enumerate(hand_result.multi_hand_landmarks):
            label = "Left"
            if hand_result.multi_handedness and i < len(hand_result.multi_handedness):
                label = hand_result.multi_handedness[i].classification[0].label
            col_line  = _COL_LINE_L  if label == "Left" else _COL_LINE_R
            col_joint = _COL_JOINT_L if label == "Left" else _COL_JOINT_R
            col_mcp   = _COL_MCP_L   if label == "Left" else _COL_MCP_R

            pts = [(int(l.x * w), int(l.y * h)) for l in lm_set.landmark]
            for a, b in _HAND_CONNS:
                cv2.line(frame_rgb, pts[a], pts[b], col_line, 2)
            for j, pt in enumerate(pts):
                r, c = (5, col_mcp) if j in _MCP_IDS else (3, col_joint)
                cv2.circle(frame_rgb, pt, r, c, -1)

    def _draw_pose_landmarks(self, frame_rgb, pose_result):
        """Draw body pose skeleton on an RGB frame (modifies in-place)."""
        if pose_result.pose_landmarks is None:
            return
        h, w, _ = frame_rgb.shape
        lm = pose_result.pose_landmarks.landmark
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        # Draw only upper body + arms connections
        conns = [
            (11, 12),                        # shoulders
            (11, 13), (13, 15),              # left arm
            (12, 14), (14, 16),              # right arm
            (11, 23), (12, 24), (23, 24),    # torso
        ]
        for a, b in conns:
            if lm[a].visibility > 0.4 and lm[b].visibility > 0.4:
                cv2.line(frame_rgb, pts[a], pts[b], _COL_POSE_LINE, 2)
        for idx in (11, 12, 13, 14, 15, 16, 23, 24):
            if lm[idx].visibility > 0.4:
                cv2.circle(frame_rgb, pts[idx], 4, _COL_POSE_JOINT, -1)

    def _read_webcam(self):
        """Capture one frame from webcam, return RGB or None."""
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Read real IMU → position + acceleration ──────────────────────────────────
    def _read_imu(self):
        """Read LPMS-B2 right IMU → (position_m, acceleration_m/s²)."""
        snap = self._imu_shm.read()
        _zero = np.zeros(3)
        if snap is None:
            return self._imu_pos.copy(), _zero

        euler = snap.get("euler")
        acc   = snap.get("acc")

        if euler is None:
            return self._imu_pos.copy(), _zero

        euler = np.array(euler, dtype=float)   # [roll, pitch, yaw] degrees

        # Auto-calibrate on first valid packet
        if self._imu_ref_euler is None:
            self._imu_ref_euler = euler.copy()
            self._status_txt.set_text("IMU + Camera connected  —  calibrated")
            self._status_txt.set_color("#3fb950")

        # Delta euler from rest → metres
        d_euler = (euler - self._imu_ref_euler) * EULER_TO_M
        # Remap: X ← pitch, Y ← roll, Z ← yaw
        d_euler = d_euler[[1, 0, 2]]

        # Accelerometer processing
        acc_remapped = _zero
        if acc is not None:
            acc_np = np.array(acc, dtype=float)
            acc_np[2] -= 9.81              # remove gravity
            acc_np = acc_np[[1, 0, 2]]     # remap same as euler
            acc_remapped = acc_np.copy()
            self._imu_vel += acc_np * ACC_DT * ACC_GAIN
            self._imu_vel *= 0.98          # damping

        # Position: euler-based + integrated acc
        euler_pos = d_euler
        acc_pos   = self._imu_vel * ACC_DT
        self._imu_pos = 0.7 * euler_pos + 0.3 * (self._imu_pos + acc_pos)

        return self._imu_pos.copy(), acc_remapped

    # ── Tick ───────────────────────────────────────────────────────────────────
    def _tick(self, hand_result=None, frame_shape=None):
        dt = ANIM_DT

        # ── Real IMU → position + acceleration ────────────────────────────────
        imu_pos, imu_acc = self._read_imu()

        # ── Kalman predict with IMU acceleration as control input ─────────────
        self.kf.predict(dt, imu_acc * ACC_GAIN)

        # ── Real camera observation (MediaPipe palm detection) ────────────────
        cam_obs, cam_delta = self.cam.observe(hand_result, frame_shape)
        if cam_obs is not None:
            self.kf.update(cam_obs)

        kal_pos = self.kf.position()
        kal_std = self.kf.std()
        kal_k   = self.kf.kalman_gain()

        i     = self.step_idx % WINDOW_STEPS
        t_now = self.step_idx * ANIM_DT

        self.times[i]    = t_now % WINDOW_SECS
        self.imu_buf[i]  = imu_pos
        self.cdelta[i]   = cam_delta if cam_delta is not None else np.full(3, np.nan)
        self.cabs[i]     = cam_obs   if cam_obs   is not None else np.full(3, np.nan)
        self.kal_buf[i]  = kal_pos
        self.std_buf[i]  = kal_std

        self.step_idx += 1
        return kal_k

    # ── Animation frame ────────────────────────────────────────────────────────
    def _animate(self, _frame):
        # ── Read webcam & run MediaPipe ───────────────────────────────────────
        rgb = self._read_webcam()
        hand_result = None
        frame_shape = None
        self._mp_frame_counter += 1
        run_pose = (self._mp_frame_counter % 3 == 0)

        if rgb is not None:
            frame_shape = rgb.shape
            # Hand detection every frame → drives position tracking
            hand_result = self._hands_det.process(rgb)
            out = rgb.copy()
            self._draw_hand_landmarks(out, hand_result)
            if run_pose:
                pose_res = self._pose_det.process(rgb)
                self._draw_pose_landmarks(out, pose_res)
            self._img_cam.set_data(out)

        # ── Tick with real sensor data ────────────────────────────────────────
        kal_k = self._tick(hand_result, frame_shape)

        order = np.argsort(self.times)
        t_s   = self.times[order]

        for row in range(3):
            k = AXES[row]
            ax0, ax1, ax2 = self.axes[row]

            im_s = self.imu_buf [order, row]
            cd_s = self.cdelta  [order, row]
            ca_s = self.cabs    [order, row]
            kl_s = self.kal_buf [order, row]
            sd_s = self.std_buf [order, row]

            # IMU
            self.ln_imu[row].set_xdata(t_s)
            self.ln_imu[row].set_ydata(im_s)

            # Camera delta
            self.ln_cdelta[row].set_xdata(t_s)
            self.ln_cdelta[row].set_ydata(cd_s)

            # Kalman + overlays
            self.ln_imu_over[row].set_xdata(t_s); self.ln_imu_over[row].set_ydata(im_s)
            self.ln_cam_over[row].set_xdata(t_s); self.ln_cam_over[row].set_ydata(ca_s)
            self.ln_kal[row].set_xdata(t_s);      self.ln_kal[row].set_ydata(kl_s)

            # Sigma fill
            if self.fill_kal[row] is not None:
                try:    self.fill_kal[row].remove()
                except Exception: pass
            self.fill_kal[row] = ax2.fill_between(
                t_s, kl_s - sd_s, kl_s + sd_s,
                color=C_KAL[k], alpha=0.14,
            )

            self.kgain_txt[row].set_text(f"K = {kal_k[row]:.3f}")

        return []

    # ── Run ────────────────────────────────────────────────────────────────────
    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=int(ANIM_DT * 1000),
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("Kalman GUI — Real IMU + Webcam MediaPipe fusion")
    print("Make sure teleop_edgard_new_setup.py is running for IMU data!")

    gui = KalmanFusionGUI()
    try:
        gui.run()
    finally:
        gui._hands_det.close()
        gui._pose_det.close()
        gui._imu_shm.close()
        if gui._cap.isOpened():
            gui._cap.release()
        print("GUI closed.")


if __name__ == "__main__":
    main()

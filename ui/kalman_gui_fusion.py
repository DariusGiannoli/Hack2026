"""
kalman_gui_fusion.py — Real IMU  +  Fake Camera  →  Kalman fusion GUI  (X / Y / Z)

  Reads real IMU data from shared memory published by teleop_edgard_new_setup.py.
  Must be launched SIMULTANEOUSLY with teleop (which owns the Bluetooth connection).

  Column 1 — IMU  (real LPMS-B2 data via shared memory)
               euler[0..2] from the right-hand IMU, converted to degrees,
               double-integrated acc → position shown as a drifty signal.

  Column 2 — Camera Δ  (fake)
               Noisy position observations built from the filtered IMU signal
               + added Gaussian noise + random dropout.  Purely synthetic.

  Column 3 — Kalman fused  ±1σ
               6-state constant-velocity filter.
               • Predict at ~30 Hz using IMU-derived velocity
               • Update when "camera" fires (with noise + dropout)

Run from repo root (teleop must be running first):
    Terminal 1:  cd hands/scripts/teleop/mina && python teleop_edgard_new_setup.py
    Terminal 2:  python ui/kalman_gui_fusion.py
"""

import os, sys, time
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ── Import the IMU shared-memory reader (teleop publishes data there) ─────────
_TELEOP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "hands", "scripts", "teleop", "mina",
)
if os.path.isdir(_TELEOP_DIR) and _TELEOP_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_TELEOP_DIR))

from imu_shm import ImuShmReader   # reads shared memory written by teleop

# ── Parameters ─────────────────────────────────────────────────────────────────
ANIM_DT       = 1.0 / 30.0    # GUI refresh rate  → ~30 Hz
WINDOW_SECS   = 10.0
WINDOW_STEPS  = int(WINDOW_SECS / ANIM_DT)

# Fake camera noise
CAM_NOISE     = 0.04           # std-dev added to "camera" observations (metres)
CAM_DROPOUT   = 0.15           # fraction of camera frames randomly dropped

# IMU → position integration gain
# euler angles are in degrees; we scale them to "metres" for display
EULER_TO_M    = 1.0 / 180.0   # 180° ≈ 1 m on the graph
ACC_DT        = ANIM_DT       # integration dt

# Kalman tuning
KF_Q          = 0.08           # process noise  (trust IMU prediction)
KF_R          = 0.015          # measurement noise (trust fake camera)

# ── Colours ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#c9d1d9"

AXES = ["x", "y", "z"]

C_IMU = {"x": "#58a6ff", "y": "#3fb950", "z": "#bc8cff"}
C_CAM = {"x": "#ff7b72", "y": "#ffa657", "z": "#f0e68c"}
C_KAL = {"x": "#79c0ff", "y": "#56d364", "z": "#e3b341"}

YLIM_POS   = (-1.2, 1.2)
YLIM_DELTA = (-0.15, 0.15)


# ── Fake camera: adds noise on top of a reference signal ──────────────────────
class FakeCamera:
    """Observe a reference position with Gaussian noise + random dropout."""
    def __init__(self):
        self._prev = np.zeros(3)

    def reset(self):
        self._prev = np.zeros(3)

    def observe(self, ref_pos: np.ndarray):
        """Returns (obs, delta) or (None, None) on dropout."""
        if np.random.rand() < CAM_DROPOUT:
            return None, None
        obs   = ref_pos + np.random.randn(3) * CAM_NOISE
        delta = obs - self._prev
        self._prev = obs.copy()
        return obs, delta


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

    def predict(self, dt: float):
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
        self.cam  = FakeCamera()
        self.kf   = Kalman()
        self._imu_shm = ImuShmReader("right")   # reads from teleop's shared memory

        # Real IMU state
        self._imu_ref_euler = None   # captured at first valid packet (calibration zero)
        self._imu_pos       = np.zeros(3)   # double-integrated position
        self._imu_vel       = np.zeros(3)   # integrated velocity

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

        self.fig = plt.figure(figsize=(19, 10), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                "Real IMU  +  Fake Camera  →  Kalman Fusion   (X / Y / Z)")
        except Exception:
            pass

        gs = GridSpec(
            3, 3, figure=self.fig,
            left=0.07, right=0.97, top=0.91, bottom=0.05,
            hspace=0.62, wspace=0.32,
        )

        col_titles = [
            "IMU  (real LPMS-B2)",
            "Camera  (fake)  —  Δ pos / frame",
            "Kalman  —  fused estimate",
        ]
        row_labels = ["X", "Y", "Z"]

        self.axes = []
        for row in range(3):
            row_axes = []
            for col in range(3):
                ax = self.fig.add_subplot(gs[row, col])
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
            0.5, 0.97,
            "Real IMU  ⊕  Fake Camera  →  Kalman Fusion        X / Y / Z",
            ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT_COL,
        )

        for x in (0.375, 0.685):
            self.fig.add_artist(
                plt.Line2D([x, x], [0.04, 0.93],
                            transform=self.fig.transFigure,
                            color=GRID_COL, lw=1.0, alpha=0.5))

        for row, lbl in enumerate(["X", "Y", "Z"]):
            col = C_IMU[AXES[row]]
            self.fig.text(0.01, 0.795 - row * 0.225, lbl,
                          fontsize=16, fontweight="bold", color=col, va="center")

        # Status text (shows IMU connection state)
        self._status_txt = self.fig.text(
            0.5, 0.005, "Waiting for teleop IMU data (shared memory) ...",
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
                            label="IMU (real)", zorder=3)
            self.ln_imu.append(li)
            ax0.legend(loc="upper left", fontsize=6, facecolor=PANEL_BG,
                        edgecolor=GRID_COL, labelcolor=TEXT_COL)

            # Column 1 — Camera Δ (fake, dots)
            ax1.axhline(0, color=GRID_COL, lw=0.8, alpha=0.6)
            ld, = ax1.plot(t, nan.copy(), ".", c=C_CAM[k], alpha=0.75,
                            ms=3.5, label="cam Δ (fake)", zorder=3)
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

    # ── Read real IMU → position ───────────────────────────────────────────────
    def _read_imu(self) -> np.ndarray:
        """Read LPMS-B2 right IMU from shared memory → return position (m)."""
        snap = self._imu_shm.read()
        if snap is None:
            return self._imu_pos.copy()

        euler = snap.get("euler")
        acc   = snap.get("acc")

        if euler is None:
            return self._imu_pos.copy()

        euler = np.array(euler, dtype=float)   # [roll, pitch, yaw] in degrees

        # Auto-calibrate on first valid packet
        if self._imu_ref_euler is None:
            self._imu_ref_euler = euler.copy()
            self._status_txt.set_text("IMU connected  —  calibrated at rest pose")
            self._status_txt.set_color("#3fb950")

        # Delta euler from rest (degrees → metres via scale)
        d_euler = (euler - self._imu_ref_euler) * EULER_TO_M

        # Also integrate accelerometer for velocity → position (double integration)
        if acc is not None:
            acc_np = np.array(acc, dtype=float)
            # Remove rough gravity estimate (assume Z up ≈ 9.81)
            acc_np[2] -= 9.81
            self._imu_vel += acc_np * ACC_DT * 0.01   # small gain to avoid blow-up
            # Damping to limit drift
            self._imu_vel *= 0.98

        # Fuse euler-based position + integrated acc (weighted sum)
        euler_pos = d_euler
        acc_pos   = self._imu_vel * ACC_DT
        self._imu_pos = 0.7 * euler_pos + 0.3 * (self._imu_pos + acc_pos)

        return self._imu_pos.copy()

    # ── Tick ───────────────────────────────────────────────────────────────────
    def _tick(self):
        dt = ANIM_DT

        # Real IMU → position
        imu_pos = self._read_imu()

        # Kalman predict (uses IMU as process model)
        self.kf.predict(dt)

        # Fake camera: observe the IMU position + noise + dropout
        cam_obs, cam_delta = self.cam.observe(imu_pos)
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
        kal_k = self._tick()

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
    print("Kalman GUI — waiting for IMU data from teleop (shared memory) ...")
    print("Make sure teleop_edgard_new_setup.py is running first!")

    gui = KalmanFusionGUI()
    try:
        gui.run()
    finally:
        gui._imu_shm.close()
        print("GUI closed.")


if __name__ == "__main__":
    main()

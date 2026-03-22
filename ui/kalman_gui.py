"""
kalman_gui.py — Real-time Kalman filter visualization for X, Y, Z translation.

Shows per-axis time series:
  • Grey dots  : noisy measurements
  • Coloured line : Kalman estimate
  • Shaded band   : ±1σ uncertainty

Also shows a live 3-D trajectory in the fourth panel.

Controls (matplotlib sliders + button):
  Q  — process noise  (higher → filter trusts measurements more)
  R  — measurement noise (higher → filter smooths more)
  [Reset] — restart simulation

Run:
    python kalman_gui.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headlessly when TkAgg unavailable → falls back
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)

# ── Simulation parameters ──────────────────────────────────────────────────────
SIM_DT       = 0.05     # seconds per step → 20 Hz
WINDOW_SECS  = 10.0     # seconds of history shown in time-series
WINDOW_STEPS = int(WINDOW_SECS / SIM_DT)
MEAS_NOISE   = 0.07     # true sensor std-dev (fixed — independent of R slider)
TRAIL_LEN    = 300      # max points in 3-D trail

# ── Colour palette (dark-ish theme) ────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL_BG  = "#16213e"
TEXT_COL  = "#e0e0e0"
GRID_COL  = "#2a2a4a"
C = {
    "x": "#ff6b6b",   # red
    "y": "#4ecdc4",   # teal
    "z": "#ffe66d",   # yellow
}
MEAS_ALPHA  = 0.35
TRUE_ALPHA  = 0.45
SIGMA_ALPHA = 0.20

# ── 1-D constant-velocity Kalman filter ───────────────────────────────────────
class KalmanFilter1D:
    """State = [position, velocity].  Measurement = position only."""

    def __init__(self, q: float = 0.05, r: float = 0.5):
        self.q = q
        self.r = r
        self._build_matrices()
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0

    def _build_matrices(self):
        dt = SIM_DT
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        q = self.q
        self.Q = np.array([
            [q * dt**3 / 3.0,  q * dt**2 / 2.0],
            [q * dt**2 / 2.0,  q * dt],
        ])
        self.R = np.array([[self.r]])

    def set_noise(self, q: float, r: float):
        self.q = q
        self.r = r
        self._build_matrices()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x = self.x + K.flatten() * (z - (self.H @ self.x)[0])
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P

    def step(self, z: float):
        """Predict + update.  Returns (estimate, std_dev)."""
        self.predict()
        self.update(z)
        return float(self.x[0]), float(np.sqrt(max(self.P[0, 0], 0.0)))

    def reset(self, z: float = 0.0):
        self.x = np.array([z, 0.0])
        self.P = np.eye(2) * 1.0


# ── Trajectory generator (fake "true" signal + noise) ─────────────────────────
class FakeTrajectory:
    """Smooth underlying motion that the Kalman filter is trying to recover."""

    def __init__(self):
        self.t = 0.0

    def reset(self):
        self.t = 0.0

    def true_xyz(self) -> np.ndarray:
        t = self.t
        x = 0.30 * np.sin(2 * np.pi * t / 8.0)
        y = 0.20 * np.sin(2 * np.pi * t / 5.0 + 1.0)
        z = 0.15 * np.sin(2 * np.pi * t / 3.5 + 2.0) + 0.35
        return np.array([x, y, z])

    def measure(self) -> np.ndarray:
        return self.true_xyz() + np.random.randn(3) * MEAS_NOISE

    def advance(self):
        self.t += SIM_DT


# ── GUI class ──────────────────────────────────────────────────────────────────
class KalmanGUI:

    # ── Initial slider values ──
    Q_INIT = 0.05
    R_INIT = 0.50

    def __init__(self):
        self.traj   = FakeTrajectory()
        self.kf_x   = KalmanFilter1D(self.Q_INIT, self.R_INIT)
        self.kf_y   = KalmanFilter1D(self.Q_INIT, self.R_INIT)
        self.kf_z   = KalmanFilter1D(self.Q_INIT, self.R_INIT)
        self._reset_buffers()
        self._build_figure()
        self._build_artists()
        self._build_widgets()

    # ── Data buffers ──────────────────────────────────────────────────────────
    def _reset_buffers(self):
        n = WINDOW_STEPS
        self.times   = np.full(n, np.nan)
        self.true_x  = np.full(n, np.nan)
        self.true_y  = np.full(n, np.nan)
        self.true_z  = np.full(n, np.nan)
        self.meas_x  = np.full(n, np.nan)
        self.meas_y  = np.full(n, np.nan)
        self.meas_z  = np.full(n, np.nan)
        self.est_x   = np.full(n, np.nan)
        self.est_y   = np.full(n, np.nan)
        self.est_z   = np.full(n, np.nan)
        self.std_x   = np.full(n, np.nan)
        self.std_y   = np.full(n, np.nan)
        self.std_z   = np.full(n, np.nan)
        self.trail_x = []
        self.trail_y = []
        self.trail_z = []
        self.step_idx = 0
        self.traj.reset()
        self.kf_x.reset()
        self.kf_y.reset()
        self.kf_z.reset()

    # ── Figure layout ─────────────────────────────────────────────────────────
    def _build_figure(self):
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
            "grid.alpha":        0.5,
            "lines.linewidth":   1.8,
            "font.family":       "monospace",
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Kalman Filter — Translation X Y Z")

        # Grid: 3 rows × 2 cols, right column merged for 3-D
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            4, 2,
            figure=self.fig,
            left=0.07, right=0.97,
            top=0.91,  bottom=0.15,
            hspace=0.55, wspace=0.30,
            height_ratios=[1, 1, 1, 0.08],
        )

        self.ax_x  = self.fig.add_subplot(gs[0, 0])
        self.ax_y  = self.fig.add_subplot(gs[1, 0])
        self.ax_z  = self.fig.add_subplot(gs[2, 0])
        self.ax_3d = self.fig.add_subplot(gs[0:3, 1], projection="3d")

        for ax, label, col in (
            (self.ax_x, "X  (m)", C["x"]),
            (self.ax_y, "Y  (m)", C["y"]),
            (self.ax_z, "Z  (m)", C["z"]),
        ):
            ax.set_ylabel(label, color=col, fontsize=9)
            ax.set_xlabel("t  (s)", fontsize=8)
            ax.grid(True)
            ax.set_xlim(0, WINDOW_SECS)
            ax.set_ylim(-0.8, 0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)

        # 3-D axes styling
        self.ax_3d.set_facecolor(PANEL_BG)
        self.ax_3d.set_xlabel("X", color=C["x"], fontsize=8)
        self.ax_3d.set_ylabel("Y", color=C["y"], fontsize=8)
        self.ax_3d.set_zlabel("Z", color=C["z"], fontsize=8)
        self.ax_3d.set_xlim(-0.5, 0.5)
        self.ax_3d.set_ylim(-0.5, 0.5)
        self.ax_3d.set_zlim( 0.0, 0.8)
        self.ax_3d.tick_params(colors=TEXT_COL, labelsize=7)
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor(GRID_COL)
        self.ax_3d.yaxis.pane.set_edgecolor(GRID_COL)
        self.ax_3d.zaxis.pane.set_edgecolor(GRID_COL)

        # Title
        self.fig.text(
            0.5, 0.97,
            "Kalman Filter  —  Translation  X / Y / Z",
            ha="center", va="top",
            fontsize=14, fontweight="bold", color=TEXT_COL,
        )

    # ── Matplotlib artists ────────────────────────────────────────────────────
    def _build_artists(self):
        t = np.linspace(0, WINDOW_SECS, WINDOW_STEPS)

        # Helper to build per-axis plots
        def _make_axis_artists(ax, col):
            meas_sc, = ax.plot(
                t, np.full_like(t, np.nan), ".",
                color=col, alpha=MEAS_ALPHA, markersize=3, label="measurement",
            )
            true_ln, = ax.plot(
                t, np.full_like(t, np.nan), "--",
                color=col, alpha=TRUE_ALPHA, linewidth=1.2, label="true",
            )
            est_ln,  = ax.plot(
                t, np.full_like(t, np.nan),
                color=col, linewidth=2.0, label="Kalman",
            )
            sigma_fill = ax.fill_between(
                t,
                np.full_like(t, np.nan),
                np.full_like(t, np.nan),
                color=col, alpha=SIGMA_ALPHA, label="±1σ",
            )
            # Kalman gain text (top-right of each panel)
            gain_txt = ax.text(
                0.98, 0.93, "K = ?",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8, color=col,
            )
            return meas_sc, true_ln, est_ln, sigma_fill, gain_txt

        (self.mx_sc, self.tx_ln, self.ex_ln, self.sx_fill, self.gx_txt
         ) = _make_axis_artists(self.ax_x, C["x"])
        (self.my_sc, self.ty_ln, self.ey_ln, self.sy_fill, self.gy_txt
         ) = _make_axis_artists(self.ax_y, C["y"])
        (self.mz_sc, self.tz_ln, self.ez_ln, self.sz_fill, self.gz_txt
         ) = _make_axis_artists(self.ax_z, C["z"])

        for ax in (self.ax_x, self.ax_y, self.ax_z):
            ax.legend(
                loc="upper left", fontsize=7,
                facecolor=PANEL_BG, edgecolor=GRID_COL,
                labelcolor=TEXT_COL, markerscale=1.5,
            )

        # 3-D trail (measurements) and estimate path
        self.trail_meas, = self.ax_3d.plot(
            [], [], [], ".", color="#aaaaaa", alpha=0.25, markersize=2,
        )
        self.trail_est, = self.ax_3d.plot(
            [], [], [],
            "-", color="#ffffff", linewidth=1.2, alpha=0.8,
        )
        self.point_est = self.ax_3d.plot(
            [], [], [], "o",
            color="#ffffff", markersize=6,
        )[0]
        # Axis projections of current estimate
        self.proj_x, = self.ax_3d.plot([], [], [], "-", color=C["x"], alpha=0.5, lw=1)
        self.proj_y, = self.ax_3d.plot([], [], [], "-", color=C["y"], alpha=0.5, lw=1)
        self.proj_z, = self.ax_3d.plot([], [], [], "-", color=C["z"], alpha=0.5, lw=1)

        self.ax_3d.set_title("3-D Trajectory", color=TEXT_COL, fontsize=10, pad=6)

    # ── Slider & button widgets ────────────────────────────────────────────────
    def _build_widgets(self):
        col_slider = "#0f3460"
        col_handle = "#e94560"

        ax_q = self.fig.add_axes([0.10, 0.07, 0.35, 0.025], facecolor=col_slider)
        ax_r = self.fig.add_axes([0.10, 0.03, 0.35, 0.025], facecolor=col_slider)

        self.sl_q = Slider(
            ax_q, "Q process noise", 1e-4, 5.0,
            valinit=self.Q_INIT, color=col_handle,
        )
        self.sl_r = Slider(
            ax_r, "R meas. noise  ", 1e-4, 5.0,
            valinit=self.R_INIT, color=col_handle,
        )
        for sl in (self.sl_q, self.sl_r):
            sl.label.set_color(TEXT_COL)
            sl.valtext.set_color(TEXT_COL)

        ax_btn = self.fig.add_axes([0.55, 0.03, 0.10, 0.055], facecolor="#0f3460")
        self.btn_reset = Button(ax_btn, "Reset", color="#0f3460", hovercolor="#e94560")
        self.btn_reset.label.set_color(TEXT_COL)
        self.btn_reset.on_clicked(self._on_reset)

        self.sl_q.on_changed(self._on_noise_change)
        self.sl_r.on_changed(self._on_noise_change)

        # Info text (bottom right)
        self.fig.text(
            0.75, 0.07,
            "Q ↑ → less smoothing\nR ↑ → more smoothing",
            ha="left", va="bottom",
            fontsize=8, color="#888888",
        )

    # ── Widget callbacks ───────────────────────────────────────────────────────
    def _on_noise_change(self, _val):
        q = float(self.sl_q.val)
        r = float(self.sl_r.val)
        for kf in (self.kf_x, self.kf_y, self.kf_z):
            kf.set_noise(q, r)

    def _on_reset(self, _event):
        self._reset_buffers()
        self._on_noise_change(None)   # re-apply slider values

    # ── Single simulation step ─────────────────────────────────────────────────
    def _tick(self):
        xyz_true = self.traj.true_xyz()
        xyz_meas = self.traj.measure()
        self.traj.advance()

        ex, sx = self.kf_x.step(xyz_meas[0])
        ey, sy = self.kf_y.step(xyz_meas[1])
        ez, sz = self.kf_z.step(xyz_meas[2])

        i = self.step_idx % WINDOW_STEPS
        t_now = self.step_idx * SIM_DT

        self.times[i]  = t_now % WINDOW_SECS   # rolling time axis
        self.true_x[i] = xyz_true[0]
        self.true_y[i] = xyz_true[1]
        self.true_z[i] = xyz_true[2]
        self.meas_x[i] = xyz_meas[0]
        self.meas_y[i] = xyz_meas[1]
        self.meas_z[i] = xyz_meas[2]
        self.est_x[i]  = ex
        self.est_y[i]  = ey
        self.est_z[i]  = ez
        self.std_x[i]  = sx
        self.std_y[i]  = sy
        self.std_z[i]  = sz

        self.trail_x.append(xyz_meas[0])
        self.trail_y.append(xyz_meas[1])
        self.trail_z.append(xyz_meas[2])
        if len(self.trail_x) > TRAIL_LEN:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
            self.trail_z.pop(0)

        self.step_idx += 1

        # Compute Kalman gain for display (approx from state covariance)
        def _gain(kf):
            h = kf.H
            P = kf.P
            R = kf.R
            S = h @ P @ h.T + R
            K = P @ h.T / S[0, 0]
            return float(K[0])

        return (ex, ey, ez, sx, sy, sz,
                _gain(self.kf_x), _gain(self.kf_y), _gain(self.kf_z))

    # ── Animation frame ───────────────────────────────────────────────────────
    def _animate(self, _frame):
        # Advance simulation one step
        ex, ey, ez, sx, sy, sz, kx, ky, kz = self._tick()

        # Build sorted time index for plotting (handles rolling buffer)
        order = np.argsort(self.times)
        t_s  = self.times[order]
        # Shift so current window always shows 0..WINDOW_SECS
        # (simply use raw buffer sorted by time)

        def _update_axis(
            ax, meas_sc, true_ln, est_ln, sigma_fill, gain_txt,
            t_s, meas, true_, est, std, col, k_val,
        ):
            m_s   = meas[order]
            tr_s  = true_[order]
            e_s   = est[order]
            sd_s  = std[order]

            meas_sc.set_xdata(t_s)
            meas_sc.set_ydata(m_s)
            true_ln.set_xdata(t_s)
            true_ln.set_ydata(tr_s)
            est_ln.set_xdata(t_s)
            est_ln.set_ydata(e_s)

            # Redraw fill_between
            for coll in ax.collections:
                coll.remove()
            ax.fill_between(
                t_s, e_s - sd_s, e_s + sd_s,
                color=col, alpha=SIGMA_ALPHA,
            )
            # Re-add projections/legends that were cleared
            gain_txt.set_text(f"K = {k_val:.3f}")

        _update_axis(
            self.ax_x,
            self.mx_sc, self.tx_ln, self.ex_ln, self.sx_fill, self.gx_txt,
            t_s, self.meas_x, self.true_x, self.est_x, self.std_x,
            C["x"], kx,
        )
        _update_axis(
            self.ax_y,
            self.my_sc, self.ty_ln, self.ey_ln, self.sy_fill, self.gy_txt,
            t_s, self.meas_y, self.true_y, self.est_y, self.std_y,
            C["y"], ky,
        )
        _update_axis(
            self.ax_z,
            self.mz_sc, self.tz_ln, self.ez_ln, self.sz_fill, self.gz_txt,
            t_s, self.meas_z, self.true_z, self.est_z, self.std_z,
            C["z"], kz,
        )

        # 3-D trail
        self.trail_meas.set_data_3d(self.trail_x, self.trail_y, self.trail_z)

        # Estimate trail from buffers
        valid = ~np.isnan(self.est_x)
        if valid.any():
            self.trail_est.set_data_3d(
                self.est_x[valid], self.est_y[valid], self.est_z[valid],
            )
        self.point_est.set_data_3d([ex], [ey], [ez])

        # Axis drop-lines from current estimate to floor/walls
        self.proj_x.set_data_3d([ex, -0.5], [ey, ey], [ez, ez])
        self.proj_y.set_data_3d([ex, ex], [ey, -0.5], [ez, ez])
        self.proj_z.set_data_3d([ex, ex], [ey, ey], [ez, 0.0])

        return []

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=int(SIM_DT * 1000),   # ms: matches simulation dt
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui = KalmanGUI()
    gui.run()

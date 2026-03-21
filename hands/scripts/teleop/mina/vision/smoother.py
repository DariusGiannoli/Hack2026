"""
One Euro Filter — speed-adaptive low-pass filter for real-time joint smoothing.

Better than a fixed EMA for teleoperation:
  - Low cutoff at low speeds  → eliminates jitter when hand is still
  - High cutoff at high speeds → minimal lag during fast movements

Original paper: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012.

Works on scalar or arbitrary-dimensional NumPy arrays (all dims filtered
independently with per-element adaptive cutoffs).
"""

import numpy as np


class OneEuroFilter:
    def __init__(self,
                 freq: float,
                 min_cutoff: float = 1.0,
                 beta: float = 0.007,
                 d_cutoff: float = 1.0):
        """
        Args:
            freq:       Expected sampling frequency (Hz). Used to scale the
                        derivative estimate. Pass 30.0 for a ~30 Hz vision loop.
            min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother at rest.
                        Typical values: 0.5–2.0 Hz.
            beta:       Speed coefficient. Higher = faster response to motion.
                        Typical values: 0.001–0.1.
            d_cutoff:   Cutoff for the internal derivative filter. Usually 1.0.
        """
        self.freq       = freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff

        self._x:  np.ndarray | None = None
        self._dx: np.ndarray | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _alpha(cutoff: np.ndarray, freq: float) -> np.ndarray:
        """Compute smoothing factor α from cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    # ------------------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a new sample.

        Args:
            x: New observation (scalar or ndarray of any shape).

        Returns:
            Filtered value with the same shape as x.
        """
        x = np.asarray(x, dtype=float)

        # First call — initialise and return raw value (no lag on startup)
        if self._x is None:
            self._x  = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()

        # --- Derivative estimate -----------------------------------------
        dx_raw = (x - self._x) * self.freq
        a_d    = self._alpha(np.full_like(x, self.d_cutoff), self.freq)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        # --- Adaptive cutoff based on signal speed -----------------------
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        alpha  = self._alpha(cutoff, self.freq)

        # --- Filter signal -----------------------------------------------
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x.copy()

    def reset(self):
        """Clear internal state (call when tracking is lost / restarted)."""
        self._x  = None
        self._dx = None

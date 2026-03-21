"""
arm_ik.py — Damped Least-Squares IK for humanoid arms.

solve() performs up to `ik_max_iters` DLS passes on the real MjData.
Improvements over the original:
  - One fewer mj_forward per call (init pass reused for iter 0).
  - Pre-cached regularisation matrix (no per-call allocation).
  - Adaptive damping: λ scales continuously with error magnitude.
  - Null-space secondary objective: pulls redundant joints (≥6 DOF)
    toward a rest pose, avoiding singularities and joint-limit creep.
  - act_indices validated at init (warns on missing actuator names).
"""

import numpy as np
import mujoco

# ── Joint / actuator / EE names per side ─────────────────────────────────────
_RIGHT_JOINTS = [
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]
_RIGHT_ACTS = [
    "hold_arm_right_shoulder_pitch_joint",
    "hold_arm_right_shoulder_roll_joint",
    "hold_arm_right_shoulder_yaw_joint",
    "hold_arm_right_elbow_pitch_joint",
    "hold_arm_right_elbow_roll_joint",
]
_RIGHT_EE = "arm_right_hand_link"

_LEFT_JOINTS = [
    "arm_left_shoulder_pitch_joint",
    "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
]
_LEFT_ACTS = [
    "hold_arm_left_shoulder_pitch_joint",
    "hold_arm_left_shoulder_roll_joint",
    "hold_arm_left_shoulder_yaw_joint",
    "hold_arm_left_elbow_pitch_joint",
    "hold_arm_left_elbow_roll_joint",
]
_LEFT_EE = "arm_left_hand_link"

# ── G1 humanoid arm config ────────────────────────────────────────────────────
_G1_RIGHT_JOINTS = [
    "g1_right_shoulder_pitch_joint",
    "g1_right_shoulder_roll_joint",
    "g1_right_shoulder_yaw_joint",
    "g1_right_elbow_joint",
    "g1_right_wrist_roll_joint",
    "g1_right_wrist_pitch_joint",
    "g1_right_wrist_yaw_joint",
]
_G1_RIGHT_EE = "g1_right_wrist_yaw_link"

_G1_LEFT_JOINTS = [
    "g1_left_shoulder_pitch_joint",
    "g1_left_shoulder_roll_joint",
    "g1_left_shoulder_yaw_joint",
    "g1_left_elbow_joint",
    "g1_left_wrist_roll_joint",
    "g1_left_wrist_pitch_joint",
    "g1_left_wrist_yaw_joint",
]
_G1_LEFT_EE = "g1_left_wrist_yaw_link"


class ArmIKSolver:
    """Damped Least-Squares IK with null-space rest-pose control.

    Parameters
    ----------
    model : MjModel
    side : str
        ``"right"`` | ``"left"`` | ``"g1_right"`` | ``"g1_left"``
    damping : float
        Base DLS regularisation λ.  Scales up automatically with error.
    ik_step : float
        Step gain ∈ (0, 1].  Lower = smoother but slower convergence.
    ik_max_iters : int
        Max iterations in normal mode (error < recovery_err_mm).
    ik_err_stop_mm : float
        Early-stop threshold in mm.
    joint_weights : list | None
        Per-joint weight vector (higher = joint penalised more).
        Joints with lower weight move first (preferred joints).
    recovery_err_mm : float
        Error threshold (mm) above which recovery mode is used
        (more iterations + stronger adaptive damping).
    recovery_max_iters : int
        Max iterations in recovery mode.
    rest_q : array-like | None
        Preferred joint configuration for null-space control.
        Defaults to all-zeros (mid-range attraction via clipping).
    null_gain : float
        Strength of the null-space pull toward rest_q ∈ [0, 1].
        0 = disabled, 1 = full pull.  Only active when n_arm >= 6.
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 side: str = "right",
                 damping: float = 1e-2,
                 ik_step: float = 0.8,
                 ik_max_iters: int = 3,
                 ik_err_stop_mm: float = 15.0,
                 joint_weights: "list | None" = None,
                 recovery_err_mm: float = 60.0,
                 recovery_max_iters: int = 20,
                 rest_q: "list | None" = None,
                 null_gain: float = 0.3):

        self.side = side
        self.damping = damping
        self.ik_step = ik_step
        self.ik_max_iters = max(1, int(ik_max_iters))
        self.ik_err_stop_m = float(ik_err_stop_mm) / 1000.0
        self.recovery_err_m = float(recovery_err_mm) / 1000.0
        self.recovery_max_iters = max(self.ik_max_iters, int(recovery_max_iters))
        self.null_gain = float(null_gain)

        if side == "right":
            jnt_names, act_names, ee_body = _RIGHT_JOINTS, _RIGHT_ACTS, _RIGHT_EE
        elif side == "left":
            jnt_names, act_names, ee_body = _LEFT_JOINTS, _LEFT_ACTS, _LEFT_EE
        elif side == "g1_right":
            jnt_names, act_names, ee_body = _G1_RIGHT_JOINTS, _G1_RIGHT_JOINTS, _G1_RIGHT_EE
        elif side == "g1_left":
            jnt_names, act_names, ee_body = _G1_LEFT_JOINTS, _G1_LEFT_JOINTS, _G1_LEFT_EE
        else:
            raise ValueError(f"side must be 'right'|'left'|'g1_right'|'g1_left', got {side!r}")

        self.n_arm = len(jnt_names)

        if joint_weights is not None:
            self._jnt_w = np.array(joint_weights, dtype=float)
        elif side in ("g1_right", "g1_left"):
            self._jnt_w = np.array([1.0, 1.0, 1.0, 0.25, 1.0, 1.0, 1.0])
        else:
            self._jnt_w = np.array([1.0, 1.0, 1.0, 0.25, 1.0])

        # Pre-cache diagonal weight matrix (re-used in every solve call)
        self._W = np.diag(self._jnt_w)

        self.ee_body_id = model.body(ee_body).id

        self.jnt_ids  = np.array([model.joint(jn).id  for jn in jnt_names], dtype=int)
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.jnt_ids], dtype=int)
        self.dof_adr  = np.array([model.jnt_dofadr[j]  for j in self.jnt_ids], dtype=int)
        self.jnt_range = np.array([model.jnt_range[j]  for j in self.jnt_ids])

        self.act_indices = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
             for an in act_names], dtype=int)

        # Validate: warn if any actuator was not found
        missing = [act_names[i] for i, ai in enumerate(self.act_indices) if ai < 0]
        if missing:
            print(f"[ArmIK {side}] AVERTISSEMENT actuateurs introuvables: {missing}")

        # Pre-allocated Jacobian buffer (3 × nv), reused every iter
        self.jacp = np.zeros((3, model.nv))

        # Rest pose for null-space control (clipped to joint limits)
        if rest_q is not None:
            rq = np.array(rest_q, dtype=float)
        else:
            # Default: midpoint of each joint range
            rq = 0.5 * (self.jnt_range[:, 0] + self.jnt_range[:, 1])
        self._rest_q = np.clip(rq, self.jnt_range[:, 0], self.jnt_range[:, 1])

        # Initialised to rest so clamp_after_step is stable from frame 0
        self._last_q = self._rest_q.copy()

    # ── Core solver ──────────────────────────────────────────────────────────

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData,
              target_pos: np.ndarray, target_quat: np.ndarray) -> dict:
        """Run up to N DLS iterations to move the EE toward target_pos.

        Adaptive damping: λ = λ_base × max(1, ‖err‖ / recovery_err_m).
        Null-space secondary objective (n_arm ≥ 6 only):
            dq = (JᵀJ + λW)⁻¹ (Jᵀ Δx + λ W × null_gain × (q_rest − q))

        Returns a dict with 'err_mm' (final position error) and 'deg' (joint angles).
        """
        # Single mj_forward before the loop — reused for iter 0
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[self.ee_body_id].copy()
        init_err = np.linalg.norm(target_pos - ee_pos)

        n_iters = (self.recovery_max_iters if init_err > self.recovery_err_m
                   else self.ik_max_iters)

        new_q   = data.qpos[self.qpos_adr].copy()
        pos_err = target_pos - ee_pos

        for i in range(n_iters):
            err = np.linalg.norm(pos_err)
            if err <= self.ik_err_stop_m:
                break

            # Adaptive λ: grows linearly beyond recovery threshold
            lam = self.damping * max(1.0, err / self.recovery_err_m)

            mujoco.mj_jacBody(model, data, self.jacp, None, self.ee_body_id)
            Jp = self.jacp[:, self.dof_adr]          # (3, n_arm)

            # DLS solve: (JᵀJ + λW) dq = rhs
            JtJ = Jp.T @ Jp + lam * self._W          # (n_arm, n_arm)

            # rhs = Jᵀ Δx  +  null-space pull toward rest_q
            rhs = Jp.T @ pos_err
            if self.n_arm >= 6 and self.null_gain > 0.0:
                rhs += lam * self._W @ (self.null_gain * (self._rest_q - new_q))

            dq    = np.linalg.solve(JtJ, rhs)
            new_q = np.clip(new_q + self.ik_step * dq,
                            self.jnt_range[:, 0], self.jnt_range[:, 1])

            data.qpos[self.qpos_adr] = new_q
            data.qvel[self.dof_adr]  = 0.0

            # Re-forward only if another iteration follows
            if i < n_iters - 1:
                mujoco.mj_forward(model, data)
                ee_pos  = data.xpos[self.ee_body_id].copy()
                pos_err = target_pos - ee_pos

        # One final forward for accurate error and ctrl update
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.xpos[self.ee_body_id]

        for i, act_idx in enumerate(self.act_indices):
            if act_idx >= 0:
                data.ctrl[act_idx] = new_q[i]

        self._last_q = new_q.copy()
        return {"err_mm": float(np.linalg.norm(pos_err) * 1000),
                "deg":    np.degrees(new_q)}

    # ── Post-step clamp ───────────────────────────────────────────────────────

    def clamp_after_step(self, data: mujoco.MjData) -> None:
        """Re-assert arm joints after mj_step to cancel physics drift."""
        data.qpos[self.qpos_adr] = self._last_q
        data.qvel[self.dof_adr]  = 0.0

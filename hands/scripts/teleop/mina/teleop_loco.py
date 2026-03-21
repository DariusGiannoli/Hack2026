"""
teleop_loco.py — Simulation unifiée : téléopération bras + locomotion.

Simulation MuJoCo unique (bhl_scene.xml) :
  - Policy locomotion (policy_humanoid_legs.yaml) → contrôle les jambes (joints 10-21)
  - Téléopération MediaPipe (torso-relative) → contrôle les bras (joints 0-9)
  - Position du buste (épaules) → command_velocity pour la policy (vx, vy, vyaw)

Lancer avec mjpython depuis la racine du projet :
    mjpython scripts/teleop/mina/teleop_loco.py --config configs/policy_humanoid_legs.yaml

Touches :
  A   — (re-)calibrer les deux bras
  R   — reset position + calibration
"""

import os
import sys
import time as _time
import multiprocessing as _mp

import numpy as np
import torch
import mujoco
import mujoco.viewer
import cv2

# Rend les imports locaux (vision/, robots/) disponibles depuis ce répertoire.
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)
# Packages source/ (berkeley_humanoid_lite, berkeley_humanoid_lite_lowlevel)
for _src_pkg in (
    "berkeley_humanoid_lite_lowlevel",
    "berkeley_humanoid_lite",
    "berkeley_humanoid_lite_assets",
):
    _pkg_path = os.path.join(_PROJECT_DIR, "source", _src_pkg)
    if _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

from vision.camera    import ZEDCamera
from vision.detectors import StereoHandTracker, ArmTracker
import vision.geometry as geo
from vision.smoother  import OneEuroFilter

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg

# ── Constantes tunable ────────────────────────────────────────────────────────
CAMERA_ID    = 4
STEREO_DEPTH = True
SHOW_CAMERA  = True

# One Euro Filter — position bras
# POS_MC  : coupure basse fréquence → plus bas = moins de latence, plus de bruit
# POS_BETA: adaptation vitesse    → plus haut = moins de lag sur mouvements rapides
POS_FREQ = 30.0
POS_MC   = 0.5    # était 0.8 → moins de lag
POS_BETA = 0.02   # était 0.005 → meilleure réactivité aux mouvements rapides

# Workspace legacy (TRANS_SCALE, START_Y/Z utilisés pour fallback pixel)
TRANS_SCALE  = 6.0
START_Y      = 0.30
START_Z      = 0.45
EPIPOLAR_TOL = 40
DEPTH_MIN_M  = 0.20
DEPTH_MAX_M  = 0.90
DEPTH_MID_M  = 0.45
DEPTH_SCALE  = 2.0
# Déplacement max par frame : augmenté pour ne pas brider les mouvements rapides
MOCAP_MAX_STEP = 0.035   # était 0.015

# Morphological calibration
MORPH_SCALE_MIN = 0.60
MORPH_SCALE_MAX = 1.50

# Bras — sensibilité par axe [x-latéral, y-profondeur, z-hauteur] (repère robot)
# Le z-axis MediaPipe Pose (profondeur mono) est le plus bruité → y-profondeur plus faible.
# Augmenter un axe = plus de déplacement du bras dans cette direction.
ARM_RIGHT_AXIS_GAIN = np.array([2.0, 0.9, 2.2])  # [latéral, profondeur, hauteur] — latéral augmenté
ARM_LEFT_AXIS_GAIN  = np.array([2.5, 0.9, 2.2])  # latéral augmenté pour bras gauche

# Torso-relative arm control
USE_TORSO_RELATIVE = True
_MP_SIGN_X = +1.0  # latéral : BHL right arm à X négatif → bras droite = target_x plus négatif
_MP_SIGN_Y = -1.0  # mp.y pointe vers le BAS dans MP world → inverser pour robot.z(up)
_MP_SIGN_Z = -1.0  # mp.z(toward-cam) → robot.y(fwd) : sens opposé
# Bras gauche : latéral inversé (symétrie miroir, left arm à X positif)
_LEFT_LATERAL_SIGN = -1.0

# Référence torse stable : mélange épaule + centre torse (hanches+épaules)
# 0.0 = épaule pure (ancien comportement)  1.0 = centre torse pur
# ~0.35 : compense les haussements d'épaules sans sur-compenser les penchements
TORSO_REF_BLEND = 0.35

# Handedness (ZED non-miroir)
TARGET_HAND = "Left"   # main physique droite sur ZED non-miroir
OTHER_HAND  = "Right"  # main physique gauche

# Auto-calibration
AUTO_CALIB_SEC = 10.0
HOLD_POSE_SEC  = 1.0

# ── Gains vitesse buste → command_velocity ────────────────────────────────────
# Déplacement du buste (midpoint épaules) en mètres/s (repère MP world)
# → mappé sur les commandes vitesse robot.
#   VX_GAIN  : profondeur MP (mp.z) → avant robot (vx)
#   VY_GAIN  : latéral MP (mp.x) → gauche robot (vy)
#   VYAW_GAIN: rotation épaules → lacet robot (vyaw)
# Ajuster le signe si le mouvement part dans la mauvaise direction.
VX_GAIN        = 8     # plus haut = robot parcourt plus de distance par mouvement de buste
VY_GAIN        = 7
VYAW_GAIN      = 1.5
BUST_VEL_ALPHA   = 0.85   # lissage équilibré : réactif sans exploser sur les spikes MP
BUST_VEL_MAX     = 0.6   # clip max (m/s) — plus haut = plus de distance possible
BUST_VEL_THRESH  = 0.08   # m/s — deadzone sur la sortie
BUST_VEL_DECAY   = 0.85   # décroissance si pose perdue (×/frame à 30Hz)

# Caméra viewer
_SHOW_EVERY   = 5
_VIEWER_SCALE = 0.35

# ── ArmIK inline pour le modèle BHL ──────────────────────────────────────────
# Même noms de joints que dans humanoid.xml.  Pas de write sur data.ctrl :
# les couples sont calculés par PD dans TeleopMujocoSimulator._apply_actions().

_LEFT_JOINTS_BHL = [
    "arm_left_shoulder_pitch_joint",
    "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
]
_RIGHT_JOINTS_BHL = [
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]
_LEFT_EE_BHL  = "arm_left_hand_link"
_RIGHT_EE_BHL = "arm_right_hand_link"


class _ArmIK:
    """DLS IK pour un bras, compatible avec le modèle BHL (actuateurs torque).

    Modifie data.qpos directement (comme l'ArmIKSolver de la téléop).
    Ne touche PAS à data.ctrl : les couples sont gérés par PD dans
    TeleopMujocoSimulator._apply_actions().
    Stocke self._last_q pour la phase de clamping post-step.
    """

    def __init__(self, model: mujoco.MjModel, side: str = "right",
                 damping: float = 1e-2, ik_step: float = 0.8,
                 ik_max_iters: int = 3, ik_err_stop_mm: float = 15.0,
                 recovery_err_mm: float = 60.0, recovery_max_iters: int = 20):
        self.side     = side
        self.damping  = damping
        self.ik_step  = ik_step
        self.ik_max_iters       = max(1, int(ik_max_iters))
        self.ik_err_stop_m      = ik_err_stop_mm / 1000.0
        self.recovery_err_m     = recovery_err_mm / 1000.0
        self.recovery_max_iters = max(self.ik_max_iters, int(recovery_max_iters))
        # Poids per-joint : elbow_pitch (idx 3) préféré (0.25×)
        # Poids DLS : plus bas = joint utilisé plus librement par l'IK
        # shoulder_roll (idx 1) et elbow_roll (idx 4) abaissés pour être utilisés
        # autant que shoulder_pitch et elbow_pitch → mouvements latéraux actifs
        self._jnt_w = np.array([1.0, 0.3, 1.0, 0.25, 0.3])

        jnt_names = _LEFT_JOINTS_BHL if side == "left" else _RIGHT_JOINTS_BHL
        ee_body   = _LEFT_EE_BHL     if side == "left" else _RIGHT_EE_BHL

        self.ee_body_id = model.body(ee_body).id
        self.n_arm      = len(jnt_names)

        self.jnt_ids   = np.array([model.joint(n).id           for n in jnt_names], dtype=int)
        self.qpos_adr  = np.array([model.jnt_qposadr[j]        for j in self.jnt_ids], dtype=int)
        self.dof_adr   = np.array([model.jnt_dofadr[j]         for j in self.jnt_ids], dtype=int)
        self.jnt_range = np.array([model.jnt_range[j]          for j in self.jnt_ids])

        self.jacp    = np.zeros((3, model.nv))
        self._last_q = None

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData,
              target_pos: np.ndarray) -> dict:
        """Multi-pass DLS IK. Retourne {'deg', 'err_mm'}."""
        new_q = data.qpos[self.qpos_adr].copy()

        mujoco.mj_forward(model, data)
        init_err = np.linalg.norm(target_pos - data.xpos[self.ee_body_id])

        if init_err > self.recovery_err_m:
            n_iters = self.recovery_max_iters
            damp    = self.damping * 4.0
        else:
            n_iters = self.ik_max_iters
            damp    = self.damping

        JtJ_reg = damp * np.diag(self._jnt_w)

        for _ in range(n_iters):
            mujoco.mj_forward(model, data)
            ee_pos  = data.xpos[self.ee_body_id]
            pos_err = target_pos - ee_pos
            if np.linalg.norm(pos_err) <= self.ik_err_stop_m:
                break

            mujoco.mj_jacBody(model, data, self.jacp, None, self.ee_body_id)
            Jp   = self.jacp[:, self.dof_adr]
            dq   = np.linalg.solve(Jp.T @ Jp + JtJ_reg, Jp.T @ pos_err)
            new_q = np.clip(data.qpos[self.qpos_adr] + self.ik_step * dq,
                            self.jnt_range[:, 0], self.jnt_range[:, 1])
            data.qpos[self.qpos_adr] = new_q
            data.qvel[self.dof_adr]  = 0.0

        mujoco.mj_forward(model, data)
        pos_err    = target_pos - data.xpos[self.ee_body_id]
        self._last_q = new_q.copy()
        return {"deg": np.degrees(new_q), "err_mm": float(np.linalg.norm(pos_err) * 1000)}

    def clamp_after_step(self, data: mujoco.MjData) -> None:
        """Réassert les positions des joints bras après mj_step."""
        if self._last_q is not None:
            data.qpos[self.qpos_adr] = self._last_q
            data.qvel[self.dof_adr]  = 0.0


# ── TeleopMujocoSimulator ──────────────────────────────────────────────────────

class TeleopMujocoSimulator(MujocoSimulator):
    """MujocoSimulator augmenté pour la téléopération combinée.

    Bras (joints 0-9)  : contrôlés par IK téléop (_arm_left_q, _arm_right_q).
    Jambes (joints 10-21) : contrôlées par la policy locomotion.
    command_velocity   : injecté depuis la position du buste.
    """

    # Indices dans le vecteur 22-joints (sensordata order = yaml joints order)
    _ARM_LEFT_IDX  = list(range(0, 5))   # left arm joints 0-4
    _ARM_RIGHT_IDX = list(range(5, 10))  # right arm joints 5-9

    def __init__(self, cfg: Cfg):
        # ── Symlinks pour que MujocoEnv trouve bhl_scene.xml + les STL ──────────
        # MujocoEnv utilise le chemin relatif "source/.../data/mjcf/bhl_scene.xml"
        # mais les fichiers réels sont dans ".../data/robots/.../mjcf/" et ".../meshes/"
        _bhl_base      = os.path.join(_PROJECT_DIR, "source", "berkeley_humanoid_lite_assets",
                                      "data", "robots", "berkeley_humanoid", "berkeley_humanoid_lite")
        _mjcf_actual   = os.path.join(_bhl_base, "mjcf")
        _meshes_actual = os.path.join(_bhl_base, "meshes")
        _mjcf_expected = os.path.join(_PROJECT_DIR,
            "source", "berkeley_humanoid_lite_assets", "data", "mjcf")
        if not os.path.exists(_mjcf_expected) and os.path.isdir(_mjcf_actual):
            os.symlink(_mjcf_actual, _mjcf_expected)
        # meshdir="assets" + mesh file="merged/*.stl" → needs assets/merged/ → meshes/
        _assets_dir  = os.path.join(_mjcf_actual, "assets")
        _merged_link = os.path.join(_assets_dir, "merged")
        if not os.path.exists(_assets_dir):
            os.makedirs(_assets_dir, exist_ok=True)
        if not os.path.exists(_merged_link) and os.path.isdir(_meshes_actual):
            os.symlink(_meshes_actual, _merged_link)

        # ── Initialisation manuelle (évite le double viewer de MujocoEnv) ───────
        # On réplique MujocoEnv.__init__ + MujocoSimulator.__init__ ici pour
        # créer le viewer UNE SEULE FOIS avec key_callback.
        import threading as _threading
        from berkeley_humanoid_lite_lowlevel.policy.gamepad import Se2Gamepad

        self.cfg = cfg

        # Charger le modèle (même logique que MujocoEnv.__init__)
        if cfg.num_joints == 22:
            _xml = "source/berkeley_humanoid_lite_assets/data/mjcf/bhl_scene.xml"
        else:
            _xml = "source/berkeley_humanoid_lite_assets/data/mjcf/bhl_biped_scene.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(_xml)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.physics_dt

        # Viewer unique avec key_callback (touches A=calib, R=reset)
        self.mj_viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, key_callback=_key_callback)

        # MujocoSimulator attributes
        self.physics_substeps    = int(np.round(cfg.policy_dt / cfg.physics_dt))
        self.sensordata_dof_size = 3 * self.mj_model.nu
        self.gravity_vector      = torch.tensor([0.0, 0.0, -1.0])

        self.joint_kp      = torch.tensor(cfg.joint_kp,      dtype=torch.float32)
        self.joint_kd      = torch.tensor(cfg.joint_kd,      dtype=torch.float32)
        self.effort_limits = torch.tensor(cfg.effort_limits,  dtype=torch.float32)
        self.n_steps       = 0

        self.is_killed          = _threading.Event()
        self.mode               = 3.0
        self.command_velocity_x = 0.0
        self.command_velocity_y = 0.0
        self.command_velocity_yaw = 0.0

        self.command_controller = Se2Gamepad()
        self.command_controller.run()

        # ── Attributs téléop ─────────────────────────────────────────────────────
        self._arm_left_q  = None
        self._arm_right_q = None
        self._bust_vx     = 0.0
        self._bust_vy     = 0.0
        self._bust_vyaw   = 0.0
        self._ik_left     = None
        self._ik_right    = None

    def _apply_actions(self, actions: torch.Tensor) -> None:
        target_positions = torch.zeros((self.cfg.num_joints,))
        target_positions[self.cfg.action_indices] = actions  # jambes [10-21]

        # Injecter les cibles IK des bras
        if self._arm_left_q is not None:
            target_positions[self._ARM_LEFT_IDX] = torch.tensor(
                self._arm_left_q, dtype=torch.float32)
        if self._arm_right_q is not None:
            target_positions[self._ARM_RIGHT_IDX] = torch.tensor(
                self._arm_right_q, dtype=torch.float32)

        output_torques = (self.joint_kp * (target_positions - self._get_joint_pos()) +
                          self.joint_kd * (-self._get_joint_vel()))
        output_torques_clipped = torch.clip(output_torques, -self.effort_limits, self.effort_limits)
        self.mj_data.ctrl[:] = output_torques_clipped.numpy()

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        step_start = _time.perf_counter()

        for _ in range(self.physics_substeps):
            self._apply_actions(actions)
            mujoco.mj_step(self.mj_model, self.mj_data)
            # Réassert positions bras après chaque substep (évite la dérive)
            if self._ik_left  is not None:
                self._ik_left.clamp_after_step(self.mj_data)
            if self._ik_right is not None:
                self._ik_right.clamp_after_step(self.mj_data)

        self.mj_viewer.sync()
        observations = self._get_observations()

        elapsed = _time.perf_counter() - step_start
        remaining = self.cfg.policy_dt - elapsed
        if remaining > 0:
            _time.sleep(remaining)

        self.n_steps += 1
        return observations

    def _get_observations(self) -> torch.Tensor:
        obs = super()._get_observations()
        # obs layout: [quat(4), ang_vel(3), joint_pos(12), joint_vel(12), mode(1), vx(1), vy(1), vyaw(1)]
        # Patcher les 3 derniers éléments avec la vitesse du buste.
        obs[-3] = float(self._bust_vx)
        obs[-2] = float(self._bust_vy)
        obs[-1] = float(self._bust_vyaw)
        self.command_velocity_x   = float(self._bust_vx)
        self.command_velocity_y   = float(self._bust_vy)
        self.command_velocity_yaw = float(self._bust_vyaw)
        return obs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mp_world_to_robot(delta_mp: np.ndarray) -> np.ndarray:
    """Vecteur MediaPipe Pose world → repère robot sim."""
    return np.array([
        _MP_SIGN_X * delta_mp[0],  # latéral
        _MP_SIGN_Z * delta_mp[2],  # profondeur → y robot (forward)
        _MP_SIGN_Y * delta_mp[1],  # hauteur    → z robot
    ])


def _stable_shoulder(plm, side: str) -> np.ndarray:
    """Épaule stabilisée = épaule pure corrigée par le centre du torse.

    Réduit l'effet des haussements d'épaules sur la commande des bras.
    side : 'right' (plm[12]) ou 'left' (plm[11])
    """
    sh_idx = 12 if side == "right" else 11
    sh = np.array([plm[sh_idx].x, plm[sh_idx].y, plm[sh_idx].z])
    # Centre torse = moyenne épaules (11,12) + hanches (23,24)
    torso = np.array([
        (plm[11].x + plm[12].x + plm[23].x + plm[24].x) / 4.0,
        (plm[11].y + plm[12].y + plm[23].y + plm[24].y) / 4.0,
        (plm[11].z + plm[12].z + plm[23].z + plm[24].z) / 4.0,
    ])
    # Mélange : épaule − TORSO_REF_BLEND × (épaule − centre_torse)
    # = épaule stabilisée vers le centre du torse
    return sh - TORSO_REF_BLEND * (sh - torso)


def _find_hand_by_label(result, label: str):
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None, 0.0
    for i, hand_class in enumerate(result.multi_handedness):
        cls = hand_class.classification[0]
        if cls.label == label:
            return result.multi_hand_landmarks[i], cls.score
    return None, 0.0


def _dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pose_morphology(pose_result):
    if pose_result is None or pose_result.pose_world_landmarks is None:
        return None
    lm = pose_result.pose_world_landmarks.landmark

    def p(i):
        return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)

    l_sh, l_el, l_wr = p(11), p(13), p(15)
    r_sh, r_el, r_wr = p(12), p(14), p(16)
    vals = np.array([_dist3(l_sh, l_el), _dist3(l_el, l_wr),
                     _dist3(r_sh, r_el), _dist3(r_el, r_wr), _dist3(l_sh, r_sh)])
    if np.any(vals < 0.05) or np.any(vals > 0.80):
        return None
    return {
        "left_reach_h":  vals[0] + vals[1],
        "right_reach_h": vals[2] + vals[3],
        "shoulder_w_h":  vals[4],
    }


# ── État global de la téléop ──────────────────────────────────────────────────

_calibrate_flag  = False
_left_calib_flag = False
_reset_flag      = None

_wrist_ref_angle = None
_pitch_ref_angle = None
_yaw_ref_angle   = None
_last_hand_time  = 0.0

_right_ref_pos    = None
_right_ee_start   = None
_right_sh_start   = None   # position épaule droite au moment de la calibration (repère monde)
_last_right_target = None  # dernière cible IK bras droit (pour MOCAP_MAX_STEP)
_left_ref_pos     = None
_left_ee_start    = None
_left_sh_start    = None   # position épaule gauche au moment de la calibration (repère monde)
_last_left_target = None
_left_mono_ref_span = None

_mp_right_hand_rel_ref  = None
_mp_left_hand_rel_ref   = None
_mp_left_smoothed_rel   = None   # signal capteur bras gauche filtré + clampé (pipeline copy.py)

_arm_scale_left  = 1.0
_arm_scale_right = 1.0
_robot_reach_left  = None
_robot_reach_right = None
_robot_shoulder_w  = None

# Buste velocity state
_prev_bust_pos       = None
_prev_shoulder_yaw   = None
_bust_vx_smooth      = 0.0
_bust_vy_smooth      = 0.0
_bust_vyaw_smooth    = 0.0

_start_time   = None
_frame_q      = None
_show_counter = 0


def _show(frame):
    global _show_counter
    if not SHOW_CAMERA or _frame_q is None:
        return
    _show_counter += 1
    if _show_counter % _SHOW_EVERY != 0:
        return
    small = cv2.resize(frame, None, fx=_VIEWER_SCALE, fy=_VIEWER_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    if _frame_q.full():
        try:
            _frame_q.get_nowait()
        except Exception:
            pass
    try:
        _frame_q.put_nowait(small)
    except Exception:
        pass


def _init_arms(model: mujoco.MjModel, data: mujoco.MjData,
               ik_right: _ArmIK, ik_left: _ArmIK) -> None:
    """Positionne les deux bras à la pose de repos courbée dans bhl_scene."""
    sp_jid = model.joint("arm_right_shoulder_pitch_joint").id
    ep_jid = model.joint("arm_right_elbow_pitch_joint").id
    data.qpos[model.jnt_qposadr[sp_jid]] =  np.pi / 2
    data.qpos[model.jnt_qposadr[ep_jid]] = -np.pi / 4

    sp_jid = model.joint("arm_left_shoulder_pitch_joint").id
    ep_jid = model.joint("arm_left_elbow_pitch_joint").id
    data.qpos[model.jnt_qposadr[sp_jid]] = -np.pi / 2
    data.qpos[model.jnt_qposadr[ep_jid]] =  np.pi / 4

    mujoco.mj_forward(model, data)

    # Initialiser _last_q pour clamp_after_step
    ik_right._last_q = data.qpos[ik_right.qpos_adr].copy()
    ik_left._last_q  = data.qpos[ik_left.qpos_adr].copy()


# ── Mise à jour combinée téléop + vitesse buste ───────────────────────────────

def _update(robot:        TeleopMujocoSimulator,
            zed:          ZEDCamera,
            tracker:      StereoHandTracker,
            pose_tracker: ArmTracker,
            pos_f:        OneEuroFilter,
            left_pos_f:   OneEuroFilter,
            right_torso_f: 'OneEuroFilter | None' = None) -> None:
    """
    Capture une frame, calcule IK bras + vitesse buste, écrit dans robot.
    Appelé une fois par itération de boucle, avant robot.step().
    """
    global _calibrate_flag, _left_calib_flag, _reset_flag
    global _wrist_ref_angle, _pitch_ref_angle, _yaw_ref_angle, _last_hand_time
    global _right_ref_pos, _right_ee_start, _right_sh_start, _last_right_target
    global _left_ref_pos, _left_ee_start, _left_sh_start, _last_left_target, _left_mono_ref_span
    global _mp_right_hand_rel_ref, _mp_left_hand_rel_ref, _mp_left_smoothed_rel
    global _arm_scale_left, _arm_scale_right
    global _robot_reach_left, _robot_reach_right, _robot_shoulder_w
    global _prev_bust_pos, _prev_shoulder_yaw
    global _bust_vx_smooth, _bust_vy_smooth, _bust_vyaw_smooth

    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    data = robot.mj_data
    model = robot.mj_model
    ik_right = robot._ik_right
    ik_left  = robot._ik_left

    # ── Reset (touche R) ──────────────────────────────────────────────────
    if _reset_flag is not None and _reset_flag.value:
        _reset_flag.value = 0
        _wrist_ref_angle = None
        _pitch_ref_angle = None
        _yaw_ref_angle   = None
        pos_f.reset()
        if left_pos_f is not None:
            left_pos_f.reset()
        if right_torso_f is not None:
            right_torso_f.reset()
        _right_ref_pos = None; _right_ee_start = None; _right_sh_start = None; _last_right_target = None
        _left_ref_pos  = None; _left_ee_start  = None; _left_sh_start  = None
        _last_left_target = None; _left_mono_ref_span = None
        _mp_right_hand_rel_ref = None
        _mp_left_hand_rel_ref  = None
        _mp_left_smoothed_rel  = None
        _prev_bust_pos = None; _prev_shoulder_yaw = None
        _bust_vx_smooth = _bust_vy_smooth = _bust_vyaw_smooth = 0.0
        robot._bust_vx  = robot._bust_vy  = robot._bust_vyaw  = 0.0
        _init_arms(model, data, ik_right, ik_left)
        # Réinitialiser les cibles bras sur la pose de repos courbée
        robot._arm_right_q = ik_right._last_q.copy()
        robot._arm_left_q  = ik_left._last_q.copy()

    h, w, _ = frame_l.shape
    res_l, res_r = tracker.process(frame_l, frame_r)
    pose_res = pose_tracker.process(frame_l) if pose_tracker is not None else None

    now = _time.monotonic()

    # ── Vitesse buste depuis épaules ─────────────────────────────────────
    if pose_res is not None and pose_res.pose_world_landmarks is not None:
        plm = pose_res.pose_world_landmarks.landmark
        lsh  = np.array([plm[11].x, plm[11].y, plm[11].z])
        rsh  = np.array([plm[12].x, plm[12].y, plm[12].z])
        lhip = np.array([plm[23].x, plm[23].y, plm[23].z])
        rhip = np.array([plm[24].x, plm[24].y, plm[24].z])
        # Moyenne épaules + hanches : les hanches sont plus stables lors des mouvements de bras
        # → réduit le faux signal de profondeur (z) quand on monte/descend les mains
        bust_pos = (lsh + rsh + lhip + rhip) / 4.0

        # Yaw épaules : angle dans le plan horizontal (x-z de MP world)
        shoulder_vec = rsh - lsh
        shoulder_yaw = float(np.arctan2(shoulder_vec[2], shoulder_vec[0]))

        if _prev_bust_pos is not None:
            dt = 1.0 / 30.0

            disp    = bust_pos - _prev_bust_pos
            raw_vel = disp / dt

            # Seul axe actif : profondeur Z MediaPipe → vx robot (avancer/reculer)
            # Se rapprocher de la caméra (z diminue) → vx positif (robot avance)
            raw_vx = VX_GAIN * (-raw_vel[2])

            _bust_vx_smooth = BUST_VEL_ALPHA * _bust_vx_smooth + (1 - BUST_VEL_ALPHA) * raw_vx

            vx_out = float(_bust_vx_smooth) if abs(_bust_vx_smooth) > BUST_VEL_THRESH else 0.0
            robot._bust_vx   = float(np.clip(vx_out, -BUST_VEL_MAX, BUST_VEL_MAX))
            robot._bust_vy   = 0.0
            robot._bust_vyaw = 0.0

            print(f"CMD  vx={robot._bust_vx:+.3f}  (raw_vz={raw_vel[2]:+.3f} m/s)")

        _prev_bust_pos     = bust_pos.copy()
        _prev_shoulder_yaw = shoulder_yaw
    else:
        # Pose non détectée → décroissance progressive vers 0 (évite que le robot parte en ligne droite)
        _bust_vx_smooth *= BUST_VEL_DECAY
        robot._bust_vx   = float(_bust_vx_smooth) if abs(_bust_vx_smooth) > BUST_VEL_THRESH else 0.0
        robot._bust_vy   = 0.0
        robot._bust_vyaw = 0.0
        _prev_bust_pos   = None   # invalide la référence pour éviter un spike au retour de la pose

    # ── Trouver les mains ────────────────────────────────────────────────
    lm_target, target_score = _find_hand_by_label(res_l, TARGET_HAND)
    lm_other,  other_score  = _find_hand_by_label(res_l, OTHER_HAND)
    lm_target_r, _ = _find_hand_by_label(res_r, TARGET_HAND)
    lm_other_r,  _ = _find_hand_by_label(res_r, OTHER_HAND)

    elapsed_since_start = now - _start_time if _start_time else 0

    # ── Main droite absente → maintien pose + arrêt progressif du robot ─
    if lm_target is None:
        # Décroissance vitesse même quand la main n'est pas visible
        _bust_vx_smooth *= BUST_VEL_DECAY
        robot._bust_vx   = float(_bust_vx_smooth) if abs(_bust_vx_smooth) > BUST_VEL_THRESH else 0.0
        robot._bust_vy   = 0.0
        robot._bust_vyaw = 0.0

        elapsed = now - _last_hand_time
        holding = elapsed < HOLD_POSE_SEC and _last_hand_time > 0
        if SHOW_CAMERA:
            hold_label = f"HOLD {HOLD_POSE_SEC - elapsed:.1f}s" if holding else "NO HAND"
            hold_col   = (0, 200, 255) if holding else (0, 0, 255)
            cv2.putText(frame_l, f"R.hand: {hold_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hold_col, 2)
            if _wrist_ref_angle is None:
                remaining = max(0, AUTO_CALIB_SEC - elapsed_since_start)
                cv2.putText(frame_l, f"CALIBRATION DANS {remaining:.1f}s", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            tracker.draw_landmarks(frame_l, res_l)
        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            _show(np.hstack([frame_l, frame_r]))
        else:
            _show(frame_l)
        return

    _last_hand_time = now
    lm_l = lm_target.landmark
    cam  = geo.ZED2I

    # ── Position mocap main droite (projection pixel) ─────────────────
    _palm_ids = (0, 5, 9, 13, 17)
    u_w = sum(lm_l[i].x for i in _palm_ids) / len(_palm_ids) * w
    v_w = sum(lm_l[i].y for i in _palm_ids) / len(_palm_ids) * h
    sim_x = -(u_w - cam.cx) / cam.fx * START_Y * TRANS_SCALE
    sim_z =  START_Z + (-(v_w - cam.cy) / cam.fy * START_Y) * TRANS_SCALE
    sim_y = START_Y

    pc = zed.get_point_cloud()
    if pc is not None:
        try:
            import pyzed.sl as sl
            px_pc = int(np.clip(u_w, 0, w - 1))
            py_pc = int(np.clip(v_w, 0, h - 1))
            err_pc, pc_val = pc.get_value(px_pc, py_pc)
            z_m = float(pc_val[2])
            if (err_pc == sl.ERROR_CODE.SUCCESS and
                    not np.isnan(z_m) and
                    DEPTH_MIN_M < z_m < DEPTH_MAX_M):
                x_m = float(pc_val[0])
                y_m = float(pc_val[1])
                sim_x = -x_m * TRANS_SCALE
                sim_y = START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE * TRANS_SCALE
                sim_z = START_Z + (-y_m) * TRANS_SCALE
        except Exception:
            pass

    # ── Auto-calibration ─────────────────────────────────────────────────
    auto_trigger = (_wrist_ref_angle is None and elapsed_since_start >= AUTO_CALIB_SEC)
    if _calibrate_flag or auto_trigger:
        # Angles de poignet de référence (pour l'affichage uniquement ici)
        idx_mcp = lm_l[5]; ring_mcp = lm_l[13]
        _wrist_ref_angle = float(np.arctan2(ring_mcp.y - idx_mcp.y,
                                            ring_mcp.x - idx_mcp.x))
        mid_mcp = lm_l[9]; wrist_lm = lm_l[0]
        _pitch_ref_angle = float(np.arctan2(mid_mcp.z - wrist_lm.z,
                                            mid_mcp.y - wrist_lm.y))
        pky_mcp = lm_l[17]; idx_mcp_y = lm_l[5]
        _yaw_ref_angle   = float((pky_mcp.z - idx_mcp_y.z) / max(abs(pky_mcp.x - idx_mcp_y.x), 0.01))
        pos_f.reset()
        _calibrate_flag = False
        _left_calib_flag = True
        _right_ref_pos  = np.array([sim_x, sim_y, sim_z], dtype=float)
        mujoco.mj_forward(model, data)
        _right_ee_start = data.xpos[ik_right.ee_body_id].copy()
        _right_sh_start = data.xpos[model.body("arm_right_shoulder_pitch").id].copy()
        _last_right_target    = None   # reset pour éviter spike au premier frame
        _mp_right_hand_rel_ref = None  # capture au prochain frame (sur signal filtré)
        _mp_left_hand_rel_ref  = None
        _mp_left_smoothed_rel  = None
        if right_torso_f is not None:
            right_torso_f.reset()

        morph = _pose_morphology(pose_res)
        if (morph is not None and _robot_reach_left is not None
                and _robot_reach_right is not None):
            _arm_scale_left  = float(np.clip(
                _robot_reach_left  / max(morph["left_reach_h"],  1e-6),
                MORPH_SCALE_MIN, MORPH_SCALE_MAX))
            _arm_scale_right = float(np.clip(
                _robot_reach_right / max(morph["right_reach_h"], 1e-6),
                MORPH_SCALE_MIN, MORPH_SCALE_MAX))
        else:
            _arm_scale_left = _arm_scale_right = 1.0

    if _wrist_ref_angle is None:
        if frame_r is not None:
            _show(np.hstack([frame_l, frame_r]))
        else:
            _show(frame_l)
        return

    # ── IK bras droit ─────────────────────────────────────────────────────
    if ik_right is not None:
        arm_target_pos = None
        if (USE_TORSO_RELATIVE
                and pose_res is not None
                and pose_res.pose_world_landmarks is not None
                and _right_ee_start is not None):
            plm = pose_res.pose_world_landmarks.landmark
            rsh = _stable_shoulder(plm, "right")          # épaule stabilisée (hanche+épaule)
            rwr = np.array([plm[16].x, plm[16].y, plm[16].z])
            hand_rel = _mp_world_to_robot(rwr - rsh)
            # Filtrage en espace capteur (pipeline copy.py)
            hand_rel_f = right_torso_f(hand_rel) if right_torso_f is not None else hand_rel
            if _mp_right_hand_rel_ref is None:
                _mp_right_hand_rel_ref = hand_rel_f.copy()
            delta_torso   = (hand_rel_f - _mp_right_hand_rel_ref) * _arm_scale_right * ARM_RIGHT_AXIS_GAIN
            # Ancre sur l'épaule courante (compense le déplacement du robot lors de la marche)
            cur_r_sh      = data.xpos[model.body("arm_right_shoulder_pitch").id].copy()
            ee_offset     = _right_ee_start - _right_sh_start
            arm_target_pos = cur_r_sh + ee_offset + delta_torso
            print(f"[HORIZ] hand_rel[0]={hand_rel[0]:+.3f} delta_lat={delta_torso[0]:+.3f} target_x={arm_target_pos[0]:+.3f}")
        elif _right_ref_pos is not None and _right_ee_start is not None:
            right_delta   = np.array([sim_x, sim_y, sim_z]) - _right_ref_pos
            arm_target_pos = _right_ee_start + right_delta * _arm_scale_right * ARM_RIGHT_AXIS_GAIN

        if arm_target_pos is not None:
            # Limite de déplacement max par frame (évite les téléportations sur jitter MediaPipe)
            if _last_right_target is not None:
                delta_rt = arm_target_pos - _last_right_target
                dist_rt  = np.linalg.norm(delta_rt)
                if dist_rt > MOCAP_MAX_STEP:
                    arm_target_pos = _last_right_target + delta_rt * (MOCAP_MAX_STEP / dist_rt)
            _last_right_target = arm_target_pos.copy()
            ik_info = ik_right.solve(model, data, arm_target_pos)
            robot._arm_right_q = ik_right._last_q.copy()

    # ── IK bras gauche ────────────────────────────────────────────────────
    if ik_left is not None:
        if (USE_TORSO_RELATIVE
                and pose_res is not None
                and pose_res.pose_world_landmarks is not None):
            plm = pose_res.pose_world_landmarks.landmark
            lsh = _stable_shoulder(plm, "left")           # épaule stabilisée (hanche+épaule)
            lwr = np.array([plm[15].x, plm[15].y, plm[15].z])
            hand_rel_l = _mp_world_to_robot(lwr - lsh)

            # 1. Filtrage en espace capteur (pipeline copy.py)
            hand_rel_l_f = left_pos_f(hand_rel_l) if left_pos_f is not None else hand_rel_l.copy()

            # 2. MOCAP_MAX_STEP en espace capteur sur le signal filtré
            if _mp_left_smoothed_rel is not None:
                d_sens    = hand_rel_l_f - _mp_left_smoothed_rel
                dist_sens = np.linalg.norm(d_sens)
                if dist_sens > MOCAP_MAX_STEP:
                    hand_rel_l_f = _mp_left_smoothed_rel + d_sens * (MOCAP_MAX_STEP / dist_sens)
            _mp_left_smoothed_rel = hand_rel_l_f.copy()

            if _left_calib_flag:
                _mp_left_hand_rel_ref = hand_rel_l_f.copy()   # référence sur signal filtré
                mujoco.mj_forward(model, data)
                _left_ee_start = data.xpos[ik_left.ee_body_id].copy()
                _left_sh_start = data.xpos[model.body("arm_left_shoulder_pitch").id].copy()
                _last_left_target = None
                if left_pos_f is not None:
                    left_pos_f.reset()
                _left_calib_flag = False

            if _mp_left_hand_rel_ref is not None and _left_ee_start is not None:
                delta_l   = (_mp_left_smoothed_rel - _mp_left_hand_rel_ref) * _arm_scale_left * ARM_LEFT_AXIS_GAIN
                delta_l[0] *= _LEFT_LATERAL_SIGN  # latéral inversé pour bras gauche (miroir)
                # Ancre sur l'épaule courante (compense le déplacement du robot lors de la marche)
                cur_l_sh  = data.xpos[model.body("arm_left_shoulder_pitch").id]
                ee_offset_l = _left_ee_start - _left_sh_start
                target_lh = cur_l_sh + ee_offset_l + delta_l
                print(f"[HORIZ L] hand_rel[0]={hand_rel_l[0]:+.3f} delta_lat={delta_l[0]:+.3f} target_x={target_lh[0]:+.3f}")
                _last_left_target = target_lh.copy()
                ik_left.solve(model, data, target_lh)
                robot._arm_left_q = ik_left._last_q.copy()

        elif lm_other is not None:
            # Fallback pixel-based pour le bras gauche
            lm_left = lm_other.landmark
            u_lh = sum(lm_left[i].x for i in _palm_ids) / len(_palm_ids) * w
            v_lh = sum(lm_left[i].y for i in _palm_ids) / len(_palm_ids) * h
            lh_x = (u_lh - cam.cx) / cam.fx * START_Y * TRANS_SCALE
            lh_z = START_Z + (-(v_lh - cam.cy) / cam.fy * START_Y) * TRANS_SCALE
            lh_y = START_Y

            lh_span = np.hypot(lm_left[9].x * w - lm_left[0].x * w,
                               lm_left[9].y * h - lm_left[0].y * h)
            if _left_calib_flag:
                _left_ref_pos = np.array([lh_x, lh_y, lh_z])
                _left_mono_ref_span = lh_span if lh_span > 10 else None
                mujoco.mj_forward(model, data)
                _left_ee_start = data.xpos[ik_left.ee_body_id].copy()
                _last_left_target = None
                if left_pos_f is not None:
                    left_pos_f.reset()
                _left_calib_flag = False

            if _left_ref_pos is not None and _left_ee_start is not None:
                delta_lh   = np.array([lh_x, lh_y, lh_z]) - _left_ref_pos
                target_lh  = _left_ee_start + delta_lh * _arm_scale_left * ARM_LEFT_GAIN
                if left_pos_f is not None:
                    target_lh = left_pos_f(target_lh)
                if _last_left_target is not None:
                    delta_lt = target_lh - _last_left_target
                    dist_lt  = np.linalg.norm(delta_lt)
                    if dist_lt > MOCAP_MAX_STEP:
                        target_lh = _last_left_target + delta_lt * (MOCAP_MAX_STEP / dist_lt)
                _last_left_target = target_lh.copy()
                ik_left.solve(model, data, target_lh)
                robot._arm_left_q = ik_left._last_q.copy()

    # ── HUD caméra ───────────────────────────────────────────────────────
    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, res_l)
        if pose_tracker is not None and pose_res is not None:
            pose_tracker.draw_landmarks(frame_l, pose_res)

        rh_col = (0, 220, 0) if lm_target is not None else (0, 0, 255)
        cv2.putText(frame_l, f"R.hand: OK ({target_score:.0%})" if lm_target else "R.hand: ---",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rh_col, 2)
        lh_col = (0, 220, 0) if lm_other is not None else (100, 100, 100)
        cv2.putText(frame_l, f"L.hand: OK ({other_score:.0%})" if lm_other else "L.hand: ---",
                    (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lh_col, 2)

        morph_txt = f"MORPH L={_arm_scale_left:.2f} R={_arm_scale_right:.2f}"
        cv2.putText(frame_l, morph_txt, (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 180, 0), 2)

        # Vitesse buste
        vel_txt = f"vx={robot._bust_vx:+.2f}  vy={robot._bust_vy:+.2f}  vyaw={robot._bust_vyaw:+.2f}"
        cv2.putText(frame_l, vel_txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)

        if _wrist_ref_angle is None:
            remaining = max(0, AUTO_CALIB_SEC - elapsed_since_start)
            cv2.putText(frame_l, f"CALIBRATION DANS {remaining:.1f}s", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Afficher les deux côtés ZED côte à côte (gauche + droite)
        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            display = np.hstack([frame_l, frame_r])
        else:
            display = frame_l
        _show(display)
    else:
        _show(frame_l)


# ── Key callback ─────────────────────────────────────────────────────────────

def _key_callback(keycode):
    global _calibrate_flag, _left_calib_flag
    if keycode == 65:   # A
        _calibrate_flag = True
        _left_calib_flag = True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _frame_q, _start_time, _reset_flag
    global _robot_reach_left, _robot_reach_right, _robot_shoulder_w

    # ── Config ─────────────────────────────────────────────────────────────
    cfg = Cfg.from_arguments()
    if not cfg:
        raise ValueError("Impossible de charger la configuration. Passer --config configs/policy_humanoid_legs.yaml")

    # ── Hardware caméra ─────────────────────────────────────────────────────
    zed          = ZEDCamera(camera_id=CAMERA_ID,
                             y_offset=geo.Y_OFFSET_PX if STEREO_DEPTH else 0)
    tracker      = StereoHandTracker()
    pose_tracker = ArmTracker()

    # ── Simulateur MuJoCo (policy + bras) ──────────────────────────────────
    robot = TeleopMujocoSimulator(cfg)
    obs   = robot.reset()

    # ── IK solvers pour les bras (modèle BHL) ──────────────────────────────
    ik_right = _ArmIK(robot.mj_model, side="right",
                      ik_max_iters=7, recovery_max_iters=25,
                      ik_err_stop_mm=8.0, damping=5e-3)
    ik_left  = _ArmIK(robot.mj_model, side="left",
                      ik_max_iters=7, recovery_max_iters=25,
                      ik_err_stop_mm=8.0, damping=5e-3)
    robot._ik_right = ik_right
    robot._ik_left  = ik_left

    # ── Filters ─────────────────────────────────────────────────────────────
    pos_f         = OneEuroFilter(POS_FREQ, min_cutoff=POS_MC, beta=POS_BETA)
    left_pos_f    = OneEuroFilter(POS_FREQ, min_cutoff=POS_MC, beta=POS_BETA)
    right_torso_f = OneEuroFilter(POS_FREQ, min_cutoff=POS_MC, beta=POS_BETA)

    # ── Init pose bras ───────────────────────────────────────────────────────
    _init_arms(robot.mj_model, robot.mj_data, ik_right, ik_left)
    # Initialiser les cibles bras sur la pose courbée dès le départ
    robot._arm_right_q = ik_right._last_q.copy()
    robot._arm_left_q  = ik_left._last_q.copy()

    # ── Morphologie robot à la pose neutre ──────────────────────────────────
    mujoco.mj_forward(robot.mj_model, robot.mj_data)
    r_sh = robot.mj_data.xpos[robot.mj_model.body("arm_right_shoulder_pitch").id].copy()
    l_sh = robot.mj_data.xpos[robot.mj_model.body("arm_left_shoulder_pitch").id].copy()
    r_ee = robot.mj_data.xpos[ik_right.ee_body_id].copy()
    l_ee = robot.mj_data.xpos[ik_left.ee_body_id].copy()
    _robot_reach_right = float(np.linalg.norm(r_ee - r_sh))
    _robot_reach_left  = float(np.linalg.norm(l_ee - l_sh))
    _robot_shoulder_w  = float(np.linalg.norm(r_sh - l_sh))

    # ── Policy locomotion ────────────────────────────────────────────────────
    controller = RlController(cfg)
    controller.load_policy()
    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    # ── Viewer caméra (subprocess séparé, évite conflit Cocoa/mjpython) ─────
    viewer_proc = None
    _frame_q = None
    if SHOW_CAMERA:
        from _camera_viewer import viewer_loop
        ctx = _mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=2)
        _reset_flag = ctx.Value('i', 0)
        viewer_proc = ctx.Process(target=viewer_loop, args=(_frame_q, _reset_flag), daemon=True)
        viewer_proc.start()

    _start_time = _time.monotonic()

    # ── Boucle principale ────────────────────────────────────────────────────
    try:
        while robot.mj_viewer.is_running():
            # 1. Téléop : camera → MediaPipe → IK bras + vitesse buste
            _update(robot, zed, tracker, pose_tracker, pos_f, left_pos_f, right_torso_f)

            # 2. Policy locomotion → actions jambes
            actions = controller.update(obs.numpy())
            if actions is None:
                actions = default_actions
            actions = torch.tensor(actions)

            # 3. Step physique (PD bras + policy jambes)
            obs = robot.step(actions)

    finally:
        # ── Arrêt propre ─────────────────────────────────────────────────────
        if SHOW_CAMERA and viewer_proc is not None and _frame_q is not None:
            try:
                _frame_q.put(None)
                viewer_proc.join(timeout=3)
            except Exception:
                pass

        try:
            zed.close()
        except Exception:
            pass
        try:
            pose_tracker.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # Force la terminaison du process (threads caméra/MediaPipe restent vivants sinon)
        os._exit(0)


if __name__ == "__main__":
    main()

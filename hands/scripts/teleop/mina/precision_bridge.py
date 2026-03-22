"""
precision_bridge.py — Direct motor control: GR00T (local ONNX) for legs + teleop IK for arms.

Instead of routing arm targets through the C++ deploy's planner topic (where the
whole-body policy overrides them), this module:

  1. Runs GR00T Balance/Walk ONNX policies locally for lower body (15 joints)
  2. Takes arm joint targets directly from teleop IK (14 joints)
  3. Publishes a merged LowCmd on rt/lowcmd at 50Hz via unitree_sdk2py

The C++ deploy is NOT needed. GR00T only controls legs. Arms are yours.

Usage (unchanged from teleop_edgard_new_setup.py):
    from precision_bridge import PrecisionBridge

    bridge = PrecisionBridge(interface="enx...")
    # ... in main loop after G1 IK + hand retargeting:
    bridge.send(g1_right_arm_ik, g1_left_arm_ik, right_fingers_12, left_fingers_12)
"""

import os
import sys
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# ── Path setup (must match gear_sonic zmq_planner_sender.py on disk) ─────────
# Wire format: b"command"|b"planner" + 1280-byte JSON header (null-padded) + payload.
# See gear_sonic/utils/teleop/zmq/zmq_planner_sender.py — HEADER_SIZE = 1280.
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.normpath(os.path.join(_DIR, "..", "..", "..", ".."))
_SENDER_REL = os.path.join(
    "gear_sonic", "utils", "teleop", "zmq", "zmq_planner_sender.py"
)


def _resolve_gr00t_wbc_root() -> str:
    """GR00T-WholeBodyControl repo root so `gear_sonic` imports work."""
    env = os.environ.get("GR00T_WBC_ROOT", "").strip()
    if env:
        p = os.path.abspath(env)
        if os.path.isfile(os.path.join(p, _SENDER_REL)):
            return p
        raise ImportError(
            f"GR00T_WBC_ROOT={env!r} does not contain {_SENDER_REL}. "
            "Set it to the top of the GR00T-WholeBodyControl clone."
        )
    candidates = [
        os.path.join(_PROJECT, "external", "GR00T-WholeBodyControl"),
        os.path.join(os.path.dirname(_PROJECT), "GR00T-WholeBodyControl"),
        os.path.join(os.path.dirname(os.path.dirname(_PROJECT)), "GR00T-WholeBodyControl"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(os.path.join(c, _SENDER_REL)):
            return c
    tried = "\n  ".join(candidates)
    raise ImportError(
        "Cannot find zmq_planner_sender.py. Tried:\n  "
        + tried
        + "\nSet GR00T_WBC_ROOT to your GR00T-WholeBodyControl directory."
    )


_GROOT = _resolve_gr00t_wbc_root()
if _GROOT not in sys.path:
    sys.path.insert(0, _GROOT)

# Use the full unitree_sdk2py bundled with GR00T (has comm/motion_switcher)
# instead of the stripped-down one from conda.
_GROOT_SDK = os.path.join(_GROOT, "external_dependencies", "unitree_sdk2_python")
if _GROOT_SDK not in sys.path:
    sys.path.insert(0, _GROOT_SDK)
    # Force re-import if the conda version was already loaded
    if "unitree_sdk2py" in sys.modules:
        del sys.modules["unitree_sdk2py"]

import onnxruntime as ort
from huggingface_hub import hf_hub_download
import cyclonedds.idl as idl
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as LowCmd,
    LowState_ as LowState,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

logger = logging.getLogger(__name__)

# ── GR00T locomotion constants (from LeRobot gr00t_locomotion.py) ────────────
GROOT_DEFAULT_ANGLES = np.zeros(29, dtype=np.float32)
GROOT_DEFAULT_ANGLES[[0, 6]] = -0.1    # Hip pitch
GROOT_DEFAULT_ANGLES[[3, 9]] = 0.3     # Knee
GROOT_DEFAULT_ANGLES[[4, 10]] = -0.2   # Ankle pitch

ACTION_SCALE = 0.25
CONTROL_DT = 0.02   # 50 Hz
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

GROOT_REPO_ID = "nepyope/GR00T-WholeBodyControl_g1"

# ── Motor gains (from LeRobot config_unitree_g1.py) ─────────────────────────
MOTOR_KP = np.array([
    150, 150, 150, 300, 40, 40,        # left leg
    150, 150, 150, 300, 40, 40,        # right leg
    250, 250, 250,                      # waist
    100, 100, 40, 40, 20, 20, 20,      # left arm
    100, 100, 40, 40, 20, 20, 20,      # right arm
], dtype=np.float32)

MOTOR_KD = np.array([
    2, 2, 2, 4, 2, 2,                  # left leg
    2, 2, 2, 4, 2, 2,                  # right leg
    5, 5, 5,                            # waist
    5, 5, 2, 2, 2, 2, 2,              # left arm
    5, 5, 2, 2, 2, 2, 2,              # right arm
], dtype=np.float32)

# Max arm joint step per control cycle (rad) — hard rate limiter.
# 0.04 rad/step × 50 Hz = 2 rad/s max arm speed. Prevents command jumps.
ARM_MAX_STEP = 0.04

MODE_PR = 0

# ── Inspire hand: 12 retargeted joints → 6 normalized motor values ───────────
_TH_YAW_MAX = 1.308
_TH_PITCH_MAX = 0.6
_TH_INT_MAX = 0.8
_TH_DIS_MAX = 0.4
_MCP_MAX = 1.47
_INT_MAX = 1.56

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


def retarget_12_to_6(q12: np.ndarray) -> list[float]:
    """Convert 12 Inspire retargeting angles (rad) -> 6 normalized [0,1] values.

    Output order matches hand_teleop_bridge.py FINGER_TOPICS:
      [pinky, ring, middle, index, thumb_bend, thumb_rot]
    Where 0.0 = open, 1.0 = closed.
    """
    return [
        float(np.clip((q12[10] / _MCP_MAX + q12[11] / _INT_MAX) / 2, 0, 1)),  # pinky
        float(np.clip((q12[8] / _MCP_MAX + q12[9] / _INT_MAX) / 2, 0, 1)),    # ring
        float(np.clip((q12[6] / _MCP_MAX + q12[7] / _INT_MAX) / 2, 0, 1)),    # middle
        float(np.clip((q12[4] / _MCP_MAX + q12[5] / _INT_MAX) / 2, 0, 1)),    # index
        float(np.clip(
            (q12[1] / _TH_PITCH_MAX + q12[2] / _TH_INT_MAX + q12[3] / _TH_DIS_MAX) / 3,
            0, 1,
        )),  # thumb_bend
        float(np.clip(q12[0] / _TH_YAW_MAX, 0, 1)),  # thumb_rot
    ]


# ── DDS FloatMsg (matches hand_teleop_bridge.py) ────────────────────────────
@dataclass
class FloatMsg(idl.IdlStruct, typename="FloatMsg"):
    value: float


def _get_gravity_orientation(quat) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quat
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class PrecisionBridge:
    """Direct motor control: GR00T ONNX (legs) + teleop IK (arms) -> rt/lowcmd.

    Runs GR00T Balance/Walk policies locally in a 50 Hz background thread for
    lower body (joints 0-14), merges with arm targets from teleop IK (joints
    15-28), and publishes LowCmd directly via DDS.

    No C++ deploy needed.

    Parameters
    ----------
    interface : str | None
        Network interface for DDS. None = default (loopback for sim).
    init_dds : bool
        If True, call ChannelFactoryInitialize. Set False if the caller
        already initialised DDS.
    """

    def __init__(
        self,
        interface: str | None = None,
        init_dds: bool = True,
    ):
        # ── DDS ───────────────────────────────────────────────────────────
        if init_dds:
            if interface:
                ChannelFactoryInitialize(0, interface)
            else:
                ChannelFactoryInitialize(0)

        # ── Release any active motion services (ai sport client, etc.) ──
        # Without this, the robot's internal controller fights our commands
        # and causes arm vibration. This is what GR00T's server does on
        # startup (see LeRobot run_g1_server.py).
        print("[BRIDGE] Releasing active motion services...")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        retries = 0
        while result is not None and "name" in result and result["name"] and retries < 10:
            print(f"[BRIDGE] Releasing active mode: {result['name']}")
            msc.ReleaseMode()
            time.sleep(1.0)
            status, result = msc.CheckMode()
            retries += 1
        if retries > 0:
            print(f"[BRIDGE] Released {retries} active mode(s)")
        else:
            print("[BRIDGE] No active motion services found")

        # LowCmd publisher — all 29 joints on rt/lowcmd
        self._lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd)
        self._lowcmd_pub.Init()
        self._lowcmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._lowcmd_msg.mode_pr = MODE_PR
        self._crc = CRC()

        # LowState subscriber (IMU + joint feedback)
        self._state_sub = ChannelSubscriber("rt/lowstate", LowState)
        self._state_sub.Init(None, 0)

        # Per-finger DDS publishers (unchanged)
        self._finger_pubs: dict[tuple[str, str], ChannelPublisher] = {}
        for name in FINGER_NAMES:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/{name}/{side}"
                pub = ChannelPublisher(topic, FloatMsg)
                pub.Init()
                self._finger_pubs[(name, side)] = pub

        # ── GR00T ONNX policies ──────────────────────────────────────────
        print("[BRIDGE] Loading GR00T Balance/Walk ONNX policies...")
        balance_path = hf_hub_download(
            repo_id=GROOT_REPO_ID,
            filename="GR00T-WholeBodyControl-Balance.onnx",
        )
        walk_path = hf_hub_download(
            repo_id=GROOT_REPO_ID,
            filename="GR00T-WholeBodyControl-Walk.onnx",
        )
        self._policy_balance = ort.InferenceSession(balance_path)
        self._policy_walk = ort.InferenceSession(walk_path)
        print("[BRIDGE] GR00T policies loaded")

        # ── Locomotion controller state ──────────────────────────────────
        self._cmd = np.zeros(3, dtype=np.float32)       # vx, vy, yaw_rate
        self._height_cmd = 0.74
        self._orientation_cmd = np.zeros(3, dtype=np.float32)
        self._action = np.zeros(15, dtype=np.float32)    # previous policy action
        self._obs_history: deque[np.ndarray] = deque(maxlen=6)
        for _ in range(6):
            self._obs_history.append(np.zeros(86, dtype=np.float32))

        # ── Shared arm targets (set by send(), read by controller thread) ─
        self._lock = threading.Lock()
        self._arm_q = np.copy(GROOT_DEFAULT_ANGLES)      # full 29-DOF buffer
        # Only joints 15-28 are written by teleop IK

        # ── Startup ramp: smoothly interpolate from current pose to target ─
        self._ramp_duration = 2.0   # seconds to reach target pose
        self._ramp_start_q = None   # captured from first lowstate read
        self._ramp_t0 = None        # time when ramp started
        self._ramp_done = False

        # ── Start 50 Hz controller thread ────────────────────────────────
        self._running = True
        self._thread = threading.Thread(target=self._controller_loop, daemon=True)
        self._thread.start()

        print("[BRIDGE] LowCmd publisher on rt/lowcmd (all 29 joints, 50 Hz)")
        print("[BRIDGE] DDS finger topics: rt/inspire_hand/{finger}/{l,r}")

    # ── Public API ────────────────────────────────────────────────────────

    def set_initial_pose(self, joint_overrides: dict[int, float]):
        """Set arm joint targets for the pre-calibration hold pose.

        Parameters
        ----------
        joint_overrides : dict[int, float]
            Mapping of joint index (0-28) to target angle (rad).
            Only joints 15-28 (arms) are meaningful here.
        """
        with self._lock:
            for idx, val in joint_overrides.items():
                self._arm_q[idx] = val

    def send(
        self,
        g1_right_arm_ik=None,
        g1_left_arm_ik=None,
        right_fingers_12: np.ndarray | None = None,
        left_fingers_12: np.ndarray | None = None,
    ):
        """Update arm targets and send finger commands.

        Called from teleop main loop at ~30 Hz. Arm targets are picked up by
        the 50 Hz controller thread on its next cycle.

        Parameters
        ----------
        g1_right_arm_ik : ArmIKSolver | None
            Already-solved right arm IK solver (7 DOF in ._last_q).
        g1_left_arm_ik : ArmIKSolver | None
            Already-solved left arm IK solver (7 DOF in ._last_q).
        right_fingers_12 : ndarray(12,) | None
            Right hand joint angles from IKRetargeter (radians).
        left_fingers_12 : ndarray(12,) | None
            Left hand joint angles from IKRetargeter (radians).
        """
        with self._lock:
            if g1_left_arm_ik is not None:
                self._arm_q[15:22] = g1_left_arm_ik._last_q
            if g1_right_arm_ik is not None:
                self._arm_q[22:29] = g1_right_arm_ik._last_q

        self._send_fingers(right_fingers_12, "r")
        self._send_fingers(left_fingers_12, "l")

    def set_locomotion_cmd(
        self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0
    ):
        """Set locomotion velocity command. Default = stand still."""
        self._cmd[:] = [vx, vy, yaw_rate]

    # ── Controller thread (50 Hz) ─────────────────────────────────────────

    def _controller_loop(self):
        _t_start = time.monotonic()
        _warned_no_state = False
        while self._running:
            t0 = time.monotonic()

            lowstate = self._state_sub.Read()  # joint positions
            if lowstate is not None:
                _warned_no_state = False
                self._step(lowstate)
            else:
                if not _warned_no_state and (time.monotonic() - _t_start) > 2.0:
                    print(
                        "[BRIDGE] WARNING: no rt/lowstate received after 2 s — "
                        "check robot connection / DDS domain. "
                        "Running arm-only fallback (default leg positions)."
                    )
                    _warned_no_state = True

            elapsed = time.monotonic() - t0
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step(self, lowstate): 
        """Run one GR00T policy step and publish merged LowCmd."""
        # ── Read joint state from lowstate ────────────────────────────────
        qj = np.zeros(29, dtype=np.float32)
        dqj = np.zeros(29, dtype=np.float32)
        for i in range(29):
            qj[i] = lowstate.motor_state[i].q
            dqj[i] = lowstate.motor_state[i].dq

        # ── IMU ───────────────────────────────────────────────────────────
        quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        gravity = _get_gravity_orientation(quat)

        obs = np.zeros(86, dtype=np.float32)
        obs[0:3] = self._cmd * CMD_SCALE
        obs[3] = self._height_cmd
        obs[4:7] = self._orientation_cmd
        obs[7:10] = ang_vel * ANG_VEL_SCALE
        obs[10:13] = gravity
        obs[13:42] = (qj - GROOT_DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[42:71] = dqj * DOF_VEL_SCALE
        obs[71:86] = self._action   # 15-D previous action

        self._obs_history.append(obs.copy())

        # Stack 6 frames -> 516-D input
        obs_stacked = np.zeros(516, dtype=np.float32)
        for i, frame in enumerate(self._obs_history):
            obs_stacked[i * 86:(i + 1) * 86] = frame

        # ── Select policy (balance vs walk) ───────────────────────────────
        cmd_mag = np.linalg.norm(self._cmd)
        policy = self._policy_balance if cmd_mag < 0.05 else self._policy_walk

        # ── Run ONNX inference -> 15 lower-body actions ───────────────────
        inputs = {policy.get_inputs()[0].name: obs_stacked[np.newaxis, :]}
        self._action = policy.run(None, inputs)[0].squeeze().astype(np.float32)

        # ── Lower body targets (joints 0-14) ─────────────────────────────
        lower_q = GROOT_DEFAULT_ANGLES[:15] + self._action * ACTION_SCALE

        # ── Build full 29-DOF target_q for rt/lowcmd ──────────────────
        target_q = np.copy(GROOT_DEFAULT_ANGLES)
        target_q[:15] = lower_q

        # ── Arm targets (joints 15-28) ───────────────────────────────────
        with self._lock:
            target_q[15:29] = self._arm_q[15:29]

        # ── Startup ramp: blend from current pose to target ──────────────
        if not self._ramp_done:
            if self._ramp_start_q is None:
                # Capture the robot's actual pose on first step
                self._ramp_start_q = qj.copy()
                self._ramp_t0 = time.monotonic()
                print(f"[BRIDGE] Ramp started from current pose")

            elapsed = time.monotonic() - self._ramp_t0
            alpha = min(elapsed / self._ramp_duration, 1.0)
            # Smooth ease-in-out
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target_q = (1.0 - alpha) * self._ramp_start_q + alpha * target_q

            if elapsed >= self._ramp_duration:
                self._ramp_done = True
                print(f"[BRIDGE] Ramp complete")

        # ── Collision diagnostic: detect a second rt/lowcmd publisher ────────
        # If rt/framereserve tick increments by >1 between our steps, another
        # process is writing LowCmd and overwriting our commands.
        try:
            tick = lowstate.tick
            if self._last_tick is not None and not self._collision_warned:
                skip = tick - self._last_tick - 1
                if skip > 5:
                    print(
                        f"[BRIDGE] WARNING: lowstate tick skipped {skip} steps — "
                        "another process is likely publishing rt/lowcmd (C++ deploy?). "
                        "Kill it to stop arm vibration."
                    )
                    self._collision_warned = True
            self._last_tick = tick
        except Exception:
            pass

        # ── Read mode_machine from lowstate (like LeRobot does) ────────
        self._lowcmd_msg.mode_machine = lowstate.mode_machine

        # ── Publish LowCmd on rt/lowcmd (all 29 joints) ──────────────────
        for i in range(29):
            self._lowcmd_msg.motor_cmd[i].mode = 1
            self._lowcmd_msg.motor_cmd[i].q = float(target_q[i])
            self._lowcmd_msg.motor_cmd[i].dq = 0.0
            self._lowcmd_msg.motor_cmd[i].kp = float(MOTOR_KP[i])
            self._lowcmd_msg.motor_cmd[i].kd = float(MOTOR_KD[i])
            self._lowcmd_msg.motor_cmd[i].tau = 0.0

        self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
        self._lowcmd_pub.Write(self._lowcmd_msg)

    # ── Finger DDS (unchanged) ────────────────────────────────────────────

    def _send_fingers(self, fingers_12: np.ndarray | None, side: str):
        if fingers_12 is None:
            return
        values_6 = retarget_12_to_6(fingers_12)
        for i, name in enumerate(FINGER_NAMES):
            self._finger_pubs[(name, side)].Write(FloatMsg(value=values_6[i]))

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        self._running = False
        self._thread.join(timeout=2)

"""
precision_bridge.py — Output adapter: teleop joint angles → GR00T (ZMQ) + Inspire (DDS).

This is NOT a standalone pipeline. It is called from teleop_edgard_new_setup.py
after the existing calibration + IK + retargeting has already run. It only handles
the transport layer:

  1. Arm joints → GR00T via ZMQ planner topic with upper_body_position override
     (keeps groot in PLANNER mode; the policy handles legs while arm joints
     are directly overridden via the upper_body_position field).

  2. Finger joints → Inspire hands via CycloneDDS per-finger FloatMsg topics
     (consumed by HandTeleopBridge running on the robot).

Usage (from teleop_edgard_new_setup.py):
    from precision_bridge import PrecisionBridge

    bridge = PrecisionBridge(interface="enx...")  # or None for sim
    # ... in main loop, after G1 IK + hand retargeting:
    bridge.send(
        g1_right_arm_ik,   # ArmIKSolver (solved) — or None
        g1_left_arm_ik,    # ArmIKSolver (solved) — or None
        right_fingers_12,  # np.ndarray(12,) from IKRetargeter — or None
        left_fingers_12,   # np.ndarray(12,) from IKRetargeter — or None
    )
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import zmq

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.normpath(os.path.join(_DIR, "..", "..", "..", ".."))
_GROOT = os.path.join(_PROJECT, "external", "GR00T-WholeBodyControl")
if _GROOT not in sys.path:
    sys.path.insert(0, _GROOT)

import cyclonedds.idl as idl
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

# ── Upper-body joint mapping (17 DOF) ────────────────────────────────────────
# The upper_body_position buffer sent via the planner topic expects 17 joints
# in IsaacLab order. Each slot i maps to MuJoCo joint index:
#   (from policy_parameters.hpp: upper_body_joint_isaaclab_order_in_mujoco_index)
UPPER_BODY_MUJOCO_INDICES = [
    12,
    13,
    14,  # waist_yaw, waist_roll, waist_pitch
    15,
    22,  # left_shoulder_pitch, right_shoulder_pitch
    16,
    23,  # left_shoulder_roll, right_shoulder_roll
    17,
    24,  # left_shoulder_yaw, right_shoulder_yaw
    18,
    25,  # left_elbow, right_elbow
    19,
    26,  # left_wrist_roll, right_wrist_roll
    20,
    27,  # left_wrist_pitch, right_wrist_pitch
    21,
    28,  # left_wrist_yaw, right_wrist_yaw
]

# ── Inspire hand: 12 retargeted joints → 6 normalized motor values ───────────
# Joint limits from inspire_hand/ik_retargeting.py
_TH_YAW_MAX = 1.308
_TH_PITCH_MAX = 0.6
_TH_INT_MAX = 0.8
_TH_DIS_MAX = 0.4
_MCP_MAX = 1.47
_INT_MAX = 1.56

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


def retarget_12_to_6(q12: np.ndarray) -> list[float]:
    """Convert 12 Inspire retargeting angles (rad) → 6 normalized [0,1] values.

    Output order matches hand_teleop_bridge.py FINGER_TOPICS:
      [pinky, ring, middle, index, thumb_bend, thumb_rot]
    Where 0.0 = open, 1.0 = closed.
    """
    return [
        float(np.clip((q12[10] / _MCP_MAX + q12[11] / _INT_MAX) / 2, 0, 1)),  # pinky
        float(np.clip((q12[8] / _MCP_MAX + q12[9] / _INT_MAX) / 2, 0, 1)),  # ring
        float(np.clip((q12[6] / _MCP_MAX + q12[7] / _INT_MAX) / 2, 0, 1)),  # middle
        float(np.clip((q12[4] / _MCP_MAX + q12[5] / _INT_MAX) / 2, 0, 1)),  # index
        float(
            np.clip(
                (q12[1] / _TH_PITCH_MAX + q12[2] / _TH_INT_MAX + q12[3] / _TH_DIS_MAX)
                / 3,
                0,
                1,
            )
        ),  # thumb_bend
        float(np.clip(q12[0] / _TH_YAW_MAX, 0, 1)),  # thumb_rot
    ]


# ── DDS FloatMsg (matches hand_teleop_bridge.py) ────────────────────────────
@dataclass
class FloatMsg(idl.IdlStruct, typename="FloatMsg"):
    value: float


class PrecisionBridge:
    """Output adapter: solved IK joint angles → GR00T (ZMQ) + Inspire (DDS).

    Instantiate once at startup, then call ``send()`` every frame from the
    teleop main loop after the G1 IK and hand retargeting have run.

    Parameters
    ----------
    zmq_port : int
        ZMQ PUB port (C++ deploy connects here with --zmq-port).
    interface : str | None
        Network interface for DDS. None = default (loopback for sim).
    init_dds : bool
        If True, call ChannelFactoryInitialize. Set False if the caller
        (teleop script) already initialised DDS.
    """

    def __init__(
        self,
        zmq_port: int = 5556,
        interface: str | None = None,
        init_dds: bool = True,
    ):
        # ── DDS ───────────────────────────────────────────────────────────
        if init_dds:
            if interface:
                ChannelFactoryInitialize(0, interface)
            else:
                ChannelFactoryInitialize(0)

        # Robot state subscriber (current joint positions, motor order)
        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_hg)
        self._state_sub.Init(None, 0)
        self._current_motor_q = np.zeros(29)

        # Per-finger DDS publishers
        self._finger_pubs: dict[tuple[str, str], ChannelPublisher] = {}
        for name in FINGER_NAMES:
            for side in ("l", "r"):
                topic = f"rt/inspire_hand/{name}/{side}"
                pub = ChannelPublisher(topic, FloatMsg)
                pub.Init()
                self._finger_pubs[(name, side)] = pub

        # ── ZMQ ───────────────────────────────────────────────────────────
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_sock.bind(f"tcp://*:{zmq_port}")
        self._frame_idx = 0

        print(f"[BRIDGE] ZMQ PUB on tcp://*:{zmq_port} (topics: 'command', 'planner')")
        print(f"[BRIDGE] DDS finger topics: rt/inspire_hand/{{finger}}/{{l,r}}")

    # ── Read current robot state ──────────────────────────────────────────
    def _read_robot_state(self) -> np.ndarray:
        """Read current 29 joint positions from rt/lowstate (motor order).

        Falls back to the last known state if no new message is available.
        """
        state = self._state_sub.Read()
        if state is not None:
            for i in range(29):
                self._current_motor_q[i] = state.motor_state[i].q
        return self._current_motor_q

    # ── Public API ────────────────────────────────────────────────────────
    def send(
        self,
        g1_right_arm_ik=None,
        g1_left_arm_ik=None,
        right_fingers_12: np.ndarray | None = None,
        left_fingers_12: np.ndarray | None = None,
    ):
        """Send one frame of retargeted data to GR00T + Inspire.

        Parameters
        ----------
        g1_right_arm_ik : ArmIKSolver | None
            Already-solved right arm IK solver. Joint angles are read from
            ``_last_q`` (7 values in MuJoCo order:
            shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw).
        g1_left_arm_ik : ArmIKSolver | None
            Already-solved left arm IK solver (same convention).
        right_fingers_12 : ndarray(12,) | None
            Right hand joint angles from IKRetargeter.retarget() (radians).
        left_fingers_12 : ndarray(12,) | None
            Left hand joint angles from IKRetargeter.retarget() (radians).
        """
        self._send_arms(g1_right_arm_ik, g1_left_arm_ik)
        self._send_fingers(right_fingers_12, "r")
        self._send_fingers(left_fingers_12, "l")

    # ── Arm joints → ZMQ (command + planner topics) ──────────────────────
    def _send_arms(self, g1_right_arm_ik, g1_left_arm_ik):
        # Send command every frame to ensure groot enters CONTROL state.
        # groot's handlePlannerInput guards with `if (start_control_ && !operator_state.start)`
        # so repeated commands are harmless once CONTROL is reached.
        # Sending continuously avoids the ZMQ slow-joiner problem (SUB sockets
        # need time to reconnect after the PUB socket binds).
        cmd = build_command_message(start=True, stop=False, planner=True)
        self._zmq_sock.send(cmd)
        if self._frame_idx == 0:
            print("[BRIDGE] Sending command every frame: start=True, planner=True")

        # Read current state for non-retargeted joints
        motor_q = self._read_robot_state().copy()

        # Override arm joints with IK output (MuJoCo order)
        # Left arm: MuJoCo indices 15-21
        if g1_left_arm_ik is not None:
            motor_q[15:22] = g1_left_arm_ik._last_q

        # Right arm: MuJoCo indices 22-28
        if g1_right_arm_ik is not None:
            motor_q[22:29] = g1_right_arm_ik._last_q

        # Extract 17 upper-body joints from motor_q (MuJoCo order)
        # using the mapping that groot expects for the planner topic
        upper_body_pos = [float(motor_q[mj]) for mj in UPPER_BODY_MUJOCO_INDICES]
        upper_body_vel = [0.0] * 17

        # Send planner message: IDLE mode (legs stand still),
        # with upper_body_position override for arms + waist
        msg = build_planner_message(
            mode=0,  # IDLE
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
            upper_body_position=upper_body_pos,
            upper_body_velocity=upper_body_vel,
        )
        self._zmq_sock.send(msg)
        self._frame_idx += 1

    # ── Finger joints → DDS ──────────────────────────────────────────────
    def _send_fingers(self, fingers_12: np.ndarray | None, side: str):
        if fingers_12 is None:
            return
        values_6 = retarget_12_to_6(fingers_12)
        for i, name in enumerate(FINGER_NAMES):
            self._finger_pubs[(name, side)].Write(FloatMsg(value=values_6[i]))

    # ── Cleanup ───────────────────────────────────────────────────────────
    def close(self):
        self._zmq_sock.close()
        self._zmq_ctx.term()
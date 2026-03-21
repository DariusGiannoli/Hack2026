"""
robots/base.py — Abstract interface shared by all robot controllers.

Every robot controller should subclass ``RobotController`` and implement the
two abstract methods.  The teleoperation loop (mujoco_teleop.py) depends only
on this interface, making it trivial to swap the LEAP hand for a Franka arm or
any other actuated model.

Example
-------
    from robots.base import RobotController

    class FrankaController(RobotController):
        def retarget(self, landmarks):
            ...  # map MediaPipe landmarks → 7 joint targets
            return q_7dof

        @classmethod
        def scene_xml_path(cls) -> str:
            return os.path.join(os.path.dirname(__file__), "franka", "scene.xml")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import mujoco


class RobotController(ABC):
    """
    Abstract base class for robot teleoperation controllers.

    A concrete subclass receives MediaPipe hand landmarks every frame and
    returns a 1-D array of target joint/actuator values that is written
    directly to ``data.ctrl``.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def retarget(self, landmarks) -> np.ndarray:
        """
        Map input perception data to robot actuator targets.

        Parameters
        ----------
        landmarks :
            result.multi_hand_landmarks[0].landmark  (21 MediaPipe entries)

        Returns
        -------
        np.ndarray
            1-D array of desired actuator values (length = number of actuators).
        """
        ...

    @classmethod
    @abstractmethod
    def scene_xml_path(cls) -> str:
        """Absolute path to this robot's MuJoCo scene XML file."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (override if needed)
    # ------------------------------------------------------------------

    def init(self, data: mujoco.MjData, start_pos: np.ndarray,
             start_quat: np.ndarray) -> None:
        """
        Called once before the simulation loop starts.

        Default: places the mocap body ``"hand_proxy"`` at ``start_pos`` /
        ``start_quat`` and runs ``mj_forward`` to initialise physics.
        Override for robots that don't use a mocap body (e.g. a fixed-base arm).
        """
        try:
            mid  = self.model.body("hand_proxy").mocapid[0]
            data.mocap_pos[mid]  = start_pos
            data.mocap_quat[mid] = start_quat
            jid  = self.model.joint("palm_free").id
            addr = self.model.jnt_qposadr[jid]
            data.qpos[addr:addr+3] = start_pos
            data.qpos[addr+3:addr+7] = start_quat
        except (KeyError, IndexError):
            pass  # robot has no mocap body — no-op
        mujoco.mj_forward(self.model, data)

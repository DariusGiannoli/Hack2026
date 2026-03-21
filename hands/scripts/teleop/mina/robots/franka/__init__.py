"""
robots/franka — Franka FR3 arm controller.

Exports
-------
  FrankaIKRetargeter   direct angle-based retargeter (MediaPipe Pose → 7 joint angles)
  FrankaController     thin RobotController wrapper around FrankaIKRetargeter
"""

from robots.franka.ik_retargeting import FrankaIKRetargeter  # noqa: F401

import os
import numpy as np
import mujoco
from robots.base import RobotController


class FrankaController(RobotController):
    """
    RobotController implementation for the Franka FR3 arm.

    Wraps FrankaIKRetargeter so the teleoperation loop can use the generic
    RobotController interface.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        self._ik = FrankaIKRetargeter(model)

    def retarget(self, landmarks) -> np.ndarray:
        """Map MediaPipe Pose result → 7 Franka actuator targets."""
        return self._ik.retarget(landmarks)

    @classmethod
    def scene_xml_path(cls) -> str:
        """Path to the Franka scene XML (co-located with this package)."""
        return os.path.join(os.path.dirname(__file__), "scene.xml")

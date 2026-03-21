"""
robots/leap_hand — LEAP Hand controller.

Exports
-------
  IKRetargeter   direct angle-based retargeter  (MediaPipe → 16 joint angles)
  palm_quat      palm orientation as MuJoCo quaternion
  LeapController thin RobotController wrapper around IKRetargeter
"""

from robots.leap_hand.ik_retargeting import IKRetargeter, palm_quat  # noqa: F401

import os
import numpy as np
import mujoco
from robots.base import RobotController


class LeapController(RobotController):
    """
    RobotController implementation for the LEAP Hand.

    Wraps ``IKRetargeter`` so the teleoperation loop can be written against the
    generic ``RobotController`` interface.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        self._ik = IKRetargeter(model)

    # ------------------------------------------------------------------
    def retarget(self, landmarks) -> np.ndarray:
        """Map MediaPipe landmarks → 16 LEAP actuator targets."""
        return self._ik.retarget(None, landmarks)

    @classmethod
    def scene_xml_path(cls) -> str:
        """Path to the LEAP scene XML (co-located with this package)."""
        return os.path.join(os.path.dirname(__file__), "scene.xml")

"""
robot/inspire_bridge.py
------------------------
Re-exports InspireHandSensors for backwards-compat imports.
Control (publishing commands) lives in:
    GR00T-WholeBodyControl/.../utils/inspire_hand.py
"""
from .inspire_hand_sensors import InspireHandSensors, DOF_NAMES, NUM_DOF

__all__ = ["InspireHandSensors", "DOF_NAMES", "NUM_DOF"]
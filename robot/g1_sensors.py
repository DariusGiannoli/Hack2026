# hardware/g1_sensors.py
# Reads two DDS streams from the Unitree G1:
#   1. rt/hand_state  → finger pressure sensors  (grip force)
#   2. rt/lowstate     → arm joint torques        (weight estimation)
#
# Both arrive via callbacks and are exposed as thread-safe properties.

import time
import threading
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_


# ── G1 arm joint indices ───────────────────────────────────────────────────
# Left arm
LEFT_ARM_JOINTS = {
    "L_ShoulderPitch": 15,
    "L_ShoulderRoll":  16,
    "L_ShoulderYaw":   17,
    "L_Elbow":         18,
    "L_WristRoll":     19,
    "L_WristPitch":    20,
    "L_WristYaw":      21,
}
# Right arm
RIGHT_ARM_JOINTS = {
    "R_ShoulderPitch": 22,
    "R_ShoulderRoll":  23,
    "R_ShoulderYaw":   24,
    "R_Elbow":         25,
    "R_WristRoll":     26,
    "R_WristPitch":    27,
    "R_WristYaw":      28,
}
ARM_JOINTS = {**LEFT_ARM_JOINTS, **RIGHT_ARM_JOINTS}


class G1Sensors:
    """
    Thread-safe reader for G1 hand pressure + arm joint torques.

    Usage:
        sensors = G1Sensors()
        sensors.connect()          # blocks until first data arrives
        ...
        forces   = sensors.finger_pressures   # dict  {finger_idx: [12 floats]}
        torques  = sensors.arm_torques        # dict  {"L_Elbow": float, ...}
        weight   = sensors.estimated_weight   # float (sum of abs arm torques)
    """

    def __init__(self, network_interface=None):
        self.interface = network_interface
        self._lock = threading.Lock()

        # latest state snapshots
        self._hand_state = None
        self._low_state = None

        # processed values
        self._finger_pressures = {}
        self._arm_torques = {}
        self._estimated_weight = 0.0

    # ── connection ──────────────────────────────────────────────────────
    def connect(self, timeout=10.0):
        print("[G1Sensors] DDS init...", flush=True)
        if self.interface:
            ChannelFactoryInitialize(0, self.interface)
        else:
            ChannelFactoryInitialize(0)
        print("[G1Sensors] DDS init done", flush=True)

        # subscribe to low state (full body motors + IMU)
        print("[G1Sensors] Subscribing to rt/lowstate...", flush=True)
        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 10)
        print("[G1Sensors] lowstate subscribed", flush=True)

        # subscribe to hand state (pressure sensors + finger motors)
        print("[G1Sensors] Subscribing to rt/hand_state...", flush=True)
        self._hand_sub = ChannelSubscriber("rt/hand_state", HandState_)
        self._hand_sub.Init(self._on_hand_state, 10)
        print("[G1Sensors] hand_state subscribed", flush=True)

        # wait for first data
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                got_hand = self._hand_state is not None
                got_low = self._low_state is not None
            if got_hand and got_low:
                print("[G1Sensors] Connected — receiving hand + arm data")
                return True
            time.sleep(0.1)

        with self._lock:
            got_hand = self._hand_state is not None
            got_low = self._low_state is not None
        if got_hand or got_low:
            missing = []
            if not got_hand:
                missing.append("hand_state")
            if not got_low:
                missing.append("lowstate")
            print(f"[G1Sensors] Partial connect — missing: {missing}")
            return True
        print("[G1Sensors] Timeout — no data received")
        return False

    # ── DDS callbacks ───────────────────────────────────────────────────
    def _on_hand_state(self, msg: HandState_):
        pressures = {}
        for i, ps in enumerate(msg.press_sensor_state):
            pressures[i] = list(ps.pressure)

        with self._lock:
            self._hand_state = msg
            self._finger_pressures = pressures

    def _on_low_state(self, msg: LowState_):
        torques = {}
        for name, idx in ARM_JOINTS.items():
            torques[name] = msg.motor_state[idx].tau_est

        # simple weight proxy: sum of absolute shoulder + elbow torques
        weight_joints = [15, 16, 18, 22, 23, 25]  # shoulders + elbows
        weight = sum(abs(msg.motor_state[j].tau_est) for j in weight_joints)

        with self._lock:
            self._low_state = msg
            self._arm_torques = torques
            self._estimated_weight = weight

    # ── public API ──────────────────────────────────────────────────────
    @property
    def finger_pressures(self) -> dict:
        """Dict of {finger_idx: [12 pressure floats]}."""
        with self._lock:
            return dict(self._finger_pressures)

    @property
    def total_finger_force(self) -> float:
        """Sum of all pressure sensor readings (0 = no contact)."""
        with self._lock:
            return sum(
                sum(vals) for vals in self._finger_pressures.values()
            )

    @property
    def max_finger_pressure(self) -> float:
        """Peak pressure across all fingers."""
        with self._lock:
            if not self._finger_pressures:
                return 0.0
            return max(
                max(vals) for vals in self._finger_pressures.values()
            )

    @property
    def arm_torques(self) -> dict:
        """Dict of {joint_name: tau_est} for all 14 arm joints."""
        with self._lock:
            return dict(self._arm_torques)

    @property
    def estimated_weight(self) -> float:
        """Rough weight proxy from shoulder + elbow torques."""
        with self._lock:
            return self._estimated_weight

    @property
    def arm_joint_positions(self) -> dict:
        """Current arm joint positions in radians."""
        with self._lock:
            if self._low_state is None:
                return {}
            return {
                name: self._low_state.motor_state[idx].q
                for name, idx in ARM_JOINTS.items()
            }

    @property
    def imu(self) -> dict:
        """Body IMU: rpy, gyro, accel."""
        with self._lock:
            if self._low_state is None:
                return {}
            imu = self._low_state.imu_state
            return {
                "rpy": list(imu.rpy),
                "gyro": list(imu.gyroscope),
                "accel": list(imu.accelerometer),
            }


if __name__ == "__main__":
    import sys
    print("[main] Starting g1_sensors...", flush=True)

    iface = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"[main] Interface: {iface}", flush=True)
    sensors = G1Sensors(network_interface=iface)

    print("[main] Connecting...", flush=True)
    if not sensors.connect():
        print("Failed to connect")
        sys.exit(1)

    print("\nStreaming sensor data (Ctrl+C to stop):\n")
    try:
        while True:
            # finger pressures
            fp = sensors.finger_pressures
            total = sensors.total_finger_force
            peak = sensors.max_finger_pressure
            print(f"Fingers: total_force={total:8.2f}  peak={peak:6.2f}  "
                  f"groups={len(fp)}")

            # arm torques
            torques = sensors.arm_torques
            weight = sensors.estimated_weight
            elbow_l = torques.get("L_Elbow", 0)
            elbow_r = torques.get("R_Elbow", 0)
            print(f"Arm:     weight_proxy={weight:8.2f}  "
                  f"L_elbow={elbow_l:+.2f}  R_elbow={elbow_r:+.2f}")

            # IMU
            imu = sensors.imu
            if imu:
                rpy = imu["rpy"]
                print(f"IMU:     R={rpy[0]:+.3f}  P={rpy[1]:+.3f}  "
                      f"Y={rpy[2]:+.3f}")

            print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped")

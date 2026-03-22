"""
sensor_check.py — Live diagnostic for all haptic-relevant sensors.

Prints a clean table every 0.2s showing:
  - Inspire hand: per-finger force + contact% (L and R)
  - G1 lowstate:  per arm-joint torque estimate
  - HapticNet inputs: the 6 pressure values + 12 torque values that
    will actually be fed into the model

Usage:
    # real robot (replace eth0 with your interface, e.g. enp131s0)
    python sensor_check.py --iface eth0

    # only G1 lowstate (no Inspire)
    python sensor_check.py --iface eth0 --no-inspire

    # only Inspire (no G1 lowstate)
    python sensor_check.py --iface eth0 --no-g1
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iface",       default=None,  help="Network interface (e.g. enp131s0)")
    p.add_argument("--no-inspire",  action="store_true", help="Skip Inspire hand sensors")
    p.add_argument("--no-g1",       action="store_true", help="Skip G1 lowstate/hand_state")
    p.add_argument("--touch",       action="store_true", help="Also subscribe to Inspire touch arrays")
    p.add_argument("--hz",          type=float, default=5.0, help="Print rate (Hz)")
    return p.parse_args()


def bar(val, max_val, width=12, char="█"):
    filled = int(round(abs(val) / max_val * width))
    filled = min(filled, width)
    return char * filled + "·" * (width - filled)


def print_inspire(inspire):
    dof = ["Pk", "Rg", "Md", "Ix", "Tb", "Tr"]
    for hand, label in [(inspire.left, "L"), (inspire.right, "R")]:
        fresh = "✓" if hand.is_fresh else "✗"
        forces = hand.force_act
        angles = hand.contact_pct
        print(f"  Inspire {label} [{fresh}]  "
              f"peak={hand.max_force:5.0f}  total={hand.total_force:6.0f}")
        # force row
        f_str = "  ".join(f"{n}={f:+5d}" for n, f in zip(dof, forces))
        print(f"    force:   {f_str}")
        # angle/contact row
        a_str = "  ".join(f"{n}={a:5.1f}" for n, a in zip(dof, angles))
        print(f"    contact: {a_str}")


def print_g1(sensors):
    torques = sensors.arm_torques
    weight  = sensors.estimated_weight
    joints_l = ["L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw",
                 "L_Elbow", "L_WristRoll", "L_WristPitch"]
    joints_r = ["R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw",
                 "R_Elbow", "R_WristRoll", "R_WristPitch"]
    print(f"  G1 torques  weight_proxy={weight:6.2f}")
    l_str = "  ".join(f"{j.split('_')[1][:3]}={torques.get(j,0):+5.2f}" for j in joints_l)
    r_str = "  ".join(f"{j.split('_')[1][:3]}={torques.get(j,0):+5.2f}" for j in joints_r)
    print(f"    L arm: {l_str}")
    print(f"    R arm: {r_str}")

    fp = sensors.finger_pressures
    if fp:
        total_fp = sensors.total_finger_force
        peak_fp  = sensors.max_finger_pressure
        print(f"  G1 hand_state  total={total_fp:.2f}  peak={peak_fp:.2f}  "
              f"finger_groups={len(fp)}")


def print_hapticnet_inputs(inspire, sensors):
    """Show the 6 pressure + 12 torque values that feed into HapticNet."""
    # Pressure: prefer Inspire right hand force, fall back to G1 hand_state
    if inspire is not None and inspire.right.is_fresh:
        raw_force = inspire.right.force_act  # 6 values
        # normalise by a rough max (Inspire force units ~0-1000)
        pressure_6 = [abs(f) / 500.0 for f in raw_force]
        src = "Inspire-R"
    elif inspire is not None and inspire.left.is_fresh:
        raw_force = inspire.left.force_act
        pressure_6 = [abs(f) / 500.0 for f in raw_force]
        src = "Inspire-L"
    elif sensors is not None:
        fp = sensors.finger_pressures
        vals = [fp[k][0] if k in fp else 0.0 for k in range(6)]
        pressure_6 = vals
        src = "G1-hand"
    else:
        pressure_6 = [0.0] * 6
        src = "none"

    # Torque: from G1 lowstate, 6L + 6R joints
    if sensors is not None:
        t = sensors.arm_torques
        joints_12 = [
            "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw",
            "L_Elbow", "L_WristRoll", "L_WristPitch",
            "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw",
            "R_Elbow", "R_WristRoll", "R_WristPitch",
        ]
        torque_12 = [t.get(j, 0.0) for j in joints_12]
    else:
        torque_12 = [0.0] * 12

    dof = ["Pk", "Rg", "Md", "Ix", "Tb", "Tr"]
    p_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(dof, pressure_6))
    t_str = "  ".join(f"{v:+.2f}" for v in torque_12)
    print(f"\n  ── HapticNet inputs ({src}) ──")
    print(f"    pressure[6]: {p_str}")
    print(f"    torque[12]:  {t_str}")


def main():
    args = parse_args()
    inspire = None
    sensors = None

    # Init DDS exactly once for the whole process
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print(f"[check] DDS init on {args.iface or 'default'}...")
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)
    print("[check] DDS ready")

    if not args.no_inspire:
        from robot.inspire_hand_sensors import InspireHandSensors
        inspire = InspireHandSensors(
            network_interface=args.iface,
            subscribe_touch=args.touch,
        )
        print("[check] Connecting to Inspire hand sensors...")
        ok = inspire.connect(timeout=8.0, init_dds=False)  # DDS already up
        if not ok:
            print("  [warn] Inspire hand sensors not responding — continuing anyway")

    if not args.no_g1:
        from robot.g1_sensors import G1Sensors
        sensors = G1Sensors(network_interface=args.iface)
        print("[check] Connecting to G1 sensors (lowstate + hand_state)...")
        sensors._skip_dds_init = True  # signal to skip re-init
        ok = sensors.connect(timeout=8.0)
        if not ok:
            print("  [warn] G1 sensors not responding — continuing anyway")

    if inspire is None and sensors is None:
        print("[check] Nothing to connect to. Use --no-inspire or --no-g1 to skip one source.")
        sys.exit(1)

    interval = 1.0 / args.hz
    print(f"\n[check] Streaming at {args.hz:.0f} Hz (Ctrl+C to stop)\n")

    try:
        while True:
            t0 = time.time()
            print("\033[H\033[J", end="")  # clear terminal
            print(f"=== Sensor check  {time.strftime('%H:%M:%S')} ===\n")

            if inspire is not None:
                print_inspire(inspire)
                print()

            if sensors is not None:
                print_g1(sensors)
                print()

            print_hapticnet_inputs(inspire, sensors)

            elapsed = time.time() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[check] Stopped")


if __name__ == "__main__":
    main()

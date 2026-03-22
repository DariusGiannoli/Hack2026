import sys
import time
import numpy as np
from precision_bridge import PrecisionBridge

interface = sys.argv[1] if len(sys.argv) > 1 else "enx00e04c681234"
bridge = PrecisionBridge(interface=interface)

dt = 0.02  # 50Hz
freq = 0.3  # Hz — slow sinusoid
t0 = time.monotonic()

print("[SCRIPT] Starting sinusoidal arm motion (Ctrl+C to stop)")

try:
    while True:
        t = time.monotonic() - t0
        s = np.sin(2 * np.pi * freq * t)

        with bridge._lock:
            # Left arm
            bridge._arm_q[15] = s * 0.3        # LeftShoulderPitch
            bridge._arm_q[16] = s * 0.3         # LeftShoulderRoll
            bridge._arm_q[19] = s * 0.8         # LeftWristYaw

            # Right arm (mirror)
            bridge._arm_q[22] = s * 0.3         # RightShoulderPitch
            bridge._arm_q[23] = -s * 0.3        # RightShoulderRoll
            bridge._arm_q[26] = -s * 0.8        # RightWristYaw

        time.sleep(dt)
except KeyboardInterrupt:
    print("\n[SCRIPT] Stopped")
    bridge.close()

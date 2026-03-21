import sys
import time
import numpy as np
from precision_bridge import PrecisionBridge

interface = sys.argv[1] if len(sys.argv) > 1 else "enx00e04c681234"
bridge = PrecisionBridge(interface=interface)

# Smoothly ramp joint 19 to target over ramp_duration seconds
target = 1.1
ramp_duration = 2.0  # seconds
dt = 0.02  # match 50Hz control loop
steps = int(ramp_duration / dt)

for i in range(1, steps + 1):
    alpha = i / steps
    with bridge._lock:
        bridge._arm_q[19] = alpha * target
        bridge._arm_q[26] = -alpha * target 
    time.sleep(dt)

print(f"[SCRIPT] Ramp complete, joint 19 = {target}")

while True:
    time.sleep(0.1)

# hardware/g1_camera.py
# Continuous camera stream from Unitree G1 head camera.
# Provides frames to the perception pipeline (DINOv2 → MLP → freq/wave).

import time
import threading
import numpy as np
import cv2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class G1Camera:
    def __init__(self, network_interface="enp131s0"):
        self.interface = network_interface
        self.client = None
        self.running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def connect(self):
        ChannelFactoryInitialize(0, self.interface)
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
        # test grab
        code, data = self.client.GetImageSample()
        if code != 0:
            print(f"[G1Camera] Failed to get test frame (code={code})")
            return False
        print("[G1Camera] Connected")
        return True

    def start(self, fps=10):
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, args=(fps,), daemon=True
        )
        self._thread.start()
        print(f"[G1Camera] Streaming at ~{fps} FPS")

    def _loop(self, fps):
        interval = 1.0 / fps
        while self.running:
            t0 = time.time()
            code, data = self.client.GetImageSample()
            if code == 0:
                buf = np.frombuffer(bytes(data), dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[G1Camera] Stopped")


if __name__ == "__main__":
    import sys
    iface = sys.argv[1] if len(sys.argv) > 1 else "enp131s0"

    cam = G1Camera(network_interface=iface)
    if not cam.connect():
        sys.exit(1)

    cam.start(fps=10)
    print("Press Q to quit")

    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("G1 Camera", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

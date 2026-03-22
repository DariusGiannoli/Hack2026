# robot/g1_camera.py
# G1 head camera via UDP stream from Orin.
# Sender runs on Orin: sudo python3.8 g1_stream_sender.py 192.168.123.100 9000 30 3

import time
import socket
import struct
import threading
import numpy as np
import cv2


class G1Camera:
    def __init__(self, port=9000, **kwargs):
        self.port = port
        self.running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self._sock = None

    def connect(self, **kwargs):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(5.0)

        print(f"[G1Camera] Listening on UDP :{self.port} — waiting for sender...")
        try:
            data, addr = self._sock.recvfrom(65535)
            if len(data) > 4:
                raw_jpg = data[4:]
                frame = cv2.imdecode(
                    np.frombuffer(raw_jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                    print(f"[G1Camera] Connected — receiving from {addr[0]}")
                    return True
        except socket.timeout:
            pass

        print("[G1Camera] No frames — is g1_stream_sender.py running on Orin?")
        return False

    def start(self, fps=30):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[G1Camera] Receiving stream on :{self.port}")

    def _loop(self):
        while self.running:
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) <= 4:
                continue

            raw_jpg = data[4:]
            frame = cv2.imdecode(
                np.frombuffer(raw_jpg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[G1Camera] Stopped")


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

    cam = G1Camera(port=port)
    if not cam.connect():
        sys.exit(1)

    cam.start()
    print("Press Q to quit")

    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("G1 Head Camera", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

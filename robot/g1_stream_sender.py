"""Run ON THE ORIN.  Grabs G1 head camera, sends JPEG over UDP.

Usage:
    sudo python3.8 g1_stream_sender.py <PC_IP> [port] [fps] [device]
    sudo python3.8 g1_stream_sender.py 192.168.123.100 9000 30 3
"""

import sys, time, socket, struct
import numpy as np
import cv2

PC_IP   = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.100"
PORT    = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
FPS     = int(sys.argv[3]) if len(sys.argv) > 3 else 30
DEVICE  = int(sys.argv[4]) if len(sys.argv) > 4 else 3
QUALITY = 70

def main():
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    ret, frame = cap.read()
    if not ret:
        print(f"[sender] Cannot open /dev/video{DEVICE}")
        sys.exit(1)

    h, w = frame.shape[:2]
    ch = frame.shape[2] if len(frame.shape) == 3 else 1
    print(f"[sender] /dev/video{DEVICE} {w}x{h} ch={ch} → {PC_IP}:{PORT} @ {FPS}fps")
    if ch == 1:
        print(f"[sender] Grayscale detected — converting to BGR for DINO")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, QUALITY]

    seq = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # convert grayscale/IR to 3-channel BGR
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # downscale to 640 wide for lower latency
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            packet = struct.pack("!I", seq) + jpg.tobytes()
            seq += 1

            if len(packet) < 65000:
                sock.sendto(packet, (PC_IP, PORT))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sock.close()
        print("[sender] Stopped")

if __name__ == "__main__":
    main()

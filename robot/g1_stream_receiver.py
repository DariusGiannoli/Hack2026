"""Run ON YOUR PC.  Receives JPEG-over-UDP from the Orin and displays.

Usage:
    python3 g1_stream_receiver.py [port]
    python3 g1_stream_receiver.py 9000
"""

import sys, os, socket, struct, time, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import cv2

PORT      = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
HTTP_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LATEST    = os.path.join(SAVE_DIR, "scene.jpg")

# shared latest JPEG bytes for MJPEG server
_lock       = threading.Lock()
_latest_jpg = b""

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence access logs

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _lock:
                    jpg = _latest_jpg
                if jpg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(0.033)  # ~30 fps to Pico
        except (BrokenPipeError, ConnectionResetError):
            pass

def start_mjpeg_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    server.serve_forever()

def main():
    global _latest_jpg

    os.makedirs(SAVE_DIR, exist_ok=True)

    # start MJPEG server in background
    threading.Thread(target=start_mjpeg_server, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(5.0)

    import socket as _s
    local_ip = _s.gethostbyname(_s.gethostname())
    print(f"[receiver] UDP  :{PORT}  — laptop display")
    print(f"[receiver] HTTP :{HTTP_PORT} — open http://{local_ip}:{HTTP_PORT} on Pico browser")
    print(f"[receiver] Press S to snapshot, Q to quit")

    last_seq = -1
    dropped  = 0

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            print("[receiver] No data — is the sender running?")
            continue

        if len(data) < 5:
            continue

        seq = struct.unpack("!I", data[:4])[0]
        if seq <= last_seq and last_seq - seq < 1000:
            dropped += 1
            continue
        last_seq = seq

        raw_jpg = data[4:]
        with _lock:
            _latest_jpg = raw_jpg

        frame = cv2.imdecode(np.frombuffer(raw_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("G1 Head Camera", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, frame)
            cv2.imwrite(LATEST, frame)
            print(f"[receiver] Saved: {path}")
        elif key == ord("q"):
            break

    sock.close()
    cv2.destroyAllWindows()
    print(f"[receiver] Done (dropped {dropped} late packets)")

if __name__ == "__main__":
    main()

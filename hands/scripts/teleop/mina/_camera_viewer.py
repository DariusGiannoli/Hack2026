"""Camera debug viewer — runs as a separate subprocess.

Why a subprocess?
-----------------
On macOS, ``mjpython`` owns the Cocoa/GLFW event loop in the main thread.
Any call to ``cv2.imshow`` in the *same* process triggers a second Cocoa
event loop, causing a crash or hang.

Solution: spawn *this* script with plain ``python`` (not ``mjpython``).
The parent process (``teleop_sim_hand.py``) encodes debug frames to JPEG
and writes them over stdin using a simple 4-byte length-prefix framing:

    [4 bytes big-endian uint32: frame_len] [frame_len bytes: JPEG data]

Sending frame_len == 0 is the shutdown signal.

Usage (called automatically by teleop_sim_hand.py — do not run directly):
    python scripts/teleop/_camera_viewer.py

Port of Binocular-Teleop/_camera_viewer.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

import struct
import sys

import cv2
import numpy as np

WINDOW_NAME = "Mina Teleop — Camera Debug"


def main() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 360)

    stdin = sys.stdin.buffer

    while True:
        # --- read 4-byte length header ---
        header = stdin.read(4)
        if len(header) < 4:
            break  # parent closed stdin → exit cleanly

        (frame_len,) = struct.unpack(">I", header)
        if frame_len == 0:
            break  # explicit shutdown signal

        # --- read JPEG payload ---
        buf = b""
        remaining = frame_len
        while remaining > 0:
            chunk = stdin.read(remaining)
            if not chunk:
                break
            buf += chunk
            remaining -= len(chunk)

        if len(buf) < frame_len:
            break  # truncated — parent died

        # --- decode and display ---
        arr  = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow(WINDOW_NAME, frame)

        # Poll for 'q' key (1 ms)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def viewer_loop(frame_q, reset_flag=None) -> None:
    """Legacy entry point used by teleop_edgard.py (multiprocessing.Queue protocol).

    Parameters
    ----------
    frame_q : multiprocessing.Queue
        Receives BGR numpy frames. ``None`` sentinel signals shutdown.
    reset_flag : multiprocessing.Value, optional
        Shared int flag — set to 1 when the user presses 'r' in the viewer.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 360)

    while True:
        try:
            frame = frame_q.get(timeout=0.1)
        except Exception:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and reset_flag is not None:
                reset_flag.value = 1
            continue

        if frame is None:
            break

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and reset_flag is not None:
            reset_flag.value = 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

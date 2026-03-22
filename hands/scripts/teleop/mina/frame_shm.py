"""frame_shm.py — Shared-memory bridge for ZED camera frames between processes.

Teleop writes downscaled left+right BGR frames; the Kalman GUI reads them.

Writer (teleop side):
    from frame_shm import FrameShmWriter
    w = FrameShmWriter(width=448, height=252)
    w.write_left(frame_bgr)
    w.write_right(frame_bgr)
    w.close()

Reader (GUI side):
    from frame_shm import FrameShmReader
    r = FrameShmReader(width=448, height=252)
    left_rgb  = r.read_left()    # numpy HxWx3 RGB or None
    right_rgb = r.read_right()   # numpy HxWx3 RGB or None
    r.close()

Layout:
    [0:1]                   left_valid   (uint8)
    [1:1+H*W*3]             left frame   (BGR, uint8, row-major)
    [1+H*W*3]               right_valid  (uint8)
    [2+H*W*3 : 2+2*H*W*3]  right frame  (BGR, uint8, row-major)
"""

import numpy as np
from multiprocessing import shared_memory

_SHM_NAME = "zed_frames"


def _total_size(h, w):
    frame_bytes = h * w * 3
    return 1 + frame_bytes + 1 + frame_bytes   # valid+frame × 2


class FrameShmWriter:
    def __init__(self, width: int = 448, height: int = 252):
        self.w, self.h = width, height
        self._fb = self.h * self.w * 3
        size = _total_size(self.h, self.w)
        try:
            self._shm = shared_memory.SharedMemory(
                name=_SHM_NAME, create=True, size=size)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(
                name=_SHM_NAME, create=False)

    def write_left(self, frame_bgr: np.ndarray):
        small = self._resize(frame_bgr)
        self._shm.buf[0] = 1
        self._shm.buf[1:1 + self._fb] = small.tobytes()

    def write_right(self, frame_bgr: np.ndarray):
        off = 1 + self._fb
        small = self._resize(frame_bgr)
        self._shm.buf[off] = 1
        self._shm.buf[off + 1:off + 1 + self._fb] = small.tobytes()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            import cv2
            return cv2.resize(frame, (self.w, self.h),
                              interpolation=cv2.INTER_NEAREST)
        return frame

    def close(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class FrameShmReader:
    def __init__(self, width: int = 448, height: int = 252):
        self.w, self.h = width, height
        self._fb = self.h * self.w * 3
        self._shm = None

    def _attach(self):
        if self._shm is not None:
            return True
        try:
            self._shm = shared_memory.SharedMemory(
                name=_SHM_NAME, create=False)
            return True
        except FileNotFoundError:
            return False

    def read_left(self):
        if not self._attach():
            return None
        if self._shm.buf[0] < 1:
            return None
        raw = bytes(self._shm.buf[1:1 + self._fb])
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        return bgr[:, :, ::-1].copy()   # BGR → RGB

    def read_right(self):
        if not self._attach():
            return None
        off = 1 + self._fb
        if self._shm.buf[off] < 1:
            return None
        raw = bytes(self._shm.buf[off + 1:off + 1 + self._fb])
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        return bgr[:, :, ::-1].copy()   # BGR → RGB

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

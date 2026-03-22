"""imu_shm.py — Shared-memory bridge for IMU data between processes.

Teleop writes real IMU snapshots here; the Kalman GUI (or any other
process) reads them without needing its own Bluetooth connection.

Usage — writer (teleop side):
    from imu_shm import ImuShmWriter
    w = ImuShmWriter("right")      # creates /dev/shm/imu_right
    w.write(euler=[r,p,y], acc=[ax,ay,az], gyr=[gx,gy,gz], quat=[w,x,y,z])
    w.close()

Usage — reader (GUI side):
    from imu_shm import ImuShmReader
    r = ImuShmReader("right")      # attaches to /dev/shm/imu_right
    snap = r.read()                # {'euler': [...], 'acc': [...], ...} or None
    r.close()

Layout (112 bytes):
    [0:8]   valid flag  (float64, 1.0 = data present)
    [8:32]  euler[3]    (float64 × 3)
    [32:56] acc[3]      (float64 × 3)
    [56:80] gyr[3]      (float64 × 3)
    [80:112] quat[4]    (float64 × 4)
"""

import struct
from multiprocessing import shared_memory

_SHM_PREFIX = "imu_"
_FMT = "d3d3d3d4d"          # 1 + 3 + 3 + 3 + 4 = 14 doubles
_SIZE = struct.calcsize(_FMT)  # 112 bytes


class ImuShmWriter:
    """Allocate (or re-attach to) a named shared-memory block and write IMU data."""

    def __init__(self, label: str = "right"):
        self._name = _SHM_PREFIX + label
        # Try to create; if it already exists, attach + overwrite
        try:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=True, size=_SIZE,
            )
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=False,
            )

    def write(self, euler=None, acc=None, gyr=None, quat=None):
        e = euler if euler is not None else [0.0, 0.0, 0.0]
        a = acc   if acc   is not None else [0.0, 0.0, 0.0]
        g = gyr   if gyr   is not None else [0.0, 0.0, 0.0]
        q = quat  if quat  is not None else [0.0, 0.0, 0.0, 0.0]
        raw = struct.pack(_FMT, 1.0, *e, *a, *g, *q)
        self._shm.buf[:_SIZE] = raw

    def close(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class ImuShmReader:
    """Attach to an existing shared-memory block and read IMU data."""

    def __init__(self, label: str = "right"):
        self._name = _SHM_PREFIX + label
        self._shm = None

    def _attach(self):
        if self._shm is not None:
            return True
        try:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=False,
            )
            return True
        except FileNotFoundError:
            return False

    def read(self):
        """Return IMU dict or None if SHM not yet available / no data."""
        if not self._attach():
            return None
        try:
            vals = struct.unpack(_FMT, bytes(self._shm.buf[:_SIZE]))
        except Exception:
            return None
        if vals[0] < 0.5:      # valid flag
            return None
        return {
            "euler": list(vals[1:4]),
            "acc":   list(vals[4:7]),
            "gyr":   list(vals[7:10]),
            "quat":  list(vals[10:14]),
        }

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

"""IMU reader — lance OpenZenImuLogger via un PTY pour éviter les erreurs BT.

Deux instances pré-créées pour l'usage dans teleop :
    from _imu_reader import imu_right, imu_left

    imu_right.start()
    imu_left.start()
    data = imu_right.get_latest()   # {'euler':[r,p,y], 'quat':[w,x,y,z], 'acc':[x,y,z], 'gyr':[x,y,z]}
    imu_right.stop()

Usage module-level (rétrocompat) :
    import _imu_reader
    _imu_reader.start()
    data = _imu_reader.get_latest()
"""
import os
import pty
import re
import subprocess
import threading
import time
from typing import Optional

# ── Chemins ───────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_IMU_SCRIPT  = os.path.join(_PROJECT_DIR, "scripts", "imu_dual.sh")

# Cherche le binaire OpenZenImuLogger dans plusieurs emplacements connus
_LOGGER_BIN_CANDIDATES = [
    os.path.join(_PROJECT_DIR, "source", "imu", "build", "examples", "OpenZenImuLogger"),
    "/home/edgard/Desktop/AITEAM/Mina/source/imu/build/examples/OpenZenImuLogger",
    "/home/edgard/Desktop/openzen/build/examples/OpenZenImuLogger",
]
_LOGGER_BIN = next((p for p in _LOGGER_BIN_CANDIDATES if os.path.isfile(p)), "")

_MAC_RIGHT = "00:04:3E:5A:2B:61"
_MAC_LEFT  = "00:04:3E:6C:52:90"

# ── ANSI stripper ─────────────────────────────────────────────────────────────
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _parse_line(line: str) -> dict:
    line = _ANSI.sub("", line)
    out  = {}

    m = re.search(r"Acc\(m/s2\):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if m:
        out["acc"] = [float(m.group(i)) for i in (1, 2, 3)]

    m = re.search(r"Gyr\(d/s\):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if m:
        out["gyr"] = [float(m.group(i)) for i in (1, 2, 3)]

    m = re.search(r"Euler\(d\):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if m:
        out["euler"] = [float(m.group(i)) for i in (1, 2, 3)]

    m = re.search(r"Quat:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if m:
        out["quat"] = [float(m.group(i)) for i in (1, 2, 3, 4)]

    return out


# ── Classe IMUReader ───────────────────────────────────────────────────────────
class IMUReader:
    """Lecteur IMU indépendant pour un capteur LPMS-B2 donné (par adresse MAC)."""

    def __init__(self, mac: str, label: str = "IMU"):
        self._mac   = mac
        self._label = label
        self._lock  = threading.Lock()
        self._latest: dict = {
            "euler": None,
            "quat":  None,
            "acc":   None,
            "gyr":   None,
        }
        self._proc:   Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._ready       = threading.Event()
        self._stop_event  = threading.Event()

    # ── Internal loop ─────────────────────────────────────────────────────────
    def _reader_loop(self) -> None:
        subprocess.run(["pkill", "-f", "OpenZenImuLogger"],
                       capture_output=True, check=False)
        time.sleep(1.5)

        if not _LOGGER_BIN:
            print(f"[{self._label}] ⚠  OpenZenImuLogger introuvable — IMU désactivé")
            return

        cmd = [_LOGGER_BIN, "--sensor", "Bluetooth", self._mac, "--all"]
        print(f"[{self._label}] Lancement binaire PTY (MAC={self._mac})")

        master_fd, slave_fd = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
            os.close(slave_fd)

            with os.fdopen(master_fd, "r", encoding="utf-8", errors="replace") as master:
                for raw_line in master:
                    if self._stop_event.is_set():
                        break
                    parsed = _parse_line(raw_line.rstrip())
                    if parsed:
                        with self._lock:
                            self._latest.update(parsed)
                        if not self._ready.is_set():
                            print(f"[{self._label}] ✓ 1er paquet reçu")
                            self._ready.set()
        except OSError:
            pass
        except Exception as exc:
            print(f"[{self._label}] Erreur : {exc}")
        finally:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            print(f"[{self._label}] Subprocess terminé.")

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self) -> bool:
        if not _LOGGER_BIN:
            print(f"[{self._label}] ⚠  Binaire IMU introuvable — IMU désactivé")
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"imu-{self._label}"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def get_latest(self) -> dict:
        with self._lock:
            return {k: (list(v) if v is not None else None)
                    for k, v in self._latest.items()}

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)


# ── Instances pré-créées ───────────────────────────────────────────────────────
imu_right = IMUReader(mac=_MAC_RIGHT, label="IMU_R")
imu_left  = IMUReader(mac=_MAC_LEFT,  label="IMU_L")


# ── Rétrocompat module-level (ancienne API : _imu_reader.start() etc.) ────────
def start() -> bool:
    if not _LOGGER_BIN:
        print("[IMU] ⚠  Script/binaire IMU introuvable — IMU désactivé")
        return False
    return imu_right.start()

def stop() -> None:
    imu_right.stop()

def get_latest() -> dict:
    return imu_right.get_latest()

def wait_ready(timeout: float = 10.0) -> bool:
    return imu_right.wait_ready(timeout)

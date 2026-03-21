"""IMU reader — lance imu_dual.sh via un PTY pour éviter les erreurs de connexion BT.

Usage dans teleop :
    import _imu_reader
    _imu_reader.start()               # lance imu_dual.sh en interne
    data = _imu_reader.get_latest()   # {'euler':[r,p,y], 'acc':[x,y,z], ...}
    _imu_reader.stop()
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
_LOGGER_BIN  = os.path.join(
    _PROJECT_DIR, "source", "imu", "build", "examples", "OpenZenImuLogger"
)
_MAC1 = "00:04:3E:5A:2B:61"
_MAC2 = "00:04:3E:6C:52:90"

# ── Shared state ──────────────────────────────────────────────────────────────
_lock   = threading.Lock()
_latest: dict = {
    "euler": None,   # [roll, pitch, yaw]°
    "quat":  None,   # [w, x, y, z]
    "acc":   None,   # [x, y, z] m/s²
    "gyr":   None,   # [x, y, z] deg/s
}
_proc:   Optional[subprocess.Popen] = None
_thread: Optional[threading.Thread] = None
_ready  = threading.Event()
_stop_event = threading.Event()

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


def _reader_loop() -> None:
    global _proc

    # Tuer toute instance précédente d'OpenZenImuLogger (évite error: 9)
    subprocess.run(["pkill", "-f", "OpenZenImuLogger"],
                   capture_output=True, check=False)
    time.sleep(1.5)

    if os.path.isfile(_IMU_SCRIPT):
        cmd = ["bash", _IMU_SCRIPT, "--mac1", _MAC1, "--mac2", _MAC2, "--all"]
        print(f"[IMU] Lancement via imu_dual.sh (PTY)")
    elif os.path.isfile(_LOGGER_BIN):
        cmd = [_LOGGER_BIN, "--sensor", "Bluetooth", _MAC1, "--all"]
        print(f"[IMU] Lancement direct binaire (PTY)")
    else:
        print("[IMU] ⚠  Ni imu_dual.sh ni OpenZenImuLogger trouvés — IMU désactivé")
        return

    # Ouvrir un PTY : le subprocess croit avoir un vrai terminal
    master_fd, slave_fd = pty.openpty()
    try:
        _proc = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        os.close(slave_fd)

        with os.fdopen(master_fd, "r", encoding="utf-8", errors="replace") as master:
            for raw_line in master:
                if _stop_event.is_set():
                    break
                line   = raw_line.rstrip()
                parsed = _parse_line(line)
                if parsed:
                    with _lock:
                        _latest.update(parsed)
                    if not _ready.is_set():
                        print("[IMU] ✓ 1er paquet reçu")
                        _ready.set()
    except OSError:
        pass
    except Exception as exc:
        print(f"[IMU] Erreur : {exc}")
    finally:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _proc.kill()
        print("[IMU] Subprocess terminé.")


def start() -> bool:
    global _thread
    if not (os.path.isfile(_IMU_SCRIPT) or os.path.isfile(_LOGGER_BIN)):
        print("[IMU] ⚠  Script/binaire IMU introuvable — IMU désactivé")
        return False
    _stop_event.clear()
    _thread = threading.Thread(
        target=_reader_loop, daemon=True, name="imu-reader"
    )
    _thread.start()
    return True


def stop() -> None:
    _stop_event.set()
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    if _thread is not None:
        _thread.join(timeout=3)


def get_latest() -> dict:
    with _lock:
        return {k: (list(v) if v is not None else None) for k, v in _latest.items()}


def wait_ready(timeout: float = 10.0) -> bool:
    return _ready.wait(timeout)

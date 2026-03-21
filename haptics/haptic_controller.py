import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hardware.serial_api import SERIAL_API
from haptics.preset_library import FINGER_ADDRS

MOCK = False

class HapticController:
    def __init__(self, mock=MOCK):
        self.mock = mock
        self.api = SERIAL_API()
        self.connected = False
        self._lock = threading.Lock()

    def connect(self, port=None):
        if self.mock:
            print("[HapticController] MOCK mode")
            self.connected = True
            return True
        devices = self.api.get_serial_devices()
        # filter out bare ttyS* ports with no description
        real_devices = [d for d in devices if "ttyS" not in d or "n/a" not in d]
        if not real_devices:
            real_devices = devices  # fallback to all if nothing else found
        if not real_devices:
            print("[HapticController] No serial devices found")
            return False
        target = next((d for d in real_devices if port in d), real_devices[0]) if port else real_devices[0]
        self.connected = self.api.connect_serial_device(target)
        return self.connected

    def disconnect(self):
        self.stop_all()
        if not self.mock:
            self.api.disconnect_serial_device()
        self.connected = False

    def send(self, addr, duty, freq, wave=1, start=True):
        duty = max(0, min(31, int(duty)))
        freq = max(0, min(7,  int(freq)))
        if self.mock:
            finger = next((f for f, a in FINGER_ADDRS.items() if a == addr), f"addr{addr}")
            print(f"[MOCK] {'ON ' if start else 'OFF'} | {finger:6s} | duty={duty:2d} freq={freq} wave={'sine' if wave else 'sq'}")
            return True
        with self._lock:
            return self.api.send_command(addr, duty, freq, start_or_stop=1 if start else 0, wave=wave)

    def send_all_fingers(self, duty, freq, wave=1):
        duty = max(0, min(31, int(duty)))
        freq = max(0, min(7,  int(freq)))
        if self.mock:
            print(f"[MOCK] ALL | duty={duty:2d} freq={freq} wave={'sine' if wave else 'sq'}")
            return True
        cmds = [{"addr": a, "duty": duty, "freq": freq, "start_or_stop": 1 if duty > 0 else 0, "wave": wave}
                for a in FINGER_ADDRS.values()]
        with self._lock:
            return self.api.send_command_list(cmds)

    def send_finger(self, finger, duty, freq, wave=1):
        addr = FINGER_ADDRS.get(finger)
        if addr is None:
            return False
        return self.send(addr, duty, freq, wave, start=duty > 0)

    def stop_all(self):
        if self.mock:
            print("[MOCK] STOP ALL")
            return
        cmds = [{"addr": a, "duty": 0, "freq": 0, "start_or_stop": 0, "wave": 0}
                for a in FINGER_ADDRS.values()]
        with self._lock:
            self.api.send_command_list(cmds)

    def pulse(self, addr, duty, freq, wave, duration_ms):
        def _p():
            self.send(addr, duty, freq, wave, start=True)
            time.sleep(duration_ms / 1000.0)
            self.send(addr, 0, freq, wave, start=False)
        threading.Thread(target=_p, daemon=True).start()

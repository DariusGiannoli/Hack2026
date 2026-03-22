"""
haptic_pipeline.py — Full 3-layer neural haptic rendering pipeline.

Layer 0 (once):     GPT-4V  → object ID + fragility cap
Layer 1 (3 Hz):     DINO MLP → material class
                    DINO → NeuralHapticRenderer → neural freq/wave (texture-aware)
Layer 2 (50 Hz):    HapticNet → neural slip detection + global duty hint
                    Weber-Fechner → per-finger psychophysical duty
                    Contact transients → FA-I pulse on finger onset
                    Blend: neural + per-finger + VLM cap + slip override

Usage:
    # Full pipeline (robot + glove + GPT-4V)
    python haptic_pipeline.py --iface enp131s0

    # Both hands
    python haptic_pipeline.py --iface enp131s0 --hand both

    # No VLM (skip GPT-4V, use DINO MLP only)
    python haptic_pipeline.py --iface enp131s0 --no-vlm

    # No neural backend (just Weber-Fechner + DINO presets)
    python haptic_pipeline.py --iface enp131s0 --no-neural

    # Mock everything (no robot, no glove)
    python haptic_pipeline.py --mock-esp --mock-hand

    # Test with a photo instead of live camera
    python haptic_pipeline.py --mock-hand --mock-esp --photo data/photos/smooth/IMG_5906.HEIC
"""

import sys, os, time, math, argparse, threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Inspire DDS paths
_SDK = os.path.expanduser("~/GR00T-WholeBodyControl/inspire_hand_ws/inspire_hand_sdk")
_DDS = os.path.join(_SDK, "inspire_sdkpy")
for _p in [_SDK, _DDS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from haptics.controller import HapticController
from haptics.presets import FINGER_ADDRS, ANCHORS

# ── Config ────────────────────────────────────────────────────────────────────

CONTROL_HZ   = 50     # main haptic loop
DINO_HZ      = 3      # visual material classification rate
TACTILE_MAX  = 100.0
DEAD_ZONE    = 0.08

# Weber-Fechner
WF_K  = 0.55
WF_P0 = 0.3
MAX_DUTY = 31

# Finger mapping
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb"]
N_FINGERS = 5

# Slip detection (rule-based fallback)
SLIP_TORQUE_THRESH   = 1.5   # Nm — torque derivative threshold
SLIP_PRESSURE_THRESH = -0.05 # pressure dropping while torque rises
SLIP_WINDOW          = 5     # frames lookback
SLIP_DUTY_BOOST      = 10
SLIP_FREQ            = 7
SLIP_WAVE            = 0     # square wave = more alarming

# Contact transient (FA-I mechanoreceptor simulation)
CONTACT_ONSET_THRESH = 0.15  # normalized pressure jump to trigger pulse
CONTACT_MIN_PRESSURE = 0.10  # minimum pressure for contact detection
CONTACT_RELEASE_THRESH = 0.03
CONTACT_PULSE_DUTY   = 28
CONTACT_PULSE_FREQ   = 6
CONTACT_PULSE_WAVE   = 0     # square = sharper impact feel
CONTACT_PULSE_MS     = 25    # short burst

# Neural blend weight (how much HapticNet influences vs pure Weber-Fechner)
NEURAL_BLEND = 0.3  # 30% HapticNet duty_raw, 70% per-finger Weber-Fechner

# Material class → haptic preset
MATERIAL_PRESETS = {
    "smooth": {"freq": 6, "wave": 0, "name": "smooth metal"},    # high freq, square
    "rough":  {"freq": 3, "wave": 1, "name": "rough fabric"},    # low freq, sine
    "soft":   {"freq": 1, "wave": 1, "name": "soft foam"},       # very low, sine
}
DEFAULT_PRESET = {"freq": 3, "wave": 1, "name": "default"}

# Inspire DDS touch fields (same for left and right — topic determines hand)
TIP_FIELDS = [
    "fingerone_tip_touch", "fingertwo_tip_touch", "fingerthree_tip_touch",
    "fingerfour_tip_touch", "fingerfive_tip_touch", None,
]
PALM_FIELDS = [
    "fingerone_palm_touch", "fingertwo_palm_touch", "fingerthree_palm_touch",
    "fingerfour_palm_touch", "fingerfive_palm_touch", None,
]

# G1 arm joint indices for torque
ARM_JOINTS_LEFT  = list(range(15, 21))  # left arm: 6 joints
ARM_JOINTS_RIGHT = list(range(22, 28))  # right arm: 6 joints


def pressure_to_duty(p: float, cap: int = MAX_DUTY) -> int:
    if p < DEAD_ZONE:
        return 0
    p_adj = p - DEAD_ZONE
    raw = WF_K * math.log(1.0 + p_adj / WF_P0)
    return max(0, min(cap, int(raw * cap)))


# ── Contact transient detector (FA-I mechanoreceptor) ─────────────────────────

class ContactDetector:
    """Detects per-finger contact onset and fires short pulses.

    Mimics fast-adapting type I (FA-I) mechanoreceptors that respond
    to initial contact and release but not sustained pressure.
    """

    def __init__(self, controller: HapticController):
        self._ctrl = controller
        self._prev = np.zeros(N_FINGERS, dtype=np.float32)
        self._in_contact = np.zeros(N_FINGERS, dtype=bool)

    def step(self, pressure_5: list):
        """Check for contact onset/release on each finger. Fire pulses on onset."""
        p = np.array(pressure_5, dtype=np.float32)

        for i in range(N_FINGERS):
            dp = p[i] - self._prev[i]

            # onset: pressure jump above threshold while not already in contact
            if (dp > CONTACT_ONSET_THRESH and p[i] > CONTACT_MIN_PRESSURE
                    and not self._in_contact[i]):
                self._in_contact[i] = True
                addr = FINGER_ADDRS[FINGER_NAMES[i]]
                self._ctrl.pulse(addr, CONTACT_PULSE_DUTY, CONTACT_PULSE_FREQ,
                                 CONTACT_PULSE_WAVE, CONTACT_PULSE_MS)

            # release: pressure drops below threshold
            if p[i] < CONTACT_RELEASE_THRESH:
                self._in_contact[i] = False

        self._prev = p.copy()


# ── Rule-based slip detector (fallback) ──────────────────────────────────────

class SlipDetector:
    def __init__(self):
        self._torque_hist = []
        self._pressure_hist = []

    def step(self, pressure_5: list, torque_12: list) -> float:
        """Returns slip probability [0, 1]."""
        t = np.array(torque_12, dtype=np.float32)
        p = np.array(pressure_5, dtype=np.float32)

        self._torque_hist.append(t)
        self._pressure_hist.append(p)

        if len(self._torque_hist) < SLIP_WINDOW + 1:
            return 0.0

        self._torque_hist = self._torque_hist[-(SLIP_WINDOW + 1):]
        self._pressure_hist = self._pressure_hist[-(SLIP_WINDOW + 1):]

        dt = np.mean(np.abs(self._torque_hist[-1] - self._torque_hist[-SLIP_WINDOW])) / SLIP_WINDOW
        dp = np.mean(self._pressure_hist[-1] - self._pressure_hist[-SLIP_WINDOW]) / SLIP_WINDOW

        if dt > SLIP_TORQUE_THRESH and dp < SLIP_PRESSURE_THRESH:
            return min(1.0, dt / (SLIP_TORQUE_THRESH * 2))
        return 0.0


# ── Neural backend ───────────────────────────────────────────────────────────

class NeuralBackend:
    """Wraps HapticNet (Conv1D+LSTM) for neural haptic prediction.

    Uses the trained model at models/hapticnet.pt to predict:
    - freq (0-7), duty_raw (0-1), wave (0/1), slip (0-1)
    from rolling windows of pressure + torque + fragility.

    Falls back gracefully if model not found.
    """

    def __init__(self):
        self._inference = None
        self._ready = False

    def load(self, model_path="models/hapticnet.pt", device="cpu"):
        if not os.path.exists(model_path):
            print(f"[neural] {model_path} not found — using rule-based fallback")
            return False
        try:
            from neural.hapticnet import HapticNetInference
            self._inference = HapticNetInference(model_path, device)
            self._ready = True
            print(f"[neural] HapticNet loaded from {model_path}")
            return True
        except Exception as e:
            print(f"[neural] Failed to load HapticNet: {e}")
            return False

    def set_fragility(self, frag_scalar: float):
        if self._inference:
            self._inference.set_fragility(frag_scalar)

    def reset(self):
        if self._inference:
            self._inference.reset()

    def step(self, pressure_6: list, torque_12: list) -> dict:
        """Returns neural predictions or None if not available."""
        if not self._ready:
            return None
        return self._inference.step(pressure_6, torque_12)

    @property
    def ready(self):
        return self._ready


class TextureRenderer:
    """Wraps DINO → MLP backbone → LSTM renderer for texture-aware freq/wave.

    Uses models/haptic_mlp.pt (frozen backbone) + models/haptic_lstm.pt.
    Receives DINO embeddings at 3 Hz, outputs freq/wave at 50 Hz.

    Falls back gracefully if models not found.
    """

    def __init__(self):
        self._inference = None
        self._ready = False
        self._prev_torque_norm = 0.0

    def load(self, mlp_path="models/haptic_mlp.pt",
             lstm_path="models/haptic_lstm.pt", device="cpu"):
        if not (os.path.exists(mlp_path) and os.path.exists(lstm_path)):
            missing = [p for p in [mlp_path, lstm_path] if not os.path.exists(p)]
            print(f"[texture] Missing: {missing} — using preset lookup")
            return False
        try:
            from haptics.models.lstm import HapticInference
            self._inference = HapticInference.from_checkpoints(mlp_path, lstm_path, device)
            self._ready = True
            print(f"[texture] LSTM renderer loaded")
            return True
        except Exception as e:
            print(f"[texture] Failed to load LSTM renderer: {e}")
            return False

    def set_material(self, dino_embedding: np.ndarray):
        """Feed a fresh 384-dim DINO embedding (called at 3 Hz)."""
        if self._inference:
            self._inference.set_material(dino_embedding)

    def reset(self):
        if self._inference:
            self._inference.reset()
        self._prev_torque_norm = 0.0

    def step(self, torque_sum: float) -> dict:
        """Returns texture-aware freq/wave/duty_raw or None."""
        if not self._ready:
            return None
        # normalize torque
        torque_norm = max(0.0, min(1.0, (torque_sum - 2.0) / 23.0))
        d_torque = torque_norm - self._prev_torque_norm
        contact = float(torque_norm > 0.05)
        self._prev_torque_norm = torque_norm

        return self._inference.step(torque_norm, d_torque, contact)

    @property
    def ready(self):
        return self._ready


# ── Hand readers ─────────────────────────────────────────────────────────────

FORCE_ACT_MAX = 600.0

class MockHand:
    def __init__(self):
        self._t0 = time.time()

    def tip_pressure(self) -> list:
        t = time.time() - self._t0
        return [max(0, 40 * math.sin(0.3 * t + i * 0.7) + 20 + np.random.normal(0, 2))
                for i in range(6)]

    def torque_12(self) -> list:
        return [np.random.normal(0, 0.5) for _ in range(12)]


class RealHandReader:
    """DDS subscriber for Inspire hand tactile + G1 arm torques.

    Parameters
    ----------
    side : str
        "right" or "left" — determines DDS topics and arm joint indices.
    """

    def __init__(self, side: str = "right"):
        self.side = side
        self._lock = threading.Lock()
        self._tips = [0.0] * 6
        self._palms = [0.0] * 6
        self._force = [0] * 6
        self._fresh = False
        self._torque = [0.0] * 12

    def connect(self, interface=None, skip_dds_init=False):
        from inspire_dds import inspire_hand_touch, inspire_hand_state
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

        if not skip_dds_init:
            if interface:
                ChannelFactoryInitialize(0, interface)
            else:
                ChannelFactoryInitialize(0)

        suffix = "r" if self.side == "right" else "l"

        sub_touch = ChannelSubscriber(f"rt/inspire_hand/touch/{suffix}", inspire_hand_touch)
        sub_touch.Init(self._on_touch, 10)
        self._sub_touch = sub_touch

        sub_state = ChannelSubscriber(f"rt/inspire_hand/state/{suffix}", inspire_hand_state)
        sub_state.Init(self._on_state, 10)
        self._sub_state = sub_state

        # G1 lowstate for arm torques
        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            sub_low = ChannelSubscriber("rt/lowstate", LowState_)
            sub_low.Init(self._on_lowstate, 10)
            self._sub_low = sub_low
        except Exception:
            pass

        print(f"[pipeline] Subscribed to {self.side} hand DDS topics — waiting...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._fresh:
                print(f"[pipeline] Receiving {self.side} hand data")
                return True
            time.sleep(0.1)
        print(f"[pipeline] Timeout on {self.side} hand — is Headless_driver running?")
        return False

    def _on_touch(self, msg):
        tips, palms = [], []
        for tip_f, palm_f in zip(TIP_FIELDS, PALM_FIELDS):
            tips.append(float(np.mean(getattr(msg, tip_f))) if tip_f else 0.0)
            palms.append(float(np.mean(getattr(msg, palm_f))) if palm_f else 0.0)
        with self._lock:
            self._tips = tips
            self._palms = palms
            self._fresh = True

    def _on_state(self, msg):
        with self._lock:
            self._force = list(msg.force_act)
            self._fresh = True

    def _on_lowstate(self, msg):
        # Both arms: left (15-21) + right (22-28)
        torques = []
        for idx in ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT:
            torques.append(float(msg.motor_state[idx].tau_est))
        with self._lock:
            self._torque = torques

    def tip_pressure(self) -> list:
        with self._lock:
            tips = list(self._tips)
            palms = list(self._palms)
            force = list(self._force)
        blended = [max(t, p) for t, p in zip(tips, palms)]
        for i in range(min(len(blended), len(force))):
            if blended[i] < 2.0 and abs(force[i]) > 10:
                blended[i] = min(TACTILE_MAX, abs(float(force[i])) / FORCE_ACT_MAX * TACTILE_MAX)
        return blended

    def torque_12(self) -> list:
        with self._lock:
            return list(self._torque)


# ── DINO MLP material classifier ─────────────────────────────────────────────

class MaterialClassifier:
    """Runs DINO + MLP at 3 Hz in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._material = "smooth"
        self._confidence = 0.0
        self._preset = MATERIAL_PRESETS["smooth"]
        self._ready = False
        self._changed = False
        self._last_frame = None
        self._last_embedding = None  # cached DINO embedding for LSTM renderer

    def load(self):
        import torch
        from haptics.perception.dino_encoder import DinoEncoder
        from haptics.training.train_dino_mlp import MaterialMLP

        self._encoder = DinoEncoder()
        self._encoder.load()

        ckpt = torch.load("models/dino_mlp.pt", map_location="cpu", weights_only=False)
        self._mlp = MaterialMLP()
        self._mlp.load_state_dict(ckpt["model"])
        self._mlp.eval()
        self._class_names = ckpt["class_names"]
        self._torch = torch
        self._ready = True
        print("[pipeline] MaterialClassifier ready")

    def classify(self, frame_bgr):
        """Run DINO + MLP on a single frame. Thread-safe."""
        if not self._ready:
            return
        emb = self._encoder.encode(frame_bgr)
        if emb is None:
            return
        with self._torch.no_grad():
            logits = self._mlp(self._torch.tensor(emb).unsqueeze(0))
            probs = self._torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        name = self._class_names[idx]
        conf = float(probs[idx])

        preset = MATERIAL_PRESETS.get(name, DEFAULT_PRESET)
        with self._lock:
            if name != self._material and conf > 0.7:
                self._changed = True
                self._last_frame = frame_bgr.copy()
                print(f"[DINO] Material changed: {self._material} → {name} ({conf:.0%})")
            self._material = name
            self._confidence = conf
            self._preset = preset
            self._last_embedding = emb  # cache for LSTM renderer

    def pop_change(self):
        """Returns (changed, frame) and resets the flag."""
        with self._lock:
            if self._changed:
                self._changed = False
                return True, self._last_frame
            return False, None

    @property
    def embedding(self):
        """Latest DINO embedding (384-dim) for LSTM renderer."""
        with self._lock:
            return self._last_embedding

    @property
    def state(self):
        with self._lock:
            return {
                "material": self._material,
                "confidence": self._confidence,
                "preset": dict(self._preset),
            }


# ── GPT-4V scene seeder ──────────────────────────────────────────────────────

def run_vlm(frame_bgr, backend="gpt4v"):
    """One-shot VLM call. Returns scene dict or None."""
    try:
        from haptics.perception.scene_seeder import SceneSeeder
        seeder = SceneSeeder(backend=backend)
        return seeder.seed(frame_bgr)
    except Exception as e:
        print(f"[pipeline] VLM failed: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full haptic rendering pipeline")
    parser.add_argument("--iface", default="enp131s0")
    parser.add_argument("--mock-esp", action="store_true")
    parser.add_argument("--mock-hand", action="store_true")
    parser.add_argument("--port", default=None)
    parser.add_argument("--no-vlm", action="store_true", help="Skip GPT-4V scene analysis")
    parser.add_argument("--vlm", default="gpt4v", choices=["gpt4v", "moondream", "gemma"])
    parser.add_argument("--photo", default=None, help="Use a static photo for DINO/VLM")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device index for live DINO")
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE)
    parser.add_argument("--hand", default="right", choices=["left", "right", "both"],
                        help="Which hand(s) to control")
    parser.add_argument("--no-neural", action="store_true",
                        help="Disable neural backends (HapticNet + LSTM renderer)")
    args = parser.parse_args()

    # update module-level dead zone
    import haptic_pipeline
    haptic_pipeline.DEAD_ZONE = args.dead_zone

    # ── sensor source ──────────────────────────────────────────────────────
    hands = {}  # side → hand reader
    sides = ["left", "right"] if args.hand == "both" else [args.hand]

    if args.mock_hand:
        for side in sides:
            hands[side] = MockHand()
        print(f"[pipeline] Using MOCK hand(s): {sides}")
    else:
        for i, side in enumerate(sides):
            reader = RealHandReader(side=side)
            # Only pass interface on first reader — DDS can only init once
            iface = args.iface if i == 0 else None
            ok = reader.connect(interface=iface, skip_dds_init=(i > 0))
            if ok:
                hands[side] = reader
            else:
                print(f"[pipeline] {side} hand timeout — falling back to mock")
                hands[side] = MockHand()

    # ── haptic output ──────────────────────────────────────────────────────
    hc = HapticController(mock=args.mock_esp)
    if not hc.connect(port=args.port):
        hc = HapticController(mock=True)
        hc.connect()

    # ── Layer 1: DINO MLP ──────────────────────────────────────────────────
    classifier = MaterialClassifier()
    try:
        classifier.load()
    except Exception as e:
        print(f"[pipeline] MaterialClassifier failed: {e} — using defaults")

    # ── Neural backends (optional) ─────────────────────────────────────────
    neural = NeuralBackend()
    texture = TextureRenderer()

    if not args.no_neural:
        neural.load()
        texture.load()
    else:
        print("[pipeline] Neural backends disabled (--no-neural)")

    # ── Load test frame for DINO / VLM ─────────────────────────────────────
    test_frame = None
    webcam_cap = None

    if args.photo:
        from PIL import Image
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        img = Image.open(args.photo).convert("RGB")
        test_frame = np.array(img)[:, :, ::-1].copy()
        print(f"[pipeline] Using photo: {args.photo}")

    if args.webcam is not None:
        import cv2
        webcam_cap = cv2.VideoCapture(args.webcam)
        if webcam_cap.isOpened():
            ret, frame = webcam_cap.read()
            if ret:
                test_frame = frame
                print(f"[pipeline] Webcam {args.webcam} connected ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"[pipeline] Webcam {args.webcam} can't read — ignoring")
                webcam_cap = None
        else:
            print(f"[pipeline] Webcam {args.webcam} can't open — ignoring")
            webcam_cap = None

    # ── Layer 0: VLM scene analysis (once) ─────────────────────────────────
    frag_cap = MAX_DUTY
    object_name = "unknown"
    frag_scalar = 0.5  # for HapticNet (0=indestructible, 1=glass)

    if not args.no_vlm and test_frame is not None:
        scene = run_vlm(test_frame, backend=args.vlm)
        if scene and scene.get("objects"):
            obj = scene["objects"][0]
            object_name = obj.get("name", "unknown")
            frag = int(obj.get("fragility", 3))
            frag_cap = {1: 31, 2: 28, 3: 22, 4: 14, 5: 8}.get(frag, 22)
            frag_scalar = (frag - 1) / 4.0
            print(f"[pipeline] VLM → {object_name} | fragility={frag} | cap={frag_cap}")

            # propagate to neural backends
            neural.set_fragility(frag_scalar)

    # ── Initial DINO classification ────────────────────────────────────────
    if test_frame is not None:
        classifier.classify(test_frame)
        mat = classifier.state
        print(f"[pipeline] DINO → {mat['material']} ({mat['confidence']:.0%})")

        # feed initial DINO embedding to texture renderer
        emb = classifier.embedding
        if emb is not None:
            texture.set_material(emb)

    # ── Background DINO thread (3 Hz from webcam) ──────────────────────────
    dino_running = True
    def dino_loop():
        """Continuously classify webcam frames at DINO_HZ."""
        interval = 1.0 / DINO_HZ
        while dino_running and webcam_cap is not None:
            t0 = time.time()
            ret, frame = webcam_cap.read()
            if ret:
                classifier.classify(frame)
                # feed embedding to texture renderer
                emb = classifier.embedding
                if emb is not None:
                    texture.set_material(emb)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    if webcam_cap is not None:
        dino_thread = threading.Thread(target=dino_loop, daemon=True)
        dino_thread.start()
        print(f"[pipeline] DINO live classification at {DINO_HZ} Hz from webcam")

    # ── Per-hand state ─────────────────────────────────────────────────────
    slip_detectors = {side: SlipDetector() for side in sides}
    contact_detectors = {side: ContactDetector(hc) for side in sides}

    # ── Print banner ───────────────────────────────────────────────────────
    mat = classifier.state
    neural_str = "HapticNet" if neural.ready else "off"
    texture_str = "LSTM renderer" if texture.ready else "preset lookup"
    hand_str = " + ".join(sides)

    print(f"\n{'='*70}")
    print(f"  HapticNet Pipeline — {CONTROL_HZ} Hz")
    print(f"  Hand(s):  {hand_str}")
    print(f"  Object:   {object_name} (cap={frag_cap})")
    print(f"  Material: {mat['material']} → freq={mat['preset']['freq']} wave={mat['preset']['wave']}")
    print(f"  Texture:  {texture_str}")
    print(f"  Neural:   {neural_str}")
    print(f"  Slip:     {'HapticNet neural' if neural.ready else 'rule-based (torque+pressure)'}")
    print(f"  Contact:  FA-I transient pulses (onset detection)")
    print(f"  Ctrl+C to stop")
    print(f"{'='*70}\n")

    # ── Background VLM re-trigger on material change ─────────────────────
    vlm_lock = threading.Lock()

    def vlm_retrigger(frame_bgr):
        """Called in background when DINO detects material switch."""
        nonlocal frag_cap, object_name, frag_scalar
        scene = run_vlm(frame_bgr, backend=args.vlm)
        if scene and scene.get("objects"):
            obj = scene["objects"][0]
            name = obj.get("name", "unknown")
            frag = int(obj.get("fragility", 3))
            cap = {1: 31, 2: 28, 3: 22, 4: 14, 5: 8}.get(frag, 22)
            scalar = (frag - 1) / 4.0
            with vlm_lock:
                frag_cap = cap
                object_name = name
                frag_scalar = scalar
            neural.set_fragility(scalar)
            neural.reset()
            texture.reset()
            print(f"[VLM] New object → {name} | fragility={frag} | cap={cap}")

    # ── 50 Hz control loop ─────────────────────────────────────────────────
    interval = 1.0 / CONTROL_HZ
    n = 0
    last_status = {}  # per-side state for status reporting

    try:
        while True:
            t0 = time.time()

            for side in sides:
                hand = hands[side]
                slip_det = slip_detectors[side]
                contact_det = contact_detectors[side]

                # read sensors
                raw_tips = hand.tip_pressure()
                torque = hand.torque_12() if hasattr(hand, 'torque_12') else [0.0] * 12

                # normalize pressure [0, 1]
                norm = [max(0.0, min(1.0, v / TACTILE_MAX)) for v in raw_tips]
                pressure_5 = [norm[0], norm[1], norm[2], norm[3], max(norm[4], norm[5])]

                # ── contact transient detection (FA-I pulses) ──────────
                contact_det.step(pressure_5)

                # ── check DINO material change → re-trigger VLM ────────
                if not args.no_vlm:
                    changed, change_frame = classifier.pop_change()
                    if changed and change_frame is not None:
                        threading.Thread(target=vlm_retrigger, args=(change_frame,), daemon=True).start()

                # ── get current state ──────────────────────────────────
                with vlm_lock:
                    current_cap = frag_cap
                mat = classifier.state

                # ── determine freq and wave ────────────────────────────
                # Priority: neural texture renderer > DINO preset > default
                neural_out = None
                freq = mat["preset"]["freq"]
                wave = mat["preset"]["wave"]

                # try DINO→LSTM texture renderer (texture-aware freq/wave)
                if texture.ready:
                    torque_sum = sum(abs(t) for t in torque)
                    tex_out = texture.step(torque_sum)
                    if tex_out:
                        freq = tex_out["freq"]
                        wave = tex_out["wave"]

                # try HapticNet for slip + duty hint
                # HapticNet expects pressure in [0,1] (same as record_data.py output)
                if neural.ready:
                    neural_out = neural.step(norm[:6], torque)

                # ── slip detection ─────────────────────────────────────
                slip_prob = 0.0
                if neural_out and neural.ready:
                    # prefer neural slip
                    slip_prob = neural_out["slip"]
                else:
                    # fall back to rule-based
                    slip_prob = slip_det.step(pressure_5, torque)

                slip_active = slip_prob > 0.6

                if slip_active:
                    freq = SLIP_FREQ
                    wave = SLIP_WAVE

                # ── per-finger duty (Weber-Fechner + optional neural blend) ─
                duties = []
                for p in pressure_5:
                    weber_duty = pressure_to_duty(p, current_cap)

                    if neural_out:
                        # blend: 70% per-finger Weber, 30% HapticNet global duty
                        neural_duty = int(neural_out["duty_raw"] * current_cap)
                        blended = int(0.7 * weber_duty + 0.3 * neural_duty)
                        duties.append(max(0, min(31, blended)))
                    else:
                        duties.append(weber_duty)

                if slip_active:
                    duties = [min(31, d + SLIP_DUTY_BOOST) for d in duties]

                # ── send to each LRA ───────────────────────────────────
                for finger, duty in zip(FINGER_NAMES, duties):
                    hc.send_finger(finger, duty, freq, wave)

                # save per-side state for status line
                last_status[side] = {
                    "duties": duties, "freq": freq, "wave": wave,
                    "slip_prob": slip_prob, "cap": current_cap,
                }

            # ── status every 1s ────────────────────────────────────────
            n += 1
            if n % CONTROL_HZ == 0:
                mat = classifier.state
                src = "neural" if neural.ready else "rules"
                for side in sides:
                    st = last_status.get(side, {})
                    d5 = st.get("duties", [0]*5)
                    bars = "  ".join(
                        f"{name[:2]}={d5[i]:2d}"
                        for i, name in enumerate(FINGER_NAMES)
                    )
                    sp = st.get("slip_prob", 0.0)
                    slip_str = " SLIP!" if sp > 0.6 else ""
                    side_tag = f" ({side[0]})" if len(sides) > 1 else ""
                    print(f"  [{mat['material']:6s} {mat['confidence']:.0%}] "
                          f"freq={st.get('freq',0)} wave={st.get('wave',0)} "
                          f"cap={st.get('cap', frag_cap)} "
                          f"slip={src} obj={object_name} | {bars}{slip_str}{side_tag}")

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    except KeyboardInterrupt:
        print("\nStopping...")
        dino_running = False
        hc.stop_all()
        hc.disconnect()
        if webcam_cap is not None:
            webcam_cap.release()
        print("Done")


if __name__ == "__main__":
    main()

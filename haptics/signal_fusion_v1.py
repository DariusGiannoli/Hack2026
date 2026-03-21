# haptics/signal_fusion_v1.py
# V1: GPT-4V preset + fingertip force → LRA command
# No DINOv2, no MLP, no training needed.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from haptics.preset_library import ANCHORS

# Preset params per material
PRESETS = {
    "smooth metal":  {"freq": 6, "wave": 0},
    "rigid plastic": {"freq": 5, "wave": 0},
    "glass":         {"freq": 7, "wave": 0},
    "rough fabric":  {"freq": 3, "wave": 1},
    "soft foam":     {"freq": 1, "wave": 1},
    "human skin":    {"freq": 2, "wave": 1},
    "wood":          {"freq": 4, "wave": 0},
    "rubber":        {"freq": 3, "wave": 1},
    "cardboard":     {"freq": 4, "wave": 1},
    "default":       {"freq": 3, "wave": 1},
}


class SignalFusionV1:
    def __init__(self):
        self._preset        = PRESETS["default"].copy()
        self._fragility_cap = 31
        self._object_name   = "unknown"
        self._active        = False

    def update_from_scene(self, scene: dict, target_object: str = None):
        """
        Call once after GPT-4V scene seed.
        target_object: name of object to interact with.
                       If None, uses first object in scene.
        """
        objects = scene.get("objects", [])
        if not objects:
            print("[FusionV1] No objects in scene, using defaults")
            return

        # pick target object
        if target_object:
            obj = next((o for o in objects
                        if target_object.lower() in o["name"].lower()), objects[0])
        else:
            obj = objects[0]

        self._object_name   = obj.get("name", "unknown")
        preset_key          = obj.get("haptic_preset", "default")
        self._preset        = PRESETS.get(preset_key, PRESETS["default"]).copy()
        self._fragility_cap = scene["fragility_caps"].get(self._object_name, 31)
        self._active        = True

        print(f"[FusionV1] Loaded: {self._object_name}")
        print(f"           preset={preset_key} "
              f"freq={self._preset['freq']} "
              f"wave={self._preset['wave']} "
              f"fragility_cap={self._fragility_cap}")

    def compute(self, force: float) -> dict:
        """
        force: 0.0-1.0 normalized fingertip force
        Returns: {freq, duty, wave} ready to send to ESP32
        """
        if not self._active:
            return {"freq": 0, "duty": 0, "wave": 0}

        # force drives duty, capped by fragility
        duty = int(force * self._fragility_cap)
        duty = max(0, min(self._fragility_cap, duty))

        return {
            "freq": self._preset["freq"],
            "duty": duty,
            "wave": self._preset["wave"],
        }

    def compute_mock(self, force: float) -> dict:
        """Same as compute() but prints instead of sending."""
        result = self.compute(force)
        print(f"[FusionV1] force={force:.2f} → "
              f"freq={result['freq']} duty={result['duty']} "
              f"wave={'sine' if result['wave'] else 'square'}")
        return result

    @property
    def object_name(self):
        return self._object_name

    @property
    def fragility_cap(self):
        return self._fragility_cap
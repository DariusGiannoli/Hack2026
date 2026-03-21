ANCHORS = {
    "smooth metal":  {"freq": 6, "duty": 22, "wave": 0},
    "rigid plastic": {"freq": 5, "duty": 20, "wave": 0},
    "glass":         {"freq": 7, "duty": 18, "wave": 0},
    "rough fabric":  {"freq": 3, "duty": 24, "wave": 1},
    "soft foam":     {"freq": 1, "duty": 16, "wave": 1},
    "human skin":    {"freq": 2, "duty": 14, "wave": 1},
    "wood":          {"freq": 4, "duty": 20, "wave": 0},
    "rubber":        {"freq": 3, "duty": 18, "wave": 1},
    "cardboard":     {"freq": 4, "duty": 16, "wave": 0},
    "default":       {"freq": 3, "duty": 18, "wave": 1},
}

FINGER_ADDRS = {
    "thumb":  0,
    "index":  1,
    "middle": 2,
    "ring":   3,
    "pinky":  4,
}

CONTACT_EVENTS = {
    "impact":  {"duty_boost": 9, "freq_override": 7, "wave": 0, "duration_ms": 80},
    "slip":    {"duty_boost": 6, "freq_override": 6, "wave": 0, "duration_ms": 200},
    "edge":    {"duty_boost": 4, "freq_override": 5, "wave": 0, "duration_ms": 40},
    "hold":    {"duty_boost": 0, "freq_override": None, "wave": None, "duration_ms": 0},
    "release": {"duty_boost": 0, "freq_override": 0, "wave": 0, "duration_ms": 30},
}

WEIGHT_CLASSES = {
    "none":   {"duty_offset": 0,  "freq": 0, "pulse_hz": 0.0},
    "light":  {"duty_offset": 0,  "freq": 0, "pulse_hz": 0.0},
    "medium": {"duty_offset": 6,  "freq": 1, "pulse_hz": 0.8},
    "heavy":  {"duty_offset": 10, "freq": 2, "pulse_hz": 0.4},
}

class PresetLibrary:
    def get(self, label):
        return ANCHORS.get(label, ANCHORS["default"]).copy()
    def get_default(self):
        return ANCHORS["default"].copy()
    def get_contact_event(self, event):
        return CONTACT_EVENTS.get(event, CONTACT_EVENTS["hold"]).copy()
    def get_weight(self, weight_class):
        return WEIGHT_CLASSES.get(weight_class, WEIGHT_CLASSES["none"]).copy()
    def all_labels(self):
        return [k for k in ANCHORS if k != "default"]
    def all_anchors(self):
        return {k: v for k, v in ANCHORS.items() if k != "default"}

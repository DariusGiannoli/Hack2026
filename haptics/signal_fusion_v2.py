import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class SignalFusionV2:
    def __init__(self):
        self._fragility_cap = 31
        self._object_name   = "unknown"
        self._active        = False

    def update_from_scene(self, scene, target_object=None):
        objects = scene.get("objects", [])
        if not objects:
            return
        obj = objects[0]
        if target_object:
            obj = next((o for o in objects if target_object.lower() in o["name"].lower()), objects[0])
        self._object_name   = obj.get("name", "unknown")
        self._fragility_cap = scene["fragility_caps"].get(self._object_name, 31)
        self._active        = True
        print(f"[FusionV2] {self._object_name} | cap={self._fragility_cap}")

    def compute(self, force, mlp_params):
        if not self._active:
            return {"freq": 0, "duty": 0, "wave": 0}
        freq = mlp_params.get("freq", 3)
        wave = mlp_params.get("wave", 1)
        duty = max(0, min(self._fragility_cap, int(force * self._fragility_cap)))
        return {"freq": freq, "duty": duty, "wave": wave}

    @property
    def object_name(self): return self._object_name
    @property
    def fragility_cap(self): return self._fragility_cap

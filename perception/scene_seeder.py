# perception/scene_seeder.py
# Analyzes the scene and provides:
#   1. Object list + materials
#   2. Fragility caps per object
#   3. Anticipatory haptic presets
#
# Currently uses GPT-4V (API).
# To swap to local VLM: change BACKEND at the top of this file.
# The rest of the pipeline never changes.

import os, base64, json, time
import cv2
import numpy as np
from abc import ABC, abstractmethod

# ── swap VLM here — one line change ───────────────────────────────────────
BACKEND = "gpt4v"   # "gpt4v" | "moondream" | "gemma"

SYSTEM_PROMPT = """You are a haptic perception system for a robot arm.
Analyze the scene and identify objects the robot hand will interact with.
Respond ONLY with valid JSON — no explanation, no markdown."""

USER_PROMPT = """List every graspable object visible.
For each output:
- name: object name
- material: primary material (metal/glass/plastic/fabric/foam/wood/rubber/cardboard/skin)
- texture: one word (smooth/rough/soft/hard/bumpy/spongy)
- fragility: 1-5 (1=indestructible, 5=extremely fragile)
- haptic_preset: best match from [smooth metal, rigid plastic, glass, rough fabric, soft foam, human skin, wood, rubber, cardboard]

Respond ONLY with a valid JSON array. Example:
[{"name":"wine glass","material":"glass","texture":"smooth","fragility":5,"haptic_preset":"glass"},
 {"name":"metal bolt","material":"metal","texture":"smooth","fragility":1,"haptic_preset":"smooth metal"}]"""

# fragility → max duty cap
FRAGILITY_DUTY_CAPS = {1: 31, 2: 28, 3: 22, 4: 14, 5: 8}


# ── base class — fixed interface regardless of backend ────────────────────

class BaseSeeder(ABC):
    @abstractmethod
    def _query(self, frame_bgr: np.ndarray) -> list:
        pass

    def seed(self, frame_bgr: np.ndarray) -> dict:
        """
        Always returns same structure regardless of backend:
        {
            objects:               [{"name", "material", "fragility", "haptic_preset"}]
            fragility_caps:        {"wine glass": 8, "metal bolt": 31}
            anticipatory_presets:  {"wine glass": "glass", "metal bolt": "smooth metal"}
            clip_labels:           ["glass", "smooth metal"]
        }
        """
        try:
            objects = self._query(frame_bgr)
            if not objects:
                return self._fallback()

            fragility_caps, anticipatory_presets, clip_labels = {}, {}, []
            for obj in objects:
                name   = obj.get("name", "unknown")
                frag   = int(obj.get("fragility", 3))
                preset = obj.get("haptic_preset", "default")
                fragility_caps[name]       = FRAGILITY_DUTY_CAPS.get(frag, 22)
                anticipatory_presets[name] = preset
                if preset not in clip_labels:
                    clip_labels.append(preset)

            result = {"objects": objects,
                      "fragility_caps": fragility_caps,
                      "anticipatory_presets": anticipatory_presets,
                      "clip_labels": clip_labels}
            self._print(result)
            return result

        except Exception as e:
            print(f"[SceneSeeder] Error: {e}")
            return self._fallback()

    def _fallback(self) -> dict:
        print("[SceneSeeder] Fallback — no scene data")
        return {"objects": [], "fragility_caps": {},
                "anticipatory_presets": {}, "clip_labels": []}

    def _print(self, result: dict):
        print("[SceneSeeder] Scene:")
        for obj in result["objects"]:
            name = obj.get("name", "?")
            print(f"  {name:20s} fragility={obj.get('fragility','?')} "
                  f"duty_cap={result['fragility_caps'].get(name,'?'):2} "
                  f"preset={result['anticipatory_presets'].get(name,'?')}")


# ── GPT-4V ────────────────────────────────────────────────────────────────

class GPT4VSeeder(BaseSeeder):
    def __init__(self, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model  = model
            print(f"[SceneSeeder] Backend: GPT-4V ({model})")
        except ImportError:
            print("[SceneSeeder] openai not installed: pip install openai")
            self.client = None

    def _query(self, frame_bgr: np.ndarray) -> list:
        if not self.client:
            return []
        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = base64.b64encode(buf).decode()
        print("[SceneSeeder] Calling GPT-4V...")
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                   "detail": "low"}},
                    {"type": "text", "text": USER_PROMPT}
                ]}
            ],
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[SceneSeeder] GPT-4V done in {time.time()-t0:.1f}s")
        # strip markdown fences robustly
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)


# ── Moondream2 — implement when switching to local ─────────────────────────

class MoondreamSeeder(BaseSeeder):
    def __init__(self):
        print("[SceneSeeder] Backend: Moondream2 (local)")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2",
                                                        trust_remote_code=True)

    def _query(self, frame_bgr: np.ndarray) -> list:
        from PIL import Image
        img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        prompt = USER_PROMPT
        result = self.model.query(img, prompt)["answer"]
        result = result.replace("```json","").replace("```","").strip()
        return json.loads(result)


# ── Gemma3 — implement when switching to local ────────────────────────────

class GemmaSeeder(BaseSeeder):
    def __init__(self):
        print("[SceneSeeder] Backend: Gemma3-4B INT4 (local)")
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        import torch
        self.processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
        self.model     = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="cuda"
        )

    def _query(self, frame_bgr: np.ndarray) -> list:
        from PIL import Image
        import torch
        img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=USER_PROMPT, images=img,
                                return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=500)
        result = self.processor.decode(out[0], skip_special_tokens=True)
        result = result.replace("```json","").replace("```","").strip()
        return json.loads(result)


# ── factory ───────────────────────────────────────────────────────────────

def SceneSeeder(backend: str = BACKEND) -> BaseSeeder:
    """Change BACKEND at top of file to swap VLM. Nothing else changes."""
    backends = {"gpt4v": GPT4VSeeder,
                "moondream": MoondreamSeeder,
                "gemma": GemmaSeeder}
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}")
    return backends[backend]()


# ── test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

    seeder = SceneSeeder()
    result = seeder.seed(frame)
    print("\nfragility_caps:       ", result["fragility_caps"])
    print("anticipatory_presets: ", result["anticipatory_presets"])
    print("clip_labels:          ", result["clip_labels"])

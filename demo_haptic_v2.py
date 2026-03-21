import sys, os, time, argparse, threading, queue
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import torch

from perception.scene_seeder import SceneSeeder
from perception.dino_encoder import DinoEncoder
from neural.haptic_mlp import HapticMLP, load_model
from haptics.signal_fusion_v2 import SignalFusionV2
from haptics.haptic_controller import HapticController

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'haptic_mlp.pt')
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real',  action='store_true')
    p.add_argument('--image', type=str, default=None)
    p.add_argument('--cam',   type=int, default=0)
    return p.parse_args()

def simulate_force(t):
    cycle = t % 4.0
    if cycle < 1.0:   return cycle
    elif cycle < 2.5: return 1.0
    elif cycle < 3.0: return (3.0 - cycle) * 2.0
    else:             return 0.0

def dino_loop(encoder, mlp, frame_q, result_q, stop):
    while not stop.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue
        emb = encoder.encode(frame)
        if emb is not None:
            params = mlp.predict(emb, device=DEVICE)
            while not result_q.empty():
                try: result_q.get_nowait()
                except: pass
            result_q.put(params)

def haptic_loop(fusion, ctrl, result_q, stop):
    t_start = time.time()
    last = {"freq": 3, "duty": 0, "wave": 1}
    while not stop.is_set():
        try: last = result_q.get_nowait()
        except queue.Empty: pass
        force = simulate_force(time.time() - t_start)
        cmd = fusion.compute(force, last)
        if cmd["duty"] > 0:
            ctrl.send_all_fingers(cmd["duty"], cmd["freq"], cmd["wave"])
        else:
            ctrl.stop_all()
        time.sleep(1/50)

def main():
    args = parse_args()
    print("\n=== Haptic V2 — DINOv2 pipeline ===\n")

    if args.image:
        frame = cv2.imread(args.image)
    else:
        cap = cv2.VideoCapture(args.cam)
        time.sleep(1.5)
        _, frame = cap.read()
        cap.release()

    print("[main] GPT-4V scene seed...")
    scene = SceneSeeder().seed(frame)

    print("[main] Loading DINOv2...")
    encoder = DinoEncoder(device=DEVICE)
    encoder.load()

    print("[main] Loading MLP...")
    if not os.path.exists(MODEL_PATH):
        print(f"No MLP at {MODEL_PATH} — run: python3 -m neural.train_mlp")
        sys.exit(1)
    mlp = load_model(MODEL_PATH, device=DEVICE)

    fusion = SignalFusionV2()
    fusion.update_from_scene(scene)

    ctrl = HapticController(mock=not args.real)
    ctrl.connect()

    frame_q  = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=1)
    stop     = threading.Event()

    threading.Thread(target=dino_loop,   args=(encoder, mlp, frame_q, result_q, stop), daemon=True).start()
    threading.Thread(target=haptic_loop, args=(fusion, ctrl, result_q, stop), daemon=True).start()

    print("[main] Running — press Q to quit\n")
    t_start = time.time()
    display = frame.copy()

    while True:
        try: frame_q.put_nowait(display.copy())
        except queue.Full: pass

        try: last = result_q.get_nowait()
        except queue.Empty: last = {"freq": 3, "duty": 0, "wave": 1, "freq_cont": 0, "duty_cont": 0}

        t = time.time() - t_start
        force = simulate_force(t)
        cmd = fusion.compute(force, last)

        h, w = display.shape[:2]
        ui = display.copy()
        cv2.rectangle(ui, (0,0), (w,140), (0,0,0), -1)
        cv2.addWeighted(ui, 0.5, display, 0.5, 0, ui)
        cv2.putText(ui, f"Object: {fusion.object_name}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,180), 2)
        cv2.putText(ui, f"Force:  {force:.2f}", (12,54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,180), 2)
        cv2.putText(ui, f"Freq:   {last.get('freq',0)}  ({last.get('freq_cont',0):.2f})  Wave: {'sine' if last.get('wave',1) else 'sq'}", (12,80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,180), 2)
        cv2.putText(ui, f"Duty:   {cmd['duty']:2d} / {fusion.fragility_cap}", (12,106), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,180), 2)
        cv2.putText(ui, "Q to quit", (12,132), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        cv2.imshow("Haptic V2", ui)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    stop.set()
    ctrl.stop_all()
    ctrl.disconnect()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

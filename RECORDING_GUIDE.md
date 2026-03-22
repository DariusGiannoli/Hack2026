# Haptic Data Recording Guide

## 1. Gather Objects

Put these on the table next to you:

**Smooth/hard (label 1):** metal can, glass bottle, phone, plastic cup — pick 2

**Rough (label 2):** cardboard box, fabric/cloth, wood block — pick 2

**Soft (label 3):** sponge, foam, rubber ball, stress ball — pick 2

**For slip:** pick the 2-3 heaviest from above (water bottle, mug with water)

## 2. Open Terminal

```bash
cd ~/Desktop/haptic_teleoperation
conda activate haptic
```

Make sure `Headless_driver_double.py` is running on the robot.

## 3. Record — Follow This Exact Sequence

### Run 1 — Baseline (automatic, 15s, don't touch anything)

```bash
python neural/record_data.py --session baseline --real --iface enp131s0
```

Keep hands open and still. It stops on its own.

---

### Run 2 — Smooth Object A

1. Pick up the metal can with the robot's right hand
2. Wrap ALL fingers around it
3. Run:

```bash
python neural/record_data.py --session material --label 1 --real --iface enp131s0 --camera
```

4. While it records:
   - Squeeze lightly (3s)
   - Squeeze firmly (3s)
   - Slide fingertips across the surface slowly (5s)
   - Vary pressure up and down (5s)
   - Watch the terminal — `active fingers` should say 4+/6
5. Press Enter to stop (~20-30s total)

---

### Run 3 — Smooth Object B

1. Pick up glass/plastic
2. Run:

```bash
python neural/record_data.py --session material --label 1 --real --iface enp131s0 --camera
```

3. Squeeze, slide, vary. Press Enter.

---

### Run 4 — Rough Object A

1. Pick up cardboard
2. Run:

```bash
python neural/record_data.py --session material --label 2 --real --iface enp131s0 --camera
```

3. Squeeze, slide, vary. Press Enter.

---

### Run 5 — Rough Object B

1. Pick up fabric/wood
2. Run:

```bash
python neural/record_data.py --session material --label 2 --real --iface enp131s0 --camera
```

3. Squeeze, slide, vary. Press Enter.

---

### Run 6 — Soft Object A

1. Pick up sponge
2. Run:

```bash
python neural/record_data.py --session material --label 3 --real --iface enp131s0 --camera
```

3. Squeeze (it compresses!), release, squeeze again, slide. Press Enter.

---

### Run 7 — Soft Object B

1. Pick up foam/rubber
2. Run:

```bash
python neural/record_data.py --session material --label 3 --real --iface enp131s0 --camera
```

3. Squeeze, slide, vary. Press Enter.

---

### Run 8 — Slip (2-3 minutes, use heavy objects)

1. Pick up the heaviest object
2. Run:

```bash
python neural/record_data.py --session slip --real --iface enp131s0 --camera
```

3. While it records, repeat this cycle:
   - Grip firmly (3s)
   - Slowly loosen until the object slides 2-3cm down
   - Re-grip firmly
   - Repeat
   - Switch to a different heavy object halfway through
4. Watch the terminal — aim for slip % between 5-15%
   - 0% means loosen your grip more
   - 30%+ means you're dropping things too fast, grip tighter between slips
5. Press Enter after 2-3 minutes

---

## Rules During Every Run

- Object in hand BEFORE you run the command
- All fingers touching — watch the `active fingers` count
- Slide fingertips across the surface — this is the texture signal
- If you see `all pressure channels read 0` at warmup — check `Headless_driver_double.py`

## Total Time: ~15 minutes

## Output Files

```
data/tactile_recordings.csv      <- all material + baseline data (appended)
data/slip_recordings.csv         <- all slip data (appended)
data/frames/material_1_*/        <- camera frames for smooth objects
data/frames/material_2_*/        <- camera frames for rough objects
data/frames/material_3_*/        <- camera frames for soft objects
data/frames/slip_*/              <- camera frames during slip
```

## What Happens Next

After recording, the data is used to train:

1. **DINO + MLP**: camera frames -> material classification (smooth/rough/soft)
2. **HapticNet LSTM**: tactile time series -> slip detection + texture rendering
3. **GPT-4V**: single photo -> object ID + fragility cap (no training needed)

These feed into the real-time haptic pipeline at 50 Hz.

# Mina Teleop — Guide d'installation et de lancement

Script principal : `scripts/teleop/mina/teleop_edgard_copy.py`

---

## Prérequis système

### Python 3.10
Le projet requiert Python **3.10** (le `python3` global peut être 3.11+ sur ton système, ça pose des conflits).

```bash
python3.10 --version   # doit afficher Python 3.10.x
```

### Outils système utiles
```bash
sudo apt install -y v4l-utils   # pour identifier les devices caméra
```

Vérifier les caméras disponibles :
```bash
v4l2-ctl --list-devices
```

Sur le setup testé :
```
ZED 2i: ZED 2i (usb-0000:80:14.0-3):
    /dev/video4   ← à utiliser (CAMERA_ID = 4)

ACER FHD User Facing:
    /dev/video0   ← webcam interne
```

---

## Environnement conda

### Création (à faire une seule fois)
```bash
conda create -n mina310 python=3.10 -y
conda activate mina310
python -m pip install -U pip
```

### Installation des dépendances
```bash
pip install "mediapipe==0.10.11" "numpy<2" mujoco opencv-python
```

> **Pourquoi ces contraintes ?**
> - `mediapipe==0.10.11` : les versions récentes (0.10.18+) ont supprimé `mp.solutions` (utilisé par `detectors.py`)
> - `numpy<2` : compatibilité avec mediapipe legacy et isaaclab
> - `pyzed` (SDK ZED) requiert `numpy>=2` → à installer dans un **env séparé** si besoin (voir section ZED SDK)

---

## Lancement

```bash
conda activate mina310
cd /home/edgard/Desktop/AITEAM/Mina
python scripts/teleop/mina/teleop_edgard_copy.py
```

> Pas besoin de `mjpython` sur Linux (nécessaire uniquement sur macOS pour les conflits Cocoa).

---

## Configuration dans `teleop_edgard_copy.py`

Constantes à ajuster en haut du fichier :

| Constante | Valeur actuelle | Description |
|---|---|---|
| `CAMERA_ID` | `4` | Index ZED 2i (`/dev/video4`). `0` = webcam ACER. |
| `STEREO_DEPTH` | `False` | Mettre `True` une fois le tracking main/pose validé. |
| `N_SUBSTEPS` | `16` | Charge physique MuJoCo (réduire si FPS trop bas). |
| `AUTO_CALIB_SEC` | `5.0` | Secondes avant auto-calibration au démarrage. |
| `SHOW_CAMERA` | `True` | Affiche la fenêtre caméra (désactiver pour perf). |

### Touches pendant le run
| Touche | Action |
|---|---|
| `A` | Re-calibration manuelle (pose de référence) |
| `R` (fenêtre caméra) | Reset complet (position + calibration) |
| `ESC` | Quitter |

---

## Architecture des fichiers

```
scripts/teleop/mina/
├── teleop_edgard_copy.py       ← script principal
├── _camera_viewer.py           ← viewer caméra (subprocess)
├── vision/
│   ├── camera.py               ← ZEDCamera (cv2.VideoCapture, SBS split)
│   ├── detectors.py            ← StereoHandTracker + ArmTracker (MediaPipe)
│   ├── geometry.py             ← modèle pinhole ZED 2i + stéréo depth
│   └── smoother.py             ← OneEuroFilter
└── robots/
    └── leap_hand/
        ├── ik_retargeting.py   ← retargeting MediaPipe → LEAP (16 joints)
        └── arm_ik.py           ← IK bras gauche/droit (DLS)
```

---

## Pipeline de fonctionnement

```
ZED 2i (SBS /dev/video4)
    ↓ split left / right (camera.py)
    ↓
MediaPipe Hands (left frame)    → retargeting doigts → MuJoCo LEAP
MediaPipe Pose  (left frame)    → IK bras droit + gauche → MuJoCo humanoid
    ↓
OneEuroFilter (lissage joints + position + orientation poignet)
    ↓
MuJoCo viewer (mjpython sur macOS, python sur Linux)
```

---

## Calibration morphologique (automatique)

Au démarrage, après `AUTO_CALIB_SEC` secondes, le script :
1. Capture la pose de référence (orientation poignet, position bras)
2. Mesure les longueurs de tes bras via MediaPipe Pose world
3. Calcule un facteur d'échelle humain → robot (`MORPH_SCALE_MIN=0.60` à `MORPH_SCALE_MAX=1.50`)

Log attendu dans le terminal :
```
[MORPH] scales L=0.85 R=0.87 (human shoulder=0.412m, robot shoulder=0.280m)
```

---

## Activer la profondeur stéréo (étape suivante)

Une fois le tracking stable en mode mono :

1. Dans `teleop_edgard_copy.py` :
```python
STEREO_DEPTH = True
```

2. Mettre les vraies intrinsèques ZED dans `vision/geometry.py` :
```python
# Récupérer via le ZED SDK Python :
# import pyzed.sl as sl
# zed = sl.Camera()
# params = zed.get_camera_information().camera_configuration.calibration_parameters
# left = params.left_cam
# print(left.fx, left.fy, left.cx, left.cy, params.get_camera_baseline())

ZED2I = PinholeCamera(
    fx=700.0,    # ← remplacer par valeur SDK
    fy=700.0,    # ← remplacer par valeur SDK
    cx=638.0,    # ← remplacer par valeur SDK
    cy=363.0,    # ← remplacer par valeur SDK
    baseline_m=0.12,
)
```

3. Ajuster `EPIPOLAR_TOL` si besoin (défaut `40` px, relâché pour ZED).

### Env conda séparé pour pyzed (numpy>=2 requis)
```bash
conda create -n mina_zed python=3.10 -y
conda activate mina_zed
pip install pyzed "numpy>=2" opencv-python
```

---

## Dépannage fréquent

| Erreur | Cause | Fix |
|---|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe trop récent (0.10.18+) ou Python 3.13 | `pip install "mediapipe==0.10.11"` dans `mina310` |
| `[TORSO] pose non détectée` en boucle | Mauvais `CAMERA_ID` ou caméra non ZED | Mettre `CAMERA_ID = 4` |
| `ModuleNotFoundError: No module named 'mujoco'` | Mauvais env Python actif | `conda activate mina310` |
| `isaaclab requires numpy<2` | Conflit avec base conda | Ne pas pip-install dans `base`, utiliser `mina310` |
| Fenêtre MuJoCo ne s'ouvre pas | Problème OpenGL/EGL | Vérifier drivers NVIDIA, `echo $DISPLAY` |

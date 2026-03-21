# Lire / déboguer les topics ZMQ vers GR00T (deploy)

Le binaire `g1_deploy_onnx_ref` (via `deploy.sh`) utilise **ZMQManager** : un **SUB** sur `tcp://<host>:<port>` (défaut `localhost:5556`).

## Les 3 topics (même port)

| Topic     | Rôle |
|-----------|------|
| `command` | Démarrer / arrêter / choisir le mode : `planner=true` → commandes `planner` ; `planner=false` → flux `pose`. Optionnel : `delta_heading` (f32). |
| `planner` | Locomotion + optionnel : `upper_body_position` (17), `upper_body_velocity` (17), `left_hand_joints` (7), `right_hand_joints` (7), champs VR… |
| `pose`    | Mode streamed motion : `joint_pos`, `joint_vel`, `body_quat`, `frame_index`, etc. (protocole v1/v3/v4 selon émetteur). |

Format commun : **`[topic ASCII][header JSON null-paddé sur 1024 ou 1280 octets][payload binaire]`**  
Le header décrit les champs (`name`, `dtype`, `shape`). **Deux tailles de padding existent** : 1024 (ex. ancien `_zmq_deploy_publisher`) et 1280 (`gear_sonic` `zmq_planner_sender.py`).

---

## Techniques pour « écouter » ce qui part vers GR00T

### 1. Script du repo : `listen_bridge.py` (recommandé)

```bash
cd /path/to/Hack2026
python3 bridge/listen_bridge.py --topic all --port 5556
python3 bridge/listen_bridge.py --topic planner
python3 bridge/listen_bridge.py --topic command
python3 bridge/listen_bridge.py --topic pose
```

- **SUB** sur `localhost:5556`, décode header + champs.
- Gère **1024 et 1280** octets de header.

### 2. Python + pyzmq (minimal)

```python
import zmq
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:5556")
s.setsockopt(zmq.SUBSCRIBE, b"planner")  # ou b"" pour tout
while True:
    msg = s.recv()
    topic_end = msg.index(b"{")
    print("topic:", msg[:topic_end])
    # puis parser JSON jusqu’au \0, puis payload…
```

### 3. Script officiel GR00T (référence wire format)

```bash
cd GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests
python3 test_zmq_manager.py --host localhost --port 5556 --test combined
```

Utile pour **générer** des messages valides et comparer avec ton émetteur.

### 4. Plusieurs subscribers (ZMQ)

Un **PUB** peut nourrir **plusieurs SUB** sur le même port : tu peux lancer `listen_bridge.py` **en plus** du deploy sans couper le robot (tous reçoivent les mêmes frames).

### 5. Affichage brut (hex) — sans parser

```bash
python3 -c "
import zmq
c=zmq.Context(); s=c.socket(zmq.SUB)
s.connect('tcp://localhost:5556'); s.setsockopt(zmq.SUBSCRIBE, b'')
while True: print(s.recv()[:200].hex())
"
```

Ou `tcpdump -i lo -A -s 0 port 5556` (moins lisible, TCP brut).

### 6. Vérifier qui bind / qui connect

- **Émetteur (téléop / bridge)** : en général `PUB` + **`bind`** `tcp://*:5556`.
- **GR00T deploy** : `SUB` + **`connect`** `tcp://localhost:5556`.
- Ton listener doit aussi **`connect`** (comme le deploy), pas `bind`.

### 7. Variables deploy (`deploy.sh`)

- `--zmq-host` (défaut `localhost`)
- `--zmq-port` (défaut `5556`)
- `--zmq-topic` : préfixe pour le flux **pose** (défaut `pose`) ; `command` et `planner` restent leurs noms fixes côté code.

### 8. Timeout / pas de message

- Si tu ne vois rien : mauvais port, firewall, ou **personne ne bind** sur 5556.
- **PUB/SUB** : les messages envoyés **avant** que le SUB soit connecté sont **perdus** → d’où les keepalive `command start` répétés.

### 9. DDS (pas ZMQ)

Les doigts / bas niveau robot passent aussi par **DDS** (`rt/lowstate`, etc.) — ce n’est **pas** du ZMQ ; utiliser `debug_dds_bridge.py` pour ça.

---

## Résumé

| Objectif | Outil |
|----------|--------|
| Lire planner + command + pose proprement | `bridge/listen_bridge.py --topic all` |
| Tester le protocole | `test_zmq_manager.py` dans gear_sonic_deploy |
| Déboguer header 1024 vs 1280 | `listen_bridge.py` (détection auto) |
| Voir des octets bruts | hex dump / tcpdump |

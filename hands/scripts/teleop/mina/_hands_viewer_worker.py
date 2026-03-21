"""Subprocess MuJoCo viewer pour la scène inspire-hand fused.

Ce module est importé dans un subprocess séparé (spawn) afin que le viewer
dispose de son propre contexte OpenGL/GLFW, évitant ainsi les conflits avec
le viewer principal (segfault avec deux launch_passive dans le même process).
"""
import time


def run(scene_xml: str,
        sh_r_pos,
        sh_r_quat,
        sh_l_pos,
        sh_l_quat,
        sh_ctrl,
        n_ctrl: int,
        sh_run) -> None:
    """Point d'entrée du subprocess.

    Args:
        scene_xml : chemin absolu vers la scène fused XML.
        sh_r_pos  : Array partagé [3]  — position mocap main droite.
        sh_r_quat : Array partagé [4]  — quat    mocap main droite.
        sh_l_pos  : Array partagé [3]  — position mocap main gauche.
        sh_l_quat : Array partagé [4]  — quat    mocap main gauche.
        sh_ctrl   : Array partagé [24] — ctrl doigts (0-11 droite, 12-23 gauche).
        n_ctrl    : taille utile de sh_ctrl.
        sh_run    : Value(bool) — mis à False pour arrêter la boucle.
    """
    import mujoco
    import mujoco.viewer
    import numpy as np

    try:
        model = mujoco.MjModel.from_xml_path(scene_xml)
    except Exception as exc:
        print(f"[HANDS VIEWER] Erreur chargement XML : {exc}")
        sh_run.value = False
        return

    data  = mujoco.MjData(model)
    mid   = model.body("hand_proxy").mocapid[0]
    _bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy_l")
    mid_l = model.body("hand_proxy_l").mocapid[0] if _bid != -1 else None
    n     = min(n_ctrl, model.nu)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running() and sh_run.value:
            data.mocap_pos[mid]  = sh_r_pos[:]
            data.mocap_quat[mid] = sh_r_quat[:]
            if mid_l is not None:
                data.mocap_pos[mid_l]  = sh_l_pos[:]
                data.mocap_quat[mid_l] = sh_l_quat[:]
            data.ctrl[:n] = sh_ctrl[:n]
            mujoco.mj_forward(model, data)
            v.sync()

    sh_run.value = False

"""
robots/
-------
One sub-package per robot.  Each package exposes a class that inherits from
``robots.base.RobotController``.

Current robots
--------------
  robots.leap_hand   — LEAP Hand (16-DOF under-actuated)
  robots.franka      — Franka FR3  (placeholder, not yet implemented)

Adding a new robot
------------------
1. ``mkdir robots/<name>``
2. Subclass ``RobotController`` in ``robots/<name>/__init__.py``
3. Implement ``retarget()`` and ``scene_xml_path()``
4. Write or symlink the MuJoCo scene XML under ``<name>/``
"""

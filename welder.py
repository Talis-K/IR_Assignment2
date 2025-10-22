import time
from math import pi
from typing import List, Optional

from spatialgeometry import Cylinder, Sphere
from spatialmath import SE3

class Welder:
    def __init__(self, base_pose: SE3, L1: float = 0.08, L2: float = 0.06, radius: float = 0.01,
                 spark_radius: float = 0.01, spark_period: float = 0.04, max_sparks: int = 200):
        """
        base_pose: initial EE pose
        L1, L2: handle/nozzle lengths
        radius: base cylinder radius factor
        spark_radius: radius of the spark spheres
        spark_period: min seconds between spark spawns (rate limit)
        max_sparks: keep at most this many sparks in the scene
        """
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.radius = float(radius)
        self.base = SE3(base_pose)
        self.tool_length = float(L1 + L2)  # maintain compat; your main reads .tool_length
        self.spark_radius = float(spark_radius)
        self.spark_period = float(spark_period)
        self.max_sparks = int(max(1, max_sparks))

        self.env = None
        self._torch_meshes_built = False
        self._last_spark_t = 0.0

        self._create_geometry(self.base)
        self.welds: List[Sphere] = []

    # ---------- internal helpers ----------
    def _safe_add(self, obj):
        """
        Thread-safe add:
        - If env has a parent with safe_add (your Environment), use that.
        - Else fall back to env.add (OK when single-threaded).
        """
        if self.env is None:
            return
        parent = getattr(self.env, "_parent", None)
        if parent is not None and hasattr(parent, "safe_add"):
            parent.safe_add(obj)
        else:
            self.env.add(obj)

    def _create_geometry(self, base_pose: SE3):
        """Create simple welder geometry with a handle and nozzle."""
        L1, L2, r = self.L1, self.L2, self.radius

        # Apply same orientation offset used in update()
        base_pose = SE3(base_pose) * SE3(0, 0, 0.02) * SE3.Ry(-pi/2)

        # Base body (mount)
        self.base_geom = Cylinder(0.04, 0.02, color=[0.8, 0.8, 0.8])  # (length, radius) in your current API
        self.base_geom.T = base_pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)

        # Welder handle
        self.handle = Cylinder(radius=r * 1.2, length=L1, color=[0.4, 0.4, 0.9])
        self.handle.T = base_pose * SE3(L1 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

        # Welder tip / nozzle
        self.nozzle = Cylinder(radius=r * 0.6, length=L2, color=[1.0, 0.4, 0.4])
        self.nozzle.T = base_pose * SE3(L1 + L2 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

        self.geometry = [self.base_geom, self.handle, self.nozzle]
        self._torch_meshes_built = True

    # ---------- lifecycle ----------
    def add_to_env(self, env):
        """Add welder geometry to Swift environment (safe to call from setup thread)."""
        self.env = env
        for geom in self.geometry:
            self._safe_add(geom)

    # ---------- control ----------
    def update(self, pose: SE3):
        """Update welder transform to follow the EE. Uses mesh .T updates (thread-safe)."""
        self.base = SE3(pose)
        pose = SE3(pose) * SE3(0, 0, 0.02) * SE3.Ry(-pi/2)

        if self._torch_meshes_built:
            self.base_geom.T = pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)
            self.handle.T    = pose * SE3(self.L1 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)
            self.nozzle.T    = pose * SE3(self.L1 + self.L2 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

    def weld(self, pose: SE3):
        """
        Emit a spark at the nozzle tip (rate-limited).
        Safe to call from worker threads thanks to _safe_add().
        """
        now = time.time()
        # Rate-limit to avoid flooding Swift socket
        if (now - self._last_spark_t) < self.spark_period:
            self.update(pose)
            return
        self._last_spark_t = now

        # Keep the tool tracking
        self.update(pose)

        # Tip position: along +X of the oriented tool by (L1 + L2)
        spark_pose = (SE3(pose) * SE3(0, 0, 0.02) * SE3.Ry(-pi/2) * SE3(self.L1 + self.L2, 0, 0))
        spark = Sphere(radius=self.spark_radius, color=[1, 1, 1], pose=spark_pose)
        self.welds.append(spark)
        if self.env is not None:
            self._safe_add(spark)

        # Trim sparks list to keep scene responsive
        if len(self.welds) > self.max_sparks:
            self.welds = self.welds[-self.max_sparks:]


if __name__ == "__main__":
    import swift
    import time as _t

    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([0.5, 0.3, 0.3], [0, 0, 0])

    welder = Welder(SE3(0, 0, 0))
    welder.add_to_env(env)

    for i in range(200):
        pose = SE3(0.1 * i/180, 0, 0)
        if i % 3 == 0:
            welder.weld(pose)
        else:
            welder.update(pose)
        env.step(0.02)
        _t.sleep(0.01)

    env.hold()

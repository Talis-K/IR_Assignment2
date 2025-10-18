#!/usr/bin/env python
import swift
from spatialgeometry import Cylinder, Sphere
from spatialmath import SE3
from math import pi
import time

class Welder:
    def __init__(self, base_pose, L1=0.08, L2=0.06, radius=0.01):
        self.L1 = L1
        self.L2 = L2
        self.radius = radius
        self.base = base_pose
        self.tool_length = 0.16
        self._create_geometry(base_pose)
        self.welds = []

    def _create_geometry(self, base_pose):
        """Create simple welder geometry with a handle and nozzle."""
        L1, L2, r = self.L1, self.L2, self.radius

        # Apply same orientation offset used in update()
        base_pose = base_pose * SE3(0, 0, 0.02) * SE3.Ry(-pi/2)

        # Base body (mount)
        self.base_geom = Cylinder(0.04, 0.02, color=[0.8, 0.8, 0.8])
        self.base_geom.T = base_pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)

        # Welder handle
        self.handle = Cylinder(radius=r * 1.2, length=L1, color=[0.4, 0.4, 0.9])
        self.handle.T = base_pose * SE3(self.L1 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

        # Welder tip / nozzle
        self.nozzle = Cylinder(radius=r * 0.6, length=L2, color=[1.0, 0.4, 0.4])
        self.nozzle.T = base_pose * SE3(self.L1 + self.L2 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

        # Collect all geometry for convenience
        self.geometry = [self.base_geom, self.handle, self.nozzle]

    def add_to_env(self, env):
        """Add welder geometry to Swift environment."""
        self.env = env
        for geom in self.geometry:
            env.add(geom)

    def update(self, pose):
        """Update position/orientation of welder based on new robot end-effector pose."""
        self.base = pose
        pose = pose * SE3(0, 0, 0.02) * SE3.Ry(-pi/2)

        self.base_geom.T = pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)
        self.handle.T = pose * SE3(self.L1 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)
        self.nozzle.T = pose * SE3(self.L1 + self.L2 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)
    
    def weld(self, pose):

        self.update(pose)

        spark_pose = (pose * SE3(0, 0, 0.02) * SE3.Ry(-pi/2) * SE3(self.L1 + self.L2, 0, 0))
        spark = Sphere(radius=0.01, color=[1, 1, 1], pose=spark_pose)
        self.welds.append(spark)
        if hasattr(self, "env"):
            self.env.add(spark)
        

if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([0.5, 0.3, 0.3], [0, 0, 0])

    welder = Welder(SE3(0, 0, 0))
    welder.add_to_env(env)

    # Move the welder to demonstrate update()
    for i in range(100):
        pose = SE3(0.1 * i/80, 0, 0)
        welder.update(pose)
        env.step(0.02)
        time.sleep(0.01)

    env.hold()

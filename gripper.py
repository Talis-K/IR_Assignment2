#!/usr/bin/env python
import swift
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid
from spatialmath import SE3
import time
from math import pi


class FingerDH(rtb.DHRobot):
    def __init__(self, pose, L1=0.08, L2=0.06, radius=0.01):
        self.L1 = L1
        self.L2 = L2
        self.radius = radius

        links = [
            rtb.RevoluteDH(a=L1, alpha=0, d=0, qlim=[-pi/2, pi/2]),
            rtb.RevoluteDH(a=L2 / 2, alpha=0, d=0, qlim=[-pi/2, pi/2])
        ]

        super().__init__(links, name='finger')
        self.base = pose
        self._create_geometry()

    def _create_geometry(self):
        L1, L2, r = self.L1, self.L2, self.radius

        joint1 = Cylinder(0.01, length=0.02, color=[0.6, 0.6, 0.6])
        joint1.T = SE3(-L1,0,0)

        cyl1 = Cylinder(radius=r, length=L1, color=[0.6, 0.6, 1.0])
        cyl1.T = SE3(-L1 / 2, 0, 0) * SE3.RPY(0, pi/2, 0)

        joint2 = Cylinder(0.01, length=0.02, color=[0.6, 0.6, 0.6])
        joint2.T = SE3(-L2/2, 0, 0)

        cyl2 = Cylinder(radius=r * 0.9, length=L2, color=[0.6, 1.0, 0.6])
        cyl2.T = SE3(0, 0, 0) * SE3.RPY(0, pi/2, 0)

        tip = Cuboid((0.05, 0.01, 0.018), color=[1, 0.6, 0.6])
        tip.T = SE3(0.005, 0.008, 0)

        self.links[0].geometry = [joint1, cyl1]
        self.links[1].geometry = [joint2, cyl2, tip]




class Gripper:
    def __init__(self, base_pose):
        
        # Offset for base geom
        base_pose = base_pose * SE3(0,0,0.02) * SE3.Ry(-pi/2)

        # Relative offsets for each finger
        self.left_offset = SE3(0, 0.05, 0) * SE3.Rx(pi)
        self.right_offset = SE3(0, -0.05, 0)

        # Create two finger robots
        self.left_finger = FingerDH(base_pose * self.left_offset)
        self.right_finger = FingerDH(base_pose * self.right_offset)

        # Base geometry (gripper palm)
        self.base_geom = Cylinder(0.07, 0.02, color=[0.8, 0.8, 0.8])
        self.base_geom.T = base_pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)

    def update(self, pose):
        # Offset for base geom
        pose = pose * SE3(0,0,0.02) * SE3.Ry(-pi/2)
        self.base_geom.T = pose * SE3(-0.01, 0, 0) * SE3.Ry(pi/2)
        self.left_finger.base = pose * self.left_offset
        self.right_finger.base = pose * self.right_offset

    def add_to_env(self, env):
        self.env = env
        env.add(self.left_finger)
        env.add(self.right_finger)
        env.add(self.base_geom)

    def actuate(self, position):
        q_open = [-pi/10, pi/10]
        q_close = [0, 0]


        if position == 'open':
            q_target = q_open
        elif position == 'close':
            q_target = q_close
        else:
            print("Invalid position:", position)
            return

        # Plan trajectory
        traj = rtb.jtraj(self.left_finger.q, q_target, 50).q

        # Step through both trajectories simultaneously
        for q in (traj):
            self.left_finger.q = q
            self.right_finger.q = q
            self.env.step(0.02)
            time.sleep(0.01)

if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([0.5, 0.3, 0.3], [0, 0, 0])

    gripper = Gripper(SE3(0, 0, 0))
    gripper.add_to_env(env)

    # Test motion
    gripper.actuate("open")
    gripper.actuate("close")
    gripper.actuate("open")

    gripper.update(SE3(1,1,1))
    gripper.actuate("close")
    gripper.actuate("open")


    env.hold()

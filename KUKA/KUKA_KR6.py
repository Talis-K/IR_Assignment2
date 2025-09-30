"""
KUKA KR6 rough DHRobot3D wrapper
Adapted from a LinearUR3 example. Uses the user's 3D model files:
  - base_link.dae
  - link_1.dae
  - link_2.dae
  - link_3.dae
  - link_4.dae
  - link_5.dae
  - link_6.dae

Notes:
 - The DH parameters provided here are illustrative/approximate. For accurate kinematics, replace
   the DH arrays in `_create_DH()` with the exact values from your robot CAD/URDF or manufacturer data.
 - Tweak `qtest_transforms` to line up the .dae models if they don't match visually.
 - This file expects `ir_support.robots.DHRobot3D` from your workspace (same base class as your LinearUR3).

Author: generated for user
Date: 2025-09-30
"""

import os
import time
from math import pi

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D


class KUKAKR6(DHRobot3D):
    def __init__(self):
        """
        Rough KUKA KR6 model that follows the same structure as your LinearUR3 class.
        Replace the DH parameter arrays with precise values for accurate kinematics.
        """
        links = self._create_DH()

        # Filenames you provided (assumed to be in same directory as this file)
        link3D_names = dict(
            link0='base_link', color0=(0.2, 0.2, 0.2, 1),
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6')

        # A reasonable home configuration and transforms for aligning the 3D parts.
        qtest = [0, -pi/4, pi/2, -pi/4, pi/2, 0]
        qtest_transforms = [
            spb.transl(0, 0, 0),                        # base
            spb.transl(0, 0, 0.08) @ spb.trotx(-pi/2),  # link1 visual offset
            spb.transl(0, 0, 0.2),                      # link2
            spb.transl(0, 0, 0.35),                     # link3
            spb.transl(0, 0, 0.5),                      # link4
            spb.transl(0, 0, 0.65),                     # link5
            spb.transl(0, 0, 0.78)                      # link6 / flange
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        print("[DEBUG] Looking for 3D model files in:", current_path)
        for key, fname in link3D_names.items():
            if key.startswith("link"):
                path_to_file = os.path.join(current_path, fname)
                print(f"[DEBUG] {key} -> {path_to_file} exists? {os.path.exists(path_to_file)}")

        super().__init__(links, link3D_names, name='KUKA_KR6', link3d_dir=current_path, qtest=qtest, qtest_transforms=qtest_transforms)

        self.q = qtest

    def _create_DH(self):
        """
        Create a list of RevoluteDH links for a 6-DOF KUKA KR6. THESE VALUES ARE APPROXIMATE.

        Standard DH arrays (theta is joint variable for RevoluteDH):
        - a: link lengths
        - d: link offsets
        - alpha: link twists

        Replace these arrays with the exact DH parameters from your robot's datasheet or URDF.
        """
        a = [0, 0.325, 0.035, 0, 0, 0]
        d = [0.4, 0, 0, 0.365, 0, 0.08]
        alpha = [pi/2, 0, -pi/2, pi/2, -pi/2, 0]
        offset = [0, 0, 0, 0, 0, 0]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i])
            links.append(link)

        return links

    def test(self):
        """
        Launch a Swift window, add the robot and run a small trajectory.
        """
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.add_to_env(env)

        q_goal = [self.q[i] + (pi/6 if i % 2 == 0 else -pi/6) for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 60).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        env.hold()
        time.sleep(2)


if __name__ == "__main__":
    robot = KUKAKR6()
    input("Press Enter to test KUKA KR6 visual and simple motion...")
    robot.test()

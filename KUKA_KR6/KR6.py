#!/usr/bin/env python3
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os
from math import pi

class KR6_Robot(DHRobot3D):
    def __init__(self):
        links = self._create_DH() # DH Links
        qtest = [0, -pi/2, 0, 0, 0, 0] # Intial joint config

        # Visual transforms
        qtest_transforms = [
            spb.transl(+0.000, 0, 0.000), # link_0 (base)
            spb.transl(+0.000, 0, 0.400), # link_1
            spb.transl(+0.025, 0, 0.400), # link_2
            spb.transl(+0.480, 0, 0.400), # link_3
            spb.transl(+0.480, 0, 0.435), # link_4 
            spb.transl(+0.900, 0, 0.435), # link_5
            spb.transl(+0.980, 0, 0.435), # link_6
        ]

        # Mesh File and Names
        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "Meshes", "KUKA_Meshes")
        
        link3D_names = dict(
            link0='link_0',
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',
        )

        # Initialise parent class
        super().__init__(
            links,
            link3D_names,
            name='KUKA_KR6',
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        # End-Effector Tool Set Up
        self.tool_length = 0.08
        self.tool = SE3(0, 0, self.tool_length)

    def _create_DH(self):
        a       = [0.025, 0.455, 0.035,  0.00,  0.00, 0.0]
        d       = [0.400, 0.000, 0.000,  0.42,  0.00, 0.0]
        alpha   = [ pi/2, 0.000,  pi/2,  pi/2, -pi/2, 0.0]
        offset  = [0.000,  pi/2,  pi/2,  0.00,  0.00, 0.0]
        qlim    = [[-pi, pi]] * 6;  
        qlim[3] = [-pi/4, pi/4]

        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i]) for i in range(6)]
        return links

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, pi/8])

        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        print("Animation beginning")

        # Test motion
        q_start = [0, 0, 0, 0, 0, 0]
        q_goal  = [0, 0, 0, 0, pi/2, 0]

        qtraj = rtb.jtraj(q_start, q_goal, 60).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        print("Animation complete")
        env.hold()

if __name__ == "__main__":
    r = KR6_Robot()
    r.test()

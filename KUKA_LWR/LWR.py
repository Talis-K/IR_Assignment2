#!/usr/bin/env python3
import os, time
from math import pi
import numpy as np

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D


class Load(DHRobot3D):
    def __init__(self):
        # ---------- DH links ----------
        links = self._create_DH()

        # ---------- Meshes ----------
        link3D_names = dict(
            link0='base',
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',
            link7='link_7'
        )

        # ---------- Robot Transform Offsets----------
        qtest = [0, 0, 0, 0, 0, 0, 0]
        qtest_transforms = [
            spb.transl(0, 0, 0)      @ spb.trotz(0) @ spb.trotx(0),      # base
            spb.transl(0, 0, 0.09)   @ spb.trotz(0) @ spb.trotx(0),      # link_1
            spb.transl(0, -0.014, 0.31) @ spb.trotz(0) @ spb.trotx(0),   # link_2
            spb.transl(0, -0.014, 0.49) @ spb.trotz(0) @ spb.trotx(0),   # link_3
            spb.transl(0, 0, 0.701)  @ spb.trotz(0) @ spb.trotx(0),      # link_4
            spb.transl(0, 0, 0.878)  @ spb.trotz(0) @ spb.trotx(0),      # link_5
            spb.transl(0, 0, 1.078)  @ spb.trotz(0) @ spb.trotx(0),      # link_6
            spb.transl(0, 0, 1.078)  @ spb.trotz(0) @ spb.trotx(0),      # link_7 (ee)
        ]

        # ---------- Mesh folder ----------
        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "meshes2/visual")

        super().__init__(
            links,
            link3D_names,
            name='LBR',                  
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest

        self.tool = SE3(0, 0, -0.04)


    def _create_DH(self):
        a     = [0, 0, 0, 0, 0, 0, 0]
        d     = [0.307, 0.013, 0.393, -0.013, 0.3769, -0.04, 0.1]
        alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0]
        qlim  = [[-pi, pi]] * 7
        qlim[1] = [-pi/4, pi/4]  # Joint 2 limit
        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]) for i in range(7)]
        return links


if __name__ == "__main__":
    r = Load()
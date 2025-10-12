#!/usr/bin/env python3
import os, time
from math import pi
import numpy as np

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D

# ===================== CONFIG =====================
USE_6DOF = False  # False -> exact 7-DoF. True -> 6-DoF (drops last wrist)

# Mesh stems in ./meshes (adjust if your file names differ)
LINK3D_NAMES_7DOF = dict(
    link0='base',
    link1='link_1',
    link2='link_2',
    link3='link_3',
    link4='link_4',
    link5='link_5',
    link6='link_6',
    link7='link_7',
)
LINK3D_NAMES_6DOF = dict(
    link0='base',
    link1='link1',
    link2='link2',
    link3='link3',
    link4='link4',
    link5='link5',
    link6='link6',
)

# If your meshes are Y-up, enable a global axis fix:
MESH_AXIS_FIX = SE3()           # default assumes Z-up meshes
# MESH_AXIS_FIX = SE3.Rx(-pi/2) # uncomment if parts look rotated 90° about X


class Load(DHRobot3D):
    """
    KUKA LWR (LWR4+), Standard DH:

    7-DoF (URDF-like):
      d = [0.11, 0.2005, 0.20, 0.20, 0.20, 0.19, 0.078]
      a = [0]*7
      alpha = [ +pi/2, -pi/2, +pi/2, -pi/2, +pi/2, -pi/2, 0 ]

    6-DoF (approx, last wrist removed → added to tool):
      d = [0.11, 0.2005, 0.20, 0.20, 0.20, 0.19]
      a = [0]*6
      alpha = [ +pi/2, -pi/2, +pi/2, -pi/2, +pi/2, -pi/2 ]
    """

    def __init__(self):
        links, link3D_names, tool_extra = self._make_dh()
        qtest = np.zeros(len(links))

        # -------- Build a temp DH robot to compute WORLD poses at qtest --------
        dh_tmp = rtb.DHRobot(links, name="LWR_tmp")
        Ts = dh_tmp.fkine_all(qtest)  # base->each link-end, WORLD SE3

        # -------- Per-link LOCAL visual tweaks (from typical LWR URDF visuals) ---
        # Most LWR visuals use rpy="0 0 pi", and some have a small z nudge -0.008
        Rz_pi = SE3.Rz(pi)
        if len(links) == 7:
            local_offsets = [
                SE3(),             # base (link0)
                SE3(0, 0, -0.008), # link1
                SE3(0, 0, 1.0),             # link2
                SE3(0, 0, -0.008), # link3
                SE3(),             # link4
                SE3(0, 0, -0.008), # link5
                SE3(),             # link6
                SE3(),             # link7
            ]
        else:  # 6-DoF version (no link7)
            local_offsets = [
                SE3(),
                SE3(0, 0, 1.0),
                SE3(0, 0, 0.1),
                SE3(0, 0, -0.008),
                SE3(),
                SE3(0, 0, -0.008),
                SE3(),
            ]

        # -------- Compose WORLD qtest transforms for DHRobot3D -------------------
        # index 0 is the base mesh pose; then one per kinematic link
        qtest_transforms = []
        # base/world:
        qtest_transforms.append((SE3() * Rz_pi * MESH_AXIS_FIX * local_offsets[0]).A)
        # link 1..N:
        for i in range(len(links)):
            T_world = Ts[i] * Rz_pi * MESH_AXIS_FIX * local_offsets[i+1]
            qtest_transforms.append(T_world.A)

        # -------- Init DHRobot3D with WORLD-registered meshes --------------------
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(here, "meshes2")

        super().__init__(
            links,
            link3D_names,
            name=f'KUKA_LWR_{"6DoF" if len(links)==6 else "7DoF"}',
            link3d_dir=mesh_dir,
            qtest=list(qtest),
            qtest_transforms=qtest_transforms
        )

        self.q = list(qtest)
        self.base = SE3(0.5, 0.5, 0.0)
        self.tool_length = 0.12 + tool_extra  # add 0.078 if 6DoF
        self.addconfiguration("tool_length", self.tool_length)

    # ---------------- helpers ----------------
    def _make_dh(self):
        if USE_6DOF:
            a     = [0, 0, 0, 0, 0, 0]
            d     = [0.11, 0.2005, 0.20, 0.20, 0.20, 0.19]
            alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2]
            links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=[-pi, pi]) for i in range(6)]
            link3D_names = LINK3D_NAMES_6DOF
            tool_extra = 0.078
        else:
            a     = [0, 0, 0, 0, 0, 0, 0]
            d     = [0.11, 0.2005, 0.20, 0.20, 0.20, 0.19, 0.078]
            alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0.0]
            links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=[-pi, pi]) for i in range(7)]
            link3D_names = LINK3D_NAMES_7DOF
            tool_extra = 0.0
        return links, link3D_names, tool_extra

    # ---------------- demo ----------------
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, pi/8])

        self.add_to_env(env)

        q0 = self.q.copy()
        if len(self.q) == 6:
            q1 = [pi/3, -pi/4, pi/3, -pi/3, pi/4, -pi/6]
        else:
            q1 = [pi/3, -pi/4, pi/3, -pi/3, pi/4, -pi/6, pi/8]

        for q in rtb.jtraj(q0, q1, 70).q:
            self.q = q
            env.step(0.02)

        time.sleep(0.5)
        env.hold()


if __name__ == "__main__":
    Load().test()

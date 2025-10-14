##  @file
#   @brief Kinova JACO2 (6-DoF) with DH kinematics + cuboid visuals (no DAE files)
#   @date Oct 2025

import os
import time
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialgeometry as geometry
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from math import pi

class JACO2_6s300(DHRobot3D):
    def __init__(self):
        links = self._create_DH()

        link3D_names = dict(
            link0="base_cuboid",
            link1="l1_cuboid",
            link2="l2_cuboid",
            link3="l3_cuboid",
            link4="l4_cuboid",
            link5="l5_cuboid",
            link6="hand_cuboid",
        )

        # preview pose
        qtest = [0, -pi/3, pi/3, 0, -pi/4, pi/4]

        # absolute frame poses for qtest (base -> frame_i)
        T = SE3()
        qtest_transforms = [SE3().A]
        for i, L in enumerate(links):
            T = T @ L.A(qtest[i])
            qtest_transforms.append(T.A)

        super().__init__(
            links,
            link3D_names,
            name="JACO2_6s300_CUBES",
            link3d_dir=os.path.abspath(os.path.dirname(__file__)),
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest

    # ===================== 3D visuals =====================
    def _apply_3dmodel(self):
        """
        n+1 visuals (frames 0..n).
        - Draw a[i] on frame i along +X_i (DH).
        - Draw d[i] on frame (i-1) along +Z_{i-1} **only if a[i]==0**.
        Visual-only: ensure small stubs for a[i]=0 or d[i]=0.
        """
        self.links_3d = []
        relation = []

        VIS_EPS = 0.02
        thick    = [0.12, 0.10, 0.085, 0.080, 0.070, 0.065, 0.065]
        fallback = [VIS_EPS] * (self.n + 1)

        colors = [
            (0.95, 0.95, 0.95, 1.0),  # 0 base
            (0.90, 0.30, 0.30, 1.0),  # 1 shoulder
            (0.30, 0.90, 0.30, 1.0),  # 2 arm
            (0.30, 0.30, 0.90, 1.0),  # 3 forearm
            (0.90, 0.90, 0.30, 1.0),  # 4 wrist 1
            (0.90, 0.30, 0.90, 1.0),  # 5 wrist 2
            (0.30, 0.90, 0.90, 1.0),  # 6 hand
        ]

        a = [float(getattr(self.links[i], "a", 0.0)) for i in range(self.n)]
        d = [float(getattr(self.links[i], "d", 0.0)) for i in range(self.n)]

        # ----- frame 0: base column extending downward (root at top) -----
        L0 = max(abs(d[0]), fallback[0]) if self.n >= 1 else fallback[0]
        base = geometry.Cuboid(scale=(thick[0], thick[0], L0), color=colors[0])
        base.T = np.eye(4)
        self.links_3d.append(base)
        relation.append(spb.transl(0, 0, -L0 / 2.0))

        # ----- frames 1..n: default to X-bars using a[i-1] -----
        for i in range(1, self.n + 1):
            t   = thick[i] if i < len(thick) else thick[-1]
            col = colors[i] if i < len(colors) else (0.8, 0.8, 0.8, 1.0)
            Lx  = max(abs(a[i - 1]), fallback[i])
            cub = geometry.Cuboid(scale=(Lx, t, t), color=col)
            cub.T = np.eye(4)
            self.links_3d.append(cub)

            # anchor at the frame origin then shift by half-length along +X
            if i == 1:
                # first link sits on top of the base column
                relation.append(spb.transl(0, 0, +L0 / 2.0) @ spb.transl(Lx / 2.0, 0, 0))
            else:
                relation.append(spb.transl(Lx / 2.0, 0, 0))

        # ----- convert to Z-columns where a[i-1] == 0 (use d on previous frame) -----
        for i in range(1, self.n + 1):
            if abs(a[i - 1]) < 1e-6:
                Lz = max(abs(d[i - 1]), fallback[i - 1])
                t  = thick[i - 1] if (i - 1) < len(thick) else thick[-1]
                col = colors[i - 1] if (i - 1) < len(colors) else (0.8, 0.8, 0.8, 1.0)
                cubZ = geometry.Cuboid(scale=(t, t, Lz), color=col)
                cubZ.T = np.eye(4)
                # replace that link's visual with a Z column centered on its previous frame
                self.links_3d[i - 1] = cubZ
                relation[i - 1] = spb.transl(0, 0, Lz / 2.0)

        self._relation_matrices = relation

    # ===================== Kinematics (Standard DH only) =====================
    def _create_DH(self):
        # Standard DH parameters for JACO2
        a     = [0.0,   0.410, 0.0,   0.0,   0.0,  0.0]
        d     = [0.276, 0.0,   0.0,   0.207, 0.0,  0.103]
        alpha = [ pi/2,   pi,  pi/2, -pi/2,  pi/2,   pi]
        qlim  = [[-2*pi, 2*pi]] * 6
        return [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]) for i in range(6)]

    # ===================== Poses & Motion =====================
    def set_pose(self, name="home"):
        poses = {
            "home":     [0, -pi/3,  pi/3,  0,  -pi/4,  pi/4],
            "folded":   [0, -pi/2,  pi/2,  0,   0,      0  ],
            "reach":    [0, -pi/6,  pi/6,  0,   0,      0  ],
            "elbow_up": [0, -pi/2,  pi/3,  pi/6, -pi/4, pi/4],
        }
        self.q = poses.get(name, poses["home"])

    def demo_motion(self, env, steps=160):
        q1 = [0, -pi/2,  pi/2,  0,    0,    0]
        q2 = [0, -pi/6,  pi/6,  0, -pi/6, pi/6]
        path = rtb.jtraj(q1, q2, steps).q
        for q in path:
            self.q = q
            env.step(0.016)

    # ===================== Test Scene =====================
    def test(self, show_frames=True):
        env = swift.Swift()
        env.launch(realtime=True)
        self.set_pose("home")

        # place robot root at base top (shoulder height) by shifting +d[0]
        d0 = float(getattr(self.links[0], "d", 0.0))
        self.base = SE3(0.5, 0.5, 0.02) @ SE3.Tz(d0)
        self.add_to_env(env)

        # joint origin markers
        joint_markers = []
        for _ in range(self.n + 1):
            s = geometry.Sphere(0.015, color=(1.0, 0.9, 0.1, 1.0))
            env.add(s)
            joint_markers.append(s)

        # optional axes
        axes = []
        if show_frames:
            ax0 = geometry.Axes(0.06); env.add(ax0); axes.append(ax0)
            for _ in range(self.n):
                ax = geometry.Axes(0.06); env.add(ax); axes.append(ax)

        try:
            env.set_camera_pose(eye=[1.3, -1.1, 0.9], target=[0.5, 0.5, 0.8])
        except Exception:
            pass

        # small animation to verify linkage continuously
        q_goal = [self.q[i] + (-1)**i * (pi/6) for i in range(self.n)]
        traj = rtb.jtraj(self.q, q_goal, 100).q

        for q in traj:
            self.q = q
            T = SE3()
            joint_markers[0].T = (self.base @ T).A
            if show_frames: axes[0].T = (self.base @ T).A
            for i, L in enumerate(self.links, start=1):
                T = T @ L.A(q[i-1])
                joint_markers[i].T = (self.base @ T).A
                if show_frames: axes[i].T = (self.base @ T).A
            env.step(0.016)

        # extra motion demo
        self.demo_motion(env, steps=140)
        env.hold()

if __name__ == "__main__":
    r = JACO2_6s300()
    r.test()

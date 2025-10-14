#!/usr/bin/env python3
import os, time
from math import pi, sin, cos, tau
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D

class Load(DHRobot3D):
    def __init__(self):
        links = self._create_DH()

        link3D_names = dict(
            link0="base",
            link1="shoulder",
            link2="arm",
            link3="forearm",
            link4="wrist_spherical_1",
            link5="wrist_spherical_2",
            link6="hand_2finger",
        )

        here = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(here, "jacorobot")

        q_zero = [0]*6

        super().__init__(
            links,
            link3D_names,
            name="kinova3s600",
            link3d_dir=mesh_dir,
            qtest=q_zero,
            qtest_transforms=None,
        )

        self.q = q_zero
        self.tool_length = 0.12
        self.addconfiguration("tool_length", self.tool_length)

    # ---------------- DH model (your numbers kept) ----------------
    def _create_DH(self):
        a     = [0.0,   -0.410, 0.0, 0.0, 0.0, 0.4]
        d     = [0.2755, 0.0,   0.0, -0.31, 0.0, 0.0]
        alpha = [ pi/2, -pi/2,  pi/2, -pi/2,  pi/2, 0.0]
        offsets = [1.75, -pi/2, pi/2, 0.0, 0.0, 0.0]
        qlim  = [[-2*pi, 2*pi]] * 6
        return [
            rtb.RevoluteDH(a=a[i], d=d[i], alpha=alpha[i], offset=offsets[i], qlim=qlim[i])
            for i in range(6)
        ]

    # --------------- helpers: motion primitives -------------------
    def movej(self, env, q_goal, steps=160, dwell=0.0):
        """Joint-space move with trapezoidal-ish profile via jtraj."""
        q_start = np.array(self.q).astype(float)
        q_goal  = np.array(q_goal).astype(float)
        path = rtb.jtraj(q_start, q_goal, steps).q
        for q in path:
            self.q = q
            env.step()
        if dwell > 0: time.sleep(dwell)

    def track_cartesian_circle(self, env, center_T, radius=0.10, plane="xy",
                               period=6.0, cycles=1, dt=0.02, lam=0.05):
        """
        Resolved-rate (damped least squares) tracking of a circle.
        - center_T: SE3 of the circle center (tool pose orientation is used)
        - radius:   meters
        - plane:    "xy", "xz" or "yz"
        """
        # Keep current tool orientation; move position on circle
        def circle_point(theta):
            cx, cy, cz = center_T.t
            if plane == "xy":
                p = np.r_[cx + radius*cos(theta), cy + radius*sin(theta), cz]
            elif plane == "xz":
                p = np.r_[cx + radius*cos(theta), cy, cz + radius*sin(theta)]
            else:  # yz
                p = np.r_[cx, cy + radius*cos(theta), cz + radius*sin(theta)]
            return SE3.RPY(*center_T.rpy(order="xyz")) * SE3(p)  # keep ori

        total = int(cycles * period / dt)
        # Start exactly at theta=0 target
        T_target = circle_point(0.0)
        self._servo_to(env, T_target, dt, lam)

        for k in range(total):
            theta = tau * (k / (period/dt))  # 0→2π per period
            T_target = circle_point(theta)
            self._servo_to(env, T_target, dt, lam)

    def _servo_to(self, env, T_target, dt, lam):
        """
        One resolved-rate step to T_target using damped least squares.
        """
        # Current pose (base frame)
        T_now = self.fkine(self.q)
        # 6×1 twist error (vx, vy, vz, wx, wy, wz)
        e = spb.tr2delta(T_now.A, T_target.A)
        # Simple gain toward target (units: per second)
        K = 2.0
        v = K * np.asarray(e).reshape(6)

        # Jacobian at current config in base frame
        J = self.jacob0(self.q)
        # DLS: qdot = Jᵀ (J Jᵀ + λ² I)⁻¹ v
        JT = J.T
        JJt = J @ JT
        qdot = JT @ np.linalg.solve(JJt + (lam**2)*np.eye(6), v)

        # Integrate
        self.q = np.asarray(self.q) + qdot * dt
        env.step(dt)

    # ----------------------- fancy demo ---------------------------
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.6, 1.3, 1.4], [0, 0, pi/8])
        self.base = SE3(0, 0, 0)
        self.add_to_env(env)

        HOME = np.array(self.q)

        # 1) coordinated joint wave
        W = 120
        amps = np.array([20, 15, 12, 18, 22, 15]) * pi/180
        centers = HOME + np.array([0, -15, +10, 0, +15, 0]) * pi/180
        for k in range(3*W):
            phase = (k / W) * tau
            self.q = centers + amps * np.sin(phase + np.linspace(0, pi, 6))
            env.step(0.015)

        # 2) go to a staging pose
        STAGE = HOME + np.array([+30, -25, +20, -20, +25, -15]) * pi/180
        self.movej(env, STAGE, steps=200, dwell=0.2)

        # 3) Cartesian circle at current tool height, in the XY plane
        T_now = self.fkine(self.q)
        center = SE3(T_now.t[0], T_now.t[1], T_now.t[2]) * SE3.Tx(0.10)
        self.track_cartesian_circle(env, center_T=center, radius=0.08,
                                    plane="xy", period=5.0, cycles=2, dt=0.02, lam=0.06)

        # 4) short “home → pose → home” in joint space to finish
        POSE = HOME + np.array([-20, -10, +15, +10, -15, +25]) * pi/180
        self.movej(env, POSE, steps=180, dwell=0.15)
        self.movej(env, HOME, steps=220, dwell=0.0)

        time.sleep(0.2)

# ----------------------------- run ------------------------------
if __name__ == "__main__":
    Load().test()

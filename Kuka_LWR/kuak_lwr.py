#!/usr/bin/env python3
import os, time
from math import pi
import numpy as np

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D

# Change this to pick the animation: "sweeps", "wave", "keyframes", "ee_circle"
ANIM_MODE = "sweeps"

class Load(DHRobot3D):
    def __init__(self):
        # ---------- DH links ----------
        links = self._create_DH()

        # ---------- Mesh names (stems) ----------
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

        # ---------- Registration pose & transforms (your values) ----------
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
            name='LBR',                      # (name only; you can rename)
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest
        self.tool_length = 0.12
        self.addconfiguration("tool_length", self.tool_length)

    def _create_DH(self):
        # Your “working” DH for this model
        a     = [0, 0, 0, 0, 0, 0, 0]
        d     = [0.307, 0.013, 0.393, -0.013, 0.3769, -0.04, 0.1]
        alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0]
        qlim  = [[-pi, pi]] * 7
        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]) for i in range(7)]
        return links

    # ---------------- MAIN TEST (selects an animation) ----------------
    def test(self, mode=ANIM_MODE):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, pi/8])

        # start from the registration pose used to build qtest_transforms
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0)  # Offset robot for visibility
        self.add_to_env(env)

        if mode == "sweeps":
            self._anim_joint_sweeps(env)
        elif mode == "wave":
            self._anim_wave(env)
        elif mode == "keyframes":
            self._anim_keyframes(env)
        elif mode == "ee_circle":
            self._anim_ee_circle(env)
        else:
            print(f"[WARN] Unknown mode '{mode}', falling back to sweeps")
            self._anim_joint_sweeps(env)

        time.sleep(0.5)
        print("Animation complete")
        env.hold()

    # ---------------- ANIMATION HELPERS ----------------
    def _anim_joint_sweeps(self, env, amp=pi/2, steps=80, settle=0.3):
        """
        Sweep each joint +amp -> -amp about q=0 (others fixed).
        Good for checking each joint pivot and link attachment.
        """
        q0 = np.zeros(7)
        # move from current to q0 first
        for q in rtb.jtraj(self.q, q0, steps//2).q:
            self.q = q; env.step(0.02)

        for j in range(7):
            q_up  = q0.copy(); q_up[j]  = +amp
            q_dn  = q0.copy(); q_dn[j]  = -amp
            for q in rtb.jtraj(self.q, q_up, steps//2).q:  # up
                self.q = q; env.step(0.02)
            for q in rtb.jtraj(self.q, q_dn, steps).q:     # down
                self.q = q; env.step(0.02)
            for q in rtb.jtraj(self.q, q0, steps//2).q:    # back to zero
                self.q = q; env.step(0.02)
            time.sleep(settle)

    def _anim_wave(self, env, secs=6.0, dt=0.02, freq=0.5):
        """
        Coordinated sinusoid across all joints.
        """
        t = 0.0
        q_off = np.zeros(7)
        phases = np.linspace(0, np.pi, 7)
        amps = np.array([0.6, 0.4, 0.5, 0.4, 0.35, 0.3, 0.25])
        while t < secs:
            q = q_off + amps * np.sin(2*np.pi*freq*t + phases)
            self.q = q
            env.step(dt)
            t += dt

    def _anim_keyframes(self, env):
        """
        A few distinct poses blended with jtraj.
        """
        keys = [
            np.zeros(7),
            np.array([ 0.8, -0.6,  0.7, -0.8,  0.5, -0.4,  0.3]),
            np.array([-0.6,  0.5, -0.7,  0.6, -0.4,  0.6, -0.2]),
            np.zeros(7),
        ]
        for i in range(len(keys)-1):
            for q in rtb.jtraj(keys[i], keys[i+1], 90).q:
                self.q = q
                env.step(0.02)

    def _anim_ee_circle(self, env, radius=0.04, revs=1.0, dt=0.02):
        """
        Draw a small circle with the end-effector in the base XY plane about
        the start pose using resolved-rate control (Jacobian pseudoinverse).
        """
        # start at zero
        q = np.zeros(7)
        for qk in rtb.jtraj(self.q, q, 40).q:
            self.q = qk; env.step(0.02)

        T0 = self.fkine(self.q)     # SE3 of EE at start
        p0 = T0.t
        omega = 2*np.pi * 0.25      # angular speed
        total_T = (revs * 2*np.pi) / omega
        t = 0.0

        while t < total_T:
            vx = -radius * omega * np.sin(omega * t)
            vy =  radius * omega * np.cos(omega * t)
            vz = 0.0
            v_lin = np.array([vx, vy, vz])
            v_ang = np.zeros(3)     # keep orientation roughly fixed
            v6 = np.r_[v_lin, v_ang]

            J = self.jacob0(self.q)            # 6x7
            dq = np.linalg.pinv(J) @ v6        # 7
            # optional clamp
            dq = np.clip(dq, -0.6, 0.6)

            self.q = self.q + dq * dt
            env.step(dt)
            t += dt

if __name__ == "__main__":
    r = Load()
    r.test(mode=ANIM_MODE)

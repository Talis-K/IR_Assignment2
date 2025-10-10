#!/usr/bin/env python3
"""
KUKA LBR (iiwa-style) • DH kinematics • Swift visualization

- Realistic-ish visuals: tapered cylinders + rounded knuckles + silver collars,
  orange/white accents, domed base.
- Corrected DH orientation for an iiwa-like vertical stack.
- Smooth Cartesian sweep using LM inverse kinematics.

Run:
  python Kuka.py
"""

from math import pi, sin
import time
from typing import List

import numpy as np
import swift
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import spatialgeometry as geom


# ---------- helpers for geometry (work across spatialgeometry versions) ----------
def CYL(radius, length, color):
    try:
        return geom.Cylinder(radius, length, color=color)
    except TypeError:
        return geom.Cylinder(radius=radius, length=length, color=color)

def SPH(radius, color):
    try:
        return geom.Sphere(radius, color=color)
    except TypeError:
        return geom.Sphere(radius=radius, color=color)


class KukaLBR7_DH(DHRobot):
    """Approximate KUKA LBR iiwa-like 7-DoF arm using DH parameters."""

    def __init__(self):
        # ---------------- DH table (approximate, oriented like an iiwa) ----------
        #  i   d (m)   a (m)   alpha (rad)
        links = [
            RevoluteDH(d=0.34, a=0.00, alpha= +pi/2),  # base → shoulder (up)
            RevoluteDH(d=0.00, a=0.27, alpha= 0.0   ), # shoulder → upper arm (forward)
            RevoluteDH(d=0.00, a=0.03, alpha= +pi/2),  # twist
            RevoluteDH(d=0.30, a=0.00, alpha= -pi/2),  # elbow
            RevoluteDH(d=0.00, a=0.25, alpha= +pi/2),  # forearm
            RevoluteDH(d=0.00, a=0.02, alpha= -pi/2),  # wrist 1
            RevoluteDH(d=0.18, a=0.00, alpha= 0.0   ), # wrist 2 → flange
        ]
        super().__init__(links, name="KUKA_LBR7_SMOOTH")

        # Sit robot correctly on pedestal and yaw it to face +x
        self.base = SE3(0, 0, 0.12) * SE3.Rz(pi/2)

        # Home pose (gentle S-curve)
        self.q_home = np.array([0.0, -0.6, 0.5, -1.2, 0.8, 0.0, 0.0])

        # Visual containers
        self._parts: list[dict] = []      # link visuals (per-link components)
        self._base_parts: list = []       # pedestal/dome

        self._build_base_visuals()
        self._build_link_visuals()

    # ---------------- base visuals ----------------
    def _build_base_visuals(self):
        white  = [0.94, 0.94, 0.97, 1.0]
        silver = [0.75, 0.75, 0.78, 1.0]

        # Pedestal cylinder + domed cap
        pedestal = CYL(0.15, 0.16, color=white)
        pedestal.T = SE3(0, 0, 0.08-0.01)
        dome = SPH(0.15, color=white)
        dome.T = SE3(0, 0, 0.16-0.01)

        # Slim silver ring between pedestal and dome
        ring = CYL(0.155, 0.02, color=silver)
        ring.T = SE3(0, 0, 0.16-0.02)

        self._base_parts = [pedestal, dome, ring]

    # ---------------- link visuals ----------------
    def _build_link_visuals(self):
        # Slight taper proximal→distal, iiwa-ish radii
        radii_p = [0.080, 0.060, 0.055, 0.060, 0.055, 0.048, 0.046]
        radii_d = [r*0.92 for r in radii_p]
        white   = [0.94, 0.94, 0.97, 1.0]
        orange  = [1.00, 0.55, 0.10, 1.0]   # KUKA-ish orange
        silver  = [0.75, 0.75, 0.78, 1.0]

        self._parts.clear()

        for i, link in enumerate(self.links):
            a_i, d_i = float(link.a), float(link.d)
            rp, rd = radii_p[i], radii_d[i]

            # Decide axis/length and local center pose
            if abs(a_i) > 1e-6:            # link aligned with +x
                L = max(0.18, abs(a_i))
                T_center = SE3(a_i/2, 0, 0) * SE3.Ry(pi/2)  # rotate z→x
            elif abs(d_i) > 1e-6:          # link aligned with +z
                L = max(0.18, abs(d_i))
                T_center = SE3(0, 0, d_i/2)
            else:                           # spacer link
                L = 0.16
                T_center = SE3()

            # Two stacked cylinders to mimic a gentle taper
            body1 = CYL(rp, L*0.55, color=white)
            body2 = CYL(rd, L*0.45, color=white)
            self._parts.append({"shape": body1, "T_offset": T_center * SE3(0, 0, -L*0.225)})
            self._parts.append({"shape": body2, "T_offset": T_center * SE3(0, 0,  L*0.275)})

            # Rounded ends (knuckles)
            s1 = SPH(rp*0.98, color=white)
            s2 = SPH(rd*0.98, color=white)
            self._parts.append({"shape": s1, "T_offset": T_center * SE3(0, 0, -L/2)})
            self._parts.append({"shape": s2, "T_offset": T_center * SE3(0, 0,  L/2)})

            # Accent collar (alternate orange/silver like branding bands)
            band_color = orange if i % 2 == 0 else silver
            collar = CYL(max(rp, rd)*1.04, L*0.065, color=band_color)
            self._parts.append({"shape": collar, "T_offset": T_center * SE3(0, 0,  L*0.32)})

        # Tool flange (small stub)
        tool = CYL(0.030, 0.12, color=silver)
        self._parts.append({"shape": tool, "T_offset": SE3()})

    # ---------------- rendering ----------------
    def add_to_swift(self, env: swift.Swift):
        for obj in self._base_parts:
            env.add(obj)
        for item in self._parts:
            env.add(item["shape"])

    def update_visuals(self, q: np.ndarray):
        Ts: List[SE3] = self.fkine_all(q)     # base→each link end frame
        # Parts per link: 5 (2 cylinders + 2 spheres + 1 collar)
        ppl = 5
        for i, linkT in enumerate(Ts):
            start, end = i*ppl, i*ppl + ppl
            for item in self._parts[start:end]:
                item["shape"].T = self.base * linkT * item["T_offset"]
        # Tool at wrist (last element)
        self._parts[-1]["shape"].T = self.base * Ts[-1] * self._parts[-1]["T_offset"]


# ---------- motions ----------
def sweep_wave(robot: KukaLBR7_DH, env: swift.Swift,
               amplitude=0.20, length=0.60, height=0.55,
               duration=10.0, freq=2.0):
    """
    Smooth sinusoidal sweep across a band in front of the robot.
    End-effector follows: x(t) in [-L/2, L/2], y(t) = A*sin(omega t) at z=height.
    """
    t0 = time.time()
    q = robot.q_home.copy()
    while True:
        t = time.time() - t0
        if t > duration:
            break
        # Linear progress along X, sinusoid in Y
        s = min(t / duration, 1.0)
        x = -length/2 + length*s
        y = amplitude * np.sin(2*np.pi*freq*s)
        T = SE3(0.35, 0, 0) * SE3(x, y, height)
        sol = robot.ikine_LM(T, q0=q, ilimit=80, slimit=20)
        if sol.success:
            q = sol.q
        robot.update_visuals(q)
        env.step(0.015)


def joint_wiggle(robot: KukaLBR7_DH, env: swift.Swift, seconds=3.0):
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t > seconds:
            break
        q = robot.q_home + np.array([
            0.25*sin(0.7*t),
            0.20*sin(0.9*t + 0.5),
            0.25*sin(0.8*t + 1.0),
            0.35*sin(1.0*t + 1.2),
            0.30*sin(1.1*t + 0.3),
            0.40*sin(1.3*t - 0.4),
            0.40*sin(1.5*t + 0.8),
        ])
        robot.update_visuals(q)
        env.step(0.02)


# ---------- main ----------
if __name__ == "__main__":
    robot = KukaLBR7_DH()

    env = swift.Swift()
    env.launch(realtime=True)

    # Floor
    try:
        floor = geom.Cuboid((2.0, 2.0, 0.02))
    except TypeError:
        floor = geom.Cuboid(side=[2.0, 2.0, 0.02])
    floor.T = SE3(0, 0, -0.01)
    env.add(floor)

    robot.add_to_swift(env)
    robot.update_visuals(robot.q_home)

    # Show life, then sweep
    joint_wiggle(robot, env, seconds=2.5)
    sweep_wave(robot, env, amplitude=0.22, length=0.70, height=0.58,
               duration=12.0, freq=3.0)

    print("Done.")

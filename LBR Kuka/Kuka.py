#!/usr/bin/env python3
"""
KUKA LBR iiwa 14 R820 • Standard DH (all a_i=0) • Swift viz
- Dimensions aligned with widely used iiwa-14 community/ROS models:
    d1=0.360 m, d3=0.420 m, d5=0.400 m, flange de=0.126 m
- Alternating ±pi/2 twists per the published non-offset DH layout
- Smooth visual model (tapered links, rounded knuckles, accent bands)
- Cartesian sweep demo with IK

Requires:
  pip install roboticstoolbox-python spatialgeometry spatialmath-python swift
"""

from math import pi, sin
import time
from typing import List

import numpy as np
import swift
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import spatialgeometry as geom


# ----- geometry helpers (handle minor API diffs) -----------------------
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


# ===== Robot ===========================================================
class KukaLBR14_DH(DHRobot):
    """
    KUKA LBR iiwa 14 R820 with STANDARD DH:

      i   θi    di      ai   αi
      1   q1    d1      0   +π/2
      2   q2    0       0   −π/2
      3   q3    d3      0   −π/2
      4   q4    0       0   +π/2
      5   q5    d5      0   +π/2
      6   q6    0       0   −π/2
      7   q7    0       0    0
      e         de      0    0      (tool flange offset)

    All a_i = 0 (non-offset chain).  d1,d3,d5, and de carry the geometry.
    """

    # --- geometric constants (meters) ---
    D1 = 0.360   # base -> shoulder can (vertical)
    D3 = 0.420   # upper arm tube (vertical in its local z when q2=0)
    D5 = 0.400   # forearm tube
    DE = 0.126   # tool flange / adapter depth

    def __init__(self):
        links = [
            RevoluteDH(d=self.D1, a=0.0, alpha= +pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=0.0,     a=0.0, alpha= -pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=self.D3, a=0.0, alpha= -pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=0.0,     a=0.0, alpha= +pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=self.D5, a=0.0, alpha= +pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=0.0,     a=0.0, alpha= -pi/2, qlim=[-pi, pi]),
            RevoluteDH(d=0.0,     a=0.0, alpha= 0.0,   qlim=[-pi, pi]),
        ]
        super().__init__(links, name="KUKA_LBR_14_R820_DH")

        # Lift the whole robot to sit on the pedestal. (No extra yaw/roll!)
        self.base = SE3(0, 0, 0.12)

        # Reasonable upright home pose for the classic “S” shape
        self.q_home = np.array([0.0, -0.8, +0.9, -1.35, +0.95, -0.25, 0.0])

        # Visual objects
        self._base_parts = []
        self._parts: list[dict] = []

        self._build_base_visuals()
        self._build_link_visuals()

    # ------------- base visuals -------------
    def _build_base_visuals(self):
        white  = [0.94, 0.94, 0.97, 1.0]
        silver = [0.75, 0.75, 0.78, 1.0]

        pedestal = CYL(0.155, 0.18, color=white); pedestal.T = SE3(0, 0, 0.09-0.01)
        dome     = SPH(0.155, color=white);       dome.T     = SE3(0, 0, 0.18-0.01)
        ring     = CYL(0.160, 0.02, color=silver); ring.T     = SE3(0, 0, 0.18-0.02)
        self._base_parts = [pedestal, dome, ring]

    # ------------- link visuals -------------
    def _build_link_visuals(self):
        # Proportions similar to photos: slight taper and rounded “knuckles”
        radii_p = [0.080, 0.062, 0.058, 0.062, 0.056, 0.050, 0.046]
        radii_d = [r*0.92 for r in radii_p]
        white   = [0.94, 0.94, 0.97, 1.0]
        orange  = [1.00, 0.55, 0.10, 1.0]
        silver  = [0.75, 0.75, 0.78, 1.0]

        # Effective “lengths” per link when a_i=0:
        # whenever d_i > 0, the visual is drawn along +z of that link
        d_vals = [self.D1, 0.0, self.D3, 0.0, self.D5, 0.0, 0.0]

        self._parts.clear()
        for i, (link, Ld) in enumerate(zip(self.links, d_vals)):
            rp, rd = radii_p[i], radii_d[i]
            if Ld <= 1e-6:
                L = 0.18
                T_center = SE3()                     # short spacer at joint
            else:
                L = Ld
                T_center = SE3(0, 0, L/2)           # extend along local +z

            # Tapered body (two cylinders)
            body1 = CYL(rp, L*0.55, color=white)
            body2 = CYL(rd, L*0.45, color=white)
            self._parts.append({"shape": body1, "T_offset": T_center * SE3(0, 0, -L*0.225)})
            self._parts.append({"shape": body2, "T_offset": T_center * SE3(0, 0,  L*0.275)})

            # Rounded ends
            s1 = SPH(rp*0.98, color=white)
            s2 = SPH(rd*0.98, color=white)
            self._parts.append({"shape": s1, "T_offset": T_center * SE3(0, 0, -L/2)})
            self._parts.append({"shape": s2, "T_offset": T_center * SE3(0, 0,  L/2)})

            # Accent collar
            band_color = orange if i % 2 == 0 else silver
            collar = CYL(max(rp, rd)*1.04, max(0.04, L*0.065), color=band_color)
            self._parts.append({"shape": collar, "T_offset": T_center * SE3(0, 0,  L*0.33)})

        # Tool flange (fixed offset DE on the end frame)
        self.tool_offset = SE3(0, 0, self.DE)
        tool = CYL(0.030, 0.12, color=silver)
        self._parts.append({"shape": tool, "T_offset": self.tool_offset})

    # ------------- rendering -------------
    def add_to_swift(self, env: swift.Swift):
        for o in self._base_parts:
            env.add(o)
        for item in self._parts:
            env.add(item["shape"])

    def update_visuals(self, q: np.ndarray):
        # fkine_all already includes self.base
        Ts: List[SE3] = self.fkine_all(q)           # base -> each link-end
        ppl = 5                                      # parts per link
        for i, linkT in enumerate(Ts):
            start, end = i*ppl, i*ppl + ppl
            for item in self._parts[start:end]:
                item["shape"].T = linkT * item["T_offset"]
        # Tool on the last link end:
        self._parts[-1]["shape"].T = Ts[-1] * self._parts[-1]["T_offset"]


# ===== Motions =========================================================
def sweep(robot: KukaLBR14_DH, env: swift.Swift,
          amplitude=0.22, length=0.72, height=0.62,
          duration=12.0, ripples=3.0):
    """Cartesian sweep (sinusoid in Y while moving in X)."""
    t0 = time.time()
    q = robot.q_home.copy()
    while True:
        t = time.time() - t0
        if t > duration:
            break
        s = min(t/duration, 1.0)
        x = -length/2 + length*s
        y = amplitude*np.sin(2*np.pi*ripples*s)
        T = SE3(0.35, 0, 0) * SE3(x, y, height)  # in front of base
        sol = robot.ikine_LM(T, q0=q, ilimit=80, slimit=20)
        if sol.success:
            q = sol.q
        robot.update_visuals(q)
        env.step(0.015)


def wiggle(robot: KukaLBR14_DH, env: swift.Swift, seconds=2.0):
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


# ===== Main ============================================================
if __name__ == "__main__":
    robot = KukaLBR14_DH()

    env = swift.Swift()
    env.launch(realtime=True)

    # floor
    try:
        floor = geom.Cuboid((2.2, 2.2, 0.02))
    except TypeError:
        floor = geom.Cuboid(side=[2.2, 2.2, 0.02])
    floor.T = SE3(0, 0, -0.01)
    env.add(floor)

    robot.add_to_swift(env)
    robot.update_visuals(robot.q_home)

    wiggle(robot, env, seconds=2.0)
    sweep(robot, env, amplitude=0.22, length=0.72, height=0.62,
          duration=12.0, ripples=3.0)

    print("Done.")

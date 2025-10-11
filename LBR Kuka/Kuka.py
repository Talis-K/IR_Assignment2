#!/usr/bin/env python3
"""
KUKA LBR iiwa • Standard DH (all a_i=0) • Swift viz • DAE meshes from ./meshes

- Uses your meshes in: LBR Kuka/meshes/{base_link.DAE, link1.DAE ... link7.DAE}
- Robust to spatialgeometry versions (scale as 3-vector, alt constructors)
- No double-base bug: fkine_all(q) already includes robot.base
- Simple wiggle + Cartesian sweep (IK) demo

Run (inside your conda env):
  python "/Users/ihtishammazid/Documents/GitHub/IR_Assignment2/LBR Kuka/Kuka.py"
"""

from math import pi, sin
import os, time
from typing import List

import numpy as np
import swift
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import spatialgeometry as geom


# ---------- helpers ----------
def _scale3(s):
    """Ensure scale is a length-3 tuple for spatialgeometry."""
    if isinstance(s, (int, float)):
        s = float(s)
        return (s, s, s)
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (float(s[0]), float(s[1]), float(s[2]))
    raise ValueError("scale must be a number or a length-3 (sx,sy,sz)")

def load_mesh(path, scale=1.0, color=None):
    """Load DAE/STL robustly across spatialgeometry versions."""
    sc = _scale3(scale)
    m = None
    try:
        m = geom.Mesh(path, scale=sc, color=color)                 # common API
    except TypeError:
        try:
            m = geom.Mesh(filename=path, scale=sc, color=color)    # alt API
        except TypeError:
            m = geom.Mesh(path)                                    # last resort
            try:
                m.scale = sc
                if color is not None:
                    m.color = color
            except Exception:
                pass
    return m

def CYL(r, L, color):
    try:
        return geom.Cylinder(r, L, color=color)
    except TypeError:
        return geom.Cylinder(radius=r, length=L, color=color)

def SPH(r, color):
    try:
        return geom.Sphere(r, color=color)
    except TypeError:
        return geom.Sphere(radius=r, color=color)


# ---------- file locations (this file lives in: .../IR_Assignment2/LBR Kuka/Kuka.py) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "meshes")  # contains base_link.DAE, link1.DAE ... link7.DAE

# Many DAE files are Y-up; robots are Z-up. Rotate mesh by Rx(-90°) to convert.
YUP_TO_ZUP = SE3.Rx(0)   # if your meshes are already Z-up, keep identity

# ---------- NEW: easy per-link mesh offsets (meters / radians) ----------
# Edit these to nudge individual meshes without touching DH.
# Example tweak: MESH_OFFSETS["3"] = {"xyz": (0.00, 0.00, 0.03), "rpy": (0.0, 0.0, 10*pi/180)}
MESH_OFFSETS = {
    "base": {"xyz": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)},
    "1":    {"xyz": (0.0, 0.0, 0.3), "rpy": (0.0, 0.0, 0.0)},
    "2":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
    "3":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
    "4":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
    "5":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
    "6":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
    "7":    {"xyz": (0.0, 0.0, 2.0), "rpy": (0.0, 0.0, 0.0)},
}

def _Toffset(key: str) -> SE3:
    """Build SE3 from per-link xyz+rpy and include the Y-up->Z-up fix."""
    xyz = MESH_OFFSETS[key]["xyz"]
    rpy = MESH_OFFSETS[key]["rpy"]
    return YUP_TO_ZUP * SE3(*xyz) * SE3.RPY(*rpy, order="xyz")

# Map base + links to meshes (use .DAE with exact names present in your repo)
MESH_MAP = {
    "base": dict(
        filename="base_link.DAE",
        scale=0.001,                    # mm→m (if your DAE is meters, set 1.0)
        T_offset=_Toffset("base"),
        color=[0.94, 0.94, 0.97, 1.0],
    ),
    1: dict(filename="link1.DAE", scale=0.001, T_offset=_Toffset("1"), color=[0.92,0.92,0.95,1.0]),
    2: dict(filename="link2.DAE", scale=0.001, T_offset=_Toffset("2"), color=[0.92,0.92,0.95,1.0]),
    3: dict(filename="link3.DAE", scale=0.001, T_offset=_Toffset("3"), color=[0.92,0.92,0.95,1.0]),
    4: dict(filename="link4.DAE", scale=0.001, T_offset=_Toffset("4"), color=[0.92,0.92,0.95,1.0]),
    5: dict(filename="link5.DAE", scale=0.001, T_offset=_Toffset("5"), color=[0.92,0.92,0.95,1.0]),
    6: dict(filename="link6.DAE", scale=0.001, T_offset=_Toffset("6"), color=[0.92,0.92,0.95,1.0]),
    7: dict(filename="link7.DAE", scale=0.001, T_offset=_Toffset("7"), color=[0.92,0.92,0.95,1.0]),
    # optional tool mesh on link 7:
    "tool": dict(
        filename=None,                  # e.g., "tool_flange.DAE"
        scale=0.001,
        T_offset=YUP_TO_ZUP * SE3(0, 0, 0),  # adjust if you add a real flange mesh
        color=[0.75, 0.75, 0.78, 1.0],
    ),
}


# ======================================================================
# Robot: LBR iiwa using Standard DH (all a_i=0) with real meshes
# ======================================================================
class KukaLBR_DH(DHRobot):
    """
    Standard DH (all a_i=0). Non-zero vertical link lengths d1, d3, d5
    match the common LBR-14 R820 community models.

      i   θi    di      ai   αi
      1   q1    d1      0   +π/2
      2   q2    0       0   −π/2
      3   q3    d3      0   −π/2
      4   q4    0       0   +π/2
      5   q5    d5      0   +π/2
      6   q6    0       0   −π/2
      7   q7    0       0    0
    """

    # Commonly used LBR-14 dimensions (meters)
    D1 = 0.360
    D3 = 0.420
    D5 = 0.400

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
        super().__init__(links, name="KUKA_LBR_DH_with_meshes")

        # Lift onto pedestal; fkine_all(q) includes this base automatically
        self.base = SE3(0, 0, 0.12)

        # A nice upright “S” home
        self.q_home = np.array([0.0, -0.85, +0.95, -1.35, +0.95, -0.25, 0.0])

        # Visual registries
        self._base_visuals = []
        self._link_visuals = []   # list of dict(shape, T_offset)
        self._tool_visual = None
        self._tool_T_offset = SE3()

        self._build_base_visual()
        self._build_link_visuals()

    # ---------- visuals ----------
    def _build_base_visual(self):
        binfo = MESH_MAP["base"]
        bpath = os.path.join(MESH_DIR, binfo["filename"]) if binfo["filename"] else None
        if bpath and os.path.exists(bpath):
            m = load_mesh(bpath, scale=binfo["scale"], color=binfo["color"])
            m.T = self.base * binfo["T_offset"]
            self._base_visuals = [m]
        else:
            # Fallback pedestal + dome
            white, silver = [0.94,0.94,0.97,1.0], [0.75,0.75,0.78,1.0]
            p = CYL(0.155, 0.18, color=white); p.T = self.base * SE3(0,0,0.09-0.01)
            d = SPH(0.155, color=white);       d.T = self.base * SE3(0,0,0.18-0.01)
            r = CYL(0.160, 0.02, color=silver); r.T = self.base * SE3(0,0,0.18-0.02)
            self._base_visuals = [p, d, r]

    def _build_link_visuals(self):
        self._link_visuals.clear()

        # Fallback sizes if a link mesh is absent
        fallback_r = [0.080, 0.062, 0.058, 0.062, 0.056, 0.050, 0.046]
        fallback_L = [self.D1, 0.18, self.D3, 0.18, self.D5, 0.16, 0.14]

        for i, link in enumerate(self.links, start=1):
            info = MESH_MAP.get(i, None)
            fpath = os.path.join(MESH_DIR, info["filename"]) if (info and info["filename"]) else None
            if fpath and os.path.exists(fpath):
                mesh = load_mesh(fpath, scale=info["scale"], color=info["color"])
                T_off = info["T_offset"]
                self._link_visuals.append(dict(shape=mesh, T_offset=T_off))
            else:
                # Simple fallback: cylinder along +z of link end
                L = fallback_L[i-1]; r = fallback_r[i-1]
                body = CYL(r, L, color=[0.94,0.94,0.97,1.0]); body.T = SE3(0,0,L/2)
                self._link_visuals.append(dict(shape=body, T_offset=SE3()))

        # Optional tool on link 7
        tinfo = MESH_MAP["tool"]
        tpath = os.path.join(MESH_DIR, tinfo["filename"]) if tinfo["filename"] else None
        if tpath and os.path.exists(tpath):
            self._tool_visual = load_mesh(tpath, scale=tinfo["scale"], color=tinfo["color"])
        else:
            self._tool_visual = CYL(0.03, 0.12, color=tinfo["color"])
        self._tool_T_offset = tinfo["T_offset"]

    # ---------- rendering ----------
    def add_to_swift(self, env: swift.Swift):
        for o in self._base_visuals:
            env.add(o)
        for item in self._link_visuals:
            env.add(item["shape"])
        env.add(self._tool_visual)

    def update_visuals(self, q: np.ndarray):
        """
        Place each link's mesh at: T_link_end * T_offset
        NOTE: fkine_all(q) already includes self.base.
        """
        Ts: List[SE3] = self.fkine_all(q)
        for T_link, vis in zip(Ts, self._link_visuals):
            vis["shape"].T = T_link * vis["T_offset"]
        # Tool on end of link 7:
        self._tool_visual.T = Ts[-1] * self._tool_T_offset


# ---------- motions ----------
def wiggle(robot: KukaLBR_DH, env: swift.Swift, seconds=2.0):
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t > seconds: break
        q = robot.q_home + np.array([
            0.25*sin(0.7*t),
            0.20*sin(0.9*t + 0.4),
            0.25*sin(0.8*t + 1.0),
            0.35*sin(1.0*t + 1.2),
            0.30*sin(1.1*t + 0.3),
            0.35*sin(1.3*t - 0.4),
            0.40*sin(1.5*t + 0.8),
        ])
        robot.update_visuals(q)
        env.step(0.02)

def sweep(robot: KukaLBR_DH, env: swift.Swift,
          amplitude=0.20, length=0.70, height=0.62,
          duration=10.0, ripples=3.0):
    """Sinusoidal sweep with IK in front of the base."""
    t0 = time.time(); q = robot.q_home.copy()
    while True:
        t = time.time() - t0
        if t > duration: break
        s = min(t/duration, 1.0)
        x = -length/2 + length*s
        y = amplitude*np.sin(2*np.pi*ripples*s)
        T = SE3(0.35, 0, 0) * SE3(x, y, height)
        sol = robot.ikine_LM(T, q0=q, ilimit=80, slimit=20)
        if sol.success:
            q = sol.q
        robot.update_visuals(q)
        env.step(0.015)


# ---------- main ----------
if __name__ == "__main__":
    robot = KukaLBR_DH()

    env = swift.Swift()
    env.launch(realtime=True)

    # simple floor
    floor = geom.Cuboid((2.0, 2.0, 0.02))
    floor.T = SE3(0, 0, -0.01)
    env.add(floor)

    robot.add_to_swift(env)
    robot.update_visuals(robot.q_home)

    wiggle(robot, env, seconds=2.0)
    sweep(robot, env, amplitude=0.22, length=0.72, height=0.62, duration=12.0, ripples=3.0)

    print("Done.")

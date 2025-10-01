#!/usr/bin/env python
"""
Visualizes the KUKA KR6 robot in Swift simulator
using DHRobot and provided mesh files.
"""
import numpy as np
import swift
import os
from roboticstoolbox import DHRobot, RevoluteMDH, jtraj
from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi


class KukaKR6(DHRobot):
    def __init__(self, meshdir=None):
        # KUKA KR6 standard DH parameters
        # [d, a, alpha] taken from datasheet (approx)
        links = [
            RevoluteMDH(d=0.400, a=0.025, alpha=pi/2, qlim=[-185*pi/180, 185*pi/180]),  # Joint 1
            RevoluteMDH(d=0.0,   a=0.455, alpha=0,    qlim=[-135*pi/180, 35*pi/180]),   # Joint 2
            RevoluteMDH(d=0.0,   a=0.035, alpha=pi/2, qlim=[-120*pi/180, 158*pi/180]),  # Joint 3
            RevoluteMDH(d=0.420, a=0.0,   alpha=-pi/2,qlim=[-350*pi/180, 350*pi/180]),  # Joint 4
            RevoluteMDH(d=0.0,   a=0.0,   alpha=pi/2, qlim=[-125*pi/180, 125*pi/180]),  # Joint 5
            RevoluteMDH(d=0.080, a=0.0,   alpha=0,    qlim=[-350*pi/180, 350*pi/180])   # Joint 6
        ]

        super().__init__(links, name="KUKA_KR6", manufacturer="KUKA")

        # Default pose
        self.q = [0, -pi/4, pi/4, 0, pi/6, 0]

        # Mesh folder
        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))

        # Base mesh
        base_mesh = os.path.join(meshdir, "base_link.dae")
        if os.path.exists(base_mesh):
            self.base = Mesh(base_mesh, color=(0.2, 0.2, 0.2, 1))
            print(f"Loaded base mesh: {base_mesh}")
        else:
            print(f"Warning: base mesh not found: {base_mesh}")

        # Load link meshes (link_1 … link_6)
        for i, link in enumerate(self.links, start=1):
            mesh_path = os.path.join(meshdir, f"link_{i}.dae")
            if os.path.exists(mesh_path):
                link.geometry = [Mesh(mesh_path)]
                print(f"Loaded mesh for link {i}: {mesh_path}")
            else:
                print(f"Warning: mesh not found for link {i}: {mesh_path}")


class LabAssignment:
    def __init__(self):
        # Swift environment
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        # Mesh directory (adjust path to where your meshes are)
        mesh_dir = os.path.join(os.getcwd(), "KUKA", "meshes")

        # Create robot
        self.robot = KukaKR6(meshdir=mesh_dir)

        # Add base if available
        if hasattr(self.robot, "base"):
            self.env.add(self.robot.base)

        # Add robot
        self.env.add(self.robot)

        # Trajectory (start → end)
        q_start = self.robot.q
        q_end = np.array([0, -pi/3, pi/6, -pi/2, pi/4, 0])
        t = np.linspace(0, 3, 100)
        traj = jtraj(q_start, q_end, t)

        for q in traj.q:
            self.robot.q = q
            self.env.step(0.03)


if __name__ == "__main__":
    assignment = LabAssignment()
    assignment.env.hold()

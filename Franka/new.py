#!/usr/bin/env python
"""
Visualizes the Franka Emika Panda robot in Swift simulator
using DHRobot and meshes from franka_ros repository.
"""
import numpy as np
import swift
import os
from roboticstoolbox import DHRobot, RevoluteMDH, jtraj
from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi

class FrankaPanda(DHRobot):
    def __init__(self, meshdir=None):
        # Panda DH parameters
        links = [
            RevoluteMDH(d=0.0, a=0,      alpha=0,      qlim=[0,0]),   # Joint 0 (base)
            RevoluteMDH(d=0.333, a=0,      alpha=0,      qlim=[-2.897, 2.897]),    # Joint 1
            RevoluteMDH(d=0.0,   a=0,      alpha=-pi/2,  qlim=[-1.7628, 1.7628]), # Joint 2
            RevoluteMDH(d=0.316, a=0,      alpha=pi/2,   qlim=[-2.8973, 2.8973]), # Joint 3
            RevoluteMDH(d=0.0,   a=0.0825, alpha=pi/2,   qlim=[-3.0718, -0.0698]),# Joint 4
            RevoluteMDH(d=0.384, a=-0.0825,alpha=-pi/2,  qlim=[-2.8973, 2.8973]), # Joint 5
            RevoluteMDH(d=0.0,   a=0,      alpha=pi/2,   qlim=[-0.0175, 3.7525]), # Joint 6
            RevoluteMDH(d=0.0,   a=0.088,  alpha=pi/2,   qlim=[-2.8973, 2.8973])  # Joint 7
        ]

        super().__init__(links, name="Franka_Panda", manufacturer="Franka Emika")

        # Default joint position
        self.q = [0, 0, -pi/4, 0, -pi/2, 0, pi/3, 0]

        # Mesh folder
        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))
        
        # Load meshes
        for i, link in enumerate(self.links, start=0):
            mesh_path = os.path.join(meshdir, f"link{i}.dae")  # or .dae depending on repo
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

        # Mesh directory: point this to franka_ros meshes
        mesh_dir = os.path.join(os.getcwd(), "franka", "meshes", "visual")

        # Create robot
        self.robot = FrankaPanda(meshdir=mesh_dir)

        # Add robot
        self.env.add(self.robot)

        # Simple trajectory
        q_start = self.robot.q
        q_end = np.array([0, 0, -pi/3, 0, -pi/2, 0, pi/4, 0])
        t = np.linspace(0, 3, 100)
        traj = jtraj(q_start, q_end, t)

        for q in traj.q:
            self.robot.q = q
            self.env.step(0.03)

if __name__ == "__main__":
    assignment = LabAssignment()
    assignment.env.hold()

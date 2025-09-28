#!/usr/bin/env python
"""
Visualizes a KUKA KR6 R900-2 robot in Swift simulator with correct link positioning using DHRobot.
"""
import numpy as np
import swift
import os
from roboticstoolbox import DHRobot, RevoluteDH
from spatialgeometry import Mesh
from roboticstoolbox import jtraj
from spatialmath import SE3
from math import pi

class KUKAKR6(DHRobot):
    def __init__(self, meshdir=None):
        # KUKA KR6 R900-2 DH parameters (based on opw_parameters_kr6r900_2.yaml)
        links = [
            RevoluteDH(d=0.400, a=0.025, alpha=pi/2, qlim=[-pi, pi]),  # Link 1
            RevoluteDH(d=0.0, a=0.455, alpha=0, qlim=[-pi*105/180, pi*105/180]),  # Link 2
            RevoluteDH(d=0.0, a=-0.025, alpha=pi/2, qlim=[-pi*155/180, pi*155/180]),  # Link 3
            RevoluteDH(d=0.420, a=0.0, alpha=-pi/2, qlim=[-pi*185/180, pi*185/180]),  # Link 4
            RevoluteDH(d=0.0, a=0.0, alpha=pi/2, qlim=[-pi*120/180, pi*120/180]),  # Link 5
            RevoluteDH(d=0.090, a=0.0, alpha=0, qlim=[-pi, pi])  # Link 6
        ]
        
        # Apply joint sign corrections
        sign_corrections = [-1, 1, 1, -1, 1, -1]
        for i, link in enumerate(links):
            link.theta = link.theta * sign_corrections[i]
        
        # Define mesh directory
        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))
        
        super().__init__(links, name="KUKA_KR6_R900_2", manufacturer="KUKA")
        
        # A joint config and corresponding transforms
        qtest = [0, -pi/2, 0, 0, 0, 0]
        qtest_transforms = [
            SE3(),  # Base
            SE3(0.025, 0, 0.4) * SE3.Rz(pi),  # Link 1
            SE3(0.48, 0, 0.4) * SE3.Rz(pi),  # Link 2
            SE3(0.455, 0, 0.4) * SE3.Rz(pi),  # Link 3
            SE3(0.455, 0, -0.02) * SE3.Rz(pi),  # Link 4
            SE3(0.455, 0, -0.02) * SE3.Rz(pi),  # Link 5
            SE3(0.455, 0, -0.11) * SE3.Rz(pi)  # Link 6
        ]
        
        # Load base mesh
        base_path = os.path.join(meshdir, "base_link.dae")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Missing mesh: {base_path}")
        self._base_mesh = Mesh(base_path, pose=qtest_transforms[0])
        print(f"Loaded base mesh: {base_path}")
        
        # Load link meshes
        for i, link in enumerate(self.links, start=1):
            mesh_path = os.path.join(meshdir, f"link_{i}.dae")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Missing mesh: {mesh_path}")
            print(f"Loaded mesh for link {i}: {mesh_path}")
            link.geometry.append(Mesh(mesh_path, pose=qtest_transforms[i]))
            print(f"Link {i} transform at qtest: {qtest_transforms[i].A}")
        
        self.q = qtest

class LabAssignment2:
    def __init__(self):
        # Create and launch Swift environment
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        
        # Current directory for mesh files
        mesh_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create KUKA KR6 robot
        self.robot = KUKAKR6(meshdir=mesh_dir)
        
        # Add base mesh and robot to environment
        self.env.add(self.robot._base_mesh)
        self.env.add(self.robot)
        
        # Define a simple joint trajectory with sign corrections
        sign_corrections = [-1, 1, 1, -1, 1, -1]
        q_start = np.zeros(6) * sign_corrections
        q_end = np.array([pi/4, -pi/6, pi/6, 0, pi/4, 0]) * sign_corrections
        t = np.linspace(0, 2, 50)  # 2-second motion
        traj = jtraj(q_start, q_end, t)
        
        # Animate the robot
        for q in traj.q:
            self.robot.q = q
            self.env.step(0.04)  # 40ms per step

if __name__ == "__main__":
    try:
        assignment = LabAssignment2()
        assignment.env.hold()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
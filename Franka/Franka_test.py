import numpy as np
import swift
import os
from roboticstoolbox import DHRobot, RevoluteDH
from spatialgeometry import Mesh
from roboticstoolbox import jtraj
from spatialmath import SE3
from math import pi
import yaml

class Franka(DHRobot):
    def __init__(self, meshdir=None):
        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))

        # --- Load joint limits ---
        yaml_path = os.path.join(meshdir, "joint_limits.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        lowers, uppers = [], []
        for _, joint_data in data.items():
            lowers.append(joint_data['limit']['lower'])
            uppers.append(joint_data['limit']['upper'])
        qlim = np.array([lowers, uppers])

        # --- Define links with limits ---
        links = [
            RevoluteDH(d=0.333, a=0, alpha=-pi/2, qlim=[lowers[0], uppers[0]]),
            RevoluteDH(d=0, a=0, alpha=pi/2, qlim=[lowers[1], uppers[1]]),
            RevoluteDH(d=0.316, a=0.0825, alpha=pi/2, qlim=[lowers[2], uppers[2]]),
            RevoluteDH(d=0, a=-0.0825, alpha=-pi/2, qlim=[lowers[3], uppers[3]]),
            RevoluteDH(d=0.384, a=0, alpha=pi/2, qlim=[lowers[4], uppers[4]]),
            RevoluteDH(d=0, a=0.088, alpha=pi/2, qlim=[lowers[5], uppers[5]]),
            RevoluteDH(d=0.107, a=0, alpha=0, qlim=[lowers[6], uppers[6]])
        ]

        # --- Call parent constructor ---
        super().__init__(links, name="Franka", manufacturer="Franka Emika")

        # --- Load meshes ---
        qtest = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]
        qtest_transforms = [
            SE3(),
            SE3(0, 0, 0.333) * SE3.Rz(pi),
            SE3(0, 0, 0.333) * SE3.Rz(pi/2) * SE3.Tx(0.0825),
            SE3(0.0825, 0, 0.649) * SE3.Rz(pi/2),
            SE3(0.0825, 0, 0.333) * SE3.Rz(pi/2),
            SE3(0.0825, 0, 0.717) * SE3.Rz(pi/2),
            SE3(0.0825, 0.088, 0.717) * SE3.Rz(pi/2),
            SE3(0.0825, 0.088, 0.824) * SE3.Rz(pi/2)
        ]

        for i, link in enumerate(self.links, start=0):
            mesh_path = os.path.join(meshdir, f"meshes/visual/link{i}.dae")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Missing mesh: {mesh_path}")
            link.geometry.append(Mesh(mesh_path, pose=qtest_transforms[i]))

        # Set initial config
        self.q = qtest


    def __init__(self, meshdir=None):

        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))
        
        yaml_path = os.path.join(meshdir, f"joint_limits.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        lowers = []
        uppers = []
        for joint_name, joint_data in data.items():
            lowers.append(joint_data['limit']['lower'])
            uppers.append(joint_data['limit']['upper'])

        qlim = np.array([lowers, uppers])

        # 3. Set qlim
        self.qlim = qlim

        # (your code that builds links/meshes continues here...)

        print("\n--- Loaded Joint Limits ---")
        for j, (lo, up) in enumerate(zip(lowers, uppers), 1):
            print(f"joint{j}: lower={lo}, upper={up}")



        # Franka Emika Panda DH parameters
        links = [
            RevoluteDH(d=0.333, a=0, alpha=-pi/2, qlim=[-2.8973, 2.8973]),  # Link 1
            RevoluteDH(d=0, a=0, alpha=pi/2, qlim=[-1.7628, 1.7628]),  # Link 2
            RevoluteDH(d=0.316, a=0.0825, alpha=pi/2, qlim=[-2.8973, 2.8973]),  # Link 3
            RevoluteDH(d=0, a=-0.0825, alpha=-pi/2, qlim=[-3.0718, -0.0698]),  # Link 4
            RevoluteDH(d=0.384, a=0, alpha=pi/2, qlim=[-2.8973, 2.8973]),  # Link 5
            RevoluteDH(d=0, a=0.088, alpha=pi/2, qlim=[-0.0175, 3.7525]),  # Link 6
            RevoluteDH(d=0.107, a=0, alpha=0, qlim=[-2.8973, 2.8973])   # Link 7
        ]
        
        # Apply joint sign corrections
        sign_corrections = [1, -1, 1, -1, 1, -1, 1]
        for i, link in enumerate(links):
            link.theta = link.theta * sign_corrections[i]
        
        # Define mesh directory
        if meshdir is None:
            meshdir = os.path.dirname(os.path.abspath(__file__))
        
        super().__init__(links, name="Franka", manufacturer="Franka Emika")
        
        # A joint config and corresponding transforms
        qtest = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]
        qtest_transforms = [
            SE3(),  # Base
            SE3(0, 0, 0.333) * SE3.Rz(pi),  # Link 1    
            SE3(0, 0, 0.333) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0),  # Link 2
            SE3(0.0825, 0, 0.649) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0),  # Link 3    
            SE3(0.0825, 0, 0.333) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0),  # Link 4
            SE3(0.0825, 0, 0.717) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0),  # Link 5
            SE3(0.0825, 0.088, 0.717) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0),  # Link 6
            SE3(0.0825, 0.088, 0.824) * SE3.Rz(pi/2) * SE3.Tx(0.0825) * SE3.Rz(-pi/2) * SE3.Ty(0)   # Link 7
        ]


        # Load link meshes
        for i, link in enumerate(self.links, start=0):
            mesh_path = os.path.join(meshdir, f"meshes/visual/link{i}.dae")
            print("Looking for:", mesh_path)

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
        self.robot = Franka(meshdir=mesh_dir)
        
        self.env.add(self.robot)
        
        # Define a simple joint trajectory with sign corrections
        sign_corrections = [-1, 1, 1, -1, 1, -1]
        q_start = np.zeros(7) * sign_corrections
        q_end = np.array([pi/4, -pi/6, pi/6, 0, pi/4, 0, -pi/6])
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
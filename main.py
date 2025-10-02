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
from spatialgeometry import Cuboid
import spatialgeometry as geometry

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

class Environment:
    def __init__(self):
        # Swift environment
        self.env = swift.Swift()  # Initialize Swift simulation environment
        self.env.launch(realTime=True)  # Launch environment with real-time rendering
        self.env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, -pi/4])  # Set camera view

        # Adding fences, ground and safety box
        self.ground_height = 0.005  # Height of the ground plane
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, 1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Front fence
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, -1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Back fence
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Right fence
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(-1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Left fence
        self.env.add(Cuboid(scale=[3, 3, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))  # Ground plane
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1, -1.1, 0.3), color=[0.5, 0.5, 0.9, 0.5]))  # Additional object

        self.safety = self.load_safety()  # Load safety objects

        # Mesh directory: point this to franka_ros meshes
        mesh_dir = os.path.join(os.getcwd(), "franka", "meshes", "visual")

        # Create robot
        self.franka = FrankaPanda(meshdir=mesh_dir)

        # Add robot
        self.env.add(self.franka)
        
    def load_safety(self):
        safety_dir = os.path.abspath("safety_models")  # Path to safety models
        stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]  # Safety object files
        safety_positions = [
            SE3(-1.3, -1.35, 0.0 + self.ground_height) * SE3.Rx(pi/2), SE3(-1, -1.4, 0.0), SE3(-1.15, -1.48, 0.5) * SE3.Rx(pi/2) * SE3.Ry(pi)
        ]  # Positions with rotations
        safety_colour = [(0.6, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]  # Colors
        safety = []
        for stl_file, pose, colour in zip(stl_files, safety_positions, safety_colour):
            stl_path = os.path.join(safety_dir, stl_file)  # Construct file path
            if not os.path.exists(stl_path):
                raise FileNotFoundError(f"STL file not found: {stl_path}")  # Error if file missing
            safety_obj = geometry.Mesh(stl_path, pose=pose * SE3(0, 0, self.ground_height), scale=(0.001, 0.001, 0.001), color=colour)  # Load mesh
            self.env.add(safety_obj)  # Add to environment
            safety.append(safety_obj)  # Store object
        return safety



class Control:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env

    def move_to(self, q_end, duration=3, steps=100):
        q_start = self.robot.q
        t = np.linspace(0, duration, steps)
        traj = jtraj(q_start, q_end, t)

        for q in traj.q:
            self.robot.q = q
            self.env.step(0.03)
       

if __name__ == "__main__":
    assignment = Environment()
    controller = Control(assignment.franka, assignment.env)
    controller.move_to([0, 0, -pi/3, 0, -pi/2, 0, pi/4, 0])
    assignment.env.hold()

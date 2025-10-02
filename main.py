import spatialgeometry as geometry
import numpy as np
import swift
import os
import time
from roboticstoolbox import jtraj, RevoluteMDH, DHRobot
from scipy.spatial import ConvexHull
from ir_support import UR3
from spatialmath import SE3
from spatialgeometry import Cuboid, Mesh
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

        super().__init__(links, name="Franka_Panda")

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

        # Add UR3 robot
        self.ur3 = UR3()  # Initialize UR3 robot
        self.ur3.q = np.array([pi/2, -pi/2, 0, -pi/2, 0, -pi/2])  # Initial joint angles
        self.ur3.base = SE3(0, 0.75, 0)  # Set base position at origin
        self.ur3.add_to_env(self.env)  # Add robot to environment

        # Add Franka Panda robot
        mesh_dir = os.path.join(os.getcwd(), "franka", "meshes", "visual")
        self.franka = FrankaPanda(meshdir=mesh_dir)
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

    def move_to(self, target_pose, steps):
        success, traj = self.check_and_calculate_joint_angles(target_pose, steps)
        if not success:
            print("Target pose is not reachable")
            return False

        for q in traj:
            self.robot.q = q
            self.env.step(0.02)  # Update rendering
            time.sleep(0.03)
        return True


    def check_and_calculate_joint_angles(self, target_pose, steps=50):
        original_q = self.robot.q.copy()

        # Solve IK
        ik_result = self.robot.ikine_LM(target_pose, q0=self.robot.q, joint_limits=False)
        if not ik_result.success:
            return False, []

        q_goal = ik_result.q
        print("IK solution found:", q_goal)

        # Generate joint trajectory
        traj = jtraj(original_q, q_goal, steps).q

        # Reset robot to original q
        self.robot.q = original_q

        return True, traj


class Mission:
    def __init__(self, env, controller_ur3, controller_franka):
        # Define sequences of poses for each robot
        self.ur3_array = [
            SE3(0.2, 0.75, 0.2),
            SE3(1, 0.75, 1),
            SE3(-0.5, -0.75, 0.5),
            SE3(1, 0.75, 1)
        ]
        self.franka_array = [
            SE3(0.2, 0.2, 0.2),
            SE3(1, 0, 1),
            SE3(-0.5, 0, 0.5),
            SE3(1, 0, 1)
        ]

        self.env = env
        self.controller_ur3 = controller_ur3
        self.controller_franka = controller_franka

    def run(self):
        # Example: move both robots to their first poses
        print("Moving UR3 to first mission pose...")
        success = self.controller_ur3.move_to(self.ur3_array[0], 50)
        if not success:
            pass

        print("Moving Franka Panda to first mission pose...")
        success = self.controller_franka.move_to(self.franka_array[0], 50)
        success = self.controller_franka.move_to(self.franka_array[1], 50)
        success = self.controller_franka.move_to(self.franka_array[2], 50)
        success = self.controller_franka.move_to(self.franka_array[3], 50)
        success = self.controller_franka.move_to(self.franka_array[4], 50)
        success = self.controller_franka.move_to(self.franka_array[5], 50)
        success = self.controller_franka.move_to(self.franka_array[6], 50)

        
        
   
       
if __name__ == "__main__":
    # Setup environment
    assignment = Environment()

    # Create controllers for each robot
    controller_ur3 = Control(assignment.ur3, assignment.env)
    controller_franka = Control(assignment.franka, assignment.env)

    # Define and run mission
    mission = Mission(assignment.env, controller_ur3, controller_franka)
    mission.run()

    assignment.env.hold()

import spatialgeometry as geometry
import numpy as np
import swift
import time
import os
from roboticstoolbox import jtraj, DHLink, DHRobot
from scipy.spatial import ConvexHull
from ir_support import UR3
from spatialmath import SE3, SO3
from spatialgeometry import Cuboid
from math import pi


# ---------------- Gripper Class ----------------
class Gripper:
    def __init__(self, env, robot):
        self.env = env  # Store the Swift environment for rendering
        self.robot = robot  # Store UR3 reference for kinematic calculations
        self.finger_len = 0.05  # Length of each finger
        self.finger_w = 0.01  # Width of each finger
        self.finger_t = 0.04  # Thickness of each finger
        self.max_opening = 0.09  # Maximum opening distance between fingers
        self.closed = 0.07  # Closed distance between fingers
        self.carrying_idx = None  # Index of carried brick, if any

        # Define two prismatic links for the fingers
        finger_L = DHLink(a=0, alpha=0, d=0, theta=0,
                          sigma=1, qlim=[0, self.max_opening/2])  # Left finger prismatic joint
        finger_R = DHLink(a=0, alpha=pi, d=0, theta=0,
                          sigma=1, qlim=[0, self.max_opening/2])  # Right finger prismatic joint

        # Internal DHRobot model for the fingers
        self.model = DHRobot([finger_L, finger_R], name="gripper")  # Create gripper DH model

        # Graphics
        self.finger_L = Cuboid([self.finger_len, self.finger_w, self.finger_t], color=[0.2,0.2,0.2,1])  # Left finger visual
        self.finger_R = Cuboid([self.finger_len, self.finger_w, self.finger_t], color=[0.2,0.2,0.2,1])  # Right finger visual
        self.connector = Cuboid([self.finger_len, self.max_opening - 0.01, self.finger_w], color=[0.2,0.2,0.2,1])  # Connector visual

        env.add(self.finger_L)  # Add left finger to environment
        env.add(self.finger_R)  # Add right finger to environment
        env.add(self.connector)  # Add connector to environment

        # Start open
        self.q = np.array([self.max_opening/2, -self.max_opening/2])  # Initial joint positions for fingers

    def open(self):
        self.q = np.array([self.max_opening/2, -self.max_opening/2])  # Set fingers to maximum opening
        self.update()  # Update gripper position

    def close(self):
        self.q = np.array([self.closed/2, -self.closed/2])  # Set fingers to closed position
        self.update()  # Update gripper position

    def update(self):
        # Get UR3 tool pose with 90-degree rotation around Z
        T_base = self.robot.fkine(self.robot.q) * SE3.Rz(pi/2) 

        # Forward kinematics for each finger relative to tool frame
        T_L = T_base * self.model.fkine([0, self.finger_t/2]) * SE3(0, self.q[0], 0)  # Left finger pose
        T_R = T_base * self.model.fkine([0, self.finger_t/2]) * SE3(0, self.q[1], 0)  # Right finger pose

        # Update meshes
        self.finger_L.T = T_L  # Set left finger transformation
        self.finger_R.T = T_R  # Set right finger transformation
        self.connector.T = T_base  # Set connector transformation

    def update_with_payload(self, bricks):
        self.update()  # Update gripper position
        if self.carrying_idx is not None:
            T_ee = self.robot.fkine(self.robot.q)  # End-effector pose
            T_offset = SE3.Rx(pi)  # 180-degree rotation around X
            # Adjusted brick pose to center it between gripper fingers
            brick_pose = T_ee * T_offset * SE3(0, 0, -self.finger_w - self.finger_t) 
            bricks[self.carrying_idx].T = brick_pose  # Update brick position


# ---------------- Environment Builder Class ----------------
class EnvironmentBuilder:
    def __init__(self):
        self.y_max = 0.8  # Maximum Y position for the rail carriage
        self.env = swift.Swift()  # Initialize Swift simulation environment
        self.env.launch(realTime=True)  # Launch environment with real-time rendering
        self.env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, -pi/4])  # Set camera view

        # Add environment objects
        self.ground_height = 0.005  # Height of the ground plane
        self.add_fences_and_ground()  # Add fences and ground
        self.add_rail()  # Add rail system
        self.safety = self.load_safety()  # Load safety objects

        # Add robot
        self.robot = UR3()  # Initialize UR3 robot
        self.robot.q = np.array([pi/2, -pi/2, 0, -pi/2, 0, -pi/2])  # Initial joint angles
        # Print initial end-effector pose
        print(f"Starting end effector pose:\n{self.robot.fkine(self.robot.q)}")
        self.robot.base = SE3(0, 0, 0)  # Set base position at origin
        self.robot.links[1].qlim = np.deg2rad([-160, -20])  # Set joint 1 limits
        self.robot.links[2].qlim = np.deg2rad([-180, 180])  # Set joint 2 limits
        # self.robot.links[3].qlim = np.deg2rad([0, 90])  # Commented out joint 3 limit
        for i in range(self.robot.n):
            print(f"Joint {i} limits: lower = {np.rad2deg(self.robot.qlim[0, i]):.2f}°, upper = {np.rad2deg(self.robot.qlim[1, i]):.2f}°")  # Print joint limits
        self.robot.add_to_env(self.env)  # Add robot to environment

        # Add gripper
        self.gripper = Gripper(self.env, self.robot)  # Initialize gripper
        # Update gripper to set initial position
        self.gripper.update()

        # Add bricks
        self.bricks = self.load_bricks()  # Load brick objects

        # Refresh environment to show initial state
        self.env.step(0)

    def add_fences_and_ground(self):
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, 1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Front fence
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, -1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Back fence
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Right fence
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(-1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))  # Left fence
        self.env.add(Cuboid(scale=[3, 3, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))  # Ground plane
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1, -1.1, 0.3), color=[0.5, 0.5, 0.9, 0.5]))  # Additional object

    def add_rail(self):
        self.env.add(Cuboid(scale=[0.05, 2 * self.y_max, 0.05], pose=SE3(0.1, 0, 0.025 + self.ground_height), color=[0.3, 0.3, 0.35, 1]))  # Left rail
        self.env.add(Cuboid(scale=[0.05, 2 * self.y_max, 0.05], pose=SE3(-0.1, 0, 0.025 + self.ground_height), color=[0.3, 0.3, 0.35, 1]))  # Right rail
        self.rail_carriage = Cuboid(scale=[0.15, 0.15, 0.05], pose=SE3(0.0, 0.0, 0.025), color=[1, 0.4, 0.7, 1])  # Rail carriage
        self.env.add(self.rail_carriage)  # Add carriage to environment

    def load_safety(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
        stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]  # Safety object files
        safety_positions = [
            SE3(-1.3, -1.35, 0.0 + self.ground_height) * SE3.Rx(pi/2), SE3(-1, -1.4, 0.0), SE3(-1.15, -1.48, 0.5) * SE3.Rx(pi/2) * SE3.Ry(pi)
        ]  # Positions with rotations
        safety_colour = [(0.6, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]  # Colors
        safety = []
        for stl_file, pose, colour in zip(stl_files, safety_positions, safety_colour):
            stl_path = os.path.join(current_dir, stl_file)  # Construct file path
            if not os.path.exists(stl_path):
                raise FileNotFoundError(f"STL file not found: {stl_path}")  # Error if file missing
            safety_obj = geometry.Mesh(stl_path, pose=pose * SE3(0, 0, self.ground_height), scale=(0.001, 0.001, 0.001), color=colour)  # Load mesh
            self.env.add(safety_obj)  # Add to environment
            safety.append(safety_obj)  # Store object
        return safety

    def load_bricks(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
        stl_path = os.path.join(current_dir, "Brick.stl")  # Brick STL file path
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")  # Error if file missing
        brick_positions = [
            SE3(-0.72, -1.2, 0.0), SE3(-0.22, 1.2, -0.2), SE3(-0.22, 0.0, 0.0),
            SE3(-0.2, 0.2, 0.0), SE3(-0.2, -0.2, 0.0), SE3(-0.3, 0.0, 0.0),
            SE3(-0.3, 0.2, 0.0), SE3(-0.3, -0.2, 0.0), SE3(-0.4, 0.0, 0.0),
            SE3(-0.4, 0.2, 0.0), SE3(-0.4, -0.2, 0.0)
        ]  # Brick positions
        bricks = []
        for pose in brick_positions:
            brick = geometry.Mesh(stl_path, pose=pose * SE3(0, 0, self.ground_height), color=(0.4, 0, 0, 1))  # Load brick mesh
            self.env.add(brick)  # Add to environment
            bricks.append(brick)  # Store brick
        return bricks


# ---------------- Controller Class ----------------
class Controller:
    def __init__(self, env_builder: EnvironmentBuilder):
        self.env_builder = env_builder  # Store environment builder reference
        self.env = env_builder.env  # Store environment
        self.robot = env_builder.robot  # Store robot
        self.bricks = env_builder.bricks  # Store bricks
        self.rail_carriage = env_builder.rail_carriage  # Store rail carriage
        self.gripper = env_builder.gripper  # Store gripper
        self.safety = env_builder.safety  # Store safety objects

        self.failed_bricks = 0  # Count of unreachable bricks
        # Wall target poses for placing bricks
        self.wall_pose = [
            SE3(0.3, 0.133, 0.0), SE3(0.3, 0.0, 0.0), SE3(0.3, -0.133, 0.0),
            SE3(0.3, 0.133, 0.033), SE3(0.3, 0.0, 0.033), SE3(0.3, -0.133, 0.033),
            SE3(0.3, 0.133, 0.066), SE3(0.3, 0.0, 0.066), SE3(0.3, -0.133, 0.066)
        ]

    def move_carriage_to_y(self, target_y, steps=25):
        start_y = self.robot.base.t[1]  # Current Y position of base
        for s in np.linspace(0, 1, steps):  # Linear interpolation over steps
            y = (1 - s) * start_y + s * target_y  # Interpolate Y position
            y = np.clip(y, -self.env_builder.y_max, self.env_builder.y_max)  # Clip to rail limits
            self.gripper.update_with_payload(self.bricks)  # Update gripper with payload
            self.robot.base = SE3(0, y, 0)  # Move robot base
            self.rail_carriage.T = SE3(0, y, 0.025)  # Move rail carriage
            self.env.step(0.02)  # Update environment
            time.sleep(0.03)  # Pause for real-time effect


# ---------------- Main ----------------
if __name__ == "__main__":
    env_builder = EnvironmentBuilder()  # Create environment builder
    # controller = Controller(env_builder)  # Create controller
    # controller.compute_reach_and_volume(env_builder)  # Compute reach and volume
    input("Press enter to start...\n")  # Wait for user input
    # controller.pick_and_place()  # Execute pick and place operation
    env_builder.env.hold()  # Hold environment open
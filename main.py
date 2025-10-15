import spatialgeometry as geometry
import numpy as np
import swift
import os
import time
from roboticstoolbox import jtraj
from ir_support import UR3
from spatialmath import SE3
from spatialgeometry import Cuboid
from math import pi
import random

# For KUKA KR6
from KUKA_KR6.KR6 import KR6_Robot as KR6
# For KUKA LBR
from KUKA_LBR.lbr_loader import Load as LBR
# For KUKA LWR
from Kuka_LWR.kuak_lwr import Load as LWR
# For gripper
from gripper import Gripper


# -------------------------------------------------------------
# RobotUnit (handles robot + gripper + control)
# -------------------------------------------------------------
class RobotUnit:
    def __init__(self, robot, env, base_pose, q_init=None):
        self.robot = robot
        self.env = env  # This is the swift.Swift object
        self.environment = env._parent if hasattr(env, '_parent') else env  # Store the Environment instance
        self.robot.base = base_pose
        self.robot.q = np.zeros(self.robot.n) if q_init is None else q_init
        self.robot.add_to_env(env)
        self.gripper = Gripper(self.robot.fkine(self.robot.q))
        self.gripper.add_to_env(env)

    def move_to(self, target_pose, steps=50, brick_idx=None):

        hover_pose = target_pose * SE3(0, 0, 0.14)
        possible, traj = self.check_and_calculate_joint_angles(hover_pose, steps)
        if not possible:
            print(f"[{self.robot.name}] Hover pose not reachable:\n{target_pose}")
            return
        possible, traj_goal = self.check_and_calculate_joint_angles(target_pose, steps, traj[-1])
        if not possible:
            print(f"[{self.robot.name}] Target pose not reachable:\n{target_pose}")
            return

        for traj_set in [traj, traj_goal]:
            for q in traj_set:
                self.robot.q = q
                self.env.step(0.02)
                if brick_idx is not None:
                    self.environment.bricks[brick_idx][1].T = self.robot.fkine(q) * SE3(0, 0, 0.14) * SE3.Rz(pi/2)  # Use environment.bricks
                self.gripper.update(self.robot.fkine(q))
                time.sleep(0.03)

    def check_and_calculate_joint_angles(self, target_pose, steps=50, q0=None):
        if q0 is None:
            q0 = self.robot.q.copy()
        target_pose_corrected = target_pose * SE3.RPY(0, pi, pi/2) * SE3(0, 0, -0.165)
        ik = self.robot.ikine_LM(target_pose_corrected, q0=np.zeros(self.robot.n), joint_limits=True)
        if not ik.success:
            return False, []
        traj = jtraj(q0, ik.q, steps).q
        return True, traj

    def pick_and_place(self, pick_pose, place_pose, steps=50, brick_idx=None):
        print(f"[{self.robot.name}] Starting pick and place task")
        self.gripper.actuate("open")
        self.move_to(pick_pose, steps)  # No brick_idx for picking (hover)
        self.gripper.actuate("close")
        self.move_to(place_pose, steps, brick_idx)  # Pass brick_idx for moving the brick
        print(f"[{self.robot.name}] Completed pick and place")

# -------------------------------------------------------------
# Environment (holds Swift, robots, safety, mission)
# -------------------------------------------------------------
class Environment:
    def __init__(self):
        # Launch Swift
        self.env = swift.Swift()
        self.env._parent = self  # Set the parent Environment instance
        self.env.launch(realTime=True)
        self.env.set_camera_pose([2, 2, 2], [0, 0, -pi/4])

        self.ground_height = 0.005
        self.ground_length = 3.5
        self.add_world()
        self.safety = self.load_safety()

        # Conveyor
        self.conveyer_height = 0.3
        self.env.add(Cuboid(scale=[0.3, 2.5, self.conveyer_height],
                            pose=SE3(0, 0, self.conveyer_height/2 + self.ground_height),
                            color=[0, 0, 0]))

        # Robots
        self.load_robots()

        # Object positions
        self.object_origin = [SE3(1.3, 1, self.ground_height),
                              SE3(1.3, 0, self.ground_height)]
        self.place_positions = [SE3(0, 1, 0.35),
                                SE3(0, 0, 0.4)*SE3.RPY(pi/2, 0, pi/2)]

        # Object tracking
        self.bricks = []

    def add_world(self):
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0,  self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0, -self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3( self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3(-self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, self.ground_length, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1.25, -1.35, 0.3), color=[0.5, 0.5, 0.9, 0.5]))

    def load_robots(self):
        self.kr6 = RobotUnit(KR6(), self.env, SE3(0.7, 1, self.ground_height))
        self.lbr = RobotUnit(LBR(), self.env, SE3(0.7, 0, self.ground_height))
        self.lwr = RobotUnit(LWR(), self.env, SE3(0.7, -1, self.ground_height))
        self.ur3 = RobotUnit(UR3(), self.env, SE3(-0.45, 0, self.ground_height),
                             q_init=np.array([pi/2, -pi/2, 0, -pi/2, 0, -pi/2]))

    def load_safety(self):
        safety_dir = os.path.abspath("Safety")
        stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]
        poses = [
            SE3(-1.55, -1.6, 0.0 + self.ground_height) * SE3.Rx(pi/2),
            SE3(-1.350, -1.65, 0.0 + self.ground_height),
            SE3(-1.4, -1.73, 0.5 + self.ground_height) * SE3.Rx(pi/2) * SE3.Ry(pi)
        ]
        colours = [(0.6, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
        safety_objs = []
        for stl, pose, colour in zip(stl_files, poses, colours):
            path = os.path.join(safety_dir, stl)
            obj = geometry.Mesh(path, pose=pose, scale=(0.001, 0.001, 0.001), color=colour)
            self.env.add(obj)
            safety_objs.append(obj)
        return safety_objs

    def load_object(self, index):
        obj_path = os.path.join(os.path.abspath("Objects"), "Brick.stl")
        pose = self.object_origin[index] * SE3(0, 0, self.ground_height)
        obj = geometry.Mesh(obj_path, pose=pose)
        self.env.add(obj)
        self.bricks.append((index, obj))
        print(f"Loaded brick at origin {index}")
        return obj

    # Run the mission directly
    def run_mission(self):
        print("Starting mission simulation...")
        self.load_object(0)
        self.load_object(1)
        self.kr6.pick_and_place(self.object_origin[0], self.place_positions[0], steps=50, brick_idx=0)
        self.kr6.gripper.actuate("open")
        self.lbr.pick_and_place(self.object_origin[1], self.place_positions[1], steps=50, brick_idx=1)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    environment = Environment()
    environment.run_mission()
    environment.env.hold()
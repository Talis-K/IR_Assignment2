import spatialgeometry as geometry
import numpy as np
import swift
import os
import time
from roboticstoolbox import jtraj, RevoluteMDH, DHRobot
from ir_support import UR3
from spatialmath import SE3
from spatialgeometry import Cuboid, Mesh
from math import pi


#For KinovaGen3
from KinovaGen3.KinovaGen3 import KinovaGen3 as KG3\
#For KUKA LBR
from KUKA_Talis.loader import Load as KUKA_LBR
#for interbotixs vx300s
from interbotix.loader import Load as vx300s

class Environment:
    def __init__(self):
       
        # Swift Set Up
        self.env = swift.Swift()
        self.env.launch(realTime=True)
        self.env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, -pi/4])
        print("Swift environment launched")


        # Adding fences, ground and safety box
        self.ground_height = 0.005
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, 1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[3, 0.05, 0.8], pose=SE3(0, -1.5, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, 3, 0.8], pose=SE3(-1.5, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[3, 3, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1, -1.1, 0.3), color=[0.5, 0.5, 0.9, 0.5]))
        self.safety = self.load_safety()


        # Add Kinova Gen3
        KG3robot = KG3()
        KG3robot.q = KG3robot.qz  # zero position
        q_rand = KG3robot.qr # Animate to some random config
        KG3robot.q = q_rand
        KG3robot.base = SE3(1, 0.5, 0.0)   # set new base pose (x, y, z)
        self.env.add(KG3robot)


        # Add KUKA LBR
        self.lbr = KUKA_LBR()
        self.lbr.base = SE3(0, 0, 0.0)   # set new base pose (x, y, z)
        self.lbr.q = self.lbr.qz  # zero position
        q_rand = self.lbr.qr # Animate to some random config
        self.lbr.q = q_rand
        self.env.add(self.lbr)

        # Add Interbotix vx300s
        self.vx300s = vx300s()
        self.vx300s.base = SE3(0, 0, 0.0)   # set new base pose (x, y, z)
        self.vx300s.q = self.vx300s.qz  # zero position
        q_rand = self.vx300s.qr # Animate to some random config
        self.vx300s.q = q_rand
        self.env.add(self.vx300s)


        # Add UR3 robot
        self.ur3 = UR3()
        self.ur3.q = np.array([pi/2, -pi/2, 0, -pi/2, 0, -pi/2])
        self.ur3.base = SE3(0, 0.75, 0)
        self.ur3.add_to_env(self.env)


        # Store bricks with their pose indices and counters
        self.bricks = []
        self.brick_counters = {0: 0, 1: 0, 2: 0}

        #load initial bricks
        for i in range(3):
            self.load_object(i)


    def load_safety(self): #Sources and Loads in Safety Objects
        safety_dir = os.path.abspath("Safety")
        stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]
        safety_positions = [
            SE3(-1.30, -1.35, 0.0 + self.ground_height) * SE3.Rx(pi/2),
            SE3(-1.00, -1.40, 0.0 + self.ground_height),
            SE3(-1.15, -1.48, 0.5 + self.ground_height) * SE3.Rx(pi/2) * SE3.Ry(pi)
        ]
        safety_colour = [
            (0.6, 0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 1.0)]
        safety = []


        for stl_file,  pose,             colour         in zip(
            stl_files, safety_positions, safety_colour):
           
            stl_path   = os.path.join(safety_dir, stl_file)          
            safety_obj = geometry.Mesh(stl_path, pose=pose, scale=(0.001, 0.001, 0.001), color=colour)
            self.env.add(safety_obj)
            safety.append(safety_obj)
        return safety
   
    def load_object(self, pose_index): #Sources and Loads in Brick Objects for Conveyor and Packing
        obj_path = os.path.join(os.path.abspath("Objects"), "Brick.stl")
        object_positions = [
            SE3(-0.5, 0, 0.5),
            SE3(0.7, 0.75, 0.1),
            SE3(0.6, 0.0, 0.0),
        ]
        counter = self.brick_counters[pose_index]
        self.brick_counters[pose_index] += 1
        pose = object_positions[pose_index] * SE3(0, 0, self.ground_height + counter * 0.01)
        obj = geometry.Mesh(obj_path, pose=pose)
        self.env.add(obj)
        self.bricks.append((pose_index, counter, obj))
        print(f"Loaded brick at pose_index {pose_index}, counter {counter}, initial pose: {obj.T[:3, 3]}")
        return obj


    def object_conveyor(self, pose_index, counter):
        brick = None
        for idx, cnt, obj in self.bricks:
            if idx == pose_index and cnt == counter:
                brick = obj
                break
        if brick is None:
            print(f"No brick found for pose_index {pose_index} and counter {counter}")
            return


        target = [
            SE3(0.1, 0.25, 0.0), SE3(0.2, -0.5, 1.0), SE3(0.9, -1.25, 0.0)
        ]
       
        initial_pose = brick.T[:3, 3]
        target_pose = target[pose_index] * SE3(0, 0, self.ground_height + counter * 0.01)
        target_position = target_pose.t
        rotation_matrix = brick.T[:3, :3]  # Keep initial rotation


        print(f"Moving brick at pose_index {pose_index}, counter {counter}")
        print(f"  From initial pose: {initial_pose}")
        print(f"  To target pose: {target_position}")


        # Linear interpolation for smooth movement
        steps = 25
        for s in np.linspace(0, 1, steps):
            # Interpolate position
            interpolated_position = (1 - s) * initial_pose + s * target_position
            # Construct new transformation matrix
            new_pose = np.eye(4)
            new_pose[:3, :3] = rotation_matrix
            new_pose[:3, 3] = interpolated_position
            brick.T = new_pose
            self.env.step(0.02)
            self.env.step(0.02)
            time.sleep(0.03)  # Match timing from move_carriage_to_y


        print(f"  Swift environment updated")


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
            self.env.step(0.02)
            time.sleep(0.03)
        return True


    def check_and_calculate_joint_angles(self, target_pose, steps=50):
        original_q = self.robot.q.copy()
        ik_result = self.robot.ikine_LM(target_pose, q0=self.robot.q, joint_limits=False)
        if not ik_result.success:
            return False, []
        q_goal = ik_result.q
        print("IK solution found:", q_goal)
        traj = jtraj(original_q, q_goal, steps).q
        self.robot.q = original_q
        return True, traj


class Mission:
    def __init__(self, env, controller_ur3, controller_lbr, controller_vx300s):
        self.ur3_array = [
            SE3(0.4, 0.75, 0.4),  # Above first brick
            SE3(1, 0.75, 1),
            SE3(-0.5, -0.75, 0.5),
            SE3(1, 0.75, 1)
        ]
        self.lbr_array = [
            SE3(0.2, 0.2, 0.2) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(1, 0, 1) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(-0.5, 0, 0.5) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(1, 0, 1) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(0.4, 0.75, 0.4) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(1, 0.75, 1) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(-0.5, -0.75, 0.5) * SE3(0,0,0.041) * SE3.Rx(pi),
            SE3(1, 0.75, 1) * SE3(0,0,0.041) * SE3.Rx(pi)
        ]
        self.vx_array = [
            SE3(0.4, 0, 0.4),  # Above first brick
            SE3(1, 0, 1),
            SE3(-0.5, 0, 0.5),
            SE3(1, 0, 1)
        ]
        self.env = env
        self.controller_ur3 = controller_ur3
        self.controller_lbr = controller_lbr
        self.controller_vx300s = controller_vx300s


    def run(self):

        # # Move vx300s to each mission pose in sequence
        # for i in range(len(self.vx_array)):
        #     print(f"Moving vx300s to mission pose {i+1}...")
        #     success = self.controller_vx300s.move_to(self.vx_array[i], 50)
        #     input("Press Enter to continue...")
        #     if not success:
        #         print(f"vx300s failed to reach pose {i+1}")
        #         continue

        # Move LBR to each mission pose in sequence
        for i in range(len(self.lbr_array)):
            print(f"Moving LBR to mission pose {i+1}...")
            success = self.controller_lbr.move_to(self.lbr_array[i], 50)
            input("Press Enter to continue...")
            if not success:
                print(f"LBR failed to reach pose {i+1}")
                continue

        # Move UR3 to each mission pose in sequence
        for i in range(len(self.ur3_array)):
            print(f"Moving UR3 to mission pose {i+1}...")
            success = self.controller_ur3.move_to(self.ur3_array[i], 50)
            input("Press Enter to continue...")
            if not success:
                print(f"UR3 failed to reach pose {i+1}")
                continue


if __name__ == "__main__":
    assignment = Environment()
    controller_ur3 = Control(assignment.ur3, assignment.env)
    controller_lbr = Control(assignment.lbr, assignment.env)
    controller_vx300s = Control(assignment.vx300s, assignment.env)
    mission = Mission(assignment, controller_ur3, controller_lbr, controller_vx300s)
    mission.run()
    assignment.env.hold()


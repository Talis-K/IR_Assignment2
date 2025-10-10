#!/usr/bin/env python3
"""
test_lbr_y_rotation_controlled.py

Visual test: moves the KUKA LBR robot in Swift using your Control.move_to() function.
Each step rotates the end-effector about its local Y axis (pitch) by 5° increments.
"""

import time
import numpy as np
import swift
from math import radians, pi
from spatialmath import SE3
from roboticstoolbox import jtraj
import os
import spatialgeometry as geometry

# Import your LBR model
from KUKA_Talis.loader import LBR as KUKA_LBR


class Control:
    """Your existing control class with IK-based movement"""

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
        ik_result = self.robot.ikine_LM(target_pose, q0=self.robot.q, joint_limits=False, end='tool0')
        if not ik_result.success:
            print("IK failed for pose:\n", target_pose)
            return False, []
        q_goal = ik_result.q
        print("IK solution found:", np.round(q_goal, 3))
        traj = jtraj(original_q, q_goal, steps).q
        self.robot.q = original_q
        return True, traj


def rotate_about_y(control, base_pose, start_deg=-30, stop_deg=30, step_deg=5):
    """
    Moves the robot to poses that are the same position but rotated around local Y axis.
    Uses IK + trajectory for smooth movement each step.
    """
    print("\nStarting controlled Y-axis rotation test...\n")

    for deg in range(start_deg, stop_deg + step_deg, step_deg):
        # Build the rotated pose
        rot = SE3.Rx(radians(deg))
        target_pose = base_pose * rot
        input("enter to continue...")
        print(f"→ Moving to pose with {deg}° about Y-axis")
        success = control.move_to(target_pose, steps=40)

        if not success:
            print(f"Pose at {deg}° not reachable, skipping.\n")
            continue

        time.sleep(0.5)  # small pause between rotations

    print("\n✅ Rotation sequence finished.")


def main():
    # Launch Swift
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, -pi / 4])

    # Instantiate LBR
    lbr = KUKA_LBR()
    lbr.base = SE3(0, 0, 0)
    lbr.q = lbr.qz  # Zero configuration
    env.add(lbr)
    obj_path = os.path.join(os.path.abspath("Objects"), "Brick.stl")
    obj = geometry.Mesh(obj_path, pose=SE3(-0.5, 0, 0.5))
    env.add(obj)

    # Make controller
    control = Control(lbr, env)

    # Define the target base pose (tool tip)
    # e.g., brick at z=0.5, tool length = 0.3 → end-effector z = 0.2
    base_pose = SE3(-0.5, 0, 0.5+0.036)*SE3.Rx(pi)
    control.move_to(base_pose, steps=10)
    env.step(0.1)
    input("Press Enter to start Y-rotation test...")
    # Run the controlled Y-rotation sequence
    rotate_about_y(control, base_pose, start_deg=-30, stop_deg=30, step_deg=5)

    print("Holding environment open...")
    env.hold()


if __name__ == "__main__":
    main()

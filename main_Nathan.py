#!/usr/bin/env python3
import os
import time
from math import pi

import numpy as np
import swift
import spatialgeometry as geometry
from spatialgeometry import Cuboid
from spatialmath import SE3
from roboticstoolbox import jtraj

# Robots & extras
from ir_support import UR3
from KUKA_KR6.KR6 import KR6_Robot as KR6
from KUKA_LBR.lbr_loader import Load as LBR
from Kuka_LWR.kuak_lwr import Load as LWR
from gripper import Gripper

# Collision detection
from collisiontester import CollisionDetector


# -------------------------------------------------------------
# Environment (holds Swift, robots, safety, mission)
# -------------------------------------------------------------
class Environment:
    def __init__(self):
        self.built = 0
        self.env = swift.Swift()
        self.env._parent = self               # Allow RobotUnit to access Environment
        self.env.launch(realTime=True)
        self.env.set_camera_pose([2, 2, 2], [0, 0, -pi / 4])

        self.ground_height = 0.005
        self.ground_length = 3.5

        self.add_world()
        self.safety = self.load_safety()

        # Conveyor
        self.conveyer_height = 0.3  # (kept original name to avoid breaking anything)
        self.env.add(
            Cuboid(
                scale=[0.3, 2.5, self.conveyer_height],
                pose=SE3(0, 0, self.conveyer_height / 2 + self.ground_height),
                color=[0, 0, 0],
            )
        )

        # Robots (now with multiple zones per robot)
        self.load_robots()

        # Brick/Object Start and End Positions
        self.brick_origin = [
            SE3(1.3, 1.0, self.ground_height),
            SE3(1.3, 0.0, self.ground_height),
        ]
        self.brick_place_pos = [
            SE3(0.0, 1.0, 0.35),
            SE3(0.0, 0.0, 0.40) * SE3.RPY(pi / 2, 0, pi / 2),
        ]

        # Track (index, geometry.Mesh)
        self.bricks = []

    def add_world(self):
        """Floor, borders, and a small platform."""
        # Long sides (Y+ / Y-)
        for y in (self.ground_length / 2, -self.ground_length / 2):
            self.env.add(
                Cuboid(
                    scale=[self.ground_length, 0.05, 0.8],
                    pose=SE3(0, y, 0.4 + self.ground_height),
                    color=[0.5, 0.9, 0.5, 0.5],
                )
            )
        # Short sides (X+ / X-)
        for x in (self.ground_length / 2, -self.ground_length / 2):
            self.env.add(
                Cuboid(
                    scale=[0.05, self.ground_length, 0.8],
                    pose=SE3(x, 0, 0.4 + self.ground_height),
                    color=[0.5, 0.9, 0.5, 0.5],
                )
            )
        # Floor
        self.env.add(
            Cuboid(
                scale=[self.ground_length, self.ground_length, 2 * self.ground_height],
                pose=SE3(0, 0, 0),
                color=[0.9, 0.9, 0.5, 1.0],
            )
        )
        # Small platform
        self.env.add(
            Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1.25, -1.35, 0.3), color=[0.5, 0.5, 0.9, 0.5])
        )
        self.built += 1

    def load_robots(self):
        """Add all robots at their bases, each with its own multiple collision detector zones."""

        # Streamlined Collision Zone Definer
        def Z(center, volume=(0.3,0.3,1.5), colour=(0.0, 1.0, 0.0, 0.45)):
            return dict(volume=volume, center=center, colour=colour)

        # Collision Zones
        kr6_zones = [
            Z((0.7,0,0.75),colour=(0.0, 0.1, 0.0, 0.001)), #For LBR Zone
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        lbr_zones = [
            Z((0.7,1,0.75),colour=(0.0, 0.1, 0.0, 0.001)),  #For KR6 Zone
            Z((0.7,-1,0.75),colour=(0.0, 0.1, 0.0, 0.001)), #For LWR Zone
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        lwr_zones = [
            Z((0.7,0,0.75),colour=(0.0, 0.1, 0.0, 0.001)), #For LBR Zone
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        ur3_zones = [
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        # KR6
        self.kr6 = RobotUnit(KR6(), self.env, SE3(0.7, 1.0, self.ground_height),
            collision_zones=kr6_zones)

        # LBR
        self.lbr = RobotUnit(LBR(), self.env, SE3(0.7, 0.0, self.ground_height),
            collision_zones=lbr_zones)

        # LWR
        self.lwr = RobotUnit(LWR(), self.env, SE3(0.7, -1.0, self.ground_height),
            collision_zones=lwr_zones)

        # UR3
        self.ur3 = RobotUnit(UR3(), self.env, SE3(-0.45, 0.0, self.ground_height),
            collision_zones=ur3_zones)

        self.built += 1

    def load_safety(self):
        """Load safety meshes with poses and colors."""
        
        safety_dir = os.path.abspath("Safety")
        stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]
        poses = [
            SE3(-1.550, -1.60, 0.0 + self.ground_height) * SE3.Rx(pi / 2),
            SE3(-1.350, -1.65, 0.0 + self.ground_height),
            SE3(-1.400, -1.73, 0.5 + self.ground_height) * SE3.Rx(pi / 2) * SE3.Ry(pi),
        ]
        colors = [(0.6, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]

        safety_objs = []
        for stl, pose, color in zip(stl_files, poses, colors):
            mesh = geometry.Mesh(
                os.path.join(safety_dir, stl),
                pose=pose,
                scale=(0.001, 0.001, 0.001),
                color=color,
            )
            safety_objs.append(mesh)
            self.env.add(mesh)

        self.built += 1
        return safety_objs

    def load_object(self, index: int):
        """Spawn a brick mesh at a start pose and track it."""
        obj_path = os.path.join(os.path.abspath("Objects"), "Brick.stl")
        pose = self.brick_origin[index]
        obj = geometry.Mesh(obj_path, pose=pose)
        self.env.add(obj)
        self.bricks.append((index, obj))

        self.built += 1
        if self.built == 4:
            print("[Start Up] World & Objects All Built")
            
        return obj

    def run_mission(self):
        """Spawn two bricks, run KR6 then LBR pick-and-place."""
        self.load_object(0)
        self.load_object(1)
        print("[Mission] Welding Factory Beginning")

        self.kr6.pick_and_place(self.brick_origin[0], self.brick_place_pos[0], steps=50, brick_idx=0)
        self.kr6.gripper.actuate("open")

        self.lbr.pick_and_place(self.brick_origin[1], self.brick_place_pos[1], steps=50, brick_idx=1)


# -------------------------------------------------------------
# RobotUnit (handles robot + gripper + control)
# -------------------------------------------------------------
class RobotUnit:
    """A single robot with a gripper and motion helpers."""

    def __init__(self, robot, env: swift.Swift, base_pose: SE3, q_init=None, collision_zones=None):
        self.robot = robot
        self.env = env
        self.environment = getattr(env, "_parent", env)  # Environment instance
        self.robot.base = base_pose
        self.robot.q = np.zeros(self.robot.n) if q_init is None else q_init
        self.robot.add_to_env(env)

        self.gripper = Gripper(self.robot.fkine(self.robot.q))
        self.gripper.add_to_env(env)

        # ---- Per-robot collision detectors (multiple + persistent) ----
        if collision_zones is None or len(collision_zones) == 0:
            collision_zones = [
                dict(volume=(0.40, 0.50, 0.50), center=(0.95, 0.00, 0.25), colour=(0.0, 1.0, 0.0, 0.45))
            ]
        self.detectors = [
            CollisionDetector(env, z["volume"], z["center"], z.get("colour", (0.0, 1.0, 0.0, 0.45)))
            for z in collision_zones
        ]

    # ------------------------- public API -------------------------

    def pick_and_place(self, pick_pose: SE3, place_pose: SE3, steps=50, brick_idx=None):
        """Open → move above pick → close → carry to place (updates brick pose)."""
        print(f"[{self.robot.name}] Starting pick and place task")
        self.gripper.actuate("open")

        # Approach pick (hover + descend)
        self.move_to(pick_pose, steps)
        self.gripper.actuate("close")

        # Move carried brick to place pose
        self.move_to(place_pose, steps, brick_idx)
        print(f"[{self.robot.name}] Completed pick and place")

    def move_to(self, target_pose: SE3, steps=50, brick_idx=None):
        """
        Go via hover (z + 0.14) then descend to target.
        If brick_idx is set, carry the tracked brick along the way.
        """
        hover_pose = target_pose * SE3(0, 0, 0.14)

        ok_hover, traj_hover = self._ik_traj(hover_pose, steps)
        if not ok_hover:
            print(f"[{self.robot.name}] Hover pose not reachable:\n{target_pose}")
            return

        ok_goal, traj_goal = self._ik_traj(target_pose, steps, q0=traj_hover[-1])
        if not ok_goal:
            print(f"[{self.robot.name}] Target pose not reachable:\n{target_pose}")
            return

        # Execute trajectories sequentially with collision checks
        self._execute_trajectory(traj_hover, brick_idx)
        self._execute_trajectory(traj_goal, brick_idx)

    # ------------------------- helpers -------------------------

    def _ik_traj(self, target_pose: SE3, steps=50, q0=None):
        """
        IK to corrected TCP pose then generate a joint trajectory.
        Keeps your original TCP correction exactly as-is.
        """
        if q0 is None:
            q0 = self.robot.q.copy()

        # Original tool orientation & TCP offset corrections:
        target_corrected = target_pose * SE3.RPY(0, pi, pi / 2) * SE3(0, 0, -0.165)

        # Use q0 as the initial guess if provided, otherwise zeros (preserves original behavior)
        q_init = q0 if q0 is not None else np.zeros(self.robot.n)
        ik = self.robot.ikine_LM(target_corrected, q0=q_init, joint_limits=True)

        if not ik.success:
            return False, []

        traj = jtraj(q0, ik.q, steps).q
        return True, traj

    def _execute_trajectory(self, traj, brick_idx):
        """
        Step through a trajectory with per-robot collision checking (across multiple zones)
        and brick carry updates.
        """
        collided = False
        for q in traj:
            if collided:
                break

            # Check THIS robot against ALL of its zones
            if any(det.check_pose(self.robot, q) for det in self.detectors):
                collided = True
                print("[Collision Detection]: Robot collided with object, stopping further motion.")
                break

            # Apply joint state
            self.robot.q = q
            self.env.step(0.016)

            # Carry brick (if applicable) – matches your original transform and rotation
            if brick_idx is not None:
                _, brick_mesh = self.environment.bricks[brick_idx]
                brick_mesh.T = self.robot.fkine(q) * SE3(0, 0, 0.14) * SE3.Rz(pi / 2)

            # Update gripper pose
            self.gripper.update(self.robot.fkine(q))

            time.sleep(0.03)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    environment = Environment()
    environment.run_mission()
    environment.env.hold()

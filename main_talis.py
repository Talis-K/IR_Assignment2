#!/usr/bin/env python3
import os
import time
from math import pi

import numpy as np
import swift
import spatialgeometry as geometry
from spatialgeometry import Cuboid, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj

# Robots & extras
from ir_support import UR3
from KUKA_KR6.KR6 import KR6_Robot as KR6
from KUKA_LBR.lbr_loader import Load as LBR
from KUKA_LWR.kuak_lwr import Load as LWR
from gripper import Gripper
from welder import Welder

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
        self.conveyer_height = 0.1  # (kept original name to avoid breaking anything)
        self.support_height = 0.2
        support1 = Cuboid(scale = [0.3, 0.1, self.support_height], pose = SE3(0,1.2, self.support_height/2+self.ground_height), color =  [0.6, 0.6, 0.6] )
        support2 = Cuboid(scale = [0.3, 0.1, self.support_height],pose =  SE3(0,-1.2, self.support_height/2+self.ground_height), color = [0.6, 0.6, 0.6] )

        self.env.add(support1)
        self.env.add(support2)
        self.env.add(
            Cuboid(
                scale=[0.3, 2.5, self.conveyer_height],
                pose=SE3(0, 0, self.conveyer_height / 2 + self.ground_height + self.support_height),
                color=[0.1, 0, 0],
            )
        )

        
        # Robots (now with multiple zones per robot)
        self.load_robots()

        # Brick/Object Start and End Positions
         # Track (index, geometry.Mesh)
        self.object_height = 0.04
        self.object_width = 0.07
        self.object_length = 0.12
        self.bricks = []
        self.brick_origin = [
            SE3(1.3, 1.0, self.ground_height + self.object_height/2),
            SE3(1.3, 0.0, self.ground_height + self.object_height/2),
        ]
        self.brick_place_pos = [
            SE3(0.0, 1.0, 0.33),
            SE3(0.0, 0.0, 0.41) * SE3.RPY(pi / 2, 0, pi/2),
        ]

       

    def add_world(self):
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0,  self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0, -self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3( self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3(-self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, self.ground_length, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1.25, -1.35, 0.3), color=[0.5, 0.5, 0.9, 0.5]))

    def load_robots(self):
        """Add all robots at their bases, each with its own multiple collision detector zones."""

        # Streamlined Collision Zone Definer
        def Z(center, volume=(0.3,0.3,0.8), colour=(0.0, 1.0, 0.0, 0.45)):
            return dict(volume=volume, center=center, colour=colour)

        # Collision Zones
        kr6_zones = [
            Z((0.7,0,0.4),colour=(0.0, 0.1, 0.0, 0.001)), #For LBR Zone
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        lbr_zones = [
            Z((0.7,1,0.4),colour=(0.0, 0.1, 0.0, 0.001)),  #For KR6 Zone
            Z((0.7,-1,0.4),colour=(0.0, 0.1, 0.0, 0.001)), #For LWR Zone
            Z((0,0,self.conveyer_height / 2 + self.ground_height),(0.5,2.7,0.3), (0.0, 0.1, 0.0, 0.001))] #For Conveyer

        lwr_zones = [
            Z((0.7,0,0.4),colour=(0.0, 0.1, 0.0, 0.001)), #For LBR Zone
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
        self.lwr = RobotUnit(LWR(), self.env, SE3(0.5, -1.0, self.ground_height),
            collision_zones=lwr_zones)

        #UR3 booster seat
        self.ur3_stand = 0.3
        self.env.add(Cuboid(scale = [0.2, 0.2, self.ur3_stand], pose = SE3(-0.7, 0, self.ur3_stand/2 + self.ground_height), color = [0.5, 0.3, 0.3]))
        # UR3
        self.ur3 = RobotUnit(UR3(), self.env, SE3(-0.7, 0.0, self.ground_height + self.ur3_stand),
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
        pose = self.brick_origin[index]
        obj = Cuboid(scale = [self.object_width, self.object_length, self.object_height],pose = pose, color = [0.6, 0.6, 0.6])
        self.env.add(obj)
        self.bricks.append((index, obj))

        self.built += 1
        print("[Start Up] World & Objects All Built")
            
        return obj
    
    def translate_object(self, target_obj, target_pose):
        if isinstance(target_obj, tuple):
            idx, target_obj = target_obj
        initial_pose = SE3(target_obj.T)
        print(f"Translating brick {idx} from, {initial_pose} to {target_pose}")
        for alpha in np.linspace(0, 1, 50):
            target_obj.T = initial_pose.interp(target_pose*SE3.Ry(pi), alpha).A

            # Step simulation
            self.env.step(0.02)
            time.sleep(0.02)

    def translate_objects(self, target_objs, translation, steps=50):
  
        # Ensure we have a list
        if not isinstance(target_objs, (list, tuple)):
            target_objs = [target_objs]

        dx, dy, dz = translation
        dT = SE3(dx, dy, dz)  # relative translation transform

        # Record initial poses
        initial_poses = []
        for obj in target_objs:
            if isinstance(obj, tuple):
                idx, obj = obj
            initial_poses.append(SE3(obj.T))

        # Interpolate and move all together
        for alpha in np.linspace(0, 1, steps):
            for i, obj_entry in enumerate(target_objs):
                if isinstance(obj_entry, tuple):
                    idx, obj = obj_entry
                else:
                    obj = obj_entry

                start = initial_poses[i]
                obj.T = start.interp(dT * start, alpha).A

            self.env.step(0.02)
            time.sleep(0.02)


    def run_mission(self):
        """Spawn two bricks, run KR6 then LBR pick-and-place."""
        self.load_object(0)
        self.load_object(1)
        print("[Mission] Welding Factory Beginning")

        self.kr6.pick_and_place(self.brick_origin[0], self.brick_place_pos[0], steps=50, brick_idx=0)
        self.kr6.gripper.actuate("open")
        self.kr6.home()
        self.translate_object(self.bricks[0], self.brick_place_pos[0] * SE3(0,-1,0))

        self.lbr.pick_and_place(self.brick_origin[1], self.brick_place_pos[1], steps=50, brick_idx=1)

        self.ur3.weld(SE3(-0.02-0.16,self.object_width/2,0.36)*SE3.RPY(0,pi/2,0), SE3(-0.02-0.16,-self.object_width/2,0.36)*SE3.RPY(0,pi/2,0))
        self.ur3.home()
        self.lbr.gripper.actuate("open")
        self.lbr.home()

        self.translate_objects([self.bricks[0], self.bricks[1]] + self.ur3.gripper.welds, (0, -1, 0))
        
        self.lwr.pick_and_place(self.brick_place_pos[1]*SE3(-1,0,0) * SE3.RPY(-pi/2, 0, 0), self.brick_place_pos[1]* SE3(-1,-0.3,1.2) * SE3.RPY(-pi/2, 0, 0), brick_idx=2)




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
        
        if self.robot.name == "UR3":
            self.gripper = Welder(self.robot.fkine(self.robot.q))
        else:
            self.gripper = Gripper(self.robot.fkine(self.robot.q))
        self.gripper.add_to_env(env)
        self.gripper_carry_offset = 0.15


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

    def weld(self, weld_begin, weld_end, num_points=5):
        """
        Move the robot in a straight line from weld_begin to weld_end,
        generating weld sparks along the path.
        """
        # --- 1. Move above the start position ---
        hover_start = weld_begin * SE3(0, 0, -0.1 - 0.16)
        ik_hover = self.robot.ikine_LM(hover_start, q0=self.robot.q, joint_limits=False)
        if ik_hover.success:
            traj_hover = jtraj(self.robot.q, ik_hover.q, 50).q
            for q in traj_hover:
                self.robot.q = q
                self.env.step(0.02)
                self.gripper.update(self.robot.fkine(q))
                time.sleep(0.02)
        else:
            print(f"[{self.robot.name}] Failed to reach hover start pose.")
            return

        # --- 2. Move down to the weld start ---
        ik_start = self.robot.ikine_LM(weld_begin, q0=self.robot.q, joint_limits=False)
        if ik_start.success:
            traj_start = jtraj(self.robot.q, ik_start.q, 50).q
            for q in traj_start:
                self.robot.q = q
                self.env.step(0.02)
                self.gripper.update(self.robot.fkine(q))
                time.sleep(0.02)
        else:
            print(f"[{self.robot.name}] Failed to reach weld start pose.")
            return

        # --- 3. Linear weld motion in Cartesian space ---
        weld_poses = [weld_begin.interp(weld_end, s) for s in np.linspace(0, 1, num_points)]
        for pose in weld_poses:
            ik = self.robot.ikine_LM(pose, q0=self.robot.q, joint_limits=False)
            if not ik.success:
                print(f"[{self.robot.name}] Failed IK at weld pose.")
                continue

            # Small smooth transition between poses
            traj = jtraj(self.robot.q, ik.q, 5).q
            for q in traj:
                self.robot.q = q
                self.env.step(0.02)
                self.gripper.weld(self.robot.fkine(q))
                time.sleep(0.02)

        
        else:
            print(f"[{self.robot.name}] Failed to retract after weld.")



    def home(self, steps=50):

        home_q = np.zeros(self.robot.n)
        if (self.robot.q == home_q).all():
            print(f"{self.robot.name} is already home")
            return
    
        current_pose = self.robot.fkine(self.robot.q)
        lifted_pose = current_pose * SE3(0, 0, -0.2)

        # Try to solve IK for the lifted pose
        ik_lift = self.robot.ikine_LM(lifted_pose, q0=self.robot.q, joint_limits=True)
        if ik_lift.success:
            traj_lift = jtraj(self.robot.q, ik_lift.q, steps).q

            for q in traj_lift:
                self.robot.q = q
                self.env.step(0.02)
                self.gripper.update(self.robot.fkine(q))
                time.sleep(0.02)
        else:
            print(f"{self.robot.name} could not find lift position before going home")
            
        traj_home = jtraj(self.robot.q, home_q, steps).q
        for q in traj_home:
            self.robot.q = q
            self.env.step(0.02)
            self.gripper.update(self.robot.fkine(q))
            time.sleep(0.02)

        print(f"[{self.robot.name}] Returned to home position")

    def pick_and_place(self, pick_pose: SE3, place_pose: SE3, steps=50, brick_idx=None):
        print(f"[{self.robot.name}] Starting pick and place task")
        self.gripper.actuate("open")

        # Approach pick (hover + descend)
        self.move_to(pick_pose, steps)
        self.gripper.actuate("close")

        # Move carried brick to place pose
        self.move_to(place_pose, steps, brick_idx)
        print(f"[{self.robot.name}] Completed pick and place")

    def move_to(self, target_pose: SE3, steps=50, brick_idx=None):
  
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

        if q0 is None:
            q0 = self.robot.q.copy()
        
        target_corrected = target_pose * SE3.RPY(0, pi, pi / 2) * SE3(0, 0, -self.gripper_carry_offset)
        q_init = q0 if q0 is not None else np.zeros(self.robot.n)
        ik = self.robot.ikine_LM(target_corrected, q0=q_init, joint_limits=True)

        if not ik.success:
            return False, []

        traj = jtraj(q0, ik.q, steps).q
        return True, traj

    def _execute_trajectory(self, traj, brick_idx):
        """
        Step through a trajectory with per-robot collision checking (across multiple zones)
        and object carry updates.
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

            # ---------------------------------------------
            # Carry brick(s) or welds as needed
            # ---------------------------------------------
            ee_pose = self.robot.fkine(q)

            if brick_idx is not None:
                # Normal pick-and-place brick motion
                if brick_idx in [0, 1]:
                    _, brick_mesh = self.environment.bricks[brick_idx]
                    brick_mesh.T = ee_pose * SE3(0, 0, self.gripper_carry_offset) * SE3.Rz(pi / 2)

                # Special case: move both bricks and welds together
                elif brick_idx == 2:
                    # Move both bricks
                    for idx in [0]:
                        _, brick_mesh = self.environment.bricks[idx]
                        brick_mesh.T = ee_pose * SE3(0, 0, self.gripper_carry_offset) * SE3.Rx(pi/2) * SE3.Ry(pi / 2)

                    for idx in [1]:
                        _, brick_mesh = self.environment.bricks[idx]
                        brick_mesh.T = ee_pose * SE3(0, 0, self.environment.object_length/2 + self.environment.object_height/2 + self.gripper_carry_offset)

                    # Move weld meshes (stored in ur3.gripper.welds)
                    welds = getattr(self.environment.ur3.gripper, "welds", [])
                    num_welds = len(welds)
                    if num_welds > 0:
                        # Evenly distribute welds along local X-axis
                        length = self.environment.object_width
                        offset = length/num_welds
                        for i, weld in enumerate(welds):
                            weld.T = (
                                ee_pose
                                * SE3(0, 0, self.gripper_carry_offset)
                                * SE3.Rz(pi / 2)
                                * SE3(-self.environment.object_width/2+offset*i, self.environment.object_height/2, self.environment.object_length/2)
                            )

            # Update gripper pose
            self.gripper.update(ee_pose)

            time.sleep(0.03)




# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    environment = Environment()
    environment.run_mission()
    environment.env.hold()

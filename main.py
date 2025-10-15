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
#For gripper
from gripper import Gripper as Gripper








class Environment:
  def __init__(self):
      # Swift Set Up
      self.env = swift.Swift()
      self.env.launch(realTime=True)
      self.env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, -pi/4])




      # Fences, ground, safety box
      self.ground_height = 0.005
      self.ground_length = 3.5
      self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0,  self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
      self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8], pose=SE3(0, -self.ground_length/2, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
      self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3( self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
      self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8], pose=SE3(-self.ground_length/2, 0, 0.4 + self.ground_height), color=[0.5, 0.9, 0.5, 0.5]))
      self.env.add(Cuboid(scale=[self.ground_length, self.ground_length, 2*self.ground_height], pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))
      self.env.add(Cuboid(scale=[0.9, 0.04, 0.6], pose=SE3(-1.25, -1.35, 0.3), color=[0.5, 0.5, 0.9, 0.5]))
      self.safety = self.load_safety()




      # Conveyer
      self.conveyer_height = 0.3
      self.env.add(Cuboid(scale=[0.3, 2.5, self.conveyer_height], pose=SE3(0,0,self.conveyer_height/2 + self.ground_height), color=[0,0,0]))




      self.load_robots()
      self.object_origin = [
          SE3(1.3,1,self.ground_height),
          SE3(1.3,0,self.ground_height),
      ]




      # Bricks
      self.bricks = []
      self.brick_counters = {0: 0, 1: 0}








  def load_robots(self):
    
      # --- KUKA KR6 ---
      self.kr6 = KR6()
      self.kr6.q = np.zeros(6)
      self.kr6.base = SE3(0.7, 1, self.ground_height)
      self.gripper_kr6 = Gripper(self.kr6.fkine(self.kr6.q))
      self.kr6.add_to_env(self.env)
      self.gripper_kr6.add_to_env(self.env)
    
      # --- KUKA LBR ---
      self.lbr = LBR()
      self.lbr.q = np.zeros(7)
      self.lbr.base = SE3(0.7, 0, self.ground_height)
      self.gripper_lbr = Gripper(self.lbr.fkine(self.lbr.q))
      self.lbr.add_to_env(self.env)
      self.gripper_lbr.add_to_env(self.env)




      # --- KUKA LWR ---
      self.lwr = LWR()
      self.lwr.q = np.zeros(7)
      self.lwr.base = SE3(0.7, -1, self.ground_height)
      self.gripper_lwr = Gripper(self.lwr.fkine(self.lwr.q))
      self.lwr.add_to_env(self.env)
      self.gripper_lwr.add_to_env(self.env)




      # --- UR3 ---
      self.ur3 = UR3()
      self.ur3.q = np.array([pi/2, -pi/2, 0, -pi/2, 0, -pi/2])
      self.ur3.base = SE3(-0.45, 0, self.ground_height)
      self.gripper_ur3 = Gripper(self.ur3.fkine(self.ur3.q))
      self.ur3.add_to_env(self.env)
      self.gripper_ur3.add_to_env(self.env)
    
  def load_safety(self):
      safety_dir = os.path.abspath("Safety")
      stl_files = ["button.stl", "Fire_extinguisher.stl", "generic_caution.STL"]
      safety_positions = [
          SE3(-1.55, -1.6, 0.0 + self.ground_height) * SE3.Rx(pi/2),
          SE3(-1.350, -1.65, 0.0 + self.ground_height),
          SE3(-1.4, -1.73, 0.5 + self.ground_height) * SE3.Rx(pi/2) * SE3.Ry(pi)
      ]
      safety_colour = [
          (0.6, 0.0, 0.0, 1.0),
          (0.5, 0.0, 0.0, 1.0),
          (1.0, 1.0, 0.0, 1.0)
      ]
      safety = []
      for stl_file, pose, colour in zip(stl_files, safety_positions, safety_colour):
          stl_path = os.path.join(safety_dir, stl_file)
          safety_obj = geometry.Mesh(stl_path, pose=pose, scale=(0.001, 0.001, 0.001), color=colour)
          self.env.add(safety_obj)
          safety.append(safety_obj)
      return safety




  def load_object(self, pose_index):
      obj_path = os.path.join(os.path.abspath("Objects"), "Brick.stl")
      counter = self.brick_counters[pose_index]
      self.brick_counters[pose_index] += 1
      pose = self.object_origin[pose_index] * SE3(0, 0, self.ground_height + counter * 0.01)
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
      initial_pose = brick.T[:3, 3]
      target_pose = self.object_position[pose_index] * SE3(0, 0, 0)
      target_position = target_pose.t
      rotation_matrix = brick.T[:3, :3]




      print(f"Moving brick at pose_index {pose_index} along conveyer, counter {counter}")
      print(f"  From: {initial_pose}  To: {target_position}")




      steps = 25
      for s in np.linspace(0, 1, steps):
          interpolated_position = (1 - s) * initial_pose + s * target_position
          new_pose = np.eye(4)
          new_pose[:3, :3] = rotation_matrix
          new_pose[:3, 3] = interpolated_position
          brick.T = new_pose
          self.env.step(0.02)
          time.sleep(0.03)








class Control:
  def __init__(self, robot, env, gripper=None):
      self.robot = robot
      self.env = env
      self.gripper = gripper




  def move_to(self, target_pose, steps):
      print(f"Robot 1 {self.robot.name} moving to object 1 collection position at {target_pose}")
      possible, traj = self.check_and_calculate_joint_angles(target_pose, steps)
      if not possible:
          print(f"[{self.robot.name}] Target pose is not reachable")
          return False
      for q in traj:
          self.robot.q = q
          if self.gripper is not None:
              self.gripper.update(self.robot.fkine(q))
          self.env.step(0.02)
          time.sleep(0.03)
      return True




  def check_and_calculate_joint_angles(self, target_pose, steps=50):
       original_q = self.robot.q.copy()
       q0 = original_q
       # IK (LM)
       target_pose_corrected = target_pose * SE3.RPY(0,pi,pi/2) * SE3(0, 0,-0.165)
       ik = self.robot.ikine_LM(target_pose_corrected, q0=q0, joint_limits=True)


       if not ik.success:
          # second try with zero seed
          q0b = np.zeros_like(q0)
          ik = self.robot.ikine_LM(target_pose_corrected, q0=q0b, joint_limits=True)
          if not ik.success:
              return False, []
       q_goal = ik.q
       print(f"[{self.robot.name}] IK solution: {q_goal}")
       traj = jtraj(original_q, q_goal, steps).q
       self.robot.q = original_q
       return True, traj








def sample_reachable_pose(xy_bounds, z_bounds, face_down=True):
  """
  Returns SE3 in the *robot base frame*.
  xy_bounds = ((xmin,xmax), (ymin,ymax)), z_bounds=(zmin,zmax)
  face_down=True adds Rx(pi) to point TCP downwards (easier IK for pick/inspect).
  """
  x = random.uniform(*xy_bounds[0])
  y = random.uniform(*xy_bounds[1])
  z = random.uniform(*z_bounds)
  T = SE3(x, y, z)
  if face_down:
      T = T * SE3.Rx(pi)
  return T








class Mission:
  def __init__(self, env, ctl_ur3, ctl_lbr, ctl_lwr, ctl_kr6):




      self.env = env
      self.ctl_ur3 = ctl_ur3
      self.ctl_lbr = ctl_lbr
      self.ctl_lwr = ctl_lwr
      self.ctl_kr6 = ctl_kr6




      # per-robot random bounds in their base frames (tuned to be conservative)
      self.ur3_bounds = ((0.25, 0.85), (-0.25, 0.25)), (0.25, 0.85)   # xy, z
      self.lwr_bounds = ((-0.15, 0.35), (-0.20, 0.25)), (0.45, 0.95)
      self.kr6_bounds = ((0.10, 0.45), (-0.20, 0.25)), (0.30, 0.70)




  def _move_random_many(self, controller, xy_bounds, z_bounds, count=4, max_tries=40):
      moved = 0
      tries = 0
      while moved < count and tries < max_tries:
          tries += 1
          T = sample_reachable_pose(xy_bounds, z_bounds, face_down=True)
          ok = controller.move_to(T, steps=50)
          if ok:
              moved += 1




  def run(self):


   print("Begining simulation")
   print("Loading bricks")
   self.env.load_object(0)
   self.env.load_object(1)


   self.ctl_kr6.gripper.actuate("open")


   success = self.ctl_kr6.move_to(self.env.object_origin[0], 50)
   if not success:
       print(f"KUKA LBR failed to reach pose {self.env.object_positions[0]}")


   success = self.ctl_lbr.move_to(self.env.object_origin[1], 50)
   if not success:
       print(f"KUKA LBR failed to reach pose {self.env.object_positions[1]}") 












  




   print("Roobot 2 (KUKA LBR) moved sucseffly to (forward kinematics)")




   print("Closing gripper")




   # self.env.gripper_lbr.actuate("close")








if __name__ == "__main__":
  assignment = Environment()
  ctl_ur3 = Control(assignment.ur3, assignment.env, assignment.gripper_ur3)
  ctl_lbr = Control(assignment.lbr, assignment.env, assignment.gripper_lbr)
  ctl_lwr = Control(assignment.lwr, assignment.env, assignment.gripper_lwr)
  ctl_kr6 = Control(assignment.kr6, assignment.env, assignment.gripper_kr6)




  mission = Mission(assignment, ctl_ur3, ctl_lbr, ctl_lwr, ctl_kr6)
  mission.run()
  assignment.env.hold()












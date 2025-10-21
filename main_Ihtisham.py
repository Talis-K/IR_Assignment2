import os
import sys
import time
from math import pi
import threading
import select
import termios
import tty

import numpy as np
import swift
import spatialgeometry as geometry
from spatialgeometry import Cuboid
from spatialmath import SE3
from roboticstoolbox import jtraj

from ir_support import UR3
from KUKA_KR6.KR6 import KR6_Robot as KR6
from KUKA_LBR.LBR import Load as LBR
from Kuka_LWR.LWR import Load as LWR
from gripper import Gripper
from welder import Welder
from collisiontester import CollisionDetector
from override import bus


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    import pygame  
except Exception:
    pygame = None



class EStopGate:
    """Latched emergency stop. When engaged, wait_if_engaged() blocks until released."""
    def __init__(self):
        self._latched = False
        self._cv = threading.Condition()

    def engage(self):
        with self._cv:
            self._latched = True

    def release(self):
        with self._cv:
            self._latched = False
            self._cv.notify_all()

    def toggle(self):
        with self._cv:
            self._latched = not self._latched
            if not self._latched:
                self._cv.notify_all()

    def is_engaged(self) -> bool:
        with self._cv:
            return self._latched

    def wait_if_engaged(self):
        with self._cv:
            while self._latched:
                self._cv.wait(timeout=0.05)


ESTOP = EStopGate()


def _start_ps4_estop_listener():
    """PS4 controller listener (pygame joystick only) + stdin fallback (e/r)."""
    def _joystick_loop():
        js = None
        if pygame is not None:
            try:
            
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    print("[E-STOP] PS4 detected; X=engage, Triangle=release.")
                else:
                    print("[E-STOP] No joystick detected; using stdin fallback (e/r).")
            except Exception as e:
                print(f"[E-STOP] pygame joystick init failed: {e}. Using stdin fallback.")
                js = None
        else:
            print("[E-STOP] pygame not available; using stdin fallback (e/r).")

        while True:
            if js is not None and pygame is not None:
                try:
                   
                    pygame.event.pump()
                  
                    if js.get_button(1):   
                        ESTOP.engage()
                    if js.get_button(3): 
                        ESTOP.release()
                except Exception:
                    pass
            time.sleep(0.01)

    def _stdin_loop():
       
        if not sys.stdin.isatty():
            return 
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print("[E-STOP] Stdin controls active: press 'e' to engage, 'r' to release.")
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == 'e':
                        ESTOP.engage()
                        print("[E-STOP] Engaged (stdin 'e').")
                    elif ch == 'r':
                        ESTOP.release()
                        print("[E-STOP] Released (stdin 'r').")
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    threading.Thread(target=_joystick_loop, daemon=True).start()
    threading.Thread(target=_stdin_loop,    daemon=True).start()



class ConveyorController:
    def __init__(self, env: swift.Swift, belt_obj: Cuboid):
        self.env = env
        self.belt = belt_obj
        self.running = True
        self._lock = threading.Lock()
        self.carried = [] 
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_running(self, running: bool):
        with self._lock:
            self.running = bool(running)

    def _loop(self):
        phase = 0.0
        while True:
            ESTOP.wait_if_engaged()  

            with self._lock:
                run = self.running

            if run and not ESTOP.is_engaged():
                for obj in list(self.carried):
                    T = SE3(obj.T)
                    obj.T = (SE3(T) * SE3(0, 0.002, 0)).A  

            self.env.step(0.02)
            time.sleep(0.02)
            phase += 0.02



class Environment:
    def __init__(self):
        self.built = 0
        self.env = swift.Swift()
        self.env._parent = self
        self.env.launch(realtime=True)
        self.env.set_camera_pose([2, 2, 2], [0, 0, -pi / 4])

        # Start E-Stop listener
        _start_ps4_estop_listener()

        self.ground_height = 0.005
        self.ground_length = 3.5
        self.add_world()
        self.safety = self.load_safety()
        self.conveyer_height = 0.1
        self.support_height = 0.2
        support1 = Cuboid(scale=[0.3, 0.1, self.support_height],
                          pose=SE3(0, 1.2, self.support_height/2 + self.ground_height),
                          color=[0.6, 0.6, 0.6])
        support2 = Cuboid(scale=[0.3, 0.1, self.support_height],
                          pose=SE3(0, -1.2, self.support_height/2 + self.ground_height),
                          color=[0.6, 0.6, 0.6])

        self.env.add(support1)
        self.env.add(support2)
        belt = Cuboid(scale=[0.3, 2.5, self.conveyer_height],
                      pose=SE3(0, 0, self.conveyer_height/2 + self.ground_height + self.support_height),
                      color=[0.1, 0, 0])
        self.env.add(belt)

        self.conveyor = ConveyorController(self.env, belt)

        self.load_robots()
        self.object_height = 0.04
        self.object_width = 0.07
        self.object_length = 0.12
        self.bricks = []
        self.brick_origin = [
            SE3(1.3, 1.0, self.ground_height + self.object_height/2) * SE3.RPY(0, pi, 0),
            SE3(1.3, 0.0, self.ground_height + self.object_height/2) * SE3.RPY(0, pi, 0),
        ]
        self._start_override_follower()

    def _start_override_follower(self):
        def _worker():
            while True:
                ESTOP.wait_if_engaged() 

                
                self.conveyor.set_running((not bus.should_pause_conveyor()) and (not ESTOP.is_engaged()))

                for unit in [self.kr6, self.lbr, self.lwr, self.ur3]:
                    if bus.is_enabled_for(unit.robot.name):
                        q_over = bus.get_q_for(unit.robot.name)
                        if q_over is not None:
                            unit.robot.q = q_over
                            unit.gripper.update(unit.robot.fkine(unit.robot.q))

                        g = bus.get_gripper_closed_for(unit.robot.name)
                        if g is not None and isinstance(unit.gripper, Gripper):
                            unit.gripper.actuate("close" if g else "open")

                self.env.step(0.02)
                time.sleep(0.02)
        threading.Thread(target=_worker, daemon=True).start()

    def add_world(self):
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8],
                            pose=SE3(0, self.ground_length/2, 0.4 + self.ground_height),
                            color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, 0.05, 0.8],
                            pose=SE3(0, -self.ground_length/2, 0.4 + self.ground_height),
                            color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8],
                            pose=SE3(self.ground_length/2, 0, 0.4 + self.ground_height),
                            color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[0.05, self.ground_length, 0.8],
                            pose=SE3(-self.ground_length/2, 0, 0.4 + self.ground_height),
                            color=[0.5, 0.9, 0.5, 0.5]))
        self.env.add(Cuboid(scale=[self.ground_length, self.ground_length, 2*self.ground_height],
                            pose=SE3(0, 0, 0), color=[0.9, 0.9, 0.5, 1]))
        self.env.add(Cuboid(scale=[0.9, 0.04, 0.6],
                            pose=SE3(-1.25, -1.35, 0.3), color=[0.5, 0.5, 0.9, 0.5]))

    def load_robots(self):
        def Z(center, volume=(0.3, 0.3, 0.8), colour=(0.0, 1.0, 0.0, 0.45)):
            return dict(volume=volume, center=center, colour=colour)

        kr6_zones = [
            Z((0.7, 0, 0.4), colour=(0.0, 0.1, 0.0, 0.001)),
            Z((0, 0, self.conveyer_height/2 + self.ground_height), (0.5, 2.7, 0.3), (0.0, 0.1, 0.0, 0.001)),
        ]
        lbr_zones = [
            Z((0.7, 1, 0.4), colour=(0.0, 0.1, 0.0, 0.001)),
            Z((0.5, -1, 0.4), colour=(0.0, 0.1, 0.0, 0.001)),
            Z((0, 0, self.conveyer_height/2 + self.ground_height), (0.5, 2.7, 0.3), (0.0, 0.1, 0.0, 0.001)),
        ]
        lwr_zones = [
            Z((0.7, 0, 0.4), colour=(0.0, 0.1, 0.0, 0.001)),
            Z((0, 0, self.conveyer_height/2 + self.ground_height), (0.5, 2.7, 0.3), (0.0, 0.1, 0.0, 0.001)),
        ]
        ur3_zones = [
            Z((0, 0, self.conveyer_height/2 + self.ground_height), (0.5, 2.7, 0.3), (0.0, 0.1, 0.0, 0.001)),
        ]

        self.kr6 = RobotUnit(KR6(), self.env, SE3(0.7, 1.0, self.ground_height), collision_zones=kr6_zones)
        self.lbr = RobotUnit(LBR(), self.env, SE3(0.7, 0.0, self.ground_height), collision_zones=lbr_zones)
        self.lwr = RobotUnit(LWR(), self.env, SE3(0.5, -1.0, self.ground_height), collision_zones=lwr_zones)

        # UR3 stand
        self.ur3_stand = 0.3
        self.env.add(Cuboid(scale=[0.2, 0.2, self.ur3_stand],
                            pose=SE3(-0.7, 0, self.ur3_stand/2 + self.ground_height),
                            color=[0.5, 0.3, 0.3]))
        self.ur3 = RobotUnit(
            UR3(), self.env,
            SE3(-0.7, 0.0, self.ground_height + self.ur3_stand),
            q_init=[pi/2, -pi/2, 0, 0, pi/2, 0],
            collision_zones=ur3_zones
        )
        self.built += 1

    def load_safety(self):
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
                pose=pose, scale=(0.001, 0.001, 0.001), color=color
            )
            safety_objs.append(mesh)
            self.env.add(mesh)

        self.built += 1
        return safety_objs

    def load_box(self):
        box_dir = os.path.abspath("Objects")
        mesh = geometry.Mesh(os.path.join(box_dir, "Pallet.stl"),
                             pose=SE3(1.1, -1, 0.1005) * SE3.Rx(pi),
                             scale=(0.008, 0.008, 0.008),
                             color=(1.0, 1.0, 0.0, 1.0))
        self.env.add(mesh)
        self.built += 1
        return mesh

    def load_object(self, index: int):
        pose = self.brick_origin[index]
        obj = Cuboid(scale=[self.object_width, self.object_length, self.object_height],
                     pose=pose, color=[0.6, 0.6, 0.6])
        self.env.add(obj)
        self.bricks.append((index, obj))
        self.built += 1
        print("[Start Up] World & Objects All Built")
        return obj

    def translate_object(self, target_obj, target_pose):
        if isinstance(target_obj, tuple):
            idx, target_obj = target_obj
        initial_pose = SE3(target_obj.T)
        print(f"Translating brick from {initial_pose} to {target_pose}")
        for alpha in np.linspace(0, 1, 50):
            ESTOP.wait_if_engaged()
            target_obj.T = initial_pose.interp(target_pose * SE3.Ry(pi), alpha).A
            self.env.step(0.02)
            time.sleep(0.02)

    def translate_objects(self, target_objs, translation, steps=50):
        if not isinstance(target_objs, (list, tuple)):
            target_objs = [target_objs]
        dx, dy, dz = translation
        dT = SE3(dx, dy, dz)
        initial_poses = []
        for obj in target_objs:
            if isinstance(obj, tuple):
                _, obj = obj
            initial_poses.append(SE3(obj.T))
        for alpha in np.linspace(0, 1, steps):
            ESTOP.wait_if_engaged()
            for i, obj_entry in enumerate(target_objs):
                obj = obj_entry[1] if isinstance(obj_entry, tuple) else obj_entry
                start = initial_poses[i]
                obj.T = start.interp(dT * start, alpha).A
            self.env.step(0.02)
            time.sleep(0.02)

    def run_mission(self):
        self.KR6_place_pos = SE3(0.0, 1.0, 0.32) * SE3.RPY(0, pi, 0)
        self.lwr_place_pos = SE3(0.8, -1, 0.2) * SE3.RPY(0, pi, pi/2)

        self.load_box()
        self.load_object(0)
        self.load_object(1)
        print("[Mission] Welding Factory Beginning")

        self.kr6.pick_and_place(SE3(self.bricks[0][1].T), self.KR6_place_pos, steps=50, brick_idx=0)
        self.kr6.gripper.actuate("open")
        self.kr6.home()

        self.translate_object(self.bricks[0], self.KR6_place_pos * SE3(0, -1, 0) * SE3.RPY(0, pi, pi))

        self.lbr.pick_and_place(
            SE3(self.bricks[1][1].T),
            SE3(self.bricks[0][1].T) * SE3(0, 0, -self.object_height/2 - self.object_length/2) * SE3.RPY(pi/2, 0, -pi/2),
            steps=50, brick_idx=1
        )

        self.ur3.weld(
            SE3(self.bricks[1][1].T) * SE3(-self.object_width/2, -self.object_length/2, self.object_height/2 + self.ur3.gripper.tool_length) * SE3.RPY(0, pi, 0),
            SE3(self.bricks[1][1].T) * SE3(+self.object_width/2, -self.object_length/2, self.object_height/2 + self.ur3.gripper.tool_length) * SE3.RPY(0, pi, 0)
        )

        self.ur3.home()
        self.lbr.gripper.actuate("open")
        self.lbr.home()

        self.translate_objects([self.bricks[0], self.bricks[1]] + self.ur3.gripper.welds, (0, -1, 0))

        self.lwr.pick_and_place(SE3(self.bricks[1][1].T) * SE3.RPY(pi/2, 0, 0), self.lwr_place_pos, brick_idx=2)


class RobotUnit:
    """A single robot with a gripper and motion helpers."""
    def __init__(self, robot, env: swift.Swift, base_pose: SE3, q_init=None, collision_zones=None):
        self.robot = robot
        self.env = env
        self.environment = getattr(env, "_parent", env)
        self.robot.base = base_pose
        self.robot.q = np.zeros(self.robot.n) if q_init is None else q_init

        if hasattr(self.robot, "add_to_env"):
            self.robot.add_to_env(env)
        else:
            self.env.add(self.robot)

        if self.robot.name == "UR3":
            self.gripper = Welder(self.robot.fkine(self.robot.q))
        else:
            self.gripper = Gripper(self.robot.fkine(self.robot.q))
        self.gripper.add_to_env(env)
        self.gripper_carry_offset = 0.15

        if not collision_zones:
            collision_zones = [
                dict(volume=(0.40, 0.50, 0.50), center=(0.95, 0.00, 0.25), colour=(0.0, 1.0, 0.0, 0.45))
            ]
        self.detectors = [
            CollisionDetector(env, z["volume"], z["center"], z.get("colour", (0.0, 1.0, 0.0, 0.45)))
            for z in collision_zones
        ]

    def weld(self, weld_begin, weld_end, num_points=5):
        if bus.is_enabled_for(self.robot.name):
            print(f"[{self.robot.name}] Override active — weld aborted")
            return

        hover_start = weld_begin * SE3(0.05, 0, 0.0)
        ik_hover = self.robot.ikine_LM(hover_start, q0=self.robot.q, joint_limits=False)
        if not ik_hover.success:
            return
        for q in jtraj(self.robot.q, ik_hover.q, 50).q:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            self._step(q)

        ik_start = self.robot.ikine_LM(weld_begin, q0=self.robot.q, joint_limits=False)
        if not ik_start.success:
            return
        for q in jtraj(self.robot.q, ik_start.q, 50).q:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            self._step(q)

        for s in np.linspace(0, 1, num_points):
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            pose = weld_begin.interp(weld_end, s)
            ik = self.robot.ikine_LM(pose, q0=self.robot.q, joint_limits=False)
            if not ik.success:
                continue
            for q in jtraj(self.robot.q, ik.q, 5).q:
                if self._maybe_abort_override(): return
                ESTOP.wait_if_engaged()
                self._step(q, weld=True)

    def home(self, steps=50):
        if bus.is_enabled_for(self.robot.name):
            print(f"[{self.robot.name}] Override active — home aborted")
            return

        home_q = np.zeros(self.robot.n)
        if self.robot.name == "UR3":
            home_q = np.array([pi/2, -pi/2, 0, 0, pi/2, 0])

        current_pose = self.robot.fkine(self.robot.q)
        lifted_pose = current_pose * SE3(0, 0, -0.2)
        ik_lift = self.robot.ikine_LM(lifted_pose, q0=self.robot.q, joint_limits=True)
        if ik_lift.success:
            for q in jtraj(self.robot.q, ik_lift.q, steps).q:
                if self._maybe_abort_override(): return
                ESTOP.wait_if_engaged()
                self._step(q)

        for q in jtraj(self.robot.q, home_q, steps).q:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            self._step(q)
        print(f"[{self.robot.name}] Returned to home position")

    def pick_and_place(self, pick_pose: SE3, place_pose: SE3, steps=50, brick_idx=None):
        if bus.is_enabled_for(self.robot.name):
            print(f"[{self.robot.name}] Override active — pick/place aborted")
            return
        print(f"[{self.robot.name}] Starting pick and place task")
        if isinstance(self.gripper, Gripper):
            self.gripper.actuate("open")
        self.move_to(pick_pose, steps)
        if isinstance(self.gripper, Gripper):
            self.gripper.actuate("close")
        self.move_to(place_pose, steps, brick_idx)
        print(f"[{self.robot.name}] Completed pick and place")

    def move_to(self, target_pose: SE3, steps=50, brick_idx=None):
        if bus.is_enabled_for(self.robot.name):
            print(f"[{self.robot.name}] Override active — move aborted")
            return

        hover_pose = target_pose * SE3(0, 0, -0.14)
        ok_hover, traj_hover = self._ik_traj(hover_pose, steps)
        if not ok_hover:
            print(f"[{self.robot.name}] Hover pose not reachable:\n{target_pose}")
            return

        ok_goal, traj_goal = self._ik_traj(target_pose, steps, q0=traj_hover[-1])
        if not ok_goal:
            print(f"[{self.robot.name}] Target pose not reachable:\n{target_pose}")
            return

        self._execute_trajectory(traj_hover, brick_idx)
        self._execute_trajectory(traj_goal, brick_idx)


    def _maybe_abort_override(self) -> bool:
        if bus.is_enabled_for(self.robot.name):
            print(f"[{self.robot.name}] Manual override engaged — aborting auto motion.")
            return True
        return False

    def _ik_traj(self, target_pose: SE3, steps=50, q0=None):
        if q0 is None:
            q0 = self.robot.q.copy()
        target_corrected = target_pose * SE3(0, 0, -self.gripper_carry_offset) * SE3.RPY(0, 0, pi/2)
        ik = self.robot.ikine_LM(target_corrected, q0=q0, joint_limits=True)
        if not ik.success:
            return False, []
        traj = jtraj(q0, ik.q, steps).q
        return True, traj

    def _execute_trajectory(self, traj, brick_idx):
        for q in traj:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            if any(det.check_pose(self.robot, q) for det in self.detectors):
                print("[Collision Detection]: Robot collided with object, stopping further motion.")
                return
            self._step(q, carry_idx=brick_idx)

    def _step(self, q, carry_idx=None, weld=False):
        ESTOP.wait_if_engaged()
        self.robot.q = q
        self.env.step(0.016)
        ee_pose = self.robot.fkine(q)

        # Carry objects
        if carry_idx is not None:
            if carry_idx in [0, 1]:
                _, brick_mesh = self.environment.bricks[carry_idx]
                brick_mesh.T = ee_pose * SE3(0, 0, self.gripper_carry_offset) * SE3.Rz(pi/2)
            elif carry_idx == 2:
                for idx in [0]:
                    _, brick_mesh = self.environment.bricks[idx]
                    brick_mesh.T = ee_pose * SE3(0, 0, self.gripper_carry_offset) * SE3.Rx(pi/2) * SE3.Ry(pi/2)
                for idx in [1]:
                    _, brick_mesh = self.environment.bricks[idx]
                    brick_mesh.T = ee_pose * SE3(0, 0, self.environment.object_length/2 + self.environment.object_height/2 + self.gripper_carry_offset)
                welds = getattr(self.environment.ur3.gripper, "welds", [])
                n = len(welds)
                if n > 0:
                    length = self.environment.object_width
                    offset = length/n
                    for i, w in enumerate(welds):
                        w.T = (ee_pose * SE3(0, 0, self.gripper_carry_offset) *
                               SE3.Rz(pi/2) * SE3(-self.environment.object_width/2 + offset*i,
                                                  self.environment.object_height/2,
                                                  self.environment.object_length/2))
        # Update tool
        if weld and hasattr(self.gripper, "weld"):
            self.gripper.weld(ee_pose)
        else:
            self.gripper.update(ee_pose)
        time.sleep(0.01)

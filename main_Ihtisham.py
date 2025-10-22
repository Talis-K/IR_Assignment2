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


import time as _time
from collections import defaultdict as _dd

DEBUG = False  

class Debug:
    _last = _dd(float)
    _counts = _dd(int)

    @staticmethod
    def stamp(cat, msg, every_sec=0.0, count_every=None):
        if not DEBUG:
            return
        now = _time.time()
        if every_sec:
            if now - Debug._last[cat] < every_sec:
                Debug._counts[cat] += 1
                return
            Debug._last[cat] = now
        if count_every:
            Debug._counts[cat] += 1
            if Debug._counts[cat] % count_every != 0:
                return
        ts = _time.strftime("%H:%M:%S")
        print(f"[{ts}][{cat}] {msg}")




class EStopGate:
    """Latched emergency stop. When engaged, wait_if_engaged() blocks until released."""
    def __init__(self):
        self._latched = False
        self._cv = threading.Condition()

    def engage(self):
        with self._cv:
            self._latched = True
            Debug.stamp("E-STOP", ">>> ENGAGED")

    def release(self):
        with self._cv:
            self._latched = False
            self._cv.notify_all()
            Debug.stamp("E-STOP", "<<< RELEASED")

    def is_engaged(self) -> bool:
        with self._cv:
            return self._latched

    def wait_if_engaged(self):
        with self._cv:
            while self._latched:
                self._cv.wait(timeout=0.05)

ESTOP = EStopGate()

def _mirror_bus_estop():
    state = None
    while True:
        try:
            s = bool(bus.is_estop())
        except Exception:
            s = False
        if s != state:
            state = s
            if s:
                ESTOP.engage()
            else:
                ESTOP.release()
            Debug.stamp("BRIDGE", f"bus.estop -> {'ENGAGED' if s else 'RELEASED'}")
        time.sleep(0.05)

def _rate_limit(period_s=0.25):
    last = {"t": 0.0}
    def ok():
        now = time.time()
        if now - last["t"] >= period_s:
            last["t"] = now
            return True
        return False
    return ok

def _pretty_axes(js, deadzone=0.08):
    vals = []
    for i in range(js.get_numaxes()):
        a = js.get_axis(i)
        if abs(a) < deadzone: a = 0.0
        vals.append(f"A{i}:{a:+.2f}")
    return " ".join(vals)

def _pretty_buttons(js):
    pressed = [str(i) for i in range(js.get_numbuttons()) if js.get_button(i)]
    return "Btns[" + (",".join(pressed) if pressed else "-") + "]"

def _learn_first_button(js) -> int:
    prev = set()
    while True:
        pygame.event.pump()
        current = {i for i in range(js.get_numbuttons()) if js.get_button(i)}
        newly = current - prev
        if newly:
            return sorted(list(newly))[0]
        prev = current
        time.sleep(0.01)

def start_ps5_estop_listener(
    estop: EStopGate,
    engage_button: int | None = None,   # default: SDL mapping Cross(X)=0
    release_button: int | None = None,  # default: SDL mapping Triangle(△)=3
    deadzone: float = 0.08,
    print_rate: float = 0.25,
    interactive_bind: bool = False
) -> None:
    def _controller_loop():
        if pygame is None:
            Debug.stamp("PS5", "pygame not available; only stdin fallback active.")
            return
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            Debug.stamp("PS5", f"pygame init failed: {e}")
            return

        try:
            n = pygame.joystick.get_count()
            if n == 0:
                Debug.stamp("PS5", "No joystick detected. Fallback: stdin 'e'/'r'.")
                return
            js = pygame.joystick.Joystick(0)
            js.init()
        except Exception as e:
            Debug.stamp("PS5", f"Joystick init failed: {e}")
            return

        name = js.get_name()
        Debug.stamp("PS5", f"Detected controller: {name}")
        Debug.stamp("PS5", f"Buttons={js.get_numbuttons()} Axes={js.get_numaxes()} Hats={js.get_numhats()}")

        if engage_button is None: eb = 0
        else: eb = engage_button
        if release_button is None: rb = 3
        else: rb = release_button

        if interactive_bind:
            Debug.stamp("PS5", "Interactive bind: Press ENGAGE (X) once…")
            eb = _learn_first_button(js)
            Debug.stamp("PS5", f"ENGAGE bound to button {eb}")
            time.sleep(0.2)
            Debug.stamp("PS5", "Interactive bind: Press RELEASE (△) once…")
            rb = _learn_first_button(js)
            Debug.stamp("PS5", f"RELEASE bound to button {rb}")

        Debug.stamp("PS5", f"Using buttons: ENGAGE={eb} (X), RELEASE={rb} (△)")
        axes_ok = _rate_limit(print_rate)
        prev_buttons = set()
        prev_hat = (0, 0)

        while True:
            try:
                pygame.event.pump()
                current = {i for i in range(js.get_numbuttons()) if js.get_button(i)}
                for b in sorted(current - prev_buttons):
                    Debug.stamp("PS5", f"Button DOWN: {b}")
                    if b == eb:
                        estop.engage()
                    if b == rb:
                        estop.release()
                for b in sorted(prev_buttons - current):
                    Debug.stamp("PS5", f"Button UP:   {b}")
                prev_buttons = current
                if axes_ok():
                    Debug.stamp("PS5", f"{_pretty_axes(js, deadzone)}  {_pretty_buttons(js)}")
                if js.get_numhats() > 0:
                    hat = js.get_hat(0)
                    if hat != prev_hat:
                        Debug.stamp("PS5", f"Hat changed: {hat}")
                        prev_hat = hat
            except Exception as e:
                Debug.stamp("PS5", f"Loop error: {e}", every_sec=1.0)
            time.sleep(0.01)
            
    

    def _stdin_loop():
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old = None
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            Debug.stamp("PS5", "Stdin active: 'e' ENGAGE, 'r' RELEASE.")
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == 'e':
                    estop.engage()
                elif ch == 'r':
                    estop.release()
        except Exception as e:
            Debug.stamp("PS5", f"stdin err: {e}")
        finally:
            try:
                if old:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    threading.Thread(target=_controller_loop, daemon=True).start()
    threading.Thread(target=_stdin_loop,    daemon=True).start()
def _clip_to_limits(robot, q):
    # If robot has joint limits, clamp; else return as-is
    try:
        qlim = getattr(robot, "qlim", None)
        if qlim is None:
            return q
        lo = np.array(qlim[0]).flatten()
        hi = np.array(qlim[1]).flatten()
        return np.minimum(np.maximum(q, lo), hi)
    except Exception:
        return q

def start_ps5_joint_driver(units_by_key,  # dict: key -> RobotUnit
                           deadzone: float = 0.12,
                           max_speed_rad_s: float = 0.6,
                           axis_map: dict | None = None):
    """
    Reads PS5 sticks and publishes joint targets for the GUI's active robot.
    Publishes ONLY when bus.is_enabled() and not bus.is_estop().
    axis_map: maps controller axes to joint indexes, e.g. {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
      Defaults: LX->J1, LY->J2, RX->J3, RY->J4, L2->J5, R2->J6
    """
    if axis_map is None:
        axis_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

    def _loop():
        if pygame is None:
            return
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                return
            js = pygame.joystick.Joystick(0)
            js.init()
        except Exception:
            return

        last_t = time.time()
        while True:
            time.sleep(0.015)
            try:
                pygame.event.pump()
            except Exception:
                continue

            # skip if not allowed
            if not bus.is_enabled() or bus.is_estop():
                last_t = time.time()
                continue

            active = bus.get_active_robot()
            if not active or active not in units_by_key:
                continue

            unit = units_by_key[active]
            dof = unit.robot.n

            # current target: prefer bus target, else current robot.q
            q_cur = bus.get_q_for(active)
            if q_cur is None:
                q_cur = unit.robot.q
            q = np.array(q_cur, dtype=float).copy()

            # time step
            now = time.time()
            dt = max(0.005, min(0.05, now - last_t))
            last_t = now

            # build per-joint velocities from axes
            dq = np.zeros(dof, dtype=float)
            for ax, j in axis_map.items():
                if j >= dof:
                    continue
                try:
                    val = float(js.get_axis(ax))
                except Exception:
                    val = 0.0
                # normalize triggers (PS5 L2/R2 are usually in [-1..+1])
                if abs(val) < deadzone:
                    val = 0.0
                dq[j] += val * max_speed_rad_s

            if not np.any(dq):
                continue

            q_new = q + dq * dt
            q_new = _clip_to_limits(unit.robot, q_new)

            # publish new target; follower thread will apply it
            bus.publish_q(active, q_new.tolist())

    threading.Thread(target=_loop, daemon=True).start()


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
        while True:
            ESTOP.wait_if_engaged()
            with self._lock:
                run = self.running
            if run and not ESTOP.is_engaged():
                for obj in list(self.carried):
                    T = SE3(obj.T)
                    obj.T = (SE3(T) * SE3(0, 0.002, 0)).A
            Debug.stamp("CONVEYOR", f"running={run}, carried={len(self.carried)}", every_sec=1.0)
            time.sleep(0.02)


KR6_KEY = "KUKA_KR6"
LBR_KEY = "KUKA_LBR"
LWR_KEY = "KUKA_LWR"
UR3_KEY = "UR3"

class Environment:
    def __init__(self):
        self._dt = 0.02
        self._render_lock = threading.Lock()
        self._running = True

        
        self._unit_map_sink = []   

        self.built = 0
        self.env = swift.Swift()
        self.env._parent = self
        self.env.launch(realtime=True)
        self.env.set_camera_pose([2, 2, 2], [0, 0, -pi / 4])

      
        start_ps5_estop_listener(ESTOP, engage_button=None, release_button=None,
                                 deadzone=0.08, print_rate=0.25, interactive_bind=False)
        threading.Thread(target=_mirror_bus_estop, daemon=True).start()

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
        threading.Thread(target=self._render_loop, daemon=True).start()

    def _render_loop(self):
        ticks = 0
        t0 = time.time()
        while self._running:
            ESTOP.wait_if_engaged()
            with self._render_lock:
                self.env.step(self._dt)
            ticks += 1
            if ticks % int(max(1, 1.0 / self._dt)) == 0:
                elapsed = time.time() - t0
                Debug.stamp("RENDER", f"alive, dt={self._dt:.3f}, ticks={ticks}, elapsed={elapsed:.1f}s", every_sec=1.0)
            time.sleep(self._dt)

    def safe_add(self, obj):
        with self._render_lock:
            self.env.add(obj)

    def _start_override_follower(self):
        unit_map = self._unit_map_sink  

        def _worker():
          
            while not unit_map:
                time.sleep(0.01)

            while True:
                ESTOP.wait_if_engaged()
                allow = (not bus.should_pause_conveyor()) and (not ESTOP.is_engaged())
                self.conveyor.set_running(allow)
                Debug.stamp("OVERRIDE", f"conveyor_allowed={allow}", every_sec=1.0)

                for key, unit in unit_map:
                    if bus.is_enabled_for(key):
                        q_over = bus.get_q_for(key)
                        g = bus.get_gripper_closed_for(key)
                        Debug.stamp("OVERRIDE", f"{key}: q_over={'yes' if q_over is not None else 'no'}, grip={g}", every_sec=0.5)
                        if q_over is not None:
                            unit.robot.q = q_over
                            unit.gripper.update(unit.robot.fkine(unit.robot.q))
                        if g is not None and isinstance(unit.gripper, Gripper):
                            unit.gripper.actuate("close" if g else "open")
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

       
        self.kr6 = RobotUnit(KR6(), self.env, SE3(0.7, 1.0, self.ground_height), override_key=KR6_KEY, collision_zones=kr6_zones)
        self.lbr = RobotUnit(LBR(), self.env, SE3(0.7, 0.0, self.ground_height), override_key=LBR_KEY, collision_zones=lbr_zones)
        self.lwr = RobotUnit(LWR(), self.env, SE3(0.5, -1.0, self.ground_height), override_key=LWR_KEY, collision_zones=lwr_zones)

       
        self.ur3_stand = 0.3
        self.env.add(Cuboid(scale=[0.2, 0.2, self.ur3_stand],
                            pose=SE3(-0.7, 0, self.ur3_stand/2 + self.ground_height),
                            color=[0.5, 0.3, 0.3]))
        self.ur3 = RobotUnit(
            UR3(), self.env,
            SE3(-0.7, 0.0, self.ground_height + self.ur3_stand),
            q_init=[pi/2, -pi/2, 0, 0, pi/2, 0],
            override_key=UR3_KEY,
            collision_zones=ur3_zones
        )
        self.built += 1

        # Register to follower map
        self._unit_map_sink.extend([
            (KR6_KEY, self.kr6),
            (LBR_KEY, self.lbr),
            (LWR_KEY, self.lwr),
            (UR3_KEY, self.ur3),
        ])
        
        
        units_by_key = {
            KR6_KEY: self.kr6,
            LBR_KEY: self.lbr,
            LWR_KEY: self.lwr,
            UR3_KEY: self.ur3,
        }
        start_ps5_joint_driver(units_by_key,
                               deadzone=0.12,
                               max_speed_rad_s=0.8,   # tweak feel here
                               axis_map={0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
        Debug.stamp("OVERRIDE", f"unit_map ready: {[k for k,_ in self._unit_map_sink]}")

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
        Debug.stamp("WORLD", "Object loaded")
        return obj

    def translate_object(self, target_obj, target_pose):
        if isinstance(target_obj, tuple):
            _, target_obj = target_obj
        initial_pose = SE3(target_obj.T)
        Debug.stamp("WORLD", f"Translating brick to {target_pose.t}")
        for alpha in np.linspace(0, 1, 50):
            ESTOP.wait_if_engaged()
            target_obj.T = initial_pose.interp(target_pose * SE3.Ry(pi), alpha).A
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
        Debug.stamp("WORLD", f"Translating {len(target_objs)} objects by {translation}")
        for alpha in np.linspace(0, 1, steps):
            ESTOP.wait_if_engaged()
            for i, obj_entry in enumerate(target_objs):
                obj = obj_entry[1] if isinstance(obj_entry, tuple) else obj_entry
                start = initial_poses[i]
                obj.T = start.interp(dT * start, alpha).A
            time.sleep(0.02)

    def run_mission(self):
        self.KR6_place_pos = SE3(0.0, 1.0, 0.32) * SE3.RPY(0, pi, 0)
        self.lwr_place_pos = SE3(0.8, -1, 0.2) * SE3.RPY(0, pi, pi/2)

        self.load_box()
        self.load_object(0)
        self.load_object(1)
        Debug.stamp("MISSION", "Welding Factory Beginning")

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
    def __init__(self, robot, env: swift.Swift, base_pose: SE3, q_init=None, override_key=None, collision_zones=None):
        self.robot = robot
        self.env = env
        self.environment = getattr(env, "_parent", env)
        self.robot.base = base_pose
        self.robot.q = np.zeros(self.robot.n) if q_init is None else q_init


        self.override_key = override_key or self.robot.name

  
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
        if bus.is_enabled_for(self.override_key):
            Debug.stamp(self.override_key, "Override active — weld aborted")
            return

        hover_start = weld_begin * SE3(0.05, 0, 0.0)
        ik_hover = self.robot.ikine_LM(hover_start, q0=self.robot.q, joint_limits=False)
        if not ik_hover.success:
            Debug.stamp(self.override_key, "weld hover IK fail")
            return
        for q in jtraj(self.robot.q, ik_hover.q, 50).q:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            self._step(q)

        ik_start = self.robot.ikine_LM(weld_begin, q0=self.robot.q, joint_limits=False)
        if not ik_start.success:
            Debug.stamp(self.override_key, "weld start IK fail")
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
                Debug.stamp(self.override_key, "weld path IK skip point")
                continue
            for q in jtraj(self.robot.q, ik.q, 5).q:
                if self._maybe_abort_override(): return
                ESTOP.wait_if_engaged()
                self._step(q, weld=True)

    def home(self, steps=50):
        if bus.is_enabled_for(self.override_key):
            Debug.stamp(self.override_key, "Override active — home aborted")
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
        Debug.stamp(self.override_key, "Returned to home position")

    def pick_and_place(self, pick_pose: SE3, place_pose: SE3, steps=50, brick_idx=None):
        if bus.is_enabled_for(self.override_key):
            Debug.stamp(self.override_key, "Override active — pick/place aborted")
            return
        Debug.stamp(self.override_key, "PICK start")
        if isinstance(self.gripper, Gripper):
            self.gripper.actuate("open")
        self.move_to(pick_pose, steps)
        if isinstance(self.gripper, Gripper):
            self.gripper.actuate("close")
        Debug.stamp(self.override_key, "PLACE start")
        self.move_to(place_pose, steps, brick_idx)
        Debug.stamp(self.override_key, "pick/place done")

    def move_to(self, target_pose: SE3, steps=50, brick_idx=None):
        if bus.is_enabled_for(self.override_key):
            Debug.stamp(self.override_key, "Override active — move aborted")
            return
        Debug.stamp(self.override_key, f"move_to steps={steps}, carry={brick_idx}")
        hover_pose = target_pose * SE3(0, 0, -0.14)
        ok_hover, traj_hover = self._ik_traj(hover_pose, steps)
        if not ok_hover:
            Debug.stamp(self.override_key, "hover IK fail")
            return

        ok_goal, traj_goal = self._ik_traj(target_pose, steps, q0=traj_hover[-1])
        if not ok_goal:
            Debug.stamp(self.override_key, "goal IK fail")
            return

        self._execute_trajectory(traj_hover, brick_idx)
        self._execute_trajectory(traj_goal, brick_idx)

    def _maybe_abort_override(self) -> bool:
        if bus.is_enabled_for(self.override_key):
            Debug.stamp(self.override_key, "Manual override engaged — aborting auto motion.")
            return True
        return False

    def _ik_traj(self, target_pose: SE3, steps=50, q0=None):
        if q0 is None:
            q0 = self.robot.q.copy()
        target_corrected = target_pose * SE3(0, 0, -self.gripper_carry_offset) * SE3.RPY(0, 0, pi/2)
        ik = self.robot.ikine_LM(target_corrected, q0=q0, joint_limits=True)
        if not ik.success:
            Debug.stamp(self.override_key, "IK failed", every_sec=0.5)
            return False, []
        traj = jtraj(q0, ik.q, steps).q
        Debug.stamp(self.override_key, f"IK ok, traj_steps={len(traj)}")
        return True, traj

    def _execute_trajectory(self, traj, brick_idx):
        Debug.stamp(self.override_key, f"exec traj len={len(traj)}, carry={brick_idx}")
        for q in traj:
            if self._maybe_abort_override(): return
            ESTOP.wait_if_engaged()
            if any(det.check_pose(self.robot, q) for det in self.detectors):
                Debug.stamp("COLLISION", f"{self.override_key}: collision flagged — halting")
                return
            self._step(q, carry_idx=brick_idx)

    def _step(self, q, carry_idx=None, weld=False):
        ESTOP.wait_if_engaged()
        self.robot.q = q
        ee_pose = self.robot.fkine(q)


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

        if weld and hasattr(self.gripper, "weld"):
            self.gripper.weld(ee_pose)
        else:
            self.gripper.update(ee_pose)

        time.sleep(0.01)


def main():
    env = Environment()
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()

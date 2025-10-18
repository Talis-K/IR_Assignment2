#!/usr/bin/env python3
# Swift-focused Robot GUI (KR6 + LBR + LWR + UR3) — UR3 listed last
# Uses exactly the same imports/paths/classes as your working script.

import time
import threading
from typing import Dict, List, Optional
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import swift
from spatialmath import SE3



from spatialgeometry import Cylinder


import roboticstoolbox as rtb
from roboticstoolbox import jtraj


def _try_imports() -> Dict[str, object]:
    robots = {}

    # KUKA KR6
    from KUKA_KR6.KR6 import KR6_Robot as KR6
    robots["KUKA_KR6"] = KR6()
  

    # KUKA LBR (Load)

    from KUKA_LBR.LBR import Load as LBR
    robots["KUKA_LBR"] = LBR()
 

    # KUKA LWR (Load)
    from Kuka_LWR.LWR import Load as LWR
    robots["KUKA_LWR"] = LWR()
 

    # UR3 (last)
  
    from ir_support import UR3
    robots["UR3"] = UR3()


    # Keep only the ones that succeeded, and enforce selector order (UR3 last)
    order = ["KUKA_KR6", "KUKA_LBR", "KUKA_LWR", "UR3"]
    robots = {k: robots[k] for k in order if k in robots}
    return robots


# -------- Uniform adapter so GUI can treat robots the same --------
class RobotAdapter:
    def __init__(self, name: str, robot):
        self.name = name
        self.robot = robot
        self.dof = int(getattr(robot, "n", 6) or 6)

        # current q
        if hasattr(robot, "q") and isinstance(robot.q, (list, np.ndarray)):
            q0 = np.array(robot.q, dtype=float).reshape(-1)
            self.q = q0 if q0.size == self.dof else np.zeros(self.dof)
        else:
            self.q = np.zeros(self.dof)

        # limits if provided
        if hasattr(robot, "qlim") and isinstance(robot.qlim, np.ndarray) and robot.qlim.shape[0] >= self.dof:
            self.qlim = robot.qlim[:self.dof]
        else:
            self.qlim = np.vstack([-np.pi*np.ones(self.dof), np.pi*np.ones(self.dof)]).T

    def set_q(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).reshape(-1)
        q = np.clip(q, self.qlim[:, 0], self.qlim[:, 1])
        self.q = q
        if hasattr(self.robot, "q"):
            try:
                self.robot.q = q.tolist() if not isinstance(self.robot.q, np.ndarray) else q
            except Exception:
                pass

    def fkine(self):
        if hasattr(self.robot, "fkine"):
            try:
                return self.robot.fkine(self.q)
            except Exception:
                return None
        return None


# -------- Swift stepping thread --------
class SwiftBridge:
    def __init__(self):
        self.env = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def launch(self):
        if self.env is None:
            self.env = swift.Swift()
            # Use the common kwarg name `realtime`, but your env can also accept realTime=True
            self.env.launch(realtime=True)
        with self.lock:
            self.running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
            try:
                if self.env is not None:
                    self.env.step()
            except Exception:
                pass
            time.sleep(0.02)


# -------- Main App --------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Swift Robot GUI (KR6 + LBR + LWR + UR3)")
        self.geometry("960x700")
        self.minsize(900, 620)

        base_robots = _try_imports()

        # Wrap into adapters
        self.robots: Dict[str, RobotAdapter] = {k: RobotAdapter(k, v) for k, v in base_robots.items()}

        self.active_key = list(self.robots.keys())[0]
        self.speed_scale = tk.DoubleVar(value=1.0)
        self.show_blast = tk.BooleanVar(value=False)

        self.swift: Optional[SwiftBridge] = None
        self._blast = None  # Cylinder

        self._build_header()
        self._build_joint_panel()
        self._build_footer()

        self._launch_swift_and_add()
        self._rebuild_sliders()
        self.after(40, self._tick)

        # Let the user know if some robots failed (without killing the GUI)
        failed = [k for k in ["KUKA_KR6", "KUKA_LBR", "KUKA_LWR", "UR3"] if k not in self.robots]
        if failed:
            messagebox.showwarning("Some robots not loaded",
                                   "Loaded: " + ", ".join(self.robots.keys()) +
                                   "\nMissing: " + ", ".join(failed))

    # ---- UI ----
    def _build_header(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(fr, text="Robot:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.robot_combo = ttk.Combobox(fr, state="readonly", values=list(self.robots.keys()), width=24)
        self.robot_combo.set(self.active_key)
        self.robot_combo.pack(side=tk.LEFT, padx=(8, 16))
        self.robot_combo.bind("<<ComboboxSelected>>", self._on_robot_change)

        ttk.Label(fr, text="Move Speed (×):").pack(side=tk.LEFT)
        ttk.Scale(fr, from_=0.1, to=2.0, variable=self.speed_scale, orient=tk.HORIZONTAL, length=180).pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(fr, text="Show 1 m blast", variable=self.show_blast).pack(side=tk.LEFT, padx=12)

        ttk.Button(fr, text="Home (zeros)", command=self._home_zeros).pack(side=tk.LEFT, padx=8)

    def _build_joint_panel(self):
        self.joint = ttk.LabelFrame(self, text="Joint Controls", padding=(10, 8))
        self.joint.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 10), pady=(8, 10))
        self.slider_vars: List[tk.DoubleVar] = []
        self.value_labels: List[ttk.Label] = []

    def _build_footer(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.BOTTOM, fill=tk.X)
        self.pose_label = ttk.Label(fr, text="TCP: —")
        self.pose_label.pack(side=tk.LEFT)

    # ---- Callbacks ----
    def _on_robot_change(self, _evt=None):
        self.active_key = self.robot_combo.get()
        self._rebuild_sliders()
        # No need to re-add to Swift; we add all robots once.

    def _home_zeros(self):
        adapter = self.robots[self.active_key]
        adapter.set_q(np.zeros(adapter.dof))
        self._sync_from_robot()

    # ---- Sliders ----
    def _rebuild_sliders(self):
        for child in list(self.joint.winfo_children()):
            child.destroy()
        self.slider_vars.clear()
        self.value_labels.clear()

        adapter = self.robots[self.active_key]
        for j in range(adapter.dof):
            row = ttk.Frame(self.joint)
            row.pack(fill=tk.X, pady=4)

            ttk.Label(row, text=f"J{j+1}", width=5, anchor="e").pack(side=tk.LEFT, padx=(0, 8))

            v = tk.DoubleVar(value=float(adapter.q[j]))
            self.slider_vars.append(v)

            qmin, qmax = adapter.qlim[j, 0], adapter.qlim[j, 1]
            rng = max(abs(qmin), abs(qmax))
            scale = ttk.Scale(row, from_=-rng, to=+rng, orient=tk.HORIZONTAL, variable=v, length=560)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            lbl = ttk.Label(row, text=f"{float(v.get()):+.3f} rad", width=14)
            lbl.pack(side=tk.LEFT, padx=(8, 0))
            self.value_labels.append(lbl)

            def _mk_cb(_var=v, _lbl=lbl):
                def _update(_evt=None):
                    _lbl.config(text=f"{float(_var.get()):+.3f} rad")
                return _update
            scale.bind("<B1-Motion>", _mk_cb())
            scale.bind("<ButtonRelease-1>", _mk_cb())

        ttk.Label(self.joint, text="Tip: drag sliders or use ←/→ on a focused slider for fine steps.",
                  foreground="#666").pack(anchor="w", pady=(6, 0))

    def _sync_from_robot(self):
        adapter = self.robots[self.active_key]
        for i, v in enumerate(self.slider_vars):
            if i < adapter.q.size:
                v.set(float(adapter.q[i]))

    # ---- Swift ----
    def _launch_swift_and_add(self):
        try:
            self.swift = SwiftBridge()
            self.swift.launch()
        except Exception as e:
            messagebox.showerror("Swift", f"Failed to launch Swift: {e}")
            return

        # Add all robots to Swift once
        for key, adapter in self.robots.items():
            try:
                # For RTB / custom models: both env.add(adapter.robot) and robot.add_to_env(env) are common.
                if hasattr(adapter.robot, "add_to_env"):
                    adapter.robot.add_to_env(self.swift.env)
                else:
                    self.swift.env.add(adapter.robot)
            except Exception as e:
                messagebox.showwarning("Swift add()", f"Failed to add {key} to Swift: {e}")

    # ---- Main loop ----
    def _tick(self):
        adapter = self.robots[self.active_key]

        # Smoothly push sliders -> robot
        q_target = np.array([float(v.get()) for v in self.slider_vars], dtype=float)
        cur = adapter.q.copy()
        alpha = max(0.1, min(1.0, float(self.speed_scale.get())))
        new = cur + (q_target - cur) * alpha
        adapter.set_q(new)

        # Pose readout
        T = adapter.fkine()
        if T is not None:
            try:
                pos = np.array(T.t).reshape(3)
                self.pose_label.config(text=f"TCP: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m")
            except Exception:
                pass

        # Optional 1 m "blast" cylinder along tool Z
        if self.show_blast.get() and self.swift and Cylinder and SE3 and T is not None:
            try:
                if self._blast is None:
                    self._blast = Cylinder(radius=0.01, length=1.0, pose=T * SE3(0, 0, 0.5), color="red")
                    self.swift.env.add(self._blast)
                else:
                    self._blast.T = T * SE3(0, 0, 0.5)
            except Exception:
                pass
        else:
            if self._blast is not None and self.swift:
                try:
                    self.swift.env.remove(self._blast)
                except Exception:
                    pass
                self._blast = None

        self.after(35, self._tick)


def launch():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    launch()

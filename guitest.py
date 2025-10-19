
import time
from typing import Dict, List
import numpy as np
import tkinter as tk
from tkinter import ttk

from override import bus

DOF_MAP = {
    "KUKA_KR6": 6,
    "KUKA_LBR": 7,
    "KUKA_LWR": 7,
    "UR3": 6,
}
ROBOT_ORDER = ["KUKA_KR6", "KUKA_LBR", "KUKA_LWR", "UR3"]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Manual Override — Joint Controller")
        self.geometry("720x560")
        self.minsize(680, 520)

        self.active_key = ROBOT_ORDER[0]
        bus.set_active_robot(self.active_key)

        self.override_enabled = tk.BooleanVar(value=False)
        self.speed_scale = tk.DoubleVar(value=1.0)

        self._build_header()
        self._build_joint_panel()
        self._build_footer()

        self._rebuild_sliders()
        self.after(40, self._tick)


    def _build_header(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(fr, text="Robot:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.robot_combo = ttk.Combobox(fr, state="readonly", values=ROBOT_ORDER, width=22)
        self.robot_combo.set(self.active_key)
        self.robot_combo.pack(side=tk.LEFT, padx=(8, 16))
        self.robot_combo.bind("<<ComboboxSelected>>", self._on_robot_change)

        ttk.Checkbutton(fr, text="Manual Override", variable=self.override_enabled,
                        command=lambda: bus.set_enabled(self.override_enabled.get())).pack(side=tk.LEFT, padx=10)

        ttk.Label(fr, text="Move Speed (×):").pack(side=tk.LEFT)
        ttk.Scale(fr, from_=0.1, to=2.0, variable=self.speed_scale, orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT, padx=6)

    def _build_joint_panel(self):
        self.joint = ttk.LabelFrame(self, text="Joint Controls", padding=(10, 8))
        self.joint.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 10), pady=(8, 10))
        self.slider_vars: List[tk.DoubleVar] = []
        self.value_labels: List[ttk.Label] = []

    def _build_footer(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.BOTTOM, fill=tk.X)
        self.pose_label = ttk.Label(fr, text="(Use the main Swift window to view the robot)")
        self.pose_label.pack(side=tk.LEFT)

    # callbacks
    def _on_robot_change(self, _evt=None):
        self.active_key = self.robot_combo.get()
        bus.set_active_robot(self.active_key)
        self._rebuild_sliders()

    # sliders
    def _rebuild_sliders(self):
        for c in list(self.joint.winfo_children()):
            c.destroy()
        self.slider_vars.clear()
        self.value_labels.clear()

        dof = DOF_MAP[self.active_key]
        # default to +/- pi range since GUI is decoupled from model limits
        for j in range(dof):
            row = ttk.Frame(self.joint)
            row.pack(fill=tk.X, pady=4)

            ttk.Label(row, text=f"J{j+1}", width=5, anchor="e").pack(side=tk.LEFT, padx=(0, 8))
            v = tk.DoubleVar(value=0.0)
            self.slider_vars.append(v)

            scale = ttk.Scale(row, from_=-np.pi, to=+np.pi, orient=tk.HORIZONTAL, variable=v, length=520)
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

    # main loop
    def _tick(self):
        # Build q from sliders
        q_target = np.array([float(v.get()) for v in self.slider_vars], dtype=float)


        speed = max(0.1, min(1.0, float(self.speed_scale.get())))
        # Keep a tiny local memory to smooth toward the last published target
        if not hasattr(self, "_last_q"):
            self._last_q = q_target.copy()
        self._last_q = self._last_q + (q_target - self._last_q) * speed

        # Publish when override is ON
        if self.override_enabled.get():
            bus.publish_q(self.active_key, self._last_q.tolist())

        self.after(35, self._tick)


def launch_gui():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()

import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import List

from override import bus  

DOF_MAP = {
    "KUKA_KR6": 6,
    "KUKA_LBR": 7,
    "KUKA_LWR": 7,
    "UR3": 6,
}
ROBOT_ORDER = ["KUKA_KR6", "KUKA_LBR", "KUKA_LWR", "UR3"]

# UI colours
ESTOP_BG_IDLE  = "#f7f7f7"
ESTOP_BG_ARMED = "#ffefef"
ESTOP_RED      = "#c62828"
ESTOP_GREEN    = "#2e7d32"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Manual Override — Joint & Gripper Controller")
        self.geometry("820x680")
        self.minsize(760, 600)

        # Active robot selection
        self.active_key = ROBOT_ORDER[0]
        bus.set_active_robot(self.active_key)

        # Manual override master switch
        self.override_enabled = tk.BooleanVar(value=False)

        # Motion smoothing factor (0.1..2.0)
        self.speed_scale = tk.DoubleVar(value=1.0)

        # Software E-Stop
        try:
            current_estop = bool(bus.is_estop())
        except Exception:
            current_estop = False
        self.estop_enabled = tk.BooleanVar(value=current_estop)

        # Gripper state (per active robot; bus is final source of truth)
        self.gripper_closed = tk.BooleanVar(value=False)

        # Build UI
        self._build_header()
        self._build_estop_banner()
        self._build_joint_panel()
        self._build_gripper_panel()
        self._build_footer()

        self._rebuild_sliders()
        self._bind_hotkeys()

        self.after(40, self._tick)

    # ---------- UI BUILD ----------
    def _build_header(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(fr, text="Robot:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.robot_combo = ttk.Combobox(fr, state="readonly", values=ROBOT_ORDER, width=22)
        self.robot_combo.set(self.active_key)
        self.robot_combo.pack(side=tk.LEFT, padx=(8, 16))
        self.robot_combo.bind("<<ComboboxSelected>>", self._on_robot_change)

        self.override_chk = ttk.Checkbutton(
            fr, text="Manual Override",
            variable=self.override_enabled,
            command=self._on_override_toggle
        )
        self.override_chk.pack(side=tk.LEFT, padx=10)

        ttk.Label(fr, text="Move Speed (×):").pack(side=tk.LEFT)
        ttk.Scale(fr, from_=0.1, to=2.0, variable=self.speed_scale,
                  orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT, padx=6)

    def _build_estop_banner(self):
        self.estop_frame = tk.Frame(self, bd=1, relief=tk.SOLID, bg=ESTOP_BG_IDLE)
        self.estop_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 0))

        inner = tk.Frame(self.estop_frame, bg=self.estop_frame["bg"])
        inner.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.estop_light = tk.Canvas(inner, width=18, height=18, highlightthickness=0, bg=self.estop_frame["bg"])
        self.estop_light.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_light(self.estop_light, ESTOP_GREEN)

        txt = tk.Label(inner, text="EMERGENCY STOP", font=("Segoe UI", 14, "bold"),
                       fg=ESTOP_RED, bg=self.estop_frame["bg"])
        txt.pack(side=tk.LEFT)

        self.estop_status = tk.Label(inner, text="Status: RELEASED", font=("Segoe UI", 11),
                                     fg="#333", bg=self.estop_frame["bg"])
        self.estop_status.pack(side=tk.LEFT, padx=16)

        self.btn_estop = tk.Button(inner, text="ENGAGE (E)", font=("Segoe UI", 11, "bold"),
                                   bg=ESTOP_RED, fg="white", activebackground="#b71c1c",
                                   command=self._engage_estop)
        self.btn_estop.pack(side=tk.RIGHT, padx=6)

        self.btn_release = tk.Button(inner, text="RELEASE (R)", font=("Segoe UI", 11, "bold"),
                                     bg=ESTOP_GREEN, fg="white", activebackground="#1b5e20",
                                     command=self._release_estop)
        self.btn_release.pack(side=tk.RIGHT, padx=6)

        self._refresh_estop_banner(initial=True)

    def _build_joint_panel(self):
        self.joint = ttk.LabelFrame(self, text="Joint Controls", padding=(10, 8))
        self.joint.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                        padx=(10, 10), pady=(8, 10))
        self.slider_vars: List[tk.DoubleVar] = []
        self.value_labels: List[ttk.Label] = []

    def _build_gripper_panel(self):
        fr = ttk.LabelFrame(self, text="Gripper", padding=(10, 8))
        fr.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        self.grip_status = ttk.Label(fr, text="State: OPEN", width=18)
        self.grip_status.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_open = ttk.Button(fr, text="Open (O)", command=self._grip_open)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_close = ttk.Button(fr, text="Close (C)", command=self._grip_close)
        self.btn_close.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_toggle = ttk.Button(fr, text="Toggle (G)", command=self._grip_toggle)
        self.btn_toggle.pack(side=tk.LEFT, padx=(12, 0))

        self.pause_conveyor_chk = tk.BooleanVar(value=True)
        self.chk_pause = ttk.Checkbutton(
            fr,
            text="Pause conveyor while overriding",
            variable=self.pause_conveyor_chk,
            command=lambda: bus.set_conveyor_pause(self.pause_conveyor_chk.get())
        )
        self.chk_pause.pack(side=tk.RIGHT)

    def _build_footer(self):
        fr = ttk.Frame(self, padding=(10, 8))
        fr.pack(side=tk.BOTTOM, fill=tk.X)
        self.pose_label = ttk.Label(fr, text="Hotkeys: E/R for E-Stop, G toggle gripper, O open, C close.")
        self.pose_label.pack(side=tk.LEFT)

    # ---------- HOTKEYS ----------
    def _bind_hotkeys(self):
        self.bind("<space>", self._toggle_estop_key)
        self.bind("<e>", lambda e: self._engage_estop())
        self.bind("<E>", lambda e: self._engage_estop())
        self.bind("<r>", lambda e: self._release_estop())
        self.bind("<R>", lambda e: self._release_estop())

        # Gripper hotkeys
        self.bind("<o>", lambda e: self._grip_open())
        self.bind("<O>", lambda e: self._grip_open())
        self.bind("<c>", lambda e: self._grip_close())
        self.bind("<C>", lambda e: self._grip_close())
        self.bind("<g>", lambda e: self._grip_toggle())
        self.bind("<G>", lambda e: self._grip_toggle())

    # ---------- CALLBACKS ----------
    def _on_robot_change(self, _evt=None):
        self.active_key = self.robot_combo.get()
        bus.set_active_robot(self.active_key)
        # adopt last known gripper state for this robot if any
        g = bus.get_gripper_closed_for(self.active_key)
        self.gripper_closed.set(bool(g) if g is not None else False)
        self._update_gripper_status()
        self._rebuild_sliders()

    def _on_override_toggle(self):
        # Disable override if E-Stop is engaged
        if self.estop_enabled.get():
            self.override_enabled.set(False)
        bus.set_enabled(self.override_enabled.get())
        bus.set_conveyor_pause(self.pause_conveyor_chk.get())

    # ---------- SLIDERS ----------
    def _rebuild_sliders(self):
        for c in list(self.joint.winfo_children()):
            c.destroy()
        self.slider_vars.clear()
        self.value_labels.clear()

        dof = DOF_MAP[self.active_key]
        for j in range(dof):
            row = ttk.Frame(self.joint)
            row.pack(fill=tk.X, pady=4)

            ttk.Label(row, text=f"J{j+1}", width=5, anchor="e").pack(side=tk.LEFT, padx=(0, 8))
            v = tk.DoubleVar(value=0.0)
            self.slider_vars.append(v)

            scale = ttk.Scale(row, from_=-np.pi, to=+np.pi, orient=tk.HORIZONTAL, variable=v, length=560)
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

    # ---------- E-STOP ----------
    def _engage_estop(self):
        self.estop_enabled.set(True)
        bus.set_estop(True)
        self.override_enabled.set(False)
        bus.set_enabled(False)
        self._refresh_estop_banner()

    def _release_estop(self):
        self.estop_enabled.set(False)
        bus.set_estop(False)
        self._refresh_estop_banner()

    def _toggle_estop_key(self, _evt=None):
        if self.estop_enabled.get():
            self._release_estop()
        else:
            self._engage_estop()

    def _refresh_estop_banner(self, initial: bool=False):
        engaged = self.estop_enabled.get() or bool(bus.is_estop())
        self.estop_frame.configure(bg=ESTOP_BG_ARMED if engaged else ESTOP_BG_IDLE)
        for child in self.estop_frame.winfo_children():
            child.configure(bg=self.estop_frame["bg"])
            for g in child.winfo_children():
                try:
                    g.configure(bg=self.estop_frame["bg"])
                except Exception:
                    pass
        self._draw_light(self.estop_light, ESTOP_RED if engaged else ESTOP_GREEN)
        self.estop_status.configure(
            text="Status: ENGAGED" if engaged else "Status: RELEASED",
            fg=ESTOP_RED if engaged else "#333"
        )
        self.btn_estop.configure(state=tk.DISABLED if engaged else tk.NORMAL)
        self.btn_release.configure(state=tk.NORMAL if engaged else tk.DISABLED)

        if engaged:
            self.override_enabled.set(False)
            bus.set_enabled(False)

    def _draw_light(self, canvas: tk.Canvas, color: str):
        canvas.delete("all")
        canvas.create_oval(2, 2, 16, 16, outline="#222", fill=color)

    # ---------- GRIPPER ----------
    def _update_gripper_status(self):
        closed = self.gripper_closed.get()
        self.grip_status.config(text=("State: CLOSED" if closed else "State: OPEN"))

    def _grip_open(self):
        self.gripper_closed.set(False)
        self._update_gripper_status()
        bus.publish_gripper_closed(self.active_key, False)

    def _grip_close(self):
        self.gripper_closed.set(True)
        self._update_gripper_status()
        bus.publish_gripper_closed(self.active_key, True)

    def _grip_toggle(self):
        self.gripper_closed.set(not self.gripper_closed.get())
        self._update_gripper_status()
        bus.publish_gripper_closed(self.active_key, self.gripper_closed.get())

    # ---------- MAIN LOOP ----------
    def _tick(self):
        # Keep GUI in sync with external E-Stop sources (e.g., PS5 thread)
        bus_estop = bool(bus.is_estop())
        if bus_estop != self.estop_enabled.get():
            self.estop_enabled.set(bus_estop)
            self._refresh_estop_banner()

        # Build q_target from sliders
        q_target = np.array([float(v.get()) for v in self.slider_vars], dtype=float)

        # Smoothing
        speed = max(0.1, min(2.0, float(self.speed_scale.get())))
        if not hasattr(self, "_last_q"):
            self._last_q = q_target.copy()
        self._last_q = self._last_q + (q_target - self._last_q) * min(1.0, speed)

        # Clamp and prepare for publish
        dof = DOF_MAP.get(self.active_key, len(self.slider_vars))
        q_pub = self._last_q[:dof] if len(self._last_q) >= dof else np.pad(self._last_q, (0, dof - len(self._last_q)))

        # Publish only when override is ON, E-Stop is NOT engaged,
        # and joystick hasn’t been moved recently
        if (
            self.override_enabled.get()
            and not self.estop_enabled.get()
            and not bus.is_joystick_active_recent(250)  # ignore while joystick moving
        ):
            bus.publish_q(self.active_key, q_pub.tolist())

        # Schedule next tick
        self.after(35, self._tick)


def launch_gui():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()

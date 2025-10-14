#!/usr/bin/env python3
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os
from math import pi

class Load(DHRobot3D):
    def __init__(self):
        # ---------- DH links (6 revolute joints) ----------
        links = self._create_DH()

        # ---------- Visual link names (base + 6 joints) ----------
        link3D_names = dict(
            link0='link_0',  # base
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',  # end-effector / wrist
        )

        # ---------- Initial configuration ----------
        qtest = [0, 0, 0, 0, 0, 0]

        # ---------- Visual transforms (adjust for your meshes) ----------
        qtest_transforms = [
            spb.transl(+0.000, 0, 0.00), # link_0 (base)
            spb.transl(+0.000, 0, 0.40), # link_1
            spb.transl(+0.025, 0, 0.40), # link_2
            spb.transl(+0.480, 0, 0.40), # link_3
            spb.transl(+0.480, 0, 0.435), # link_4 Z-axis is flipped by the offset onlyt for dh[2} so now we add the a figures instead of d figure for this set of dh parameters
            spb.transl(+0.880, 0, 0.435), # link_5
            spb.transl(+0.960, 0, 0.435), # link_6
            # spb.transl(+0.000, 0, 0.00), # link_0 (base)
            # spb.transl(+0.000, 0, 0.40), # link_1
            # spb.transl(+0.020, 0, 0.40), # link_2
            # spb.transl(+0.480, 0, 0.40), # link_3
            # spb.transl(+0.460, 0, 0.43), # link_4
            # spb.transl(+0.880, 0, 0.43), # link_5
            # spb.transl(+0.960, 0, 0.43), # link_6
        ]

            

        # ---------- Mesh loading ----------
        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "Meshes", "KUKA_Meshes")

        try_exts = [".dae", ".stl", ".obj"]
        found = 0
        for i in range(7):
            stem = link3D_names[f"link{i}"]
            if any(os.path.exists(os.path.join(mesh_dir, stem + ext)) for ext in try_exts):
                found += 1
        print("[INFO] Mesh directory:", mesh_dir)
        print(f"[INFO] Found {found}/7 meshes")

        # ---------- Initialize parent class ----------
        super().__init__(
            links,
            link3D_names,
            name='KUKA_KR6',
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        self.q = qtest
        self.tool_length = 0.12
        self.addconfiguration("tool_length", self.tool_length)

    # ==============================================================
    #                   DH TABLE CREATION
    # ==============================================================

    def _create_DH(self):
        """
        Define DH parameters for the 6-DOF KUKA-like manipulator.
        Target axis directions:
        Joint 1 -> Z
        Joint 2 -> Y
        Joint 3 -> Y
        Joint 4 -> Z
        Joint 5 -> Y
        Joint 6 -> Z
        """
        a      = [0.025, 0.455, 0.035,  0.0,   0.0,   0.0  ]
        d      = [0.4,   0.0,   0.0,    0.4,   0.0,   0]
        alpha  = [pi/2,  0,     pi/2,   pi/2,  -pi/2, 0.0  ]
        offset = [0,     0,     pi/2,   0,     0,     0    ]


        # a     = [0,     0.025, 0.315, 0.035, 0.0,   0.0  ]
        # d     = [0.400, 0.0,   0.0,   0.365, 0.0,   0.080]
        # alpha = [pi/2,  0.0,   0.0,   pi/2,  -pi/2, 0.0  ]


        # Lengths in meters (example estimates)
        # a = [0.02, 0.0, 0.0, 0.20, 0.0, 0.0] # Lengths in x-direction
        # d = [0.00, 0.0, -1.0, 0.0, 0.0, 0.0] # Offsets in z-direction

        # # α_i defines the tilt between successive joint z-axes
        # alpha = [
        #     -pi/2,   # z0 → +y, joint2 rotates about y
        #     -pi,   # z1 → -z or flipped plane
        #     -pi/2,   # z2 → +y, joint4 rotates about y
        #     0.0,     # z3 → z
        #     -pi/2,   # z4 → +y
        #     0.0      # z5 → z
        # ]

        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i]) for i in range(6)]
        return links

    # ==============================================================
    #                   TEST FUNCTION
    # ==============================================================

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, pi/8])

        # Base offset for visibility
        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        # --- Print joint axes at q=0 ---
        q0 = [0]*6
        J0 = self.jacob0(q0)      # 6x6 Jacobian in base frame
        axes = J0[3:6, :]         # Angular velocity part (joint z-axes)
        print("Joint axes in base frame at q=0 (J1..J6):")
        for j in range(6):
            a = axes[:, j]
            n = (a @ a) ** 0.5
            a = a / (n if n > 0 else 1.0)
            print(f"J{j+1}: {a}")

        # --- Test motion ---
        q_start = [0, 0, 0, 0, 0, 0]
        q_goal  = [pi/2]*6

        qtraj = rtb.jtraj(q_start, q_goal, 60).q



        # put this inside Load.test() before running the animation loop
        q_debug = q_goal  # final pose you expect
        print("\n=== FK debug at q_goal ===")
        # Try using fkine_all() if available (SerialLink API)
        try:
            fk_all = self.fkine_all(q_debug)   # list/array of SE3 frames (base->link_i)
            for i, T in enumerate(fk_all):
                print(f"link{i} origin: {T.t}")   # T.t is xyz translational part
        except Exception as e:
            print("fkine_all() not available, falling back to incremental A() multiplication:", e)
            T = SE3()
            for i in range(len(self.links)):
                # self.A(i, q) returns homogeneous transform for link i (if API present)
                try:
                    A_i = self.A(i, q_debug)
                    T = T * A_i
                    print(f"link{i+1} origin: {T.t}")
                except Exception as e2:
                    print("Couldn't use A(); last known T:", T.t, "error:", e2)
                    break

        # Also print the end-effector pose
        try:
            T_ee = self.fkine(q_debug)
            print("end-effector (fkine) origin:", T_ee.t)
        except:
            pass
        print("========================\n")



        for q in qtraj:
            self.q = q
            env.step(0.02)

        print("Animation complete")
        env.hold()

# ==============================================================
#                   RUN DIRECTLY
# ==============================================================

if __name__ == "__main__":
    r = Load()
    r.test()

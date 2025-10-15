#!/usr/bin/env python3
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os
from math import pi

class KR6_Robot(DHRobot3D):
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
        qtest = [0, -pi/2, 0, 0, 0, 0]

        # ---------- Visual transforms (adjust for your meshes) ----------
        qtest_transforms = [
            spb.transl(+0.000, 0, 0.00), # link_0 (base)
            spb.transl(+0.000, 0, 0.40), # link_1
            spb.transl(+0.025, 0, 0.40), # link_2
            spb.transl(+0.480, 0, 0.40), # link_3
            spb.transl(+0.480, 0, 0.435), # link_4 Z-axis is flipped by the offset onlyt for dh[2} so now we add the a figures instead of d figure for this set of dh parameters
            spb.transl(+0.900, 0, 0.435), # link_5
            spb.transl(+0.98, 0, 0.435), # link_6
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


        self.tool_length = 0.08
        self.tool = SE3(0, 0, self.tool_length)

    # ==============================================================
    #                   DH TABLE CREATION
    # ==============================================================

    def _create_DH(self):
        """
        Define DH parameters for the 6-DOF KUKA KR6.
        """
        a       = [0.025, 0.455, 0.035,  0.00,  0.00, 0.0]
        d       = [0.400, 0.000, 0.000,  0.42,  0.00, 0.0]
        alpha   = [ pi/2, 0.000,  pi/2,  pi/2, -pi/2, 0.0]
        offset  = [0.000,  pi/2,  pi/2,  0.00,  0.00, 0.0]
        qlim    = [[-pi, pi]] * 6;  
        qlim[3] = [-pi/4, pi/4]


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

        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i]) for i in range(6)]
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

        # --- Test motion ---
        q_start = [0, 0, 0, 0, 0, 0]
        q_goal  = [0, 0, 0, 0, pi/2, 0]

        qtraj = rtb.jtraj(q_start, q_goal, 60).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        print("Animation complete")
        env.hold()

# ==============================================================
#                   RUN DIRECTLY
# ==============================================================

if __name__ == "__main__":
    r = KR6_Robot()
    r.test()

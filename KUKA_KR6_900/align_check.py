#!/usr/bin/env python3
"""
align_check.py

Compare DH model joint-frame transforms to URDF (or reference) transforms and print correction transforms.
Usage:
  - Put your KR6 URDF path in URDF_PATH (or set to None to only compare DH -> mesh transforms if you already
    embed transforms).
  - Update the `dh_params` to match your _create_DH() values (the ones currently used).
  - The script will print per-joint corrections and also print Python expressions to paste into your qtest_transforms.
"""

import numpy as np
from math import pi
import spatialmath.base as spb
from spatialmath import SE3
import roboticstoolbox as rtb

# Attempt to import URDF parser (if you have URDF file)
try:
    from urdf_parser_py.urdf import URDF
    URDF_AVAILABLE = True
except Exception as e:
    URDF_AVAILABLE = False
    # print("urdf_parser_py not found. Install with `pip install urdf-parser-py` if you want URDF comparison.")

# ---------- USER EDIT: point this at your URDF if you have one ----------
URDF_PATH = "/home/nathan/git/IR_Assignment2/KUKA/kr6r900sixx.urdf"  # <-- edit to your local URDF file, or None
# ---------------------------------------------------------------------

# ---------- USER EDIT: your current DH parameters from _create_DH ----------
# Keep them in the same order used by your code (6 joints)
a = [0.0, 0.18, 0.45, 0.11, 0.32, 0.08]
d = [0.400, 0.0, 0.0, 0.420, 0.0, 0.080]
alpha = [pi/2, 0.0, 0.0, pi/2, -pi/2, 0.0]
# qtest you use for visual alignment
qtest = [0, -pi/2, 0, 0, 0, 0]
# The 3D mesh transform you currently apply per link (qtest_transforms in your file).
# Provide the same list order as link0..link6 (7 entries: base + 6 links)
qtest_transforms = [
    spb.transl(0,0,0),                                # base_link → DH frame 0
    spb.transl(0,0,0.400),                            # link_1 offset
    spb.transl(0.02,0,0.4) @ spb.rpy2tr(0,-pi/2,0),   # link_2 offset (your current values)
    spb.transl(0.02,0,0.855) @ spb.rpy2tr(0,-pi/2,0),
    spb.transl(-0.015,0,0.86) @ spb.rpy2tr(0,-pi/2,0),
    spb.transl(-0.015,0,1.28) @ spb.rpy2tr(0,-pi/2,0),
    spb.transl(-0.015,0,1.36) @ spb.rpy2tr(0,-pi/2,0)
]
# ---------------------------------------------------------------------

def make_dh_robot(a,d,alpha,qlim=None):
    links = []
    if qlim is None:
        qlim = [[-2*pi, 2*pi] for _ in range(len(a))]
    for i in range(len(a)):
        links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
    robot = rtb.DHRobot(links, name='KR6_candidate')
    return robot

def homog_to_rpy(T):
    # Return roll, pitch, yaw in radians using spatialmath SE3 convenience
    s = SE3(T)
    rpy = s.rpy(order='xyz')  # roll, pitch, yaw
    return rpy

def print_vec(v):
    return f"[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]"

def print_rpy(rpy):
    return f"[{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}]"

def load_urdf_transforms(urdf_path):
    robot = URDF.from_xml_file(urdf_path)
    # Build base->link transforms by traversing the tree (like earlier example)
    base = robot.get_root()
    transforms = { base: np.eye(4) }

    def homogeneous_from_xyz_rpy(xyz, rpy):
        return spb.trvec(xyz) @ spb.rpy2tr(rpy)

    # simple traversal
    pending = [base]
    while pending:
        parent = pending.pop(0)
        for joint in robot.joints:
            if joint.parent == parent:
                T_parent_to_child = homogeneous_from_xyz_rpy(joint.origin.xyz, joint.origin.rpy)
                transforms[joint.child] = transforms[parent].dot(T_parent_to_child)
                pending.append(joint.child)
    # Map joint frames: URDF origins are joint origins. We'll map them to the link names for comparison.
    return robot, transforms

def main():
    # Make DH robot and compute base->joint (frame i) transforms for qtest
    dh_robot = make_dh_robot(a,d,alpha)
    # robot.fkine returns end-effector transforms; we need per-link transforms. Robotics Toolbox provides
    # .A(i, theta) for link transforms; accumulate them to get base->each joint frame.
    T_dh = [np.eye(4)]
    T_acc = np.eye(4)
    for i in range(dh_robot.n):
        Ai = dh_robot.links[i].A(qtest[i]).A  # get raw 4x4 numpy array
        T_acc = T_acc @ Ai
        T_dh.append(T_acc.copy())
    # Now we have T_dh[0..6] corresponding to base and link frames after each joint

    # If URDF available, load and compute base->link transforms for URDF
    T_urdf = None
    urdf_robot = None
    if URDF_AVAILABLE and URDF_PATH:
        try:
            urdf_robot, T_urdf = load_urdf_transforms(URDF_PATH)
            print("Loaded URDF and computed base->link transforms.")
        except Exception as e:
            print("Error loading URDF:", e)
            T_urdf = None

    # If URDF not present, use existing qtest_transforms as reference for visual mesh positions
    if T_urdf is None:
        print("No URDF transforms found; will use your qtest_transforms as reference (visual alignment).")
        # convert qtest_transforms (spatialmath matrix or 4x4 arrays) to numpy arrays
        T_ref = [np.array(t) for t in qtest_transforms]
    else:
        # Try to make a list of transforms that correspond to the chain order:
        # We need an ordered list of link frames: base_link, link_1, link_2, ..., link_6
        # Heuristic: find link names in URDF that match your mesh naming (base_link, link_1, ...)
        # Otherwise print available link names for you to map manually.
        expected = ['base_link','link_1','link_2','link_3','link_4','link_5','link_6']
        missing = [n for n in expected if n not in T_urdf]
        if missing:
            print("Warning: expected link names not all found in URDF transforms. Available links:")
            print(list(T_urdf.keys()))
            # fallback: use whatever order we can; but this is brittle
            # For now, try to use the root and the children traversal order
            # Convert dict to list by sorted keys (not ideal)
            T_ref = []
            for k in sorted(T_urdf.keys()):
                T_ref.append(T_urdf[k])
            # ensure length at least 7 by padding last
            while len(T_ref) < 7:
                T_ref.append(np.eye(4))
            T_ref = T_ref[:7]
        else:
            T_ref = [T_urdf[n] for n in expected]

    # Now we have T_ref[0..6] and T_dh[0..6], compute corrections
    print("\nPer-link corrections (T_corr = T_ref * inv(T_dh))")
    corrections = []
    for i in range(min(len(T_ref), len(T_dh))):
        Td = T_dh[i]
        Tr = T_ref[i]
        Tcorr = Tr.dot(np.linalg.inv(Td))
        corrections.append(Tcorr)
        trans = Tcorr[:3,3]
        rpy = homog_to_rpy(Tcorr)
        print(f"Link {i}: translation (m) = {print_vec(trans)}, rpy (rad) = {print_rpy(rpy)}")
        # also show small-angle degrees for convenience
        print(f"        (deg) rpy = [{rpy[0]*180/pi:.2f}, {rpy[1]*180/pi:.2f}, {rpy[2]*180/pi:.2f}]")
        print("        Python matrix expression (use in qtest_transforms):")
        print(f"        spb.trvec({print_vec(trans)}) @ spb.rpy2tr({print_rpy(rpy)})")
        print("")

    # Provide suggested quick-fixes:
    print("SUGGESTIONS:")
    print(" - If corrections for joints 3..6 are mostly pure rotations about X or Y of +/-90deg,")
    print("   you likely used the wrong DH convention (MDH vs SDH). Try using RevoluteMDH for links or")
    print("   re-derive the table in the other convention.")
    print(" - If corrections show small translations (e.g. a few cm) along X or Z, add those offsets into")
    print("   your DH 'a' or 'd' values for the affected links (or apply an extra fixed transform in qtest_transforms).")
    print(" - The simplest/fastest fix for visualization: replace qtest_transforms[i] with qtest_transforms[i] @ correction")
    print("   or directly set qtest_transforms[i] = spb.trvec(...) @ spb.rpy2tr(...) using the printed expression above.")
    print("")
    print("If you want, paste the printed spb expressions into your class's qtest_transforms list (for links 3..6),")
    print("then re-run and see if the wrist sits correctly. That won't change your FK math — only how meshes are placed.")
    print("\nDONE.")

if __name__ == "__main__":
    main()

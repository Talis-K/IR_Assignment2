#!/usr/bin/env python3
"""
Swift demo for KUKA KR150 using your IR_Assignment2 folder structure.

Folders (from your upload):
  - MODEL_ROOT = /mnt/data/IR_A2/IR_Assignment2-main/LBR Kuka/KUKA_KR150
    ├── urdf/kr150_2.xacro, kr150_2_macro.xacro
    └── meshes/kr150_2/{visual,collision}/*.dae|*.stl

This script:
  1) Converts Xacro -> URDF (needs: pip install xacro)
  2) Rewrites any 'package://...' mesh URIs to relative paths
  3) Loads the URDF into Robotics Toolbox ERobot and displays in Swift
  4) Plays a small joint-space motion

Dependencies:
  pip install roboticstoolbox-python spatialmath-python swift xacro
"""

import os
import re
import sys
import tempfile
import subprocess
import numpy as np

import swift
import roboticstoolbox as rtb

MODEL_ROOT = r"/mnt/data/IR_A2/IR_Assignment2-main/LBR Kuka/KUKA_KR150"
XACRO_MAIN = r"/mnt/data/IR_A2/IR_Assignment2-main/LBR Kuka/KUKA_KR150/urdf/kr150_2.xacro"

def ensure(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")


def run_xacro(xacro_in: str, urdf_out: str):
    """Generate URDF from Xacro using the xacro CLI."""
    cmd = ["xacro", xacro_in, "-o", urdf_out]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("ERROR: `xacro` CLI not found. Install with: pip install xacro", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print("xacro failed:\n", e.stderr, file=sys.stderr)
        raise


def fix_mesh_uris(urdf_path: str):
    """Replace package:// URIs with relative paths that resolve under MODEL_ROOT."""
    with open(urdf_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    patterns = [
        r"package://kuka_description/kuka_kr150/meshes/kr150_2/",
        r"package://kuka_kr150/meshes/kr150_2/",
        r"package://kuka_description/meshes/kr150_2/",
        r"package://kuka_description/meshes/",
        r"package://kuka_kr150/meshes/",
    ]
    for pat in patterns:
        txt = txt.replace(pat, "meshes/kr150_2/")

    # normalize any residual prefixes
    txt = re.sub(r'filename="(?:\./)?meshes/', 'filename="meshes/', txt)

    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(txt)


def build_urdf() -> str:
    ensure(MODEL_ROOT)
    ensure(XACRO_MAIN)

    tmp = tempfile.mkdtemp(prefix="kr150_urdf_")
    urdf_out = os.path.join(tmp, "kr150_generated.urdf")
    run_xacro(XACRO_MAIN, urdf_out)
    fix_mesh_uris(urdf_out)
    return urdf_out


def play_traj(robot, q0, q1, steps=120, dt=0.02, env=None):
    traj = rtb.jtraj(q0, q1, steps)
    for q in traj.q:
        robot.q = q
        if env is not None:
            env.step(dt)


def main():
    urdf = build_urdf()
    robot = rtb.ERobot.URDF(urdf, meshpath=MODEL_ROOT)

    env = swift.Swift()
    env.launch(realtime=True, title="KUKA KR150 • Swift Demo")
    env.add(robot)

    q_ready = robot.qr if robot.qr is not None else np.zeros(robot.n)
    robot.q = q_ready
    env.step(0.5)

    q_target = q_ready.copy()
    for i, delta_deg in enumerate([20, -30, 25, -40, 30, 15]):
        if i < robot.n:
            q_target[i] += np.deg2rad(delta_deg)

    play_traj(robot, q_ready, q_target, env=env)
    play_traj(robot, q_target, q_ready, env=env)
    print("Done. You can pan/zoom the Swift view in your browser.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nIf this fails at the xacro step, be sure you have `xacro` installed:") 
        print("  pip install xacro\n")
        raise

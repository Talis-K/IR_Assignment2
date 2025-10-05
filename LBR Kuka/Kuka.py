#!/usr/bin/env python3
"""
KUKA KR150 • Swift (No ROS).
- Patches Xacro to remove ROS substitutions (incl. include path).
- Runs xacro on the patched copy (you already have xacro).
- Absolutizes mesh paths so RTB/Swift can always find geometry.
- Loads in Swift and plays a short trajectory.
- Includes detailed DEBUG prints + robust Swift launcher.

Folder structure (relative to THIS file):
  LBR Kuka/
    Kuka.py
    KUKA_KR150/
      urdf/
        kr150_2.xacro
        kr150_2_macro.xacro
      meshes/
        kr150_2/
          visual/*.dae
          collision/*.stl
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import swift
import roboticstoolbox as rtb

# ---------------- Paths ----------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
KR150_ROOT   = os.path.join(SCRIPT_DIR, "KUKA_KR150")
URDF_DIR     = os.path.join(KR150_ROOT, "urdf")
MESH_ROOT    = os.path.join(KR150_ROOT, "meshes")
XACRO_MAIN   = os.path.join(URDF_DIR, "kr150_2.xacro")
XACRO_MACRO  = os.path.join(URDF_DIR, "kr150_2_macro.xacro")

def ensure(p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")

# ---------- Step 1: Patch Xacro (ROS-free) ----------
def patch_xacro_to_temp(src_main: str, src_macro: str, mesh_root: str) -> str:
    """
    Copy Xacros to a temp folder and:
      - Rewrite the include '$(find kuka_kr150_support)/urdf/kr150_2_macro.xacro'
        to 'kr150_2_macro.xacro' (no ROS).
      - Replace any '$(find kuka_kr150_support)' with '.'.
      - Map package://.../meshes and $(find ...)/meshes to 'meshes/...'.
    Returns path to patched main xacro in the temp folder.
    """
    ensure(src_main); ensure(src_macro); ensure(mesh_root)
    tmpdir = tempfile.mkdtemp(prefix="kr150_xacro_")
    main_tmp  = os.path.join(tmpdir, os.path.basename(src_main))
    macro_tmp = os.path.join(tmpdir, os.path.basename(src_macro))
    shutil.copy2(src_main,  main_tmp)
    shutil.copy2(src_macro, macro_tmp)

    print(f"\n[DEBUG] Patched copies created in temp folder: {tmpdir}")

    def patch_file(path: str):
        print(f"[DEBUG] Patching file: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        # 1) Fix include path that required ROS
        txt = re.sub(
            r'<\s*xacro:include\s+filename="\$\(\s*find\s+kuka_kr150_support\s*\)/urdf/kr150_2_macro\.xacro"\s*/\s*>',
            '<xacro:include filename="kr150_2_macro.xacro" />',
            txt
        )

        # 2) Any remaining $(find kuka_kr150_support) → "."
        txt = re.sub(r'\$\(\s*find\s+kuka_kr150_support\s*\)', '.', txt)

        # 3) Map common package:// URIs to local meshes/
        for pat in [
            r'package://kuka_description/kuka_kr150/meshes/kr150_2/',
            r'package://kuka_kr150/meshes/kr150_2/',
            r'package://kuka_description/meshes/kr150_2/',
            r'package://kuka_description/meshes/',
            r'package://kuka_kr150/meshes/',
        ]:
            txt = txt.replace(pat, 'meshes/kr150_2/')

        # 4) $(find ...)/meshes/... → meshes/...
        txt = re.sub(r'\$\(\s*find[-\w]*\s+[^\)]+\)/meshes/kr150_2/', 'meshes/kr150_2/', txt)
        txt = re.sub(r'\$\(\s*find[-\w]*\s+[^\)]+\)/meshes/', 'meshes/', txt)

        # 5) $(find-pkg-share ...) → .
        txt = re.sub(r'\$\(\s*find-pkg-share\s+[^\)]+\)', '.', txt)

        # 6) Normalize filename="./meshes/..." → "meshes/..."
        txt = re.sub(r'filename="(?:\./)?meshes/', 'filename="meshes/', txt)

        with open(path, "w", encoding="utf-8") as fw:
            fw.write(txt)
        print(f"[DEBUG] Finished patching: {path}")

    patch_file(main_tmp)
    patch_file(macro_tmp)
    return main_tmp

# ---------- Step 2: Run xacro ----------
def run_xacro(xacro_in: str, urdf_out: str):
    print(f"\n[DEBUG] Running xacro on: {xacro_in}")
    print(f"[DEBUG] Output URDF will be written to: {urdf_out}")
    print("[DEBUG] Current working directory:", os.getcwd())
    cmd = ["xacro", xacro_in, "-o", urdf_out]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[DEBUG] ✅ Xacro conversion succeeded.")
    except subprocess.CalledProcessError as e:
        sys.stderr.write("\n[xacro failed after patching]\n")
        sys.stderr.write(e.stderr or str(e))
        raise

# ---------- Step 3: Absolutize mesh paths ----------
def absolutize_mesh_paths(urdf_in: str, mesh_root: str) -> str:
    print(f"\n[DEBUG] Absolutizing mesh paths in URDF: {urdf_in}")
    print(f"[DEBUG] Mesh root: {mesh_root}")
    ensure(urdf_in); ensure(mesh_root)
    with open(urdf_in, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    # Ensure 'meshes/...' then prefix with absolute path
    txt = re.sub(r'filename="(?:\./)?meshes/', 'filename="meshes/', txt)
    mesh_root_abs = mesh_root.replace("\\", "/")
    txt = re.sub(r'filename="meshes/', f'filename="{mesh_root_abs}/', txt)

    tmpdir = tempfile.mkdtemp(prefix="kr150_urdf_")
    out = os.path.join(tmpdir, "kr150_abs.urdf")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[DEBUG] ✅ Wrote absolutized URDF to: {out}")
    return out

# ---------- Helper: small motion ----------
def play_traj(robot, q0, q1, steps=120, dt=0.02, env=None):
    traj = rtb.jtraj(q0, q1, steps)
    for q in traj.q:
        robot.q = q
        if env is not None:
            env.step(dt)

# ---------- Main ----------
def main():
    print("\n========== [DEBUG] Starting KUKA KR150 Loader ==========")
    ensure(KR150_ROOT); ensure(URDF_DIR); ensure(MESH_ROOT); ensure(XACRO_MAIN); ensure(XACRO_MACRO)

    # 1) Patch Xacro to a temp location
    main_patched = patch_xacro_to_temp(XACRO_MAIN, XACRO_MACRO, MESH_ROOT)

    # 2) Run xacro -> URDF
    tmp_build = tempfile.mkdtemp(prefix="kr150_build_")
    urdf_out  = os.path.join(tmp_build, "kr150_generated.urdf")
    run_xacro(main_patched, urdf_out)

    print(f"\n[DEBUG] URDF generated at: {urdf_out}")
    with open(urdf_out, "r", encoding="utf-8", errors="ignore") as f:
        head = ''.join(f.readlines()[:30])
    print("[DEBUG] --- First 30 lines of URDF ---")
    print(head)
    print("[DEBUG] --- End preview ---")

    # 3) Absolutize mesh paths
    urdf_abs = absolutize_mesh_paths(urdf_out, MESH_ROOT)

    # 4) Load in RTB (version-agnostic)
    print(f"[DEBUG] Loading URDF into Robotics Toolbox: {urdf_abs}")
    try:
        robot = rtb.ERobot.URDF(urdf_abs)
        print("[DEBUG] ✅ Robot loaded successfully via ERobot.URDF()")
    except Exception as e:
        print(f"[DEBUG] ❌ ERobot.URDF() failed: {e}")
        from roboticstoolbox.tools.urdf import URDF as URDFLoader
        from spatialmath import SE3
        model = URDFLoader.load(urdf_abs)
        robot = rtb.ERobot(model, base=SE3(), tool=SE3())
        print("[DEBUG] ✅ Loaded using fallback URDFLoader.load()")
    print(f"[DEBUG] Robot DOF: {robot.n}")

    # 5) Launch Swift (robust launcher; don't use boolean browser arg)
    print("[DEBUG] Launching Swift environment...")
    env = swift.Swift()
    swift_port = 8123
    launched = False
    for candidate in [os.environ.get("BROWSER", ""), "chrome", "safari", "firefox", None]:
        try:
            if candidate is None or candidate == "":
                env.launch(realtime=True, port=swift_port)  # no browser kwarg (avoids bool issues)
            else:
                env.launch(realtime=True, browser=candidate, port=swift_port)  # string only
            launched = True
            break
        except TypeError as e:
            print(f"[DEBUG] Swift launch retry with browser={candidate!r}: {e}")
            continue
        except Exception as e:
            print(f"[DEBUG] Swift launch failed with browser={candidate!r}: {e}")
            continue

    if not launched:
        swift_port = 8125
        print("[DEBUG] Final fallback: launching Swift without browser param on port", swift_port)
        env.launch(realtime=True, port=swift_port)

    # Older Swift versions don't expose env.url — print the URL ourselves
    manual_url = f"http://localhost:{swift_port}/"
    print(f"[DEBUG] Swift launched. If no tab opened automatically, open this URL:\n{manual_url}\n")

    # 6) Add robot & move
    env.add(robot)
    print("[DEBUG] Robot added to Swift environment.")

    q_ready = robot.qr if robot.qr is not None else np.zeros(robot.n)
    robot.q = q_ready
    env.step(0.5)

    q_target = q_ready.copy()
    for i, delta_deg in enumerate([20, -30, 25, -40, 30, 15]):
        if i < robot.n:
            q_target[i] += np.deg2rad(delta_deg)

    print("[DEBUG] Playing trajectory forward...")
    play_traj(robot, q_ready, q_target, env=env)
    print("[DEBUG] Playing trajectory back...")
    play_traj(robot, q_target, q_ready, env=env)

    print("✅ [DEBUG] KR150 loaded successfully in Swift. Pan/zoom in browser.")
    print("============================================================\n")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Missing path: {e}")
        print("Ensure this script is next to 'KUKA_KR150' with 'urdf' and 'meshes' inside.")
        raise
    except Exception as e:
        print("\nIf it fails at import time, confirm you have:")
        print("  pip install roboticstoolbox-python spatialmath-python swift")
        print("(xacro is already installed in your env.)\n")
        raise

#!/usr/bin/env python3
"""
KUKA KR150 • Swift (No ROS) — hardened loader + joint sweep demo

- Removes ROS $(find ...) and includes
- Expands xacro without ROS
- Absolutizes all mesh paths (package://, file://, ./meshes, etc.)
- Two safe rendering options:
  * COLLISION_ONLY = True  → display collision STLs (fast & stable)
  * COLLISION_ONLY = False → try to show visuals; if only DAEs exist,
                             auto-convert to STL (needs trimesh)

If the browser shows "client-side error", keep COLLISION_ONLY=True.
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
from typing import List

import numpy as np
import swift
import roboticstoolbox as rtb

# ---------------- Config ----------------
COLLISION_ONLY = True            # safest; avoids DAE visuals
AUTO_CONVERT_DAE_TO_STL = True   # used when COLLISION_ONLY is False
SWIFT_PORT = int(os.environ.get("SWIFT_PORT", "8123"))

# ---------------- Paths ----------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
KR150_ROOT  = os.path.join(SCRIPT_DIR, "KUKA_KR150")
URDF_DIR    = os.path.join(KR150_ROOT, "urdf")
MESH_ROOT   = os.path.join(KR150_ROOT, "meshes")
XACRO_MAIN  = os.path.join(URDF_DIR, "kr150_2.xacro")
XACRO_MACRO = os.path.join(URDF_DIR, "kr150_2_macro.xacro")

def ensure(p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")

def debug(msg: str):
    print(f"[DEBUG] {msg}")

# ---------- Step 0: optional DAE→STL conversion ----------
def _convert_dae_to_stl_if_needed(mesh_root: str):
    """Convert visual/kr150_2/*.dae → collision/kr150_2/*.stl if missing (needs trimesh)."""
    try:
        import trimesh  # type: ignore
    except Exception:
        debug("trimesh not installed; skipping auto-conversion.")
        return
    src_dir = os.path.join(mesh_root, "visual", "kr150_2")
    dst_dir = os.path.join(mesh_root, "collision", "kr150_2")
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if not name.lower().endswith(".dae"):
            continue
        stem = os.path.splitext(name)[0]
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, stem + ".stl")
        if os.path.isfile(dst):
            continue
        try:
            debug(f"Converting {name} → {os.path.basename(dst)}")
            mesh = trimesh.load(src, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            mesh.export(dst)
        except Exception as e:
            debug(f"  ↳ conversion failed for {name}: {e}")

# ---------- Step 1: Patch Xacro (ROS-free) ----------
def patch_xacro_to_temp(src_main: str, src_macro: str) -> str:
    ensure(src_main); ensure(src_macro); ensure(MESH_ROOT)
    tmpdir = tempfile.mkdtemp(prefix="kr150_xacro_")
    main_tmp  = os.path.join(tmpdir, os.path.basename(src_main))
    macro_tmp = os.path.join(tmpdir, os.path.basename(src_macro))
    shutil.copy2(src_main,  main_tmp)
    shutil.copy2(src_macro, macro_tmp)
    debug(f"Patched copies created in temp folder: {tmpdir}")

    def patch_file(path: str):
        debug(f"Patching file: {path}")
        txt = open(path, "r", encoding="utf-8", errors="ignore").read()

        # Replace ROS-dependent include → local macro file
        txt = re.sub(
            r'<\s*xacro:include\s+filename="[^"]*kr150_2_macro\.xacro"\s*/\s*>',
            '<xacro:include filename="kr150_2_macro.xacro" />',
            txt
        )
        # Kill remaining ROS find macros (catch-all)
        txt = re.sub(r'\$\(\s*find[-\w]*\s+[^\)]+\)', '.', txt)

        # Map package:// URIs to meshes/kr150_2/ or meshes/
        txt = re.sub(r'package://[A-Za-z0-9_\-]+/meshes/kr150_2/', 'meshes/kr150_2/', txt)
        txt = re.sub(r'package://[A-Za-z0-9_\-]+/meshes/', 'meshes/', txt)

        # Normalize filename paths (strip ./)
        txt = re.sub(r'filename="(?:\./)+meshes/', 'filename="meshes/', txt)

        open(path, "w", encoding="utf-8").write(txt)
        debug(f"Finished patching: {path}")

    patch_file(main_tmp)
    patch_file(macro_tmp)
    return main_tmp

# ---------- Step 2: Run xacro ----------
def run_xacro(xacro_in: str, urdf_out: str):
    debug(f"Running xacro on: {xacro_in}")
    cmd = ["xacro", xacro_in, "-o", urdf_out]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        debug("✅ Xacro conversion succeeded.")
    except subprocess.CalledProcessError as e:
        sys.stderr.write("\n[xacro failed after patching]\n")
        sys.stderr.write(e.stderr or str(e))
        raise

# ---------- Step 3a: collision-only visuals ----------
def make_collision_only(urdf_in: str) -> str:
    debug(f"Converting visuals to collision-only in: {urdf_in}")
    txt = open(urdf_in, "r", encoding="utf-8", errors="ignore").read()
    # Swap 'visual/' → 'collision/' and '.dae"' → '.stl"'
    txt = txt.replace('visual/', 'collision/')
    txt = re.sub(r'\.dae"', '.stl"', txt)

    out_dir = tempfile.mkdtemp(prefix="kr150_collision_")
    out = os.path.join(out_dir, "kr150_collision_only.urdf")
    open(out, "w", encoding="utf-8").write(txt)
    debug(f"✅ Collision-only URDF at: {out}")
    return out

# ---------- Step 3b: absolutize mesh paths ----------
def absolutize_mesh_paths(urdf_in: str, mesh_root: str) -> str:
    debug(f"Absolutizing mesh paths: {urdf_in}")
    txt = open(urdf_in, "r", encoding="utf-8", errors="ignore").read()

    # package://… → absolute (assume meshes under MESH_ROOT)
    def repl_pkg(m):
        # groups: pkg, rel
        rel = m.group(2)
        path = os.path.join(mesh_root, rel).replace("\\", "/")
        return f'filename="{path}"'

    txt = re.sub(r'filename="package://([A-Za-z0-9_\-]+)/meshes/([^"]+)"', repl_pkg, txt)

    # file:// → strip scheme
    txt = re.sub(r'filename="file://', 'filename="', txt)

    # ./meshes or meshes → absolute
    mesh_root_fixed = mesh_root.replace("\\", "/")
    txt = re.sub(r'filename="(?:\./)?meshes/', f'filename="{mesh_root_fixed}/', txt)

    out_dir = tempfile.mkdtemp(prefix="kr150_abs_")
    out = os.path.join(out_dir, "kr150_abs.urdf")
    open(out, "w", encoding="utf-8").write(txt)

    # Quick sanity print
    refs = re.findall(r'filename="([^"]+)"', txt)[:8]
    debug("Sample mesh refs: " + ", ".join(refs))
    return out

# ---------- Mesh existence check ----------
def check_meshes_exist(urdf_path: str) -> List[str]:
    txt = open(urdf_path, "r", encoding="utf-8", errors="ignore").read()
    paths = re.findall(r'filename="([^"]+)"', txt)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("\n[WARN] Missing meshes (showing up to 8):")
        for m in missing[:8]:
            print("   -", m)
    return missing

# ---------- Small motion helper ----------
def play_traj(robot, q0, q1, steps=120, dt=0.02, env=None):
    traj = rtb.jtraj(q0, q1, steps)
    for q in traj.q:
        robot.q = q
        if env is not None:
            env.step(dt)

# ---------- Main ----------
def main():
    print("\n========== KUKA KR150 Loader ==========")
    ensure(KR150_ROOT); ensure(URDF_DIR); ensure(MESH_ROOT); ensure(XACRO_MAIN); ensure(XACRO_MACRO)

    # If we’ll need STLs but only have DAEs, auto-convert
    if not COLLISION_ONLY and AUTO_CONVERT_DAE_TO_STL:
        _convert_dae_to_stl_if_needed(MESH_ROOT)

    # 1) Patch Xacro
    main_patched = patch_xacro_to_temp(XACRO_MAIN, XACRO_MACRO)

    # 2) Xacro → URDF
    tmp_build = tempfile.mkdtemp(prefix="kr150_build_")
    urdf_out  = os.path.join(tmp_build, "kr150_generated.urdf")
    run_xacro(main_patched, urdf_out)
    debug(f"URDF generated at: {urdf_out}")

    # 3) Choose visuals mode
    urdf_for_paths = make_collision_only(urdf_out) if COLLISION_ONLY else urdf_out

    # 4) Absolutize mesh paths
    urdf_abs = absolutize_mesh_paths(urdf_for_paths, MESH_ROOT)

    # 5) Mesh existence check (not fatal)
    _ = check_meshes_exist(urdf_abs)

    # 6) Load robot
    debug(f"Loading URDF into RTB: {urdf_abs}")
    try:
        robot = rtb.ERobot.URDF(urdf_abs)
        debug("✅ ERobot.URDF() succeeded")
    except Exception as e:
        debug(f"ERobot.URDF() failed: {e}; trying legacy loader")
        from roboticstoolbox.tools.urdf import URDF as URDFLoader
        from spatialmath import SE3
        model = URDFLoader.load(urdf_abs)
        robot = rtb.ERobot(model, base=SE3(), tool=SE3())
        debug("✅ Fallback URDFLoader worked")
    print(f"[INFO] Robot DOF: {robot.n}")

    # 7) Swift launch (robust)
    debug("Launching Swift…")
    env = swift.Swift()
    try:
        env.launch(realtime=True, port=SWIFT_PORT)
    except TypeError:
        env.launch(realtime=True, port=SWIFT_PORT)  # older swift; no browser kw
    print(f"[INFO] If needed, open http://localhost:{SWIFT_PORT}/")

    # 8) Add robot & sweep ALL joints safely within limits
    env.add(robot)

    # Start from mid-configuration if qlim available, else qz/zeros
    if getattr(robot, "qlim", None) is not None and robot.qlim.size == 2 * robot.n:
        qmin = robot.qlim[0, :]
        qmax = robot.qlim[1, :]
        q_mid = 0.5 * (qmin + qmax)
    else:
        q_mid = getattr(robot, "qz", None)
        if q_mid is None or len(q_mid) != robot.n:
            q_mid = np.zeros(robot.n)

    robot.q = q_mid.copy()
    env.step(0.5)

    def safe_amp_for_joint(j):
        """Use 30% of joint span (if available), capped at ~35° otherwise."""
        if getattr(robot, "qlim", None) is not None and robot.qlim.size == 2 * robot.n:
            span = float(robot.qlim[1, j] - robot.qlim[0, j])
            amp = 0.3 * span
        else:
            amp = np.deg2rad(35.0)
        return float(amp)

    def clamp_to_limits(q):
        """Clamp config to qlim if available."""
        if getattr(robot, "qlim", None) is not None and robot.qlim.size == 2 * robot.n:
            return np.minimum(np.maximum(q, robot.qlim[0, :]), robot.qlim[1, :])
        return q

    print("[INFO] Sweeping each joint safely within limits…")
    q_curr = robot.q.copy()

    # One full sweep per joint: +amp → center → -amp → center
    for j in range(robot.n):
        amp = safe_amp_for_joint(j)

        # +amp
        q_plus = q_curr.copy()
        q_plus[j] = q_plus[j] + amp
        q_plus = clamp_to_limits(q_plus)
        play_traj(robot, q_curr, q_plus, steps=120, dt=0.015, env=env)

        # back to center
        play_traj(robot, q_plus, q_mid, steps=90, dt=0.015, env=env)
        q_curr = q_mid.copy()

        # -amp
        q_minus = q_curr.copy()
        q_minus[j] = q_minus[j] - amp
        q_minus = clamp_to_limits(q_minus)
        play_traj(robot, q_curr, q_minus, steps=120, dt=0.015, env=env)

        # back to center
        play_traj(robot, q_minus, q_mid, steps=90, dt=0.015, env=env)
        q_curr = q_mid.copy()

    print("✅ All joints moved. Increase amps/steps for bigger/smoother motion.")
    print("If visuals crash in the browser, set COLLISION_ONLY=True (already default).")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Missing path: {e}")
        print("Ensure this script is next to 'KUKA_KR150' with 'urdf' and 'meshes' inside.")
        raise

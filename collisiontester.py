#!/usr/bin/env python3
import os
import time
from math import pi
from itertools import combinations
from typing import List

import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Sphere

from ir_support.robots.DHRobot3D import DHRobot3D
from ir_support import RectangularPrism, line_plane_intersection


# ---------- helpers ----------
def _is_point_inside_triangle(intersect_p: np.ndarray, triangle_verts: np.ndarray) -> bool:
    """
    Barycentric point-in-triangle test (same as your lab util).
    triangle_verts: (3, 3) array (A,B,C).
    """
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv
    if abs(D) < 1e-12:
        return False

    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False

    return True

def _stack_transforms(frames):
    """
    Normalise fkine_all output to a 4x4xN ndarray.

    Accepts any of:
      • An object with .A (either 4x4xN or N x 4 x 4)
      • A numpy ndarray of shape (4,4,N) or (N,4,4) or (4,4)
      • A list/tuple of SE3 objects
      • A list/tuple of 4x4 ndarrays
    """
    # Case 0: numpy ndarray directly
    if isinstance(frames, np.ndarray):
        A = frames
        if A.ndim == 3:
            if A.shape[:2] == (4, 4):          # (4,4,N)
                return A
            if A.shape[1:] == (4, 4):          # (N,4,4) -> transpose to (4,4,N)
                return np.transpose(A, (1, 2, 0))
        if A.shape == (4, 4):                  # single transform -> expand to (4,4,1)
            return A[..., None]

    # Case 1: object with .A (SE3Array-like)
    if hasattr(frames, "A"):
        A = np.asarray(frames.A)
        if A.ndim == 3:
            if A.shape[:2] == (4, 4):          # (4,4,N)
                return A
            if A.shape[1:] == (4, 4):          # (N,4,4) -> transpose to (4,4,N)
                return np.transpose(A, (1, 2, 0))

    # Case 2: list/tuple of SE3 objects, or 4x4 arrays
    if isinstance(frames, (list, tuple)) and len(frames) > 0:
        first = frames[0]
        # SE3-like objects
        if hasattr(first, "A"):
            mats = [np.asarray(T.A) for T in frames]
            A = np.stack(mats, axis=2)  # (4,4,N)
            return A
        # raw 4x4 matrices
        first_arr = np.asarray(first)
        if first_arr.shape == (4, 4):
            mats = [np.asarray(M) for M in frames]
            A = np.stack(mats, axis=2)  # (4,4,N)
            return A

    # Fallthrough: unknown format
    raise TypeError(f"Unsupported fkine_all return type for stacking transforms (type={type(frames)})")

class CollisionDetector:
    """
    Segment-vs-mesh collision checker based on link line segments between consecutive joint frames.
    """

    def __init__(self, env: swift.Swift,volume,center,colour):
        self.env = env
        self.collisions: List[Sphere] = []

        self.cuboid = Cuboid(scale=list(volume), color=list(colour))
        self.cuboid.T = spb.transl(*center)
        env.add(self.cuboid)

        # Mesh data for geometric tests
        self.vertices, self.faces, self.face_normals = RectangularPrism(volume[0], volume[1], volume[2], center=center).get_data()

    def check_pose(self, robot, q) -> bool:
        """
        Returns True if any links intersect with cuboid objects.
        """
        # Normalise fkine_all output to 4x4xN
        frames = robot.fkine_all(q)
        tr = _stack_transforms(frames)

        hit_any = False

        # For each link segment between consecutive frames
        for i in range(tr.shape[2] - 1):
            p0 = tr[:3, 3, i]
            p1 = tr[:3, 3, i + 1]

            # For each mesh face (with triangles)
            for j, face in enumerate(self.faces):
                vert_on_plane = self.vertices[face][0] #
                collision_location, check = line_plane_intersection(self.face_normals[j], vert_on_plane, p0, p1)

                if check != 1:  #Checks if the point has not intersected with the face meaning not crash has occured
                    continue

                # Test intersection point against all triangles that make up the (quad) face
                tri_idxs = np.array(list(combinations(face, 3)), dtype=int)
                for tri in tri_idxs:
                    tri_verts = self.vertices[tri]
                    if _is_point_inside_triangle(collision_location, tri_verts):
                        hit_any = True
                        break  # mark one point per face
        return hit_any

    def check_traj(self, robot: rtb.DHRobot, q_matrix: np.ndarray, stop_on_first=True) -> bool:
        """
        Return True if any configuration collides.
        """
        for q in q_matrix:
            if self.check_pose(robot, q):
                if stop_on_first:
                    return True
        return False

class RobotCollision():
    def __init__(self, env, robot, q_start, q_finished, volume, center, colour=(0.0, 1.0, 0.0, 0.45)):

        detector = CollisionDetector(env,volume,center,colour)

        q_traj = rtb.jtraj(q_start, q_finished, 80).q

        collided = False
        stop_message = False

        for k, q in enumerate(q_traj):
            if not collided:
                robot.q = q #Move the robot
                hit = detector.check_pose(robot, q) #Collision Test
                if hit:
                    collided = True
                    print(f"[Collision Detection]: Collision detected at step {k+1}/{len(q_traj)}")

            if collided and not stop_message:
                stop_message = True
                print("[Collision Detection]: Robot collided with object, stopping further motion.")

            env.step(0.016) #Update the environment

class KR6_Robot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()  # DH Links
        qtest = [0, -pi/2, 0, 0, 0, 0]  # Initial joint config

        # Visual transforms for your meshes
        qtest_transforms = [
            spb.transl(+0.000, 0, 0.000),  # link_0 (base)
            spb.transl(+0.000, 0, 0.400),  # link_1
            spb.transl(+0.025, 0, 0.400),  # link_2
            spb.transl(+0.480, 0, 0.400),  # link_3
            spb.transl(+0.480, 0, 0.435),  # link_4
            spb.transl(+0.900, 0, 0.435),  # link_5
            spb.transl(+0.980, 0, 0.435),  # link_6
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "KUKA_KR6", "Meshes", "KUKA_Meshes")

        link3D_names = dict(
            link0='link_0',
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',
        )

        super().__init__(
            links,
            link3D_names,
            name='KUKA_KR6',
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        # End-effector tool
        self.tool_length = 0.08
        self.tool = SE3(0, 0, self.tool_length)

    def _create_DH(self):
        a =      [0.025, 0.455, 0.035, 0.00, 0.00, 0.0]
        d =      [0.400, 0.000, 0.000, 0.42, 0.00, 0.0]
        alpha =  [ pi/2, 0.000,  pi/2,  pi/2, -pi/2, 0.0]
        offset = [0.000,  pi/2,  pi/2,  0.00,  0.00, 0.0]
        qlim = [[-pi, pi]] * 6
        qlim[3] = [-pi/4, pi/4]

        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i]) for i in range(6)]
        return links

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.55, 1.35, 1.40], [0.45, 0.0, 0.25])

        self.base = SE3(0.50, 0.00, 0.0)
        self.add_to_env(env)

        q1 = [0.0, -40*pi/180,  40*pi/180,  0.0,  0.0,  0.0]
        q2 = [0.0,   -pi/2,     -pi/2,      pi/2, pi/2, 0.0]

        RobotCollision(env, self, q1, q2,             
            volume=(0.40, 0.50, 0.50),
            center=(0.95, 0.00, 0.25),
            colour=(0.0, 1.0, 0.0, 0.45)
        )
            
        print("Test complete.")
        env.hold()

if __name__ == "__main__":
    robot = KR6_Robot()
    robot.test()

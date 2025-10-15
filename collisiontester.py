#!/usr/bin/env python3
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialgeometry import Cuboid
import spatialmath.base as spb

#Robot Imports
import os
from math import pi
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialmath import SE3


class CollisionDetector:
    def __init__(self, env: swift.Swift, volume, center, colour):
        self.env = env

        # Visual Box for environment
        self.cuboid = Cuboid(scale=volume, color=colour)
        self.cuboid.T = spb.transl(center)
        env.add(self.cuboid)

        #Box bounds
        half = np.array(volume) * 0.5
        self.box_min = center - half
        self.box_max = center + half

    def intersection(point, next_point, box_min, box_max):
        eps = 1e-12 # Number very close to 0 for future parallel calcs
        d = next_point - point #Defining vector between points
        tmin, tmax = 0.0, 1.0

        for i in range(3): #For all 3 axes
            if abs(d[i]) < eps: #Checking if point vector is parallel to box slab therefore meaning it hit it
                if point[i] < box_min[i] or point[i] > box_max[i]:
                    return False #If the points are outside of the box though the box has not been hit
            else:
                inv = 1.0 / d[i] #Creating inverse for future use
                t1 = (box_min[i] - point[i]) * inv #Plane the robot will hit
                t2 = (box_max[i] - point[i]) * inv #Plane the robot will exit from if it passed through the box
                if t1 > t2: 
                    t1, t2 = t2, t1 #Making sure the closer plane is t1 as that is the one the robot will hit
                #Updating global planes if the robot is more likely to hit these new planes
                if t1 > tmin: tmin = t1
                if t2 < tmax: tmax = t2
                if tmin > tmax: #If this occurs it means the robot has passed the box and not hit it
                    return False

        #If tmin is before the current point or tmin is past the next point then it hasn't hit it yet
        if tmin < 0.0 or tmin > 1.0:
            return False

        return True #The robot has hit something

    def check_pose(self, robot, q) -> bool:
        hit_any = False

        A = np.array(robot.fkine_all(q).A) # Array of all frame transforms of robot

        # Extract positions in shape (N,3)
        # A is (N,4,4): pose i is A[i]; translation is A[i,:3,3]
        P = A[:, :3, 3] #Get joint origins for all the frames
        for i in range(P.shape[0] - 1): #For the range of all the joint origin points
            p0 = P[i] #Starting point for frame
            p1 = P[i + 1] #Next point ending the frame
            hit = CollisionDetector.intersection(p0, p1, self.box_min, self.box_max) #Checks if hit
            if hit:
                hit_any = True
        return hit_any #Says if the robot hit the box


class RobotCollision():
    def __init__(self, env, robot, q_start, q_finished, volume, center, colour=(0.0, 1.0, 0.0, 0.45),steps = 80):
        detector = CollisionDetector(env, volume, center, colour)
        q_traj = rtb.jtraj(q_start, q_finished, steps).q

        collided = False
        announced = False

        for k, q in enumerate(q_traj):
            if not collided:
                if detector.check_pose(robot, q):
                    collided = True
                    print(f"[Collision Detection]: Collision detected at step {k+1}/{len(q_traj)}")
                else:
                    robot.q = q  #Move the robot
            if collided and not announced:
                announced = True
                print("[Collision Detection]: Robot collided with object, stopping further motion.")

            env.step(0.016) #Update environment


class KR6_Robot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()
        qtest = [0, -pi/2, 0, 0, 0, 0]

        qtest_transforms = [
            spb.transl(+0.000, 0, 0.000),
            spb.transl(+0.000, 0, 0.400),
            spb.transl(+0.025, 0, 0.400),
            spb.transl(+0.480, 0, 0.400),
            spb.transl(+0.480, 0, 0.435),
            spb.transl(+0.900, 0, 0.435),
            spb.transl(+0.980, 0, 0.435),
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

        self.tool_length = 0.08
        self.tool = SE3(0, 0, self.tool_length)

    def _create_DH(self):
        a =      [0.025, 0.455, 0.035, 0.00, 0.00, 0.0]
        d =      [0.400, 0.000, 0.000, 0.42, 0.00, 0.0]
        alpha =  [ pi/2, 0.000,  pi/2,  pi/2, -pi/2, 0.0]
        offset = [0.000,  pi/2,  pi/2,  0.00,  0.00, 0.0]
        qlim = [[-pi, pi]] * 6
        qlim[3] = [-pi/4, pi/4]
        return [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i]) for i in range(6)]

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.55, 1.35, 1.40], [0.45, 0.0, 0.25])

        self.base = SE3(0.50, 0.00, 0.0)
        self.add_to_env(env)

        q1 = [pi/2,-pi/2,  0.0, 0.0, 0.0,0.0] #Start
        q2 = [ 0.0,-pi/2,-pi/2,pi/2,pi/2,0.0] #End

        RobotCollision(
            env, self, q1, q2,
            volume=(0.40, 0.50, 0.50),
            center=(0.95, 0.00, 0.25),
            colour=(0.0, 1.0, 0.0, 0.45),
            steps = 50
        )

        print("Test complete.")
        env.hold()


if __name__ == "__main__":
    robot = KR6_Robot()
    robot.test()

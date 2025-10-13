#!/usr/bin/env python
import swift
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid
from spatialmath import SE3
import time
from math import pi

class FingerDH(rtb.DHRobot):
    def __init__(self, pose, L1=0.08, L2=0.06, radius=0.01):
        """
        2-DOF robotic finger model with two revolute joints.
        L1, L2: lengths of phalanges
        radius: cylinder radius for visualization
        """
        self.L1 = L1
        self.L2 = L2
        self.radius = radius
        

        # Create DH links
        links = self._create_DH()

        # Initialize DHRobot
        super().__init__(links, name='finger')

        self.base = pose

        # Create visual geometry
        self._create_geometry()

    def _create_DH(self):
        """
        Create DH model for the 2-link finger.
        - Joint 1 at base (pivot at bottom)
        - Joint 2 at the connection between proximal and distal phalanges
        """
        L1, L2 = self.L1, self.L2
        links = [
            rtb.RevoluteDH(a=L1, alpha=0, d=0, qlim=[-pi/2, pi/2]),
            rtb.RevoluteDH(a=L2/2, alpha=0, d=0, qlim=[-pi/2, pi/2])
        ]
        return links

    def _create_geometry(self, ):
        """
        Attach cylinder geometries to the robot links for visualization.
        """
        L1, L2 = self.L1, self.L2
        r = self.radius

        # Proximal phalange
        cyl1 = Cylinder(radius=r, length=L1, color=[0.6, 0.6, 1.0])
        # Position so its base aligns with joint axis
        cyl1.T = SE3(-L1/2, 0, 0) * SE3.RPY(0,pi/2,0)

    
        # Distal phalange
        cyl2 = Cylinder(radius=r * 0.9, length=L2, color=[0.6, 1.0, 0.6])
        cyl2.T = SE3(0, 0, 0) * SE3.RPY(0,pi/2,0)


        cbd2 = Cuboid((0.05,0.01,0.018), color = [1,0.6,0.6])
        cbd2.T = SE3(0.005, 0.008, 0)

        # Attach to corresponding links
        self.links[0].geometry = [cyl1]
        self.links[1].geometry = [cyl2, cbd2]

    def actuate(self, position):

        q_open =[0.0, 0.0]
        q_open_r =[0.0, 0.0]
        q_close = [pi/4, pi/4]
        q_close_r = [pi/4, pi/4]

        if position == 'open':
            q_target = q_open
        elif position == 'close':
            q_target = q_close
        else:
            print("No position selected")

        q_start = self.q
        traj = rtb.jtraj(q_start, q_target, 50).q

        for q in traj:
            self.q = q
            env.step(0.02)
            time.sleep(0.01)


    def test(self):
        """
        Launch Swift and animate open-close motion.
        """
        

        q_open = [0.0, 0.0]
        q_close = [pi/4, pi/4]

        traj = rtb.jtraj(q_open, q_close, 50).q
        for q in traj:
            self.q = q
            env.step(0.02)

        time.sleep(0.3)

        traj = rtb.jtraj(q_close, q_open, 50).q
        for q in traj:
            self.q = q
            env.step(0.02)
        


if __name__ == "__main__":
    finger_l = FingerDH(SE3(0,0.05,0.1) * SE3.Rx(pi))
    finger_r = FingerDH(SE3(0,0,0.1))
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([0.5, 0.3, 0.3], [0, 0, 0])
    env.add(finger_l)
    env.add(finger_r)
    finger_l.test()
    finger_r.actuate("open")
    finger_r.actuate("close")
    env.hold

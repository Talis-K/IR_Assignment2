import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from math import pi

class Load(DHRobot3D):
    def __init__(self):
       
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(
            link0='base_link',
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',
            link7='link_7'
        ) # Limit to number of links (3 currently)

        # A joint config and the 3D object transforms to match that config
        qtest = [0,0,0,0,0,0,0]
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a1
            spb.transl(0, 0, 0) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a2
            spb.transl(0, 0, 0.36) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a3
            spb.transl(0, 0, 0.36) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a4
            spb.transl(0, 0, 0.78) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a5
            spb.transl(0, 0, 0.78) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a6
            spb.transl(0, 0, 1.18) @ spb.trotz(0) @ spb.trotx(0),  # Joint_a7
            spb.transl(0, 0, 1.18) @ spb.trotz(0) @ spb.trotx(0),  # ee
        ]  # Truncate to match number of links

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "meshes/visual")
        super().__init__(links, link3D_names, name='LBR', link3d_dir=mesh_dir, qtest=qtest, qtest_transforms=qtest_transforms)
        self.q = qtest

        self.tool_length = 0.12
        self.addconfiguration("tool_length", self.tool_length)

    def _create_DH(self):
        a = [0, 0, 0, 0, 0, 0, 0] 
        d = [0.36, 0, 0.42, 0, 0.4, 0, 0] 
        alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0]
        qlim = [[-pi, pi]] * 7
        links = []
        for i in range(7):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i])
            links.append(link)
        return links

    def test(self):

        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, 1.3, 1.4], [0, 0, pi/8])
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0)  # Offset robot for visibility
        self.add_to_env(env)

        q_goal = [pi/2, pi/2, pi/2, pi/2, pi/2 ,pi/2, pi/2]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
            print(f"Joint angles: {q}")
        time.sleep(3)
        env.hold()

if __name__ == "__main__":
    r = Load()
    r.test()
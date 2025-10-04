import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from math import pi

class KUKA_KR6_900(DHRobot3D):
    def __init__(self):
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(link0='base_kr6',
                            link1='link1_kr6',
                            link2='link2_kr6',
                            link3='link3_kr6',
                            link4='link4_kr6',
                            link5='link5_kr6',
                            link6='link6_kr6')

        # Example joint configuration
        qtest = [0, -pi/2, pi/2, 0, pi/2, 0]

        # Compute kinematic transforms (for reference)
        self.qtest_transforms = [(SE3(0, 0, 0)).A]  # Base
        T = SE3()
        for i in range(len(links)):
            T = T * links[i].A(qtest[i])
            self.qtest_transforms.append(T.A)

        # Override with adjustments for mesh alignment
        self.qtest_transforms[1] = (SE3(0, 0, 0.4)).A  # Link1: vertical offset
        self.qtest_transforms[2] = (SE3(0.025, 0, 0.4) * SE3.Ry(-pi/2)).A  # Link2: horizontal link, rotated
        self.qtest_transforms[3] = (SE3(0.025, 0, 0.855) * SE3.Ry(-pi/2) * SE3.Rx(-pi)).A  # Link3: adjusted offset, keep flip
        self.qtest_transforms[4] = (SE3(0.055, 0, 0.85) * SE3.Ry(-pi/2) * SE3.Rx(pi)).A  # Link4: counter flip
        self.qtest_transforms[5] = (SE3(0.055, 0, 1.27) * SE3.Ry(-pi/2)).A  # Link5: small offset
        self.qtest_transforms[6] = (SE3(0.055, 0, 1.35) * SE3.Ry(-pi/2)).A  # Link6: final small offset

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name='KUKA_KR6_900',
                        link3d_dir=os.path.join(current_path, "meshes", "visual"),
                        qtest=qtest, qtest_transforms=self.qtest_transforms)
        self.q = qtest

        print("End-effector pose at qtest:")
        print(self.fkine(qtest))
        self.check_qtest_transforms()

    def _create_DH(self):
        a = [0, 0.025, 0.455, 0, 0.420, 0.080]
        d = [0.4, 0, 0, 0.035, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        qlim = [[-pi, pi],
                [-pi/2, pi/2],
                [-pi/2, pi/2],
                [-pi, pi],
                [-pi/2, pi/2],
                [-pi/2, pi/2]]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i])
            links.append(link)
        return links

    def check_qtest_transforms(self):
        print("Checking qtest_transforms against DH kinematics at qtest:")
        qtest = [0, -pi/2, pi/2, 0, pi/2, 0]
        T = SE3()
        for i in range(len(self.links)):
            T = T * self.links[i].A(qtest[i])
            print(f"Link {i+1} FK transform:\n{T.A}")
            print(f"Link {i+1} qtest_transform:\n{self.qtest_transforms[i+1]}")
            print("---")

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.base = SE3(0.5, 0.5, 0)
        self.add_to_env(env)

        for i in range(self.n):
            q_goal = self.q.copy()
            q_goal[i] -= pi/6
            print(f"Moving joint {i+1} to {q_goal}")
            qtraj = rtb.jtraj(self.q, q_goal, 50).q
            for q in qtraj:
                self.q = q
                T = SE3()
                for j in range(len(self.links)):
                    T = T * self.links[j].A(q[j])
                    print(f"Step {i+1}, Link {j+1} transform:\n{T.A}")
                env.step(0.1)
                time.sleep(0.2)
            self.q = q_goal
            time.sleep(2)
        env.hold()

if __name__ == "__main__":
    r = KUKA_KR6_900()
    r.test()
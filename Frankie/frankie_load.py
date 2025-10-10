import numpy as np
from roboticstoolbox.robot.Robot import Robot


class Load(Robot):
    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "franka_description/robots/frankieOmni_arm.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Franka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
            # gripper_links=elinks[9]
        )

        # self.qdlim = np.array([
        #     2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.qr = np.array([0, -0.3, 0, -1.9, 0, 1.5, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__": 
    robot = Load()
    print(robot)
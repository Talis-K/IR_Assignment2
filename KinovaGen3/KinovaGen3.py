import numpy as np
from roboticstoolbox.robot.Robot import Robot


class KinovaGen3(Robot):
    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "kortex_description/robots/gen3.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Kinova",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qr = np.array([np.pi, -0.3, 0, -1.6, 0, -1.0, np.pi / 2])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__": 
    robot = KinovaGen3()
    print(robot)
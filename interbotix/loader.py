import numpy as np
from roboticstoolbox.robot.Robot import Robot


class Load(Robot):
    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "interbotix_descriptions/urdf/vx300s.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Interbotix",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0, 0])
        self.qz = np.zeros(9)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = Load()
    print(robot)
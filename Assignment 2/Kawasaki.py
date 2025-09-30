import swift
from spatialmath import SE3
from spatialgeometry import Mesh  # comes with Robotics Toolbox ecosystem

# --- Swift scene ---
env = swift.Swift()
env.launch(realtime=True)

#
kawasaki_mesh = Mesh(
    filename="RS020N-A001.stl",     # <— your converted mesh
    pose=SE3(0.2, -0.6, 0) * SE3.Rz(1.57),   
)
env.add(kawasaki_mesh)

env.hold()

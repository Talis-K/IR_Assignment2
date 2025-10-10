from swift import Swift
import loader as px100

# Load the Kinova Gen3 (7 DOF)
robot = px100.Load()

# Launch Swift
env = Swift()
env.launch()

robot.q = robot.qz  # zero position

# Animate to some random config
import numpy as np
q_rand = robot.qr
robot.q = q_rand

# Add robot at zero pose
env.add(robot)
env.hold()  # keep the viewer alive


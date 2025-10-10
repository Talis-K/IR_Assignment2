from swift import Swift
import frankie_load as frankie

# Load the Kinova Gen3 (7 DOF)
robot = frankie.Load()

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


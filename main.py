import numpy as np
from PyFlyt.core import Aviary

# Define starting positions and orientations
start_pos = np.array([[0.0, 0.0, 0.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# Instantiate aviary with rgb camera enabled
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="quadx",
    drone_options={"use_camera": True},
)

# Add a red sphere at position [2, 0, 1]
sphere_visual_id = env.createVisualShape(
    shapeType=env.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1]  # Red color
)
sphere_id = env.createMultiBody(
    baseMass=0,  # Static object
    baseVisualShapeIndex=sphere_visual_id,
    basePosition=[2, 0, 1],
)

# Define control mode
env.set_mode(7)

# Define a setpoint
setpoint = np.array([1.0, 0.0, 0.0, 1.0])
env.set_setpoint(0, setpoint)

# Step the physics
for i in range(10000):
    obs_dict = env.step()

env.close()

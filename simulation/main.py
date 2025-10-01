import numpy as np
from PyFlyt.core import Aviary
import cv2

start_pos = np.array([[0.0, 0.0, 0.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="quadx",
    drone_options={"use_camera": True},
)

sphere_visual_id = env.createVisualShape(
    shapeType=env.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1]
)
sphere_id = env.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=sphere_visual_id,
    basePosition=[2, 0, 1],
)

env.set_mode(0)

setpoint = np.array([0.0, 0.0, 1, 0.3])
env.set_setpoint(0, setpoint)

for i in range(10000):
    obs_dict = env.step()
    frame = env.drones[0].rgbaImg
    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    sensors = env.state(0)
    gyro_data = sensors[0]
    attitude_data = sensors[1] * 57.3  # Convert radians to degrees
    # print(
    #     round(attitude_data[0], 3),
    #     round(attitude_data[1], 3),
    #     round(attitude_data[2], 3),
    # )
    # print(round(gyro_data[0], 3), round(gyro_data[1], 3), round(gyro_data[2], 3))
    print(env.aux_state(0))
    cv2.waitKey(1)
    cv2.imshow("Camera View", frame)

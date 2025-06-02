import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from PyFlyt.core import Aviary
from stable_baselines3.common.env_checker import check_env


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        # [<-1, 1>: roll, <-1, 1>: pitch, <-1, 1>: yaw, <-1, 1>: throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, target_pos_x, target_pos_y]
        low_bounds = np.array([-np.inf] * 6 + [0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([np.inf] * 6 + [1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(8,), dtype=np.float32
        )

        start_pos = np.array([[0.0, 0.0, 0.0]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=True,
            drone_type="quadx",
            drone_options={"use_camera": True},
        )

        sphere_visual_id = self.env.createVisualShape(
            shapeType=self.env.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1]
        )
        self.env.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual_id,
            basePosition=[2, 0, 1],
        )

        self.env.set_mode(0)

    def create_observation(self, sensors, frame):
        # Extract gyro and accel data from sensors
        gyro_data = sensors[0]
        accel_data = sensors[2]

        return np.concatenate((gyro_data, accel_data, [0.5, 0.5]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.env.reset()
        sensors = self.env.state(0)
        frame = self.env.drones[0].rgbaImg
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)

        obs = self.create_observation(sensors, frame)
        return obs, {}

    def step(self, action):
        self.env.set_setpoint(0, action)
        obs_dict = self.env.step()

        sensors = self.env.state(0)
        frame = self.env.drones[0].rgbaImg
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)

        # TODO compute reward
        reward = 0

        done = False
        info = {}

        obs = self.create_observation(sensors, frame)
        return obs, reward, done, False, info

    def render(self, mode="human"):
        frame = self.env.drones[0].rgbaImg
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)

    def close(self):
        pass


if __name__ == "__main__":
    env = DroneEnv()
    check_env(env)
    print("Environment is valid!")

    obs, _ = env.reset()

    for _ in range(400):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print("Observation:", obs, "Reward:", reward)

        env.render()

    env.close()

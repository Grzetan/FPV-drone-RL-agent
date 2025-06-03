import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from PyFlyt.core import Aviary
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.target_pos = np.array([0, 0, 3], dtype=np.float32)

        # [<-1, 1>: roll, <-1, 1>: pitch, <-1, 1>: yaw, <-1, 1>: throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [gyro_xyz, attitude_xyz, target_distance, previous_action_yrpt]
        low_bounds = np.array(
            [-np.inf] * 3 + [-1] * 3 + [-np.inf] + [-1] * 4, dtype=np.float32
        )
        high_bounds = np.array(
            [np.inf] * 3 + [1] * 3 + [np.inf] + [1] * 4, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(11,), dtype=np.float32
        )

        start_pos = np.array([[0.0, 0.0, 0.3]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=True,
            drone_type="quadx",
            drone_options={"use_camera": False},
        )

        self.env.set_mode(0)

        self.action = np.zeros(4, dtype=np.float32)

        self.max_steps = 300
        self.step_count = 0
        self.flight_dome_size = 20.0
        self.info = {}

    def create_observation(self, sensors):
        gyro_data = sensors[0]
        attitude_data = sensors[1]
        target_distance = np.linalg.norm(sensors[3] - self.target_pos)

        return np.concatenate(
            (gyro_data, attitude_data, target_distance, self.action),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        sensors = self.env.state(0)
        self.action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        obs = self.create_observation(sensors)
        return obs, {}

    def step(self, action):
        self.action = action.copy()

        self.env.set_setpoint(0, action)
        self.env.step()

        sensors = self.env.state(0)

        self.reward = -0.1

        if self.step_count > self.max_steps:
            self.truncation |= True

        # if anything hits the floor, basically game over
        if np.any(self.env.contact_array[self.env.planeId]):
            self.reward = -100.0
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(sensors[3]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

        linear_distance = np.linalg.norm(
            self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
        )
        # Negative Reward For High Yaw rate, To prevent high yaw while training
        yaw_rate = abs(sensors[0][2])  # Assuming z-axis is the last component
        yaw_rate_penalty = 0.01 * yaw_rate**2  # Add penalty for high yaw rate
        self.reward -= (
            yaw_rate_penalty  # You can adjust the coefficient (0.01) as needed
        )

        # how far are we from 0 roll pitch
        angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

        self.reward -= linear_distance + angular_distance
        self.reward += 1.0

        self.step_count += 1
        obs = self.create_observation(sensors)
        return obs, self.reward, self.termination, self.truncation, self.info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # Initialize environment with rendering disabled for faster training
    env = DroneEnv()
    check_env(env)
    print("Environment is valid!")

    # Set up logging

    # Create log directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with monitor for logging
    env = Monitor(env, log_dir)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Save model every 1000 steps
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="drone_model",
        verbose=1,
    )

    # Initialize and train model with logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")
    model.learn(total_timesteps=25_000, callback=checkpoint_callback)
    model.save(f"{log_dir}/final_drone_model")
    print("Model trained and saved!")

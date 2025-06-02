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

        # Observation space: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, target_pos_x, target_pos_y]
        low_bounds = np.array([-np.inf] * 9, dtype=np.float32)
        high_bounds = np.array([np.inf] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(9,), dtype=np.float32
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

    def create_observation(self, sensors):
        # Extract gyro and accel data from sensors
        gyro_data = sensors[0]
        accel_data = sensors[2]
        global_position = sensors[3] - self.target_pos

        return np.concatenate(
            (gyro_data, accel_data, global_position), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        sensors = self.env.state(0)

        obs = self.create_observation(sensors)
        return obs, {}

    def step(self, action):
        self.env.set_setpoint(0, action)
        obs_dict = self.env.step()

        sensors = self.env.state(0)

        # Compute reward based on distance to target position
        current_pos = sensors[3]
        distance_to_target = np.linalg.norm(current_pos - self.target_pos)

        # Negative distance as reward (closer = higher reward)
        reward = -distance_to_target

        # Bonus for being very close to target
        if distance_to_target < 0.5:
            reward += 5.0

        # Check if done conditions are met
        done = False

        # Condition 1: Drone crashed (too low or inverted)
        if current_pos[2] < 0.1:  # Height near ground
            done = True
            reward -= 10.0  # Penalty for crashing

        # Condition 2: Drone too far from target
        if distance_to_target > 10.0:  # Far away from target
            done = True
            reward -= 5.0  # Penalty for going out of bounds

        info = {}

        obs = self.create_observation(sensors)
        return obs, reward, done, False, info

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

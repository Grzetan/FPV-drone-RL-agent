import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os


class DroneEnv(gym.Env):
    def __init__(self, render=False):
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
            render=render,
            drone_type="quadx",
            drone_options={"use_camera": False},
        )

        self.env.set_mode(0)

        self.action = np.zeros(4, dtype=np.float32)

        self.termination = False
        self.truncation = False
        self.max_steps = 300
        self.step_count = 0
        self.flight_dome_size = 20.0
        self.info = {}

    def create_observation(self, sensors):
        gyro_data = sensors[0]
        attitude_data = sensors[1]
        target_distance = [np.linalg.norm(sensors[3] - self.target_pos)]

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

        for _ in range(5):
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
    models_dir = "models"
    logs_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = DroneEnv()
    env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # Train the model
    total_timesteps = 20000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(f"{models_dir}/ppo_quadx_hover")

    # # Test the trained model
    env = DroneEnv()
    obs, _ = env.reset()
    term, trunc = False, False

    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)

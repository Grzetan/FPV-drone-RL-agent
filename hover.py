import cv2
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

        # Observation space: [gyro_xyz, attitude_xyz, target_distance, previous_action_yrpt, sphere_center_xy]
        low_bounds = np.array(
            [-np.inf] * 3 + [-1] * 3 + [-np.inf] + [-1] * 4 + [-1] * 2, dtype=np.float32
        )
        high_bounds = np.array(
            [np.inf] * 3 + [1] * 3 + [np.inf] + [1] * 4 + [1] * 2, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(13,), dtype=np.float32
        )

        start_pos = np.array([[0.0, 0.0, 0.0]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=render,
            drone_type="quadx",
            drone_options={"use_camera": True, "camera_angle_degrees": -25},
        )

        self.add_sphere()

        self.env.set_mode(0)

        self.action = np.zeros(4, dtype=np.float32)

        self.termination = False
        self.truncation = False
        self.max_steps = 300
        self.step_count = 0
        self.flight_dome_size = 20.0
        self.info = {}

    def add_sphere(self):
        self.sphere_visual_id = self.env.createVisualShape(
            shapeType=self.env.GEOM_SPHERE, radius=1, rgbaColor=[1, 0, 0, 1]
        )
        self.sphere_id = self.env.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.sphere_visual_id,
            basePosition=[10, 0, 15],
        )

    def detect_red_sphere_center(self, rgba_image):
        rgb_image = rgba_image[:, :, :3]
        red_channel = rgb_image[:, :, 0].astype(np.float32)
        green_channel = rgb_image[:, :, 1].astype(np.float32)
        blue_channel = rgb_image[:, :, 2].astype(np.float32)

        mask = (
            (red_channel >= 3 * green_channel)
            & (red_channel >= 3 * blue_channel)
            & (red_channel > 50)
        ).astype(np.uint8) * 255

        red_pixels = np.where(mask > 0)

        if len(red_pixels[0]) > 0:
            min_y, max_y = np.min(red_pixels[0]), np.max(red_pixels[0])
            min_x, max_x = np.min(red_pixels[1]), np.max(red_pixels[1])

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            height, width = rgb_image.shape[:2]
            normalized_x = (2.0 * center_x / width) - 1.0
            normalized_y = (2.0 * center_y / height) - 1.0

            return np.array([normalized_x, normalized_y], dtype=np.float32)

        return np.array([-1, -1], dtype=np.float32)  # ? What should be here?

    def create_observation(self, sensors):
        gyro_data = sensors[0]
        attitude_data = sensors[1]
        target_distance = [np.linalg.norm(sensors[3] - self.target_pos)]

        # Get camera image and detect red sphere center
        rgba_image = self.env.drones[0].rgbaImg
        sphere_center = self.detect_red_sphere_center(rgba_image)

        return np.concatenate(
            (gyro_data, attitude_data, target_distance, self.action, sphere_center),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.add_sphere()
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

    def test_env(self):
        print("Controls: UP arrow = throttle 0.5, any other key = throttle 0.0")
        print("Press 'q' to quit, 'ESC' to exit")

        for i in range(10000):
            # Keyboard control for throttle
            key = cv2.waitKey(1) & 0xFF

            # Set throttle based on key input
            if (
                key == 82 or key == 0
            ):  # UP arrow key (different codes on different systems)
                throttle = 1
            else:
                throttle = 0.0

            # Exit conditions
            if key == ord("q") or key == 27:  # 'q' or ESC key
                break

            # Set action with controlled throttle [roll, pitch, yaw, throttle]
            action = np.array([0.0, 0.0, 0.0, throttle])
            self.env.set_setpoint(0, action)

            self.env.step()
            rgba_frame = self.env.drones[0].rgbaImg

            # Get full observation space
            sensors = self.env.state(0)
            observation = self.create_observation(sensors)

            # Print observation space with labels (less frequently to reduce spam)
            if i % 10 == 0:  # Print every 10 iterations instead of every iteration
                print(f"Iteration {i} - Throttle: {throttle}")
                print(
                    f"  Gyro [x,y,z]: [{observation[0]:.3f}, {observation[1]:.3f}, {observation[2]:.3f}]"
                )
                print(
                    f"  Attitude [x,y,z]: [{observation[3]:.3f}, {observation[4]:.3f}, {observation[5]:.3f}]"
                )
                print(f"  Target distance: {observation[6]:.3f}")
                print(
                    f"  Previous action [r,p,y,t]: [{observation[7]:.3f}, {observation[8]:.3f}, {observation[9]:.3f}, {observation[10]:.3f}]"
                )
                print(
                    f"  Sphere center [x,y]: [{observation[11]:.3f}, {observation[12]:.3f}]"
                )
                print("-" * 50)

            # Convert to BGR for display
            frame = cv2.cvtColor(rgba_frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)
            cv2.imshow("Camera View", frame)


if __name__ == "__main__":
    # Just display the environment without training
    env = DroneEnv(render=True)  # Enable rendering
    obs, _ = env.reset()

    print("Environment loaded. Drone will do nothing - just displaying the simulation.")
    print("Press Ctrl+C to exit.")

    env.test_env()

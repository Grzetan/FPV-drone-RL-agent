import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
import pygame
import time
import threading
from queue import Queue
from stable_baselines3.common.env_checker import check_env


# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
import os


class RadioMasterJoystick:
    """
    RadioMaster Pocket joystick controller for drone control.
    Maps axis values from RadioMaster Pocket to drone controls.
    """
    
    def __init__(self, joystick_index=0):
        """
        Initialize the RadioMaster joystick controller.
        
        Args:
            joystick_index: Index of the joystick device (default: 0)
        """
        pygame.init()
        
        # Check for connected joysticks
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No joystick detected. Please connect your RadioMaster Pocket.")
        
        if joystick_index >= joystick_count:
            raise RuntimeError(f"Joystick index {joystick_index} not available. Found {joystick_count} joystick(s).")
        
        # Initialize the joystick
        self.joystick = pygame.joystick.Joystick(joystick_index)
        self.joystick.init()
        
        print(f"Initialized RadioMaster Joystick: {self.joystick.get_name()}")
        print(f"Number of Axes: {self.joystick.get_numaxes()}")
        print(f"Number of Buttons: {self.joystick.get_numbuttons()}")
        
        # Control mappings
        self.axis_mappings = {
            'roll': 0,      # Axis 0: Roll (left/right)
            'pitch': 1,     # Axis 1: Pitch (forward/backward)
            'throttle': 2,  # Axis 2: Throttle (up/down)
            'yaw': 3        # Axis 3: Yaw (rotation)
        }
        
        # Control scaling factors
        self.scale_factors = {
            'roll': 1.0,    # Full range for roll
            'pitch': 1.0,   # Full range for pitch
            'throttle': 1.0, # Full range for throttle
            'yaw': 1.0      # Full range for yaw
        }
        
        # Dead zone to prevent small movements
        self.dead_zone = 0.05
        
    def get_controls(self):
        """
        Get current control values from the RadioMaster Pocket.
        
        Returns:
            dict: Dictionary with control values {'roll': float, 'pitch': float, 'throttle': float, 'yaw': float}
        """
        # Process pygame events
        pygame.event.pump()
        
        controls = {}
        
        for control_name, axis_index in self.axis_mappings.items():
            if axis_index < self.joystick.get_numaxes():
                # Get raw axis value (-1 to 1)
                raw_value = self.joystick.get_axis(axis_index)
                
                # Apply dead zone
                if abs(raw_value) < self.dead_zone:
                    raw_value = 0.0
                
                # Apply scaling
                scaled_value = raw_value * self.scale_factors[control_name]
                
                # Clamp to [-1, 1] range
                controls[control_name] = np.clip(scaled_value, -1.0, 1.0)
            else:
                controls[control_name] = 0.0
        
        return controls
    
    def get_action_array(self):
        """
        Get control values as numpy array for drone control.
        
        Returns:
            np.array: Array with [roll, pitch, yaw, throttle] values in [-1, 1] range
        """
        controls = self.get_controls()
        return np.array([
            controls['roll'],
            controls['pitch'], 
            controls['yaw'],
            controls['throttle']
        ], dtype=np.float32)
    
    def set_scale_factor(self, control_name, scale):
        """
        Set scaling factor for a specific control.
        
        Args:
            control_name: Name of the control ('roll', 'pitch', 'throttle', 'yaw')
            scale: Scaling factor (float)
        """
        if control_name in self.scale_factors:
            self.scale_factors[control_name] = scale
        else:
            raise ValueError(f"Invalid control name: {control_name}")
    
    def set_dead_zone(self, dead_zone):
        """
        Set dead zone for all controls.
        
        Args:
            dead_zone: Dead zone value (0.0 to 1.0)
        """
        self.dead_zone = np.clip(dead_zone, 0.0, 1.0)
    
    def print_status(self):
        """
        Print current control values for debugging.
        """
        controls = self.get_controls()
        print(f"Roll: {controls['roll']:6.3f} | Pitch: {controls['pitch']:6.3f} | "
              f"Throttle: {controls['throttle']:6.3f} | Yaw: {controls['yaw']:6.3f}", end="\r")
    
    def cleanup(self):
        """
        Clean up pygame resources.
        """
        pygame.quit()


class DroneEnv(gym.Env):
    def __init__(self, render=False):
        super(DroneEnv, self).__init__()
        # [<-1, 1>: roll, <-1, 1>: pitch, <-1, 1>: yaw, <-1, 1>: throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [attitude_xyz (3), sphere_center_xy (2), previous_raw_action (4)]
        low_bounds = np.array([-1] * 3 + [-1] * 2 + [-1] * 4, dtype=np.float32)
        high_bounds = np.array([1] * 3 + [1] * 2 + [1] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(9,), dtype=np.float32
        )

        start_pos = np.array([[0.0, 0.0, 0.0]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=render,
            drone_type="quadx",
            drone_options={
                "use_camera": True,
                "camera_angle_degrees": -25,
                "model_dir": "./drone_models",  # Path to your drone models directory, there you can change the mass, thrust, model etc.
                "drone_model": "cf2x",
            },
        )

        self.add_sphere()
        # self.add_reference_structures()

        self.env.set_mode(0)

        self.action = np.zeros(4, dtype=np.float32)
        self.last_raw_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_raw_action = np.zeros(4, dtype=np.float32)

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

    def add_reference_structures(self):
        """
        Add various reference structures to help with spatial awareness during flight.
        """
        
        # Building-like structures
        self.add_buildings()
        
        # Colored reference objects
        self.add_reference_objects()

    def add_buildings(self):
        """Add building-like structures for reference."""
        buildings = [
            # Building 1: Low building
            {"pos": [8, 8, 0], "size": [2, 2, 2], "color": [0.7, 0.4, 0.4, 1.0]},
            # Building 2: Medium building
            {"pos": [-8, 8, 0], "size": [1.5, 1.5, 4], "color": [0.4, 0.7, 0.4, 1.0]},
            # Building 3: Tall building
            {"pos": [8, -8, 0], "size": [1, 1, 6], "color": [0.4, 0.4, 0.7, 1.0]},
            # Building 4: Wide building
            {"pos": [-8, -8, 0], "size": [3, 1, 2.5], "color": [0.7, 0.7, 0.4, 1.0]},
        ]
        
        for building in buildings:
            pos = building["pos"]
            size = building["size"]
            color = building["color"]
            
            # Create building box
            building_shape_id = self.env.createVisualShape(
                shapeType=self.env.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                rgbaColor=color
            )
            self.env.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=building_shape_id,
                basePosition=[pos[0], pos[1], size[2]/2]
            )

    def add_reference_objects(self):
        """Add colored reference objects at different locations."""
        objects = [
            # Red cube
            {"pos": [3, 0, 1], "shape": "box", "size": [0.5, 0.5, 0.5], "color": [1.0, 0.0, 0.0, 1.0]},
            # Green sphere
            {"pos": [-3, 0, 1], "shape": "sphere", "size": [0.3], "color": [0.0, 1.0, 0.0, 1.0]},
            # Blue cylinder
            {"pos": [0, 3, 1], "shape": "cylinder", "size": [0.2, 0.8], "color": [0.0, 0.0, 1.0, 1.0]},
            # Yellow pyramid (box approximation)
            {"pos": [0, -3, 1], "shape": "box", "size": [0.4, 0.4, 0.8], "color": [1.0, 1.0, 0.0, 1.0]},
        ]
        
        for obj in objects:
            pos = obj["pos"]
            color = obj["color"]
            
            if obj["shape"] == "box":
                size = obj["size"]
                shape_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_BOX,
                    halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                    rgbaColor=color
                )
            elif obj["shape"] == "sphere":
                radius = obj["size"][0]
                shape_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=color
                )
            elif obj["shape"] == "cylinder":
                radius, height = obj["size"]
                shape_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=color
                )
            
            self.env.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shape_id,
                basePosition=pos
            )

    def create_observation(self, sensors):
        attitude_data = sensors[1]

        # Get camera image and detect red sphere center
        rgba_image = self.env.drones[0].rgbaImg
        sphere_center = self.detect_red_sphere_center(rgba_image)

        return np.concatenate(
            (attitude_data, sphere_center, self.last_raw_action),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.add_sphere()
        # self.add_reference_structures()
        sensors = self.env.state(0)
        self.action = np.zeros(4, dtype=np.float32)
        self.last_raw_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_raw_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.info = {}

        obs = self.create_observation(sensors)
        return obs, {}

    def calculate_reward(self, obs):
        sphere_x = obs[3]
        sphere_y = obs[4]

        self.reward = 0.0

        sphere_visible = not (sphere_x == -1 and sphere_y == -1)
        if sphere_visible:
            self.reward += 1.0

            distance = float(np.sqrt(sphere_x ** 2 + sphere_y ** 2))
            max_dist = np.sqrt(2.0) 
            proximity_score = max(0.0, 1.0 - distance / max_dist)
            self.reward += 0.5 * proximity_score

        action_delta = self.current_raw_action - self.last_raw_action
        delta_norm = float(np.linalg.norm(action_delta))
        self.reward -= 100 * (delta_norm ** 2)

        roll_rad = float(obs[0])
        pitch_rad = float(obs[1])
        self.reward -= 1 * (abs(roll_rad) + abs(pitch_rad))
        # print(f"Reward: {self.reward}")
        return self.reward
        

    def step(self, action):
        # capture current raw action in [-1, 1] range; keep last_raw_action for reward/obs
        self.current_raw_action = action.copy()
        action[0] *= 30.0 
        action[1] *= 30.0
        action[2] *= -30.0
        action[3] = (action[3] + 1) / 2  # Throttle: keep normalized 0-1

        self.action = action.copy()

        self.env.set_setpoint(0, action)

        for _ in range(1):
            self.env.step()

        sensors = self.env.state(0)

        # reset info dict for this step
        self.info = {}

        if self.step_count >= self.max_steps:
            self.truncation = True

        if np.linalg.norm(sensors[3]) > self.flight_dome_size:
            self.info["out_of_bounds"] = True
            self.termination = True

        self.step_count += 1
        obs = self.create_observation(sensors)
        reward = self.calculate_reward(obs)

        # after computing reward/obs, update last_raw_action for next step
        self.last_raw_action = self.current_raw_action.copy()

        return obs, reward, self.termination, self.truncation, self.info

    def test_env_with_joystick(self, joystick_index=0):
        """
        Test the environment using RadioMaster Pocket joystick control.
        
        Args:
            joystick_index: Index of the joystick device (default: 0)
        """
        try:
            # Initialize RadioMaster joystick
            joystick = RadioMasterJoystick(joystick_index)
            
            print("RadioMaster Pocket Joystick Controls:")
            print("  Axis 0: Roll (left/right)")
            print("  Axis 1: Pitch (forward/backward)")
            print("  Axis 2: Throttle (up/down)")
            print("  Axis 3: Yaw (rotation)")
            print("  Press 'q' or ESC to quit")
            print("-" * 50)
            
            iteration = 0
            while True:
                # Get joystick controls
                action = joystick.get_action_array()
                # Apply action to drone
                self.env.set_setpoint(0, action)
                observation, reward, termination, truncation, info = self.step(action)
                rgba_frame = self.env.drones[0].rgbaImg
                
                # Print status every 10 iterations
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}")
                    print(action)
                    print(f"  Attitude [x,y,z]: [{np.degrees(observation[0]):.3f}°, "
                          f"{np.degrees(observation[1]):.3f}°, {np.degrees(observation[2]):.3f}°]")
                    print(f"  Sphere center [x,y]: [{observation[3]:.3f}, {observation[4]:.3f}]")
                    print("-" * 50)
                
                # Display camera view
                frame = cv2.cvtColor(rgba_frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)
                cv2.imshow("RadioMaster Pocket Control", frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                iteration += 1
                
        except RuntimeError as e:
            print(f"Joystick Error: {e}")
            print("Falling back to keyboard controls...")
            self.test_env()
        except KeyboardInterrupt:
            print("\nStopping joystick control...")
        finally:
            try:
                joystick.cleanup()
            except:
                pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Just display the environment without training
    env = DroneEnv(render=True)  # Enable rendering
    check_env(env)
    obs, _ = env.reset()

    print("Environment loaded. Starting RadioMaster Pocket control...")
    print("Press Ctrl+C to exit.")

    # Try joystick control first, fall back to keyboard if not available
    env.test_env_with_joystick()

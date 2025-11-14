import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
import pygame
from stable_baselines3.common.env_checker import check_env
from collections import deque
import random

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
    def __init__(self, render=False, history_length=8):
        super(DroneEnv, self).__init__()
        
        # History configuration
        self.history_length = history_length  # Number of timesteps to track for actions and sphere observations
        
        # [<-1, 1>: roll, <-1, 1>: pitch, <-1, 1>: yaw, <-1, 1>: throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [attitude_xyz (3), angular_velocity_xyz (3), last_N_actions (N*4), last_N_sphere_history (N*3)]
        # Total: 3 + 3 + (N*4) + (N*3) where N = history_length
        # Angular velocity is clipped to [-20, 20] rad/s and normalized to [-1, 1]
        # Sphere history: N entries of [center_x, center_y, size] each (last entry is most recent)
        # Each center_x, center_y, size: [-1, 1]
        action_history_size = self.history_length * 4
        sphere_history_size = self.history_length * 3
        total_size = 3 + 3 + action_history_size + sphere_history_size
        
        low_bounds = np.array(
            [-1] * 3 +  # attitude
            [-1] * 3 +  # angular_velocity
            [-1] * action_history_size +  # actions history
            [-1] * sphere_history_size,  # sphere history
            dtype=np.float32
        )
        high_bounds = np.array(
            [1] * 3 +  # attitude
            [1] * 3 +  # angular_velocity
            [1] * action_history_size +  # actions history
            [1] * sphere_history_size,  # sphere history
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(total_size,), dtype=np.float32
        )

        start_pos = np.array([[0.0, 0.0, 0.0]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        self.physics_hz = 240.0  # PyFlyt simulation runs at 240 Hz
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=render,
            drone_type="quadx",
            physics_hz=self.physics_hz,
            drone_options={
                "use_camera": True,
                "camera_angle_degrees": -30,
                "model_dir": "./drone_models",  # Path to your drone models directory, there you can change the mass, thrust, model etc.
                "drone_model": "cf2x",
            },
        )

        self.add_sphere()
        # self.add_reference_structures()

        self.env.set_mode(0)

        self.action = np.zeros(4, dtype=np.float32)

        # Physics parameters
        self.dt = 1.0 / self.physics_hz  # Time step between physics updates
        
        # Buffer to store last 5 attitude positions for angular velocity calculation
        # Each entry is just the attitude array (roll, pitch, yaw)
        self.attitude_history = deque(maxlen=5)
        # Initialize with zeros
        for _ in range(5):
            self.attitude_history.append(np.zeros(3, dtype=np.float32))
        
        # Buffer to store last N actions
        default_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        self.action_history = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.action_history.append(default_action.copy())
        
        # Buffer to store last N sphere observations (center_x, center_y, size)
        default_sphere = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.sphere_history = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.sphere_history.append(default_sphere.copy())
        
        # Angular velocity clipping and normalization parameters per axis [roll, pitch, yaw]
        self.max_angular_velocity = np.array([20.0, 20.0, 35.0], dtype=np.float32)  # rad/s

        self.termination = False
        self.truncation = False
        self.max_steps = 10000
        self.step_count = 0
        self.flight_dome_size = 10.0
        self.max_height = 35.0
        self.info = {}

    def add_sphere(self):
        # Generate random position: 10m away from (0,0) in XY plane, random height 10-25m
        random_angle = random.uniform(0, 2 * np.pi)
        x = 10.0 * np.cos(random_angle)
        y = 10.0 * np.sin(random_angle)
        z = random.uniform(10, 25)
        
        # Store sphere position for reward calculation
        self.sphere_position = np.array([x, y, z], dtype=np.float32)
        
        self.sphere_visual_id = self.env.createVisualShape(
            shapeType=self.env.GEOM_SPHERE, radius=1, rgbaColor=[1, 0, 0, 1]
        )
        self.sphere_id = self.env.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.sphere_visual_id,
            basePosition=[x, y, z],
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
            
            # Calculate sphere size (X dimension normalized by image width to [-1, 1])
            sphere_width = max_x - min_x
            normalized_size = 2.0 * (sphere_width / width) - 1.0

            return np.array([normalized_x, normalized_y], dtype=np.float32), normalized_size

        return np.array([-1, -1], dtype=np.float32), -1.0

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

    def calculate_angular_velocity(self):
        if len(self.attitude_history) < 5:
            return np.zeros(3, dtype=np.float32)
        
        # Extract attitudes (no timestamps needed - we use fixed dt)
        attitudes = np.array(list(self.attitude_history), dtype=np.float32)
        
        # Create time array with fixed dt: [0, dt, 2*dt, 3*dt, 4*dt]
        # This represents time steps with indices 0-4
        timestamps = np.arange(5, dtype=np.float64) * self.dt
        
        # Normalize timestamps to improve numerical stability (center around 0)
        t_mean = np.mean(timestamps)
        t_normalized = timestamps - t_mean
        
        # Calculate angular velocity for each axis using least squares
        # We're fitting: attitude[i] = velocity * t + offset
        # For simple linear regression: velocity = cov(t, attitude) / var(t)
        angular_velocity = np.zeros(3, dtype=np.float32)
        
        for axis in range(3):
            attitude_axis = attitudes[:, axis]
            
            # Simple least squares solution for slope
            # slope = sum((t - t_mean) * (att - att_mean)) / sum((t - t_mean)^2)
            att_mean = np.mean(attitude_axis)
            numerator = np.sum(t_normalized * (attitude_axis - att_mean))
            denominator = np.sum(t_normalized ** 2)
            
            if abs(denominator) > 1e-10:
                angular_velocity[axis] = numerator / denominator
            else:
                angular_velocity[axis] = 0.0
            
        angular_velocity_clipped = np.clip(
            angular_velocity, 
            -self.max_angular_velocity, 
            self.max_angular_velocity
        )
        
        # Normalize to [-1, 1] range per axis by dividing by max for each axis
        angular_velocity_normalized = angular_velocity_clipped / self.max_angular_velocity        
        return angular_velocity_normalized.astype(np.float32)
    
    def create_observation(self, sensors):
        attitude_data = sensors[1] / np.pi

        # Get camera image and detect red sphere center and size
        rgba_image = self.env.drones[0].rgbaImg
        sphere_center, sphere_size = self.detect_red_sphere_center(rgba_image)

        # Update sphere history with current observation (last element is most recent)
        current_sphere = np.array([sphere_center[0], sphere_center[1], sphere_size], dtype=np.float32)
        self.sphere_history.append(current_sphere)

        # Calculate angular velocity from attitude history
        angular_velocity = self.calculate_angular_velocity()
        
        # Flatten last N actions into a single array
        last_N_actions = np.concatenate(list(self.action_history), dtype=np.float32)
        
        # Flatten last N sphere observations into a single array (last entry is most recent)
        last_N_sphere_obs = np.concatenate(list(self.sphere_history), dtype=np.float32)

        # print("angular_velocity", np.round(angular_velocity, 3))
        return np.concatenate(
            (attitude_data, angular_velocity, last_N_actions, last_N_sphere_obs),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.env.start_orn = np.array([[0.0, 0.0, 0.0]])

        self.add_sphere()
        # self.add_reference_structures()
        sensors = self.env.state(0)
        self.action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        
        # Reset attitude history buffer
        self.attitude_history.clear()
        for _ in range(5):
            self.attitude_history.append(np.zeros(3, dtype=np.float32))
        
        # Reset action history buffer
        default_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        self.action_history.clear()
        for _ in range(self.history_length):
            self.action_history.append(default_action.copy())
        
        # Reset sphere history buffer
        default_sphere = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.sphere_history.clear()
        for _ in range(self.history_length):
            self.sphere_history.append(default_sphere.copy())
        
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.info = {}

        obs = self.create_observation(sensors)
        return obs, {}

    def calculate_reward(self, obs, sensors):
        self.reward = 0.0

        # Large penalty for termination due to failure conditions
        if self.termination:
            if self.info.get("out_of_bounds", False):
                self.reward -= 100.0
            if self.info.get("excessive_tilt", False):
                self.reward -= 100.0
            return self.reward

        # Survival reward - encourages staying operational
        self.reward += 0.3
        
        # Target position: (0, 0, sphere_z - 10) - 10m below the sphere
        target_position = np.array([0.0, 0.0, self.sphere_position[2] - 10.0], dtype=np.float32)
        current_position = sensors[3]  # Global position [x, y, z]
        
        # Position error (Euclidean distance from target)
        position_error = np.linalg.norm(current_position - target_position)
        # Reward decreases with distance, maximum reward when at target
        self.reward += np.exp(-position_error)
        
        # Target orientation: yaw towards sphere, pitch and roll always 0
        # Calculate yaw angle to face the sphere from origin (0, 0)
        target_yaw = np.arctan2(self.sphere_position[1], self.sphere_position[0])
        
        target_orientation = np.array([0.0, 0.0, target_yaw], dtype=np.float32)
        current_orientation = sensors[1]  # Attitude [roll, pitch, yaw] in radians
        
        # Orientation error (sum of absolute deviations)
        orientation_error = np.sum(np.abs(current_orientation - target_orientation))
        self.reward += 0.7 * np.exp(-orientation_error)

        if len(self.action_history) >= 2:
            last_raw_action = self.action_history[-2]
            current_raw_action = self.action_history[-1]
            action_diff = current_raw_action - last_raw_action
            self.reward -= np.sum(np.square(action_diff)) * 0.05

        # print(f"Reward: {self.reward:.3f} | Pos Error: {position_error:.3f} | Orient Error: {orientation_error:.3f}")
        return self.reward
        

    def step(self, action):
        # Store the raw action in history before scaling
        raw_action = action.copy()
        
        action[0] *= 30.0 
        action[1] *= 30.0
        action[2] *= -30.0
        action[3] = (action[3] + 1) / 2  # Throttle: keep normalized 0-1

        self.action = action.copy()

        self.env.set_setpoint(0, action)

        for _ in range(1):
            self.env.step()

        sensors = self.env.state(0)
        
        # Update attitude history with current attitude (no timestamp needed - fixed dt)
        attitude_data = sensors[1]
        self.attitude_history.append(attitude_data.copy())
        
        # Update action history with raw action (before scaling)
        self.action_history.append(raw_action)

        # reset info dict for this step
        self.info = {}

        if self.step_count >= self.max_steps:
            self.truncation = True

        if abs(sensors[3][0]) > self.flight_dome_size or abs(sensors[3][1]) > self.flight_dome_size or sensors[3][2] > self.max_height:
            self.info["out_of_bounds"] = True
            self.termination = True

        # Terminate if roll or pitch exceeds 90 degrees (π/2 radians)
        roll = attitude_data[0]
        pitch = attitude_data[1]
        if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
            self.info["excessive_tilt"] = True
            self.termination = True

        self.step_count += 1
        obs = self.create_observation(sensors)
        reward = self.calculate_reward(obs, sensors)

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
            print("  Press 'R' to reset environment")
            print("  Press 'q' or ESC to quit")
            print("-" * 50)
            
            iteration = 0
            while True:
                # Get joystick controls
                action = joystick.get_action_array()
                # Apply action to drone
                self.env.set_setpoint(0, action)
                observation, reward, termination, truncation, info = self.step(action)

                if termination:
                    print("Termination: ", info)
                    self.reset()
                    continue

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
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r') or key == ord('R'):  # 'r' or 'R' to reset
                    print("\nResetting environment...")
                    observation, _ = self.reset()
                    iteration = 0
                    print("Environment reset complete!")
                    print("-" * 50)
                    continue
                
                iteration += 1
                
        except RuntimeError as e:
            print(f"Joystick Error: {e}")
            print("Falling back to keyboard controls...")
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

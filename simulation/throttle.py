import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from collections import deque
import cv2

from hover import DroneEnv as HoverDroneEnv, RadioMasterJoystick as BaseJoystick


class RadioMasterJoystick(BaseJoystick):
    """Simplified joystick controller that only reads throttle axis."""
    
    def get_action_array(self):
        """
        Get throttle control value as numpy array for drone control.
        
        Returns:
            np.array: Array with [throttle] value in [-1, 1] range
        """
        controls = self.get_controls()
        return np.array([controls['throttle']], dtype=np.float32)


class DroneEnv(HoverDroneEnv):
    """
    Throttle-only control environment.
    Inherits from hover DroneEnv but constrains action space to throttle only.
    Roll, pitch, and yaw are hardcoded.
    """
    
    def __init__(self, render=False):
        # Call parent init first
        super().__init__(render=render)
        
        # Override action space to be 1D (throttle only)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Override observation space: [attitude_xyz (3), sphere_center_xy (2), angular_velocity_xyz (3), last_4_throttle_actions (4)]
        # Total: 3 + 2 + 3 + 4 = 12
        low_bounds = np.array([-1] * 3 + [-1] * 2 + [-1] * 3 + [-1] * 4, dtype=np.float32)
        high_bounds = np.array([1] * 3 + [1] * 2 + [1] * 3 + [1] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(12,), dtype=np.float32
        )
        
        # Hardcoded action values
        self.hardcoded_roll = 0.0
        self.hardcoded_pitch = 0.0
        self.hardcoded_yaw = 0.0
        
        # Override action history to store scalar throttle values instead of 4D arrays
        self.action_history = deque(maxlen=4)
        for _ in range(4):
            self.action_history.append(-1.0)  # Default throttle is -1
    
    def create_observation(self, sensors):
        """Override to use scalar action history instead of 4D arrays."""
        attitude_data = sensors[1] / np.pi

        # Get camera image and detect red sphere center
        rgba_image = self.env.drones[0].rgbaImg
        sphere_center = self.detect_red_sphere_center(rgba_image)

        # Calculate angular velocity from attitude history
        angular_velocity = self.calculate_angular_velocity()
        
        # Last 4 throttle actions as array (scalars not 4D arrays)
        last_4_throttle_actions = np.array(list(self.action_history), dtype=np.float32)

        return np.concatenate(
            (attitude_data, sphere_center, angular_velocity, last_4_throttle_actions),
            dtype=np.float32,
        )
    
    def reset(self, seed=None, options=None):
        """Override to reset scalar action history."""
        self.env.reset()
        self.env.start_orn = np.array([[0.0, 0.0, 0.0]])

        self.add_sphere()
        sensors = self.env.state(0)
        
        # Reset attitude history buffer
        self.attitude_history.clear()
        for _ in range(5):
            self.attitude_history.append(np.zeros(3, dtype=np.float32))
        
        # Reset action history buffer with scalar throttle values
        self.action_history.clear()
        for _ in range(4):
            self.action_history.append(-1.0)
        
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.info = {}

        obs = self.create_observation(sensors)
        return obs, {}
    
    def step(self, action):
        """Override to handle 1D action (throttle only) and construct full 4D action."""
        # Extract throttle from action (1D array)
        throttle_action = float(action[0])
        
        # Store the raw throttle action in history
        self.action_history.append(throttle_action)
        
        # Construct full action with hardcoded values: [roll, pitch, yaw, throttle]
        full_action = np.array([
            self.hardcoded_roll,
            self.hardcoded_pitch,
            self.hardcoded_yaw,
            throttle_action
        ], dtype=np.float32)
        
        # Scale the full action for the drone
        full_action[0] *= 30.0  # roll (already 0)
        full_action[1] *= 30.0  # pitch (already 0)
        full_action[2] *= -30.0  # yaw (already 0)
        full_action[3] = (full_action[3] + 1) / 2  # Throttle: normalize to 0-1

        self.env.set_setpoint(0, full_action)

        for _ in range(1):
            self.env.step()

        sensors = self.env.state(0)
        
        # Update attitude history with current attitude
        attitude_data = sensors[1]
        self.attitude_history.append(attitude_data.copy())

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

        return obs, reward, self.termination, self.truncation, self.info
    
    def test_env_with_joystick(self, joystick_index=0):
        """Override to display throttle-only control instructions."""
        try:
            # Initialize RadioMaster joystick
            joystick = RadioMasterJoystick(joystick_index)
            
            print("RadioMaster Pocket Joystick Controls:")
            print("  Axis 2: Throttle (up/down) - ONLY CONTROL")
            print("  Roll, Pitch, Yaw: Hardcoded to 0")
            print("  Press 'R' to reset environment")
            print("  Press 'q' or ESC to quit")
            print("-" * 50)
            
            iteration = 0
            while True:
                # Get joystick controls (only throttle)
                action = joystick.get_action_array()
                # Apply action to drone
                observation, reward, termination, truncation, info = self.step(action)

                rgba_frame = self.env.drones[0].rgbaImg
                
                # Print status every 10 iterations
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}")
                    print(f"  Throttle action: {action[0]:.3f}")
                    print(f"  Attitude [x,y,z]: [{np.degrees(observation[0]):.3f}°, "
                          f"{np.degrees(observation[1]):.3f}°, {np.degrees(observation[2]):.3f}°]")
                    print(f"  Sphere center [x,y]: [{observation[3]:.3f}, {observation[4]:.3f}]")
                    print(f"  Reward: {reward:.3f}")
                    print("-" * 50)
                
                # Display camera view
                frame = cv2.cvtColor(rgba_frame.astype(np.uint8), cv2.COLOR_RGBA2BGR)
                cv2.imshow("RadioMaster Pocket Control - Throttle Only", frame)
                
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

    print("Environment loaded. Starting RadioMaster Pocket control (Throttle only)...")
    print("Press Ctrl+C to exit.")

    # Try joystick control
    env.test_env_with_joystick()


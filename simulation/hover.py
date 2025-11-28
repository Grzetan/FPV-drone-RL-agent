import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
import pybullet as p
from radio_controller import RadioMasterJoystick
import cv2

class QuadXHoverEnv(gym.Env):
    def __init__(
        self,
        flight_mode: int = 0,
        agent_hz: int = 40,
        render: bool = False,
    ):
        super().__init__()

        self.flight_mode = flight_mode
        self.agent_hz = agent_hz
        self.render = render

        self.env_step_ratio = int(120 / self.agent_hz)

        self._target_pos = np.array([0.0, 0.0, 1.0])

        self.action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.termination = False
        self.truncation = False
        self.max_steps = 400
        self.step_count = 0
        self.flight_dome_size = 3.0
        self.floor_threshold = 0.1  # Height threshold to consider "on the floor"
        
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False
        self.info["on_floor"] = False
        
        high = np.ones(4)
        low = -np.ones(4)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Observation Space:
        # Structure: [ang_vel (3), quaternion (4), lin_vel (3), lin_pos (3), action (4)]
        # Total size: 3 + 4 + 3 + 3 + 4 = 17
        obs_dim = 17
        
        high = np.inf * np.ones(obs_dim)
        low = -np.inf * np.ones(obs_dim)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

        self.aviary = Aviary(
            start_pos=np.array([[0.0, 0.0, 0.0]]), # Starting position set to 0,0,0
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.render,
            drone_type="quadx",
            drone_options={
                "use_camera": self.render,
                "camera_angle_degrees": -25,
                "camera_FOV_degrees": 90,
                "camera_resolution": (128, 128),
                "model_dir": "./drone_models",
                "drone_model": "cf2x",
            },
        )
        self.aviary.set_mode(0)
        self.aviary.reset()
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.action = np.zeros(4, dtype=np.float32)
        self.info = {"out_of_bounds": False, "collision": False, "env_complete": False, "on_floor": False}
        
        for _ in range(10):
            self.aviary.step()

        self.compute_state()
        return self.state, self.info

    def compute_attitude(self):
        """
        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        raw_state = self.aviary.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quaternion angles
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quaternion

    def compute_state(self):
        """Computes the state of the current timestep."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        
        # Combined state without auxiliary info
        # [ang_vel, quaternion, lin_vel, lin_pos, action]
        self.state = np.concatenate(
            [
                ang_vel, 
                quaternion, 
                lin_vel, 
                lin_pos, 
                self.action
            ], 
            axis=-1
        )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        
        # --- Base Termination/Truncation Logic ---
        # Exceed step count
        if self.step_count > self.max_steps:
            self.truncation = True

        # Exceed flight dome
        if np.linalg.norm(self.aviary.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination = True
        
        # Check if drone touches the floor after step 30
        if self.step_count > 30 and not self.render:
            lin_pos = self.aviary.state(0)[-1]
            z_position = lin_pos[2]
            
            if z_position < self.floor_threshold:
                self.reward = -100.0
                self.info["on_floor"] = True
                self.termination = True

        linear_distance = np.linalg.norm(
            self.aviary.state(0)[-1] - self._target_pos
        )
        
        # Negative Reward For High Yaw rate
        yaw_rate = abs(self.aviary.state(0)[0][2])
        yaw_rate_penalty = 0.01 * yaw_rate**2
        self.reward -= yaw_rate_penalty

        # Orientation penalty (roll/pitch)
        angular_distance = np.linalg.norm(self.aviary.state(0)[1][:2])

        self.reward -= linear_distance + angular_distance

        action_diff = self.action - self.prev_action
        smoothness_penalty = np.linalg.norm(action_diff)
        self.reward -= smoothness_penalty * 0.2

        self.reward += 1.0

    def step(self, action):
        """Steps the environment."""
        self.action = action.copy()

        action[0] *= 30.0 
        action[1] *= 30.0
        action[2] *= -30.0
        action[3] = (action[3] + 1) / 2  # Throttle: keep normalized 0-1

        self.reward = -0.1
        self.aviary.set_setpoint(0, action)

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break

            self.aviary.step()

            self.compute_state()
            self.compute_term_trunc_reward()

        self.step_count += 1
        self.prev_action = self.action.copy()
        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """Renders the environment."""
        return self.aviary.render()

    def close(self):
        """Cleans up the environment."""
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def test_env_with_joystick(self, joystick_index=0):
        """Test the environment with a RadioMaster joystick controller."""
        joystick = None
        try:
            # Initialize RadioMaster joystick
            joystick = RadioMasterJoystick(joystick_index)
            
            # Reset environment to initialize
            observation, _ = self.reset()
            
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
                # Step the environment (set_setpoint is called inside step())
                observation, reward, termination, truncation, info = self.step(action)

                if termination or truncation:
                    print(f"\nTermination: {info}")
                    print("Resetting environment...")
                    observation, _ = self.reset()
                    iteration = 0
                    continue
                
                # Print status every iteration
                if iteration % 10 == 0:
                    # Observation structure: [ang_vel (3), quaternion (4), lin_vel (3), lin_pos (3), action (4)]
                    ang_vel = observation[0:3]
                    lin_pos = observation[10:13]  # lin_pos starts at index 10 (3+4+3)
                    lin_vel = observation[7:10]   # lin_vel starts at index 7 (3+4)
                    
                    print(f"Iteration {iteration}")
                    print(f"  Reward: {reward:.3f}")
                    print(f"  Position [x,y,z]: [{lin_pos[0]:.3f}, {lin_pos[1]:.3f}, {lin_pos[2]:.3f}]")
                    print(f"  Velocity [x,y,z]: [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]")
                    print(f"  Angular velocity [x,y,z]: [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")
                    print(f"  Step count: {self.step_count}")
                    print("-" * 50)
                
                # Check for keyboard input (non-blocking)
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
            if joystick is not None:
                try:
                    joystick.cleanup()
                except:
                    pass
            cv2.destroyAllWindows()

if __name__ == "__main__":
    env = QuadXHoverEnv(render=True)
    env.test_env_with_joystick()
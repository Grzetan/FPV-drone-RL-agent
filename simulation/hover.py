import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
import pybullet as p

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
                # "use_camera": True,
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
        if self.step_count > 30:
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
        self.reward += 1.0

    def step(self, action):
        """Steps the environment."""
        self.action = action.copy()

        self.reward = -0.1
        self.aviary.set_setpoint(0, action)

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break

            self.aviary.step()

            self.compute_state()
            self.compute_term_trunc_reward()

        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """Renders the environment."""
        return self.aviary.render()

    def close(self):
        """Cleans up the environment."""
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
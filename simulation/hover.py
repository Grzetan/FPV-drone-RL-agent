import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyFlyt.core import Aviary
import pybullet as p
from radio_controller import RadioMasterJoystick
import cv2
import random

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

        self.physics_hz = 240.0
        self.env_step_ratio = int(self.physics_hz / self.agent_hz)
        self.agent_dt = 1.0 / self.agent_hz
        self.previous_ang_pos = np.zeros(3, dtype=np.float32)

        self._target_pos = np.array([0.0, 0.0, 1.0])

        self.action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.termination = False
        self.truncation = False
        self.max_steps = 400
        self.step_count = 0
        self.flight_dome_size = 3.0
        self.floor_threshold = 0.1

        # Rectangle detection state variables
        self.previous_center = np.zeros(2, dtype=np.float32)
        self.previous_area = 0.0
        self.previous_ratio = 0.0
        
        # Target values for reward calculation
        # Target center: [0, 0] means centered in camera view
        # Target area: desired area of the rectangle (normalized by image area)
        # Target ratio: 1.0 for a perfect square
        self.target_center = np.array([0.0, 0.0], dtype=np.float32)
        self.target_area = 0.013  # Adjust this based on desired size (10% of image area)
        self.target_ratio = 1.53  # Perfect square

        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False
        self.info["on_floor"] = False

        high = np.ones(4)
        low = -np.ones(4)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Observation: [ang_vel (3), quaternion (4), current_center (2), previous_center (2), 
        #                current_area (1), previous_area (1), target_visible (1), 
        #                bbox_width_height_ratio (1), previous_radio (1), action (4)]
        # Total: 3 + 4 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 4 = 20
        obs_dim = 20
        high = np.inf * np.ones(obs_dim)
        low = -np.inf * np.ones(obs_dim)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

        self.aviary = Aviary(
            start_pos=np.array([[0.0, 0.0, 0.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.render,
            drone_type="quadx",
            physics_hz=self.physics_hz,
            drone_options={
                "use_camera": True,  # Always enable camera for rectangle detection
                "camera_angle_degrees": -25,
                "camera_FOV_degrees": 90,
                "camera_resolution": (128, 128),
                "model_dir": "./drone_models",
                "drone_model": "cf2x",
            },
        )
        self.aviary.set_mode(0)
        self.aviary.reset()
        
        # Add reference rectangle
        self.add_reference_object()
        
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.action = np.zeros(4, dtype=np.float32)
        self.info = {"out_of_bounds": False, "collision": False, "env_complete": False, "on_floor": False}
        
        # Reset rectangle detection state
        self.previous_center = np.zeros(2, dtype=np.float32)
        self.previous_area = 0.0
        self.previous_ratio = 0.0

        for _ in range(10):
            self.aviary.step()

        self.previous_ang_pos = self.aviary.state(0)[1].copy()
        self.compute_state()
        return self.state, self.info

    def add_reference_object(self):
        """Add a reference rectangle (red face) in the environment."""
        # Fixed position at [6, 0, 6]
        x = 6.0
        y = 0.0
        z = 6.0
        
        # Store the position for reference
        self.reference_object_position = np.array([x, y, z], dtype=np.float32)
        
        # Create a square red face (2m x 2m square)
        square_size = 2.0  # 2m square
        
        # Create a thin red square face (a flat box)
        # The red face is placed on the +X face (forward direction)
        red_face_visual_id = self.aviary.createVisualShape(
            shapeType=self.aviary.GEOM_BOX,
            halfExtents=[0.02, square_size/2, square_size/2],  # Thin red square panel (thin in X, square in YZ)
            rgbaColor=[1, 0, 0, 1],  # Red
            visualFramePosition=[square_size/2, 0, 0]  # Position it on the +X face
        )
        
        # Calculate yaw rotation to face origin (only Z-axis rotation)
        # The red face is on the +X local face, so we need to rotate so that +X points toward origin
        yaw_to_origin = np.arctan2(-y, -x)  # Angle from cube position to origin in XY plane
        yaw_rotation = yaw_to_origin
        
        # Create orientation quaternion with only yaw rotation (roll=0, pitch=0)
        # Quaternion for rotation around Z axis: [x, y, z, w] = [0, 0, sin(yaw/2), cos(yaw/2)]
        quat_z = np.sin(yaw_rotation / 2)
        quat_w = np.cos(yaw_rotation / 2)
        orientation = [0, 0, quat_z, quat_w]
        
        # Create the red face as a separate body at the same position with same orientation
        self.red_face_id = self.aviary.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=red_face_visual_id,
            basePosition=[x, y, z],
            baseOrientation=orientation
        )

    def detect_rectangle(self, rgba_image, return_corners=False):
        """
        Detect red rectangle in the camera image and extract features.
        Returns: (target_visible, current_center, current_area, bbox_width_height_ratio, corners)
        corners is only returned if return_corners=True
        """
        # Default values when target is not visible
        default_center = np.zeros(2, dtype=np.float32)
        default_area = 0.0
        default_ratio = 0.0
        default_corners = None
        
        # Get image dimensions (128x128 as configured)
        img_height, img_width = rgba_image.shape[:2]
        
        # Create red mask
        red_mask = (
            (rgba_image[:, :, 0] > 100) &  # Red channel > 100
            (rgba_image[:, :, 1] == 0) &   # Green channel == 0
            (rgba_image[:, :, 2] == 0)     # Blue channel == 0
        ).astype(np.uint8) * 255

        # Check if red is at edges (target partially out of view)
        red_at_edges = (
            np.any(red_mask[0, :] > 0) or      # Top edge
            np.any(red_mask[-1, :] > 0) or     # Bottom edge
            np.any(red_mask[:, 0] > 0) or        # Left edge
            np.any(red_mask[:, -1] > 0)        # Right edge
        )

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and not red_at_edges:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                corners = approx.reshape((4, 2))
                
                # Calculate center (normalized to [-1, 1])
                center_x = np.mean(corners[:, 0])
                center_y = np.mean(corners[:, 1])
                # Normalize: X: left edge (0) -> -1, right edge (img_width) -> 1
                normalized_center_x = (center_x / (img_width / 2.0)) - 1.0
                # Normalize: Y: top edge (0) -> -1, bottom edge (img_height) -> 1
                normalized_center_y = (center_y / (img_height / 2.0)) - 1.0
                current_center = np.array([normalized_center_x, normalized_center_y], dtype=np.float32)
                
                # Calculate area (normalized by image area)
                current_area = cv2.contourArea(contour) / (img_width * img_height)
                
                # Calculate bounding box and width/height ratio
                x, y, w, h = cv2.boundingRect(contour)
                if h > 0:
                    bbox_width_height_ratio = w / h
                else:
                    bbox_width_height_ratio = 0.0
                
                if return_corners:
                    return True, current_center, current_area, bbox_width_height_ratio, corners
                return True, current_center, current_area, bbox_width_height_ratio

        # Target not visible
        if return_corners:
            return False, default_center, default_area, default_ratio, default_corners
        return False, default_center, default_area, default_ratio

    def compute_attitude(self):
        raw_state = self.aviary.state(0)
        current_ang_pos = raw_state[1]

        angle_diff = current_ang_pos - self.previous_ang_pos
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        ang_vel = angle_diff / self.agent_dt

        ang_pos = current_ang_pos
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, quaternion

    def compute_state(self):
        ang_vel, ang_pos, quaternion = self.compute_attitude()
        
        # Get camera image and detect rectangle
        rgba_image = self.aviary.drones[0].rgbaImg
        target_visible, current_center, current_area, bbox_width_height_ratio = self.detect_rectangle(rgba_image)
        
        # If target is not visible, set all values to 0 (neutral state)
        if not target_visible:
            current_center = np.zeros(2, dtype=np.float32)
            current_area = 0.0
            bbox_width_height_ratio = 0.0
        
        # Build observation: [ang_vel (3), quaternion (4), current_center (2), previous_center (2),
        #                     current_area (1), previous_area (1), target_visible (1),
        #                     bbox_width_height_ratio (1), previous_radio (1), action (4)]
        self.state = np.concatenate(
            [
                ang_vel,
                quaternion,
                current_center,
                self.previous_center,
                np.array([current_area]),
                np.array([self.previous_area]),
                np.array([1.0 if target_visible else 0.0]),
                np.array([bbox_width_height_ratio]),
                np.array([self.previous_ratio]),
                self.action
            ],
            axis=-1
        )
        
        # Update previous values for next step (always, regardless of visibility)
        self.previous_center = current_center.copy()
        self.previous_area = current_area
        self.previous_ratio = bbox_width_height_ratio

    def compute_term_trunc_reward(self):
        if self.step_count > self.max_steps:
            self.truncation = True

        if np.linalg.norm(self.aviary.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination = True

        if self.step_count > 30 and not self.render:
            lin_pos = self.aviary.state(0)[-1]
            z_position = lin_pos[2]

            if z_position < self.floor_threshold:
                self.reward = -100.0
                self.info["on_floor"] = True
                self.termination = True

        # Target-based rewards (replace linear_distance)
        # Extract values from state: [ang_vel (3), quaternion (4), current_center (2), previous_center (2),
        #                             current_area (1), previous_area (1), target_visible (1),
        #                             bbox_width_height_ratio (1), previous_radio (1), action (4)]
        target_visible = self.state[13] > 0.5
        
        if target_visible:
            # Extract current values from state
            current_center = self.state[7:9]
            current_area = self.state[11]
            current_ratio = self.state[14]
            
            # Target center reward: penalize distance from target center [0, 0]
            center_distance = np.linalg.norm(current_center - self.target_center)
            center_reward = -center_distance
            
            # Target area reward: penalize distance from target area
            area_distance = abs(current_area - self.target_area)
            area_reward = -area_distance
            
            # Target ratio reward: penalize distance from target ratio (1.0 for square)
            ratio_distance = abs(current_ratio - self.target_ratio)
            ratio_reward = -ratio_distance
            
            # Combine target rewards
            target_reward = center_reward + area_reward + ratio_reward
        else:
            # Large penalty when target is not visible
            target_reward = -2.0

        yaw_rate = abs(self.aviary.state(0)[0][2])
        yaw_rate_penalty = 0.01 * yaw_rate**2
        self.reward -= yaw_rate_penalty

        angular_distance = np.linalg.norm(self.aviary.state(0)[1][:2])
        self.reward += target_reward - angular_distance

        action_diff = self.action - self.prev_action
        smoothness_penalty = np.linalg.norm(action_diff)
        self.reward -= smoothness_penalty * 0.2
        self.reward += 1.0

    def step(self, action):
        self.action = action.copy()

        action_scaled = action.copy()
        action_scaled[0] *= 30.0
        action_scaled[1] *= 30.0
        action_scaled[2] *= -30.0
        action_scaled[3] = (action_scaled[3] + 1) / 2

        self.reward = -0.1
        self.aviary.set_setpoint(0, action_scaled)

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break
            self.aviary.step()

        self.compute_state()
        self.compute_term_trunc_reward()

        self.previous_ang_pos = self.aviary.state(0)[1].copy()

        self.step_count += 1
        self.prev_action = self.action.copy()
        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        return self.aviary.render()

    def close(self):
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def test_env_with_joystick(self, joystick_index=0):
        joystick = None
        try:
            joystick = RadioMasterJoystick(joystick_index)
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
                action = joystick.get_action_array()
                observation, reward, termination, truncation, info = self.step(action)

                if termination or truncation:
                    print(f"\nTermination: {info}")
                    print("Resetting environment...")
                    observation, _ = self.reset()
                    iteration = 0
                    continue

                # Get camera image and detect rectangle with corners for visualization
                rgba_image = self.aviary.drones[0].rgbaImg
                target_visible, current_center, current_area, bbox_width_height_ratio, corners = self.detect_rectangle(rgba_image, return_corners=True)
                
                # Convert RGBA to BGR for OpenCV display
                frame = cv2.cvtColor(rgba_image.astype(np.uint8), cv2.COLOR_RGBA2BGR)
                
                # Draw corners if target is visible
                if target_visible and corners is not None:
                    # Draw corners as green circles
                    for corner in corners:
                        cv2.circle(frame, (int(corner[0]), int(corner[1])), 1, (0, 255, 0), -1)
                                        
                    # Draw center point
                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))
                    cv2.circle(frame, (center_x, center_y), 1, (255, 0, 0), -1)  # Blue center point
                
                # Display the camera view
                cv2.imshow("Camera View - Rectangle Detection", frame)

                if iteration % 10 == 0:
                    ang_vel = observation[0:3]
                    current_center = observation[7:9]
                    previous_center = observation[9:11]
                    current_area = observation[11]
                    previous_area = observation[12]
                    target_visible = observation[13]
                    bbox_ratio = observation[14]
                    previous_radio = observation[15]

                    print(f"Iteration {iteration}")
                    print(f"  Reward: {reward:.3f}")
                    print(f"  Target visible: {target_visible:.0f}")
                    print(f"  Current center [x,y]: [{current_center[0]:.3f}, {current_center[1]:.3f}]")
                    print(f"  Previous center [x,y]: [{previous_center[0]:.3f}, {previous_center[1]:.3f}]")
                    print(f"  Current area: {current_area:.3f}, Previous area: {previous_area:.3f}")
                    print(f"  Bbox ratio: {bbox_ratio:.3f}, Previous radio: {previous_radio:.3f}")
                    print(f"  Angular velocity [x,y,z]: [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")
                    print(f"  Step count: {self.step_count}")
                    print("-" * 50)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r') or key == ord('R'):
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
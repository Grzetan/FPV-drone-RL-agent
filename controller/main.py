import logging
import toml
import time
from collections import deque
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import serial

from camera import Camera
from crsf_parser import get_attitude_frames, send_control_data, create_channels_packet

CONFIG_PATH = 'controller/config.toml'
LAST_X = 10 

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config():
    try:
        return toml.load(CONFIG_PATH)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {CONFIG_PATH}")
        exit()
    except toml.TomlDecodeError:
        logging.error(f"Error decoding the TOML configuration file.")
        exit()

def scale_action_to_cfsr(action):
    return np.interp(action, [-1, 1], [988, 2012]).astype(int)

def construct_observation(last_actions, last_ref_points, attitude, ang_velocity):
    flat_actions = np.array(last_actions).flatten()
    flat_points = np.array(last_ref_points).flatten()
    attitude_data = np.array([attitude['pitch'], attitude['roll'], attitude['yaw']])
    
    return np.concatenate([flat_actions, flat_points, attitude_data, ang_velocity])

def main():
    config = load_config()
    setup_logging(config['logging']['file'])

    cam = Camera(config['camera']['id'], config['camera']['preview'])
    
    env = DummyVecEnv([lambda: None]) 
    vec_env = VecNormalize.load(config['model']['path'] + ".pkl", venv=env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = SAC.load(config['model']['path'] + ".zip", env=vec_env)

    try:
        ser = serial.Serial(config['serial']['port'], config['serial']['baud_rate'])
    except serial.SerialException as e:
        logging.error(f"Failed to open serial port {config['serial']['port']}: {e}")
        return

    attitude_generator = get_attitude_frames(ser.port, ser.baudrate)

    last_actions = deque(maxlen=LAST_X)
    last_ref_points = deque(maxlen=LAST_X)
    
    for _ in range(LAST_X):
        last_actions.append(np.zeros(4)) 
        last_ref_points.append(np.zeros((4, 2)))

    try:
        logging.info("Controller loop started.")
        for attitude in attitude_generator:
            frame = cam.get_frame()
            if frame is None:
                break
            
            corners = cam.find_reference_corners(frame)
            if corners is not None:
                last_ref_points.append(corners)

            angular_velocity = np.zeros(3)

            obs = construct_observation(last_actions, last_ref_points, attitude, angular_velocity)
            
            normalized_obs = vec_env.normalize_obs(obs)

            action, _ = model.predict(normalized_obs, deterministic=True)
            last_actions.append(action)

            channels = [1500] * 16
            scaled_action = scale_action_to_cfsr(action)
            channels[0] = scaled_action[0] # Throttle
            channels[1] = scaled_action[1] # Roll
            channels[2] = scaled_action[2] # Pitch
            channels[3] = scaled_action[3] # Yaw

            send_control_data(ser, channels)
            
            if not cam.show_preview(frame):
                break

    except KeyboardInterrupt:
        logging.info("Controller loop stopped by user.")
    finally:
        cam.release()
        ser.close()
        logging.info("Resources released.")

if __name__ == "__main__":
    main()

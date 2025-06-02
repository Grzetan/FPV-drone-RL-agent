from stable_baselines3.common.env_checker import check_env
from env import DroneEnv

check_env(DroneEnv())
print("Environment is valid!")

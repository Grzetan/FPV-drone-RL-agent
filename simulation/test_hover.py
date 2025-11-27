import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage
from stable_baselines3 import PPO, SAC
from hover import DroneEnv
from yaw import DroneEnv as YawDroneEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import time
env = DummyVecEnv([lambda: DroneEnv(render=True)])
vec_env = VecNormalize.load("./ppo_hover_checkpoint3/ppo_hover_2000000_steps_None_lenNone_rewNone.pkl", venv=env)
vec_env.training = False
vec_env.norm_reward = False
model = SAC.load("./ppo_hover_checkpoint3/ppo_hover_2000000_steps_None_lenNone_rewNone.zip", env=vec_env)


obs = vec_env.reset()
for _ in range(2000):
    start_time = time.time()
    action, _ = model.predict(obs, deterministic=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    print(action)
    # input()
    obs, reward, done, info = vec_env.step(action)
    # print(obs)
    # print(f"Velocity: {obs[3:6]}")
    if done[0]:
        obs = vec_env.reset()
        print("Episode finished, resetting...")

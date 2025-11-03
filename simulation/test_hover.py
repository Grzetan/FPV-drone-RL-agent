from stable_baselines3 import PPO, SAC
from hover import DroneEnv
from yaw import DroneEnv as YawDroneEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

env = DummyVecEnv([lambda: DroneEnv(render=True)])
vec_env = VecNormalize.load("ppo_hover_checkpoint2/ppo_hover_960000_steps_None_lenNone_rewNone.pkl", venv=env)
vec_env.training = False
vec_env.norm_reward = False
model = SAC.load("ppo_hover_checkpoint2/ppo_hover_960000_steps_None_lenNone_rewNone.zip", env=vec_env)


obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    # input()
    obs, reward, done, info = vec_env.step(action)
    if done[0]:
        obs = vec_env.reset()
        print("Episode finished, resetting...")

from stable_baselines3 import PPO
from hover import DroneEnv

model = PPO.load("ppo_hover")
env = DroneEnv(render=True)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

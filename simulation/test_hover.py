from stable_baselines3 import PPO
from hover import DroneEnv

model = PPO.load("ppo_hover_checkpoints/ppo_hover_4096_steps_1182364.5.zip")
env = DroneEnv(render=True)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    print(action)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

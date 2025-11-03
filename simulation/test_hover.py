from stable_baselines3 import PPO
from hover import DroneEnv
from yaw import DroneEnv as YawDroneEnv

model = PPO.load("ppo_hover_checkpoints/ppo_hover_180224_steps_0.28.zip")
env = YawDroneEnv(render=True)

obs, _ = env.reset()
for _ in range(3000):
    action, _ = model.predict(obs)
    print(action)
    input()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

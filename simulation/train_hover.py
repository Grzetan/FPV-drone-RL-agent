from hover import DroneEnv
from stable_baselines3 import PPO

env = DroneEnv(render=False)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_drone_tensorboard/"
)

model.learn(total_timesteps=100000)

model.save("ppo_hover")

obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
    
print("Training complete.")
import gymnasium
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

env = gymnasium.make("PyFlyt/QuadX-Hover-v1")
env = DummyVecEnv([lambda: env])

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logs_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# Train the model
total_timesteps = 500000
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save(f"{models_dir}/ppo_quadx_hover")

# Test the trained model
env = gymnasium.make("PyFlyt/QuadX-Hover-v1", render_mode="human")
obs, _ = env.reset()
term, trunc = False, False

while not (term or trunc):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)

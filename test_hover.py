from stable_baselines3 import PPO
from hover import DroneEnv  # Assuming you have the DroneEnv class defined in hover.py

# Load the trained model - replace 'PPO' with the algorithm you used and 'path_to_model' with your model's path
model = PPO.load("models/ppo_quadx_hover")

# Test the trained model
env = DroneEnv(render=True)
obs, _ = env.reset()
term, trunc = False, False

while not (term or trunc):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)

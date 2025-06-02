import numpy as np
import gymnasium as gym
from hover import DroneEnv
from stable_baselines3 import PPO
import time

# Path to the saved model
model_path = "logs/final_drone_model"


def test_drone_model(model_path, episodes=5, max_steps=500):
    # Load the trained model
    model = PPO.load(model_path)
    print(f"Model loaded from {model_path}")

    # Create the environment
    env = DroneEnv()
    print("Environment created")

    total_episode_rewards = []

    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")

        # Reset the environment
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # Use the model to predict the next action
            action, _ = model.predict(obs, deterministic=True)

            # Take a step in the environment
            obs, reward, done, truncated, info = env.step(action)

            # Update metrics
            episode_reward += reward
            steps += 1

            # Get current position and distance to target
            sensors = env.env.state(0)
            current_pos = sensors[3]
            distance = np.linalg.norm(current_pos - env.target_pos)

            # Print status periodically
            if steps % 20 == 0:
                print(f"Step: {steps}, Distance: {distance:.2f}m, Reward: {reward:.2f}")
                print(f"Position: {current_pos}, Target: {env.target_pos}")

            if done or truncated:
                break

        total_episode_rewards.append(episode_reward)
        print(
            f"Episode {episode+1} finished: Steps: {steps}, Total Reward: {episode_reward:.2f}"
        )
        print(f"Final distance to target: {distance:.2f}m")

    # Display overall performance
    avg_reward = np.mean(total_episode_rewards)
    print(f"\nTesting completed! Average episode reward: {avg_reward:.2f}")

    # Close the environment
    env.close()
    return avg_reward


if __name__ == "__main__":
    # Test the model
    test_drone_model(model_path)

    # Optionally, test the last checkpoint as well
    # checkpoint_path = "logs/checkpoints/drone_model_24000_steps"
    # test_drone_model(checkpoint_path)

from hover import DroneEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

env = DroneEnv(render=False)
env = DummyVecEnv([lambda: env])

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self) -> bool:
        logger_values = self.model.logger.name_to_value
        value_loss = round(logger_values.get("train/loss"), 2) if logger_values.get("train/loss") is not None else "N/A"
        checkpoint_name = f"{self.name_prefix}_{self.num_timesteps}_steps_{value_loss}.zip"
        model_path = os.path.join(self.save_path, checkpoint_name)
        self.model.save(model_path)
        if self.verbose > 0:
            print(f"Saving model checkpoint to {model_path}\n")

        return True
    
    def _on_step(self) -> bool:
        return True

checkpoint_callback = CheckpointCallback(save_path="./ppo_hover_checkpoints/", name_prefix="ppo_hover", verbose=1)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./tensorboard",
    n_steps=2048,
    batch_size=64,
    n_epochs=10
)

model.learn(total_timesteps=100000, callback=checkpoint_callback)

model.save("ppo_hover")
    
print("Training complete.")
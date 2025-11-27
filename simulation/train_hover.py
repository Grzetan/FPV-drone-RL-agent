from hover import DroneEnv
from yaw import DroneEnv as YawDroneEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, min_steps_between_checkpoints: int = 20000):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.min_steps_between_checkpoints = min_steps_between_checkpoints
        os.makedirs(self.save_path, exist_ok=True)
        self._last_checkpoint_step = 0

    def _on_rollout_end(self) -> bool:
        # Only save if enough steps since last checkpoint
        if self.num_timesteps - self._last_checkpoint_step >= self.min_steps_between_checkpoints:
            logger_values = self.model.logger.name_to_value
            value_loss = round(logger_values.get("train/loss"), 2) if logger_values.get("train/loss") is not None else "None"
            ep_len_mean = round(logger_values.get("rollout/ep_len_mean"), 2) if logger_values.get("rollout/ep_len_mean") is not None else "None"
            rp_rew_mean = round(logger_values.get("rollout/ep_rew_mean"), 2) if logger_values.get("rollout/ep_rew_mean") is not None else "None"
            checkpoint_name = f"{self.name_prefix}_{self.num_timesteps}_steps_{value_loss}_len{ep_len_mean}_rew{rp_rew_mean}.zip"
            model_path = os.path.join(self.save_path, checkpoint_name)
            self.model.save(model_path)
            self.model.env.save(model_path.replace(".zip", ".pkl"))
            self._last_checkpoint_step = self.num_timesteps
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}\n")
        return True

    def _on_step(self) -> bool:
        return True

def make_env():
    return DroneEnv(render=False)

if __name__ == "__main__":
    num_env = 16
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    checkpoint_callback = CheckpointCallback(save_path="./ppo_hover_checkpoint3/", name_prefix="ppo_hover", verbose=1)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard",
        buffer_size=200_000,      # Size of the replay buffer (e.g., 200k transitions)
        train_freq=(1, "step"),   # Train the model after each step in the environment
        gradient_steps=1,         # How many gradient updates to perform after each train_freq
        batch_size=256,
        learning_rate=0.0003,
        policy_kwargs=dict(net_arch=[512, 512])
    )

    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)

    model.save("hover")
    env.save("hover")
    print("Training complete.")
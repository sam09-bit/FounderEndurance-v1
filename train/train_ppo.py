# train/train_ppo.py
import gymnasium as gym
import founder_endurance  # triggers registration
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def main():
    env_id = "FounderEndurance-v1"

    # Vectorised training: 8 parallel environments
    train_env = make_vec_env(env_id, n_envs=8)
    eval_env = make_vec_env(env_id, n_envs=2)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
    )

    model.save("./models/founder_ppo_final")


if __name__ == "__main__":
    main()
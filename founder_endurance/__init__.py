from gymnasium.envs.registration import register

register(
    id="FounderEndurance-v1",
    entry_point="founder_endurance.envs.founder_env:FounderEnv",
    max_episode_steps=90,
    reward_threshold=400.0,
)
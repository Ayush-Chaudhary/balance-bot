from gymnasium.envs.registration import register

register(
    id='balancebot-v0',
    entry_point='balance_bot.envs:BalancebotEnv',
    max_episode_steps=5000,  # Optional: Define the max steps for each episode
)

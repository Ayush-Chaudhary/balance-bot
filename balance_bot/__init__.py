from gymnasium.envs.registration import register

register(
    id='balancebot-v0',
    entry_point='balance_bot.envs:BalancebotEnv',
)

register(
    id='balancebot_pid-v0',
    entry_point='balance_bot.envs:BalancebotEnvWithPID',
)

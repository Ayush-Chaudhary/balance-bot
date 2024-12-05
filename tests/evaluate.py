import gymnasium as gym
from balance_bot.envs.pid_navigation_env import BalancebotEnvWithNavigation
from stable_baselines3 import PPO, DQN

# Load the trained model
model = PPO.load("balancebot_navigation_model_PPO.zip")

# Create the environment
env = BalancebotEnvWithNavigation(render_mode="human")

observation, info = env.reset()
done = False
cumulative_reward = 0

while not done:
    env.render()
    # The model predicts the next action
    action, _states = model.predict(observation, deterministic=True)

    # Take the action in the environment
    observation, reward, done, truncated, info = env.step(action)

    # Accumulate rewards
    cumulative_reward += reward

    # (Optional) Terminate manually if needed
    if truncated:
        break

print(f"Cumulative Reward: {cumulative_reward}")
env.close()

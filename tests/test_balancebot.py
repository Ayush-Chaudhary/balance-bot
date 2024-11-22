import gymnasium as gym
import balance_bot

# Load the custom environment
env = gym.make('balancebot-v0', render_mode='human')

# Reset the environment
obs, info = env.reset()
print("Initial Observation:", obs)

# Simulate one episode
done = False
while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

# Close the environment
env.close()

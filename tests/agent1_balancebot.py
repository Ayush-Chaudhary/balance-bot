import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import balance_bot  # Assuming this is the custom BalanceBot environment

class TrainingCallback(BaseCallback):
    """
    Custom callback to stop training when the reward exceeds 199
    """
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if the episode reward list exists in locals
        if 'episode_rewards' in self.locals:
            # Ensure you are accessing the reward history correctly
            episode_rewards = self.locals['episode_rewards']
            if len(episode_rewards) > 100:
                # Check if average reward over last 100 episodes exceeds 199
                if sum(episode_rewards[-100:]) / 100 >= 199:
                    print("Solved!")
                    return False  # Stop training if solved
        return True


def main():
    # Create the environment
    env = gym.make("balancebot-v0")#, render_mode='human')

    # Create the learning agent using DQN from Stable-Baselines3
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000,
                exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=10)

    # Train the agent with the callback to stop training once solved
    model.learn(total_timesteps=200000, callback=TrainingCallback())

    # Save the trained model
    model.save("balance_model")

if __name__ == '__main__':
    main()

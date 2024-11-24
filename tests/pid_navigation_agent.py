from balance_bot.envs.pid_navigation_env import BalancebotEnvWithNavigation
from stable_baselines3 import DQN, PPO, A2C

def main():

    # Create the environment with the PID controller and custom terrain
    env = BalancebotEnvWithNavigation(render_mode='human')

    model_type = 'A2C'
    # Create the learning agent using the chosen RL algorithm
    if model_type == 'DQN':
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=100000,
                    exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=10)
    elif model_type == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-2, n_steps=20480, batch_size=64, n_epochs=10)
    elif model_type == 'A2C':
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, n_steps=50)
    else:
        raise ValueError("Unsupported model type")

    # Train the agent
    total_timesteps = 1000000
    env.render()
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save(f"balancebot_navigation_model_{model_type}")

    env.close()

if __name__ == '__main__':
    main()
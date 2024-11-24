import time
from balance_bot.envs.keyboard_nav_env import BalancebotEnvWithKeyboard

def main():
    # Create the environment with the PID controller and keyboard controls
    env = BalancebotEnvWithKeyboard(render_mode='human')

    # Reset the environment
    observation, info = env.reset()

    # Run the environment with the PID controller and keyboard controls
    done = False
    while not done:
        observation, reward, done, truncated, info = env.step(action=None)  # No action needed, PID controller and keyboard controls are used
        env.render()
        time.sleep(0.01)  # Adjust the sleep time to match the simulation time step

    env.close()

if __name__ == '__main__':
    main()
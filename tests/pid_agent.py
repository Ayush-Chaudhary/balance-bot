import gymnasium as gym
import time
import balance_bot  # Assuming this is the custom BalanceBot environment
from balance_bot.envs.balancebot_env_pid import BalancebotEnvWithPID
from balance_bot.envs.pid_yaw_pitch_env import BalancebotEnvWithPIDYawPitch

def main():
    # Create the environment
    # env = gym.make("balancebot-v0", render_mode='human')
    # env = BalancebotEnvWithPID(render_mode='human')
    env = BalancebotEnvWithPIDYawPitch(render_mode='human')

    # Reset the environment
    observation = env.reset()

    # Run the environment with the PID controller
    done = False
    while not done:
        observation, reward, done, truncated, info = env.step(action=None)  # No action needed, PID controller is used
        env.render()
        time.sleep(0.01)  # Adjust the sleep time to match the simulation time step

    env.close()

if __name__ == '__main__':
    main()
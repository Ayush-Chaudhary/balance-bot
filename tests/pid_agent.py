import gymnasium as gym
import time
import balance_bot  # Assuming this is the custom BalanceBot environment
from balance_bot.envs.balancebot_env_pid import BalancebotEnvWithPID
from balance_bot.envs.pid_yaw_pitch import BalancebotEnvWithPIDYawPitch

def main():
    # Create the environment
    # env = gym.make("balancebot-v0", render_mode='human')
    # env = BalancebotEnvWithPID(render_mode='human', kp=1500, ki=0.01, kd=0.1)
    env = BalancebotEnvWithPIDYawPitch(render_mode='human', kp_pitch=1500, ki_pitch=2, kd_pitch=0.1, kp_yaw=20, ki_yaw=0.1, kd_yaw=5)

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
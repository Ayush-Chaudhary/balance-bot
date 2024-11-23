import numpy as np
from scipy.optimize import minimize
import gymnasium as gym
import balance_bot  # Assuming this is the custom BalanceBot environment
from balance_bot.envs.pid_controller import PIDController
import pybullet as p
from balance_bot.envs.balancebot_env import BalancebotEnv


class BalancebotEnvWithPID(BalancebotEnv):
    def __init__(self, kp, ki, kd, render_mode=None):
        super().__init__(render_mode)
        self.pid = PIDController(kp, ki, kd)

    def step(self, action):
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        control_signal = self.pid.compute(euler[0], dt=0.01)
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=control_signal
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-control_signal
        )
        p.stepSimulation()
        self._env_step_counter += 1
        observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        truncated = False
        return np.array(observation, dtype=np.float32), reward, done, truncated, {}

def objective(params):
    kp, ki, kd = params
    env = BalancebotEnvWithPID(kp, ki, kd)
    observation, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        observation, reward, done, truncated, info = env.step(action=None)
        total_reward += reward
    env.close()
    return -total_reward  # Minimize the negative reward to maximize the reward

initial_guess = [1.0, 0.1, 0.05]
result = minimize(objective, initial_guess, method='Nelder-Mead')
optimal_kp, optimal_ki, optimal_kd = result.x

print(f"Optimal PID parameters: kp={optimal_kp}, ki={optimal_ki}, kd={optimal_kd}")
import numpy as np
import gymnasium as gym
import balance_bot  # Assuming this is the custom BalanceBot environment
import pybullet as p
from balance_bot.envs.balancebot_env import BalancebotEnv
from balance_bot.helper.pid_controller import PIDController
from balance_bot.helper import config


class BalancebotEnvWithPID(BalancebotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.pid = PIDController(config.KP_PITCH, config.KI_PITCH, config.KD_PITCH)

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
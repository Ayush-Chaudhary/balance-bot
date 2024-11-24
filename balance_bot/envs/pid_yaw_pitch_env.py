import pybullet as p
import numpy as np
from balance_bot.envs.balancebot_env import BalancebotEnv
from balance_bot.helper.pid_controller import PIDController
from balance_bot.helper import config

class BalancebotEnvWithPIDYawPitch(BalancebotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.pid_pitch = PIDController(config.KP_PITCH, config.KI_PITCH, config.KD_PITCH)
        self.pid_yaw = PIDController(config.KP_YAW, config.KI_YAW, config.KD_YAW)
        self.initial_orientation = None

    def reset(self):
        observation, info = super().reset()
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        self.initial_orientation = p.getEulerFromQuaternion(orientation)
        return observation, info

    def step(self, action):
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)

        # Compute control signals for pitch and yaw
        control_signal_pitch = self.pid_pitch.compute(euler[0], dt=0.01)
        control_signal_yaw = self.pid_yaw.compute(euler[2] - self.initial_orientation[2], dt=0.01)

        # Apply control signals to the robot
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=control_signal_pitch - control_signal_yaw
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-control_signal_pitch - control_signal_yaw
        )
        p.stepSimulation()
        self._env_step_counter += 1
        observation = self._compute_observation()
        reward = self._compute_reward(euler)
        done = self._compute_done()
        truncated = False
        return np.array(observation, dtype=np.float32), reward, done, truncated, {}

    def _compute_reward(self, euler):
        # Define a reward function that includes penalties for pitch and yaw deviations
        pitch_penalty = abs(euler[0])
        yaw_penalty = abs(euler[2] - self.initial_orientation[2])
        return np.float32((1 - pitch_penalty) * 0.1 - yaw_penalty * 0.01 - abs(self.vt) * 0.01)
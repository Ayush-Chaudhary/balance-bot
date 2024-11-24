import pybullet as p
import numpy as np
from balance_bot.envs.balancebot_env import BalancebotEnv
import keyboard
from balance_bot.helper import config
from balance_bot.helper.pid_controller import PIDController

class BalancebotEnvWithKeyboard(BalancebotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.pid_pitch = PIDController(config.KP_PITCH, config.KI_PITCH, config.KD_PITCH)
        self.pid_yaw = PIDController(config.KP_YAW, config.KI_YAW, config.KD_YAW)  # Secondary PID for yaw control
        self.initial_orientation = None

    def reset(self):
        observation, info = super().reset()
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        self.initial_orientation = p.getEulerFromQuaternion(orientation)
        return observation, info

    def step(self, action=None):
        # Get the current orientation and compute PID for pitch stabilization
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        
        # Compute pitch control signal for stabilization
        control_signal_pitch = self.pid_pitch.compute(euler[0], dt=0.01)
        
        # Apply keyboard controls for navigation
        throttle_force, left_turning_differential, right_turning_differential = self._apply_keyboard_controls()
        
        # Print the commands input by keyboard
        if throttle_force > 0:
            print("front")
        elif throttle_force < 0:
            print("back")
        if left_turning_differential > 0:
            print("left")
        elif left_turning_differential < 0:
            print("right")
        # Apply forces for forward/backward motion
        if throttle_force != 0:
            # Apply a direct force to move the robot base forward/backward
            p.applyExternalForce(
                objectUniqueId=self.bot_id,
                linkIndex=-1,  # Apply force to the base
                forceObj=[0, throttle_force, 0],  # Force vector (x, y, z)
                posObj=[0, 0, 0],  # Apply at the base center of mass
                flags=p.LINK_FRAME
            )
        
        # Combine stabilization and turning differential for wheel velocities
        left_wheel_velocity = control_signal_pitch - left_turning_differential
        right_wheel_velocity = -control_signal_pitch - right_turning_differential

        # Apply the combined velocities to the wheels
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=0,  # Left wheel
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_wheel_velocity
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,  # Right wheel
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_wheel_velocity
        )

        # Step the simulation
        p.stepSimulation()
        self._env_step_counter += 1
        observation = self._compute_observation()
        reward = self._compute_reward(euler)
        done = self._compute_done()
        truncated = False
        return np.array(observation, dtype=np.float32), reward, done, truncated, {}


    def _apply_keyboard_controls(self):
        # Define force and turning differential for keyboard controls
        throttle_force = 0
        left_turning_differential = 0
        right_turning_differential = 0

        if keyboard.is_pressed('up'):  # Move forward
            throttle_force += config.FORCE_MAGNITUDE
        if keyboard.is_pressed('down'):  # Move backward
            throttle_force -= config.FORCE_MAGNITUDE
        if keyboard.is_pressed('left'):  # Turn left
            left_turning_differential += config.TURNING_SPEED
            right_turning_differential += config.TURNING_SPEED
        if keyboard.is_pressed('right'):
            left_turning_differential -= config.TURNING_SPEED
            right_turning_differential -= config.TURNING_SPEED

        return throttle_force, left_turning_differential, right_turning_differential

    def _compute_reward(self, euler):
        # Define a reward function that includes penalties for pitch and yaw deviations
        pitch_penalty = abs(euler[0])
        yaw_penalty = abs(euler[2] - self.initial_orientation[2])
        return np.float32((1 - pitch_penalty) * 0.1 - yaw_penalty * 0.01 - abs(self.vt) * 0.01)
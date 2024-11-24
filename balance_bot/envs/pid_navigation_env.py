import os
import numpy as np
import pybullet as p
import gymnasium as gym
from scipy.ndimage import gaussian_filter
from balance_bot.envs.balancebot_env import BalancebotEnv
from balance_bot.helper.pid_controller import PIDController
from balance_bot.helper import config
import keyboard

class BalancebotEnvWithNavigation(BalancebotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.pid_pitch = PIDController(config.KP_PITCH, config.KI_PITCH, config.KD_PITCH)
        self.initial_orientation = None
        self.target_position = np.array(config.TARGET_POSITION)
        self.target_marker_id = None

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Define the action space: [yaw adjustment, force]
        self.action_space = gym.spaces.Box(
            low=np.array([-config.MAX_YAW_ADJUSTMENT, -config.MAX_FORCE]),
            high=np.array([config.MAX_YAW_ADJUSTMENT, config.MAX_FORCE]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed)
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        self.initial_orientation = p.getEulerFromQuaternion(orientation)

        # Add a visual marker at the target position
        self._add_target_marker()

        return observation, info

    def step(self, action):
        # Action format: [yaw_adjustment, forward_force]
        yaw_adjustment, forward_force = action

        # Apply force-based navigation
        self._apply_force_navigation(forward_force)

        # Get the current orientation
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)

        # Compute pitch stabilization
        control_signal_pitch = self.pid_pitch.compute(euler[0], dt=0.01)

        # Compute yaw adjustments
        yaw = euler[2] - self.initial_orientation[2]
        left_wheel_velocity = control_signal_pitch - yaw_adjustment
        right_wheel_velocity = -control_signal_pitch - yaw_adjustment

        # Apply velocities to the wheels
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_wheel_velocity
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_wheel_velocity
        )

        # Step the simulation
        p.stepSimulation()
        self._env_step_counter += 1
        observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        truncated = False
        return np.array(observation, dtype=np.float32), reward, done, truncated, {}

    def _apply_force_navigation(self, forward_force):
        """Apply forward/backward force to the robot's base."""
        if forward_force != 0:
            p.applyExternalForce(
                objectUniqueId=self.bot_id,
                linkIndex=-1,  # Apply force to the base
                forceObj=[forward_force, 0, 0],  # Force vector (x, y, z)
                posObj=[0, 0, 0],  # Apply at the base center of mass
                flags=p.LINK_FRAME
            )

    def _compute_observation(self):
        """Include useful states for RL."""
        position, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        linear_vel, angular_vel = p.getBaseVelocity(self.bot_id)

        # Observation: [pitch, pitch velocity, yaw, yaw velocity, x, y]
        return [
            np.float32(euler[0]),         # Pitch angle
            np.float32(angular_vel[0]),  # Pitch velocity
            np.float32(euler[2]),        # Yaw angle
            np.float32(angular_vel[2]),  # Yaw velocity
            np.float32(position[0]),     # x-position
            np.float32(position[1])      # y-position
        ]
    
    def _add_target_marker(self):
        """Add a visual marker at the target position."""
        if self.target_marker_id is not None:
            p.removeBody(self.target_marker_id)
        self.target_marker_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0, 0, 1]  # Red color
        )
        p.createMultiBody(
            baseVisualShapeIndex=self.target_marker_id,
            basePosition=self.target_position
        )

    def _compute_reward(self):
        """Reward function with penalties for instability and distance."""
        position, _ = p.getBasePositionAndOrientation(self.bot_id)
        distance_to_target = np.linalg.norm(np.array(position[:3]) - self.target_position)

        # Reward: Proximity to target, penalties for pitch instability and yaw error
        pitch_penalty = abs(self._compute_observation()[0])  # Pitch angle
        yaw_penalty = abs(self._compute_observation()[2])   # Yaw angle
        reward = -distance_to_target - 0.1 * pitch_penalty - 0.5 * yaw_penalty
        return reward

    def _compute_done(self):
        """Terminate episode if close to the target or if unstable."""
        position, _ = p.getBasePositionAndOrientation(self.bot_id)
        distance_to_target = np.linalg.norm(np.array(position[:3]) - self.target_position)

        # End episode if robot is close to the target or has fallen over
        pitch_angle = abs(self._compute_observation()[0])
        if distance_to_target < 0.1 :
            print("Target reached!")
            return True
        elif pitch_angle > np.pi / 4:
            print("Robot fell over!")
            return True
        elif keyboard.is_pressed('q'):
            print("User terminated the episode!")
            return True
        else:
            return False
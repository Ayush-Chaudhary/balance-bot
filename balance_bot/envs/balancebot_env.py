import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
from .pid_controller import PIDController

class BalancebotEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50,
        'video.frames_per_second': 50
    }

    def __init__(self, render_mode=None):
        super(BalancebotEnv, self).__init__()
        self.render_mode = render_mode
        self._observation = np.array([], dtype=np.float32)
        self.action_space = spaces.Discrete(9)  # Discrete actions
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -5], dtype=np.float32),  # Example: pitch, angular velocity, throttle
            high=np.array([np.pi, np.pi, 5], dtype=np.float32),
            dtype=np.float32
        )
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)  # Choose render mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Allow PyBullet to find URDFs
        self.bot_id = None
        self._seed()  # Optionally set a seed for reproducibility
        
        self._env_step_counter = 0
        self.vt = np.float32(0)  # Initialize velocity
        self.pid = PIDController(kp=1, ki=0, kd=1)  # Initialize PID controller

        # Initialize simulation
        self._initialize_simulation()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _initialize_simulation(self):
        # Connect to PyBullet
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -10)  # Set gravity
        p.setTimeStep(0.01)
        p.loadURDF("plane.urdf")  # Load ground plane
        self.vt = np.float32(0)  # Reset velocity to initial value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(0.01)  # Time step

        # Load plane and robot
        p.loadURDF("plane.urdf")
        cube_start_pos = [0, 0, 0.1]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.bot_id = p.loadURDF(
            os.path.join(path, "balancebot.xml"),
            cube_start_pos,
            cube_start_orientation
        )

        # Initialize observation
        self._observation = self._compute_observation()
        self._env_step_counter = 0  # Reset step counter
        info = {}  # Any extra information
        return np.array(self._observation, dtype=np.float32), info

    def step_old(self, action):
        self._apply_action(action)
        p.stepSimulation()

        observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        truncated = self._env_step_counter >= 1500  # Maximum steps for truncation

        self._env_step_counter += 1

        return np.array(self._observation, dtype=np.float32), reward, done, truncated, {}
    
    def step(self, action):
        # Compute control signal using PID controller
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        control_signal = self.pid.compute(euler[0], dt=0.01)  # Assuming a time step of 0.01s

        # Apply control signal to the robot
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=control_signal
        )

        # Step the simulation
        p.stepSimulation()
        self._env_step_counter += 1

        # Compute observation, reward, and done
        observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        truncated = False  # You can set this based on your environment's logic

        return np.array(observation, dtype=np.float32), reward, done, truncated, {}

    def _apply_action(self, action):
        # Map action to velocity changes
        delta_v = np.float32(0.1)
        throttle = [-10 * delta_v, -5 * delta_v, -2 * delta_v, -delta_v, 0, delta_v, 2 * delta_v, 5 * delta_v, 10 * delta_v][action]
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=throttle
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.bot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-throttle
        )

    def _compute_observation(self):
        # Obtain the robot's state
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        linear_vel, angular_vel = p.getBaseVelocity(self.bot_id)

        # Return observation (customize as needed)
        return [np.float32(euler[0]), np.float32(angular_vel[0]), self.vt]

    def _compute_reward(self):
        # Define a reward function (customize for your task)
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        return np.float32((1 - abs(euler[0])) * 0.1 - abs(self.vt) * 0.01)  # Example reward

    def _compute_done(self):
        # Check termination conditions
        position, _ = p.getBasePositionAndOrientation(self.bot_id)
        return position[2] < 0.05 or self._env_step_counter >= 1500  # If the robot falls over

    def render(self):
        if self.render_mode == "human":
            # Set the camera position and target
            p.resetDebugVisualizerCamera(
                cameraDistance=1,  # Adjust the distance to zoom in 4 times
                cameraYaw=50,        # Adjust the yaw angle
                cameraPitch=-35,     # Adjust the pitch angle
                cameraTargetPosition=[0, 0, 0.5]  # Adjust the target position
            )
        elif self.render_mode == "rgb_array":
            # Handle RGB array rendering if needed
            pass

    def close(self):
        # Clean up the environment
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

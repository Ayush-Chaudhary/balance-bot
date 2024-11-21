import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class BalancebotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(BalancebotEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)  # Discrete actions
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -5]),  # Example: pitch, angular velocity, throttle
            high=np.array([np.pi, np.pi, 5]),
            dtype=np.float32
        )
        self.physics_client = None
        self.bot_id = None

        # Initialize simulation
        self._initialize_simulation()

    def _initialize_simulation(self):
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load URDF files
        p.setGravity(0, 0, -10)  # Set gravity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -10)

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
        observation = self._get_observation()
        info = {}  # Any extra information
        return observation, info

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()

        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        info = {}

        return observation, reward, done, False, info

    def _apply_action(self, action):
        # Map action to velocity changes
        delta_v = 0.1
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

    def _get_observation(self):
        # Obtain the robot's state
        _, orientation = p.getBasePositionAndOrientation(self.bot_id)
        euler = p.getEulerFromQuaternion(orientation)
        linear_vel, angular_vel = p.getBaseVelocity(self.bot_id)

        # Return observation (customize as needed)
        return np.array([euler[0], angular_vel[0], 0.0], dtype=np.float32)

    def _compute_reward(self):
        # Define a reward function (customize for your task)
        return 0.0  # Placeholder

    def _compute_done(self):
        # Check termination conditions
        position, _ = p.getBasePositionAndOrientation(self.bot_id)
        return position[2] < 0.1  # If the robot falls over

    def render(self):
        if self.render_mode == "human":
            # Render is handled by PyBullet GUI
            pass
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
        # Clean up the environment
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

import robamine.utils as arl
import math

class Clutter(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        """
        The constructor of the environment.
        """

        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/clutter.xml")

        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self._viewers = {}

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1, -1, -1]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.float32)

        self.object_names = ['object1', 'object2', 'object3']

        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)
        self.seed()

    def reset_model(self):

        # Randomize the position of the obstracting objects
        random_qpos = self.init_qpos
        for object_name in self.object_names:
            index = self.sim.model.get_joint_qpos_addr(object_name)
            r = abs(np.random.normal(0, 0.01)) + 0.05
            theta = np.random.uniform(0, 2*math.pi)
            random_qpos[index[0]] = r * math.cos(theta)
            random_qpos[index[0]+1] = r * math.sin(theta)

        self.set_state(random_qpos, self.init_qvel)
        return self.get_obs()

    def get_obs(self):
        return self.observation_space.sample()

    def step(self, action):
        done = False
        obs = self.get_obs()
        reward = self.get_reward(obs)
        time = self.do_simulation(action)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {}

    def do_simulation(self, action):
        time = self.sim.data.time

        # Move forward the simulation
        self.sim.step()
        return time

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def get_reward(self, observation):
        reward = 0
        return reward

    def terminal_state(self, observation):
        return False

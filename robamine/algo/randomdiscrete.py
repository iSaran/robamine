"""
Random discrete policy
===========

A dummy agent which produces random actions. Used for testing.
"""

from robamine.algo.core import Agent
import numpy as np

class RandomDiscrete(Agent):
    def __init__(self, state_dim, action_dim, params = {}):
        super(RandomDiscrete, self).__init__(state_dim, action_dim, 'RandomDiscrete')
        self.rng = np.random.RandomState()

    def explore(self, state):
        return self.rng.randint(self.action_dim)

    def predict(self, state):
        return self.rng.randint(self.action_dim)

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

    def seed(self, seed):
        self.rng.seed(seed)

    def save(self, path):
        pass

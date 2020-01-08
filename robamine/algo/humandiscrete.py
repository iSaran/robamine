"""
Human discrete policy
===========

A dummy agent which produces random actions. Used for testing.
"""

from robamine.algo.core import RLAgent
import numpy as np

class HumanDiscrete(RLAgent):
    def __init__(self, state_dim, action_dim, params = {}):
        super(HumanDiscrete, self).__init__(state_dim, action_dim, 'HumanDiscrete')
        self.rng = np.random.RandomState()

    def explore(self, state):
        return self.rng.randint(self.action_dim)

    def predict(self, state):
        return int(input('Enter action (0-' + str(self.action_dim) + '): '))

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

    def seed(self, seed):
        self.rng.seed(seed)

    def save(self, path):
        pass

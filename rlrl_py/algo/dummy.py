"""
Dummy Agent
===========

A dummy agent which produces random actions. Used for testing.
"""

from robamine.algo.core import Agent

class Dummy(Agent):
    def __init__(self, sess, env, random_seed=999, log_dir='/tmp', console=True):
        super(Dummy, self).__init__(sess, env, random_seed, log_dir, "Dummy", console)

    def explore(self, state):
        return self.env.action_space.sample()

    def predict(self, state):
        return self.env.action_space.sample()

    def learn(self, state, action, reward, next_state, done):
        pass

    def q_value(self, state, action):
        return 0.0

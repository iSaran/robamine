"""
Dummy Agent
===========

A dummy agent which produces random actions. Used for testing.
"""

from robamine.algo.core import Agent
from robamine.algo.util import Logger, Stats

class Dummy(Agent):
    def __init__(self, sess, env, random_seed=999, log_dir='/tmp', console=True):
        super(Dummy, self).__init__(sess, env, random_seed, log_dir, "Dummy")

        self.logger = Logger(self.sess, self.log_dir, self.name, self.env.spec.id)
        self.train_stats = Stats(dt=0.02, logger=self.logger, timestep_stats = ['reward', 'q_value'], name = "train")
        self.eval_stats = Stats(dt=0.02, logger=self.logger, timestep_stats = ['reward', 'q_value'], name = "eval")
        self.eval_episode_batch = 0
        self.logger.init_tf_writer()

    def explore(self, state):
        return self.env.action_space.sample()

    def predict(self, state):
        return self.env.action_space.sample()

    def learn(self, state, action, reward, next_state, done):
        pass

    def q_value(self, state, action):
        return 0.0

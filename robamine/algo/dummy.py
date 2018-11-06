"""
Dummy Agent
===========

A dummy agent which produces random actions. Used for testing.
"""

from robamine.algo.core import Agent, AgentParams
from robamine.algo.util import Logger, Stats

class DummyParams(AgentParams):
    def __init__(self):
        super().__init__(name="Dummy")

class Dummy(Agent):
    def __init__(self, sess, params, action_space):
        super(Dummy, self).__init__(sess, params)
        self.action_space = action_space

        # self.logger = Logger(self.sess, self.log_dir, self.name, self.env.spec.id)
        # self.train_stats = Stats(dt=0.02, logger=self.logger, timestep_stats = ['reward', 'q_value'], name = "train")
        # self.eval_stats = Stats(dt=0.02, logger=self.logger, timestep_stats = ['reward', 'q_value'], name = "eval")
        # self.eval_episode_batch = 0
        # self.logger.init_tf_writer()

    def explore(self, state):
        return self.action_space.sample()

    def predict(self, state):
        return self.action_space.sample()

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

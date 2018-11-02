import unittest

import tensorflow as tf
import numpy as np
import os

from robamine.algo.dummy import Dummy
from robamine.algo.util import Plotter

class TestAgent(unittest.TestCase):
    def test_construct(self):
        with tf.Session() as sess:
            agent = Dummy(sess, 'Pendulum-v0', random_seed=999, console = False)
            agent.train(n_episodes=50, episode_batch_size=5, episodes_to_evaluate=10)

            streams = ['train_episode', 'train_batch', 'eval_episode', 'eval_batch']
            pl = Plotter(agent.logger.log_path, streams)
            pl_2 = Plotter(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robamine_logger_Dummy_Pendulum-v0_2018.10.30.16.54.48.378300'), streams)

            for stream in streams:
                x, y = pl.extract_data(stream)
                x_d, y_d = pl_2.extract_data(stream)
                for y_var in y_d:
                    error = np.array(y_d[y_var]) - np.array(y[y_var])
                    self.assertEqual(np.max(error), 0.0)

if __name__ == '__main__':
    unittest.main()

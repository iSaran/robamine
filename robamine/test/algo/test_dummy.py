import unittest

import tensorflow as tf
import numpy as np
import os

import robamine as rm

class TestAgent(unittest.TestCase):
    def test_construct(self):
        rm.rb_logging.init(console_level=rm.rb_logging.ERROR, file_level=rm.rb_logging.ERROR)
        world = rm.World('Dummy', 'Pendulum-v0')
        world.seed(999)
        world.train_and_eval(n_episodes_to_train=50, n_episodes_to_evaluate=10, evaluate_every=5)
        world.plot(10)
        streams = ['train', 'batch_train', 'eval', 'batch_eval']
        pl = rm.Plotter(world.log_dir, streams)
        pl_2 = rm.Plotter(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robamine_logs_2019.04.19.16.25.29.987332/Dummy_Pendulum-v0'), streams)


        for stream in streams:
            x, y, _, _= pl.extract_data(stream)
            x_d, y_d, _, _ = pl_2.extract_data(stream)
            for y_var in y_d:
                error = np.array(y_d[y_var]) - np.array(y[y_var])
                self.assertEqual(np.max(error), 0.0)

if __name__ == '__main__':
    unittest.main()

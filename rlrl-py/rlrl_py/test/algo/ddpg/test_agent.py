import unittest
from rlrl_py.algo.ddpg.agent import DDPG
import tensorflow as tf
import numpy as np

class Agent(unittest.TestCase):
    def test_construct(self):
        with tf.Session() as sess:
            agent = DDPG(sess, 'SphereReacher-v1').train(10, render=False)

if __name__ == '__main__':
    unittest.main()

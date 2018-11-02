import unittest
import robamine.algo.util as util
import tensorflow as tf

class TestUtil(unittest.TestCase):
    def test_logger(self):
        with tf.Session() as sess:
            logger = util.Logger(sess, '/tmp', 'mitsos', 'env', console=False)

if __name__ == '__main__':
    unittest.main()

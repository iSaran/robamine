import unittest
import robamine.algo.util as util
import tensorflow as tf

class TestUtil(unittest.TestCase):
    def test_datastream(self):
        with tf.Session() as sess:

            tf_writer = tf.summary.FileWriter('/tmp', sess.graph)
            datastream = util.DataStream(sess, '/tmp', tf_writer, ['var1', 'var2'], 'mitsos')

if __name__ == '__main__':
    unittest.main()

import unittest
import robamine.algo.util as util
import tensorflow as tf
import numpy as np

class TestUtil(unittest.TestCase):
    def test_datastream(self):
        with tf.Session() as sess:

            tf_writer = tf.summary.FileWriter('/tmp', sess.graph)
            datastream = util.DataStream(sess, '/tmp', tf_writer, ['var1', 'var2'], 'mitsos')

class TestDataset(unittest.TestCase):
    def test_normalization(self):
        dataset = util.Dataset()
        dataset.append(util.Datapoint(x=np.array([12, 232]), y=np.array([18, 28, 21])))
        dataset.append(util.Datapoint(x=np.array([2, -3]), y=np.array([180, -8, 0])))
        dataset.append(util.Datapoint(x=np.array([54, -22]), y=np.array([138, 18, 451])))
        x, y = dataset.to_array()
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

        dataset.normalize()

        for i in range(len(dataset)):
            for j in range(len(dataset[i].x)):
                self.assertEqual(dataset[i].x[j], x[i][j])

            for j in range(len(dataset[i].y)):
                self.assertEqual(dataset[i].y[j], y[i][j])

    def test_rescaling(self):
        dataset = util.Dataset()
        dataset.append(util.Datapoint(x=np.array([12, 232]), y=np.array([18, 28, 21])))
        dataset.append(util.Datapoint(x=np.array([2, -3]), y=np.array([180, -8, 0])))
        dataset.append(util.Datapoint(x=np.array([54, -22]), y=np.array([138, 18, 451])))

        result = util.Dataset()
        result.append(util.Datapoint(x=np.array([0.19230769230769232, 1.0]), y=np.array([0, 1, 0.04656319290465632])))
        result.append(util.Datapoint(x=np.array([0, 0.07480314960629922]), y=np.array([1, 0, 0])))
        result.append(util.Datapoint(x=np.array([1, 0]), y=np.array([0.7407407407407407, 0.72222222222222222, 1])))

        dataset.rescale()

        for i in range(len(dataset)):
            for j in range(len(dataset[i].x)):
                self.assertEqual(dataset[i].x[j], result[i].x[j])

            for j in range(len(dataset[i].y)):
                self.assertEqual(dataset[i].y[j], result[i].y[j])

    def test_split(self):
        dataset = util.Dataset()
        for i in range(100):
            dataset.append(util.Datapoint(x=np.array([12, 232]), y=np.array([18, 28, 21])))
        train, test = dataset.split(0.7)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(test), 30)

        dataset = util.Dataset()
        for i in range(213):
            dataset.append(util.Datapoint(x=np.array([12, 232]), y=np.array([18, 28, 21])))
        train, test = dataset.split(0.7)
        self.assertEqual(len(train), 149)
        self.assertEqual(len(test), 64)

        dataset = util.Dataset()
        dataset.append(util.Datapoint(x=np.array([12, 232]), y=np.array([18, 28, 21])))
        dataset.append(util.Datapoint(x=np.array([2, -3]), y=np.array([180, -8, 0])))
        dataset.append(util.Datapoint(x=np.array([54, -22]), y=np.array([138, 18, 451])))
        train, test = dataset.split(0.7)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 1)

if __name__ == '__main__':
    unittest.main()

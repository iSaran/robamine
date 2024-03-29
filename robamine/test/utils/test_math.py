import unittest
import numpy as np
from robamine.utils.math import rescale, rescale_array, Signal
import numpy.testing as np_test

class TestRescale(unittest.TestCase):
    def test_rescale(self):
        self.assertEqual(rescale(5, 0, 10), 0.5)

    def test_rescale_array(self):
        # Test a 2D array with shape (2, 3)
        x = np.array([[1, 2], [-2, 4], [0., -9]])
        result = np.array([[1, 0.84615385], [0, 1], [0.666666667, 0]])
        rescale = rescale_array(x, range=[0, 1], axis=0)
        reversed = rescale_array(rescale, min=np.min(x, axis=0), max=np.max(x, axis=0), range=[0, 1], axis=0, reverse=True)
        np_test.assert_array_almost_equal(rescale, result)
        np_test.assert_array_almost_equal(reversed, x)

        # Test a 1D array with shape (4,)
        x = np.array([1, 2, 3, 7])
        result = np.array([0, 0.166666667, 0.3333333, 1])
        rescale = rescale_array(x, range=[0, 1], axis=0)
        reversed = rescale_array(rescale, min=np.min(x), max=np.max(x), range=[0, 1], axis=0, reverse=True)
        np_test.assert_array_almost_equal(rescale, result)
        np_test.assert_array_almost_equal(reversed, x)

        # Test a 1D array with shape (1,)
        x = np.array([5])
        result = np.array([0, 0.166666667, 0.3333333, 1])
        with self.assertRaises(AssertionError):
            rescale = rescale_array(x, range=[0, 1], axis=0)

        # Test a 1D array with shape (2,)
        x = np.array([5, 2])
        result = np.array([1, 0])
        rescale = rescale_array(x, range=[0, 1], axis=0)
        np_test.assert_array_almost_equal(rescale, result)
        reversed = rescale_array(rescale, min=np.min(x), max=np.max(x), range=[0, 1], axis=0, reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

        # None min, max with matrix along axis=0
        x = np.array([[5, 2], [-5, 20], [15, 4], [3, 0]])
        result = np.array([[0.5, 0.1], [0, 1], [1, 0.2], [0.4, 0]])
        rescale = rescale_array(x, range=[0, 1], axis=0)
        np_test.assert_array_almost_equal(rescale, result)
        reversed = rescale_array(rescale, min=np.min(x, axis=0), max=np.max(x, axis=0), range=[0, 1], axis=0, reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

        # Scalar min max for a vector
        x = np.array([5, 2, 1, 2, 10, 100])
        result = x / 10
        rescale = rescale_array(x, min=0, max=10, range=[0, 1])
        np_test.assert_array_equal(rescale, result)
        reversed = rescale_array(rescale, min=0, max=10, range=[0, 1], reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

        # Scalar min max for a matrix
        x = np.array([[5, 2, 1, 2, 10, 100], [5, 2, 1, 2, 10, 100]])
        result = x / 10
        rescale = rescale_array(x, min=0, max=10, range=[0, 1])
        np_test.assert_array_equal(rescale, result)
        reversed = rescale_array(rescale, min=0, max=10, range=[0, 1], reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

        # Vector min max for a vector x
        x = np.array([5, 2, 1, 2, 10, 100])
        result = np.array([1, 0.5, 1, 1, 0.1, 0])
        rescale = rescale_array(x, min=np.array([0, 0, 0, 1, 0, 100]), max=np.array([5, 4, 1, 2, 100, 200]), range=[0, 1])
        np_test.assert_array_equal(rescale, result)
        reversed = rescale_array(rescale, min=np.array([0, 0, 0, 1, 0, 100]), max=np.array([5, 4, 1, 2, 100, 200]), range=[0, 1], reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

        # Vector min max for a matrix x
        x = np.array([[5, 2, 1, 2, 10, 100], [5, 2, 1, 2, 10, 100]])
        result = np.array([[1, 0.5, 1, 1, 0.1, 0], [1, 0.5, 1, 1, 0.1, 0]])
        rescale = rescale_array(x, min=np.array([0, 0, 0, 1, 0, 100]), max=np.array([5, 4, 1, 2, 100, 200]), range=[0, 1])
        np_test.assert_array_equal(rescale, result)
        reversed = rescale_array(rescale, min=np.array([0, 0, 0, 1, 0, 100]), max=np.array([5, 4, 1, 2, 100, 200]), range=[0, 1], reverse=True)
        np_test.assert_array_almost_equal(reversed, x)

class TestSignal(unittest.TestCase):
    def test_average_filter(self):

        # Test that it throws assertion error
        x = np.arange(1002)
        y = np.sin(2 * np.pi * 0.001 * x) + 0.2 * np.random.randn(1002)
        signal = Signal(y.reshape(-1, 1))
        with self.assertRaises(AssertionError):
            signal.average_filter(segments=100)

        with self.assertRaises(AssertionError):
            signal.segment_last_element(segments=100)

        # Test one dimensional
        x = np.arange(1000)
        y = np.sin(2 * np.pi * 0.001 * x) + 0.2 * np.random.randn(1000)
        signal = Signal(y.reshape(-1, 1))
        signal2 = Signal(y.reshape(-1, 1))
        # signal2.plot()
        signal.average_filter(segments=100)
        signal2.segment_last_element(segments=100)
        # signal2.plot()

        # Test multidimensional
        x = np.arange(1000)
        y = np.sin(2 * np.pi * 0.001 * x) + 0.2 * np.random.randn(1000)
        y = y.reshape(-1, 1)
        y2 = y + np.ones(y.shape) * 0.2
        yy = np.concatenate((y, y2), axis=1)
        y2 = y + np.ones(y.shape) * 0.4
        yy = np.concatenate((yy, y2), axis=1)
        y2 = y + np.ones(y.shape) * 0.6
        yy = np.concatenate((yy, y2), axis=1)
        signal = Signal(yy)
        signal2 = Signal(yy)
        # signal.plot()
        # print(signal2.array()[:20, :])
        signal.average_filter(segments=100)
        signal2.segment_last_element(segments=100)
        # print(signal2.array())
        # signal.plot()

if __name__ == '__main__':
    unittest.main()

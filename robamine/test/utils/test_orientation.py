import unittest
from robamine.utils.orientation import Quaternion
import numpy.testing as np_test
import numpy as np

class TestQuaternion(unittest.TestCase):
    def test_to_rotation_matrix(self):
        q = Quaternion(w=1, x=0, y=0, z=0)
        np_test.assert_array_almost_equal(np.eye(3), q.rotation_matrix())

    def test_from_rotation_matrix(self):
        q = Quaternion.from_rotation_matrix(np.eye(3))

        self.assertEqual(q.w, 1)
        self.assertEqual(q.x, 0)
        self.assertEqual(q.y, 0)
        self.assertEqual(q.z, 0)

if __name__ == '__main__':
    unittest.main()

import unittest
from robamine.utils.orientation import Quaternion, Affine3
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

    def test_rot_z(self):
        q = Quaternion(w=1, x=0, y=0, z=0)
        q.rot_z(3.1415)
        self.assertAlmostEqual(q.w, 0, 3)
        self.assertAlmostEqual(q.x, 0, 3)
        self.assertAlmostEqual(q.y, 0, 3)
        self.assertAlmostEqual(q.z, 1, 3)


class TestAffine3(unittest.TestCase):
    def test_init(self):
        x = Affine3()
        np.testing.assert_equal(x.matrix(), np.eye(4))
        np.testing.assert_equal(x.translation, np.zeros(3))
        np.testing.assert_equal(x.linear, np.eye(3))

    def test_from_matrix(self):
        y = np.array([[1, 2, 3, 10], [4, 5, 6, 20], [7, 8, 9, 30]])
        x = Affine3.from_matrix(y)

        expected = np.array([[1., 2., 3., 10.],
                             [4., 5., 6., 20.],
                             [7., 8., 9., 30.],
                             [0., 0., 0.,  1.]])
        np.testing.assert_equal(x.matrix(), expected)

        np.testing.assert_equal(x.translation, np.array([10, 20, 30]))

        expected = np.array([[1., 2., 3.],
                             [4., 5., 6.],
                             [7., 8., 9.]])
        np.testing.assert_equal(x.linear, expected)

    def test_from_pos_quat(self):
        x = Affine3.from_vec_quat(np.array([0, 1, 2]), Quaternion(1, 0, 0, 0))

        expected = np.array([[1, 0, 0, 0.],
                             [0, 1, 0, 1.],
                             [0, 0, 1, 2.],
                             [0, 0, 0, 1.]])
        np.testing.assert_equal(x.matrix(), expected)

        np.testing.assert_equal(x.translation, np.array([0, 1, 2]))
        np.testing.assert_equal(x.linear, np.eye(3))

    def test_mul(self):
        rot_a = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        affine_a = Affine3.from_vec_quat(np.array([4, 3, 0]), Quaternion.from_rotation_matrix(rot_a))
        rot_b = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        affine_b = Affine3.from_vec_quat(np.array([5, -2, 0]), Quaternion.from_rotation_matrix(rot_b))
        diff = affine_a.inv() * affine_b
        expected = np.array([[1, 0, 0, 5], [0, 0, 1, 1], [0, -1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_almost_equal(diff.matrix(), expected)



if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import numpy.testing as npt

from fftoptionlib.helper import (
    to_array_atleast_1d,
    to_array_with_same_dimension,
)


class HelperTest(unittest.TestCase):
    def test_to_array_atleast_1d(self):
        res = to_array_atleast_1d(1, 2, 3)
        exp_res = np.array([1]), np.array([2]), np.array([3])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 1)

    def test_to_array_atleast_1d_2(self):
        res = to_array_atleast_1d(1, 2, None)
        exp_res = np.array([1]), np.array([2]), np.array([None])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 1)

    def test_to_array_atleast_1d_3(self):
        res = to_array_atleast_1d([1, 2, 3], 2, None)
        exp_res = np.array([1, 2, 3]), np.array([2]), np.array([None])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 3)

    def test_to_array_with_same_dimension(self):
        res = to_array_with_same_dimension([1, 2, 3], 2, None)
        exp_res = np.array([1, 2, 3]), np.array([2, 2, 2]), np.array([None, None, None])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 3)

    def test_to_array_with_same_dimension_2(self):
        res = to_array_with_same_dimension(1, 2, 3)
        exp_res = np.array([1]), np.array([2]), np.array([3])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 1)

    def test_to_array_with_same_dimension_3(self):
        res = to_array_with_same_dimension(np.array([1, 2, 3]), np.array(2), None)
        exp_res = np.array([1, 2, 3]), np.array([2, 2, 2]), np.array([None, None, None])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])
        self.assertEqual(res[0].shape[0], 3)

    def test_to_array_with_same_dimension_4(self):
        self.assertRaises(ValueError, to_array_with_same_dimension, np.array([1, 2, 3]), [2, 3], None)

    def test_to_array_with_same_dimension_5(self):
        res = to_array_with_same_dimension([1, 2, 3], 'put', None)
        exp_res = np.array([1, 2, 3]), np.array(['put', 'put', 'put']), np.array([None, None, None])
        npt.assert_array_equal(res[0], exp_res[0])
        npt.assert_array_equal(res[1], exp_res[1])
        npt.assert_array_equal(res[2], exp_res[2])

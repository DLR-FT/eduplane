# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.utils.utils.

Run from top-level package directory:

   python3 -m unittest tests.utils.test_utils

"""

import unittest

from eduplane.utils.utils import *


class UtilsTest(unittest.TestCase):

    def test_forward_func(self):
        pass
        # print(proxy_func.__signature__)
        # print(proxy_func(1, 2, 4))

    def test_angular_diff(self):
        rad = numpy.radians

        # 0° -> 10°
        self.assertAlmostEqual(angular_diff(0, rad(10)), rad(10))
        self.assertAlmostEqual(angular_diff(0, rad(10), 1), rad(10))
        self.assertAlmostEqual(angular_diff(0, rad(10), -1), -rad(350))

        # 0° -> -10°
        self.assertAlmostEqual(angular_diff(0, rad(-10)), rad(-10))
        self.assertAlmostEqual(angular_diff(0, rad(-10), 1), rad(350))
        self.assertAlmostEqual(angular_diff(0, rad(-10), -1), rad(-10))

        # 0° -> 350°
        self.assertAlmostEqual(angular_diff(0, rad(350)), rad(-10))
        self.assertAlmostEqual(angular_diff(0, rad(350), 1), rad(350))
        self.assertAlmostEqual(angular_diff(0, rad(350), -1), rad(-10))

        # 0° -> -350°
        self.assertAlmostEqual(angular_diff(0, rad(-350)), rad(10))
        self.assertAlmostEqual(angular_diff(0, rad(-350), 1), rad(10))
        self.assertAlmostEqual(angular_diff(0, rad(-350), -1), rad(-350))

        # 10° -> 350°
        self.assertAlmostEqual(angular_diff(rad(10), rad(350)), rad(-20))
        self.assertAlmostEqual(angular_diff(rad(10), rad(350), 1), rad(340))
        self.assertAlmostEqual(angular_diff(rad(10), rad(350), -1), rad(-20))

        # 10° -> -350°
        self.assertAlmostEqual(angular_diff(rad(10), rad(-350)), rad(0))
        self.assertAlmostEqual(angular_diff(rad(10), rad(-350), 1), rad(0))
        self.assertAlmostEqual(angular_diff(rad(10), rad(-350), -1), rad(0))

        # Test vectorization
        self.assertTrue(
            numpy.allclose(
                angular_diff(0, rad([10, 20, 30])), rad([10, 20, 30])))
        self.assertTrue(
            numpy.allclose(
                angular_diff(0, rad([10, 20, 30]), 1), rad([10, 20, 30])))
        self.assertTrue(
            numpy.allclose(
                angular_diff(0, rad([10, 20, 30]), -1), rad([-350, -340,
                                                             -330])))

    def test_listify(self):
        self.assertEqual(listify("one"), ["one"])
        self.assertEqual(listify(["one"]), ["one"])
        self.assertEqual(listify("one", [[], []]), [["one"], ["one"]])
        self.assertEqual(listify(["one"], [[], []]), [["one"], ["one"]])
        self.assertEqual(
            listify(["one", "two"], [[], []]), [["one", "two"], ["one", "two"]])
        self.assertEqual(
            listify([["one"], ["two"]], [[], []]), [["one"], ["two"]])
        self.assertEqual(listify("one", repeat=3), ["one", "one", "one"])

    def test_clip_abs(self):
        self.assertAlmostEqual(clip_abs(5, 10), 5)
        self.assertAlmostEqual(clip_abs(5, 5), 5)
        self.assertAlmostEqual(clip_abs(5, 4), 4)
        self.assertAlmostEqual(clip_abs(-5, 10), -5)
        self.assertAlmostEqual(clip_abs(-5, 5), -5)
        self.assertAlmostEqual(clip_abs(-5, 4), -4)

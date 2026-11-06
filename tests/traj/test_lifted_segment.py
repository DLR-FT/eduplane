# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.lifted_segment.

Run from top-level package directory:

   python3 -m unittest tests.traj.lifted_segment

"""

import math
import unittest

from eduplane.traj.horizontal_profiles import ConstSpeedStraightFlight
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile


class LiftedSegmentTests(unittest.TestCase):

    def test_interface(self):
        # Interface test with simplest lifted segment
        seg = LiftedSegment(
            horiz_profile=ConstSpeedStraightFlight(tf=10, V_A_xy=20, chi=0),
            vert_profile=ConstVertSpeedProfile(tf=10, z_g_t0=0, w_A_g=10))
        values = seg.x0
        self.assertTupleEqual(tuple(values.keys()), LiftedSegment.states)
        for k, v in values.items():
            self.assertFalse(math.isnan(v))
        df = seg.discretize().df
        for (k1, v1), (k2, v2) in zip(df.iloc[0].to_dict().items(),
                                      seg.x0.items()):
            self.assertEqual(k1, k2)
            self.assertAlmostEqual(v1, v2)
        self.assertAlmostEqual(seg.x0["x_g"], 0)
        self.assertAlmostEqual(seg.x0["y_g"], 0)
        self.assertAlmostEqual(seg.x0["z_g"], 0)
        self.assertAlmostEqual(seg.xf["x_g"], 200)
        self.assertAlmostEqual(seg.xf["y_g"], 0)
        self.assertAlmostEqual(seg.xf["z_g"], 100)
        # Initial time and states can be shifted
        seg.shift(10, x_g=10, y_g=20, z_g=30)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)
        self.assertAlmostEqual(seg.x0["x_g"], 10)
        self.assertAlmostEqual(seg.x0["y_g"], 20)
        self.assertAlmostEqual(seg.x0["z_g"], 30)
        self.assertAlmostEqual(seg.xf["x_g"], 210)
        self.assertAlmostEqual(seg.xf["y_g"], 20)
        self.assertAlmostEqual(seg.xf["z_g"], 130)
        df = seg.discretize().df
        for (k1, v1), (k2, v2) in zip(df.iloc[0].to_dict().items(),
                                      seg.x0.items()):
            self.assertEqual(k1, k2)
            self.assertAlmostEqual(v1, v2)
        # Note that trajectory length, l, must be integrated and thus is not
        # included in xf.
        for (k1, v1), (k2, v2) in zip(df.iloc[-1].to_dict().items(),
                                      seg.xf.items()):
            if k1 != "l":
                self.assertEqual(k1, k2)
                self.assertAlmostEqual(v1, v2)

    def test_kinematic_consistency(self):
        seg = LiftedSegment(
            horiz_profile=ConstSpeedStraightFlight(tf=10, V_A_xy=20, chi=0),
            vert_profile=ConstVertSpeedProfile(z_g_t0=0, tf=10, w_A_g=10))
        seg.discretize().check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

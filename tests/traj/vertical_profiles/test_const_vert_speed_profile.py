# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.vertical_profiles.const_vert_speed_profile.

Run from top-level package directory:

   python3 -m unittest tests.traj.vertical_profiles.test_const_vert_speed_profile

"""

import math
import unittest

from eduplane.traj.vertical_profiles.const_vert_speed_profile import *


class ConstVertSpeedProfileTests(unittest.TestCase):

    def test_segment_interface(self):
        # Implements a fully parametric vertical profile
        seg = ConstVertSpeedProfile(z_g_t0=20, w_A_g=10, tf=10)
        values = seg.evaluate(0)
        self.assertTupleEqual(tuple(values.keys()), VertProfile.states)
        for v in values.values():
            self.assertFalse(math.isnan(v))
        df = seg.discretize().df
        self.assertDictEqual(df.iloc[0].to_dict(), seg.x0)
        self.assertDictEqual(df.iloc[-1].to_dict(), seg.xf)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 10)
        self.assertAlmostEqual(seg.x0["z_g"], 20)
        self.assertAlmostEqual(seg.xf["z_g"], 120)

        # Initial time and altitude can be shifted
        seg.shift(10, z_g=30)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)
        self.assertAlmostEqual(seg.x0["z_g"], 30)
        self.assertAlmostEqual(seg.xf["z_g"], 130)

    def test_initialization(self):
        # Can be initialized from vertical velocity component
        seg = ConstVertSpeedProfile.init_from(
            z_g_t0=20, w_K_g=10, w_W_g=5, tf=10)
        self.assertEqual(
            seg, ConstVertSpeedProfile(z_g_t0=20, w_A_g=5, w_W_g=5, tf=10))
        # Can be initialized from final instead of initial altitude
        seg = ConstVertSpeedProfile.init_from(
            z_g_tf=120, w_K_g=10, w_W_g=5, tf=10)
        self.assertEqual(
            seg, ConstVertSpeedProfile(z_g_t0=20, w_A_g=5, w_W_g=5, tf=10))

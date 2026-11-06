# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.vertical_profiles.linear_vert_speed_profile.

Run from top-level package directory:

   python3 -m unittest tests.traj.vertical_profiles.test_linear_vert_speed_profile

"""

import math
import unittest

from eduplane.traj.vertical_profiles.linear_vert_speed_profile import *


class LinearVertSpeedProfileTests(unittest.TestCase):

    def test_segment_interface(self):
        # Implements a fully parametric vertical profile
        seg = LinearVertSpeedProfile(
            z_g_t0=20, w_A_g_t0=10, w_A_g_dot=1, w_W_g=5, tf=10)
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
        self.assertAlmostEqual(seg.x0["w_A_g"], 10)
        self.assertAlmostEqual(seg.x0["w_K_g"], 15)
        self.assertAlmostEqual(seg.xf["z_g"], 20*10 + 20)
        self.assertAlmostEqual(seg.xf["w_A_g"], 20)
        self.assertAlmostEqual(seg.xf["w_K_g"], 25)

        # Initial time and altitude can be shifted
        seg.shift(10, z_g=30)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)
        self.assertAlmostEqual(seg.x0["z_g"], 30)
        self.assertAlmostEqual(seg.xf["z_g"], 20*10 + 30)

    def test_initialization(self):
        # Can be initialized from vertical velocity and acceleration component
        seg1 = LinearVertSpeedProfile(
            z_g_t0=20, w_A_g_t0=10, w_A_g_dot=1, w_W_g=5, tf=10)
        seg2 = LinearVertSpeedProfile.init_from(
            z_g_t0=20, w_K_g_t0=15, w_K_g_dot=1, w_W_g=0, tf=10)
        self.assertAlmostEqual(seg1.z_g_tf, seg2.z_g_tf)
        self.assertAlmostEqual(seg1.w_K_g_tf, seg2.w_K_g_tf)

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.airspeed_profiles.linear_airspeed_profile.

Run from top-level package directory:

   python3 -m unittest tests.traj.airspeed_profiles.test_linear_airspeed_profile

"""

import math
import unittest

from eduplane.traj.airspeed_profiles.linear_airspeed_profile import *


class LinearAirspeedProfilesTest(unittest.TestCase):

    def test_segment_interface(self):
        # Implements a fully parametric airspeed profile
        seg = LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10)
        values = seg.evaluate(0)
        self.assertTupleEqual(tuple(values.keys()), AirspeedProfile.states)
        for v in values.values():
            self.assertFalse(math.isnan(v))
        df = seg.discretize().df
        self.assertDictEqual(df.iloc[0].to_dict(), seg.x0)
        self.assertDictEqual(df.iloc[-1].to_dict(), seg.xf)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 10)
        self.assertAlmostEqual(seg.x0["V_A"], 20)
        self.assertAlmostEqual(seg.xf["V_A"], 40)

        # Initial time and airspeed can be shifted
        seg.shift(10, V_A=30)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)

    def test_stretch(self):
        # Stetching will decrease whichever boundary airspeed rate is higher
        # while keeping initial/final airspeed constant
        seg = LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10)
        seg_stretched = seg.stretch(20)
        self.assertAlmostEqual(seg_stretched.V_A_t0, seg.V_A_t0)
        self.assertAlmostEqual(seg_stretched.V_A_tf, seg.V_A_tf)
        self.assertLess(seg_stretched.V_A_dot, seg.V_A_dot)

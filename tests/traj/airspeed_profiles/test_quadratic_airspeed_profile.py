# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.airspeed_profiles.quadratic_airspeed_profile.

Run from top-level package directory:

   python3 -m unittest tests.traj.airspeed_profiles.test_quadratic_airspeed_profile

"""

import math
import unittest

from eduplane.traj.airspeed_profiles.quadratic_airspeed_profile import *


class QuadraticAirspeedProfileTest(unittest.TestCase):

    def test_segment_interface(self):
        # Implements a fully parametric airspeed profile
        seg = QuadraticAirspeedProfile(
            V_A_t0=20, V_A_dot_t0=-5, V_A_dot2=1, tf=10)
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
        self.assertAlmostEqual(seg.xf["V_A"], 20 - 5*10 + 0.5*10**2)

        # Initial time can be shifted
        seg.shift(10)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)

    def test_init(self):
        # Initialize from initial/final airspeed and derivative:
        # Governing equations:
        # V_A_tf = V_A_t0 + V_A_dot_t0*tf + 0.5*V_A_dot2*tf**2
        # V_A_dot2 = (V_A_dot_tf - V_A_dot_t0)/tf
        # V_A_tf = V_A_t0 + 0.5*tf*(V_A_dot_t0 + V_A_dot_tf)
        # tf = (V_A_tf - V_A_t0)/(0.5*(V_A_dot_t0 + V_A_dot_tf))
        tf = 10
        V_A_t0 = 20
        V_A_dot_t0 = 5
        V_A_dot2 = -0.05
        V_A_dot_tf = V_A_dot_t0 + V_A_dot2*tf
        V_A_tf = V_A_t0 + 0.5*tf*(V_A_dot_t0+V_A_dot_tf)
        seg = QuadraticAirspeedProfile.init_from(
            V_A_t0=V_A_t0,
            V_A_tf=V_A_tf,
            V_A_dot_t0=V_A_dot_t0,
            V_A_dot_tf=V_A_dot_tf)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, tf)
        self.assertAlmostEqual(seg.V_A_t0, V_A_t0)
        self.assertAlmostEqual(seg.V_A_tf, V_A_tf)
        self.assertAlmostEqual(seg.V_A_dot_t0, V_A_dot_t0)
        self.assertAlmostEqual(seg.V_A_dot_tf, V_A_dot_tf)
        self.assertAlmostEqual(seg.V_A_dot2, V_A_dot2)

    def test_stretch(self):
        # If positive airspeed rate increases, the initial airspeed rate will
        # be kept constant
        seg = QuadraticAirspeedProfile(
            V_A_t0=10, V_A_dot_t0=2, V_A_dot2=1, tf=10)
        seg_stretched = seg.stretch(20)
        self.assertAlmostEqual(seg_stretched.V_A_t0, seg.V_A_t0)
        self.assertAlmostEqual(seg_stretched.V_A_tf, seg.V_A_tf)
        self.assertAlmostEqual(seg_stretched.V_A_dot_t0, seg.V_A_dot_t0)
        self.assertLess(seg_stretched.V_A_dot_tf, seg.V_A_dot_tf)
        self.assertLess(abs(seg_stretched.V_A_dot2), abs(seg.V_A_dot_t0))
        # If negative airspeed rate decreases, the initial airspeed rate will
        # be kept constant
        seg = QuadraticAirspeedProfile(
            V_A_t0=10, V_A_dot_t0=-2, V_A_dot2=-1, tf=10)
        seg_stretched = seg.stretch(20)
        self.assertAlmostEqual(seg_stretched.V_A_t0, seg.V_A_t0)
        self.assertAlmostEqual(seg_stretched.V_A_tf, seg.V_A_tf)
        self.assertAlmostEqual(seg_stretched.V_A_dot_t0, seg.V_A_dot_t0)
        self.assertGreater(seg_stretched.V_A_dot_tf, seg.V_A_dot_tf)
        self.assertLess(abs(seg_stretched.V_A_dot2), abs(seg.V_A_dot_t0))
        # If positive airspeed rate decreases, the final airspeed rate will
        # be kept constant
        seg = QuadraticAirspeedProfile(
            V_A_t0=10, V_A_dot_t0=10, V_A_dot2=-1, tf=10)
        seg_stretched = seg.stretch(20)
        self.assertAlmostEqual(seg_stretched.V_A_t0, seg.V_A_t0)
        self.assertAlmostEqual(seg_stretched.V_A_tf, seg.V_A_tf)
        self.assertAlmostEqual(seg_stretched.V_A_dot_tf, seg.V_A_dot_tf)
        self.assertLess(seg_stretched.V_A_dot_t0, seg.V_A_dot_t0)
        self.assertLess(abs(seg_stretched.V_A_dot2), abs(seg.V_A_dot_t0))

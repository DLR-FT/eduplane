# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.horizontal_profiles.const_speed_straight_flight.

Run from top-level package directory:

   python3 -m unittest tests.traj.horizontal_profiles.test_const_speed_straight_flight

"""

import math
import unittest

from eduplane.traj.horizontal_profiles.const_speed_straight_flight import *


class ConstSpeedStraightFlightTest(unittest.TestCase):

    def test_interface(self):
        # Implements a fully parametric horizontal trajectory segment
        seg = ConstSpeedStraightFlight.init_from(
            x_g_t0=0, y_g_t0=0, tf=10, V_A_xy=20, chi_A=0, V_W_xy=0, chi_W=0)
        values = seg.evaluate(0)
        self.assertTupleEqual(tuple(values.keys()), HorizProfile.states)
        for v in values.values():
            self.assertFalse(math.isnan(v))
        df = seg.discretize().df
        self.assertDictEqual(df.iloc[0].to_dict(), seg.x0)
        self.assertDictEqual(df.iloc[-1].to_dict(), seg.xf)
        # Initial time and states can be shifted
        seg.shift(10, x_g=10, y_g=20)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)
        self.assertAlmostEqual(seg.x0["x_g"], 10)
        self.assertAlmostEqual(seg.x0["y_g"], 20)
        self.assertAlmostEqual(seg.xf["x_g"], 210)
        self.assertAlmostEqual(seg.xf["y_g"], 20)

    def test_no_wind(self):
        # Linear segment going from (-100, -100) to (100, 100) in 10 seconds
        x_g_t0 = -100
        y_g_t0 = -100
        l_xy_tf = math.sqrt(2*200**2)
        tf = 10
        V_A_xy = l_xy_tf/tf
        chi = math.radians(45)
        chi_A = chi
        # Initialize from segment time and air path azimuth
        lin_seg_from_tf_chi_A = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy=V_A_xy,
            chi_A=chi_A,
            V_W_xy=0,
            chi_W=0)
        # Initialize from segment time and path azimuth (default)
        lin_seg_from_tf_chi = ConstSpeedStraightFlight(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy=V_A_xy,
            chi=chi,
            V_W_xy=0,
            chi_W=0)
        # Initialize from segment length and air path azimuth
        lin_seg_from_l_xy_tf_chi_A = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            l_xy_tf=l_xy_tf,
            V_A_xy=V_A_xy,
            chi_A=chi_A,
            V_W_xy=0,
            chi_W=0)
        # Initialize from segment length and path azimuth
        lin_seg_from_l_xy_tf_chi = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            l_xy_tf=l_xy_tf,
            V_A_xy=V_A_xy,
            chi=chi,
            V_W_xy=0,
            chi_W=0)
        for lin_seg in [lin_seg_from_tf_chi_A, lin_seg_from_tf_chi,
                        lin_seg_from_l_xy_tf_chi_A, lin_seg_from_l_xy_tf_chi]:
            self.assertAlmostEqual(lin_seg.tf, tf)
            self.assertAlmostEqual(lin_seg.V_A_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.V_K_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.l_xy_tf, l_xy_tf)
            self.assertAlmostEqual(lin_seg.chi_A, chi_A)
            self.assertAlmostEqual(lin_seg.chi, chi)
            self.assertAlmostEqual(lin_seg.V_K_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.V_A_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.x_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.y_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.x_g(t=tf/2), 0)
            self.assertAlmostEqual(lin_seg.y_g(t=tf/2), 0)
            self.assertAlmostEqual(lin_seg.x_g(t=tf), 100)
            self.assertAlmostEqual(lin_seg.y_g(t=tf), 100)

    def test_with_wind(self):
        # Linear segment going from (-100, -100) to (100, 100) with headwind.
        x_g_t0 = -100
        y_g_t0 = -100
        l_xy_tf = math.sqrt(2*200**2)
        V_A_xy = 10
        V_W_xy = 5
        V_K_xy = V_A_xy - V_W_xy
        tf = l_xy_tf/V_K_xy
        chi = math.radians(45)
        chi_A = chi
        chi_W = chi + math.pi
        # Initialize from segment time and air path azimuth
        lin_seg_from_tf_chi_A = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy=V_A_xy,
            chi_A=chi_A,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        # Initialize from segment time and path azimuth (default)
        lin_seg_from_tf_chi = ConstSpeedStraightFlight(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy=V_A_xy,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        # Initialize from segment length and air path azimuth
        lin_seg_from_l_xy_tf_chi_A = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            l_xy_tf=l_xy_tf,
            V_A_xy=V_A_xy,
            chi_A=chi_A,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        # Initialize from segment length and path azimuth
        lin_seg_from_l_xy_tf_chi = ConstSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            l_xy_tf=l_xy_tf,
            V_A_xy=V_A_xy,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        for lin_seg in [lin_seg_from_tf_chi_A, lin_seg_from_tf_chi,
                        lin_seg_from_l_xy_tf_chi_A, lin_seg_from_l_xy_tf_chi]:
            self.assertAlmostEqual(lin_seg.tf, tf)
            self.assertAlmostEqual(lin_seg.V_A_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.V_K_xy, V_A_xy*0.5)
            self.assertAlmostEqual(lin_seg.V_W_xy, V_A_xy*0.5)
            self.assertAlmostEqual(lin_seg.l_xy_tf, l_xy_tf)
            self.assertAlmostEqual(lin_seg.chi, chi)
            self.assertAlmostEqual(lin_seg.chi_A, chi_A)
            self.assertAlmostEqual(lin_seg.chi_W, chi_W)
            self.assertAlmostEqual(lin_seg.V_K_xy, V_A_xy*0.5)
            self.assertAlmostEqual(lin_seg.V_A_xy, V_A_xy)
            self.assertAlmostEqual(lin_seg.x_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.y_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.x_g(t=tf/2), 0)
            self.assertAlmostEqual(lin_seg.y_g(t=tf/2), 0)
            self.assertAlmostEqual(lin_seg.x_g(t=tf), 100)
            self.assertAlmostEqual(lin_seg.y_g(t=tf), 100)

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.discrete_segment.

Run from top-level package directory:

   python3 -m unittest tests.traj.test_discrete_segment

"""

import unittest

import pandas

from eduplane.traj.discrete_segment import *
from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, LinearSpeedStraightFlight)
from eduplane.traj.horizontal_profiles.steady_turn import SteadyTurn
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile


class DiscreteSegmentTests(unittest.TestCase):

    def test_segment_interface(self):
        # Discretized segments can be default-initialized
        self.assertEqual(DiscreteSegment().t0, 0)
        self.assertEqual(DiscreteSegment().tf, 0)
        self.assertTrue(DiscreteSegment().df.empty, 0)
        self.assertEqual(DiscreteSegment().duration, 0)
        # Start and final state and times are retrieved from the dataframe
        segment = DiscreteSegment(
            df=pandas.DataFrame(
                {
                    "x_g": [0, 1],
                    "y_g": [2, 3]
                }, index=[0.1, 0.2]))
        self.assertEqual(segment.t0, 0.1)
        self.assertEqual(segment.tf, 0.2)
        self.assertAlmostEqual(segment.duration, 0.1)
        self.assertEqual(segment.x0, {"x_g": 0, "y_g": 2})
        self.assertEqual(segment.xf, {"x_g": 1, "y_g": 3})
        # A segment's time interval and other initial values can be shifted
        segment.shift(0.5, x_g=10)
        self.assertEqual(segment.t0, 0.5)
        self.assertAlmostEqual(segment.tf, 0.6)
        self.assertAlmostEqual(segment.duration, 0.1)
        self.assertEqual(segment.x0, {"x_g": 10, "y_g": 2})
        self.assertEqual(segment.xf, {"x_g": 11, "y_g": 3})
        # We can interpolate linearly in a discrete segment
        x_int = segment.evaluate(0.55)
        self.assertAlmostEqual(x_int["x_g"], 10.5)
        self.assertAlmostEqual(x_int["y_g"], 2.5)
        x_int = segment.evaluate(0.51)
        self.assertAlmostEqual(x_int["x_g"], 10.1)
        self.assertAlmostEqual(x_int["y_g"], 2.1)

    def test_check_kinematic_consistency(self):
        # Straight climb segment with constant airspeed, no wind
        seg = LiftedSegment(
            ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=-2, tf=15),
        ).discretize()
        self.assertGreater(seg.df.gamma.max(), 0)
        self.assertGreater(seg.df.gamma_A.max(), 0)
        seg.check_kinematic_consistency(raise_err=True)
        # If we mess with the data, the check will fail
        seg.df.loc[5, "x_g"] -= seg.df["x_g"].loc[5] - 1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                warn=False,
                raise_err=True)

        # Straight descent segment with constant airspeed, cross wind
        seg = LiftedSegment(
            ConstSpeedStraightFlight(
                V_A_xy=20, chi=0.5*numpy.pi, V_W_xy=5, chi_W=numpy.pi, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=2, tf=15),
        ).discretize()
        seg.check_kinematic_consistency(raise_err=True)
        # If we mess with the data, the check will fail
        #        seg.df.loc[5, "V_W_xy"] = seg.df["V_W_xy"].loc[5] + 1
        seg.df["V_W_xy"] = seg.df["V_W_xy"] + 1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                warn=False,
                raise_err=True)

        # Straight descent segment with linear airspeed, no wind
        seg = LiftedSegment(
            LinearSpeedStraightFlight(
                V_A_xy_t0=20, V_A_xy_dot=0.5, chi=0.5*numpy.pi, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=2, tf=15),
        ).discretize()
        # # DEBUG:
        # from matplotlib import pyplot
        # seg.df[["v_A_g", "v_A_g_dot"]].plot()
        # seg.df[["u_A_g", "u_A_g_dot"]].plot()
        # seg.df[["V_A_xy", "chi_A"]].plot()
        # pyplot.show()
        seg.check_kinematic_consistency(raise_err=True)
        # If we mess with the data, the check will fail.
        # NOTE: Corrupt the state V_A_xy rather than its derivative
        # V_A_xy_dot. A single-point jump in the derivative stays within the
        # (conservative) numerical integration error margin of _check_int,
        # so it would not be detected.
        seg.df.loc[5, "V_A_xy"] = seg.df["V_A_xy"].loc[5] - 1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                horizontal_airpath_continuity=1,
                airspeed_continuity=1,
                warn=False,
                raise_err=True)

        # Straight descent segment with linear airspeed, cross wind
        seg = LiftedSegment(
            LinearSpeedStraightFlight(
                V_A_xy_t0=20,
                V_A_xy_dot=0.5,
                chi=0.5*numpy.pi,
                V_W_xy=5,
                chi_W=numpy.pi,
                tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=2, tf=15),
        ).discretize()
        seg.check_kinematic_consistency(raise_err=True)
        # If we mess with the data, the check will fail.
        # NOTE: See above - corrupt the state, not its derivative.
        seg.df.loc[5, "V_A_xy"] = seg.df["V_A_xy"].loc[5] - 1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                horizontal_airpath_continuity=1,
                airspeed_continuity=1,
                warn=False,
                raise_err=True)

        # Steady level turn with constant airspeed, no wind
        seg = LiftedSegment(
            SteadyTurn(V_A_xy=20, chi_t0=0, chi_A_dot=numpy.pi/60, tf=240),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=240),
        ).discretize()
        seg.check_kinematic_consistency(raise_err=True)
        # If we mess with the data, the check will fail
        seg.df.loc[5, "chi_A"] = seg.df["chi_A"].loc[5] - 0.1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                warn=False,
                raise_err=True)

        # Steady level turn with constant airspeed, with wind
        seg = LiftedSegment(
            SteadyTurn(
                V_A_xy=20,
                chi_t0=0.5*numpy.pi,
                chi_A_dot=numpy.pi/60,
                V_W_xy=5,
                chi_W=numpy.pi,
                tf=240),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=240),
        ).discretize()
        seg.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)
        # If we mess with the data, the check will fail
        seg.df.loc[5, "V_W_xy"] = seg.df["V_W_xy"].loc[5] - 1
        with self.assertRaises(ValueError):
            seg.check_kinematic_consistency(
                horizontal_continuity=2,
                vertical_continuity=2,
                warn=False,
                raise_err=True)

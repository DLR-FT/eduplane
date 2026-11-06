# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.trajectory.

Run from top-level package directory:

   python3 -m unittest tests.traj.test_trajectory

"""

import unittest

import numpy
import pandas

from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, LinearSpeedStraightFlight, SteadyTurn)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.trajectory import *
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile


class TrajectoryTests(unittest.TestCase):

    def test_segment_interface(self):
        # Trajectories may be initialized with only an initial state and time
        traj = Trajectory(x0={"x_g": 10, "y_g": 20}, t0=10)
        # Segments can be appended and are aligned automatically
        traj.append(
            DiscreteSegment(
                df=pandas.DataFrame(
                    data={
                        "x_g": [0, 1],
                        "y_g": [2, 3],
                    }, index=[0.1, 0.2])))
        traj.append(
            DiscreteSegment(
                df=pandas.DataFrame(
                    {
                        "x_g": [1, 3],
                        "y_g": [3, 5]
                    }, index=[1234.1, 1234.2])))
        self.assertEqual(len(traj.segments), 3)
        self.assertAlmostEqual(traj.t0, traj.segments[0].t0)
        self.assertAlmostEqual(traj.tf, traj.segments[-1].tf)
        self.assertDictEqual(traj.x0, traj.segments[0].x0)
        self.assertDictEqual(traj.xf, traj.segments[-1].xf)
        self.assertAlmostEqual(traj.segments[0].tf, traj.segments[1].t0)
        self.assertAlmostEqual(traj.segments[1].tf, traj.segments[2].t0)
        self.assertAlmostEqual(traj.segments[0].tf, 10)
        self.assertAlmostEqual(traj.segments[1].tf, 10.1)
        self.assertAlmostEqual(traj.segments[2].tf, 10.2)
        self.assertDictEqual(traj.segments[0].x0, {"x_g": 10, "y_g": 20})
        self.assertDictEqual(traj.segments[0].xf, {"x_g": 10, "y_g": 20})
        self.assertDictEqual(traj.segments[1].x0, {"x_g": 10, "y_g": 20})
        self.assertDictEqual(traj.segments[1].xf, {"x_g": 11, "y_g": 21})
        self.assertDictEqual(traj.segments[2].x0, {"x_g": 11, "y_g": 21})
        self.assertDictEqual(traj.segments[2].xf, {"x_g": 13, "y_g": 23})
        # Values at segment knots are from the last segment at the knot
        traj = Trajectory(x0={"x_g": 10, "y_g": 0}, t0=10)
        traj.append(
            DiscreteSegment(
                df=pandas.DataFrame(
                    data={
                        "x_g": [0, 1],
                        "y_g": [2, 3],
                    }, index=[0.1, 0.2])))
        self.assertDictEqual(traj.x0, {"x_g": 10, "y_g": 0})
        self.assertDictEqual(traj.segments[0].x0, {"x_g": 10, "y_g": 0})
        self.assertDictEqual(traj.segments[0].xf, {"x_g": 10, "y_g": 0})
        self.assertDictEqual(traj.segments[1].x0, {"x_g": 10, "y_g": 0})
        self.assertDictEqual(traj.segments[1].xf, {"x_g": 11, "y_g": 1})
        self.assertDictEqual(traj.xf, {"x_g": 11, "y_g": 1})
        discr_traj = traj.discretize()
        self.assertDictEqual(discr_traj.df.iloc[0].to_dict(), traj.x0)
        self.assertDictEqual(discr_traj.df.iloc[-1].to_dict(), traj.xf)
        # Trajectories can be shifted, which affects all segments
        traj.shift(100, x_g=100)
        self.assertAlmostEqual(traj.t0, 100)
        self.assertAlmostEqual(traj.tf, 100.1)
        self.assertDictEqual(traj.x0, {"x_g": 100, "y_g": 0})
        self.assertDictEqual(traj.xf, {"x_g": 101, "y_g": 1})

    def test_kinematic_consistency(self):
        # Concatenate C1-continuous segments (i.e. speeds/rates continuous,
        # accelerations may be discontinuous)
        # Segment #1: straight level flight at constant airspeed
        seg1 = LiftedSegment(
            horiz_profile=ConstSpeedStraightFlight(tf=10, V_A_xy=20, chi=0),
            vert_profile=ConstVertSpeedProfile(z_g_t0=0, tf=10, w_A_g=0))
        # Segment #2: level right turn 180°
        seg2 = LiftedSegment(
            SteadyTurn(V_A_xy=20, chi_t0=0, chi_A_dot=numpy.pi/60, tf=60),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=60),
        )
        # Segment #3: accelerating straight level flight
        seg3 = LiftedSegment(
            horiz_profile=LinearSpeedStraightFlight(
                tf=10, V_A_xy_t0=20, V_A_xy_dot=2, chi=numpy.pi),
            vert_profile=ConstVertSpeedProfile(z_g_t0=0, tf=10, w_A_g=0))
        traj = Trajectory([seg1, seg2, seg3])
        traj_discretized = traj.discretize()

        # from matplotlib import pyplot
        # # traj_discretized.df[["V_K_xy", "V_K_xy_dot", "u_K_g",
        # #                      "u_K_g_dot"]].plot()
        # traj_discretized.df[["l", "l_xy"]].plot()
        # traj_discretized.df[["x_g", "y_g"]].plot(x="y_g", y="x_g")
        # pyplot.show()

        traj_discretized.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

    def test_time_index_clipping(self):
        # When concatenating very short segments (i.e. with parameter intervals
        # which are numerically close to zero), we still want to have the time
        # index evenly distributed with the desired resolution.
        traj = Trajectory(
            [
                ConstSpeedStraightFlight(tf=0.1, V_A_xy=20, chi=0),
                ConstSpeedStraightFlight(tf=0.0001, V_A_xy=30, chi=0),
                ConstSpeedStraightFlight(tf=0.1, V_A_xy=40, chi=0)
            ])
        traj_discr = traj.discretize(t_res=0.05)
        idx_diff = traj_discr.df.index.to_series().diff().dropna()
        expected_diff = 0.2001/5
        self.assertTrue(numpy.allclose(idx_diff, expected_diff))
        self.assertLessEqual(idx_diff.max(), 0.05)
        self.assertGreater(idx_diff.min(), 0.0001)

    def test_zero_length_segments(self):
        # Zero-length segments are not evaluated at all to avoid duplication of
        # points towards the neighouring segments
        traj = Trajectory(
            [
                ConstSpeedStraightFlight(tf=0.1, V_A_xy=20, chi=0),
                ConstSpeedStraightFlight(t0=0.1, tf=0.1, V_A_xy=30, chi=0),
                ConstSpeedStraightFlight(tf=0.1, V_A_xy=40, chi=0)
            ])
        # Discretized with dt=0.05 will give t values: 0, 0.05, 0.1, 0.15, 0.2
        traj_discr = traj.discretize(t_res=0.05)
        idx_diff = traj_discr.df.index.to_series().diff().dropna()
        self.assertTrue(numpy.allclose(idx_diff, 0.05))
        self.assertEqual(len(traj_discr.df), 5)
        # Discretized with dt=0.04 will give t values: 0, 0.04, 0.08, 0.12, 0.16, 0.2
        traj_discr = traj.discretize(t_res=0.04)
        idx_diff = traj_discr.df.index.to_series().diff().dropna()
        self.assertTrue(numpy.allclose(idx_diff, 0.04))
        self.assertEqual(len(traj_discr.df), 6)


# Run from folder containing the top-level package:
#   python3 -m eduplane.traj.trajectory
if __name__ == "__main__":
    unittest.main()

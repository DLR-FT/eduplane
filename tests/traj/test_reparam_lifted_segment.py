# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.reparam_lifted_segment.

Run from top-level package directory:

   python3 -m unittest tests.traj.test_reparam_lifted_segment

"""

import math
import unittest

from eduplane.traj.airspeed_profiles import LinearAirspeedProfile
from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, SteadyTurn)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.reparam_lifted_segment import *
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile


class ReparamLiftedSegmentTests(unittest.TestCase):

    def test_reparam_lifted_segment(self):

        # Tests with straight horizontal path
        # ==============================================

        # Generate straight-level segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        horiz_path = ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15)
        lifted_seg = LiftedSegment(
            horiz_path,
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=15),
        )
        lifted_seg_discr = lifted_seg.discretize()
        lifted_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(lifted_seg.t0, 0)
        self.assertAlmostEqual(lifted_seg.tf, 15)
        self.assertAlmostEqual(lifted_seg.x0["x_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["y_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["z_g"], 0)
        self.assertAlmostEqual(lifted_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(lifted_seg.xf["l_xy"], 300)
        self.assertAlmostEqual(lifted_seg.x0["l"], 0)
        self.assertAlmostEqual(lifted_seg_discr.xf["l"], 300)
        self.assertAlmostEqual(lifted_seg.x0["V_A"], 20)
        self.assertAlmostEqual(lifted_seg.xf["V_A"], 20)

        # Re-parametrize with linear airspeed, no wind
        # 300 m at 20-40 m/s (linear increase -> 30 m/s mean) in 10 s
        # with identical vertical profile
        reparam_seg = ReparamLiftedSegment(
            lifted_seg.horiz_profile, lifted_seg.vert_profile,
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10)).discretize()

        # from matplotlib import pyplot
        # reparam_seg.discretize().df[["u_A_g", "u_A_g_dot"]].plot()
        # pyplot.show()
        reparam_seg_discr = reparam_seg.discretize()
        reparam_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(reparam_seg.t0, 0)
        self.assertAlmostEqual(reparam_seg.tf, 10)
        self.assertAlmostEqual(reparam_seg.x0["x_g"], lifted_seg.x0["x_g"])
        self.assertAlmostEqual(reparam_seg.xf["y_g"], lifted_seg.x0["y_g"])
        self.assertAlmostEqual(reparam_seg_discr.x0["l"], 0)
        self.assertAlmostEqual(
            reparam_seg_discr.xf["l"], lifted_seg_discr.xf["l"])
        self.assertAlmostEqual(reparam_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(reparam_seg.xf["l_xy"], lifted_seg.xf["l_xy"])
        self.assertAlmostEqual(reparam_seg.x0["V_A"], 20)
        self.assertAlmostEqual(reparam_seg.xf["V_A"], 40)

        # Re-parametrize with linear airspeed and apply a climb vertical
        # profile, still no wind
        # 300 m at 20-40 m/s (linear increase -> 30 m/s mean) in 10 s
        # with steady 5 m/s climb
        # -> V_A_xy = (V_A(t)^2 - (5 m/s)^2)^1/2
        # -> l is the same
        # -> l_xy is less -> l_xy ~= 300*cos(gamma_mean)
        reparam_seg = ReparamLiftedSegment(
            lifted_seg.horiz_profile,
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=10),
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10))
        reparam_seg_discr = reparam_seg.discretize()
        reparam_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        # DEBUG:
        # from matplotlib import pyplot
        # # reparam_seg.discretize().df[["u_A_g", "u_A_g_dot"]].plot()
        # # reparam_seg.df[["l", "l_xy"]].plot()
        # reparam_seg.df[["V_A", "V_A_xy", "w_A_g"]].plot()
        # pyplot.show()

        self.assertAlmostEqual(reparam_seg_discr.t0, 0)
        self.assertAlmostEqual(reparam_seg_discr.tf, 10)
        self.assertAlmostEqual(reparam_seg_discr.x0["l"], 0)
        self.assertAlmostEqual(reparam_seg_discr.xf["l"], 300, places=3)
        self.assertAlmostEqual(reparam_seg_discr.x0["l_xy"], 0)
        self.assertAlmostEqual(
            reparam_seg_discr.xf["l_xy"],
            300*math.cos(reparam_seg_discr.df["gamma"].mean()),
            places=0)
        self.assertAlmostEqual(reparam_seg_discr.x0["V_A"], 20)
        self.assertAlmostEqual(reparam_seg_discr.xf["V_A"], 40)

        # Tests with steady 1-minute turn horizontal path
        # ==============================================
        lifted_seg = LiftedSegment(
            SteadyTurn(V_A_xy=20, chi_t0=0, chi_A_dot=2*math.pi/60, tf=60),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=60),
        )
        fullturn_l_xy = 2*math.pi*20/(2*math.pi/60)
        seg_discr = lifted_seg.discretize()
        seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(lifted_seg.t0, 0)
        self.assertAlmostEqual(lifted_seg.tf, 60)
        self.assertAlmostEqual(lifted_seg.x0["x_g"], 0)
        self.assertAlmostEqual(lifted_seg.x0["y_g"], 0)
        self.assertAlmostEqual(lifted_seg.x0["z_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["x_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["y_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["z_g"], 0)
        self.assertAlmostEqual(lifted_seg.x0["V_A"], 20)
        self.assertAlmostEqual(lifted_seg.xf["V_A"], 20)
        self.assertAlmostEqual(lifted_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(lifted_seg.xf["l_xy"], fullturn_l_xy)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(lifted_seg.x0["l"], 0)
        self.assertAlmostEqual(seg_discr.xf["l"], fullturn_l_xy, places=1)

        # Re-parametrize with linear airspeed, no wind
        # Same circumference at 20-40 m/s (linear increase -> 30 m/s mean) in 40 s
        tf_at_mean_airspeed_30 = fullturn_l_xy/30
        seg_reparam = ReparamLiftedSegment(
            lifted_seg.horiz_profile, lifted_seg.vert_profile,
            LinearAirspeedProfile.init_from(
                V_A_t0=20, V_A_tf=40, tf=tf_at_mean_airspeed_30)).discretize(
                    t_res=0.1, tf=fullturn_l_xy/30)
        seg_reparam_discr = seg_reparam.discretize()
        seg_reparam_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(seg_reparam.t0, 0)
        self.assertAlmostEqual(seg_reparam.tf, tf_at_mean_airspeed_30)
        self.assertAlmostEqual(seg_reparam.x0["x_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["y_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["z_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["x_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["y_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["z_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], fullturn_l_xy, places=1)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(seg_reparam.x0["l"], 0)
        self.assertAlmostEqual(
            seg_reparam_discr.xf["l"], fullturn_l_xy, places=1)

        # Re-parametrize with linear airspeed and apply a climb vertical
        # profile, still no wind
        # Same circumference at 20-40 m/s (linear increase -> 30 m/s mean)
        # with steady 5 m/s climb
        V_A_xy_mean = math.sqrt(30**2 - 5**2)
        tf_at_V_A_xy_mean = fullturn_l_xy/V_A_xy_mean
        # -> l_xy is less -> l_xy ~= 300*cos(gamma_mean)
        reparam_seg = ReparamLiftedSegment(
            lifted_seg.horiz_profile,
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=tf_at_V_A_xy_mean),
            LinearAirspeedProfile.init_from(
                V_A_t0=20, V_A_tf=40, tf=tf_at_V_A_xy_mean))
        reparam_seg_discr = reparam_seg.discretize()
        reparam_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        # from matplotlib import pyplot
        # # ax = lifted_seg.discretize().df[["x_g", "y_g"]].plot(x="x_g", y="y_g")
        # ax = None
        # reparam_seg_discr.df[["x_g", "y_g"]].plot(x="x_g", y="y_g", ax=ax)
        # # ax = lifted_seg.discretize().df[["V_A", "V_K"]].plot()
        # ax = None
        # reparam_seg_discr.df[["V_A", "V_K", "x_g"]].plot(ax=ax)
        # # ax = lifted_seg.discretize().df[["chi", "chi_dot"]].plot()
        # # seg_reparam.discretize().df[["chi", "chi_dot"]].plot(ax=ax)
        # pyplot.show()

        self.assertAlmostEqual(seg_reparam.t0, 0)
        self.assertAlmostEqual(seg_reparam.tf, tf_at_mean_airspeed_30)
        self.assertAlmostEqual(seg_reparam.x0["x_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["y_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["z_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["x_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["y_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["z_g"], 0)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], fullturn_l_xy, places=1)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(seg_reparam.x0["l"], 0)
        self.assertAlmostEqual(
            seg_reparam_discr.xf["l"], fullturn_l_xy, places=1)

    def test_integrate_backwards(self):
        # Tests with straight horizontal path
        # ==============================================

        # Generate straight-level segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        horiz_path = ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15)
        lifted_seg = LiftedSegment(
            horiz_path,
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=15),
        )
        lifted_seg_discr = lifted_seg.discretize()
        lifted_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(lifted_seg.t0, 0)
        self.assertAlmostEqual(lifted_seg.tf, 15)
        self.assertAlmostEqual(lifted_seg.x0["x_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["x_g"], 300)
        self.assertAlmostEqual(lifted_seg.x0["z_g"], 0)
        self.assertAlmostEqual(lifted_seg.xf["z_g"], 0)
        self.assertAlmostEqual(lifted_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(lifted_seg.xf["l_xy"], 300)
        self.assertAlmostEqual(lifted_seg.x0["l"], 0)
        self.assertAlmostEqual(lifted_seg_discr.xf["l"], 300)
        self.assertAlmostEqual(lifted_seg.x0["V_A"], 20)
        self.assertAlmostEqual(lifted_seg.xf["V_A"], 20)

        # Re-parametrization with constant airspeed profile, i.e. without any change
        reparam_seg = ReparamLiftedSegment(
            lifted_seg.horiz_profile,
            lifted_seg.vert_profile,
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=0, tf=15),
            align_left=False)

        # from matplotlib import pyplot
        # reparam_seg.discretize().df[["V_K_xy", "x_g", "l_xy", "l"]].plot()
        # pyplot.show()

        self.assertAlmostEqual(reparam_seg.t0, 0)
        self.assertAlmostEqual(reparam_seg.tf, 15)
        self.assertAlmostEqual(reparam_seg.x0["x_g"], 0)
        self.assertAlmostEqual(reparam_seg.xf["x_g"], 300)
        self.assertAlmostEqual(reparam_seg.x0["z_g"], 0)
        self.assertAlmostEqual(reparam_seg.xf["z_g"], 0)
        self.assertAlmostEqual(reparam_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(reparam_seg.xf["l_xy"], 300)
        self.assertAlmostEqual(reparam_seg.x0["V_A"], 20)
        self.assertAlmostEqual(reparam_seg.xf["V_A"], 20)

        # Re-parametrize with linear airspeed, no wind
        # 300 m at 20-40 m/s (linear increase -> 30 m/s mean) in 10 s
        # with identical vertical profile
        reparam_seg = ReparamLiftedSegment(
            lifted_seg.horiz_profile,
            lifted_seg.vert_profile,
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10),
            align_left=False)

        # # DEBUG PLOT
        # from matplotlib import pyplot

        # reparam_seg.discretize().df[["l_xy", "l"]].plot()
        # # # reparam_seg.discretize().df[["V_K_xy", "x_g", "l_xy", "l"]].plot()
        # # # reparam_seg.discretize().df[["l_xy", "x_g", "u_K_g",
        # # #                              "u_K_g_dot"]].plot()
        # # # reparam_seg.discretize().df[["V_A"]].plot()
        # pyplot.show()

        reparam_seg_discr = reparam_seg.discretize()
        reparam_seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(reparam_seg.t0, 0)
        self.assertAlmostEqual(reparam_seg.tf, 10)
        self.assertAlmostEqual(reparam_seg.x0["x_g"], lifted_seg.x0["x_g"])
        self.assertAlmostEqual(reparam_seg.xf["y_g"], lifted_seg.x0["y_g"])
        self.assertAlmostEqual(reparam_seg_discr.x0["l"], 0)
        self.assertAlmostEqual(
            reparam_seg_discr.xf["l"], lifted_seg_discr.xf["l"])
        self.assertAlmostEqual(reparam_seg.x0["l_xy"], 0)
        self.assertAlmostEqual(reparam_seg.xf["l_xy"], lifted_seg.xf["l_xy"])
        self.assertAlmostEqual(reparam_seg.x0["V_A"], 20)
        self.assertAlmostEqual(reparam_seg.xf["V_A"], 40)

    def test_alignment(self):
        # Tests with straight horizontal path
        # ==============================================

        # Generate straight-level segment with constant airspeed, no wind
        # 240 m at 12 m/s in 20 s
        horiz_path = ConstSpeedStraightFlight(V_A_xy=12, chi=0, tf=20)
        self.assertAlmostEqual(horiz_path.xf["l_xy"], 240)

        # Re-parametrize with constant airspeed and 5m/s climb vertical profile,
        # V_A_xy stays at 12 m/s
        # V_A is 13 m/s
        # -> tf is 20
        target_tf = 20
        reparam_lifted_seg = ReparamLiftedSegment(
            horiz_path, ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=target_tf),
            LinearAirspeedProfile(V_A_t0=13, V_A_dot=0, tf=target_tf))
        self.assertTrue(reparam_lifted_seg.check_alignment())

        # Re-parametrize with shorter linear airspeed profile
        reparam_lifted_seg = ReparamLiftedSegment(
            horiz_path, ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=target_tf),
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=0, tf=target_tf - 2))
        self.assertFalse(reparam_lifted_seg.check_alignment())

        # Re-parametrize with longer linear airspeed profile
        reparam_lifted_seg = ReparamLiftedSegment(
            horiz_path, ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=target_tf),
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=0, tf=target_tf + 2))
        self.assertFalse(reparam_lifted_seg.check_alignment())

        # Re-parametrize with longer linear airspeed and vertical profiles
        reparam_lifted_seg = ReparamLiftedSegment(
            horiz_path,
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=target_tf + 2),
            LinearAirspeedProfile(V_A_t0=20, V_A_dot=0, tf=target_tf + 2))
        self.assertFalse(reparam_lifted_seg.check_alignment())

        # Calculate aligned profiles for accelerated straight climbing flight
        seg_reparam = ReparamLiftedSegment.align_airspeed_and_vertical_profiles(
            horiz_path,
            lambda tf: ConstVertSpeedProfile(z_g_t0=0, w_A_g=5, tf=tf),
            lambda tf: LinearAirspeedProfile.init_from(
                V_A_t0=20, V_A_tf=40, tf=tf),
            t_res=1e-2)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20, places=3)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40, places=3)
        self.assertAlmostEqual(seg_reparam.x0["w_A_g"], 5, places=3)
        self.assertAlmostEqual(seg_reparam.xf["w_A_g"], 5, places=3)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0, places=3)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 240, places=3)

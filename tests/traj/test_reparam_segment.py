# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.reparam_segment.

Run from top-level package directory:

   python3 -m unittest tests.traj.test_reparam_segment

"""

import math
import random
import unittest

from eduplane.traj.airspeed_profiles import LinearAirspeedProfile
from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, SteadyTurn)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.reparam_segment import *
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile


class ReparamSegmentTests(unittest.TestCase):

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_reparam_segment(self):

        # Tests with straight horizontal path
        # ==============================================

        # Generate straight-level segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        seg = LiftedSegment(
            ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=15),
        )
        seg_discr = seg.discretize()
        seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 15)
        self.assertAlmostEqual(seg.x0["V_A"], 20)
        self.assertAlmostEqual(seg.xf["V_A"], 20)
        self.assertAlmostEqual(seg.x0["l_xy"], 0)
        self.assertAlmostEqual(seg.xf["l_xy"], 300)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(seg_discr.x0["l"], 0)
        self.assertAlmostEqual(seg_discr.xf["l"], 300)

        # Re-parametrize with linear airspeed, no wind
        # 300 m at 20-40 m/s (linear increase -> 30 m/s mean) in 10 s
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10)).discretize(
                t_res=0.1, tf=10)

        # from matplotlib import pyplot
        # seg_reparam.discretize().df[["u_A_g", "u_A_g_dot"]].plot()
        # pyplot.show()

        seg_reparam.discretize().check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(seg_reparam.t0, 0)
        self.assertAlmostEqual(seg_reparam.tf, 10)
        self.assertAlmostEqual(seg_reparam.x0["l"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l"], 300)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 300)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40)

        # Re-parametrize the already re-parametrized segment with a new linear
        # airspeed profile:
        #   300 m at 40-80 m/s (linear increase -> 60 m/s mean) in 5 s
        # seg_rereparam = ReparamSegment(
        #     seg_reparam, LinearAirspeedProfile(V_A_t0=40, V_A_dot=8,
        #                                        tf=5)).discretize(
        #                                            t_res=0.1, tf=5)
        seg_rereparam = ReparamSegment(
            seg_reparam, LinearAirspeedProfile(V_A_t0=40, V_A_dot=8, tf=5))
        seg_rereparam.discretize().check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        # DEBUG:
        # with pandas.option_context('display.max_rows', None,
        #                            'display.max_columns', None):
        #     print(
        #         seg_rereparam.df[[
        #             "t_orig", "V_K_xy", "V_K_xy_dot", "V_A_xy", "V_A_xy_dot",
        #
        #  "V_K", "V_K_dot", "l_xy"
        #         ]])
        #     print(seg_reparam.df[["t_orig", "l_xy"]])
        self.assertAlmostEqual(seg_rereparam.t0, 0)
        self.assertAlmostEqual(seg_rereparam.tf, 5)
        self.assertAlmostEqual(seg_rereparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_rereparam.xf["l_xy"], 300)
        self.assertAlmostEqual(seg_rereparam.x0["V_A"], 40)
        self.assertAlmostEqual(seg_rereparam.xf["V_A"], 80)

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
        # Same circumference at 20-40 m/s (linear increase -> 30 m/s mean) in 45 s
        tf_at_mean_airspeed_30 = fullturn_l_xy/30
        seg_reparam = ReparamSegment(
            lifted_seg,
            LinearAirspeedProfile.init_from(
                V_A_t0=20, V_A_tf=40, tf=tf_at_mean_airspeed_30)).discretize(
                    t_res=0.1, tf=fullturn_l_xy/30)

        # from matplotlib import pyplot
        # # ax = lifted_seg.discretize().df[["x_g", "y_g"]].plot(x="x_g", y="y_g")
        # # seg_reparam.discretize().df[["x_g", "y_g"]].plot(
        # #     x="x_g", y="y_g", ax=ax)
        # ax = lifted_seg.discretize().df[["V_A", "V_K"]].plot()
        # seg_reparam.discretize().df[["V_A", "V_K"]].plot(ax=ax)
        # # ax = lifted_seg.discretize().df[["chi", "chi_dot"]].plot()
        # # seg_reparam.discretize().df[["chi", "chi_dot"]].plot(ax=ax)
        # pyplot.show()

        seg_reparam_discr = seg_reparam.discretize()
        seg_reparam_discr.check_kinematic_consistency(raise_err=True)

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
        # print(fullturn_l_xy)
        # print(seg_reparam.xf["l_xy"])
        # print(lifted_seg.horiz_profile.xf["l_xy"])
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], fullturn_l_xy, places=1)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(seg_reparam.x0["l"], 0)
        self.assertAlmostEqual(
            seg_reparam_discr.xf["l"], fullturn_l_xy, places=1)

        # Tests with straight climbing path
        # ==============================================

        # Generate straight-climb segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        seg = LiftedSegment(
            ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=-5, tf=15),
        )
        seg_discr = seg.discretize()
        seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 15)
        self.assertAlmostEqual(seg.x0["V_A_xy"], 20)
        self.assertAlmostEqual(seg.xf["V_A_xy"], 20)
        self.assertAlmostEqual(seg.x0["l_xy"], 0)
        self.assertAlmostEqual(seg.xf["l_xy"], 300)
        self.assertAlmostEqual(seg.x0["V_A"], math.sqrt(20**2 + 5**2))
        self.assertAlmostEqual(seg.xf["V_A"], math.sqrt(20**2 + 5**2))
        self.assertAlmostEqual(seg.x0["z_g"], 0)
        self.assertAlmostEqual(seg.xf["z_g"], -5*15)
        # Note how non-parametric parameter 'l' is available only from the discretized segment
        self.assertAlmostEqual(seg_discr.x0["l"], 0)
        self.assertAlmostEqual(
            seg_discr.xf["l"], math.sqrt(300**2 + (-5*15)**2))

        # Re-parametrize with linear airspeed, no wind
        # Linear increase from 20-40 m/s
        seg_reparam = ReparamSegment(
            seg,
            LinearAirspeedProfile.init_from(
                V_A_t0=20, V_A_tf=40, tf=seg_discr.xf["l"]/30))

        self.assertAlmostEqual(seg_reparam.t0, 0)
        self.assertAlmostEqual(seg_reparam.tf, seg_discr.xf["l"]/30, places=1)
        self.assertAlmostEqual(seg_reparam.x0["l"], seg_discr.x0["l"])
        self.assertAlmostEqual(seg_reparam.xf["l"], seg_discr.xf["l"])
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 300)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40)
        self.assertAlmostEqual(seg_reparam.x0["z_g"], 0)
        self.assertAlmostEqual(seg_reparam.xf["x_g"], seg_discr.xf["x_g"])
        self.assertAlmostEqual(seg_reparam.xf["y_g"], seg_discr.xf["y_g"])
        self.assertAlmostEqual(seg_reparam.xf["z_g"], seg_discr.xf["z_g"])

        seg_reparam_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        # # Re-parametrize with a shorter linear airspeed profile:
        # #   20-40 m/s in 5 seconds (shorter than previous segment)
        # seg_reparam = ReparamSegment(
        #     seg, LinearAirspeedProfile.init_from(V_A_t0=20, V_A_tf=40,
        #                                          tf=15)).discretize()
        # seg_reparam.discretize().check_kinematic_consistency(
        #     horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        # seg_reparam_discr = seg_reparam.discretize()
        # from eduplane.utils import plotting
        # plotting.plot_ref_trajectory(
        #     [seg_reparam_discr.df], ["3D traj."],
        #     0,
        #     0,
        #     xy_projection=True,
        #     timelines=["w_A_g", "V_A", "V_A_dot"],
        #     POIs=[
        #         dict(
        #             x_g=seg_discr.x0["x_g"],
        #             y_g=seg_discr.x0["y_g"],
        #             z_g=seg_discr.x0["z_g"],
        #             label="start",
        #             style="go"),
        #         dict(
        #             x_g=seg_discr.xf["x_g"],
        #             y_g=seg_discr.xf["y_g"],
        #             z_g=seg_discr.xf["z_g"],
        #             label="goal",
        #             style="gx")
        #     ])
        # # plotting.plot_3D_view(
        # #     [traj_discr.df], z_scale=1, xy_projection=True)
        # plotting.plt.show()

        # # Test wrapping with Trajectory
        # traj = Trajectory([ReparamSegment])
        # traj_discr = traj.discretize()
        # traj_discr.check_kinematic_consistency(
        #     horizontal_continuity=2, vertical_continuity=2, raise_err=True)

    def test_no_final_time(self):
        # Generate straight-level segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        seg = LiftedSegment(
            ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=15),
        )
        seg.discretize().check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)

        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 15)
        self.assertAlmostEqual(seg.x0["l_xy"], 0)
        self.assertAlmostEqual(seg.xf["l_xy"], 300)
        self.assertAlmostEqual(seg.x0["V_A"], 20)
        self.assertAlmostEqual(seg.xf["V_A"], 20)

        # Re-parametrize with half airspeed (i.e. double time) without a
        # final time given.
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=10, V_A_dot=0, tf=30))
        self.assertAlmostEqual(seg_reparam.tf, seg.tf*2)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 300)
        self.assertTrue(seg_reparam.check_alignment())

        # If the airspeed profile is longer than the resulting re-parametrized
        # segment's duration, the airspeed profile will be truncated
        # (i.e. same result for constant airspeed here)
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=10, V_A_dot=0, tf=35))
        self.assertAlmostEqual(seg_reparam.tf, seg.tf*2)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 300)
        self.assertFalse(seg_reparam.check_alignment())

        # If the airspeed profile is shorter than the resulting re-parametrized
        # segment's duration, the segment will be truncated
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=10, V_A_dot=0, tf=25))
        self.assertAlmostEqual(seg_reparam.tf, 25)
        self.assertAlmostEqual(seg_reparam.x0["l_xy"], 0)
        self.assertAlmostEqual(seg_reparam.xf["l_xy"], 250)
        self.assertFalse(seg_reparam.check_alignment())

    def test_alignment(self):
        # Straight horizontal path
        # ==============================================

        # Generate straight-level segment with constant airspeed, no wind
        # 300 m at 20 m/s in 15 s
        seg = LiftedSegment(
            ConstSpeedStraightFlight(V_A_xy=20, chi=0, tf=15),
            ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=15),
        )

        # Re-parametrize with linear airspeed profile
        # 300 m at 20-40 m/s (linear increase -> 30 m/s mean) in 10 s
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=10))
        self.assertTrue(seg_reparam.check_alignment())

        # Re-parametrize with shorter linear airspeed profile
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=8))
        self.assertFalse(seg_reparam.check_alignment())

        # Re-parametrize with longer linear airspeed profile
        seg_reparam = ReparamSegment(
            seg, LinearAirspeedProfile(V_A_t0=20, V_A_dot=2, tf=12))
        self.assertFalse(seg_reparam.check_alignment())

        seg_reparam = ReparamSegment.align_linear_airspeed(
            seg, 20, 40, t_res=1e-2)
        self.assertAlmostEqual(seg_reparam.tf, 10, places=3)
        self.assertAlmostEqual(seg_reparam.x0["V_A"], 20, places=3)
        self.assertAlmostEqual(seg_reparam.xf["V_A"], 40, places=3)
        self.assertAlmostEqual(seg_reparam.x0["V_A_dot"], 2, places=3)
        self.assertTrue(seg_reparam.check_alignment())

    def test_dt_from_dl_xy(self):
        random.seed(1234567890)

        for _ in range(1000):
            V_K_xy_t0 = self.random(1, 100)
            V_K_xy_tf = self.random(1, 100)
            dt = self.random(0.1, 1.0)
            V_K_xy_dot = (V_K_xy_tf-V_K_xy_t0)/dt
            dl_xy = dl_xy_from_dt(dt, V_K_xy_t0, V_K_xy_dot)
            # Since we ensured V_K_xy > 0, we should get a positive length
            self.assertGreater(dl_xy, 0)
            po2 = V_K_xy_t0/V_K_xy_dot
            q = -2*dl_xy/V_K_xy_dot
            radicand = po2**2 - q
            dt1 = -po2 + numpy.sqrt(radicand)
            dt2 = -po2 - numpy.sqrt(radicand)

            self.assertAlmostEqual(
                dt_from_dl_xy(dl_xy, V_K_xy_t0, V_K_xy_dot), dt)

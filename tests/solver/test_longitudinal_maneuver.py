# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.longitudinal_maneuver.

Run from top-level package directory:

    python3 -m unittest tests.solver.test_longitudinal_maneuver

"""

import math
import random
import unittest

import numpy

from eduplane.model.ext_dubins_airplane import ExtDubinsAirplane
from eduplane.solver.longitudinal_maneuver import *
from eduplane.traj.horizontal_profiles.const_speed_straight_flight import (
    ConstSpeedStraightFlight)
from eduplane.traj.horizontal_profiles.linear_speed_straight_flight import (
    LinearSpeedStraightFlight)
from eduplane.traj.horizontal_profiles.steady_turn import SteadyTurn
from eduplane.traj.trajectory import Trajectory
from eduplane.utils import plotting

# from qbstyles import mpl_style

# mpl_style(dark=True, minor_ticks=False)

DEBUG_OUTPUT = False
DEBUG_PLOTS = False
# DEBUG_PLOTS = True


def get_steady_state_lifted_segment(
        tf, x_g_t0, y_g_t0, z_g_t0, V_A_xy, w_A_g, chi, V_W_xy, chi_W):
    """Construct a lifted segment corresponding to steady-state flight with the given parameters."""
    horiz_profile = ConstSpeedStraightFlight(
        x_g_t0=x_g_t0,
        y_g_t0=y_g_t0,
        tf=tf,
        V_A_xy=V_A_xy,
        chi=chi,
        V_W_xy=V_W_xy,
        chi_W=chi_W)
    vert_profile = ConstVertSpeedProfile(z_g_t0=z_g_t0, w_A_g=w_A_g, tf=tf)
    return LiftedSegment(horiz_profile, vert_profile)


def get_steady_to_steady_transition(
        m: LongitudinalManeuver, steady_segment_time=5):
    """Construct a trajectory with initial and final steady-state segments, and the longitudinal maneuver in between."""
    seg_before = get_steady_state_lifted_segment(
        tf=steady_segment_time,
        x_g_t0=0,
        y_g_t0=0,
        z_g_t0=0,
        V_A_xy=m.V_A_xy_t0,
        w_A_g=m.w_A_g_t0,
        chi=m.chi,
        V_W_xy=m.V_W_xy,
        chi_W=m.chi_W,
    )
    lon_man_lifted = m.getLiftedSegment(
        x_g_t0=seg_before.xf["x_g"], y_g_t0=seg_before.xf["y_g"])
    seg_before.shift(seg_before.t0, z_g_t0=m.z_g_t0 - seg_before.xf["z_g"])
    seg_after = get_steady_state_lifted_segment(
        tf=steady_segment_time,
        x_g_t0=lon_man_lifted.xf["x_g"],
        y_g_t0=lon_man_lifted.xf["y_g"],
        z_g_t0=lon_man_lifted.xf["z_g"],
        V_A_xy=m.V_A_xy_tf,
        w_A_g=m.w_A_g_tf,
        chi=m.chi,
        V_W_xy=m.V_W_xy,
        chi_W=m.chi_W,
    )
    return Trajectory([seg_before, lon_man_lifted, seg_after])


class LongitudinalManeuverTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_rapid_climb_transition(self):
        ac = ExtDubinsAirplane(
            g=9.81,
            m=4,
            rho=1.225,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=5000,
            n_min=0.2,
            n_max=1.2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        V_A_t0 = 20
        gamma_A_t0 = -0.1

        test_case = dict(
            z_g_t0=-50,
            xy_dist=100,
            V_A_xy_t0=vt.V_A_xy_from_V_A_gamma_A(V_A_t0, gamma_A_t0),
            w_A_g_t0=vt.w_A_g_from_V_A_gamma_A(V_A_t0, gamma_A_t0),
            chi=0,
            V_W_xy=5,
            chi_W=math.radians(45),
        )

        T_dec = 2

        m = LongitudinalManeuver.solve_rapid_climb_fixed_length(
            ac, T_dec=T_dec, **test_case)

        traj = get_steady_to_steady_transition(m, 5)

        traj_discr = traj.discretize()

        lon_man_discr = traj.segments[1].discretize()
        self.assertAlmostEqual(
            lon_man_discr.xf["l_xy"] - lon_man_discr.x0["l_xy"],
            test_case["xy_dist"])

        if DEBUG_PLOTS:
            plotting.plot_ref_trajectory(traj_discr.df)
            ac.fly_trajectory(traj_discr)
            plotting.plot_timeline(
                traj_discr.df,
                ["V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "gamma_A", "T"])
            plotting.show()

    def test_rapid_descent_transition_fixed_airspeed(self):
        ac = ExtDubinsAirplane(
            g=9.81,
            m=4,
            rho=1.225,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=5000,
            n_min=0.2,
            n_max=1.2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        xy_dist = 100
        z_g_t0 = -100
        gamma_A_t0 = 0
        gamma_A_min_range = numpy.linspace(-math.radians(30), gamma_A_t0, 10)
        V_A_t0 = 20
        V_A_xy_t0 = vt.V_A_xy_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        w_A_g_t0 = vt.w_A_g_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        V_A_max = ac.V_A_max
        chi = 0
        V_W_xy = 5
        chi_W = math.radians(45)
        T_acc = 0

        # Rapid descent transition with constant airspeed
        # ============================================

        discr_paths = []
        for gamma_A_min in gamma_A_min_range:
            m = LongitudinalManeuver.solve_rapid_descent_fixed_length_fixed_airspeed(
                ac,
                V_A_xy_t0,
                w_A_g_t0,
                z_g_t0,
                chi,
                V_W_xy,
                chi_W,
                xy_dist,
                gamma_A_min=gamma_A_min)

            # Construct trajectory with initial final steady-state segments
            seg1_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=0,
                y_g_t0=0,
                tf=5,
                V_A_xy=V_A_xy_t0,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg2_horiz_profile = LinearSpeedStraightFlight(
                x_g_t0=seg1_horiz_profile.xf["x_g"],
                y_g_t0=seg1_horiz_profile.xf["y_g"],
                tf=m.tf,
                V_A_xy_t0=m.V_A_xy_t0,
                V_A_xy_dot=m.V_A_xy_dot,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg3_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=seg2_horiz_profile.xf["x_g"],
                y_g_t0=seg2_horiz_profile.xf["y_g"],
                tf=5,
                V_A_xy=seg2_horiz_profile.xf["V_A_xy"],
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg1_vert_profile = ConstVertSpeedProfile(
                z_g_t0=z_g_t0, w_A_g=w_A_g_t0, tf=5)
            seg2_vert_profile = m.getVerticalProfile()
            seg3_vert_profile = ConstVertSpeedProfile(
                z_g_t0=seg2_vert_profile.xf["z_g"],
                w_A_g=seg2_vert_profile.xf["w_A_g"],
                tf=5)
            lifted_seg = LiftedSegment(
                Trajectory(
                    [
                        seg1_horiz_profile, seg2_horiz_profile,
                        seg3_horiz_profile
                    ]),
                Trajectory(
                    [seg1_vert_profile, seg2_vert_profile, seg3_vert_profile]))
            discr_path = lifted_seg.discretize()

            ac.fly_trajectory(discr_path)
            discr_paths.append(discr_path)

            self.assertAlmostEqual(
                seg2_horiz_profile.xf["l_xy"] - seg2_horiz_profile.x0["l_xy"],
                xy_dist,
                places=0)
            self.assertGreaterEqual(lifted_seg.xf["gamma_A"], gamma_A_min)

        if DEBUG_PLOTS:
            plotting.plot_ref_trajectory([p.df for p in discr_paths])
            ac.fly_trajectory(discr_path)
            plotting.plot_timeline(
                [p.df for p in discr_paths],
                ["V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "gamma_A", "T"])
            plotting.show()

        # Rapid descent transition with increasing airspeed
        # =================================================
        V_A_xy_delta = 2.5

        discr_paths = []
        for gamma_A_min in gamma_A_min_range:
            lon_man = LongitudinalManeuver.solve_rapid_descent_fixed_length_fixed_airspeed(
                ac,
                V_A_xy_t0,
                w_A_g_t0,
                z_g_t0,
                chi,
                V_W_xy,
                chi_W,
                xy_dist,
                gamma_A_min=gamma_A_min,
                V_A_xy_tf=V_A_xy_t0 + V_A_xy_delta)

            # Construct trajectory with initial final steady-state segments
            seg1_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=0,
                y_g_t0=0,
                tf=5,
                V_A_xy=V_A_xy_t0,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg2_horiz_profile = LinearSpeedStraightFlight(
                x_g_t0=seg1_horiz_profile.xf["x_g"],
                y_g_t0=seg1_horiz_profile.xf["y_g"],
                tf=lon_man.tf,
                V_A_xy_t0=lon_man.V_A_xy_t0,
                V_A_xy_dot=lon_man.V_A_xy_dot,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg3_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=seg2_horiz_profile.xf["x_g"],
                y_g_t0=seg2_horiz_profile.xf["y_g"],
                tf=5,
                V_A_xy=seg2_horiz_profile.xf["V_A_xy"],
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg1_vert_profile = ConstVertSpeedProfile(
                z_g_t0=z_g_t0, w_A_g=w_A_g_t0, tf=5)
            seg2_vert_profile = lon_man.getVerticalProfile()
            seg3_vert_profile = ConstVertSpeedProfile(
                z_g_t0=seg2_vert_profile.xf["z_g"],
                w_A_g=seg2_vert_profile.xf["w_A_g"],
                tf=5)
            # lifted_seg = LiftedSegment(
            #     Trajectory(
            #         [
            #             seg1_horiz_profile, seg2_horiz_profile,
            #             seg3_horiz_profile
            #         ]),
            #     Trajectory(
            #         [seg1_vert_profile, seg2_vert_profile, seg3_vert_profile]))
            lifted_seg = Trajectory(
                [
                    LiftedSegment(seg1_horiz_profile, seg1_vert_profile),
                    LiftedSegment(seg2_horiz_profile, seg2_vert_profile),
                    LiftedSegment(seg3_horiz_profile, seg3_vert_profile)
                ])
            discr_path = lifted_seg.discretize()

            ac.fly_trajectory(discr_path)
            discr_paths.append(discr_path)

            self.assertAlmostEqual(
                seg2_horiz_profile.xf["l_xy"] - seg2_horiz_profile.x0["l_xy"],
                xy_dist,
                places=0)
            self.assertGreaterEqual(lifted_seg.xf["gamma_A"], gamma_A_min)
            self.assertAlmostEqual(seg2_horiz_profile.x0["V_A_xy"], V_A_xy_t0)
            self.assertAlmostEqual(
                seg2_horiz_profile.xf["V_A_xy"], V_A_xy_t0 + V_A_xy_delta)

            # print("#### DEBUG THURST")
            # seg0 = lifted_seg.segments[0].discretize()
            # ac.fly_trajectory(seg0)
            # seg1 = lifted_seg.segments[1].discretize()
            # ac.fly_trajectory(seg1)
            # print(seg0.xf["T"])
            # print(seg1.x0["T"])
            # print(ac.thrust(V_A=lon_man.V_A_t0, w_A_g=0))
            # print(
            #     ac.thrust(
            #         V_A=lon_man.V_A_t0, w_A_g=0, w_A_g_dot=lon_man.w_A_g_dot))

        if DEBUG_PLOTS:
            plotting.plot_ref_trajectory([p.df for p in discr_paths])
            ac.fly_trajectory(discr_path)
            plotting.plot_timeline(
                [p.df for p in discr_paths], [
                    "V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "w_A_g_dot",
                    "gamma_A", "T"
                ])
            plotting.show()

    def test_rapid_descent_transition_free_airspeed(self):
        ac = ExtDubinsAirplane(
            g=9.81,
            m=4,
            rho=1.225,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=5000,
            n_min=0.2,
            n_max=1.2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        xy_dist = 100
        z_g_t0 = -100
        gamma_A_t0 = 0
        gamma_A_min = -math.radians(15)
        V_A_t0 = 15
        V_A_xy_t0 = vt.V_A_xy_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        w_A_g_t0 = vt.w_A_g_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        V_A_max = ac.V_A_max
        chi = 0
        V_W_xy = 5
        chi_W = math.radians(45)

        # Rapid descent transition with free final airspeed
        # =================================================

        discr_paths = []
        gamma_A_min_values = numpy.linspace(gamma_A_min, gamma_A_t0, 10)
        for gamma_A_min in gamma_A_min_values:
            m = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                ac,
                V_A_xy_t0,
                w_A_g_t0,
                z_g_t0,
                chi,
                V_W_xy,
                chi_W,
                xy_dist,
                gamma_A_min=gamma_A_min)

            # Construct trajectory with initial final steady-state segments
            seg1_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=0,
                y_g_t0=0,
                tf=5,
                V_A_xy=V_A_xy_t0,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg2_horiz_profile = LinearSpeedStraightFlight(
                x_g_t0=seg1_horiz_profile.xf["x_g"],
                y_g_t0=seg1_horiz_profile.xf["y_g"],
                tf=m.tf,
                V_A_xy_t0=m.V_A_xy_t0,
                V_A_xy_dot=m.V_A_xy_dot,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg3_horiz_profile = ConstSpeedStraightFlight(
                x_g_t0=seg2_horiz_profile.xf["x_g"],
                y_g_t0=seg2_horiz_profile.xf["y_g"],
                tf=5,
                V_A_xy=seg2_horiz_profile.xf["V_A_xy"],
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg1_vert_profile = ConstVertSpeedProfile(
                z_g_t0=z_g_t0, w_A_g=w_A_g_t0, tf=5)
            seg2_vert_profile = m.getVerticalProfile()
            seg3_vert_profile = ConstVertSpeedProfile(
                z_g_t0=seg2_vert_profile.xf["z_g"],
                w_A_g=seg2_vert_profile.xf["w_A_g"],
                tf=5)
            lifted_seg = LiftedSegment(
                Trajectory(
                    [
                        seg1_horiz_profile, seg2_horiz_profile,
                        seg3_horiz_profile
                    ]),
                Trajectory(
                    [seg1_vert_profile, seg2_vert_profile, seg3_vert_profile]))
            discr_path = lifted_seg.discretize()
            ac.fly_trajectory(discr_path)
            discr_paths.append(discr_path)

            self.assertAlmostEqual(
                seg2_horiz_profile.xf["l_xy"] - seg2_horiz_profile.x0["l_xy"],
                xy_dist,
                places=0)
            self.assertGreaterEqual(lifted_seg.xf["gamma_A"], gamma_A_min)

        if DEBUG_PLOTS:
            plotting.plot_ref_trajectory([p.df for p in discr_paths])
            ac.fly_trajectory(discr_path)
            plotting.plot_timeline(
                [p.df for p in discr_paths],
                ["V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "gamma_A", "T"])
            plotting.show()

    def test_rapid_descent_transition_between_turns(self):
        ac = ExtDubinsAirplane(
            g=9.81,
            m=4,
            rho=1.225,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=5000,
            n_min=0.2,
            n_max=1.2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        xy_dist = 500
        z_g_t0 = -100
        gamma_A_t0 = 0
        gamma_A_min = -math.radians(45)
        V_A_t0 = 25
        V_A_xy_t0 = vt.V_A_xy_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        w_A_g_t0 = vt.w_A_g_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        V_A_max = ac.V_A_max
        chi = 0
        turn_rate = math.radians(60)
        V_W_xy = 5
        chi_W = math.radians(45)

        T_turn = ac.thrust_required(
            V_A=V_A_t0, gamma_A=gamma_A_t0, chi_A_dot=turn_rate)

        discr_paths = []
        gamma_A_min_values = numpy.linspace(gamma_A_min, gamma_A_t0, 10)
        for gamma_A_min in gamma_A_min_values:
            # Construct trajectory with initial final steady-state segments
            seg1_horiz_profile = SteadyTurn(
                x_g_t0=0,
                y_g_t0=0,
                tf=5,
                V_A_xy=V_A_xy_t0,
                chi_t0=chi,
                chi_A_dot=turn_rate,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            m = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                ac,
                V_A_xy_t0,
                w_A_g_t0,
                z_g_t0,
                seg1_horiz_profile.xf["chi"],
                V_W_xy,
                chi_W,
                xy_dist,
                T_max=T_turn,
                gamma_A_min=gamma_A_min)
            seg2_horiz_profile = LinearSpeedStraightFlight(
                x_g_t0=seg1_horiz_profile.xf["x_g"],
                y_g_t0=seg1_horiz_profile.xf["y_g"],
                tf=m.tf,
                V_A_xy_t0=m.V_A_xy_t0,
                V_A_xy_dot=m.V_A_xy_dot,
                chi=m.chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg3_horiz_profile = SteadyTurn(
                x_g_t0=seg2_horiz_profile.xf["x_g"],
                y_g_t0=seg2_horiz_profile.xf["y_g"],
                tf=5,
                V_A_xy=seg2_horiz_profile.xf["V_A_xy"],
                chi_t0=seg2_horiz_profile.xf["chi"],
                chi_A_dot=turn_rate,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            seg1_vert_profile = ConstVertSpeedProfile(
                z_g_t0=z_g_t0, w_A_g=w_A_g_t0, tf=5)
            seg2_vert_profile = m.getVerticalProfile()
            seg3_vert_profile = ConstVertSpeedProfile(
                z_g_t0=seg2_vert_profile.xf["z_g"],
                w_A_g=seg2_vert_profile.xf["w_A_g"],
                tf=5)
            lifted_seg = LiftedSegment(
                Trajectory(
                    [
                        seg1_horiz_profile, seg2_horiz_profile,
                        seg3_horiz_profile
                    ]),
                Trajectory(
                    [seg1_vert_profile, seg2_vert_profile, seg3_vert_profile]))
            discr_path = lifted_seg.discretize()
            ac.fly_trajectory(discr_path)
            discr_paths.append(discr_path)

            seg2_horiz_profile_discr = seg2_horiz_profile.discretize()
            self.assertAlmostEqual(
                seg2_horiz_profile_discr.xf["l_xy"] -
                seg2_horiz_profile_discr.x0["l_xy"],
                xy_dist,
                places=1)
            self.assertGreaterEqual(lifted_seg.xf["gamma_A"], gamma_A_min)

        if DEBUG_PLOTS:
            plotting.plot_ref_trajectory([p.df for p in discr_paths])
            plotting.plot_timeline(
                [p.df for p in discr_paths],
                ["V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "gamma_A", "T"])
            plotting.show()

    def test_segmented_rapid_descent_transition(self):
        ac = ExtDubinsAirplane(
            g=9.81,
            m=4,
            rho=1.225,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=5000,
            n_min=0.2,
            n_max=1.2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        # w_A_g_max, V_A_w_A_g_max = ac.descent_rate_max()
        # print(f"w_A_g_max: {w_A_g_max}, V_A_w_A_g_max: {V_A_w_A_g_max}")
        # w_A_g_max_at_20 = ac.descent_rate_max(V_A=20)
        # print(f"w_A_g_max_at_20: {w_A_g_max_at_20}")
        # w_A_g_max_at_15 = ac.descent_rate_max(V_A=15)
        # print(f"w_A_g_max_at_15: {w_A_g_max_at_15}")

        xy_dist = 100
        z_g_t0 = -100
        gamma_A_t0 = 0
        gamma_A_min = -math.radians(45)
        V_A_t0 = 15
        V_A_xy_t0 = vt.V_A_xy_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        w_A_g_t0 = vt.w_A_g_from_V_A_gamma_A(V_A_t0, gamma_A_t0)
        V_A_max = ac.V_A_max
        chi = 0
        V_W_xy = 5
        chi_W = math.radians(45)
        T_acc = 0.1

        horiz_profiles = [
            ConstSpeedStraightFlight(
                x_g_t0=0,
                y_g_t0=0,
                tf=5,
                V_A_xy=V_A_xy_t0,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
        ]
        vert_profiles = [
            ConstVertSpeedProfile(
                z_g_t0=z_g_t0,
                w_A_g=w_A_g_t0,
                tf=5,
            )
        ]

        T_t0 = ac.thrust_required(w_A_g=w_A_g_t0, V_A=V_A_t0)
        n_segments = 1
        for i in range(n_segments):
            m = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                ac,
                horiz_profiles[-1].xf["V_A_xy"],
                vert_profiles[-1].xf["w_A_g"],
                vert_profiles[-1].xf["z_g"],
                chi,
                V_W_xy,
                chi_W,
                xy_dist/n_segments,
                T_min=0,
            )
            T_delta = m.T_t0 - T_t0
            m = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                ac,
                horiz_profiles[-1].xf["V_A_xy"],
                vert_profiles[-1].xf["w_A_g"],
                vert_profiles[-1].xf["z_g"],
                chi,
                V_W_xy,
                chi_W,
                xy_dist/n_segments,
                T_min=T_delta,
            )
            T_t0 = m.T_tf
            horiz_profiles.append(
                LinearSpeedStraightFlight(
                    x_g_t0=horiz_profiles[-1].xf["x_g"],
                    y_g_t0=horiz_profiles[-1].xf["y_g"],
                    tf=m.tf,
                    V_A_xy_t0=m.V_A_xy_t0,
                    V_A_xy_dot=m.V_A_xy_dot,
                    chi=chi,
                    V_W_xy=V_W_xy,
                    chi_W=chi_W))
            vert_profiles.append(m.getVerticalProfile())

        # n_segments = 180
        # for i in range(n_segments):
        #     m = LongitudinalManeuver.solve_rapid_descent_fixed_length(
        #         ac,
        #         horiz_profiles[-1].xf["V_A_xy"],
        #         vert_profiles[-1].xf["w_A_g"],
        #         vert_profiles[-1].xf["z_g"],
        #         chi,
        #         V_W_xy,
        #         chi_W,
        #         xy_dist/n_segments,
        #         # T_acc,
        #         T_acc if i > 0 else 0,
        #     )
        #     horiz_profiles.append(
        #         LinearSpeedStraightFlight(
        #             x_g_t0=horiz_profiles[-1].xf["x_g"],
        #             y_g_t0=horiz_profiles[-1].xf["y_g"],
        #             tf=m.tf,
        #             V_A_xy_t0=m.V_A_xy_t0,
        #             V_A_xy_dot=m.V_A_xy_dot,
        #             chi=chi,
        #             V_W_xy=V_W_xy,
        #             chi_W=chi_W))
        #     vert_profiles.append(m.getVerticalProfile())

        horiz_profiles.append(
            ConstSpeedStraightFlight(
                x_g_t0=horiz_profiles[-1].xf["x_g"],
                y_g_t0=horiz_profiles[-1].xf["y_g"],
                tf=5,
                V_A_xy=horiz_profiles[-1].xf["V_A_xy"],
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W))
        vert_profiles.append(
            ConstVertSpeedProfile(
                z_g_t0=vert_profiles[-1].xf["z_g"],
                w_A_g=vert_profiles[-1].xf["w_A_g"],
                tf=5))

        # TODO: Add checks
        self.assertIsNotNone(vert_profiles)
        # lifted_seg = LiftedSegment(
        #     Trajectory(horiz_profiles), Trajectory(vert_profiles))
        # discr_path = lifted_seg.discretize()

        # plotting.plot_ref_trajectory(discr_path.df)
        # ac.fly_trajectory(discr_path)
        # plotting.plot_timeline(
        #     discr_path.df, [
        #         "V_A", "V_A_xy", "V_A_xy_dot", "w_A_g", "w_A_g_dot", "gamma_A",
        #         "T", "n"
        #     ])
        # plotting.show()

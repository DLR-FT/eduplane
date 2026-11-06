# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.trochoid_path.

Run from top-level package directory:

   python3 -m unittest tests.traj.horizontal_profiles.test_circular_turn

"""

import math
import random
import unittest

import numpy

from eduplane.traj.horizontal_profiles.circular_turn import CircularTurn
from eduplane.utils import utils

DEBUG_OUTPUT = False
DEBUG_PLOTS = False
# DEBUG_PLOTS = True


class CircularTurnTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_circular_turn_no_wind(self):
        radius = 100
        airspeed = 15
        start_x = 100
        start_y = 100
        start_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        end_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        turn_dir_values = [1, -1]
        turn_rate = airspeed/radius
        for turn_dir in turn_dir_values:
            for start_dir in start_dir_values:
                for end_dir in end_dir_values:
                    # start_dir = 0.25*math.pi
                    # end_dir = 0
                    # turn_dir = -1
                    # print(f"start_dir: {math.degrees(start_dir)}")
                    # print(f"end_dir: {math.degrees(end_dir)}")

                    # Create a 270 steady turn with no wind
                    turn = CircularTurn(
                        x_g_t0=start_x,
                        y_g_t0=start_y,
                        V_A_xy=airspeed,
                        chi_t0=start_dir,
                        chi_tf=end_dir,
                        radius=radius,
                        V_W_xy=0,
                        chi_W=0,
                        turn_dir=turn_dir)
                    self.assertAlmostEqual(
                        turn.tf*turn_rate*turn_dir,
                        utils.angular_diff(start_dir, end_dir, turn_dir))
                    self.assertAlmostEqual(turn.x0["chi"], start_dir)
                    self.assertAlmostEqual(
                        utils.angular_diff(turn.xf["chi"], end_dir), 0)
                    self.assertAlmostEqual(turn.x0["x_g"], start_x)
                    self.assertAlmostEqual(turn.x0["y_g"], start_y)
                    turn_discr = turn.discretize()
                    turn_discr.check_kinematic_consistency(
                        horizontal_continuity=2)

                    # TODO: Implement wind re-parametrization
                    # turn_discr.df["V_W_xy"] = wind_speed
                    # turn_reparam = ReparamLiftedSegment(
                    #     turn_discr,
                    #     ConstVertSpeedProfile(z_g_t0=0, w_A_g=0, tf=turn_discr.tf),
                    #     LinearAirspeedProfile(V_A_t0=airspeed, V_A_dot=0, tf=turn_discr.tf))
                    # Add "air-path"
                    turn_discr.df["x_a"] = turn.x_a(turn_discr.df.index.values)
                    turn_discr.df["y_a"] = turn.y_a(turn_discr.df.index.values)
                    # Plot horizontal path
                    # plotting.plot_topview_with_airpath(turn_discr.df)
                    # plotting.show()

    def test_circular_turn_with_wind(self):
        wind_speed = 5
        wind_dir = math.pi*0.25
        radius = 100
        airspeed = 15
        start_x = 100
        start_y = 100
        start_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        end_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        turn_dir_values = [1, -1]
        for turn_dir in turn_dir_values:
            for start_dir in start_dir_values:
                for end_dir in end_dir_values:
                    # start_dir = 0.25*math.pi
                    # end_dir = 0
                    # turn_dir = -1
                    # print(f"start_dir: {math.degrees(start_dir)}")
                    # print(f"end_dir: {math.degrees(end_dir)}")

                    # Create a 270 steady turn with no wind
                    turn = CircularTurn(
                        x_g_t0=start_x,
                        y_g_t0=start_y,
                        V_A_xy=airspeed,
                        chi_t0=start_dir,
                        chi_tf=end_dir,
                        radius=radius,
                        V_W_xy=wind_speed,
                        chi_W=wind_dir,
                        turn_dir=turn_dir)
                    self.assertAlmostEqual(turn.x0["chi"], start_dir)
                    self.assertAlmostEqual(
                        utils.angular_diff(turn.xf["chi"], end_dir), 0)
                    self.assertAlmostEqual(turn.x0["x_g"], start_x)
                    self.assertAlmostEqual(turn.x0["y_g"], start_y)
                    turn_discr = turn.discretize()
                    turn_discr.check_kinematic_consistency(
                        horizontal_continuity=2)
                    # # Add "air-path"
                    # turn_discr.df["x_a"] = turn.x_a(turn_discr.df.index.values)
                    # turn_discr.df["y_a"] = turn.y_a(turn_discr.df.index.values)
                    # # Plot horizontal path
                    # plotting.plot_topview_with_airpath(turn_discr.df)
                    # plotting.show()

    def test_circular_turn_full_turns(self):
        wind_speed = 0
        wind_dir = 0
        radius = 100
        airspeed = 15
        start_x = 100
        start_y = 100
        start_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        end_dir_values = math.pi*numpy.array(
            [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        # start_dir_values = math.pi*numpy.array([0])
        # end_dir_values = math.pi*numpy.array([0.25])
        n_full_turn_values = [0, 1, 2]
        turn_dir_values = [1, -1]
        turn_rate = airspeed/radius
        for turn_dir in turn_dir_values:
            for start_dir in start_dir_values:
                for end_dir in end_dir_values:
                    for n_full_turns in n_full_turn_values:
                        # Create a 270 steady turn with no wind
                        turn = CircularTurn(
                            x_g_t0=start_x,
                            y_g_t0=start_y,
                            V_A_xy=airspeed,
                            chi_t0=start_dir,
                            chi_tf=end_dir,
                            radius=radius,
                            V_W_xy=wind_speed,
                            chi_W=wind_dir,
                            turn_dir=turn_dir,
                            n_full_turns=n_full_turns)
                        self.assertGreaterEqual(
                            turn.duration, n_full_turns*2*math.pi/turn_rate)
                        self.assertAlmostEqual(turn.x0["chi"], start_dir)
                        self.assertAlmostEqual(
                            utils.angular_diff(turn.xf["chi"], end_dir), 0)
                        self.assertAlmostEqual(turn.x0["x_g"], start_x)
                        self.assertAlmostEqual(turn.x0["y_g"], start_y)
                        turn_discr = turn.discretize()
                        turn_discr.check_kinematic_consistency(
                            horizontal_continuity=2)
                        # # Add "air-path"
                        # turn_discr.df["x_a"] = turn.x_a(
                        #     turn_discr.df.index.values)
                        # turn_discr.df["y_a"] = turn.y_a(
                        #     turn_discr.df.index.values)
                        # # Plot horizontal path
                        # plotting.plot_topview_with_airpath(turn_discr.df)
                        # plotting.show()


# Run from folder containing the top-level package:
#   python3 -m eduplane.traj.trajectory
if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import math

from eduplane.model import velocity_triangle as vt
from eduplane.traj import Trajectory
from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, SteadyTurn)
from eduplane.utils import utils


def get_full_turn(
        start_x, start_y, start_dir, airspeed_straight, airspeed_turn,
        turn_rate, wind_speed, wind_dir):
    """Just a full turn - not really a periodic path as wind drift may place the aircraft at constant offsets each round."""
    return SteadyTurn(
        x_g_t0=start_x,
        y_g_t0=start_y,
        V_A_xy=airspeed_turn,
        chi_A_dot=turn_rate,
        chi_t0=start_dir,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        tf=2*math.pi/abs(turn_rate),
    )


def get_trochoid_periodic_path(
        start_x, start_y, start_dir, airspeed_straight, airspeed_turn,
        turn_rate, wind_speed, wind_dir):
    t2pi = 2*math.pi/abs(turn_rate)
    # Turn into the wind
    seg1_chi_A_t0 = vt.chi_A_from_V_A_xy_chi(
        airspeed_turn, start_dir, wind_speed, wind_dir)
    seg1_chi_A_tf = wind_dir + math.pi
    seg1_delta_chi_A = utils.angular_diff(
        seg1_chi_A_t0, seg1_chi_A_tf, turn_rate)
    seg1_tf = seg1_delta_chi_A/turn_rate
    seg1 = SteadyTurn(
        x_g_t0=start_x,
        y_g_t0=start_y,
        V_A_xy=airspeed_turn,
        chi_A_dot=turn_rate,
        chi_t0=start_dir,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        tf=seg1_tf,
    )
    # Straight segment to compensate full turn wind drift
    seg2_length = wind_speed*t2pi
    seg2 = ConstSpeedStraightFlight.init_from(
        y_g_t0=seg1.xf["y_g"],
        x_g_t0=seg1.xf["x_g"],
        l_xy_tf=seg2_length,
        V_A_xy=airspeed_straight,
        chi=wind_dir + math.pi,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
    )
    # Turn back to initial course
    seg3_tf = t2pi - seg1_tf
    seg3 = SteadyTurn(
        y_g_t0=seg2.xf["y_g"],
        x_g_t0=seg2.xf["x_g"],
        V_A_xy=airspeed_turn,
        chi_A_dot=turn_rate,
        chi_t0=wind_dir + math.pi,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        tf=seg3_tf,
    )
    return Trajectory([seg1, seg2, seg3])

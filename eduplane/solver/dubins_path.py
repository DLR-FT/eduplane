# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Solvers to calculate Dubins paths:
  - CSC: curve-straight-curve path
"""
import math

from eduplane.model import flight_performance_model as fpm
from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.traj import Trajectory
from eduplane.traj.horizontal_profiles import (
    CircularTurn, ConstSpeedStraightFlight)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.vertical_profiles import ConstVertSpeedProfile
from eduplane.utils import logger, utils


def get_turn_dir(path_type, segment_index):
    if path_type == "LSL":
        return (-1, 0, -1)[segment_index]
    elif path_type == "RSR":
        return (1, 0, 1)[segment_index]
    elif path_type == "LSR":
        return (-1, 0, 1)[segment_index]
    elif path_type == "RSL":
        return (1, 0, -1)[segment_index]
    else:
        raise ValueError("Path type not supported: {}".format(path_type))


def lsl(alpha, beta, d, eps=1e-9):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = 2 + (d*d) - (2*c_ab) + (2*d*(sa-sb))
    if p_squared < -eps:
        return None, None, None
    p_squared = utils.clip(p_squared, 0)
    p = math.sqrt(p_squared)
    tmp = math.atan2((cb - ca), d + sa - sb)
    t = (-alpha + tmp) % (2*math.pi)
    q = (beta-tmp) % (2*math.pi)
    return t, p, q


def rsr(alpha, beta, d, eps=1e-9):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp = d - sa + sb
    p_squared = 2 + (d*d) - (2*c_ab) + (2*d*(sb-sa))
    if p_squared < -eps:
        return None, None, None
    p_squared = utils.clip(p_squared, 0)
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca - cb), tmp)
    t = (alpha-tmp2) % (2*math.pi)
    q = (-beta + tmp2) % (2*math.pi)
    return t, p, q


def lsr(alpha, beta, d, eps=1e-9):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d*d) + (2*c_ab) + (2*d*(sa+sb))
    if p_squared < -eps:
        return None, None, None
    p_squared = utils.clip(p_squared, 0)

    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)

    t = (-alpha + tmp2) % (2*math.pi)
    q = (-beta + tmp2) % (2*math.pi)

    return t, p, q


def rsl(alpha, beta, d, eps=1e-9):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = d*d - 2 + 2*c_ab - 2*d*(sa+sb)
    if p_squared < -eps:
        return None, None, None
    p_squared = utils.clip(p_squared, 0)

    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), d - sa - sb) - math.atan2(2.0, p)

    t = (alpha-tmp2) % (2*math.pi)
    q = (beta-tmp2) % (2*math.pi)

    return t, p, q


def solve_csc_path(
        start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeed,
        turn_radius, path_type):

    R = turn_radius

    dx = goal_y - start_y
    dy = goal_x - start_x

    D = math.sqrt(dx*dx + dy*dy)/R

    theta = math.atan2(dy, dx)
    alpha = (0.5*math.pi - start_dir - theta) % (2*math.pi)
    beta = (0.5*math.pi - goal_dir - theta) % (2*math.pi)

    if path_type == "LSL":
        t, p, q = lsl(alpha, beta, D)
    elif path_type == "RSR":
        t, p, q = rsr(alpha, beta, D)
    elif path_type == "LSR":
        t, p, q = lsr(alpha, beta, D)
    elif path_type == "RSL":
        t, p, q = rsl(alpha, beta, D)
    else:
        raise ValueError("Invalid path type: {}".format(path_type))

    if t is None:
        return None

    return path_type, t, p, q


def get_dubins_path_2D(
        start_x,
        start_y,
        start_dir,
        goal_x,
        goal_y,
        goal_dir,
        airspeed,
        turn_radius,
        wind_speed,
        wind_dir,
        path_type,
        start_rounds=0,
        end_rounds=0):

    solution = solve_csc_path(
        start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeed,
        turn_radius, path_type)
    if not solution:
        return None

    path_type, t, p, q = solution

    turn_dir1 = get_turn_dir(path_type, 0)
    turn_dir2 = get_turn_dir(path_type, 2)

    # Segment 1: Turn
    seg1 = CircularTurn(
        x_g_t0=start_x,
        y_g_t0=start_y,
        V_A_xy=airspeed,
        chi_t0=start_dir,
        chi_tf=start_dir + t*turn_dir1,
        radius=turn_radius,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        turn_dir=turn_dir1)

    # State at end of seg1
    x_g_t1, y_g_t1 = seg1.xf["x_g"], seg1.xf["y_g"]
    chi_t1 = seg1.xf["chi"]

    # Segment 2: Straight
    seg2 = ConstSpeedStraightFlight.init_from(
        x_g_t0=x_g_t1,
        y_g_t0=y_g_t1,
        V_A_xy=airspeed,
        chi=chi_t1,
        l_xy_tf=p*turn_radius,
        V_W_xy=wind_speed,
        chi_W=wind_dir)

    # State at end of seg2
    x_g_t2, y_g_t2 = seg2.xf["x_g"], seg2.xf["y_g"]

    # Segment 3: Turn
    seg3 = CircularTurn(
        x_g_t0=x_g_t2,
        y_g_t0=y_g_t2,
        V_A_xy=airspeed,
        chi_t0=chi_t1,
        chi_tf=chi_t1 + q*turn_dir2,
        radius=turn_radius,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        turn_dir=turn_dir2)

    segments = [seg1, seg2, seg3]
    if start_rounds:
        segments.insert(
            0,
            CircularTurn(
                x_g_t0=start_x,
                y_g_t0=start_y,
                V_A_xy=airspeed,
                chi_t0=start_dir,
                chi_tf=start_dir,
                radius=turn_radius,
                V_W_xy=wind_speed,
                chi_W=wind_dir,
                turn_dir=turn_dir1,
                n_full_turns=start_rounds))
    if end_rounds:
        segments.append(
            CircularTurn(
                x_g_t0=goal_x,
                y_g_t0=goal_y,
                V_A_xy=airspeed,
                chi_t0=goal_dir,
                chi_tf=goal_dir,
                radius=turn_radius,
                V_W_xy=wind_speed,
                chi_W=wind_dir,
                turn_dir=turn_dir2,
                n_full_turns=end_rounds))
    return Trajectory(segments)


def get_dubins_path_3D(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_radius,
        wind_speed,
        wind_dir,
        path_type,
        turn_n_max=None,
        vert_speed_constraint=None,
        t_precision=1e-3,
        speed_eps=1e-3):
    """Calculate 3D Dubins paths for uniform wind and with global vertical speed constraint.

    Limitations:
        - Currently considers only CSC horizontal paths (no CCC paths).
        - Only low-altitude and high-altidude cases are considered (no medium-altitude case).
    """
    start_rounds = 0
    end_rounds = 0
    logger.push("dubins_csc3D")
    w_A_g_values = logger.track_values("w_A_g", [0])

    # Upper bound on |w_A_g|: the horizontal (air-relative) airspeed component
    # V_A_xy = sqrt(V_A^2 - w_A_g^2) must exceed the wind speed to make forward
    # progress. Iterates beyond this bound make V_A_xy (and hence the turn
    # radius) non-real, so they must be rejected rather than evaluated.
    w_A_g_abs_max = vt.w_A_g_abs_from_V_A_V_A_xy(airspeed, wind_speed)

    def horizontal_path(w_A_g):
        """Horizontal CSC path for a given vertical speed ``w_A_g``.

        The vertical speed fixes the horizontal airspeed component
        V_A_xy = sqrt(V_A^2 - w_A_g^2) and -- when ``turn_n_max`` is given --
        the achievable turn rate via the load-factor bound. Returns ``None``
        if the horizontal path is infeasible.
        """
        V_A_xy = vt.V_A_xy_from_V_A_w_A_g(airspeed, w_A_g)
        turn_radius_eff = turn_radius
        if turn_n_max is not None:
            # Conservative turn radius feasible in the down-wind direction.
            turn_rate_safe = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                turn_n_max, airspeed,
                vt.gamma_A_from_V_A_w_A_g(airspeed, w_A_g), 0, 9.81)
            chi_dot_safe = km.chi_dot_from_chi_A_dot_V_A_xy_dot(
                V_A_xy, 0, 0, turn_rate_safe, wind_speed, 0)
            turn_radius_eff = (V_A_xy+wind_speed+speed_eps)/chi_dot_safe
        return get_dubins_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir, V_A_xy,
            turn_radius_eff, wind_speed, wind_dir, path_type, start_rounds,
            end_rounds)

    def fixed_point_map(w_A_g):
        """Raw fixed-point map F(w) = dz / t_f(w), or ``None`` if infeasible."""
        horiz_path = horizontal_path(w_A_g)
        if horiz_path is None:
            return None
        return (goal_z-start_z)/horiz_path.duration

    # Fixed-point iteration on w_A_g using Steffensen's method / Aitken
    # acceleration. The raw map F(w) = dz / t_f(w) can form a 2-cycle because
    # the horizontal path geometry reacts to w_A_g through the airspeed and
    # turn-rate coupling
    n_passes = 0
    while not utils.converged(w_A_g_values, abs_tol=t_precision):
        n_passes += 1
        w_A_g = w_A_g_values[-1]

        x1 = fixed_point_map(w_A_g)
        if x1 is None:
            logger.pop(success=False, n_passes=n_passes)
            return None

        # If the first iterate already exceeds the feasible vertical-speed
        # range, add a start or end round.
        if abs(x1) > w_A_g_abs_max:
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue

        x2 = fixed_point_map(x1)
        if x2 is None:
            # x1 is feasible but the map is ill-conditioned. Treat this like
            # the out-of-range case: add a round to shorten the required path
            # time and retry from the current w_A_g.
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue

        denom = x2 - 2.0*x1 + w_A_g
        if abs(denom) > 1e-12 and math.isfinite(denom):
            new_w_A_g = w_A_g - (x1 - w_A_g)**2/denom
        else:
            # Negligible second difference: the raw map is (locally) linear,
            # fall back to the raw iterate.
            new_w_A_g = x2
        if not math.isfinite(new_w_A_g):
            new_w_A_g = x2

        exceeds_airspeed = abs(new_w_A_g) > w_A_g_abs_max
        violates_constraint = (
            vert_speed_constraint is not None
            and not vert_speed_constraint(new_w_A_g))
        if exceeds_airspeed or violates_constraint:
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue
        w_A_g_values.append(new_w_A_g)

    # Re-evaluate the horizontal path at the converged vertical speed.
    horiz_path = horizontal_path(w_A_g_values[-1])
    if horiz_path is None:
        # The converged point is ill-conditioned; fall back to the last
        # well-posed iterate.
        horiz_path = horizontal_path(w_A_g_values[-2])
    if horiz_path is None:
        logger.pop(success=False, n_passes=n_passes)
        return None

    logger.pop(
        success=True,
        iterations=w_A_g_values.n_iterations,
        n_passes=n_passes,
        start_rounds=start_rounds,
        end_rounds=end_rounds)

    vert_profile = ConstVertSpeedProfile(
        z_g_t0=start_z, z_g_tf=goal_z, w_A_g=w_A_g_values[-1], tf=horiz_path.tf)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)
    return lifted_path


def get_dubins_path(
    start_x,
    start_y,
    start_z,
    start_dir,
    goal_x,
    goal_y,
    goal_z,
    goal_dir,
    airspeed,
    turn_radius,
    wind_speed,
    wind_dir,
    path_type,
):
    if start_z is not None:
        assert (goal_z is not None)
        assert (goal_dir is not None)
        return get_dubins_path_3D(
            start_x, start_y, start_z, start_dir, goal_x, goal_y, goal_z,
            goal_dir, airspeed, turn_radius, wind_speed, wind_dir, path_type)
    else:
        return get_dubins_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeed,
            turn_radius, wind_speed, wind_dir, path_type)


def get_dubins_paths(
    start_x,
    start_y,
    start_z,
    start_dir,
    goal_x,
    goal_y,
    goal_z,
    goal_dir,
    airspeed,
    turn_radius,
    wind_speed,
    wind_dir,
    path_types=["LSL", "RSR", "LSR", "RSL"],
):
    result_paths = {}
    for path_type in path_types:
        dubins_path = get_dubins_path(
            start_x, start_y, start_z, start_dir, goal_x, goal_y, goal_z,
            goal_dir, airspeed, turn_radius, wind_speed, wind_dir, path_type)
        if dubins_path:
            result_paths[path_type] = dubins_path
    return result_paths

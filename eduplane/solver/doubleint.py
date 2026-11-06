# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Solver to calculate optimal controls for asymmetrically constrainted
double integrator systems.
"""

from __future__ import division

import math
from dataclasses import dataclass

import numpy

from eduplane.utils.utils import BetterDataClass

# Verbosity levels
ERROR = 1
WARNING = 2
INFO = 3


@dataclass
class FreeFinalTimeProblem(BetterDataClass):
    x0: float
    v0: float
    xg: float
    vg: float
    v_max: float
    v_min: float
    u0_max: float
    u0_min: float
    u2_max: float
    u2_min: float

    @classmethod
    def from_fixed_final_time(cls, fixed_problem):
        return cls(
            x0=fixed_problem.x0,
            v0=fixed_problem.v0,
            xg=fixed_problem.xg,
            vg=fixed_problem.vg,
            v_max=fixed_problem.v_max,
            v_min=fixed_problem.v_min,
            u0_max=fixed_problem.u0_max,
            u0_min=fixed_problem.u0_min,
            u2_max=fixed_problem.u2_max,
            u2_min=fixed_problem.u2_min)


@dataclass
class Solution(BetterDataClass):
    u0: float
    u2: float
    t1: float
    t2: float
    tf: float
    v1: float


@dataclass
class FixedFinalTimeProblem(BetterDataClass):

    x0: float
    v0: float
    xg: float
    vg: float
    tf: float
    v_max: float
    v_min: float
    u0_max: float
    u0_min: float
    u2_max: float
    u2_min: float

    @classmethod
    def from_free_final_time(cls, topt_problem: FreeFinalTimeProblem, tf):
        return cls(
            x0=topt_problem.x0,
            v0=topt_problem.v0,
            xg=topt_problem.xg,
            vg=topt_problem.vg,
            tf=tf,
            v_max=topt_problem.v_max,
            v_min=topt_problem.v_min,
            u0_max=topt_problem.u0_max,
            u0_min=topt_problem.u0_min,
            u2_max=topt_problem.u2_max,
            u2_min=topt_problem.u2_min)


@dataclass
class FreeFinalPositionProblem(BetterDataClass):
    x0: float
    v0: float
    u0: float
    vg: float
    tf: float
    v_max: float
    v_min: float
    u0_max: float
    u0_min: float
    u2_max: float
    u2_min: float


# Time-optimal solution
###############################################################################


# Get control inputs for first and last segment
def doubleint_topt_inputs(problem, verbose=False):
    # Unpack problem
    x0, v0, xg, vg, u0_max, u0_min, u2_max, u2_min = (
        problem.x0, problem.v0, problem.xg, problem.vg, problem.u0_max,
        problem.u0_min, problem.u2_max, problem.u2_min)
    # Decide on inputs for first and last segment.
    # This is done by calculating the x-value of the switching curve
    # at the initial velocity and comparing it to the intial x-value:
    # If left of / below the switching curve, then u_a must be upper limit
    # (u_max) and u_d must be lower limit (u_min) and vice versa.
    u_s = u2_min if vg - v0 < 0 else u2_max
    x_s = xg + (v0**2 - vg**2)/(2*u_s) if abs(u_s) > 0 else xg
    if x0 > x_s:
        u0 = u0_min
        u2 = u2_max
    else:
        u0 = u0_max
        u2 = u2_min

    if verbose >= INFO:
        print("Time-optimal inputs: u0={}, u2={}".format(u0, u2))

    return u0, u2


def doubleint_topt(problem, t_eps=0.001, permissive=True, verbose=False):
    # Unpack problem for brevity
    x0, v0, xg, vg, v_min, v_max, u0_max, u0_min, u2_max, u2_min = (
        problem.x0, problem.v0, problem.xg, problem.vg, problem.v_min,
        problem.v_max, problem.u0_max, problem.u0_min, problem.u2_max,
        problem.u2_min)

    if verbose >= INFO:
        print(f"Solving time-optimal double integrator problem: {problem}")

    solutions = []
    u0_candidates = []
    u2_candidates = []

    # Allow initial/final speed to violate speed constraint?
    if not permissive:
        if v0 < v_min or v0 > v_max:
            if verbose >= WARNING:
                print("Initial velocity violates constraints")
            return solutions
        if vg < v_min or vg > v_max:
            if verbose >= WARNING:
                print("Final velocity violates constraints")
            return solutions

    u0_topt, u2_topt = doubleint_topt_inputs(problem)
    u0_candidates.append(u0_topt)
    u2_candidates.append(u2_topt)
    if verbose >= INFO:
        print(f"Calculated time-optimal inputs u0={u0_topt}, u2={u2_topt}")

    # Also add all max. and all min. accelerations.
    # This captures cases where the signs are not intended as for assumed for
    # optimal input calculation, i.e. if bounds are all negative/positive.
    # This also captures cases where the goal state is not reachable with
    # optimal inputs due to asymmetric constraints.
    u0_candidates.append(u0_min)
    u2_candidates.append(u2_min)
    u0_candidates.append(u0_max)
    u2_candidates.append(u2_max)
    if permissive:
        u0_candidates.append(u0_min if u0_topt == u0_max else u0_max)
        u2_candidates.append(u2_min if u2_topt == u2_max else u2_max)

    for u0, u2 in zip(u0_candidates, u2_candidates):
        if verbose == INFO:
            print(f"\nTrying inputs u0={u0}, u2={u2}")
        v_s_candidates = []
        # Calculate intersection of the phase plane trajectories going through
        # start and goal state with inputs from above.
        # This will be the switching velocity for time-optimal case without
        # limited maneuverability.
        try:
            v_s_plusminus_squared = (2*u2*u0*
                                     (x0-xg) - u2*v0**2 + u0*vg**2)/(u0-u2)
            v_s_plusminus = math.sqrt(v_s_plusminus_squared)
            v_s_candidates.append(v_s_plusminus)
            v_s_candidates.append(-v_s_plusminus)
            if verbose >= INFO:
                print(f"Switching velocity is +/-{v_s_plusminus}")
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass
        # Add initial/final and min/max velocity to list of candidate switching velocities
        if vg < v0:
            v_s_candidates.append(v0)
            v_s_candidates.append(v_min)
            v_s_candidates.append(vg)
        if vg > v0:
            v_s_candidates.append(v0)
            v_s_candidates.append(v_max)
            v_s_candidates.append(vg)

        # # Calculate intersection of the phase plane trajectories going through
        # # start and goal state with inputs from above.
        for v_s in v_s_candidates:
            v_s = max(v_s, v_min)
            v_s = min(v_s, v_max)
            if abs(v_s) < 1e-6:
                continue
            else:
                if verbose >= INFO:
                    print(f"Trying constrained switching velocity {v_s}")

                # Calculate x_values at (possibly saturated) velocity for both
                # parabola legs
                x_w_0 = x0 + (v_s**2 - v0**2)/(2*u0) if abs(u0) > 0 else x0
                x_w_2 = xg + (v_s**2 - vg**2)/(2*u2) if abs(u2) > 0 else xg

                # print(f"x_w_0={x_w_0}, x_w_2={x_w_2}")

                if abs(u0) <= 0:
                    if abs(v_s - v0) > 0:
                        if verbose >= INFO:
                            print(
                                f"Cannot reach switching velocity with u0={u0}")
                        continue
                    else:
                        if verbose >= INFO:
                            print("Zero-length initial segment (t1=0)")
                        t1 = 0
                else:
                    t1 = (v_s-v0)/u0

                t2 = t1 + (x_w_2-x_w_0)/v_s

                if abs(u2) <= 0:
                    if abs(vg - v_s) > 0:
                        if verbose >= INFO:
                            print(
                                f"Cannot reach switching velocity with u2={u2}")
                        continue
                    else:
                        if verbose >= INFO:
                            print("Zero-length final segment (t2=tf)")
                        tf = t2
                else:
                    tf = t2 + (vg-v_s)/u2

                # print(f" t2 = {t1} + ({x_w_2}-{x_w_0})/{v_s} = {t2}")

                # Try to numerically sanitize results
                if abs(t1) < t_eps:
                    t1 = 0
                if abs(t2 - t1) < t_eps:
                    t2 = t1
                if abs(tf - t2) < t_eps:
                    t2 = tf
                if t1 >= 0 and t1 <= t2 and t2 <= tf:
                    if verbose >= INFO:
                        print(
                            "Valid solution with u0={}, u2={}, v_s={}, t1={}, "
                            "t2={}, tf={}".format(u0, u2, v_s, t1, t2, tf))
                    solutions.append(
                        Solution(u0=u0, u2=u2, t1=t1, t2=t2, tf=tf, v1=v_s))
                elif verbose >= INFO:
                    print(
                        "Invalid solution with u0={}, u2={}, v_s={}, t1={}, "
                        "t2={}, tf={}".format(u0, u2, v_s, t1, t2, tf))

    solutions = sorted(solutions, key=lambda s: s.tf)
    if verbose >= INFO:
        print(
            "Found {} time-optimal solutions with final times: {}".format(
                len(solutions), [s.tf for s in solutions]))

    return solutions


# Fixed Final Time Solution
###############################################################################


def fixed_min_acc_u0_unconstrained(R, tf, v0, vg, x0, xg):
    p = 2*v0/tf - 2*vg/(R*tf) - ((xg-x0)*(2*R - 2))/(R*tf**2)
    q = (vg**2 + v0**2 - 2*v0*vg)/(R*tf**2)
    return -0.5*p - numpy.sqrt(0.25*p**2 -
                               q), -0.5*p + numpy.sqrt(0.25*p**2 - q)


def fixed_min_acc_u0_from_vs(R, tf, v0, vg, x0, xg, vs):
    # return ((vg-vs)/R+vs-v0)/(tf-(xg-x0)/vs)
    return (2*vs*vg - vs**2 - vg**2 + R*
            (v0**2 + vs**2 - 2*v0*vs))/(2*R*(vs*tf + x0 - xg))


# TODO: Handle and test ambiguous cases and cases with x0=xg, v0=vg and tf=0
def doubleint_fixed_candidate(
        problem, u_0, u_2, t_eps=0.001, permissive=True, verbose=False):
    # Unpack into local variables for brevity
    x0, v0, xg, vg, tf, v_max, v_min = (
        problem.x0, problem.v0, problem.xg, problem.vg, problem.tf,
        problem.v_max, problem.v_min)

    solutions = []

    if v0 < v_min or v0 > v_max:
        if verbose >= WARNING:
            print("Initial velocity violates constraints")
        if not permissive:
            return solutions
    if vg < v_min or vg > v_max:
        if verbose >= WARNING:
            print("Final velocity violates constraints")
        if not permissive:
            return solutions

    if verbose >= INFO:
        print(
            "Solving fixed final time with inputs "
            "u0={}, u2={}".format(u_0, u_2))
    # Calculate solution candidates (valid t1, v1, t2)
    if abs(u_0 - u_2) < t_eps:
        denom = 2*u_0*u_2*tf - 2*u_0*(vg-v0)
        if denom != 0:
            t1 = (2*u_0*((xg-x0) - v0*tf) - (vg - v0)**2)/denom

            v1_upper = v0 + (t1+t_eps)*u_0
            v1_lower = v0 + (t1-t_eps)*u_0
            v1 = v0 + t1*u_0
            if v1_upper < v_min or v1_lower > v_max:
                if verbose >= INFO:
                    print("Invalid solution with t1={}, v1={}".format(t1, v1))
            else:
                t2 = (v0 - vg + u_0*t1 + u_2*tf)/u_2
                # Try to numerically sanitize results
                if abs(t1) < t_eps:
                    t1 = 0
                if abs(t2 - t1) < t_eps:
                    t2 = t1
                if abs(tf - t2) < t_eps:
                    t2 = tf
                if t1 >= 0 and t1 <= t2 and t2 <= tf:
                    if verbose >= INFO:
                        print(
                            "Valid solution with u0={}, u2={}, v_s={}, "
                            "t1={}, t2={}, tf={}".format(
                                u_0, u_2, v1, t1, t2, tf))
                    solutions.append(Solution(u_0, u_2, t1, t2, tf, v1))
                elif verbose >= INFO:
                    print(
                        "Invalid solution with u0=u2={}, t1={}, t2={}, "
                        "tf={} for problem\n{}".format(
                            u_0, t1, t2, tf, problem))
        elif verbose >= INFO:
            print("Denominator is zero with u0=u2={}".format(u_0))
    else:
        denom = u_0**2 - u_0*u_2
        A = u_0*((vg-v0) - u_2*tf)/denom
        B = (2*u_2*((x0-xg) + v0*tf) + (vg - v0)**2)/denom
        radicand = A**2 - B
        # Take care of rounding errors
        radicand = (0 if abs(radicand) < t_eps**2 else radicand)
        if radicand >= 0:
            for sqrt_sign in [-1, 1]:
                t1 = A + sqrt_sign*math.sqrt(radicand)
                v1_upper = v0 + (t1+t_eps)*u_0
                v1_lower = v0 + (t1-t_eps)*u_0
                v1 = v0 + t1*u_0
                if max(v1_upper, v1_lower) < v_min or min(v1_upper,
                                                          v1_lower) > v_max:
                    if verbose >= INFO:
                        print(
                            "Invalid solution with t1={}, v1={}".format(t1, v1))
                else:
                    t2 = (v0 - vg + u_0*t1 + u_2*tf)/u_2
                    # Try to numerically sanitize results
                    if abs(t1) < t_eps:
                        t1 = 0
                    if abs(t2 - t1) < t_eps:
                        t2 = t1
                    if abs(tf - t2) < t_eps:
                        t2 = tf
                    if t1 >= 0 and t1 <= t2 and t2 <= tf:
                        if verbose >= INFO:
                            print(
                                "Valid solution with u0={}, u2={}, v_s={}, "
                                "t1={}, t2={}, tf={}".format(
                                    u_0, u_2, v1, t1, t2, tf))
                        solutions.append(Solution(u_0, u_2, t1, t2, tf, v1))
                    elif verbose >= INFO:
                        print(
                            "Invalid solution with u0={}, u2={}, t1={}, t2={}, "
                            "tf={}".format(u_0, u_2, t1, t2, tf))
        elif verbose >= WARNING:
            print(
                "Negative radicand (A^2-B ({}) < 0) for "
                "u0={}, u2={}".format(radicand, u_0, u_2))
    return solutions


def doubleint_fixed(problem, permissive=True, verbose=False):
    u0_topt, u2_topt = doubleint_topt_inputs(problem, verbose=verbose)

    u0_candidates = [u0_topt]
    u2_candidates = [u2_topt]
    u0_candidates.append(problem.u0_min)
    u2_candidates.append(problem.u2_min)
    u0_candidates.append(problem.u0_max)
    u2_candidates.append(problem.u2_max)

    solutions = []
    for u0, u2 in zip(u0_candidates, u2_candidates):
        solutions += doubleint_fixed_candidate(
            problem, u0, u2, permissive=permissive, verbose=verbose)

    if verbose >= INFO:
        print(
            "Found {} solutions for fixed final time problem: {}".format(
                len(solutions), problem))

    return solutions


def doubleint_fixed_min_acc(problem, permissive=True, verbose=False):
    u0_topt, u2_topt = doubleint_topt_inputs(problem, verbose=verbose)

    u0_candidates = []
    u2_candidates = []

    u0_u2_ratio = u2_topt/u0_topt

    u0_abs_unconstr_1, u0_abs_unconstr_2 = fixed_min_acc_u0_unconstrained(
        u0_u2_ratio, problem.tf, problem.v0, problem.vg, problem.x0, problem.xg)
    u0_candidates.append(u0_abs_unconstr_1)
    u2_candidates.append(u0_abs_unconstr_1*u0_u2_ratio)
    u0_candidates.append(u0_abs_unconstr_2)
    u2_candidates.append(u0_abs_unconstr_2*u0_u2_ratio)

    try:
        u0_abs_constr_1 = fixed_min_acc_u0_from_vs(
            u0_u2_ratio, problem.tf, problem.v0, problem.vg, problem.x0,
            problem.xg, problem.v_min)
        u0_candidates.append(u0_abs_constr_1)
        u2_candidates.append(u0_abs_constr_1*u0_u2_ratio)
    except ZeroDivisionError:
        pass
    try:
        u0_abs_constr_2 = fixed_min_acc_u0_from_vs(
            u0_u2_ratio, problem.tf, problem.v0, problem.vg, problem.x0,
            problem.xg, problem.v_max)
        u0_candidates.append(u0_abs_constr_2)
        u2_candidates.append(u0_abs_constr_2*u0_u2_ratio)
    except ZeroDivisionError:
        pass

    u0_candidates.append(problem.u0_min)
    u2_candidates.append(problem.u2_min)
    u0_candidates.append(problem.u0_max)
    u2_candidates.append(problem.u2_max)

    solutions = []
    for u0, u2 in zip(u0_candidates, u2_candidates):
        solutions += doubleint_fixed_candidate(
            problem, u0, u2, permissive=permissive, verbose=verbose)

    if verbose >= INFO:
        print(
            "Found {} solutions for fixed final time problem with min. acceleration: {}"
            .format(len(solutions), problem))

    solutions = sorted(solutions, key=lambda s: s.u0**2 + s.u2**2)

    return solutions


# Free Final Position (aka "open-loop") Solution
###############################################################################


def doubleint_free(problem, t_eps=0.001, verbose=False):
    # Unpack into local variables for brevity
    v0, u0, vg, tf, v_max, v_min, u0_max, u0_min, u2_max, u2_min = (
        problem.v0, problem.u0, problem.vg, problem.tf, problem.v_max,
        problem.v_min, problem.u0_max, problem.u0_min, problem.u2_max,
        problem.u2_min)
    assert (u0 == u0_min or u0 == u0_max)
    assert (v0 >= v_min and v0 <= v_max)
    assert (vg >= v_min and vg <= v_max)

    # How much time from v0 to vg?
    dv = vg - v0
    dt_vg = dv/(u0_min if dv < 0 else u0_max)
    dt_free = tf - dt_vg

    # print("dt_vg: {}, dt_free: {}".format(dt_vg, dt_free))

    # "Bang-bang" control inputs are straight-forward
    u2 = u2_max if u0 == u0_min else u2_min

    # Two cases:
    # 1) (vg-v0)/u0 > 0 : Start acceleration towards target velocity
    #    Excess time is used to "overshoot" target velocity and come back to it
    # 2) (vg-v0)/u0 < 0 : Start acceleration in opposite direction of target
    #    velocity. Excess time is used to accelerate and come back to start
    #    velocity
    solutions = []
    if dv/u0 >= 0:
        # Calculate unsatured solution
        t1_unsat = dt_vg + dt_free/(1 + abs(u0/u2))
        v1_unsat = v0 + t1_unsat*u0
        # print("t1_unsat: {}, v1_unsat: {}".format(t1_unsat, v1_unsat))
        # Handle saturation
        v1 = max(v1_unsat, v_min) if u0 == u0_min else min(v1_unsat, v_max)
        t1 = dt_vg + (v1-vg)/u0
        t2 = tf - (vg-v1)/u2
        # Try to numerically sanitize results
        if abs(t1) < t_eps:
            t1 = 0
        if abs(t2 - t1) < t_eps:
            t2 = t1
        if abs(tf - t2) < t_eps:
            t2 = tf
        solutions.append(Solution(u0, u2, t1, t2, tf, v1))
    else:
        # Calculate unsatured solution
        t1_unsat = dt_free/(1 + abs(u0/u2))
        v1_unsat = v0 + t1_unsat*u0
        # print("t1_unsat: {}, v1_unsat: {}".format(t1_unsat, v1_unsat))
        # Handle saturation
        v1 = max(v1_unsat, v_min) if u0 == u0_min else min(v1_unsat, v_max)
        t1 = (v1-v0)/u0
        t2 = tf - (vg-v1)/u2
        # Try to numerically sanitize results
        if abs(t1) < t_eps:
            t1 = 0
        if abs(t2 - t1) < t_eps:
            t2 = t1
        if abs(tf - t2) < t_eps:
            t2 = tf
        solutions.append(Solution(u0, u2, t1, t2, tf, v1))

    if verbose >= INFO:
        print(
            "Found {} solutions for free final position problem\n{}".format(
                len(solutions), problem))

    return solutions

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
This modules provides the class `ExtDubinsAirplane`.
The Extended Dubins Airplane class provides algorithms and convenience methods
to work with the kinematic and flight performance equations and to calculate
parameter values for flight envelope boundaries and various specific flight
states (e.g. max. thrust climbs or min. thrust horizontal turning flight).
"""

import math
import warnings
from dataclasses import dataclass, field
from math import cos, sin, sqrt

import numpy
import pandas
from scipy import integrate, optimize

from eduplane.model import flight_performance_model as fpm
from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.utils import utils


@dataclass
class FPMParameters(object):
    m: float
    S: float
    C_D0: float
    C_L0: float
    k: float
    T_min: float
    T_max: float
    P_max: float
    n_min: float
    n_max: float
    C_L_min: float
    C_L_max: float
    mu_A_max: float
    V_A_max: float
    g: float = 9.81
    rho: float = 1.225
    eps: float = 1e-9
    W: float = field(init=False)

    def __post_init__(self):
        self.W = self.m*self.g


class ExtDubinsAirplane(FPMParameters):

    def max_thrust(self, V_A, T_max=None, P_max=None):
        """Calculate max. thrust at given airspeed respecting both thrust and
        power constraints."""
        if P_max is not None and V_A > 0:
            return min(T_max, P_max/V_A)
        else:
            return T_max

    def max_power(self, V_A, T_max=None, P_max=None):
        """Calculate max. power at given airspeed respecting both thrust and
        power constraints."""
        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max
        if V_A > 0:
            return min(T_max*V_A, P_max)
        else:
            return T_max

    def get_stall_speed(
            self, n=None, gamma_A=0, chi_A_dot=0, gamma_A_dot=0, V_A_eps=1e-3):
        if n is not None:
            return fpm.V_A_from_C_L_n(
                self.C_L_max, n, self.m, self.g, self.S, self.rho)

        if n is None:
            n_func = lambda V_A: fpm.n_from_chi_A_dot_gamma_A_dot(
                V_A, chi_A_dot, gamma_A, gamma_A_dot, self.g)
            V_A = [self.V_A_max]
            while not utils.converged(V_A, abs_tol=V_A_eps):
                n = n_func(V_A[-1])
                V_A.append(
                    fpm.V_A_from_C_L_n(
                        self.C_L_max, n, self.m, self.g, self.S, self.rho))
            return V_A[-1]

    def air_path_inclination_rate(
            self,
            *,
            V_A,
            gamma_A,
            n=None,
            V_A_dot=None,
            w_A_g_dot=None,
            V_A_xy_dot=None,
            mu_A=None,
            chi_A_dot=None):
        """Calculate air-path inclination rate from current airspeed and
        air-path inclination angle plus one of the following parameter sets:

        - vertical acceleration and airspeed rate
        - vertical acceleration and horizontal airspeed rate
        - airspeed rate and horizontal airspeed rate
        - air-path azimuth rate and load factor
        - air-path azimuth rate and roll angle (must be > 0)
        """
        # gamma_A_dot = f(V_A_dot, w_A_g_dot)
        if None not in (V_A_dot, w_A_g_dot):
            # print(f"air_path_inclination_rate(V_A={V_A:.1f}, V_A_dot={V_A_dot:.1f}, gamma_A={gamma_A:.1f}, w_A_g_dot={w_A_g_dot:.1f}")
            return km.gamma_A_dot_from_V_A_dot_w_A_g_dot(
                V_A, V_A_dot, w_A_g_dot, gamma_A)
        # gamma_A_dot = f(V_A_dot, V_A_xy_dot)
        elif None not in (V_A_dot, V_A_xy_dot):
            # print(f"air_path_inclination_rate(V_A={V_A:.1f}, V_A_dot={V_A_dot:.1f}, gamma_A={gamma_A:.1f}, V_A_xy_dot={V_A_xy_dot:.1f}")
            return km.gamma_A_dot_from_V_A_dot_V_A_xy_dot(
                V_A, V_A_dot, V_A_xy_dot, gamma_A)
        # gamma_A_dot = f(V_A_xy_dot, w_A_g_dot)
        elif None not in (V_A_xy_dot, w_A_g_dot):
            # print(f"air_path_inclination_rate(V_A={V_A:.1f}, gamma_A={gamma_A:.1f}, w_A_g_dot={w_A_g_dot:.1f}, V_A_xy_dot={V_A_xy_dot:.1f}")
            return km.gamma_A_dot_from_V_A_xy_dot_w_A_g_dot(
                V_A, V_A_xy_dot, w_A_g_dot, gamma_A)
        # gamma_A_dot = f(n, chi_A_dot)
        elif None not in (n, chi_A_dot):
            # print(f"air_path_inclination_rate(n={n:.1f}, V_A={V_A:.1f}, chi_A_dot={chi_A_dot:.1f}, gamma_A={gamma_A:.1f}")
            # radicand = (self.g*n)**2 - (chi_A_dot*V_A*cos(gamma_A))**2
            return fpm.gamma_A_dot_from_n_chi_A_dot(
                n, V_A, chi_A_dot, gamma_A, self.g)
        # gamma_A_dot = f(chi_A_dot, mu_A)
        elif None not in (chi_A_dot, mu_A) and abs(mu_A) > 0:
            # print(f"air_path_inclination_rate(V_A={V_A:.1f}, chi_A_dot={chi_A_dot:.1f}, gamma_A={gamma_A:.1f}, mu_A={mu_A:.1f}")
            return km.gamma_A_dot_from_mu_A(
                V_A, chi_A_dot, gamma_A, mu_A, self.g)
        else:
            raise ValueError(
                "Failed to calculate air-path inclination rate "
                "from given arguments.")

    def roll_angle(self, *, V_A, gamma_A, n, gamma_A_dot=None, chi_A_dot=None):
        """Calculate roll angle (mu_A) from airspeed, air-path inclination
        angle and either air-path inclination or azimuth angle rate."""
        if gamma_A_dot is not None:
            return fpm.mu_A_from_n_gamma_A_dot(
                V_A, gamma_A, gamma_A_dot, n, self.g)
        elif chi_A_dot is not None:
            return fpm.mu_A_from_n_chi_A_dot(V_A, gamma_A, chi_A_dot, n, self.g)
        else:
            raise ValueError(
                "Failed to calculate roll angle from given arguments.")

    def vert_acc(
            self,
            *,
            V_A,
            gamma_A=None,
            w_A_g=None,
            n=None,
            chi_A_dot=None,
            gamma_A_dot=None,
            V_A_dot=None):
        """Calculate vertical acceleration (w_A_g_dot) from airspeed, air-path
        inclination angle or vertical speed and either one of these parameter
        sets:
        - air-path azimuth rate and load factor
        - air-path inclination angle rate and airspeed rate
        """
        if gamma_A is None and w_A_g is not None:
            gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        if gamma_A is None:
            raise ValueError(
                "Failed to calculate vert. acceleration from given arguments.")
        if gamma_A_dot is None and None not in (chi_A_dot, n):
            gamma_A_dot = self.air_path_inclination_rate(
                V_A=V_A,
                gamma_A=gamma_A,
                n=n,
                V_A_dot=V_A_dot,
                chi_A_dot=chi_A_dot)
        if None not in (V_A_dot, gamma_A_dot):
            # print(
            #     f"vert_acc(V_A={V_A:.1f}, V_A_dot={V_A_dot:.1f}, "
            #     f"gamma_A={gamma_A:.1f}, gamma_A_dot={gamma_A_dot:.1f}")
            w_A_g_dot = km.w_A_g_dot_from_gamma_A_dot(
                V_A, V_A_dot, gamma_A, gamma_A_dot)
        else:
            raise ValueError(
                "Failed to calculate vert. acceleration from given arguments.")
        return w_A_g_dot

    def max_vert_acc(self, *, V_A, w_A_g_t0, w_A_g_tf, chi_A_dot=0, V_A_dot=0):
        """Calculate maximum feasible constant vertical acceleration between
        two reference steady flight states."""
        gamma_A_t0 = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g_t0)
        gamma_A_tf = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g_tf)

        T_max = self.T_max
        n_push_t0 = fpm.n_push_from_T_w_A_g(
            T_max, V_A, V_A_dot, w_A_g_t0, self.C_D0, self.C_L0, self.m, self.g,
            self.k, self.S, self.rho)
        n_push_tf = fpm.n_push_from_T_w_A_g(
            T_max, V_A, V_A_dot, w_A_g_tf, self.C_D0, self.C_L0, self.m, self.g,
            self.k, self.S, self.rho)
        gamma_A_dot_t0 = fpm.gamma_A_dot_from_n_chi_A_dot(
            n_push_t0, V_A, chi_A_dot, gamma_A_t0, self.g)
        gamma_A_dot_tf = fpm.gamma_A_dot_from_n_chi_A_dot(
            n_push_tf, V_A, chi_A_dot, gamma_A_tf, self.g)
        w_A_g_dot_t0 = km.w_A_g_dot_from_gamma_A_dot(
            V_A, V_A_dot, gamma_A_t0, gamma_A_dot_t0)
        w_A_g_dot_tf = km.w_A_g_dot_from_gamma_A_dot(
            V_A, V_A_dot, gamma_A_tf, gamma_A_dot_tf)
        # print(
        #     f"n_push_t0={n_push_t0}, gamma_A_dot_t0={gamma_A_dot_t0}, w_A_g_dot_t0={w_A_g_dot_t0}"
        # )
        # print(
        #     f"n_push_tf={n_push_tf}, gamma_A_dot_tf={gamma_A_dot_tf}, w_A_g_dot_tf={w_A_g_dot_tf}"
        # )
        return min(w_A_g_dot_t0, w_A_g_dot_tf)

    def turn_rate(
            self,
            *,
            V_A,
            gamma_A,
            mu_A=None,
            n=None,
            gamma_A_dot=None,
            w_A_g_dot=None,
            V_A_dot=None):
        """Calculate turn rate from airspeed, air-path inclination angle and
        rate and either roll angle (mu_A) or load factor.

        If not given, air-path inclination angle rate may be calculated from
        vertical acceleration, see method `air_path_inclination_rate`."""
        if gamma_A_dot is None:
            gamma_A_dot = self.air_path_inclination_rate(
                V_A=V_A,
                gamma_A=gamma_A,
                V_A_dot=V_A_dot,
                w_A_g_dot=w_A_g_dot,
                mu_A=mu_A)
        # f(mu_A, gamma_A_dot)
        if None not in (gamma_A_dot, mu_A):
            # print(f"turn_rate(V_A={V_A:.1f}, gamma_A={gamma_A:.1f}, gamma_A_dot={gamma_A_dot:.1f}, mu_A={mu_A:.1f}) = {km.chi_A_dot_from_mu_A(V_A, gamma_A, gamma_A_dot, mu_A, self.g):.1f}")
            return km.chi_A_dot_from_mu_A(
                V_A, gamma_A, gamma_A_dot, mu_A, self.g)
        # f(n, gamma_A_dot)
        elif None not in (gamma_A_dot, n):
            mu_A = self.roll_angle(
                V_A=V_A, gamma_A=gamma_A, n=n, gamma_A_dot=gamma_A_dot)
            # print(f"turn_rate(V_A={V_A:.1f}, gamma_A={gamma_A:.1f}, gamma_A_dot={gamma_A_dot:.1f}, mu_A={mu_A:.1f}) = {km.chi_A_dot_from_mu_A(V_A, gamma_A, gamma_A_dot, mu_A, self.g):.1f}")
            return km.chi_A_dot_from_mu_A(
                V_A, gamma_A, gamma_A_dot, mu_A, self.g)
        else:
            raise ValueError(
                "Failed to calculate turn rate from given arguments.")

    def load_factor(
            self,
            *,
            V_A,
            gamma_A=None,
            w_A_g=None,
            mu_A=None,
            chi_A_dot=None,
            gamma_A_dot=None,
            w_A_g_dot=None,
            T=None,
            V_A_dot=0):
        """Calculate load factor from airspeed and air-path inclination angle
        or vertical speed.

        Depending on input parameters, one of two strategies is used to
        calculate load factor value(s):

        1) If a thrust value is provided, the corresponding push/pull load factor
        values are calculated and returned as a tuple (n_push, n_pull).

        2) If no thrust value is provided, the load factor is calculated
        from air-path azimuth and inclination angle rates. Values for these
        angular rates may be calculated from other input parameters using
        methods `turn_rate` and `air_path_inclination_rate`.

        Note that for both strategies, the airspeed rate is taken into account
        which defaults to zero.
        """
        if gamma_A is None and w_A_g is not None:
            gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        if gamma_A is None:
            raise ValueError(
                "Failed to calculate load factor from given arguments.")

        # n = f(gamma_A_dot, chi_A_dot)
        if T is None:
            if chi_A_dot is None:
                chi_A_dot = self.turn_rate(
                    V_A=V_A,
                    gamma_A=gamma_A,
                    mu_A=mu_A,
                    gamma_A_dot=gamma_A_dot,
                    w_A_g_dot=w_A_g_dot,
                    V_A_dot=V_A_dot)
            if gamma_A_dot is None and w_A_g_dot is not None:
                gamma_A_dot = self.air_path_inclination_rate(
                    V_A=V_A,
                    gamma_A=gamma_A,
                    V_A_dot=V_A_dot,
                    w_A_g_dot=w_A_g_dot,
                    mu_A=mu_A,
                    chi_A_dot=chi_A_dot)
            # If n is close to zero, rounding errors may lead to a negative
            # radicand. The resulting error is caught and handled here.
            try:
                n = fpm.n_from_chi_A_dot_gamma_A_dot(
                    V_A, chi_A_dot, gamma_A, gamma_A_dot, self.g)
            except FloatingPointError as err:
                radicand = (
                    self.g*
                    cos(gamma_A))**2 + 2*self.g*gamma_A_dot*V_A*cos(gamma_A) + (
                        gamma_A_dot*V_A)**2 + (chi_A_dot*V_A*cos(gamma_A))**2
                if abs(radicand) < 1e-9:
                    n = 0
                else:
                    raise err
            return n
        # n = f(T, V_A, V_A_dot, gamma_A)
        else:
            # FOR DEBUGGING PURPOSES:
            # print(f"n(T={T} V_A={V_A}, V_A_dot={V_A_dot} gamma_A={gamma_A})")
            # print(
            #     f"n(T={T:.1f} V_A={V_A:.1f}, V_A_dot={V_A_dot:.1f} gamma_A={gamma_A:.1f})"
            # )
            # radicand = 2*T - 2*self.m*self.g*numpy.sin(
            #     gamma_A) - 2*V_A_dot*self.m - self.rho*V_A**2*self.S*self.C_D0
            # denom = 2*self.m*self.g*numpy.sqrt(self.k)
            # result = V_A/denom*(
            #     self.rho*V_A*self.S*numpy.sqrt(self.k)*self.C_L0 +
            #     numpy.sqrt(self.rho*self.S)*numpy.sqrt(radicand))
            # print(f"radicand={radicand} denom={denom}")

            try:
                n_max_from_T = fpm.n_pull_from_T_gamma_A(
                    T, V_A, V_A_dot, gamma_A, self.C_D0, self.C_L0, self.m,
                    self.g, self.k, self.S, self.rho)
            except FloatingPointError:
                n_max_from_T = None
            try:
                n_min_from_T = fpm.n_push_from_T_gamma_A(
                    T, V_A, V_A_dot, gamma_A, self.C_D0, self.C_L0, self.m,
                    self.g, self.k, self.S, self.rho)
            except FloatingPointError:
                n_min_from_T = None
            # print(
            #     f"n(T={T:.1f} V_A={V_A:.1f}, V_A_dot={V_A_dot:.1f} gamma_A={gamma_A:.1f}) = [{n_min_from_T:.1f}, {n_max_from_T:.1f}]"
            # )
            return n_min_from_T, n_max_from_T

    def thrust_required(
            self,
            *,
            V_A,
            gamma_A=None,
            w_A_g=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            gamma_A_dot=None,
            w_A_g_dot=None):
        """Calculate thrust required for a given flight state.

        Airspeed and either air-path inclination rate or vertical acceleration
        must be provided.

        Depending on input parameters, one of two strategies is used to
        calculate thrust:

        - If a load factor values is given, thrust is calculated directly from
          the thrust equation.
        - If no load factor value is given, thrust is calculated from air-path
          azimuth rate and air-path inclination angle rate or vertical
          acceleration. Note that if neither gamma_A_dot nor w_A_g_dot is
          given, vertical acceleration is assumed to be zero.
        """
        if n is None:
            if gamma_A is None:
                gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
            if gamma_A_dot is None:
                # If neither gamma_A_dot nor w_A_g_dot is given,
                # we'll assume steady flight, i.e. gamma_A_dot=0
                if w_A_g_dot is None:
                    gamma_A_dot = 0
                else:
                    gamma_A_dot = km.gamma_A_dot_from_V_A_dot_w_A_g_dot(
                        V_A, V_A_dot, w_A_g_dot, gamma_A)
            n = self.load_factor(
                V_A=V_A,
                V_A_dot=V_A_dot,
                gamma_A=gamma_A,
                chi_A_dot=chi_A_dot,
                gamma_A_dot=gamma_A_dot)
        # T = f(n, V_A, V_A_dot, w_A_g)
        if w_A_g is not None:
            return fpm.T_from_n_w_A_g(
                n, V_A, V_A_dot, w_A_g, self.C_D0, self.C_L0, self.m, self.g,
                self.k, self.S, self.rho)
        # T = f(n, V_A, V_A_dot, gamma_A)
        elif gamma_A is not None:
            return fpm.T_from_n_gamma_A(
                n, V_A, V_A_dot, gamma_A, self.C_D0, self.C_L0, self.m, self.g,
                self.k, self.S, self.rho)
        else:
            raise ValueError("Failed to calculate thrust from given arguments.")

    def excess_thrust(
            self,
            V_A,
            gamma_A=None,
            w_A_g=None,
            T_avail=None,
            n=None,
            chi_A_dot=0,
            V_A_dot=0,
            T_eps=1e-3):
        """
        """
        if n is None:
            if gamma_A is None:
                # if w_A_g is None:
                #     w_A_g = self.vert_speed(
                #         V_A=V_A,
                #         T=self.max_thrust(V_A, T_max, P_max),
                #         n=n,
                #         gamma_A_dot=0,
                #         chi_A_dot=chi_A_dot)
                gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)

            n = fpm.n_s_from_V_A_gamma_A(V_A, gamma_A, chi_A_dot, self.g)

        return T_avail - self.thrust_required(
            V_A=V_A, n=n, gamma_A=gamma_A, w_A_g=w_A_g, V_A_dot=V_A_dot)

    def power_required(
            self,
            *,
            V_A,
            T_req=None,
            gamma_A=None,
            w_A_g=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            gamma_A_dot=None,
            w_A_g_dot=None):
        """Calculate power required for a given flight state.

        See method `thrust` for required input parameters.
        """
        if T_req is not None:
            return T_req*V_A
        else:
            return V_A*self.thrust_required(
                V_A=V_A,
                gamma_A=gamma_A,
                w_A_g=w_A_g,
                n=n,
                V_A_dot=V_A_dot,
                chi_A_dot=chi_A_dot,
                gamma_A_dot=gamma_A_dot,
                w_A_g_dot=w_A_g_dot)

    def drag(self, V_A, n):
        """Calculate drag force from airspeed and load factor."""
        return fpm.D_from_n(
            n, V_A, self.C_D0, self.C_L0, self.m, self.g, self.k, self.S,
            self.rho)

    def lift(self, n):
        """Calculate lift force from load factor."""
        return n*self.m*self.g

    def lift_coefficient(
            self,
            *,
            V_A,
            n=None,
            w_A_g=None,
            gamma_A=None,
            gamma_A_dot=None,
            w_A_g_dot=None,
            V_A_dot=0,
            chi_A_dot=0):
        """Calculate lift coefficient from

        - airspeed and load factor
        - airspeed, air-path inclination angle, air-path azimuth rate and
          air-path inclination angle rate or vertical acceleration.
        """
        if n is not None:
            return fpm.C_L_from_n(V_A, n, self.m, self.g, self.S, self.rho)
        if gamma_A is None and w_A_g is not None:
            gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        if gamma_A_dot is None:
            if w_A_g_dot is not None:
                gamma_A_dot = self.air_path_inclination_rate(
                    V_A=V_A,
                    gamma_A=gamma_A,
                    V_A_dot=V_A_dot,
                    w_A_g_dot=w_A_g_dot)
            else:
                gamma_A_dot = 0
        return fpm.C_L_from_chi_A_dot_gamma_A_dot(
            V_A, chi_A_dot, gamma_A, gamma_A_dot, self.m, self.g, self.S,
            self.rho)

    def airspeed_bounds(
            self,
            *,
            n=None,
            gamma_A=None,
            w_A_g=None,
            chi_A_dot=0,
            V_A_dot=0,
            T_min=None,
            T_max=None,
            P_max=None,
            V_A_eps=1e-1,
            V_A_ref=None):
        """Calculate (V_A_min, V_A_max) from thrust, power and lift-coefficient constraints.

        By default returns global bounds; with ``V_A_ref`` set, returns bounds
        *reachable* from that reference airspeed (or from ``"min"``/``"max"``),
        using ``optimize.newton``.

        NOTE: The returned ``V_A_max`` is the same physical quantity as
        ``airspeed_max()`` (max steady airspeed at max thrust/power), but
        computed with a different solver — expect possible small disagreement.
        See ``airspeed_min()``/``airspeed_max()`` for the single-bound, min-/max
        -thrust variants.
        """

        # Get default values for thrust and power constraint, if none specified
        if T_min is None:
            T_min = self.T_min
        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max

        # Calculate min. airspeed constraint based on max. lift coefficient
        # =====================================================================
        if n is None:
            if gamma_A is not None:
                V_A_min_from_C_L = fpm.V_A_from_C_L_gamma_A(
                    self.C_L_max, gamma_A, chi_A_dot, self.m, self.g, self.S,
                    self.rho)
                # print(f"V_A_min at gamma_A={gamma_A}")
                # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")
            elif w_A_g is not None:
                V_A_min_from_C_L = [self.V_A_max]
                # print(f"### Iterating for w_A_g={w_A_g}")
                while not utils.converged(V_A_min_from_C_L, 1e-6):
                    # print(f"V_A_min_from_C_L={V_A_min_from_C_L[-1]}")
                    gamma_A_iterated = vt.gamma_A_from_V_A_w_A_g(
                        V_A_min_from_C_L[-1], w_A_g)
                    # print(f"gamma_A_iterated={gamma_A_iterated}")
                    V_A_min_from_C_L.append(
                        max(
                            abs(w_A_g),
                            fpm.V_A_from_C_L_gamma_A(
                                self.C_L_max, gamma_A_iterated, chi_A_dot,
                                self.m, self.g, self.S, self.rho)))
                V_A_min_from_C_L = V_A_min_from_C_L[-1]
                # print(f"V_A_min_from_C_L for w_A_g={w_A_g} and gamma_A_iterated={gamma_A_iterated}")
                # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")
            else:
                raise ValueError(
                    "If no load factor is given, steady flight is assumed and "
                    "either gamma_A or w_A_g must be supplied.")
        else:
            if abs(n) <= 0:
                V_A_min_from_C_L = 0
            elif n > 0:
                V_A_min_from_C_L = fpm.V_A_from_C_L_n(
                    self.C_L_max, n, self.m, self.g, self.S, self.rho)
            else:
                V_A_min_from_C_L = fpm.V_A_from_C_L_n(
                    self.C_L_min, n, self.m, self.g, self.S, self.rho)
        # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")

        # If neither w_A_g nor gamma_A are given, thrust or power constraints
        # cannot and are not considered, instead the min. airspeed due to
        # max. lift coefficient is returned.
        if gamma_A is None and w_A_g is None:
            return V_A_min_from_C_L, self.V_A_max

        # Find strategies calculating n (and gamma_A) iteratively if required
        if n is None:
            if gamma_A is not None and w_A_g is None:
                gamma_A_func = lambda _: gamma_A
                n_func = lambda new_V_A: fpm.n_s_from_V_A_gamma_A(
                    new_V_A, gamma_A, chi_A_dot, self.g)
            elif w_A_g is not None and gamma_A is None:
                gamma_A_func = lambda V_A: vt.gamma_A_from_V_A_w_A_g(
                    numpy.maximum(
                        V_A_eps, numpy.maximum(numpy.abs(w_A_g), V_A)), w_A_g)
                n_func = lambda new_V_A: fpm.n_s_from_V_A_gamma_A(
                    new_V_A, gamma_A_func(new_V_A), chi_A_dot, self.g)
        else:
            n_func = lambda _: n

        # Decide which equation to use for calculating airspeed bounds
        if gamma_A is not None:
            T_func = fpm.T_from_n_gamma_A
            gamma_A_or_w_A_g = gamma_A
        elif w_A_g is not None:
            T_func = fpm.T_from_n_w_A_g
            gamma_A_or_w_A_g = w_A_g
        else:
            raise ValueError("Either gamma_A or w_A_g must be supplied.")

        # def debug_plot():
        #     from eduplane.utils import plotting
        #     V_A_range = numpy.linspace(0.1, 60, 100)
        #     T_range = T_func(
        #         n_func(V_A_range), V_A_range, V_A_dot, gamma_A_or_w_A_g,
        #         self.C_D0, self.C_L0, self.m, self.g, self.k, self.S, self.rho)
        #     P_range = T_range*V_A_range
        #     ax = pandas.DataFrame(dict(V_A=V_A_range, T=T_range,
        #                                P=P_range)).plot(
        #                                    x='V_A', y=['T', 'P'])
        #     ax.axhline(T_min, color='r', linestyle='--')
        #     ax.axhline(T_max, color='r', linestyle='--')
        #     ax.axhline(P_max, color='b', linestyle='--')
        #     plotting.show()

        # Calculate global min. and max. airspeeds based on max. thrust constraint
        try:
            V_A_min_T_max = optimize.newton(
                lambda V_A: T_func(
                    n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
                    .C_L0, self.m, self.g, self.k, self.S, self.rho) - T_max,
                V_A_eps)
        except RuntimeError:
            # print("airspeed_bounds: V_A_min_T_max failed")
            # debug_plot()
            V_A_min_T_max = numpy.nan
        try:
            V_A_max_T_max = optimize.newton(
                lambda V_A: T_func(
                    n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
                    .C_L0, self.m, self.g, self.k, self.S, self.rho) - T_max,
                self.V_A_max)
        except RuntimeError:
            # print("airspeed_bounds: V_A_max_T_max failed")
            # debug_plot()
            V_A_max_T_max = numpy.nan

        # Calculate global min. and max. airspeeds based on max. power constraint
        try:
            V_A_min_P_max = optimize.newton(
                lambda V_A: T_func(
                    n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
                    .C_L0, self.m, self.g, self.k, self.S, self.rho)*V_A -
                P_max, V_A_eps)
        except RuntimeError:
            V_A_min_P_max = numpy.nan
        try:
            V_A_max_P_max = optimize.newton(
                lambda V_A: T_func(
                    n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
                    .C_L0, self.m, self.g, self.k, self.S, self.rho)*V_A -
                P_max, self.V_A_max)
        except RuntimeError:
            V_A_max_P_max = numpy.nan

        # Only take V_A_min values into account if they are not NaN and actually differ from V_A_max values. If min.
        # and max. values are equal, the T(V_A) curve is monotonically increasing for V_A > 0 (i.e. there is no
        # negative arc), hence both root-finding procedures above yield the same result.
        # V_A_min values equal to their corresponding V_A_max values must not be considered to avoid situations were
        # V_A_min > V_A_max.
        V_A_min = V_A_min_from_C_L
        if abs(V_A_min_T_max - V_A_max_T_max) > V_A_eps:
            V_A_min = max(V_A_min, V_A_min_T_max)
        if abs(V_A_min_P_max - V_A_max_P_max) > V_A_eps:
            V_A_min = max(V_A_min, V_A_min_P_max)
        V_A_max = min(V_A_max_T_max, V_A_max_P_max)
        if V_A_min > V_A_max:
            raise ValueError(
                f"Infeasible airspeed bounds: {V_A_min=} {V_A_max=} {V_A_min_from_C_L=} {V_A_min_T_max=} {V_A_min_P_max=} {V_A_max_T_max=} {V_A_max_P_max=}"
            )

        if V_A_ref is None:
            # Return global airspeed bounds
            return V_A_min, V_A_max
        else:
            # Assess local airspeed bounds around reference airspeed, i.e. reachability from given airspeed.

            # If special values "min" or "max" are for V_A_ref argument, replace with corresponding numerical value
            if V_A_ref == "max":
                V_A_ref = V_A_max
            elif V_A_ref == "min":
                V_A_ref = V_A_min

            # Case 0: Reference airspeed is outside global constraints
            if V_A_ref < V_A_min or V_A_ref > V_A_max:
                # # DEBUG PLOT:
                # from eduplane.utils import plotting
                # V_A_range = numpy.linspace(0.1, 60, 100)
                # T_range = T_func(
                #     n_func(V_A_range), V_A_range, V_A_dot, gamma_A_or_w_A_g,
                #     self.C_D0, self.C_L0, self.m, self.g, self.k, self.S,
                #     self.rho)
                # P_range = T_range*V_A_range
                # ax = pandas.DataFrame(
                #     dict(V_A=V_A_range, T=T_range, P=P_range)).plot(
                #         x='V_A', y=['T', 'P'])
                # ax.axhline(T_min, color='r', linestyle='--')
                # ax.axhline(T_max, color='r', linestyle='--')
                # ax.axhline(P_max, color='r', linestyle='--')
                # plotting.show()
                raise ValueError(
                    f"Reference airspeed {V_A_ref=} is outside feasible range {V_A_min=} {V_A_max=}"
                )

            # Calculate "local" min. and max. airspeeds based on min. thrust constraint.
            # Note that both values can turn out to be NaN, if the min. thrust constraint does not intersect the
            # V_A(T) curve.

            # NOTE: Cannot simply apply newton method as it's unclear wether a root exists, i.e. if the T(V_A) curve
            #       hits the T_min constraint
            # V_A_min_T_min = optimize.newton(
            #     lambda V_A: T_func(
            #         n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
            #         .C_L0, self.m, self.g, self.k, self.S, self.rho) - T_min,
            #     V_A_eps)
            # V_A_max_T_min = optimize.newton(
            #     lambda V_A: T_func(
            #         n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0, self
            #         .C_L0, self.m, self.g, self.k, self.S, self.rho) - T_min,
            #     self.V_A_max)
            V_A_min_T_min_result = optimize.minimize(
                lambda V_A: (
                    T_func(
                        n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0,
                        self.C_L0, self.m, self.g, self.k, self.S, self.rho) -
                    T_min)**2, V_A_eps)
            if V_A_min_T_min_result["fun"] < self.eps:
                V_A_min_T_min = V_A_min_T_min_result.x.item()
            else:
                V_A_min_T_min = numpy.nan

            V_A_max_T_min_result = optimize.minimize(
                lambda V_A: (
                    T_func(
                        n_func(V_A), V_A, V_A_dot, gamma_A_or_w_A_g, self.C_D0,
                        self.C_L0, self.m, self.g, self.k, self.S, self.rho) -
                    T_min)**2, self.V_A_max)
            if V_A_max_T_min_result["fun"] < self.eps:
                V_A_max_T_min = V_A_max_T_min_result.x.item()
            else:
                V_A_max_T_min = numpy.nan

            # Case 1: If the T(V_A) curve does not hit the T_min constraint, the global airspeed bounds are effective
            if numpy.isnan(V_A_min_T_min) and numpy.isnan(V_A_max_T_min):
                return V_A_min, V_A_max
            # Case 2: Reference airspeed is within left/lower valid airspeed range
            if V_A_ref <= V_A_min_T_min:
                return V_A_min, V_A_min_T_min
            # Case 3: Reference airspeed is within right/upper valid airspeed range
            if V_A_ref >= V_A_max_T_min:
                return V_A_max_T_min, V_A_max
            # Case 4: Reference airspeed is in between local/inner constraints
            # Note that this is the only option left since we checked against global min/max constraints above.
            raise ValueError(
                f"Reference airspeed {V_A_ref=} is outside feasible range, between local/inner constraints {V_A_min_T_min=} {V_A_max_T_min=}"
            )

    def airspeed_min(
            self,
            *,
            n=None,
            gamma_A=None,
            w_A_g=None,
            chi_A_dot=0,
            V_A_dot=0,
            T_min=None,
            T_max=None,
            P_max=None):
        """Calculate the minimum feasible airspeed.

        Combines stall-speed (lift-coefficient) and thrust/power constraints.
        Unique among the airspeed helpers: accounts for the *minimum-thrust*
        boundary (positive thrust-over-airspeed arc) plus slow-flight max
        thrust/power (negative arc).

        Prefer over ``airspeed_bounds`` when only V_A_min is needed; see
        ``airspeed_max()`` for the single-bound max-thrust/power counterpart.
        """
        # Get default values for thrust and power constraint, if none specified
        if T_min is None:
            T_min = self.T_min
        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max

        # Calculate min. airspeed constraint based on max. lift coefficient
        # =====================================================================
        if n is None:
            if gamma_A is not None:
                V_A_min_from_C_L = fpm.V_A_from_C_L_gamma_A(
                    self.C_L_max, gamma_A, chi_A_dot, self.m, self.g, self.S,
                    self.rho)
                # print(f"V_A_min at gamma_A={gamma_A}")
                # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")
            elif w_A_g is not None:
                V_A_min_from_C_L = [self.V_A_max]
                # print(f"### Iterating for w_A_g={w_A_g}")
                while not utils.converged(V_A_min_from_C_L, 1e-6):
                    # print(f"V_A_min_from_C_L={V_A_min_from_C_L[-1]}")
                    gamma_A_iterated = vt.gamma_A_from_V_A_w_A_g(
                        V_A_min_from_C_L[-1], w_A_g)
                    # print(f"gamma_A_iterated={gamma_A_iterated}")
                    V_A_min_from_C_L.append(
                        max(
                            abs(w_A_g),
                            fpm.V_A_from_C_L_gamma_A(
                                self.C_L_max, gamma_A_iterated, chi_A_dot,
                                self.m, self.g, self.S, self.rho)))
                V_A_min_from_C_L = V_A_min_from_C_L[-1]
                # print(f"V_A_min_from_C_L for w_A_g={w_A_g} and gamma_A_iterated={gamma_A_iterated}")
                # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")
            else:
                raise ValueError(
                    "If no load factor is given, steady flight is assumed and "
                    "either gamma_A or w_A_g must be supplied.")
        else:
            if abs(n) <= 0:
                V_A_min_from_C_L = 0
            elif n > 0:
                V_A_min_from_C_L = fpm.V_A_from_C_L_n(
                    self.C_L_max, n, self.m, self.g, self.S, self.rho)
            else:
                V_A_min_from_C_L = fpm.V_A_from_C_L_n(
                    self.C_L_min, n, self.m, self.g, self.S, self.rho)
        # print(f"V_A_min_from_C_L={V_A_min_from_C_L}")

        # # DEBUGGING: Plot T_over_V_A
        # # =====================================================================
        # from eduplane.utils import plotting
        # V_A_range = numpy.linspace(2, self.V_A_max, 100)
        # n_over_V_A = [utils.error2constant(n_func)(V_A) for V_A in V_A_range]
        # gamma_A_over_V_A = [gamma_A_func(V_A) for V_A in V_A_range]
        # T_over_V_A = [
        #     utils.error2constant(fpm.T_from_n_gamma_A)(
        #         n_value, V_A, V_A_dot, gamma_A_value, self.C_D0, self.C_L0,
        #         self.m, self.g, self.k, self.S, self.rho)
        #     for V_A, n_value, gamma_A_value in zip(
        #         V_A_range, n_over_V_A, gamma_A_over_V_A)
        # ]
        # df = pandas.DataFrame(
        #     {
        #         "V_A": V_A_range,
        #         "T_over_V_A": T_over_V_A,
        #         "n_over_V_A": n_over_V_A,
        #         "gamma_A_over_V_A": gamma_A_over_V_A,
        #     })
        # df.plot(x="V_A", y=["T_over_V_A", "n_over_V_A", "gamma_A_over_V_A"])
        # plotting.show()
        # # ---------------------------------------------------------------------

        # Calculate min. airspeed constraint based on minimum thrust.
        # =====================================================================
        # Note that the given gamma_A/w_A_g may not be feasible at min. thrust,
        # so we have to catch this case when trying to calculate min. airspeed
        # constrained by thrust.

        # Find strategies calculating gamma_A and n iteratively if required
        n_func = lambda _: n
        if gamma_A is not None and w_A_g is None:
            gamma_A_func = lambda _: gamma_A
            if n is None:
                n_func = lambda new_V_A: fpm.n_s_from_V_A_gamma_A(
                    new_V_A, gamma_A, chi_A_dot, self.g)
        elif w_A_g is not None and gamma_A is None:
            gamma_A_func = lambda V_A: vt.gamma_A_from_V_A_w_A_g(
                max(abs(w_A_g), V_A), w_A_g)
            if n is None:
                n_func = lambda new_V_A: fpm.n_s_from_V_A_gamma_A(
                    new_V_A, gamma_A_func(new_V_A), chi_A_dot, self.g)
        else:
            gamma_A_func = None

        V_A_min_from_T_min = 0
        if gamma_A_func is not None:
            try:
                V_A_min_from_T_min = [V_A_min_from_C_L]
                while not utils.converged(V_A_min_from_T_min, abs_tol=1e-3):
                    gamma_A_value = gamma_A_func(V_A_min_from_T_min[-1])
                    n_value = n_func(V_A_min_from_T_min[-1])
                    V_A_from_T_min = fpm.V_A_from_T_gamma_A(
                        T_min, n_value, gamma_A_value, V_A_dot, self.C_D0,
                        self.C_L0, self.m, self.g, self.k, self.S, self.rho)
                    V_A_min_from_T_min.append(V_A_from_T_min)
                    # print(
                    #     f"airspeed_min: V_A_min[-1]={V_A_min[-1]}, n[-1]={n[-1]}, gamma_A_value={gamma_A_value}"
                    # )
                V_A_min_from_T_min = V_A_min_from_T_min[-1]
            except (ValueError, FloatingPointError):
                V_A_min_from_T_min = 0

        # Calculate airspeed constraint based on max. thrust and/or power
        # =====================================================================

        # TODO: Consider moving this to flight_performance_model module
        def V_A_min_from_T_gamma_A(
                T,
                n,
                gamma_A,
                V_A_dot,
                C_D0=self.C_D0,
                C_L0=self.C_L0,
                m=self.m,
                g=self.g,
                k=self.k,
                S_=self.S,
                rho=self.rho):
            # po2 = (m*g*sin(gamma_A) - T -
            #        2*k*g*m*C_L0*n)/(rho*S_*C_D0 + k*rho*S_*C_L0**2)
            neg_po2 = (T - V_A_dot*m + g*m*
                       (2*k*n*C_L0 - sin(gamma_A)))/(rho*S_*(C_D0 + k*C_L0**2))
            q = 4*k*(m*g*n)**2/((rho*S_)**2*(C_D0 + k*C_L0**2))
            inner_radicand = neg_po2**2 - q
            if inner_radicand < 0:
                return numpy.nan
            inner_sqrt_term = sqrt(inner_radicand)
            outer_radicand_neg = neg_po2 - inner_sqrt_term
            return sqrt(outer_radicand_neg)
            # if outer_radicand_neg >= 0:
            #     return sqrt(outer_radicand_neg)
            # else:
            #     return sqrt(neg_po2 + inner_sqrt_term)

        # # DEBUGGING: Plot T_over_V_A
        # # =====================================================================
        # from eduplane.utils import plotting
        # V_A_range = numpy.linspace(2, self.V_A_max, 100)
        # n_over_V_A = [utils.error2constant(n_func)(V_A) for V_A in V_A_range]
        # gamma_A_over_V_A = [gamma_A_func(V_A) for V_A in V_A_range]
        # T_over_V_A = [
        #     utils.error2constant(fpm.T_from_n_gamma_A)(
        #         n_value, V_A, V_A_dot, gamma_A_value, self.C_D0, self.C_L0,
        #         self.m, self.g, self.k, self.S, self.rho)
        #     for V_A, n_value, gamma_A_value in zip(
        #         V_A_range, n_over_V_A, gamma_A_over_V_A)
        # ]
        # df = pandas.DataFrame(
        #     {
        #         "V_A": V_A_range,
        #         "T_over_V_A": T_over_V_A,
        #         "n_over_V_A": n_over_V_A,
        #         "gamma_A_over_V_A": gamma_A_over_V_A,
        #     })
        # df.plot(x="V_A", y=["T_over_V_A", "n_over_V_A", "gamma_A_over_V_A"])
        # plotting.show()

        # # V_A_min_over_T_values
        # T_range = numpy.linspace(25, 30, 100)
        # V_A_min_over_T = []
        # for T_value in T_range:
        #     V_A_values = [10]
        #     while not utils.converged(V_A_values, 1e-6):
        #         last_V_A = V_A_values[-1] if len(
        #             V_A_values) > 0 else self.V_A_max
        #         gamma_A_value = gamma_A_func(last_V_A)
        #         n_value = n_func(last_V_A)
        #         print(f"{T_value=}, {n_value=}, {gamma_A_value=}")
        #         V_A_values.append(
        #             fpm.V_A_min_from_T_gamma_A(
        #                 T_value, n_value, gamma_A_value, V_A_dot, self.C_D0,
        #                 self.C_L0, self.m, self.g, self.k, self.S, self.rho))
        #         # if numpy.isnan(V_A_values[-1]):
        #         #     break
        #     V_A_min_over_T.append(V_A_values[-1])
        # df = pandas.DataFrame(
        #     {
        #         "V_A_min_over_T": V_A_min_over_T,
        #         "T_range": T_range,
        #     })
        # df.plot(x="V_A_min_over_T", y="T_range")
        # plotting.show()
        # # ---------------------------------------------------------------------

        V_A_min_from_T_max = 0
        V_A_min_from_P_max = 0
        if gamma_A_func is not None:
            try:
                V_A_min_from_T_max = [V_A_min_from_C_L]
                while not utils.converged(V_A_min_from_T_max, abs_tol=1e-3):
                    gamma_A_value = gamma_A_func(V_A_min_from_T_max[-1])
                    n_value = n_func(V_A_min_from_T_max[-1])
                    V_A_from_T_max = V_A_min_from_T_gamma_A(
                        T_max, n_value, gamma_A_value, V_A_dot, self.C_D0,
                        self.C_L0, self.m, self.g, self.k, self.S, self.rho)
                    V_A_min_from_T_max.append(V_A_from_T_max)
                    # print(
                    #     f"airspeed_min: V_A_min[-1]={V_A_min[-1]}, n[-1]={n[-1]}, gamma_A_value={gamma_A_value}"
                    # )
                V_A_min_from_T_max = V_A_min_from_T_max[-1]
            except (ValueError, FloatingPointError):
                V_A_min_from_T_max = 0

            try:
                V_A_min_from_P_max = [V_A_min_from_C_L]
                while not utils.converged(V_A_min_from_P_max, abs_tol=1e-3):
                    gamma_A_value = gamma_A_func(V_A_min_from_P_max[-1])
                    n_value = n_func(V_A_min_from_P_max[-1])
                    T_value = P_max/V_A_min_from_P_max[-1]
                    V_A_from_P_max = V_A_min_from_T_gamma_A(
                        T_value, n_value, gamma_A_value, V_A_dot, self.C_D0,
                        self.C_L0, self.m, self.g, self.k, self.S, self.rho)
                    V_A_min_from_P_max.append(V_A_from_P_max)
                    # print(
                    #     f"airspeed_min: V_A_min[-1]={V_A_min[-1]}, n[-1]={n[-1]}, gamma_A_value={gamma_A_value}"
                    # )
                V_A_min_from_P_max = V_A_min_from_P_max[-1]
            except (ValueError, FloatingPointError, ZeroDivisionError):
                V_A_min_from_P_max = 0

        return max(
            V_A_min_from_C_L, V_A_min_from_T_min, V_A_min_from_T_max,
            V_A_min_from_P_max)

    def vert_speed(
        self,
        V_A,
        T,
        n=None,
        gamma_A_dot=None,
        w_A_g_dot=None,
        chi_A_dot=0,
        V_A_dot=0,
        gamma_A_lower=math.radians(-89),
        gamma_A_upper=math.radians(89)):
        """Finds the steady-state equilibrium vertical speed for a given airspeed
        and thrust.

        If airspeed is set to "min", the steady-state equilibrium is searched
        for along the stall speed boundary.
        """

        if n is None:
            if gamma_A_dot is not None:
                gamma_A_dot_func = lambda _: gamma_A_dot
            elif w_A_g_dot is not None:
                gamma_A_dot_func = lambda new_gamma_A: self.air_path_inclination_rate(
                    V_A=V_A,
                    gamma_A=new_gamma_A,
                    V_A_dot=V_A_dot,
                    w_A_g_dot=w_A_g_dot,
                    chi_A_dot=chi_A_dot)
            else:
                gamma_A_dot_func = lambda _: 0
            n_func = lambda gamma_A: fpm.n_from_chi_A_dot_gamma_A_dot(
                V_A, chi_A_dot, gamma_A, gamma_A_dot_func(gamma_A), self.g)
        else:
            n_func = lambda _: n

        if V_A == "min":
            V_A_func = lambda _n: self.get_stall_speed(n=_n)
        else:
            V_A_func = lambda _: V_A

        def axial_force_balance(gamma_A):
            n_value = n_func(gamma_A)
            V_A_value = V_A_func(n_value)
            return self.excess_thrust(
                T_avail=T,
                n=n_value,
                V_A=V_A_value,
                V_A_dot=V_A_dot,
                gamma_A=gamma_A)

        try:
            if V_A == "min":
                gamma_A_upper = 0

            # # DEBUG PLOT
            # from matplotlib import pyplot as plt
            # fig, ax = plt.subplots(1)
            # gamma_A_range = numpy.linspace(gamma_A_lower, gamma_A_upper, 100)
            # T_exc = [axial_force_balance(g) for g in gamma_A_range]
            # ax.plot(gamma_A_range, T_exc)
            # plt.show()

            gamma_A = optimize.brentq(
                axial_force_balance, gamma_A_lower, gamma_A_upper)
            n_value = n_func(gamma_A)
            w_A_g = vt.w_A_g_from_V_A_gamma_A(V_A_func(n_value), gamma_A)
            return w_A_g
        except ValueError:
            # # DEBUG PLOT
            # from matplotlib import pyplot as plt
            # fig, ax = plt.subplots(1)
            # gamma_A_range = numpy.linspace(gamma_A_lower, gamma_A_upper, 100)
            # T_exc = [axial_force_balance(g) for g in gamma_A_range]
            # ax.plot(gamma_A_range, T_exc)
            # plt.show()
            raise

    def airspeed_max(
            self,
            *,
            gamma_A=None,
            w_A_g=None,
            chi_A_dot=0,
            V_A_dot=0,
            T_max=None,
            P_max=None):
        """Calculate the maximum feasible/admissable airspeed, at steady flight
        and max thrust or power (``T_max``/``P_max``).

        Same physical quantity as ``airspeed_bounds()``'s max output, but solved
        via a fixed-point iteration (not ``newton``). Requires either
        ``gamma_A`` or ``w_A_g``. See ``airspeed_min()`` for the min-thrust
        counterpart.
        """
        # Get default values for thrust and power constraint, if none specified
        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max

        # gamma_A is given either directly or calculated from w_A_g and V_A
        if gamma_A is not None and w_A_g is None:
            gamma_A_func = lambda _: gamma_A
        elif w_A_g is not None and gamma_A is None:
            # gamma_A_func = lambda V_A: vt.gamma_A_from_V_A_w_A_g(
            #     max(abs(w_A_g), V_A), w_A_g)
            gamma_A_func = lambda V_A: vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        else:
            raise ValueError("Either gamma_A or w_A_g must be provided")

        V_A_max = [self.V_A_max]
        n = [0]
        while not utils.converged(V_A_max, abs_tol=1e-3):
            # print(
            #     f"airspeed_max: iteration #{len(V_A_max)} with V_A_max={V_A_max[-1]}, w_A_g={w_A_g}"
            # )
            gamma_A_value = gamma_A_func(V_A_max[-1])
            n.append(
                fpm.n_s_from_V_A_gamma_A(
                    V_A_max[-1], gamma_A_value, chi_A_dot, self.g))

            V_A_from_T_max = fpm.V_A_from_T_gamma_A(
                T_max, n[-1], gamma_A_value, V_A_dot, self.C_D0, self.C_L0,
                self.m, self.g, self.k, self.S, self.rho)
            try:
                V_A_from_P_max = optimize.newton(
                    lambda V_A: fpm.P_from_T_V_A(
                        fpm.T_from_n_gamma_A(
                            fpm.n_s_from_V_A_gamma_A(
                                V_A, gamma_A_func(V_A),
                                chi_A_dot, self.g), V_A, V_A_dot,
                            gamma_A_func(V_A), self.C_D0, self.C_L0, self.m,
                            self.g, self.k, self.S, self.rho), V_A) - P_max,
                    V_A_from_T_max)
            except (FloatingPointError, RuntimeError):
                V_A_from_P_max = V_A_from_T_max

            V_A_max.append(min(V_A_from_T_max, V_A_from_P_max))
            # print(
            #     f"    V_A_from_T_max={V_A_from_T_max}, V_A_from_P_max={V_A_from_P_max}"
            # )

        # Apply minimum/maximum admissable airspeed constraint
        return min(V_A_max[-1], self.V_A_max)

    def airspeed(
            self,
            *,
            w_A_g=None,
            gamma_A=None,
            T=None,
            P=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            V_A_i0="max"):
        """Calculate airspeed at given vertical speed or air-path inclination
        angle for given thrust or power and (optionally) load factor.

        By default, the maximum feasible airspeed at the given thrust/power
        setting is calculated. Optionally, the minimum airspeed may be
        calculated by setting the initial iteration value V_A_i0="min".

        If the given input reference states are infeasible, a ValueError is
        raised.

        Instead of an air-path inclination angle, a vertical speed may be given.
        The air-path inclination angle is then calculted iteratively with the
        airspeed

        If no load factor is given, steady flight is assumed and load factor is
        calculated iteratively with airspeed.

        The airspeed returned is constrained to upper bound V_max.
        """
        if n is not None:
            n_func = lambda *_: n
        else:
            n_func = lambda V_A, gamma_A: fpm.n_s_from_V_A_gamma_A(
                V_A, gamma_A, chi_A_dot, self.g)

        if gamma_A is not None:
            gamma_A_func = lambda *_: gamma_A
        elif w_A_g is not None:
            gamma_A_func = lambda V_A: vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        else:
            raise ValueError("Either gamma_A or w_A_g must be specified")

        if V_A_i0 == "max":
            V_A_result = [self.V_A_max]
            V_A_from_T_func = lambda T, n, gamma_A, V_A_dot: fpm.V_A_from_T_gamma_A(
                T, n, gamma_A, V_A_dot, self.C_D0, self.C_L0, self.m, self.g,
                self.k, self.S, self.rho)
        elif V_A_i0 == "min":
            V_A_result = [self.eps]
            V_A_from_T_func = lambda T, n, gamma_A, V_A_dot: fpm.V_A_min_from_T_gamma_A(
                T, n, gamma_A, V_A_dot, self.C_D0, self.C_L0, self.m, self.g,
                self.k, self.S, self.rho)

        if T is not None:
            T_goal_func = lambda _: T
        elif P is not None:
            T_goal_func = lambda V_A: P/V_A
        else:
            raise ValueError("Either T or P must be specified")

        # DEBUGGING: Plot T_over_V_A and V_A_over_T
        # =====================================================================

        # V_A_range = numpy.linspace(.2, self.V_A_max, 100)
        # n_over_V_A = [n_func(V_A, gamma_A_func(V_A)) for V_A in V_A_range]
        # gamma_A_over_V_A = [gamma_A_func(V_A) for V_A in V_A_range]
        # T_over_V_A = [
        #     fpm.T_from_n_gamma_A(
        #         n_value, V_A, V_A_dot, gamma_A_value, self.C_D0, self.C_L0,
        #         self.m, self.g, self.k, self.S, self.rho)
        #     for V_A, n_value, gamma_A_value in zip(
        #         V_A_range, n_over_V_A, gamma_A_over_V_A)
        # ]
        # V_A_over_T = [
        #     V_A_from_T_func(
        #         T_value, n_func(V_A_result[-1], gamma_A_value), gamma_A_value,
        #         V_A_dot)
        #     for T_value, gamma_A_value in zip(T_over_V_A, gamma_A_over_V_A)
        # ]
        # df = pandas.DataFrame(
        #     {
        #         "V_A": V_A_range,
        #         "T_over_V_A": T_over_V_A,
        #         "n_over_V_A": n_over_V_A,
        #         "gamma_A_over_V_A": gamma_A_over_V_A,
        #         "V_A_over_T": V_A_over_T
        #     })

        # def iterate_V_A_from_T_gamma_A(
        #         T, n, gamma_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
        #     # po2 = (m*g*sin(gamma_A) - T -
        #     #        2*k*g*m*C_L0*n)/(rho*S_*C_D0 + k*rho*S_*C_L0**2)
        #     neg_po2 = (T - V_A_dot*m + g*m*
        #                (2*k*n*C_L0 - sin(gamma_A)))/(rho*S_*(C_D0 + k*C_L0**2))
        #     q = 4*k*(m*g*n)**2/((rho*S_)**2*(C_D0 + k*C_L0**2))
        #     inner_radicand = neg_po2**2 - q
        #     if inner_radicand < 0:
        #         # inner_sqrt_term = -sqrt(abs(inner_radicand))
        #         inner_sqrt_term = 0
        #     else:
        #         inner_sqrt_term = sqrt(inner_radicand)
        #         # inner_radicand = 0
        #         # return numpy.nan
        #     outer_radicand = neg_po2 + inner_sqrt_term
        #     if outer_radicand < 0:
        #         outer_radicand = 0
        #         # return -sqrt(abs(outer_radicand))
        #     return sqrt(outer_radicand)

        # T_range = numpy.linspace(self.T_min, self.T_max, 100)
        # V_A_over_T = []
        # for T_value in T_range:
        #     V_A_values = []
        #     while not utils.converged(V_A_values, 1e-6):
        #         last_V_A = V_A_values[-1] if len(
        #             V_A_values) > 0 else self.V_A_max
        #         gamma_A_value = gamma_A_func(last_V_A)
        #         n_value = n_func(last_V_A, gamma_A_value)
        #         V_A_values.append(
        #             iterate_V_A_from_T_gamma_A(
        #                 T_value, n_value, gamma_A_value, V_A_dot, self.C_D0,
        #                 self.C_L0, self.m, self.g, self.k, self.S, self.rho))
        #         if numpy.isnan(V_A_values[-1]):
        #             break
        #     V_A_over_T.append(V_A_values[-1])
        # df = pandas.DataFrame({
        #     "V_A_over_T": V_A_over_T,
        #     "T_range": T_range,
        # })
        # df.plot(x="V_A_over_T", y="T_range")
        # import matplotlib.pyplot as plt
        # plt.show()

        # =====================================================================

        if V_A_dot != 0 or P is not None:
            while not utils.converged(V_A_result, 1e-6):
                # gamma_A_value = gamma_A_func(V_A_result[-1])
                # n_value = n_func(V_A_result[-1], gamma_A_value)
                # T_goal = T_goal_func(V_A_result[-1])
                V_A_result.append(
                    optimize.newton(
                        lambda V_A: fpm.T_from_n_gamma_A(
                            n_func(V_A, gamma_A_func(V_A)), V_A, V_A_dot,
                            gamma_A_func(V_A), self.C_D0, self.C_L0, self.m,
                            self.g, self.k, self.S, self.rho) - T_goal_func(
                                V_A), V_A_result[-1]))
                # print(
                #     f"airspeed from T: V_A={V_A_result[-1]}, gamma_A={gamma_A_value}, n={n_value}, T={T}"
                # )
        else:
            while not utils.converged(V_A_result, 1e-6):
                gamma_A_value = gamma_A_func(V_A_result[-1])
                n_value = n_func(V_A_result[-1], gamma_A_value)
                T_goal = T_goal_func(V_A_result[-1])
                # print(
                #     f"airspeed from T_s: V_A={V_A_result[-1]}, gamma_A={gamma_A_value}, n={n_value}, T={T_goal}"
                # )
                V_A_result.append(
                    V_A_from_T_func(T_goal, n_value, gamma_A_value, V_A_dot))

        return min(self.V_A_max, V_A_result[-1])

    def airspeed_opt(
            self,
            *,
            opt_goal="min thrust",
            gamma_A=None,
            w_A_g=None,
            n=None,
            V_A_guess=None,
            V_A_dot_from_V_A=None,
            V_A_min=None,
            V_A_max=None,
            chi_A_dot=0):
        """Calculate optimal (i.e. min/max thrust/power) airspeed at given
        air-path azimuth angle or vertical speed.

        The optimization goal may be specified with the `opt_goal` argument:
          - "min thrust" (default): calculate minimum thrust airspeed
          - "max thrust": calculate maximum thrust airspeed
          - "min power": calculate minimum power airspeed

        Optionally, a load factor can be specified.
        If no load factor is specified, a quasi-steady flight state is assumed
        with n=f(gamma_A, chi_A_dot). A turn rate != 0 may be specified for
        this case to account for quasi-steady turning flight.

        Optionally, a function V_A_dot(V_A) may be provided for non-linear
        airspeed profiles which may result in significant changes of the thrust
        over airspeed profile (minimum may shift, maximum may occur).

        Optionally, min. and/or max. airspeed bounds may be provided to limit
        the optimization to the given range. Bounded optimization is activated
        if either one of the bounds is specified. The maximum airspeed bound
        is default-initialized to the aircraft model's maximum airspeed. The
        lower airspeed bound is default-initialized to the minimum airspeed.
        """

        if n is not None and abs(n) < self.eps:
            return self.airspeed_bounds(
                n=n, gamma_A=gamma_A, w_A_g=w_A_g, chi_A_dot=chi_A_dot)[0]

        if V_A_dot_from_V_A is None:
            V_A_dot_func = lambda *_: 0
        else:
            V_A_dot_func = V_A_dot_from_V_A
        # Only use bounded-optimization if caller specified min/max airspeed
        # or if caller specified w_A_g instead of gamma_A (in the latter case
        # at least V_A_min >= w_A_g must be respected in optimization).
        bounded_opt = (
            V_A_min is not None or V_A_max is not None or w_A_g is not None)

        # Use user-specified or default airspeed bounds
        # Note that default airspeed bounds are limited to the "right"/the
        # positive arc of the T(V_A) curve.
        V_A_min_default, V_A_max_default = self.airspeed_bounds(
            n=n,
            gamma_A=gamma_A,
            w_A_g=w_A_g,
            chi_A_dot=chi_A_dot,
            V_A_ref="max")
        if V_A_min is None:
            V_A_min = V_A_min_default
        if V_A_max is None:
            V_A_max = V_A_max_default

        # Start slightly below the max. airspeed
        V_A_guess = V_A_max*(1 - self.eps)

        # Select proper sign depending on wether to maximize or minimize
        # thrust or power
        sign = 1.0 if "min" in opt_goal else -1.0

        # Select proper thrust equation depending on given inputs
        if gamma_A is not None:
            if n is None:
                T_func = lambda V_A: sign*fpm.T_s_from_gamma_A(
                    V_A, gamma_A, chi_A_dot, self.C_D0, self.C_L0, self.m, self.
                    g, self.k, self.S, self.rho)
            else:
                T_func = lambda V_A: sign*fpm.T_from_n_gamma_A(
                    n, V_A, V_A_dot_func(V_A), gamma_A, self.C_D0, self.C_L0,
                    self.m, self.g, self.k, self.S, self.rho)
        elif w_A_g is not None:
            if n is None:
                T_func = lambda V_A: sign*fpm.T_s_from_w_A_g(
                    V_A, w_A_g, chi_A_dot, self.C_D0, self.C_L0, self.m, self.g,
                    self.k, self.S, self.rho)
            else:
                T_func = lambda V_A: sign*fpm.T_from_n_w_A_g(
                    n, V_A, V_A_dot_func(V_A), w_A_g, self.C_D0, self.C_L0, self
                    .m, self.g, self.k, self.S, self.rho)

        # Wrap thrust equation if the optimization goal is to minimize/maximize
        # power instead
        if "power" in opt_goal:
            opt_func = lambda V_A: V_A*T_func(V_A)
        else:
            opt_func = T_func

        # Select optimization method and set bounds (if any)
        if bounded_opt:
            method = None
            # method = "trust-constr"
            bounds = optimize.Bounds(V_A_min, V_A_max)
        else:
            # Use default method for unconstrained optimization
            method = None
            bounds = None

        # # DEBUGGING:
        # # ===================================================================
        # print("\nairspeed_opt:")
        # if gamma_A is not None:
        #     print(f"  gamma_A={math.degrees(gamma_A)}, n={n}")
        # elif w_A_g is not None:
        #     print(f"  w_A_g={w_A_g}, n={n}")
        # print(f"  V_A_min={V_A_min}, V_A_guess={V_A_guess}")
        # if bounds is not None:
        #     print(f"  bounds={bounds}")
        # if method is not None:
        #     print(f"  method={method}")
        # import matplotlib.pyplot as plt
        # import pandas
        # V_A_range = numpy.linspace(
        #     V_A_min if V_A_min is not None else 5, V_A_max, 100)
        # if gamma_A is not None:
        #     df = pandas.DataFrame(
        #         {
        #             "T":
        #             fpm.T_from_n_gamma_A(
        #                 n if n is not None else 1.0, V_A_range,
        #                 V_A_dot_func(V_A_range), gamma_A, self.C_D0, self.C_L0,
        #                 self.m, self.g, self.k, self.S, self.rho),
        #             # "T_s":
        #             # fpm.T_s_from_gamma_A(
        #             #     V_A_range, gamma_A, chi_A_dot, self.C_D0, self.C_L0,
        #             #     self.m, self.g, self.k, self.S, self.rho),
        #             "V_A_dot":
        #             V_A_dot_func(V_A_range),
        #             "opt_func":
        #             opt_func(V_A_range)
        #         },
        #         index=V_A_range)
        #     df["P"] = df["T"]*V_A_range
        #     df.plot(style={"opt_func": ".-"})
        # elif w_A_g is not None:
        #     df = pandas.DataFrame(
        #         {
        #             "T":
        #             fpm.T_from_n_w_A_g(
        #                 n if n is not None else 1.0, V_A_range,
        #                 V_A_dot_func(V_A_range), w_A_g, self.C_D0, self.C_L0,
        #                 self.m, self.g, self.k, self.S, self.rho),
        #             # "T_s":
        #             # fpm.T_s_from_w_A_g(
        #             #     V_A_range, V_A_dot_func(V_A_range), w_A_g,
        #             #     chi_A_dot, self.C_D0, self.C_L0, self.m, self.g,
        #             #     self.k, self.S, self.rho),
        #             "V_A_dot":
        #             V_A_dot_func(V_A_range)
        #         },
        #         index=V_A_range)
        #     df["P"] = df["T"]*V_A_range
        #     df.plot()
        # plt.show()
        # # -------------------------------------------------------------------

        # Run optimization.
        # Suppress user warnings that may occur with constrained optimization
        # method 'trust-constr'.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            V_A_opt = optimize.minimize(
                opt_func, V_A_guess, method=method, bounds=bounds).x.item()

        # Apply minimum and maximum airspeed constraint
        V_A_opt = utils.clip(V_A_opt, V_A_min, V_A_max)
        return V_A_opt

    def airspeed_rate(
            self,
            *,
            V_A,
            gamma_A=None,
            w_A_g=None,
            T=None,
            n=None,
            chi_A_dot=None,
            gamma_A_dot=None,
            w_A_g_dot=None,
            V_A_dot_eps=1e-4):
        """Calculate airspeed rate from current airspeed, thrust, load factor,
        air-path inclination angle or vertical speed, """
        if gamma_A is None and w_A_g is not None:
            gamma_A = vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g)
        if T is None:
            T = self.max_thrust(V_A)
        if n is not None:
            n_func = lambda _: n
        else:
            if gamma_A_dot is not None:
                n_func = lambda V_A_dot: self.load_factor(
                    V_A=V_A,
                    V_A_dot=V_A_dot,
                    gamma_A=gamma_A,
                    chi_A_dot=chi_A_dot,
                    gamma_A_dot=gamma_A_dot)
            elif w_A_g_dot is not None:
                n_func = lambda V_A_dot: self.load_factor(
                    V_A=V_A,
                    V_A_dot=V_A_dot,
                    gamma_A=gamma_A,
                    chi_A_dot=chi_A_dot,
                    w_A_g_dot=w_A_g_dot)
            else:
                raise ValueError(
                    "Either gamma_A_dot or w_A_g_dot must be specified.")

        V_A_dot = [
            fpm.V_A_dot_from_T(
                T, n_func(0), V_A, gamma_A, self.C_D0, self.C_L0, self.m,
                self.g, self.k, self.S, self.rho)
        ]
        while not utils.converged(V_A_dot, V_A_dot_eps):
            # print(f"V_A_dot={V_A_dot[-1]:.1f} , n={n_func(V_A_dot[-1]):.1f}")
            V_A_dot.append(
                fpm.V_A_dot_from_T(
                    T, n_func(V_A_dot[-1]), V_A, gamma_A, self.C_D0, self.C_L0,
                    self.m, self.g, self.k, self.S, self.rho))
        return V_A_dot[-1]

    def climb_angle_max(
            self,
            *,
            T_max=None,
            P_max=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            V_A_eps=1e-3):
        """Calculate max. feasible climb angle from max thrust and/or power and
        (optionally) load factor.

        If no load factor is specified, steady flight is assumed and the load
        factor is calculated iteratively with airspeed and air-path azimuth
        angle.

        Returns the max. feasible air-path inclination angle and corresponding
        airspeed as a tuple.
        """
        if n is not None:
            n_func = lambda *_: n
        else:
            n_func = lambda V_A, gamma_A: utils.bounded_value(
                fpm.n_s_from_V_A_gamma_A(V_A, gamma_A, chi_A_dot, self.g), self.
                n_min, self.n_max)

        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max

        V_A_gamma_A_max = [self.V_A_max]
        gamma_A_max = [0]

        while not utils.converged(gamma_A_max, abs_tol=1e-9, max_iter=1e3):
            n_func_of_V_A = lambda V_A: n_func(V_A, gamma_A_max[-1])

            # # DEBUG PLOT:
            # from matplotlib import pyplot as plt
            # # V_A_range = numpy.linspace(
            # #     V_A_bounds[0], V_A_bounds[1], 100)
            # V_A_range = numpy.linspace(0.5, self.V_A_max, 100)
            # T_ex = [
            #     self.drag(V_A=V_A, n=n_func_of_V_A(V_A)) -
            #     self.max_thrust(V_A=V_A, T_max=T_max, P_max=P_max)
            #     for V_A in V_A_range
            # ]
            # T_max_avail = [
            #     self.max_thrust(V_A=V_A, T_max=T_max, P_max=P_max)
            #     for V_A in V_A_range
            # ]
            # fig, ax = plt.subplots(1)
            # ax.plot(V_A_range, T_ex, label="T_ex")
            # ax.plot(V_A_range, T_max_avail, label="T_max")
            # plt.show()

            # Find airspeed maximizing excess thrust (thrust-drag)
            opt_result = optimize.minimize(
                lambda V_A: self.drag(V_A=V_A, n=n_func_of_V_A(V_A)) - self.
                max_thrust(V_A=V_A, T_max=T_max, P_max=P_max),
                V_A_gamma_A_max[-1],
                bounds=optimize.Bounds(V_A_eps, self.V_A_max))
            # print(opt_result)
            V_A_gamma_A_max_new = opt_result.x[0]

            # Calculate corresponding air-path inclination angle
            gamma_A_max_new = fpm.gamma_A_from_T(
                self.max_thrust(
                    V_A=V_A_gamma_A_max_new, T_max=T_max, P_max=P_max),
                n_func_of_V_A(V_A_gamma_A_max_new), V_A_gamma_A_max_new,
                V_A_dot, self.C_D0, self.C_L0, self.m, self.g, self.k, self.S,
                self.rho)
            gamma_A_max.append(gamma_A_max_new)
            V_A_gamma_A_max.append(V_A_gamma_A_max_new)

            # print(
            #     f"climb_angle_max iteration #{len(gamma_A_max)}: gamma_A_max={math.degrees(gamma_A_max[-1])}, V_A_gamma_max={V_A_gamma_A_max[-1]}"
            # )

        return gamma_A_max[-1], V_A_gamma_A_max[-1]

    def climb_angle_min(
            self,
            *,
            T=None,
            P=None,
            n=None,
            V_A_max=None,
            V_A_dot=0,
            chi_A_dot=0):
        """Find minimum air-path incl. angle (i.e. min. climb angle or max.
        descent angle) and corresponding airspeed for given thrust or power and
        (optionally) load factor

        If no load factor is specified, quasi-steady flight is assumed. The
        load factor is then calculated iteratively along with airspeed and
        air-path inclination angle.

        If no max. airspeed is specified, it is initialized to the aircraft
        model's maximum airspeed
        """
        if V_A_max is None:
            V_A_max = self.V_A_max
        if n is not None:
            n_func = lambda *_: n
        else:
            n_func = lambda _V_A, _gamma_A: utils.bounded_value(
                fpm.n_s_from_V_A_gamma_A(_V_A, _gamma_A, chi_A_dot, self.g),
                self.n_min, self.n_max)

        if T is not None:
            T_func = lambda *_: T
        elif P is not None:
            T_func = lambda V_A: P/V_A
        else:
            raise ValueError("Either T or P must be specified")

        # Iterate gamma_A_min and V_A
        gamma_A_min = [-0.5*math.pi]
        V_A_max_gamma_A_min = [
            self.airspeed_max(
                T_max=T,
                P_max=P,
                gamma_A=gamma_A_min[-1],
                chi_A_dot=chi_A_dot,
                V_A_dot=V_A_dot)
        ]
        # print(f"initial guess gamma_A_min={gamma_A_min[-1]}")
        while not utils.converged(gamma_A_min, abs_tol=1e-9, max_iter=1e3):
            V_A_max_gamma_A_min.append(
                min(
                    V_A_max,
                    self.airspeed_max(
                        T_max=T_func(V_A_max_gamma_A_min[-1]),
                        gamma_A=gamma_A_min[-1],
                        chi_A_dot=chi_A_dot,
                        V_A_dot=V_A_dot)))
            n_gamma_A_min = n_func(V_A_max_gamma_A_min[-1], gamma_A_min[-1])
            gamma_A_min.append(
                fpm.gamma_A_from_T(
                    T_func(V_A_max_gamma_A_min[-1]), n_gamma_A_min,
                    V_A_max_gamma_A_min[-1], V_A_dot, self.C_D0, self.C_L0,
                    self.m, self.g, self.k, self.S, self.rho))
            # print(
            #     f"gamma_A_min={gamma_A_min[-1]}, V_A_max_gamma_A_min={V_A_max_gamma_A_min}, n_gamma_A_min={n_gamma_A_min}"
            # )
        return gamma_A_min[-1], V_A_max_gamma_A_min[-1]

    def climb_rate_max(
            self,
            *,
            V_A=None,
            T_max=None,
            P_max=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            V_A_eps=1e-3):
        """Calculate max. feasible climb rate (-w_A_g) from max thrust and/or
        power and (optionally) airspeed and load factor.

        If no airspeed is specified, the max. climb rate airspeed is calculated
        iteratively.

        If no load factor is specified, steady flight is assumed and the load
        factor is calculated iteratively with airspeed and air-path azimuth
        angle.

        Returns the resulting climb rate (-w_A_g) and corresponding airspeed.
        """

        # TODO: This could be extended to respect max. lift coefficient
        if n is not None:
            n_func = lambda *_: n
        else:
            n_func = lambda V_A, gamma_A: utils.bounded_value(
                fpm.n_s_from_V_A_gamma_A(V_A, gamma_A, chi_A_dot, self.g), self.
                n_min, self.n_max)

        if T_max is None:
            T_max = self.T_max
        if P_max is None:
            P_max = self.P_max

        if V_A is not None:
            return -self.vert_speed(
                V_A=V_A,
                T=self.max_thrust(V_A, T_max, P_max),
                V_A_dot=V_A_dot,
                chi_A_dot=chi_A_dot,
                gamma_A_dot=0,
                n=n), V_A

        V_A_w_A_g_min = [self.V_A_max]
        w_A_g_min = [0]
        while not utils.converged(w_A_g_min, abs_tol=V_A_eps):
            gamma_A_w_A_g_min = vt.gamma_A_from_V_A_w_A_g(
                V_A_w_A_g_min[-1], w_A_g_min[-1])
            n_func_of_V_A = lambda V_A: n_func(V_A, gamma_A_w_A_g_min)

            # # DEBUG PLOT:
            # # ===============================================================
            # from matplotlib import pyplot as plt
            # # V_A_range = numpy.linspace(
            # #     V_A_bounds[0], V_A_bounds[1], 100)
            # V_A_range = numpy.linspace(0.5, self.V_A_max, 100)
            # P_ex = [
            #     V_A*(self.drag(V_A=V_A, n=n_func_of_V_A(V_A)) + weight_force) -
            #     self.max_power(V_A=V_A, T_max=T_max, P_max=P_max)
            #     for V_A in V_A_range
            # ]
            # P_at_n1 = [
            #     V_A*(self.drag(V_A=V_A, n=1) + weight_force) -
            #     self.max_power(V_A=V_A, T_max=T_max, P_max=P_max)
            #     for V_A in V_A_range
            # ]
            # # P_max_avail = [
            # #     self.max_power(V_A=V_A, T_max=T_max, P_max=P_max)
            # #     for V_A in V_A_range
            # # ]
            # drag = [
            #     self.drag(V_A=V_A, n=n_func_of_V_A(V_A)) for V_A in V_A_range
            # ]
            # drag_at_n1 = [self.drag(V_A=V_A, n=1) for V_A in V_A_range]
            # fig, ax = plt.subplots(1)
            # ax.plot(V_A_range, P_ex, label="T_ex")
            # ax.plot(V_A_range, P_at_n1, label="T_max")
            # # ax.plot(V_A_range, drag, label="T_ex", linestyle=":")
            # # ax.plot(V_A_range, drag_at_n1, label="T_max", linestyle=":")
            # plt.show()
            # # ---------------------------------------------------------------

            # Find airspeed maximizing excess power
            opt_result = optimize.minimize(
                lambda V_A: V_A*self.drag(V_A=V_A, n=n_func_of_V_A(V_A)) - self.
                max_power(V_A=V_A, T_max=T_max, P_max=P_max),
                V_A_w_A_g_min[-1],
                bounds=optimize.Bounds(V_A_eps, self.V_A_max))
            # print(opt_result)
            V_A_w_A_g_min_new = opt_result.x[0]

            # Calculate corresponding air-path inclination angle
            w_A_g_min_new = fpm.w_A_g_from_T(
                self.max_power(V_A=V_A_w_A_g_min_new, T_max=T_max, P_max=P_max)/
                V_A_w_A_g_min_new, n_func_of_V_A(V_A_w_A_g_min_new),
                V_A_w_A_g_min_new, V_A_dot, self.C_D0, self.C_L0, self.m,
                self.g, self.k, self.S, self.rho)
            w_A_g_min.append(w_A_g_min_new)
            V_A_w_A_g_min.append(V_A_w_A_g_min_new)

            # print(
            #     f"climb_rate_max iteration #{len(w_A_g_min)}: w_A_g_min={w_A_g_min[-1]}, V_A_w_A_g_min={V_A_w_A_g_min[-1]}, n={n_func_of_V_A(V_A_w_A_g_min[-1])}"
            # )
        return -w_A_g_min[-1], V_A_w_A_g_min[-1]

    def climb_rate_min(
            self, T_min=None, P_min=None, n=None, V_A_dot=0, chi_A_dot=0):
        """Find minimum feasible climb rate for a given thrust and power or
        setting and (optionally) airspeed and load factor.

        If no airspeed is given, the minimum climb rate airspeed is calculated
        iteratively.

        If no load factor is specified, quasi-steady flight is assumed. The
        load factor is then calculated iteratively along with airspeed and
        descent rate.

        If no max. airspeed is specified, it is initialized to the aircraft
        model's maximum airspeed.

        If no min. thrust is specified, it is initialized to the aircraft
        model's min. thrust.

        Returns corresponding pairs of climb rate (-w_A_g) and airspeed.
        """
        descent_rate_max, V_A_w_A_g_max = self.descent_rate_max(
            T_min=T_min, P_min=P_min, n=n, V_A_dot=V_A_dot, chi_A_dot=chi_A_dot)
        return -descent_rate_max, V_A_w_A_g_max

    def descent_rate_min(
            self,
            *,
            V_A=None,
            T_max=None,
            P_max=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0):
        """Calculate min. descent rate from thrust.
        See also `climb_rate_max`

        Returns corresponding pairs of descent rate (w_A_g) and airspeed.
        """
        climb_rate_max, V_A_w_A_g_min = self.climb_rate_max(
            V_A=V_A,
            T_max=T_max,
            P_max=P_max,
            n=n,
            V_A_dot=V_A_dot,
            chi_A_dot=chi_A_dot)
        return -climb_rate_max, V_A_w_A_g_min

    def terminal_dive_airspeed(
            self, *, T_min=None, P_min=None, gamma_A_lower=None, V_A_eps=1e-3):
        """Calculate the terminal-dive airspeed.

        This is the airspeed at which, in a vertical dive at the flight-path
        angle `gamma_A_lower`, the excess thrust of the aircraft equals zero.
        Above this airspeed, no steady vertical speed exists for the given
        min. thrust/power setting (the aircraft decelerates even in a vertical
        dive), i.e. ``vert_speed`` has no solution.

        If T_min/P_min are not given, the aircraft model's T_min is used.
        `gamma_A_lower` defaults to the inner bound of ``vert_speed``'s root
        search interval, so the two remain consistent.

        Returns the terminal-dive airspeed, clamped to V_A_max if the terminal
        velocity exceeds the structural limit.
        """
        if T_min is None and P_min is None:
            T_min = self.T_min

        if gamma_A_lower is None:
            # Match the lower bound of vert_speed's root-finding interval.
            gamma_A_lower = math.radians(-89)

        def excess_at_dive_bound(V_A):
            # Load factor at the dive bound, computed consistently with
            # vert_speed (chi_A_dot=0, quasi-steady).
            n = fpm.n_from_chi_A_dot_gamma_A_dot(
                V_A, 0, gamma_A_lower, 0, self.g)
            return self.excess_thrust(
                T_avail=self.max_thrust(V_A, T_min, P_min),
                n=n,
                V_A=V_A,
                V_A_dot=0,
                gamma_A=gamma_A_lower)

        # If excess > 0 at V_A_max, terminal velocity exceeds the structural
        # limit, so the dive is limited by V_A_max instead.
        if excess_at_dive_bound(self.V_A_max) > 0:
            return self.V_A_max

        V_A_lower = self.get_stall_speed(n=1)
        return optimize.brentq(
            excess_at_dive_bound, V_A_lower, self.V_A_max) - V_A_eps

    def descent_rate_max(
            self,
            *,
            T_min=None,
            P_min=None,
            n=None,
            V_A_dot=0,
            chi_A_dot=0,
            V_A_eps=1e-3):
        """Find maximum feasible descent rate for a given thrust and/or power
        setting and (optionally) airspeed and load factor.

        If no load factor is specified, quasi-steady flight is assumed. The
        load factor is then calculated iteratively along with airspeed and
        descent rate.

        If no min. thrust is specified, it is initialized to the aircraft
        model's min. thrust (T_min). If no min. power is specified, it is
        assumed to be zero.

        Returns corresponding pairs of descent rate (w_A_g) and airspeed.

        """

        # TODO: This could be extended to take into account stall speed constrained / max. lift coefficient

        if T_min is None and P_min is None:
            T_min = self.T_min

        # Lower bound is stall speed at level flight with n=1
        V_A_lower = self.get_stall_speed(n=1)

        # Upper bound is terminal-dive airspeed (or V_A_max)
        V_A_upper = self.terminal_dive_airspeed(T_min=T_min, P_min=P_min)

        res = optimize.minimize_scalar(
            lambda V_A: -self.vert_speed(
                V_A=V_A,
                T=self.max_thrust(V_A, T_min, P_min),
                n=n,
                V_A_dot=V_A_dot,
                chi_A_dot=chi_A_dot),
            bounds=(V_A_lower, V_A_upper),
            method='bounded')
        V_A_w_A_g_max = res.x
        w_A_g_max = -res.fun
        return w_A_g_max, V_A_w_A_g_max

    def get_speed_polar(self, *, T_range, n=1, V_A_dot=0, dV_A=0.1):
        V_A_max = self.airspeed_max(gamma_A=-0.5*math.pi, T_max=max(T_range))
        V_A_range = numpy.arange(dV_A, V_A_max, dV_A)
        data = {"V_A": V_A_range}
        for T in T_range:
            w_A_g_range = [
                fpm.w_A_g_from_T(
                    T, n, V_A, V_A_dot, self.C_D0, self.C_L0, self.m, self.g,
                    self.k, self.S, self.rho) for V_A in V_A_range
            ]
            data[f"w_A_g_T{T:d}"] = w_A_g_range
        df = pandas.DataFrame(data).set_index("V_A")
        return df

    # TODO: Refactor, document
    def fly_trajectory(self, traj):
        traj.df["chi_A_dot"] = km.chi_A_dot_from_chi_dot_V_A_xy_dot.eval_if(
            mask=traj.df["V_W_xy"] > 0,
            fill_value=traj.df["chi_dot"],
            V_A_xy=traj.df["V_A_xy"],
            V_A_xy_dot=traj.df["V_A_xy_dot"],
            chi=traj.df["chi"],
            chi_dot=traj.df["chi_dot"],
            V_W_xy=traj.df["V_W_xy"],
            chi_W=traj.df["chi_W"])
        traj.df["n"] = fpm.n_from_chi_A_dot_gamma_A_dot(
            traj.df.V_A, traj.df.chi_A_dot, traj.df.gamma_A,
            traj.df.gamma_A_dot, self.g)
        traj.df["T"] = fpm.T_from_n_gamma_A(
            traj.df.n, traj.df.V_A, traj.df.V_A_dot, traj.df.gamma_A, self.C_D0,
            self.C_L0, self.m, self.g, self.k, self.S, self.rho)
        traj.df["C_L"] = fpm.C_L_from_chi_A_dot_gamma_A_dot(
            traj.df.V_A, traj.df.chi_A_dot, traj.df.gamma_A,
            traj.df.gamma_A_dot, self.m, self.g, self.S, self.rho)
        traj.df["P"] = traj.df["T"]*traj.df["V_A"]
        # Integrate power (P) to obtain energy (E) using scipy integrate
        traj.df["E"] = integrate.cumulative_trapezoid(
            traj.df["P"], traj.df.index, initial=0)

        return traj

    def check_trajectory(self, traj, raise_on_fail=False):
        self.fly_trajectory(traj)
        min_T = traj.df["T"].min()
        max_T = traj.df["T"].max()
        min_n = traj.df["n"].min()
        max_n = traj.df["n"].max()
        min_C_L = traj.df["C_L"].min()
        max_C_L = traj.df["C_L"].max()
        check_T = min_T >= self.T_min and max_T <= self.T_max
        check_n = min_n >= self.n_min and max_n <= self.n_max
        check_C_L = min_C_L >= self.C_L_min and max_C_L <= self.C_L_max
        check = check_T and check_n and check_C_L
        if not check and raise_on_fail:
            raise ValueError("Trajectory not feasible")
        return check

    def check_feasible(
            self,
            V_A,
            gamma_A=None,
            w_A_g=None,
            V_A_dot=0,
            chi_A_dot=0,
            gamma_A_dot=0,
            w_A_g_dot=None,
            raise_on_fail=False,
            T_max=None,
            P_max=None,
            verbose=False):
        if w_A_g is None and gamma_A is not None:
            w_A_g = vt.w_A_g_from_V_A_gamma_A(V_A, gamma_A)
        if gamma_A_dot is None and w_A_g_dot is not None:
            gamma_A_dot = km.gamma_A_dot_from_V_A_dot_w_A_g_dot(
                V_A, V_A_dot, w_A_g_dot, gamma_A)
        if gamma_A_dot != 0:
            n = fpm.n_from_chi_A_dot_gamma_A_dot(
                V_A, chi_A_dot, gamma_A, gamma_A_dot, self.g)
        else:
            n = fpm.n_s_from_V_A_w_A_g(V_A, w_A_g, chi_A_dot, self.g)
        T = fpm.T_from_n_w_A_g(
            n, V_A, V_A_dot, w_A_g, self.C_D0, self.C_L0, self.m, self.g,
            self.k, self.S, self.rho)
        C_L = fpm.C_L_from_n(V_A, n, self.m, self.g, self.S, self.rho)
        T_max = self.max_thrust(V_A=V_A, T_max=T_max, P_max=P_max)
        check_T = T >= self.T_min and T <= T_max
        check_n = n >= self.n_min and n <= self.n_max
        check_C_L = C_L >= self.C_L_min and C_L <= self.C_L_max
        check = check_T and check_n and check_C_L
        if not check:
            violation_msg = ""
            if not check_T:
                violation_msg += f"\n  T={T} not within [{self.T_min},{T_max}]"
            if not check_n:
                violation_msg += f"\n  n={n} not within [{self.n_min},{self.n_max}]"
            if not check_C_L:
                violation_msg += f"\n  C_L={C_L} not within [{self.C_L_min},{self.C_L_max}]"
            if raise_on_fail:
                raise ValueError(
                    f"State not feasible: T={T}, n={n}, , C_L={C_L}" +
                    violation_msg)
            elif verbose:
                print(
                    f"State not feasible: T={T}, n={n}, C_L={C_L}" +
                    violation_msg)
        return check

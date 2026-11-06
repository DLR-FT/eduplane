# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Solvers to calculate segmented trochoid paths:
  - CSC: curve-straight-curve path like a classical Dubins path
  - CSCA: asymmetrical curve-straight-curve path like a Dubins path but with
          different turn rates for initial and final turn
  - CSCAA: asymmetrical curve-straight-curve path like a Dubins path but with
           different turn rates and different airspeeds for initial and final
           turn
  - CS: curve-straight path to reach a waypoint without heading/path azimuth
        constraint
"""

import copy
import math
from collections import namedtuple
from dataclasses import dataclass, field

import numpy
import pandas
import scipy.optimize

from eduplane.model import flight_performance_model as fpm
from eduplane.model import velocity_triangle as vt
from eduplane.model.ext_dubins_airplane import ExtDubinsAirplane
from eduplane.solver.longitudinal_maneuver import LongitudinalManeuver
from eduplane.solver.trochoid_periodic_path import (
    get_full_turn, get_trochoid_periodic_path)
from eduplane.traj import Trajectory
from eduplane.traj.horizontal_profiles import (
    ConstSpeedStraightFlight, LinearSpeedStraightFlight, SteadyTurn)
from eduplane.traj.horizontal_profiles.linear_speed_straight_flight import (
    tf_from_l_xy_tf_V_A_xy_tf)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.vertical_profiles import (
    ConstVertSpeedProfile, LinearVertSpeedProfile)
from eduplane.utils import logger, utils

# DEBUG_OUTPUT = True
DEBUG_OUTPUT = False


def get_signed_turnrate(turnrate, path_type, segment_index):
    if path_type == "LSL" or path_type == "LS":
        return turnrate*(-1, 0, -1)[segment_index]
    elif path_type == "RSR" or path_type == "RS":
        return turnrate*(1, 0, 1)[segment_index]
    elif path_type == "LSR":
        return turnrate*(-1, 0, 1)[segment_index]
    elif path_type == "RSL":
        return turnrate*(1, 0, -1)[segment_index]
    else:
        raise ValueError("Path type not supported: {}".format(path_type))


class CSC(object):

    def __init__(
        self,
        v_w,
        psi_w,
        v_a,
        omega,
        k,
        x_t10,
        y_t10,
        x_t20,
        y_t20,
        phi_1,
        phi_2,
        d1,
        d2,
    ):
        self.v_w = v_w
        self.psi_w = psi_w
        self.v_a = v_a
        self.omega = omega
        self.k = k
        self.x_t10 = x_t10
        self.y_t10 = y_t10
        self.x_t20 = x_t20
        self.y_t20 = y_t20
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.d1 = d1
        self.d2 = d2
        # Constants
        self.dphi_star = (phi_1 - phi_2 + k*2*math.pi)/(d2*omega)
        self.dx = x_t20 - x_t10
        self.dy = y_t20 - y_t10
        self.E = self.calc_E()
        self.G = self.calc_G()
        self.mem_F = (None, None)
        self.mem_sincos_term = (None, None)

    def calc_E(self):
        return self.v_a*(
            self.v_w*(self.d1 - self.d2)/(self.d1*self.d2*self.omega) - self.dy)

    def calc_G(self):
        return self.v_w*self.dy + (
            self.v_a*self.v_a*(self.d2 - self.d1)/(self.d1*self.d2*self.omega))

    def calc_F(self, t_a):
        if t_a == self.mem_F[0]:
            return self.mem_F[1]
        else:
            F = self.v_a*(
                self.dx + self.v_w*(t_a*(self.d1/self.d2 - 1) + self.dphi_star))
            self.mem_F = (t_a, F)
            return F

    def calc_sincos_term(self, t_a):
        if t_a == self.mem_sincos_term[0]:
            return self.mem_sincos_term[1]
        else:
            arg = self.d1*self.omega*t_a + self.phi_1
            sincos_term = (math.sin(arg), math.cos(arg))
            self.mem_sincos_term = (t_a, sincos_term)
            return sincos_term

    def f(self, t_a):
        F = self.calc_F(t_a)
        sin_term, cos_term = self.calc_sincos_term(t_a)
        f = self.E*cos_term + F*sin_term - self.G
        return f

    def f_dot(self, t_a):
        F = self.calc_F(t_a)
        sin_term, cos_term = self.calc_sincos_term(t_a)
        f_dot = (
            -self.E*self.d1*self.omega*sin_term +
            F*self.d1*self.omega*cos_term + self.v_a*self.v_w*
            (self.d1/self.d2 - 1)*sin_term)
        return f_dot


# CSC-asymmetric, i.e. with different turn rate turns
# TODO: Optimize like the regular CSC functor
CSCABase = namedtuple(
    "CSCA",
    [
        "v_w",
        "psi_w",
        "v_a",
        "omega1",
        "omega2",
        "k",
        "x_t10",
        "y_t10",
        "x_t20",
        "y_t20",
        "phi_1",
        "phi_2",
        "E",
        "G",
    ],
)


class CSCA(CSCABase):

    def __new__(
        cls,
        v_w,
        psi_w,
        v_a,
        omega1,
        omega2,
        k,
        x_t10,
        y_t10,
        x_t20,
        y_t20,
        phi_1,
        phi_2,
    ):
        return super(CSCA, cls).__new__(
            cls,
            v_w,
            psi_w,
            v_a,
            omega1,
            omega2,
            k,
            x_t10,
            y_t10,
            x_t20,
            y_t20,
            phi_1,
            phi_2,
            cls.calc_E(v_w, v_a, omega1, omega2, y_t10, y_t20),
            cls.calc_G(v_w, v_a, omega1, omega2, y_t10, y_t20),
        )

    @staticmethod
    def calc_E(v_w, v_a, omega1, omega2, y_t10, y_t20):
        return v_a*(v_w*(1/omega2 - 1/omega1) - (y_t20-y_t10))

    @staticmethod
    def calc_G(v_w, v_a, omega1, omega2, y_t10, y_t20):
        return v_a*v_a*(1/omega1 - 1/omega2) + v_w*(y_t20-y_t10)

    @staticmethod
    def calc_F(v_w, v_a, omega1, omega2, x_t10, x_t20, phi_1, phi_2, k, t_a):
        return v_a*(
            v_w*
            (t_a*(omega1/omega2 - 1) +
             (phi_1 - phi_2 + k*2*math.pi)/omega2) + x_t20 - x_t10)

    def f(self, t_a):
        sin_term = math.sin(self.omega1*t_a + self.phi_1)
        cos_term = math.cos(self.omega1*t_a + self.phi_1)
        F = self.calc_F(
            self.v_w,
            self.v_a,
            self.omega1,
            self.omega2,
            self.x_t10,
            self.x_t20,
            self.phi_1,
            self.phi_2,
            self.k,
            t_a,
        )
        f = self.E*cos_term + F*sin_term - self.G
        return f

    def f_dot(self, t_a):
        sin_term = math.sin(self.omega1*t_a + self.phi_1)
        cos_term = math.cos(self.omega1*t_a + self.phi_1)
        F = self.calc_F(
            self.v_w,
            self.v_a,
            self.omega1,
            self.omega2,
            self.x_t10,
            self.x_t20,
            self.phi_1,
            self.phi_2,
            self.k,
            t_a,
        )
        F_dot = self.v_a*self.v_w*(self.omega1/self.omega2 - 1)
        f_dot = (
            -self.E*self.omega1*sin_term + F*self.omega1*cos_term +
            F_dot*sin_term)
        return f_dot


# CSC 3D case with asymmetric (different turn rates) and constant vertical speed
@dataclass(kw_only=True)
class CSC3D(object):

    x_t0: float
    y_t0: float
    z_t0: float
    x_tf: float
    y_tf: float
    z_tf: float
    v_a_1: float
    v_a_2: float
    chi_t0: float
    chi_tf: float
    omega1: float
    omega2: float
    v_w: float
    psi_w: float
    k: int
    t_precision: float
    t_2pi1: float = field(default=None, init=False)
    t_2pi2: float = field(default=None, init=False)
    dx_t: float = field(default=None, init=False)
    dy_t: float = field(default=None, init=False)
    x_t10: float = field(default=None, init=False)
    y_t10: float = field(default=None, init=False)
    x_t20: float = field(default=None, init=False)
    y_t20: float = field(default=None, init=False)
    x1_t_a: float = field(default=None, init=False)
    y1_t_a: float = field(default=None, init=False)
    x2_t_b: float = field(default=None, init=False)
    y2_t_b: float = field(default=None, init=False)
    psi_1: float = field(default=None, init=False)
    psi_2: float = field(default=None, init=False)
    phi_1: float = field(default=None, init=False)
    phi_2: float = field(default=None, init=False)
    chi_t_a: float = field(default=None, init=False)
    v_a_xy_1: float = field(default=None, init=False)
    v_a_xy_2: float = field(default=None, init=False)
    w_a_g: float = field(default=None, init=False)
    t_beta: float = field(default=None, init=False)
    t_a: float = field(default=None, init=False)
    t_b: float = field(default=None, init=False)
    t_f: float = field(default=None, init=False)
    dpsi_c12: float = field(default=None, init=False)
    feasible: bool = field(default=True, init=False)

    def __post_init__(self, ):
        self.t_2pi1 = 2*math.pi/abs(self.omega1)
        self.t_2pi2 = 2*math.pi/abs(self.omega2)
        # Get x- and y-distance in trochoidal frame
        dx = self.x_tf - self.x_t0
        dy = self.y_tf - self.y_t0
        self.dx_t = dx*math.cos(self.psi_w) + dy*math.sin(self.psi_w)
        self.dy_t = -dx*math.sin(self.psi_w) + dy*math.cos(self.psi_w)
        # Initialize v_a_xy to v_a
        self.v_a_xy_1 = self.v_a_1
        self.v_a_xy_2 = self.v_a_2

    @staticmethod
    def calc_dpsi_c12(v_a_xy_1, v_a_xy_2, v_w, psi_w, omega1, phi_1, t_a):
        """Calculate difference in wind correction angles: dpsi_c12 = dpsi_c2 - dpsi_c1"""
        # Note that we need the track of the original/non-transformed trochoid
        chi_t_a = math.atan2(
            v_a_xy_1*math.sin(omega1*t_a + phi_1 + psi_w) + v_w*math.sin(psi_w),
            v_a_xy_1*math.cos(omega1*t_a + phi_1 + psi_w) + v_w*math.cos(psi_w),
        )
        # Direct difference:
        dpsi_c2 = -math.asin(v_w/v_a_xy_2*math.sin(psi_w - chi_t_a))
        dpsi_c1 = -math.asin(v_w/v_a_xy_1*math.sin(psi_w - chi_t_a))
        dpsi_c12 = dpsi_c2 - dpsi_c1
        return dpsi_c12

    @staticmethod
    def calc_E(v_w, v_a_xy_1, v_a_xy_2, omega1, omega2, y_t10, y_t20, dpsi_c12):
        return v_w*(v_a_xy_1/omega1 - v_a_xy_2/omega2*
                    math.cos(dpsi_c12)) + v_a_xy_1*(y_t20-y_t10)

    @staticmethod
    def calc_G(v_w, v_a_xy_1, v_a_xy_2, omega1, omega2, y_t10, y_t20, dpsi_c12):
        return v_a_xy_1*(v_a_xy_2/omega2*math.cos(dpsi_c12) -
                         v_a_xy_1/omega1) - v_w*(y_t20-y_t10)

    @staticmethod
    def calc_F(
        v_w,
        v_a_xy_1,
        v_a_xy_2,
        omega1,
        omega2,
        x_t10,
        x_t20,
        phi_1,
        phi_2,
        k,
        dpsi_c12,
        t_a,
    ):
        return v_a_xy_1*(
            v_w*(
                t_a*(omega1/omega2 - 1) +
                (phi_1 - phi_2 + dpsi_c12 + k*2*math.pi)/omega2) + x_t20 -
            x_t10) - v_w*v_a_xy_2/omega2*math.sin(dpsi_c12)

    def update_parameters(self, t_a):
        """
        Update any parameters regarding candidate trochoid path given a t_a value
        which are intermediate values for the root-finding equation
        """

        # Store t_a to use it as part of the final solution
        self.t_a = t_a

        # NOTE: Once below zero, we can abort the root-finding
        # t_a = utils.clip(t_a, 0, 2*self.t_2pi1)
        if self.t_a < 0:
            raise ValueError("t_a must be non-negative")

        w_a_g_values = [0 if self.w_a_g is None else self.w_a_g]
        while not utils.converged(w_a_g_values, abs_tol=self.t_precision):

            # Update air-path azimuth angles at start and goal
            self.psi_1 = vt.chi_A_from_V_A_xy_chi(
                self.v_a_xy_1, self.chi_t0, self.v_w, self.psi_w)
            self.psi_2 = vt.chi_A_from_V_A_xy_chi(
                self.v_a_xy_2, self.chi_tf, self.v_w, self.psi_w)

            # Get phase angles, i.e. angles between the air-relative speed vector (heading) and the wind vector
            self.phi_1 = self.psi_1 - self.psi_w
            self.phi_2 = self.psi_2 - self.psi_w - self.omega2*self.t_2pi2
            # print(
            #     f"phi_2: {self.phi_2} = {self.psi_2} - {self.psi_w} - {self.omega2}*{self.t_2pi2}"
            # )
            # print(
            #     f"{self.psi_2} - {self.psi_w} - {abs(self.omega2)*self.t_2pi2} = {self.phi_2}"
            # )

            # Calculate constants for both trochoids
            # (x_t10, y_t10) and (x_t20, y_t20) are the "instantaneous center points" of the first and second
            # trochoid at t1=0 and t2=0 such that both trochoids fulfill the boundary conditions, the first one
            # at t1=0 and the second one at t2=t_2pi.
            self.x_t10 = 0 - self.v_a_xy_1/self.omega1*math.sin(self.phi_1)
            self.y_t10 = 0 + self.v_a_xy_1/self.omega1*math.cos(self.phi_1)
            self.x_t20 = (
                self.dx_t - self.v_a_xy_2/self.omega2*
                math.sin(self.omega2*self.t_2pi2 + self.phi_2) -
                self.v_w*self.t_2pi2)
            self.y_t20 = self.dy_t + self.v_a_xy_2/self.omega2*math.cos(
                self.omega2*self.t_2pi2 + self.phi_2)

            self.dpsi_c12 = self.calc_dpsi_c12(
                self.v_a_xy_1, self.v_a_xy_2, self.v_w, self.psi_w, self.omega1,
                self.phi_1, self.t_a)
            self.t_b = (
                self.omega1/self.omega2*self.t_a +
                (self.phi_1 - self.phi_2 + self.dpsi_c12 + self.k*2*math.pi)/
                self.omega2)
            if -self.t_2pi2 <= self.t_b <= self.t_2pi2:
                self.feasible = True
            else:
                self.feasible = False
                self.t_b = utils.clip(self.t_b, -self.t_2pi2, self.t_2pi2)

            self.x1_t_a = self.v_a_xy_1/self.omega1*math.sin(
                self.omega1*self.t_a +
                self.phi_1) + self.v_w*self.t_a + self.x_t10
            self.y1_t_a = -self.v_a_xy_1/self.omega1*math.cos(
                self.omega1*self.t_a + self.phi_1) + self.y_t10
            self.x2_t_b = self.v_a_xy_2/self.omega2*math.sin(
                self.omega2*self.t_b +
                self.phi_2) + self.v_w*self.t_b + self.x_t20
            self.y2_t_b = -self.v_a_xy_2/self.omega2*math.cos(
                self.omega2*self.t_b + self.phi_2) + self.y_t20
            self.l_beta = math.sqrt(
                (self.x2_t_b - self.x1_t_a)**2 + (self.y2_t_b - self.y1_t_a)**2)
            self.chi_t_a = vt.chi_from_V_A_xy_chi_A(
                self.v_a_xy_1, self.psi_1 + self.omega1*self.t_a, self.v_w,
                self.psi_w)
            if self.v_a_xy_1 != self.v_a_xy_2:
                try:
                    self.t_beta = self.t_a + tf_from_l_xy_tf_V_A_xy_tf(
                        0,
                        self.l_beta,
                        self.v_a_xy_1,
                        self.v_a_xy_2,
                        self.chi_t_a,
                        self.v_w,
                        self.psi_w,
                        eps=self.t_precision)
                except FloatingPointError:
                    raise ValueError("Failed to calculate t_beta")
                self.t_f = self.t_beta + self.t_2pi2 - self.t_b
            else:
                x1_dot_t_a = self.v_a_xy_1*math.cos(
                    self.omega1*t_a + self.phi_1) + self.v_w
                y1_dot_t_a = self.v_a_xy_1*math.sin(
                    self.omega1*t_a + self.phi_1)
                v_k_beta = math.sqrt(x1_dot_t_a**2 + y1_dot_t_a**2)
                self.t_beta = t_a + self.l_beta/v_k_beta
                self.t_f = self.t_beta + self.t_2pi2 - self.t_b

            # print(
            #     f"t_a={self.t_a:.3f}, t_beta={self.t_beta:.3f}, t_f={self.t_f:.3f}, t_b={self.t_b:.3f}, feasible={self.feasible}"
            # )
            assert (self.t_f >= 0)
            # Update vertical and horizontal airspeed components based on final time
            try:
                self.w_a_g = (self.z_tf - self.z_t0)/self.t_f
                self.w_a_g = utils.clip_abs(
                    self.w_a_g,
                    min(self.v_a_1, self.v_a_2) - 1e-12)
            except FloatingPointError:
                self.w_a_g = 0.0
            assert (self.v_a_1 >= 0)
            assert (self.v_a_2 >= 0)
            assert (abs(self.w_a_g) <= self.v_a_1)
            assert (abs(self.w_a_g) <= self.v_a_2)
            self.v_a_xy_1 = math.sqrt(self.v_a_1**2 - self.w_a_g**2)
            self.v_a_xy_2 = math.sqrt(self.v_a_2**2 - self.w_a_g**2)

            self.v_a_xy_1 = max(self.v_a_xy_1, self.v_w)
            self.v_a_xy_2 = max(self.v_a_xy_2, self.v_w)

            # w_a_g_values.append(
            #     utils.round_by_eps(self.w_a_g, self.t_precision))
            w_a_g_values.append(self.w_a_g)

        return

    def f(self, t_a):
        self.update_parameters(t_a)

        sin_term = math.sin(self.omega1*self.t_a + self.phi_1)
        cos_term = math.cos(self.omega1*self.t_a + self.phi_1)

        E = self.calc_E(
            self.v_w, self.v_a_xy_1, self.v_a_xy_2, self.omega1, self.omega2,
            self.y_t10, self.y_t20, self.dpsi_c12)
        G = self.calc_G(
            self.v_w, self.v_a_xy_1, self.v_a_xy_2, self.omega1, self.omega2,
            self.y_t10, self.y_t20, self.dpsi_c12)
        F = self.calc_F(
            self.v_w, self.v_a_xy_1, self.v_a_xy_2, self.omega1, self.omega2,
            self.x_t10, self.x_t20, self.phi_1, self.phi_2, self.k,
            self.dpsi_c12, self.t_a)
        f = F*sin_term - E*cos_term + G
        return f


@dataclass(kw_only=True)
class CSC3D_ConstThrust(CSC3D):
    ac: ExtDubinsAirplane
    w_A_g_1: float = field(default=0)
    w_A_g_2: float = field(default=0)
    w_A_g_3: float = field(default=0)
    z_t1: float = field(default=0)
    z_t2: float = field(default=0)

    gain_w_A_g: float = field(default=0.5)

    def __post_init__(self, ):
        super().__post_init__()

        # thrust: weight force compensation: -m*g*w_A_g/V_A
        T_weight_force = self.ac.m*self.ac.g
        self.calc_T_weight_force = lambda V_A, w_A_g: -T_weight_force*w_A_g/V_A
        # thrust: parasite drag term: C_D0*V_A**2*rho*S_/2
        T_parasite_drag_term = self.ac.C_D0*self.ac.rho*self.ac.S*0.5
        self.calc_T_parasite_drag = lambda V_A_sq: T_parasite_drag_term*V_A_sq
        # thrust: induced drag: k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
        T_k_over_2_rho_S = self.ac.k/(2*self.ac.rho*self.ac.S)
        C_L0_rho_S = self.ac.C_L0*self.ac.rho*self.ac.S
        self.calc_T_induced_drag = lambda V_A_sq, n: T_k_over_2_rho_S/V_A_sq*(
            C_L0_rho_S*V_A_sq - 2*T_weight_force*n)**2
        # w_A_g_from_T: parasite drag term
        w_A_g_parasite_drag_term = self.ac.C_D0*self.ac.rho*self.ac.S/(
            2*self.ac.g*self.ac.m)
        self.calc_w_A_g_parasite_drag = lambda V_A, V_A_sq: w_A_g_parasite_drag_term*V_A*V_A_sq
        w_A_g_k_over_2_rho_S_m_g = T_k_over_2_rho_S/(self.ac.m*self.ac.g)
        self.calc_w_A_g_induced_drag = lambda V_A, V_A_sq, n: w_A_g_k_over_2_rho_S_m_g/V_A*(
            C_L0_rho_S*V_A_sq - 2*T_weight_force*n)**2

    def n_s_from_V_A_gamma_A(self, V_A, gamma_A, chi_A_dot, g):
        cos = numpy.cos
        sqrt = numpy.sqrt
        return cos(gamma_A)*sqrt(1 + (V_A*chi_A_dot/g)**2)

    def T_from_n_w_A_g(
            self, n, V_A, V_A_dot, w_A_g, C_D0, C_L0, m, g, k, S_, rho):
        # inertial_force = V_A_dot*m
        # weight_force_comp = -m*g*w_A_g/V_A
        # parasite_drag = C_D0*V_A**2*rho*S_/2
        # induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
        # print(
        #     f"T_from_n_w_A_g: {inertial_force=}, {weight_force_comp=}, {parasite_drag=}, {induced_drag=}, result={inertial_force + weight_force_comp + parasite_drag + induced_drag}"
        # )
        # return inertial_force + weight_force_comp + parasite_drag + induced_drag
        V_A_sq = V_A**2
        inertial_force = V_A_dot*self.ac.m
        weight_force_comp = self.calc_T_weight_force(V_A, w_A_g)
        parasite_drag = self.calc_T_parasite_drag(V_A_sq)
        induced_drag = self.calc_T_induced_drag(V_A_sq, n)
        # print(
        #     f"T_from_n_w_A_g: {inertial_force=}, {weight_force_comp=}, {parasite_drag=}, {induced_drag=}, result={inertial_force + weight_force_comp + parasite_drag + induced_drag}"
        # )
        return inertial_force + weight_force_comp + parasite_drag + induced_drag

    def w_A_g_from_T(self, T, n, V_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
        # inertial_force_term = V_A_dot*V_A/g
        # parasite_drag_term = C_D0*V_A**3*rho*S_/(2*g*m)
        # induced_drag_term = k*(C_L0*V_A**2*rho*S_ -
        #                        2*g*m*n)**2/(2*V_A*rho*g*m*S_)
        # thrust_term = T*V_A/(g*m)
        # print(
        #     f"w_A_g_from_T: {inertial_force_term=}, {parasite_drag_term=}, {induced_drag_term=}, {thrust_term=}, result={inertial_force_term + parasite_drag_term + induced_drag_term - thrust_term}"
        # )
        # return inertial_force_term + parasite_drag_term + induced_drag_term - thrust_term
        V_A_sq = V_A**2
        inertial_force_term = V_A_dot*V_A/g
        parasite_drag_term = self.calc_w_A_g_parasite_drag(V_A, V_A_sq)
        induced_drag_term = self.calc_w_A_g_induced_drag(V_A, V_A_sq, n)
        thrust_term = T*V_A/(g*m)
        # print(
        #     f"w_A_g_from_T: {inertial_force_term=}, {parasite_drag_term=}, {induced_drag_term=}, {thrust_term=}, result={inertial_force_term + parasite_drag_term + induced_drag_term - thrust_term}"
        # )
        return inertial_force_term + parasite_drag_term + induced_drag_term - thrust_term

    def update_parameters(self, t_a):
        """
        Update any parameters regarding candidate trochoid path given a t_a value
        which are intermediate values for the root-finding equation
        """

        # print(f"CSC3D_ConstThrust: update parameters for {t_a=}")
        # Store t_a to use it as part of the final solution
        # t_a = utils.clip(t_a, 0, 2*self.t_2pi1)
        self.t_a = t_a

        if t_a < 0:
            raise ValueError("t_a must be non-negative")

        t_f_values = [t_a]
        w_A_g_1 = w_A_g_2 = w_A_g_3 = 0
        while not utils.converged(t_f_values, abs_tol=self.t_precision):

            # print(
            #     f"> t_f iteration {len(t_f_values)} with t_f={t_f_values[-1]}")

            # Update air-path azimuth angles at start and goal
            self.psi_1 = vt.chi_A_from_V_A_xy_chi(
                self.v_a_xy_1, self.chi_t0, self.v_w, self.psi_w)
            self.psi_2 = vt.chi_A_from_V_A_xy_chi(
                self.v_a_xy_2, self.chi_tf, self.v_w, self.psi_w)

            # Get phase angles, i.e. angles between the air-relative speed vector (heading) and the wind vector
            self.phi_1 = self.psi_1 - self.psi_w
            self.phi_2 = self.psi_2 - self.psi_w - self.omega2*self.t_2pi2
            # print(
            #     f"phi_2: {self.phi_2} = {self.psi_2} - {self.psi_w} - {self.omega2}*{self.t_2pi2}"
            # )
            # print(
            #     f"{self.psi_2} - {self.psi_w} - {abs(self.omega2)*self.t_2pi2} = {self.phi_2}"
            # )

            # Calculate constants for both trochoids
            # (x_t10, y_t10) and (x_t20, y_t20) are the "instantaneous center points" of the first and second
            # trochoid at t1=0 and t2=0 such that both trochoids fulfill the boundary conditions, the first one
            # at t1=0 and the second one at t2=t_2pi.
            self.x_t10 = 0 - self.v_a_xy_1/self.omega1*math.sin(self.phi_1)
            self.y_t10 = 0 + self.v_a_xy_1/self.omega1*math.cos(self.phi_1)
            self.x_t20 = (
                self.dx_t - self.v_a_xy_2/self.omega2*
                math.sin(self.omega2*self.t_2pi2 + self.phi_2) -
                self.v_w*self.t_2pi2)
            self.y_t20 = self.dy_t + self.v_a_xy_2/self.omega2*math.cos(
                self.omega2*self.t_2pi2 + self.phi_2)

            self.dpsi_c12 = self.calc_dpsi_c12(
                self.v_a_xy_1, self.v_a_xy_2, self.v_w, self.psi_w, self.omega1,
                self.phi_1, self.t_a)
            self.t_b = (
                self.omega1/self.omega2*t_a +
                (self.phi_1 - self.phi_2 + self.dpsi_c12 + self.k*2*math.pi)/
                self.omega2)
            if -self.t_2pi2 <= self.t_b <= self.t_2pi2:
                self.feasible = True
            else:
                self.feasible = False
                self.t_b = utils.clip(self.t_b, -self.t_2pi2, self.t_2pi2)

            self.x1_t_a = self.v_a_xy_1/self.omega1*math.sin(
                self.omega1*t_a + self.phi_1) + self.v_w*t_a + self.x_t10
            self.y1_t_a = -self.v_a_xy_1/self.omega1*math.cos(
                self.omega1*t_a + self.phi_1) + self.y_t10
            self.x2_t_b = self.v_a_xy_2/self.omega2*math.sin(
                self.omega2*self.t_b +
                self.phi_2) + self.v_w*self.t_b + self.x_t20
            self.y2_t_b = -self.v_a_xy_2/self.omega2*math.cos(
                self.omega2*self.t_b + self.phi_2) + self.y_t20
            self.l_beta = math.sqrt(
                (self.x2_t_b - self.x1_t_a)**2 + (self.y2_t_b - self.y1_t_a)**2)
            self.chi_t_a = vt.chi_from_V_A_xy_chi_A(
                self.v_a_xy_1, self.psi_1 + self.omega1*t_a, self.v_w,
                self.psi_w)
            if self.v_a_xy_1 != self.v_a_xy_2:
                try:
                    self.t_beta = t_a + tf_from_l_xy_tf_V_A_xy_tf(
                        0,
                        self.l_beta,
                        self.v_a_xy_1,
                        self.v_a_xy_2,
                        self.chi_t_a,
                        self.v_w,
                        self.psi_w,
                        eps=self.t_precision)
                except FloatingPointError:
                    raise ValueError("Failed to calculate t_beta")
                self.t_f = self.t_beta + self.t_2pi2 - self.t_b
            else:
                x1_dot_t_a = self.v_a_xy_1*math.cos(
                    self.omega1*t_a + self.phi_1) + self.v_w
                y1_dot_t_a = self.v_a_xy_1*math.sin(
                    self.omega1*t_a + self.phi_1)
                v_k_beta = math.sqrt(x1_dot_t_a**2 + y1_dot_t_a**2)
                self.t_beta = t_a + self.l_beta/v_k_beta
                self.t_f = self.t_beta + self.t_2pi2 - self.t_b

            # Iteratively calculate segment-wise vertical speed for constant thrust
            # Variables denoted with double index: [<segment-idx>[<0/1 for segment start or end>]]
            # TODO: Make notation consistent (e.g. currently v_a_xy_2/omega2 are final turn segment)
            V_A_xy_dot_2 = (self.v_a_xy_2 -
                            self.v_a_xy_1)/(self.t_beta - self.t_a)
            # On first iteration only: initialize values with mean vertical speed
            if len(t_f_values) == 0:
                w_A_g_mean = (self.z_tf - self.z_t0)/self.t_f
                w_A_g_1 = w_A_g_2 = w_A_g_3 = w_A_g_mean

            # NOTE: interestingly, a nested iteration of tf (outer) and vertical speeds (inner) seems not required
            # while not utils.converged(w_A_g_2_values, abs_tol=1e-3):
            w_A_g_2_values = [w_A_g_2]
            while not utils.converged(w_A_g_2_values, abs_tol=self.t_precision):
                # for _ in [0]:

                # print(
                #     f"> > vert. speed. iteration {len(w_A_g_2_values)}: w_A_g_1={w_A_g_1:.3f}, w_A_g_2={w_A_g_2:.3f}, w_A_g_3={w_A_g_3:.3f}"
                # )

                gamma_A_1 = vt.gamma_A_from_V_A_xy_w_A_g(self.v_a_xy_1, w_A_g_1)
                gamma_A_20 = vt.gamma_A_from_V_A_xy_w_A_g(
                    self.v_a_xy_1, w_A_g_2)
                gamma_A_21 = vt.gamma_A_from_V_A_xy_w_A_g(
                    self.v_a_xy_2, w_A_g_2)
                gamma_A_3 = vt.gamma_A_from_V_A_xy_w_A_g(self.v_a_xy_2, w_A_g_3)

                n_1 = self.n_s_from_V_A_gamma_A(
                    self.v_a_xy_1, gamma_A_1, self.omega1, self.ac.g)
                n_20 = self.n_s_from_V_A_gamma_A(
                    self.v_a_xy_1, gamma_A_20, 0, self.ac.g)
                n_21 = self.n_s_from_V_A_gamma_A(
                    self.v_a_xy_2, gamma_A_21, 0, self.ac.g)
                n_3 = self.n_s_from_V_A_gamma_A(
                    self.v_a_xy_2, gamma_A_3, self.omega2, self.ac.g)
                # T_1 = self.T_from_n_w_A_g(
                #     n_1, self.v_a_1, 0, w_A_g_1, self.ac.C_D0, self.ac.C_L0,
                #     self.ac.m, self.ac.g, self.ac.k, self.ac.S, self.ac.rho)
                T_20 = self.T_from_n_w_A_g(
                    n_20, self.v_a_1, V_A_xy_dot_2, w_A_g_2, self.ac.C_D0,
                    self.ac.C_L0, self.ac.m, self.ac.g, self.ac.k, self.ac.S,
                    self.ac.rho)
                T_21 = self.T_from_n_w_A_g(
                    n_21, self.v_a_2, V_A_xy_dot_2, w_A_g_2, self.ac.C_D0,
                    self.ac.C_L0, self.ac.m, self.ac.g, self.ac.k, self.ac.S,
                    self.ac.rho)
                # print(f"> > {T_20=:.3f}, {T_21=:.3f}")
                # T_3 = self.T_from_n_w_A_g(
                #     n_3, self.v_a_2, 0, w_A_g_3, self.ac.C_D0, self.ac.C_L0,
                #     self.ac.m, self.ac.g, self.ac.k, self.ac.S, self.ac.rho)
                # Update turn segments vertical speeds to match thrust level of straight segment
                w_A_g_1_new = self.w_A_g_from_T(
                    T_20, n_1, self.v_a_1, 0, self.ac.C_D0, self.ac.C_L0,
                    self.ac.m, self.ac.g, self.ac.k, self.ac.S, self.ac.rho)
                w_A_g_3_new = self.w_A_g_from_T(
                    T_21, n_3, self.v_a_2, 0, self.ac.C_D0, self.ac.C_L0,
                    self.ac.m, self.ac.g, self.ac.k, self.ac.S, self.ac.rho)
                w_A_g_1 = w_A_g_1 + (w_A_g_1_new-w_A_g_1)*self.gain_w_A_g
                w_A_g_3 = w_A_g_3 + (w_A_g_3_new-w_A_g_3)*self.gain_w_A_g
                w_A_g_1 = utils.clip_abs(w_A_g_1, self.v_a_1 - 1e-12)
                w_A_g_3 = utils.clip_abs(w_A_g_3, self.v_a_2 - 1e-12)
                # if abs(w_A_g_1) > self.v_a_1:
                #     raise ValueError("w_A_g_1 out of bounds")
                # if abs(w_A_g_3) > self.v_a_2:
                #     raise ValueError("w_A_g_3 out of bounds")
                w_A_g_1 = utils.clip_abs(w_A_g_1, self.v_a_1)
                w_A_g_3 = utils.clip_abs(w_A_g_3, self.v_a_2)
                # Update straight segment vertical speed based on remaining altitutde transition
                dz1 = self.t_a*w_A_g_1
                dz3 = (self.t_f - self.t_beta)*w_A_g_3
                self.z_t1 = self.z_t0 + dz1
                self.z_t2 = self.z_tf - dz3
                w_A_g_2 = (self.z_t2 - self.z_t1)/(self.t_beta - self.t_a)
                w_A_g_2 = utils.clip_abs(w_A_g_2, min(self.v_a_1, self.v_a_2))
                if abs(w_A_g_2) > min(self.v_a_1, self.v_a_2):
                    raise ValueError("w_A_g_2 out of bounds")
                w_A_g_2_values.append(w_A_g_2)
                # print(
                #     f"> > {w_A_g_1=:.3f}, {w_A_g_2=:.3f}, {w_A_g_3=:.3f}, {dz1=:.3f}, {dz3=:.3f}"
                # )

            self.w_A_g_1 = w_A_g_1
            self.w_A_g_2 = w_A_g_2
            self.w_A_g_3 = w_A_g_3
            self.v_a_xy_1 = math.sqrt(self.v_a_1**2 - self.w_A_g_1**2)
            self.v_a_xy_2 = math.sqrt(self.v_a_2**2 - self.w_A_g_3**2)
            # if abs(self.v_a_xy_1) > self.v_w:
            #     raise ValueError("v_a_xy_1 out of bounds")
            # if abs(self.v_a_xy_2) > self.v_w:
            #     raise ValueError("v_a_xy_2 out of bounds")
            self.v_a_xy_1 = max(self.v_a_xy_1, self.v_w)
            self.v_a_xy_2 = max(self.v_a_xy_2, self.v_w)

            # NOTE: rounding t_f values results in faster detection of
            # non-convergence via checking for unique values
            t_f_values.append(self.t_f)
            # t_f_values.append(utils.round_by_eps(self.t_f, self.t_precision))

        # print(f"{self.z_t1=} {self.z_t0 + self.t_a*self.w_A_g_1}")
        # print(
        #     f"{self.z_t2=} {self.z_t0 + self.t_a*self.w_A_g_1 + (self.t_beta - self.t_a)*self.w_A_g_2} {self.z_tf - (self.t_f-self.t_beta)*self.w_A_g_3}"
        # )
        # print(
        #     f"{self.z_tf=} {self.z_t2 + (self.t_f - self.t_beta)*self.w_A_g_3}")
        # if abs(self.z_t0 + self.t_a*self.w_A_g_1 - self.z_t1) > 1e-3:
        #     raise ValueError()
        # if abs(self.z_tf - (self.t_beta - self.t_a)*self.w_A_g_2 -
        #        self.z_t2) > 1e-3:
        #     raise ValueError()
        # if abs(self.z_t1 + (self.t_beta - self.t_a)*self.w_A_g_2 -
        #        self.z_t2) > 1e-3:
        #     raise ValueError()

        # print(f"CSC3D_ConstThrust: update parameters finished with {t_a=}")
        return


# CSCAA - asymmetric turn rates and airspeeds
CSCAABase = namedtuple(
    "CSCAA",
    [
        "v_w",
        "psi_w",
        "v_a1",
        "v_a2",
        "omega1",
        "omega2",
        "k",
        "x_t10",
        "y_t10",
        "x_t20",
        "y_t20",
        "phi_1",
        "phi_2",
    ],
)


class CSCAA(CSCAABase):

    def __new__(
        cls,
        v_w,
        psi_w,
        v_a1,
        v_a2,
        omega1,
        omega2,
        k,
        x_t10,
        y_t10,
        x_t20,
        y_t20,
        phi_1,
        phi_2,
    ):
        return super(CSCAA, cls).__new__(
            cls,
            v_w,
            psi_w,
            v_a1,
            v_a2,
            omega1,
            omega2,
            k,
            x_t10,
            y_t10,
            x_t20,
            y_t20,
            phi_1,
            phi_2,
        )

    @staticmethod
    def calc_dpsi_c12(v_a1, v_a2, v_w, psi_w, omega1, phi_1, x_t10, y_t10, t_a):
        """Calculate difference in wind correction angles: dpsi_c12 = dpsi_c2 - dpsi_c1"""
        # Note that we need the track of the original/non-transformed trochoid
        chi = math.atan2(
            v_a1*math.sin(omega1*t_a + phi_1 + psi_w) + v_w*math.sin(psi_w),
            v_a1*math.cos(omega1*t_a + phi_1 + psi_w) + v_w*math.cos(psi_w),
        )
        # Direct difference:
        dpsi_c2 = -math.asin(v_w/v_a2*math.sin(psi_w - chi))
        dpsi_c1 = -math.asin(v_w/v_a1*math.sin(psi_w - chi))
        dpsi_c12 = dpsi_c2 - dpsi_c1
        return dpsi_c12

    @staticmethod
    def calc_E(v_w, v_a1, v_a2, omega1, omega2, y_t10, y_t20, dpsi_c12):
        return v_w*(v_a1/omega1 -
                    v_a2/omega2*math.cos(dpsi_c12)) + v_a1*(y_t20-y_t10)

    @staticmethod
    def calc_G(v_w, v_a1, v_a2, omega1, omega2, y_t10, y_t20, dpsi_c12):
        return v_a1*(v_a2/omega2*math.cos(dpsi_c12) -
                     v_a1/omega1) - v_w*(y_t20-y_t10)

    @staticmethod
    def calc_F(
        v_w,
        v_a1,
        v_a2,
        omega1,
        omega2,
        x_t10,
        x_t20,
        phi_1,
        phi_2,
        psi_w,
        k,
        dpsi_c12,
        t_a,
    ):
        return v_a1*(
            v_w*(
                t_a*(omega1/omega2 - 1) +
                (phi_1 - phi_2 + dpsi_c12 + k*2*math.pi)/omega2) + x_t20 -
            x_t10) - v_w*v_a2/omega2*math.sin(dpsi_c12)

    def f(self, t_a):
        sin_term = math.sin(self.omega1*t_a + self.phi_1)
        cos_term = math.cos(self.omega1*t_a + self.phi_1)
        dpsi_c12 = self.calc_dpsi_c12(
            self.v_a1,
            self.v_a2,
            self.v_w,
            self.psi_w,
            self.omega1,
            self.phi_1,
            self.x_t10,
            self.y_t10,
            t_a,
        )
        F = self.calc_F(
            self.v_w,
            self.v_a1,
            self.v_a2,
            self.omega1,
            self.omega2,
            self.x_t10,
            self.x_t20,
            self.phi_1,
            self.phi_2,
            self.psi_w,
            self.k,
            dpsi_c12,
            t_a,
        )
        E = self.calc_E(
            self.v_w,
            self.v_a1,
            self.v_a2,
            self.omega1,
            self.omega2,
            self.y_t10,
            self.y_t20,
            dpsi_c12,
        )
        G = self.calc_G(
            self.v_w,
            self.v_a1,
            self.v_a2,
            self.omega1,
            self.omega2,
            self.y_t10,
            self.y_t20,
            dpsi_c12,
        )
        f = F*sin_term - E*cos_term + G
        return f


# CS-path, i.e. single turn towards waypoint
# TODO: Optimize like the regular CSC functor
CSBase = namedtuple(
    "CS",
    ["v_w", "psi_w", "v_a", "omega", "psi_t0", "x_t0", "y_t0", "x_g", "y_g"])


class CS(CSBase):

    def __new__(cls, v_w, psi_w, v_a, omega, psi_t0, x_t0, y_t0, x_g, y_g):
        return super(CS, cls).__new__(
            cls, v_w, psi_w, v_a, omega, psi_t0, x_t0, y_t0, x_g, y_g)

    def f(self, t_1):
        sin_term = math.sin(self.omega*t_1 + self.psi_t0)*(
            self.v_a*(self.x_g - self.x_t0) - self.v_a*self.v_w*
            (math.sin(self.psi_w)/self.omega + math.cos(self.psi_w)*t_1))
        cos_term = math.cos(self.omega*t_1 + self.psi_t0)*(
            self.v_a*(self.y_g - self.y_t0) + self.v_a*self.v_w*
            (math.cos(self.psi_w)/self.omega - math.sin(self.psi_w)*t_1))
        const_term = (
            self.v_w*(
                math.sin(self.psi_w)*
                (self.x_g - self.x_t0) - math.cos(self.psi_w)*
                (self.y_g - self.y_t0)) - self.v_a**2/self.omega)
        return sin_term - cos_term + const_term

    def f_dot(self, t_1):
        sin_term = math.sin(self.omega*t_1 + self.psi_t0)*(
            self.omega*self.v_a*(self.y_g - self.y_t0) -
            self.v_a*self.v_w*self.omega*math.sin(self.psi_w)*t_1)
        cos_term = math.cos(self.omega*t_1 + self.psi_t0)*(
            self.omega*self.v_a*(self.x_g - self.x_t0) -
            self.v_a*self.v_w*self.omega*math.cos(self.psi_w)*t_1)
        return cos_term + sin_term


# For debugging mainly: Plot the functions used for root-finding
def plot_f(func, right_border):
    n_samples = 1000
    t_range = numpy.linspace(0, right_border, n_samples)
    f_data = [func.f(t) for t in t_range]
    f_dot_data = [func.f_dot(t) for t in t_range]
    df = pandas.DataFrame({"f": f_data, "f_dot": f_dot_data}, index=t_range)
    df.plot()


def solve_csc_path(
        start_x,
        start_y,
        start_dir,
        goal_x,
        goal_y,
        goal_dir,
        airspeed,
        turn_rate,
        wind_speed,
        wind_dir,
        path_type,
        precision=1e-2):
    """Solve CSC Dubins-Trochoid path.

    """
    rel_radius = airspeed/turn_rate

    if DEBUG_OUTPUT:
        print(
            "\nTrochoid path planning from ",
            (start_x, start_y),
            "with chi=",
            math.degrees(start_dir),
            "to",
            (goal_x, goal_y),
            "with chi",
            math.degrees(goal_dir),
            ", airspeed:",
            airspeed,
            ", air-path turn radius:",
            rel_radius,
            ", wind speed:",
            wind_speed,
            " wind dir:",
            math.degrees(wind_dir),
        )

    t_precision = precision/(airspeed+wind_speed)

    # Note that angular_precision is used only to rule out opposite direction
    # stitching points... it would still be interesting to find out what the
    # limiting factor for angular precision really is.
    # angular_precision = math.radians(1)
    angular_precision = t_precision*turn_rate

    psi_1 = vt.chi_A_from_V_A_xy_chi(airspeed, start_dir, wind_speed, wind_dir)
    psi_2 = vt.chi_A_from_V_A_xy_chi(airspeed, goal_dir, wind_speed, wind_dir)

    freq = 1/turn_rate
    t2pi = 2*math.pi*freq

    if DEBUG_OUTPUT:
        print("    t_2pi =", t2pi)
        print("    psi_1 =", math.degrees(psi_1))
        print("    psi_2 =", math.degrees(psi_2))

    # Calculate time required for one full turn
    t_2pi = 2*math.pi/turn_rate
    t_2pi_plus_tolerance = t_2pi + t_precision

    # Get signed turn rate values for first and second turn
    omega1 = get_signed_turnrate(turn_rate, path_type, 0)
    omega2 = get_signed_turnrate(turn_rate, path_type, 2)

    # Get phase angles, i.e. angles between the air-relative speed vector (heading) and the wind vector
    phi_t1 = psi_1 - wind_dir
    phi_t2 = psi_2 - wind_dir - omega2*t_2pi
    dphi = phi_t1 - phi_t2
    # Get x- and y-distance in trochoidal frame
    dx = goal_x - start_x
    dy = goal_y - start_y
    dx_t = dx*math.cos(wind_dir) + dy*math.sin(wind_dir)
    dy_t = -dx*math.sin(wind_dir) + dy*math.cos(wind_dir)
    if DEBUG_OUTPUT:
        print("(dx, dy) in inertial ref. frame: (", dx, ", ", dy, ")")
        print("(dx, dy) in trochoidal frame: (", dx_t, ", ", dy_t, ")")
        print(
            "phi_t1 = ",
            math.degrees(phi_t1),
            ", phi_t2 = ",
            math.degrees(phi_t2),
            ", dphi = ",
            math.degrees(dphi),
        )

    # if abs(dx) < precision and abs(dy) < precision and utils.angular_diff(
    #         psi_1, psi_2) < angular_precision:
    #     return [(path_type, 0, 0, 0)]

    # Calculate constant values for both trochoids
    # (x_t10, y_t10) and (x_t20, y_t20) are the "instantaneous center points" of the first and second
    # trochoid at t1=0 and t2=0 such that both trochoids fulfill the boundary conditions, the first one
    # at t1=0 and the second one at t2=t_2pi.
    x_t10 = 0 - airspeed/omega1*math.sin(phi_t1)
    y_t10 = 0 + airspeed/omega1*math.cos(phi_t1)
    x_t20 = (
        dx_t - airspeed/omega2*math.sin(omega2*t_2pi + phi_t2) -
        wind_speed*t_2pi)
    y_t20 = dy_t + airspeed/omega2*math.cos(omega2*t_2pi + phi_t2)
    if DEBUG_OUTPUT:
        print(
            "Constants t1=0 and t2=0 to fulfill the boundary conditions at t1=0 and t2=t_2pi\n",
            "(x_t10, y_t10): (",
            x_t10,
            ", ",
            y_t10,
            ")\n",
            "(x_t20, y_t20): (",
            x_t20,
            ", ",
            y_t20,
            ")",
        )

    solutions = []
    # Solve for CSC paths:
    # LSL and RSR paths (analytical solution)
    if path_type == "LSL" or path_type == "RSR":
        # For each k in [-3,-2,-1,0,1,2]... This is the maximum range of k for which optimal paths may occur.
        # NOTE: The order matters here, as we want to break on first solution, we want to inspect "good" solutions first.
        for k in [0, 1, -1, 2, -2, -3]:
            # Calculate the tangent angle
            alpha = math.atan2(
                (y_t20 - y_t10),
                ((x_t20-x_t10) + wind_speed*(dphi + k*2*math.pi)/omega2),
            )
            # Compute the first switching time
            t_a = (
                math.asin(wind_speed/airspeed*math.sin(alpha)) + alpha -
                phi_t1)/omega1
            # t_a must be in range [0,2*t_2pi)
            while t_a < -t_precision:
                t_a += t_2pi
            # If t_a is within the tolerance, limit its value to the interval [0, t_2pi]
            t_a = max(0.0, t_a)
            t_a = math.fmod(t_a, t_2pi)
            # The "other" valid solution will be t_a1 + t_2pi, however, it seems not to be required
            # probabyl due to checking the full range of k and above sanitization of t_a value
            if DEBUG_OUTPUT:
                print(
                    "LSL/RSR: k=", k, ", alpha=", math.degrees(alpha), ", t_a=",
                    t_a)
            # The second switching time expressed in parameter of second trochoid
            t_b = t_a + (dphi + k*2*math.pi)/omega2
            # t_b must be in range (-t_2pi,t_2pi]
            if t_b <= -t_2pi_plus_tolerance or t_b > t_2pi_plus_tolerance:
                if DEBUG_OUTPUT:
                    print(", t_b=", t_b, " is out of range")
                continue
            # If t_b is within the tolerance, limit its value to the valid interval
            t_b = max(t_b, -t_2pi)
            t_b = min(t_b, t_2pi)

            if DEBUG_OUTPUT:
                print(", t_b=", t_b)

            # Calculate the exit/entry point of the first/second trochoid
            x_t1_ta = (
                airspeed/omega1*math.sin(omega1*t_a + phi_t1) + wind_speed*t_a +
                x_t10)
            y_t1_ta = -airspeed/omega1*math.cos(omega1*t_a + phi_t1) + y_t10
            x_t2_tb = (
                airspeed/omega2*math.sin(omega2*t_b + phi_t2) + wind_speed*t_b +
                x_t20)
            y_t2_tb = -airspeed/omega2*math.cos(omega2*t_b + phi_t2) + y_t20
            x_t1_ta_dot = airspeed*math.cos(omega1*t_a + phi_t1) + wind_speed
            y_t1_ta_dot = airspeed*math.sin(omega1*t_a + phi_t1)

            # Check if tangents of the trochoids align (this *should* never happen for analytically
            # derived LSL/RSR paths, however, it feels much better to check this anyway)
            x_t2_tb_dot = airspeed*math.cos(omega2*t_b + phi_t2) + wind_speed
            y_t2_tb_dot = airspeed*math.sin(omega2*t_b + phi_t2)
            if (abs(x_t1_ta_dot - x_t2_tb_dot) > precision
                    or abs(y_t1_ta_dot - y_t2_tb_dot) > precision):
                if DEBUG_OUTPUT:
                    print(" -> tangents not aligned")
                continue

            # Calculate time spent on straight segment
            t_beta = t_a + (
                math.sqrt(
                    math.pow(x_t2_tb - x_t1_ta, 2) +
                    math.pow(y_t2_tb - y_t1_ta, 2))/
                math.sqrt(math.pow(x_t1_ta_dot, 2) + math.pow(y_t1_ta_dot, 2)))

            # Calculate final time
            t_final = t_beta + (t_2pi-t_b)
            if DEBUG_OUTPUT:
                print(", t_beta=", t_beta, ", t_f=", t_final)
            # Store result
            solutions.append((type, t_a, t_beta, t_final))

    # LSR and RSL paths (numerical solution)
    elif path_type == "LSR" or path_type == "RSL":
        # For each k in [-1,0,1,2,3]... This is the maximum range of k for which optimal LSR/RSL paths may occur.
        # for k in [-1, 0, 1, 2, 3]:
        for k in [0, 1, -1, 2, -2, -3]:
            # Construct the csc functor object that will be used for numerical solving and calculate the constants
            csc = CSC(
                v_a=airspeed,
                v_w=wind_speed,
                psi_w=wind_dir,
                x_t10=x_t10,
                y_t10=y_t10,
                x_t20=x_t20,
                y_t20=y_t20,
                omega=turn_rate,
                d1=get_signed_turnrate(1, path_type, 0),
                d2=get_signed_turnrate(1, path_type, 2),
                k=k,
                phi_1=phi_t1,
                phi_2=phi_t2,
            )
            # The grid of start values for the numerical solver has an impact for finding border cases, especially
            # when the distance between start and end position approaches 2 times the air-relative turn radius.
            # Note that reducing the list of initial values may lead to suboptimal paths or compromise completeness
            # guesses = [0, 1./2*t2pi, t2pi, 3./2.*t_2pi, 2*t_2pi]
            guesses = [
                0,
                1.0/4*t2pi,
                1.0/2*t2pi,
                3.0/4*t2pi,
                t2pi,
                5.0/4*t_2pi,
                3.0/2.0*t_2pi,
                7.0/4*t_2pi,
                2*t_2pi,
            ]
            for guess in guesses:
                # Solve for t_a in range of [0, 2*t_2pi]
                try:
                    logger.push("csc_newton", guess=guess)
                    # NOTE: For a scalar initial guess, scipy's newton returns
                    # ``(root, RootResults)`` when full_output is set.
                    t_a, info = scipy.optimize.newton(
                        logger.wrap_fn(csc.f, name="csc.f", arg_name="t_a"),
                        guess,
                        fprime=csc.f_dot,
                        tol=t_precision,
                        maxiter=100,
                        full_output=True)
                except RuntimeError:
                    # if True:
                    #     print(f"Failed to converge: {err}")
                    #     from matplotlib import pyplot as plt
                    #     t_range = numpy.arange(0, 2*t_2pi, t_precision)
                    #     f_values = [csc.f(t) for t in t_range]
                    #     fig, ax = plt.subplots(1)
                    #     ax.plot(t_range, f_values)
                    #     plt.show()
                    if DEBUG_OUTPUT:
                        print("Failed to converge for problem: " + str(csc))
                    logger.pop(converged=False)
                    continue
                logger.pop(
                    converged=bool(info.converged),
                    iterations=int(info.iterations),
                    function_calls=int(info.function_calls))
                if t_a < 0 or t_a > 2*t_2pi:
                    continue

                # The "other" valid solution will be t_a1 + t_2pi, however, it seems not to be required
                # probabyl due to checking the full range of k and above sanitization of t_a value
                t_a = math.fmod(t_a, t_2pi)
                # # t_a_candidates = [t_a1, t_a1 + t_2pi]
                # t_a_candidates = [t_a1]
                # for t_a in t_a_candidates:
                if DEBUG_OUTPUT:
                    print("RSL/LSR: k=", k, ", guess=", guess, ", t_a=", t_a)
                # The second switching time expressed in parameter of second trochoid
                t_b = (
                    get_signed_turnrate(1, path_type, 0)/
                    get_signed_turnrate(1, path_type, 2)*t_a +
                    (dphi + k*2*math.pi)/omega2)
                # t_b must be in range (-t_2pi,t_2pi]
                if t_b <= -t_2pi_plus_tolerance or t_b > t_2pi_plus_tolerance:
                    if DEBUG_OUTPUT:
                        print(", t_b=", t_b, " is out of range")
                    continue
                # If t_b is within the tolerance, limit its value to the valid interval
                t_b = max(t_b, -t_2pi)
                t_b = min(t_b, t_2pi)

                if DEBUG_OUTPUT:
                    print(", t_b=", t_b)

                # Calculate the exit/entry point of the first/second trochoid
                x_t1_ta = (
                    airspeed/omega1*math.sin(omega1*t_a + phi_t1) +
                    wind_speed*t_a + x_t10)
                y_t1_ta = (
                    -airspeed/omega1*math.cos(omega1*t_a + phi_t1) + y_t10)
                x_t2_tb = (
                    airspeed/omega2*math.sin(omega2*t_b + phi_t2) +
                    wind_speed*t_b + x_t20)
                y_t2_tb = (
                    -airspeed/omega2*math.cos(omega2*t_b + phi_t2) + y_t20)

                # Check if tangents of the trochoids align (*should* always be the case...)
                x_t1_ta_dot = (
                    airspeed*math.cos(omega1*t_a + phi_t1) + wind_speed)
                y_t1_ta_dot = airspeed*math.sin(omega1*t_a + phi_t1)
                x_t2_tb_dot = (
                    airspeed*math.cos(omega2*t_b + phi_t2) + wind_speed)
                y_t2_tb_dot = airspeed*math.sin(omega2*t_b + phi_t2)
                if (abs(x_t1_ta_dot - x_t2_tb_dot) > precision
                        or abs(y_t1_ta_dot - y_t2_tb_dot) > precision):
                    if DEBUG_OUTPUT:
                        print(
                            " -> tangents not aligned: ",
                            (x_t1_ta_dot, y_t1_ta_dot),
                            " != ",
                            (x_t2_tb_dot, y_t2_tb_dot),
                        )
                    continue
                # Check if tangents of the trochoids and the straight segment align. This is required as the root finding
                # method may find solutions where the trochoid tangents are aligned but the linear segment in between the
                # entry and exit points (t_a and t_b) is in opposite direction to the tangents.
                chi_tangent = math.atan2(y_t1_ta_dot, x_t1_ta_dot)
                chi_linear_segment = math.atan2(
                    y_t2_tb - y_t1_ta, x_t2_tb - x_t1_ta)
                if (abs(utils.angular_diff(chi_tangent, chi_linear_segment))
                        > angular_precision):
                    if DEBUG_OUTPUT:
                        print(
                            " -> infeasible solution: ",
                            math.degrees(chi_tangent),
                            " != ",
                            math.degrees(chi_linear_segment),
                        )
                    continue
                # Calculate time spent on straight segment
                t_beta = t_a + math.sqrt(
                    math.pow(x_t2_tb - x_t1_ta, 2) +
                    math.pow(y_t2_tb - y_t1_ta, 2))/math.sqrt(
                        math.pow(x_t1_ta_dot, 2) + math.pow(y_t1_ta_dot, 2))
                # Calculate final time
                t_final = t_beta + (t_2pi-t_b)
                if DEBUG_OUTPUT:
                    print(", t_beta=", t_beta, ", t_f=", t_final)
                # Store result
                solutions.append((type, t_a, t_beta, t_final))

    elif path_type == "LRL" or path_type == "RLR":
        print("Warning: Solving for CCC paths not implemented!")
    solutions = sorted(solutions, key=lambda x: x[-1])
    if DEBUG_OUTPUT and solutions:
        t1 = solutions[0][1]
        t2 = solutions[0][2]
        t3 = solutions[0][3]
        print("Selected solution with t_1=", t1, ", t_2=", t2, ", t_f=", t3)
    return solutions


def solve_csc_asym_path(
        start_x,
        start_y,
        start_dir,
        goal_x,
        goal_y,
        goal_dir,
        turn_rate1,
        turn_rate2,
        airspeed1,
        airspeed2,
        wind_speed,
        wind_dir,
        path_type,
        precision=0.001):
    rel_radius1 = airspeed1/turn_rate1
    rel_radius2 = airspeed2/turn_rate2

    if DEBUG_OUTPUT:
        print(
            "\nTrochoid (asymmetric) path planning from ",
            (start_x, start_y),
            "with chi=",
            math.degrees(start_dir),
            "to",
            (goal_x, goal_y),
            "with chi",
            math.degrees(goal_dir),
            ", airspeeds:",
            (airspeed1, airspeed2),
            ", air-path turn radii:",
            (rel_radius1, rel_radius2),
            ", wind speed:",
            wind_speed,
            " wind dir:",
            math.degrees(wind_dir),
        )

    # Precision value for comparing floating point values and interval checking.
    # For common problems (i.e. normal airspeeds, wind speeds, distances),
    # this value approximately gives the order of magnitude of the precision of
    # reaching the goal point with a curve-straight-curve path.
    t_precision = precision/(max(airspeed1, airspeed2) + wind_speed)
    angular_precision = t_precision*min(turn_rate1, turn_rate2)

    psi_1 = vt.chi_A_from_V_A_xy_chi(airspeed1, start_dir, wind_speed, wind_dir)
    psi_2 = vt.chi_A_from_V_A_xy_chi(airspeed2, goal_dir, wind_speed, wind_dir)

    # Calculate time required for one full turn
    t_2pi1 = 2*math.pi/turn_rate1
    t_2pi1_plus_tolerance = t_2pi1 + t_precision
    t_2pi2 = 2*math.pi/turn_rate2
    t_2pi2_plus_tolerance = t_2pi2 + t_precision

    if DEBUG_OUTPUT:
        print("    t_2pi =", (t_2pi1, t_2pi2))
        print("    psi_1 =", math.degrees(psi_1))
        print("    psi_2 =", math.degrees(psi_2))

    # Get signed turn rate values for first and second turn
    omega1 = get_signed_turnrate(turn_rate1, path_type, 0)
    omega2 = get_signed_turnrate(turn_rate2, path_type, 2)

    # Get phase angles, i.e. angles between the air-relative speed vector (heading) and the wind vector
    phi_t1 = psi_1 - wind_dir
    phi_t2 = psi_2 - wind_dir - omega2*t_2pi2
    # print(f"phi_t2: {phi_t2} = {psi_2} - {wind_dir} - {omega2}*{t_2pi2}")

    # Get x- and y-distance in trochoidal frame
    dx = goal_x - start_x
    dy = goal_y - start_y
    dx_t = dx*math.cos(wind_dir) + dy*math.sin(wind_dir)
    dy_t = -dx*math.sin(wind_dir) + dy*math.cos(wind_dir)
    if DEBUG_OUTPUT:
        print(f"(dx, dy) in inertial ref. frame: ({dx}, {dy})")
        print(f"(dx, dy) in trochoidal frame: {dx_t}, {dy_t}")
        print(f"{psi_1=}, {psi_2=}")
        print(f"{phi_t1=}, {phi_t2=}")
    # Calculate constant values for both trochoids
    # (x_t10, y_t10) and (x_t20, y_t20) are the "instantaneous center points" of the first and second
    # trochoid at t1=0 and t2=0 such that both trochoids fulfill the boundary conditions, the first one
    # at t1=0 and the second one at t2=t_2pi.
    x_t10 = 0 - airspeed1/omega1*math.sin(phi_t1)
    y_t10 = 0 + airspeed1/omega1*math.cos(phi_t1)
    x_t20 = (
        dx_t - airspeed2/omega2*math.sin(omega2*t_2pi2 + phi_t2) -
        wind_speed*t_2pi2)
    # print(f"x_t20={x_t20}, omega2={omega2}, t_2pi2={t_2pi2}, phi_t2={phi_t2}")

    y_t20 = dy_t + airspeed2/omega2*math.cos(omega2*t_2pi2 + phi_t2)
    if DEBUG_OUTPUT:
        print(
            "Constants t1=0 and t2=0 to fulfill the boundary conditions at t1=0 and t2=t_2pi\n",
            f"(x_t10, y_t10): ({x_t10}, {y_t10})\n",
            f"(x_t20, y_t20): ({x_t20}, {y_t20})\n",
        )

    solutions = []
    # Solve for CSC paths (numerical solution)
    if path_type in ["LSL", "RSR", "LSR", "RSL"]:
        # For each k in [-1,0,1,2,3]... This is the maximum range of k for which optimal LSR/RSL paths may occur.
        # for k in [-2, -1, 0, 1, 2, 3]:

        for k in [0, 1, -1, 2, -2, 3]:
            # Construct the csc functor object that will be used for numerical solving and calculate the constants
            csc = CSCAA(
                v_a1=airspeed1,
                v_a2=airspeed2,
                v_w=wind_speed,
                psi_w=wind_dir,
                x_t10=x_t10,
                y_t10=y_t10,
                x_t20=x_t20,
                y_t20=y_t20,
                omega1=omega1,
                omega2=omega2,
                k=k,
                phi_1=phi_t1,
                phi_2=phi_t2,
            )
            # The grid of start values for the numerical solver has an impact for finding border cases, especially
            # when the distance between start and end position approaches 2 times the air-relative turn radius.
            # guesses = [0, 1/8*t_2pi1, 2/8*t_2pi1, 3/8*t_2pi1, 4/8*t_2pi1,
            #            5/8*t_2pi1, 6/8*t_2pi1, 7/8*t_2pi1, 8/8*t_2pi1,
            #            9/8*t_2pi1, 10/8*t_2pi1, 11/8*t_2pi1, 12/8*t_2pi1,
            #            13/8*t_2pi1, 14/8*t_2pi1, 15/8*t_2pi1, 16/8*t_2pi1]
            guesses = [
                # 0,
                1.0/4*t_2pi1,
                1.0/2*t_2pi1,
                3.0/4*t_2pi1,
                t_2pi1,
                5.0/4*t_2pi1,
                3.0/2.0*t_2pi1,
                7.0/4*t_2pi1,
                # 2*t_2pi1,
            ]
            # guesses = [0, 1./2*t_2pi1, t_2pi1, 3./2.*t_2pi1, 2*t_2pi1]
            for guess in guesses:
                # Solve for t_a in range of [0, 2*t_2pi]
                try:
                    logger.push("csca_newton", guess=guess)
                    t_a, info = scipy.optimize.newton(
                        logger.wrap_fn(csc.f, name="csca.f", arg_name="t_a"),
                        guess,
                        tol=t_precision,
                        maxiter=100,
                        full_output=True)
                except (RuntimeError, ValueError) as err:
                    # if True:
                    #     print(f"Failed to converge: {err}")
                    #     from matplotlib import pyplot as plt
                    #     t_range = numpy.arange(0, 2*t_2pi1, t_precision)
                    #     f_values = [csc.f(t) for t in t_range]
                    #     fig, ax = plt.subplots(1)
                    #     ax.plot(t_range, f_values)
                    #     plt.show()
                    if DEBUG_OUTPUT:
                        print(f"Failed to converge: {err}")
                        # print(
                        #     "Failed to converge for problem: " + str(csc) +
                        #     " with guess: {}".format(guess))
                    logger.pop(converged=False)
                    continue
                logger.pop(
                    converged=bool(info.converged),
                    iterations=int(info.iterations),
                    function_calls=int(info.function_calls))
                if t_a < -t_precision or t_a > 2*t_2pi1_plus_tolerance:
                    # if t_a < 0 or t_a > 2*t_2pi1:
                    if DEBUG_OUTPUT:
                        print(
                            f"{path_type}: k={k}, guess={guess}, t_a={t_a} is out of range"
                        )
                    continue
                # # Enforce t_a in [0, 2*t_2pi1] interval
                t_a = utils.clip(t_a, 0, 2*t_2pi1)

                # for t_a in [t_a1, t_a1 + t_2pi1]:
                if DEBUG_OUTPUT:
                    print(f"{path_type}: k={k}, guess={guess}, t_a={t_a}")
                # The second switching time expressed in parameter of second trochoid
                dpsi_c12 = csc.calc_dpsi_c12(
                    csc.v_a1,
                    csc.v_a2,
                    csc.v_w,
                    csc.psi_w,
                    csc.omega1,
                    csc.phi_1,
                    csc.x_t10,
                    csc.y_t10,
                    t_a,
                )
                t_b = (
                    omega1/omega2*t_a +
                    (phi_t1 - phi_t2 + dpsi_c12 + k*2*math.pi)/omega2)

                # print(
                #     f"t_b: {t_b} = {omega1/omega2}*{t_a} + ({phi_t1} - {phi_t2} + {dpsi_c12} + {k}*2*math.pi)/{omega2}"
                # )

                # t_b must be in range (-t_2pi,t_2pi]
                if t_b <= -t_2pi2_plus_tolerance or t_b > t_2pi2_plus_tolerance:
                    if DEBUG_OUTPUT:
                        print(", t_b=", t_b, " is out of range")
                    continue
                # Enforce t_b in [-t_2pi2, t_2pi2] interval
                t_b = utils.clip(t_b, -t_2pi2, t_2pi2)

                if DEBUG_OUTPUT:
                    print(", t_b=", t_b)

                # Calculate the exit/entry point of the first/second trochoid
                x_t1_ta = (
                    airspeed1/omega1*math.sin(omega1*t_a + phi_t1) +
                    wind_speed*t_a + x_t10)
                y_t1_ta = (
                    -airspeed1/omega1*math.cos(omega1*t_a + phi_t1) + y_t10)
                x_t2_tb = (
                    airspeed2/omega2*math.sin(omega2*t_b + phi_t2) +
                    wind_speed*t_b + x_t20)
                y_t2_tb = (
                    -airspeed2/omega2*math.cos(omega2*t_b + phi_t2) + y_t20)
                x_t1_ta_dot = (
                    airspeed1*math.cos(omega1*t_a + phi_t1) + wind_speed)
                y_t1_ta_dot = airspeed1*math.sin(omega1*t_a + phi_t1)

                # Check if tangents of the trochoids align (*should* always be the case...)
                x_t2_tb_dot = (
                    airspeed2*math.cos(omega2*t_b + phi_t2) + wind_speed)
                y_t2_tb_dot = airspeed2*math.sin(omega2*t_b + phi_t2)
                if (abs(utils.angular_diff(
                        math.atan2(y_t1_ta_dot, x_t1_ta_dot),
                        math.atan2(y_t2_tb_dot, x_t2_tb_dot),
                )) > angular_precision):
                    # if abs(x_t1_ta_dot-x_t2_tb_dot) > precision \
                    # or abs(y_t1_ta_dot-y_t2_tb_dot) > precision:
                    if DEBUG_OUTPUT:
                        print(
                            " -> tangents not aligned: "
                            # ,(x_t1_ta_dot, y_t1_ta_dot)
                            ,
                            math.degrees(math.atan2(y_t1_ta_dot, x_t1_ta_dot)),
                            " != "
                            # ,(x_t2_tb_dot, y_t2_tb_dot))
                            ,
                            math.degrees(math.atan2(y_t2_tb_dot, x_t2_tb_dot)),
                        )
                    continue
                # Check if tangents of the trochoids and the straight segment align. This is required as the root finding
                # method may find solutions where the trochoid tangents are aligned but the linear segment in between the
                # entry and exit points (t_a and t_b) is in opposite direction to the tangents.
                chi_tangent = math.atan2(y_t1_ta_dot, x_t1_ta_dot)
                chi_linear_segment = math.atan2(
                    y_t2_tb - y_t1_ta, x_t2_tb - x_t1_ta)
                if (abs(utils.angular_diff(chi_tangent, chi_linear_segment))
                        > angular_precision):
                    if DEBUG_OUTPUT:
                        print(
                            " -> infeasible solution: ",
                            math.degrees(chi_tangent),
                            " != ",
                            math.degrees(chi_linear_segment),
                        )
                        print("xy(ta): {}".format((x_t1_ta, y_t1_ta)))
                        print("xy(tb): {}".format((x_t2_tb, y_t2_tb)))
                    continue
                # Calculate time spent on straight segment
                # This solution assumes airspeed1 to be continued for linear segment
                t_beta = t_a + math.sqrt(
                    math.pow(x_t2_tb - x_t1_ta, 2) +
                    math.pow(y_t2_tb - y_t1_ta, 2))/math.sqrt(
                        math.pow(x_t1_ta_dot, 2) + math.pow(y_t1_ta_dot, 2))
                t_beta = tf_from_l_xy_tf_V_A_xy_tf(
                    0,
                    math.sqrt(
                        math.pow(x_t2_tb - x_t1_ta, 2) +
                        math.pow(y_t2_tb - y_t1_ta, 2)), airspeed1, airspeed2,
                    chi_tangent, wind_speed, wind_dir)
                # Calculate final time
                t_final = t_beta + (t_2pi2-t_b)
                if DEBUG_OUTPUT:
                    print(", t_beta=", t_beta, ", t_f=", t_final)
                    print("xy(ta): {}".format((x_t1_ta, y_t1_ta)))
                    print("xy(tb): {}".format((x_t2_tb, y_t2_tb)))
                    # print("dist: {}".format(Vec2D(x_t1_ta,y_t1_ta).dist(Vec2D(x_t2_tb,y_t2_tb))))
                    # print("gs: {}".format(groundspeed_linear_segment))
                # Store result
                solutions.append((type, t_a, t_beta, t_final))

    elif path_type == "LRL" or path_type == "RLR":
        print("Warning: Solving for CCC paths not implemented!")
    if solutions:
        solutions = sorted(solutions, key=lambda x: x[-1])
        if DEBUG_OUTPUT:
            t1 = solutions[0][1]
            t2 = solutions[0][2]
            t3 = solutions[0][3]
            print("Selected solution with t_1=", t1, ", t_2=", t2, ", t_f=", t3)
        return solutions
    else:
        return None


def solve_csc_3d_path(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        turn_rate1,
        turn_rate2,
        airspeed1,
        airspeed2,
        wind_speed,
        wind_dir,
        path_type,
        precision=1e-4,
        strategy="const. vertical speed",
        **kwargs):
    if strategy == "const. thrust":
        CSC_Solver = CSC3D_ConstThrust
    elif strategy == "const. vertical speed":
        CSC_Solver = CSC3D
    else:
        raise RuntimeError(
            f"Unsupported strategy for solving 3D CSC paths: {strategy}")

    rel_radius1 = airspeed1/turn_rate1
    rel_radius2 = airspeed2/turn_rate2

    if DEBUG_OUTPUT:
        print(
            f"\nTrochoid 3D path planning from {(start_x, start_y, start_z)}"
            f", with chi={start_dir} ({math.degrees(start_dir)} deg)"
            f", to {(goal_x, goal_y, goal_z)}"
            f", with chi={goal_dir} ({math.degrees(goal_dir)} deg)"
            f", airspeed={(airspeed1, airspeed2)}"
            f", turn_rate={(turn_rate1, turn_rate2)}"
            f", air-path turn radii: {(rel_radius1, rel_radius2)}"
            f", wind speed={wind_speed}"
            f", wind dir={math.degrees(wind_dir)}"
            f", path_type={path_type}")

    # Precision value for comparing floating point values and interval checking.
    # For common problems (i.e. normal airspeeds, wind speeds, distances),
    # this value approximately gives the order of magnitude of the precision of
    # reaching the goal point with a curve-straight-curve path.
    t_precision = precision/(max(airspeed1, airspeed2) + wind_speed)
    angular_precision = t_precision*min(turn_rate1, turn_rate2)

    # Calculate time required for one full turn
    t_2pi1 = 2*math.pi/turn_rate1

    # if DEBUG_OUTPUT:
    #     print("    t_2pi =", (t_2pi1, t_2pi2))
    #     print("    psi_1 =", math.degrees(psi_1))
    #     print("    psi_2 =", math.degrees(psi_2))

    # Get signed turn rate values for first and second turn
    omega1 = get_signed_turnrate(turn_rate1, path_type, 0)
    omega2 = get_signed_turnrate(turn_rate2, path_type, 2)

    # # Get phase angles, i.e. angles between the air-relative speed vector (heading) and the wind vector
    # phi_t1 = psi_1 - wind_dir
    # phi_t2 = psi_2 - wind_dir - omega2*t_2pi2
    # # Get x- and y-distance in trochoidal frame
    # dx = goal_x - start_x
    # dy = goal_y - start_y
    # dx_t = dx*math.cos(wind_dir) + dy*math.sin(wind_dir)
    # dy_t = -dx*math.sin(wind_dir) + dy*math.cos(wind_dir)
    # if DEBUG_OUTPUT:
    #     print("(dx, dy) in inertial ref. frame: (", dx, ", ", dy, ")")
    #     print("(dx, dy) in trochoidal frame: (", dx_t, ", ", dy_t, ")")
    #     print(
    #         "phi_t1 = ", math.degrees(phi_t1), ", phi_t2 = ",
    #         math.degrees(phi_t2))

    # # Calculate constant values for both trochoids
    # # (x_t10, y_t10) and (x_t20, y_t20) are the "instantaneous center points" of the first and second
    # # trochoid at t1=0 and t2=0 such that both trochoids fulfill the boundary conditions, the first one
    # # at t1=0 and the second one at t2=t_2pi.
    # x_t10 = 0 - airspeed1/omega1*math.sin(phi_t1)
    # y_t10 = 0 + airspeed1/omega1*math.cos(phi_t1)
    # x_t20 = (
    #     dx_t - airspeed2/omega2*math.sin(omega2*t_2pi2 + phi_t2) -
    #     wind_speed*t_2pi2)
    # y_t20 = dy_t + airspeed2/omega2*math.cos(omega2*t_2pi2 + phi_t2)
    # if DEBUG_OUTPUT:
    #     print(
    #         "Constants t1=0 and t2=0 to fulfill the boundary conditions at t1=0 and t2=t_2pi\n",
    #         "(x_t10, y_t10): (",
    #         x_t10,
    #         ", ",
    #         y_t10,
    #         ")\n",
    #         "(x_t20, y_t20): (",
    #         x_t20,
    #         ", ",
    #         y_t20,
    #         ")",
    #     )

    k_values = {
        "LSL": [-2, -1, 0, 1, 2, 3],
        "RSR": [-2, -1, 0, 1, 2, 3],
        "LSR": [-1, 0, 1, 2, 3],
        "RSL": [-1, 0, 1, 2, 3]
    }
    solutions = []
    # Solve for CSC paths (numerical solution)
    if path_type in ["LSL", "RSR", "LSR", "RSL"]:
        for k in k_values[path_type]:

            # Construct the csc functor object that will be used for numerical solving and calculate the constants
            # print(
            #     f"Solving {path_type} path with strategy {strategy} and k={k}")
            csc = CSC_Solver(
                x_t0=start_x,
                y_t0=start_y,
                z_t0=start_z,
                x_tf=goal_x,
                y_tf=goal_y,
                z_tf=goal_z,
                v_a_1=airspeed1,
                v_a_2=airspeed2,
                chi_t0=start_dir,
                chi_tf=goal_dir,
                omega1=omega1,
                omega2=omega2,
                v_w=wind_speed,
                psi_w=wind_dir,
                k=k,
                t_precision=t_precision,
                **kwargs)
            # The grid of start values for the numerical solver has an impact for finding border cases, especially
            # when the distance between start and end position approaches 2 times the air-relative turn radius.
            # guesses = [0, 1/8*t_2pi1, 2/8*t_2pi1, 3/8*t_2pi1, 4/8*t_2pi1,
            #            5/8*t_2pi1, 6/8*t_2pi1, 7/8*t_2pi1, 8/8*t_2pi1,
            #            9/8*t_2pi1, 10/8*t_2pi1, 11/8*t_2pi1, 12/8*t_2pi1,
            #            13/8*t_2pi1, 14/8*t_2pi1, 15/8*t_2pi1, 16/8*t_2pi1]
            guesses = [
                # 0,
                1.0/4*t_2pi1,
                1.0/2*t_2pi1,
                3.0/4*t_2pi1,
                t_2pi1,
                5.0/4*t_2pi1,
                3.0/2.0*t_2pi1,
                7.0/4*t_2pi1,
                # 2*t_2pi1,
            ]
            for guess in guesses:
                if DEBUG_OUTPUT:
                    print(f"\nSolving {path_type} with {k=} and {guess=}")
                # Solve for t_a in range of [0, 2*t_2pi]
                try:
                    logger.push("csc3D_ta_newton", guess=guess)
                    t_a, info = scipy.optimize.newton(
                        logger.wrap_fn(csc.f, name="csc3D.f", arg_name="t_a"),
                        guess,
                        tol=t_precision,
                        maxiter=1000,
                        full_output=True)
                except (RuntimeError, ValueError, FloatingPointError) as err:
                    if DEBUG_OUTPUT:
                        print(f"-> Failed to converge: {err}")
                    logger.pop(converged=False)
                    continue

                if not csc.feasible:
                    logger.pop(infeasible=True)
                    continue
                else:
                    logger.pop(
                        converged=bool(info.converged),
                        iterations=int(info.iterations),
                        function_calls=int(info.function_calls))

                # Enforce t_beta >= t_a
                csc.t_beta = max(csc.t_beta, t_a)

                if csc.t_f < csc.t_beta - t_precision:
                    if DEBUG_OUTPUT:
                        print(
                            f"{path_type}: k={k}, guess={guess}, t_a={t_a}, t_f is out of range"
                        )
                        # print({str(csc)})
                    continue

                # Enforce t_f >= t_beta
                csc.t_f = max(csc.t_f, csc.t_beta)

                # Check if tangents of the trochoids and the straight segment align. This is required as the root finding
                # method may find solutions where the trochoid tangents are aligned but the linear segment in between the
                # entry and exit points (t_a and t_b) is in opposite direction to the tangents.
                x1_t_a_dot = (
                    csc.v_a_xy_1*math.cos(csc.omega1*t_a + csc.phi_1) +
                    wind_speed)
                y1_t_a_dot = csc.v_a_xy_1*math.sin(csc.omega1*t_a + csc.phi_1)
                chi_tangent = math.atan2(y1_t_a_dot, x1_t_a_dot)
                chi_linear_segment = math.atan2(
                    csc.y2_t_b - csc.y1_t_a, csc.x2_t_b - csc.x1_t_a)
                if (abs(utils.angular_diff(chi_tangent, chi_linear_segment))
                        > angular_precision):
                    if DEBUG_OUTPUT:
                        print(
                            f"{path_type}: k={k}, guess={guess}, t_a={t_a}, t_beta={csc.t_beta}, t_f={csc.t_f} infeasible solution, tangents not aligned"
                            f", {math.degrees(chi_tangent)} != {math.degrees(chi_linear_segment)}"
                            f", {angular_precision=}")
                        # print({str(csc)})
                    continue

                # Return a copy to the CSC3D functor object as it includes all required paramters of the solution
                solutions.append(copy.copy(csc))

                if DEBUG_OUTPUT:
                    print(
                        f"-> new candidate solution: {path_type}: k={k}, guess={guess}, t_a={t_a}, t_beta={csc.t_beta}, t_f={csc.t_f}"
                    )

    elif path_type == "LRL" or path_type == "RLR":
        print("Warning: Solving for CCC paths not implemented!")
    if solutions:
        solutions = sorted(solutions, key=lambda x: x.t_f)
        if DEBUG_OUTPUT:
            print(
                f"Selected solution with t_1={solutions[0].t_a}, t_2={solutions[0].t_beta}, t_f={solutions[0].t_f}, w_A_g={solutions[0].w_a_g}, v_A_xy={solutions[0].v_a_xy_1, solutions[0].v_a_xy_2}"
            )
        return solutions[0]
    else:
        print("Solver failed! Last CSC functor: ", csc)
        return None


def solve_cs_path(
    start_x,
    start_y,
    start_dir,
    goal_x,
    goal_y,
    airspeed,
    turn_rate,
    wind_speed,
    wind_dir,
    path_type,
):
    if DEBUG_OUTPUT:
        print(
            "\nSolving for path type {} from start_pos={} with chi_0={} and turn_rate={} to goal_pos={}"
            .format(
                path_type,
                (start_x, start_y),
                math.degrees(start_dir),
                math.degrees(turn_rate),
                (goal_x, goal_y),
            ))

    solutions = []

    # Get signed turn rate
    omega = get_signed_turnrate(turn_rate, path_type, 0)

    # Get initial heading
    psi_0 = vt.chi_A_from_V_A_xy_chi(airspeed, start_dir, wind_speed, wind_dir)

    precision = 1
    t_precision = precision/(airspeed+wind_speed)
    angular_precision = t_precision*turn_rate

    # Calculate time required for one full turn
    t_2pi = 2*math.pi/turn_rate

    # Calculate constant values for trochoid
    # (x_t0, y_t0) is the "instantaneous center point" of the
    # trochoid at t1=0 and t2=0 such that it fulfill the boundary condition at t0=0
    x_t0 = start_x - airspeed/omega*math.sin(psi_0)
    y_t0 = start_y + airspeed/omega*math.cos(psi_0)

    # LS and RS paths (numerical solution)
    if path_type == "LS" or path_type == "RS":
        # Construct the csc functor object that will be used for numerical solving and calculate the constants
        cs_func = CS(
            v_a=airspeed,
            v_w=wind_speed,
            psi_w=wind_dir,
            x_t0=x_t0,
            y_t0=y_t0,
            psi_t0=psi_0,
            omega=omega,
            x_g=goal_x,
            y_g=goal_y,
        )
        # NOTE: For debugging purposes, the function used for root finding can be plotted like that:
        # plot_f(cs_func, t_2pi)
        # The grid of start values for the numerical solver has an impact for finding border cases, especially
        # when the distance between start and end position approaches 2 times the air-relative turn radius.
        # Note that reducing the list of initial values may lead to suboptimal paths or compromise completeness
        guesses = [0, 0.25*t_2pi, 0.5*t_2pi, 0.75*t_2pi, t_2pi]
        for guess in guesses:
            # Solve for t_1 in range of [0, 2*t_2pi]
            try:
                t_1 = scipy.optimize.newton(
                    cs_func.f,
                    guess,
                    fprime=cs_func.f_dot,
                    tol=t_precision,
                    maxiter=100)
            except RuntimeError:
                if DEBUG_OUTPUT:
                    print("Failed to converge for problem: " + str(cs_func))
                continue
            if t_1 < 0 or t_1 > 2*t_2pi:
                if DEBUG_OUTPUT:
                    print("t_1={} is out-of-bounds".format(t_1))
                continue
            # Calculate the exit point of the trochoid
            x_t1 = (
                airspeed/omega*math.sin(omega*t_1 + psi_0) +
                wind_speed*math.cos(wind_dir)*t_1 + x_t0)
            y_t1 = (
                -airspeed/omega*math.cos(omega*t_1 + psi_0) +
                wind_speed*math.sin(wind_dir)*t_1 + y_t0)
            x_t1_dot = airspeed*math.cos(omega*t_1 +
                                         psi_0) + wind_speed*math.cos(wind_dir)
            y_t1_dot = airspeed*math.sin(omega*t_1 +
                                         psi_0) + wind_speed*math.sin(wind_dir)
            # Check if this is a feasible solution
            # TODO: It could be worth to further investigate the roots of these infeasible solutions to possibly rule/filter
            # them out at an earlier point (possibly look at sign of f_dot?)
            chi_goal = math.atan2((goal_y - y_t1), (goal_x - x_t1))
            chi_tangent = math.atan2(y_t1_dot, x_t1_dot)
            if abs(utils.angular_diff(chi_goal,
                                      chi_tangent)) > angular_precision:
                if DEBUG_OUTPUT:
                    print(
                        "t_1={} is an infeasible solution: {} != {}".format(
                            t_1, math.degrees(chi_goal),
                            math.degrees(chi_tangent)))
                    print("x_t1,y_t1", x_t1, y_t1)
                continue
            # Compute t_final
            t_final = t_1 + math.sqrt(
                math.pow(goal_x - x_t1, 2) + math.pow(goal_y - y_t1, 2)
            )/math.sqrt(math.pow(x_t1_dot, 2) + math.pow(y_t1_dot, 2))
            if DEBUG_OUTPUT:
                print(
                    "Found CS solution: chi_t1={}, t_1={}, t_final={}".format(
                        math.degrees(chi_tangent), t_1, t_final))
            # Store result
            solutions.append((path_type, t_1, t_final))
    if solutions:
        solutions = sorted(solutions, key=lambda x: x[-1])
        if DEBUG_OUTPUT:
            t1 = solutions[0][1]
            tf = solutions[0][2]
            print("Selected solution with t_1=", t1, ", t_f=", tf)
        return solutions
    else:
        return None


def get_trochoid_path(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        wind_speed,
        wind_dir,
        path_type,
        strategy=None,
        **kwargs):
    if start_z is not None:
        assert (goal_z is not None)
        assert (goal_dir is not None)
        return get_trochoid_path_3D_const_vert_speed(
            start_x, start_y, start_z, start_dir, goal_x, goal_y, goal_z,
            goal_dir, airspeed, turn_rate, wind_speed, wind_dir, path_type,
            strategy, **kwargs)
    else:
        return get_trochoid_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeed,
            turn_rate, wind_speed, wind_dir, path_type, **kwargs)


def get_trochoid_path_2D(
    start_x,
    start_y,
    start_dir,
    goal_x,
    goal_y,
    goal_dir,
    airspeed,
    turn_rate,
    wind_speed,
    wind_dir,
    path_type,
    start_rounds=0,
    end_rounds=0,
    start_end_round_func=get_trochoid_periodic_path,
    **kwargs,
):
    """Get a 2D Trochoid-Dubins path from start to goal configuration, optionally with start and end rounds.
    """
    # Get initial and final turn rate
    if isinstance(turn_rate, tuple) or isinstance(turn_rate, list):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        turn_rate2 = turn_rate

    # Get initial and final airspeed
    if isinstance(airspeed, tuple) or isinstance(airspeed, list):
        if len(airspeed) == 6:
            airspeed_start_straight = airspeed[0]
            airspeed_start_turn = airspeed[1]
            airspeed_start_acc = airspeed[2]
            airspeed_end_acc = airspeed[3]
            airspeed_end_turn = airspeed[4]
            airspeed_end_straight = airspeed[5]
        elif len(airspeed) == 2:
            airspeed_start_straight = airspeed[0]
            airspeed_start_turn = airspeed[0]
            airspeed_start_acc = airspeed[0]
            airspeed_end_acc = airspeed[1]
            airspeed_end_turn = airspeed[1]
            airspeed_end_straight = airspeed[1]
        else:
            raise ValueError(
                "Argument airspeed must either be scalar or 2 or 6 values")
    else:
        airspeed_start_straight = airspeed
        airspeed_start_turn = airspeed
        airspeed_start_acc = airspeed
        airspeed_end_acc = airspeed
        airspeed_end_turn = airspeed
        airspeed_end_straight = airspeed

    # Construct start/end rounds first - this will give us
    # the "true" start goal configurations for non-periodic
    # start/end rounds
    start_round_segments = []
    start_round_offset_x = 0
    start_round_offset_y = 0
    if start_rounds:
        start_round_seg = start_end_round_func(
            start_x, start_y, start_dir,
            airspeed_start_straight, airspeed_start_turn,
            get_signed_turnrate(turn_rate2, path_type, 0), wind_speed, wind_dir)
        start_round_offset_x = (
            start_round_seg.xf["x_g"] - start_round_seg.x0["x_g"])*start_rounds
        start_round_offset_y = (
            start_round_seg.xf["y_g"] - start_round_seg.x0["y_g"])*start_rounds
    for _ in range(start_rounds):
        start_round_segments.append(copy.deepcopy(start_round_seg))

    end_round_segments = []
    end_round_offset_x = 0
    end_round_offset_y = 0
    if end_rounds:
        end_round_seg = start_end_round_func(
            goal_x, goal_y, goal_dir, airspeed_end_straight, airspeed_end_turn,
            get_signed_turnrate(turn_rate2, path_type, 2), wind_speed, wind_dir)
        end_round_offset_x = (
            end_round_seg.xf["x_g"] - end_round_seg.x0["x_g"])*end_rounds
        end_round_offset_y = (
            end_round_seg.xf["y_g"] - end_round_seg.x0["y_g"])*end_rounds
    for _ in range(end_rounds):
        end_round_segments.append(copy.deepcopy(end_round_seg))

    # Apply offsets from start/end rounds
    start_x += start_round_offset_x
    start_y += start_round_offset_y
    goal_x -= end_round_offset_x
    goal_y -= end_round_offset_y

    # Call symmetric or asymmetric 2D solver
    if airspeed_start_turn != airspeed_end_turn or turn_rate1 != turn_rate2:
        solutions = solve_csc_asym_path(
            start_x,
            start_y,
            start_dir,
            goal_x,
            goal_y,
            goal_dir,
            turn_rate1,
            turn_rate2,
            airspeed_start_turn,
            airspeed_end_turn,
            wind_speed,
            wind_dir,
            path_type,
            **kwargs,
        )
    else:
        # Call CS or CSC path solver
        if goal_dir is not None:
            solutions = solve_csc_path(
                start_x,
                start_y,
                start_dir,
                goal_x,
                goal_y,
                goal_dir,
                airspeed_start_turn,
                turn_rate1,
                wind_speed,
                wind_dir,
                path_type,
                **kwargs,
            )
        else:
            solutions = solve_cs_path(
                start_x,
                start_y,
                start_dir,
                goal_x,
                goal_y,
                airspeed_start_turn,
                turn_rate1,
                wind_speed,
                wind_dir,
                path_type,
            )

    if not solutions:
        return None
    else:
        solution = solutions[0]

    # First segment: trochoid
    omega1 = get_signed_turnrate(turn_rate1, path_type, 0)
    # hdg_at_t0 = vt.chi_A_from_V_A_xy_chi(
    #     airspeed1, start_dir, wind_speed, wind_dir)
    # seg1_path = Trochoid(
    #     start_pos, airspeed1, hdg_at_t0, omega1, wind_speed, wind_dir)
    seg1 = SteadyTurn(
        x_g_t0=start_x,
        y_g_t0=start_y,
        V_A_xy=airspeed_start_turn,
        chi_A_dot=omega1,
        chi_t0=start_dir,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        tf=solution[1],
    )

    # Third segment: trochoid
    seg3 = None
    if len(solution) > 3:
        omega3 = get_signed_turnrate(turn_rate2, path_type, 2)
        seg3 = SteadyTurn.init_from(
            x_g_tf=goal_x,
            y_g_tf=goal_y,
            V_A_xy=airspeed_end_turn,
            chi_A_dot=omega3,
            chi_tf=goal_dir,
            V_W_xy=wind_speed,
            chi_W=wind_dir,
            tf=solution[3] - solution[2],
        )

    # Second segment: linear segment
    seg2 = None
    if len(solution) > 2:
        # Note that the current implementations of solve_xyz_path assume the
        # linear segment to have same constant airspeed as the first segment.
        # Thus, we have to calculate the linear segment end point (t2) based
        # on this constant-speed assumption here.
        x_g_t1 = seg1.xf["x_g"]
        y_g_t1 = seg1.xf["y_g"]
        if len(solution) > 3:
            x_g_t2 = seg3.x0["x_g"]
            y_g_t2 = seg3.x0["y_g"]
        else:
            x_g_t2 = x_g_t1 + seg1.u_K_g(
                solution[1])*(solution[2] - solution[1])
            y_g_t2 = y_g_t1 + seg1.v_K_g(
                solution[1])*(solution[2] - solution[1])
        seg2_length = math.sqrt((x_g_t2 - x_g_t1)**2 + (y_g_t2 - y_g_t1)**2)
        # print("pos at t1: {}".format(pos_at_t1))
        # print("pos at t2: {}".format(pos_at_t2))
        # print("dist: {}".format(pos_at_t2.dist(pos_at_t1)))
        # print("gs: {}".format(seg1_path.velocity(solution[1]).mag()))
        # hdg_at_t1 = util.heading(airspeed2, track_at_t1, wind_speed, wind_dir)
        # if False:
        # NOTE: The asymmetric path (if any) was planned for turn airspeeds,
        # to limit complexity of the solver, however, here we construct
        # the final accelerated linear segment with the caller-given
        # start and stop airspeeds.
        if airspeed_start_acc != airspeed_end_acc:
            # print(
            #     "initializing linear speed straight flight: ",
            #     dict(
            #         x_g_t0=x_g_t1,
            #         y_g_t0=y_g_t1,
            #         l_xy_tf=seg2_length,
            #         V_A_xy_t0=airspeed_start_acc,
            #         V_A_xy_tf=airspeed_end_acc,
            #         chi=states_t1["chi"],
            #         V_W_xy=wind_speed,
            #         chi_W=wind_dir))
            seg2 = LinearSpeedStraightFlight.init_from(
                x_g_t0=x_g_t1,
                y_g_t0=y_g_t1,
                l_xy_tf=seg2_length,
                V_A_xy_t0=airspeed_start_acc,
                V_A_xy_tf=airspeed_end_acc,
                chi=seg1.xf["chi"],
                V_W_xy=wind_speed,
                chi_W=wind_dir,
            )

            # === DEBUG ================================
            # print("seg2.x0: {}".format(seg2.x0))
            # print("seg2.xf: {}".format(seg2.xf))
            # # print("x_g_t1: {}".format(x_g_t1))
            # # print("y_g_t1: {}".format(y_g_t1))
            # # print("x_g_t2: {}".format(x_g_t2))
            # # print("y_g_t2: {}".format(y_g_t2))
            # # ax = seg2.discretize().df.plot(x="y_g", y="x_g")
            # # ax.scatter(x=[y_g_t1, y_g_t2], y=[x_g_t1, x_g_t2])
            # ax = seg2.discretize().df[["V_A_xy"]].plot()
            # ax.scatter(x=[0, seg2.tf], y=[airspeed1, airspeed2])
            # from matplotlib import pyplot as plt
            # plt.show()
            # === DEBUG ================================

            eps = 1e-2
            assert abs(seg2.x0["x_g"] - x_g_t1) < eps
            assert abs(seg2.x0["y_g"] - y_g_t1) < eps
            assert abs(seg2.xf["x_g"] - x_g_t2) < eps
            assert abs(seg2.xf["y_g"] - y_g_t2) < eps
            assert abs(seg2.x0["V_A_xy"] - airspeed_start_acc) < eps
            assert abs(seg2.xf["V_A_xy"] - airspeed_end_acc) < eps
            assert (
                abs(
                    (seg2.xf["V_A_xy"] - seg2.x0["V_A_xy"])/seg2.V_A_xy_dot -
                    seg2.duration) < eps)
        else:
            seg2 = ConstSpeedStraightFlight.init_from(
                x_g_t0=x_g_t1,
                y_g_t0=y_g_t1,
                l_xy_tf=seg2_length,
                V_A_xy=airspeed_start_acc,
                chi=seg1.xf["chi"],
                V_W_xy=wind_speed,
                chi_W=wind_dir,
            )

    segments = [seg1]
    if seg2:
        segments.append(seg2)
    if seg3:
        segments.append(seg3)

    segments = start_round_segments + segments + end_round_segments

    return Trajectory(segments, align_initial_states=True)


def get_trochoid_path_3D(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        wind_speed,
        wind_dir,
        path_type,
        strategy="const. vertical speed",
        **kwargs):
    if isinstance(turn_rate, (tuple, list)):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        turn_rate2 = turn_rate
    if isinstance(airspeed, (tuple, list)):
        airspeed1 = airspeed[0]
        airspeed2 = airspeed[-1]
    else:
        airspeed1 = airspeed
        airspeed2 = airspeed

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")
    else:
        csc = solve_csc_3d_path(
            start_x,
            start_y,
            start_z,
            start_dir,
            goal_x,
            goal_y,
            goal_z,
            goal_dir,
            turn_rate1,
            turn_rate2,
            airspeed1,
            airspeed2,
            wind_speed,
            wind_dir,
            path_type,
            strategy=strategy,
            **kwargs)
        if not csc:
            return None

    # Create path based on CSC solution
    # First segment: trochoid
    seg1 = SteadyTurn(
        x_g_t0=start_x,
        y_g_t0=start_y,
        V_A_xy=csc.v_a_xy_1,
        chi_A_dot=csc.omega1,
        chi_t0=start_dir,
        V_W_xy=wind_speed,
        chi_W=wind_dir,
        tf=csc.t_a,
    )
    horiz_path = Trajectory([seg1])
    # Second segment: linear segment
    if len(path_type) >= 2:
        x_g_t1 = seg1.xf["x_g"]
        y_g_t1 = seg1.xf["y_g"]

        if airspeed1 != airspeed2:
            seg2 = LinearSpeedStraightFlight.init_from(
                x_g_t0=x_g_t1,
                y_g_t0=y_g_t1,
                l_xy_tf=csc.l_beta,
                V_A_xy_t0=csc.v_a_xy_1,
                V_A_xy_tf=csc.v_a_xy_2,
                chi=csc.chi_t_a,
                V_W_xy=wind_speed,
                chi_W=wind_dir,
            )
            assert (numpy.isclose(seg2.tf, csc.t_beta - csc.t_a))
        else:
            seg2 = ConstSpeedStraightFlight.init_from(
                x_g_t0=x_g_t1,
                y_g_t0=y_g_t1,
                l_xy_tf=csc.l_beta,
                V_A_xy=csc.v_a_xy_1,
                chi=csc.chi_t_a,
                V_W_xy=wind_speed,
                chi_W=wind_dir,
            )

        horiz_path.append(seg2)
    # Third segment: trochoid
    if len(path_type) >= 3:
        x_g_t2 = seg2.xf["x_g"]
        y_g_t2 = seg2.xf["y_g"]
        seg3 = SteadyTurn(
            x_g_t0=x_g_t2,
            y_g_t0=y_g_t2,
            V_A_xy=csc.v_a_xy_2,
            chi_A_dot=csc.omega2,
            chi_t0=csc.chi_t_a,
            V_W_xy=wind_speed,
            chi_W=wind_dir,
            tf=csc.t_f - csc.t_beta,
        )
        horiz_path.append(seg3)

    # assert (numpy.isclose(horiz_path.tf, solution.t_final))

    # TODO: implement getVerticalProfile functions directly in the solver class
    # to avoid this type distinction here. Alternative: have all solvers have
    # segmented profile and only "optimize" away the use of Trajectory here if
    # all vertical speeds are the same.
    if isinstance(csc, CSC3D):
        vert_profile = ConstVertSpeedProfile(
            z_g_t0=start_z, z_g_tf=goal_z, w_A_g=csc.w_a_g, tf=horiz_path.tf)
    if isinstance(csc, CSC3D_ConstThrust):
        vert_profile = Trajectory(
            [
                ConstVertSpeedProfile(
                    w_A_g=csc.w_A_g_1, t0=0, tf=csc.t_a, z_g_t0=start_z),
                ConstVertSpeedProfile(
                    w_A_g=csc.w_A_g_2,
                    t0=csc.t_a,
                    tf=csc.t_beta,
                    z_g_t0=csc.z_t1),
                ConstVertSpeedProfile(
                    w_A_g=csc.w_A_g_3,
                    t0=csc.t_beta,
                    tf=horiz_path.tf,
                    z_g_t0=csc.z_t2),
            ])
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)
    return lifted_path


def get_trochoid_path_3D_const_vert_speed(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        turn_n_max,
        wind_speed,
        wind_dir,
        path_type,
        vert_speed_constraint=None,
        high_alt_strategy="full turns",
        speed_eps=1e-3,
        **kwargs):
    """Calculate 3D Trochoid-Dubins paths for uniform wind and with constrained constant vertical speed.

    Limitations:
        - Currently considers only CSC horizontal paths (no CCC paths).
        - Only low-altitude and high-altidude cases are considered (no medium-altitude case).
    """
    if isinstance(airspeed, (tuple, list)):
        airspeed1 = airspeed[0]
        airspeed2 = airspeed[-1]
    else:
        airspeed1 = airspeed
        airspeed2 = airspeed

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")

    if high_alt_strategy == "full turns":
        start_end_round_func = get_full_turn
    elif high_alt_strategy == "periodic paths":
        start_end_round_func = get_trochoid_periodic_path

    logger.push("trochoid_CSC3D_const_vert_speed")

    start_rounds = 0
    end_rounds = 0
    w_A_g_values = logger.track_values("w_A_g", [0.0])

    # Upper bound on |w_A_g|: the horizontal (air-relative) airspeed component
    # V_A_xy = sqrt(V_A^2 - w_A_g^2) must exceed the wind speed to make forward
    # progress: V_A_xy < V_W_xy
    w_A_g_abs_max = min(
        vt.w_A_g_abs_from_V_A_V_A_xy(airspeed1, wind_speed),
        vt.w_A_g_abs_from_V_A_V_A_xy(airspeed2, wind_speed),
    )
    if w_A_g_abs_max < speed_eps:
        # Horizontal path is never feasible (airspeed <= wind speed).
        logger.pop(success=False)
        return None

    def horizontal_path(w_A_g):
        """Closure to return the horizontal path for the given problem at a
        given vertical speed ``w_A_g``.

        The vertical speed changes the horizontal (air-relative) airspeed
        component v_a_xy = sqrt(V_A^2 - w_A_g^2), and -- when ``turn_n_max`` is
        given -- the achievable turn rates via the load-factor bound. Returns
        ``None`` if the horizontal path is infeasible.
        """
        v_a_xy_1 = vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g)
        v_a_xy_2 = vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g)

        tr = turn_rate
        if turn_n_max is not None:
            gamma_A_1 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g)
            gamma_A_2 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_2, w_A_g)
            tr_1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                turn_n_max, airspeed1, gamma_A_1, 0, 9.81)
            tr_2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                turn_n_max, airspeed2, gamma_A_2, 0, 9.81)
            tr = (tr_1, tr_2)
        try:
            return get_trochoid_path_2D(
                start_x, start_y, start_dir, goal_x, goal_y, goal_dir,
                (v_a_xy_1, v_a_xy_2), tr, wind_speed, wind_dir, path_type,
                start_rounds, end_rounds, start_end_round_func)
        except Exception:
            return None

    def fixed_point_map(w_A_g):
        """Raw fixed-point map F(w) = dz / t_f(w), or ``None`` if infeasible."""
        horiz_path = horizontal_path(w_A_g)
        if horiz_path is None:
            return None
        return (goal_z-start_z)/horiz_path.duration

    # Fixed point iteration on w_A_g using Steffensen's method / Aitken acceleration
    # Fixed point map is F(w_A_g) = d(z_g_tf) / t_f(w_A_g)
    # This may give a 2-cycle around the fixed-point as the horizontal path
    # gemoetry reacts to changes in vertical speed via turn rate and airspeed coupling.
    # With Aitken acceleration it converges fast and reliable.
    # Number of loop-body executions. Passes that add a start/end round or
    # reject an out-of-range iterate do not append to w_A_g_values.
    n_passes = 0
    while not utils.converged(w_A_g_values, abs_tol=speed_eps):
        n_passes += 1
        w_A_g = w_A_g_values[-1]

        x1 = fixed_point_map(w_A_g)
        if x1 is None:
            logger.pop(success=False)
            return None

        # If the first iterate already exceeds the feasible vertical-speed range,
        # add start or end round.
        if abs(x1) > w_A_g_abs_max:
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue

        x2 = fixed_point_map(x1)
        if x2 is None:
            # x1 is feasible but the map is ill-conditioned.
            # Treat this like the out-of-range case: add a round to shorten the
            # required path time and retry from the current w_A_g.
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
        violates_constraint = vert_speed_constraint is not None and not vert_speed_constraint(
            new_w_A_g)
        if exceeds_airspeed or violates_constraint:
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue
        else:
            w_A_g_values.append(new_w_A_g)

    logger.pop(
        success=True,
        start_rounds=start_rounds,
        end_rounds=end_rounds,
        iterations=w_A_g_values.n_iterations,
        n_passes=n_passes)

    # Re-evaluate the horizontal path at the converged vertical speed.
    horiz_path = horizontal_path(w_A_g_values[-1])
    if horiz_path is None:
        # The converged point is ill-conditioned; fall back to the last
        # well-posed iterate.
        horiz_path = horizontal_path(w_A_g_values[-2])
    if horiz_path is None:
        logger.pop(success=False)
        return None

    w_A_g = w_A_g_values[-1]

    vert_profile = ConstVertSpeedProfile(
        z_g_t0=start_z, z_g_tf=goal_z, w_A_g=w_A_g, tf=horiz_path.tf)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path


def get_trochoid_path_3D_const_thrust_alt(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        turn_n_max,
        wind_speed,
        wind_dir,
        path_type,
        ac: ExtDubinsAirplane,
        thrust_constraint=None,
        high_alt_strategy="full turns",
        T_eps=1e-3,
        z_eps=1e-1,
        **kwargs):
    """Calculate 3D Trochoid-Dubins paths for uniform wind with constrained constant thrust.

    Limitations:
        - Currently considers only CSC horizontal paths (no CCC paths).
        - Only low-altitude and high-altidude cases are considered (no medium-altitude case).

    TODO: Standardize notation for segment-wise indexed variables. Currently it's a mix of different
          indexing conventions that come from the history and evolution of the algorithm
    """
    # Get initial and final airspeed
    if isinstance(airspeed, (tuple, list)):
        airspeed1 = airspeed[0]
        airspeed2 = airspeed[-1]
    else:
        airspeed1 = airspeed
        airspeed2 = airspeed
    # Get initial and final turn rate
    if isinstance(turn_rate, tuple) or isinstance(turn_rate, list):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        turn_rate2 = turn_rate

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")

    if high_alt_strategy == "full turns":
        start_end_round_func = get_full_turn
    elif high_alt_strategy == "periodic paths":
        start_end_round_func = get_trochoid_periodic_path

    # NOTE: We will have up to 4 vertical speeds and corresponding horizontal
    # airspeeds:
    # - straight on start/end round (periodic path): w_A_g_straight
    # - initial turn of CSC path or start round turn: w_A_g_1
    # - straight segment of CSC path: w_A_g_2
    # - final turn of CSC path or end round turn: w_A_g_3
    # This gives even more (5) horizontal airspeeds:
    # - straight on start round:v_a_xy_1_straight
    # - initial turn of CSC path or start round turn: v_a_xy_1
    # - start of straight segment of CSC path: v_a_xy_20
    # - end of straight segment of CSC path: v_a_xy_21
    # - final turn of CSC path or end round turn: v_a_xy_3
    # - straight on end round: v_a_xy_3_straight
    v_a_xy_1_straight = airspeed1
    v_a_xy_1 = airspeed1
    v_a_xy_20 = airspeed1
    v_a_xy_21 = airspeed2
    v_a_xy_3 = airspeed2
    v_a_xy_3_straight = airspeed2
    w_A_g_1_straight = 0
    w_A_g_1 = 0
    w_A_g_3 = 0
    w_A_g_3_straight = 0
    start_rounds = 0
    end_rounds = 0

    logger.push("trochoid_CSC3D_const_thrust")

    # Thrust bound at which the previous round's root was pinned, or None if
    # the root was interior. Guards the cheap feasibility probe below.
    T_pinned = None

    # Iteratively add start/end rounds until thrust constraint is satisfied
    n_passes = 0
    while True:
        n_passes += 1
        # Thrust bracket for the bisection. If the previous round left the root
        # pinned at a thrust bound, probe that bound first with a degenerate
        # bracket (a single residual evaluation), see the shortcut below. The
        # probe bracket lies just inside the valid thrust range so that no
        # out-of-range thrust is ever evaluated.
        probing = T_pinned is not None
        shortcut_taken = False
        if probing:
            if T_pinned == ac.T_min:
                T_lower = ac.T_min
                T_upper = ac.T_min + 2*T_eps
            else:
                T_lower = ac.T_max - 2*T_eps
                T_upper = ac.T_max
        else:
            T_lower = ac.T_min
            T_upper = ac.T_max
        T_current = T_lower + 0.5*(T_upper-T_lower)

        z_g_tf = 1e12
        logger.push("thrust_iteration")
        n_bisection = 0
        while (T_upper - T_lower) > T_eps or probing:
            n_bisection += 1
            # Relax the coupled airspeed/turn-rate state twice per bisection
            # step so the altitude evaluation is consistent with the geometry
            # it was built from (removes the history dependence of z_g_tf(T)).
            for _ in range(2):
                # Update turn rate if n_max is given
                if turn_n_max is not None:
                    gamma_A_1 = vt.gamma_A_from_V_A_w_A_g(airspeed1, w_A_g_1)
                    gamma_A_2 = vt.gamma_A_from_V_A_w_A_g(airspeed2, w_A_g_3)
                    turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                        turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1),
                        gamma_A_1, 0, 9.81)
                    turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                        turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3),
                        gamma_A_2, 0, 9.81)
                    turn_rate = (turn_rate1, turn_rate2)

                # Update horizontal path
                airspeeds = (
                    v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
                    v_a_xy_3_straight)
                horiz_path = get_trochoid_path_2D(
                    start_x, start_y, start_dir, goal_x, goal_y, goal_dir,
                    airspeeds, turn_rate, wind_speed, wind_dir, path_type,
                    start_rounds, end_rounds, start_end_round_func)
                if horiz_path is None:
                    logger.pop(success=False, iterations=n_bisection)
                    logger.pop(success=False, iterations=n_passes)
                    return None

                # Calculate airspeed rate via midsegment duration
                straight_segment_duration = horiz_path.segments[start_rounds +
                                                                1].duration
                V_A_xy_dot_2 = (v_a_xy_21-v_a_xy_20)/straight_segment_duration

                # Calculate vertical speeds from current thrust
                w_A_g_1_straight = ac.vert_speed(
                    V_A=airspeed1, T=T_current, gamma_A_dot=0)
                w_A_g_1 = ac.vert_speed(
                    V_A=airspeed1,
                    T=T_current,
                    chi_A_dot=turn_rate1,
                    gamma_A_dot=0)
                w_A_g_20 = ac.vert_speed(
                    V_A=airspeed1,
                    T=T_current,
                    V_A_dot=V_A_xy_dot_2,
                    gamma_A_dot=0)
                w_A_g_21 = ac.vert_speed(
                    V_A=airspeed2,
                    T=T_current,
                    V_A_dot=V_A_xy_dot_2,
                    gamma_A_dot=0)
                w_A_g_3 = ac.vert_speed(
                    V_A=airspeed2,
                    T=T_current,
                    chi_A_dot=turn_rate2,
                    gamma_A_dot=0)
                w_A_g_3_straight = ac.vert_speed(
                    V_A=airspeed2, T=T_current, gamma_A_dot=0)

                w_A_g_1_straight = utils.clip_abs(w_A_g_1_straight, airspeed1)
                w_A_g_1 = utils.clip_abs(w_A_g_1, airspeed1)
                w_A_g_20 = utils.clip_abs(w_A_g_20, airspeed1)
                w_A_g_21 = utils.clip_abs(w_A_g_21, airspeed2)
                w_A_g_3 = utils.clip_abs(w_A_g_3, airspeed2)
                w_A_g_3_straight = utils.clip_abs(w_A_g_3_straight, airspeed2)

                # Update horizontal airspeeds
                v_a_xy_1_straight = vt.V_A_xy_from_V_A_w_A_g(
                    airspeed1, w_A_g_1_straight)
                v_a_xy_1 = vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g_1)
                v_a_xy_20 = vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g_20)
                v_a_xy_21 = vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g_21)
                v_a_xy_3 = vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g_3)
                v_a_xy_3_straight = vt.V_A_xy_from_V_A_w_A_g(
                    airspeed2, w_A_g_3_straight)

                v_a_xy_1_straight = utils.clip(v_a_xy_1_straight, wind_speed)
                v_a_xy_1 = utils.clip(v_a_xy_1, wind_speed)
                v_a_xy_20 = utils.clip(v_a_xy_20, wind_speed)
                v_a_xy_21 = utils.clip(v_a_xy_21, wind_speed)
                v_a_xy_3 = utils.clip(v_a_xy_3, wind_speed)
                v_a_xy_3_straight = utils.clip(v_a_xy_3_straight, wind_speed)

            # Update final altitude
            z_g_tf = start_z
            for _ in range(start_rounds):
                if high_alt_strategy == "full turns":
                    z_g_tf += w_A_g_1*horiz_path.segments[0].duration
                else:
                    for seg in horiz_path.segments[0].segments:
                        if seg.x0["chi_A_dot"] != 0:
                            z_g_tf += w_A_g_1*seg.duration
                        else:
                            z_g_tf += w_A_g_1_straight*seg.duration
            for _ in range(end_rounds):
                if high_alt_strategy == "full turns":
                    z_g_tf += w_A_g_3*horiz_path.segments[-1].duration
                else:
                    for seg in horiz_path.segments[-1].segments:
                        if seg.x0["chi_A_dot"] != 0:
                            z_g_tf += w_A_g_3*seg.duration
                        else:
                            z_g_tf += w_A_g_3_straight*seg.duration
            z_g_tf += w_A_g_1*horiz_path.segments[start_rounds].duration
            z_g_tf += 0.5*(w_A_g_20+w_A_g_21)*horiz_path.segments[start_rounds +
                                                                  1].duration
            z_g_tf += w_A_g_3*horiz_path.segments[start_rounds + 2].duration

            # Update thrust bounds and thrust estimate
            if z_g_tf < goal_z:
                # Too high, decrease thrust
                T_upper = T_current
            else:
                # Too low, increase thrust
                T_lower = T_current
            T_current = T_lower + 0.5*(T_upper-T_lower)

            # Trace the bisection: altitude residual and bracket width
            logger.add_value(
                "z_g_tf_residual",
                abs(z_g_tf - goal_z),
                T_current=T_current,
                T_bracket=T_upper - T_lower)

            # print(
            #     f"dz={z_g_tf - goal_z:.3f}, z_step={z_g_tf - z_g_tf_before:.3f}, {T_lower:.3f} <= {T_current:.3f} <= {T_upper:.3f}"
            # )

            if probing and (T_upper - T_lower) <= T_eps:
                # The reachable final altitude is monotone in thrust, so if the
                # pinned bound still falls clearly short of the target, no
                # thrust in [T_min, T_max] can reach it: accept the shortcut
                # and let the caller add another start/end round. The margin
                # keeps the decision identical to the full bisection, which
                # rejects the round whenever the residual exceeds 1e-1.
                shortcut = abs(z_g_tf - goal_z) > 2e-1 and (
                    (T_pinned == ac.T_min and z_g_tf < goal_z) or
                    (T_pinned == ac.T_max and z_g_tf > goal_z))
                if shortcut:
                    shortcut_taken = True
                    break
                # Otherwise the root moved inside the bracket (the added round
                # changed the geometry), so bisect the full thrust range.
                probing = False
                T_lower = ac.T_min
                T_upper = ac.T_max
                T_current = T_lower + 0.5*(T_upper-T_lower)

        # Remember whether the root is pinned at a thrust bound so that the next
        # round can probe that bound directly instead of bisecting. When the
        # shortcut fired, T_pinned still holds the bound that was probed.
        if not shortcut_taken:
            if abs(T_current - ac.T_min) <= 3*T_eps:
                T_pinned = ac.T_min
            elif abs(T_current - ac.T_max) <= 3*T_eps:
                T_pinned = ac.T_max
            else:
                T_pinned = None

        logger.pop(
            success=True, iterations=n_bisection, shortcut=shortcut_taken)

        # Check if target altitude was reached (within z_g delta corresponding to T_eps stepwidth)
        # if abs(z_g_tf - goal_z) > abs(z_g_tf - z_g_tf_before):
        if abs(z_g_tf - goal_z) > z_eps:
            altitude_reached = False
        else:
            altitude_reached = True

        if not altitude_reached or thrust_constraint is not None and not thrust_constraint(
                T_current):
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            # print(
            #     f"{start_rounds=}, {end_rounds=}, {T_current=:.3f} violates constraint"
            # )
            continue
        else:
            break

    # Post-convergence consistency pass: re-build the horizontal path based on
    # final converged vertical speed to ensure satisfaction of load factor constraint
    # with sufficient accuracy.
    if turn_n_max is not None:
        gamma_A_1 = vt.gamma_A_from_V_A_w_A_g(airspeed1, w_A_g_1)
        gamma_A_2 = vt.gamma_A_from_V_A_w_A_g(airspeed2, w_A_g_3)
        turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1), gamma_A_1,
            0, 9.81)
        turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3), gamma_A_2,
            0, 9.81)
        turn_rate = (turn_rate1, turn_rate2)
        airspeeds = (
            v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
            v_a_xy_3_straight)
        horiz_path = get_trochoid_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeeds,
            turn_rate, wind_speed, wind_dir, path_type, start_rounds,
            end_rounds, start_end_round_func)
        if horiz_path is None:
            logger.pop(success=False)
            return None

    logger.pop(
        success=altitude_reached,
        start_rounds=start_rounds,
        end_rounds=end_rounds,
        n_passes=n_passes)

    # Construct vertical profile
    vert_prof_segments = []
    if start_rounds:
        start_round = horiz_path.segments[0]
        if high_alt_strategy == "full turns":
            start_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=start_z,
                w_A_g=w_A_g_1,
                tf=start_round.duration,
            )
        else:
            # Note that initial states, i.e. start altitude and time, will be
            # aligned auomatically.
            start_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1,
                        tf=start_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1_straight,
                        tf=start_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1,
                        tf=start_round.segments[2].duration,
                    ),
                ])
    if end_rounds:
        end_round = horiz_path.segments[-1]

        if high_alt_strategy == "full turns":
            end_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=start_z,
                w_A_g=w_A_g_3,
                tf=end_round.duration,
            )
        else:
            # Note that initial states, i.e. start altitude and time, will be
            # aligned auomatically later when we construct the final vertical profile
            end_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3_straight,
                        tf=end_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[2].duration,
                    ),
                ])
    csc_horiz_path = horiz_path.segments[start_rounds:start_rounds + 3]
    csc_vert_prof = Trajectory(
        [
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_1, tf=csc_horiz_path[0].duration),
            # ConstVertSpeedProfile(
            #     z_g_t0=start_z, w_A_g=w_A_g_2, tf=csc_horiz_path[1].duration),
            LinearVertSpeedProfile.init_from(
                z_g_t0=start_z,
                w_A_g_t0=w_A_g_20,
                w_A_g_tf=w_A_g_21,
                tf=csc_horiz_path[1].duration),
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_3, tf=csc_horiz_path[2].duration),
        ])
    vert_prof_segments = []
    for _ in range(start_rounds):
        vert_prof_segments.append(copy.deepcopy(start_round_vert_prof))
    vert_prof_segments.append(csc_vert_prof)
    for _ in range(end_rounds):
        vert_prof_segments.append(copy.deepcopy(end_round_vert_prof))
    vert_profile = Trajectory(vert_prof_segments)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path


def get_trochoid_path_3D_const_thrust(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        turn_n_max,
        wind_speed,
        wind_dir,
        path_type,
        ac: ExtDubinsAirplane,
        thrust_constraint=None,
        high_alt_strategy="full turns",
        T_eps=1e-3,
        speed_eps=1e-6,
        z_eps=1e-1,
        **kwargs):
    """Calculate 3D Trochoid-Dubins paths with constrained constant thrust.

    More robust variant of :func:`get_trochoid_path_3D_const_thrust_alt`.

    The original solver evaluates the altitude residual ``z_g_tf(T)`` while
    mutating the coupled airspeed/turn-rate state inside the bisection loop.
    That makes ``z_g_tf`` history-dependent (non-monotone), so the bisection can
    stop on a spurious root and request unnecessary start/end rounds, leaving
    large thrust slack.

    Here the map is made pure: the horizontal geometry (and its airspeed state)
    is frozen, ``z_g_tf(T)`` is bisected exactly, and the frozen state is then
    updated once. Repeating this outer fixed point converges to the consistent
    (settled) solution, and the pass/round decision is always made against the
    settled state.
    """
    # Get initial and final airspeed
    if isinstance(airspeed, (tuple, list)):
        airspeed1 = airspeed[0]
        airspeed2 = airspeed[-1]
    else:
        airspeed1 = airspeed
        airspeed2 = airspeed
    # Get initial and final turn rate
    if isinstance(turn_rate, (tuple, list)):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        turn_rate2 = turn_rate

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")

    if high_alt_strategy == "full turns":
        start_end_round_func = get_full_turn
    elif high_alt_strategy == "periodic paths":
        start_end_round_func = get_trochoid_periodic_path

    def update_horiz_path(new_airspeeds, new_turn_rates):
        """Horizontal path for the given horizontal-airspeed/turn-rate state."""
        return get_trochoid_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir,
            new_airspeeds, new_turn_rates, wind_speed, wind_dir, path_type,
            start_rounds, end_rounds, start_end_round_func)

    def int_final_altitude(horiz_path_, w1s, w1, w20, w21, w3, w3s):
        """Integrated final altitude for the given vertical-speed state."""
        z_g_tf_ = start_z
        for _ in range(start_rounds):
            if high_alt_strategy == "full turns":
                z_g_tf_ += w1*horiz_path_.segments[0].duration
            else:
                for seg in horiz_path_.segments[0].segments:
                    if seg.x0["chi_A_dot"] != 0:
                        z_g_tf_ += w1*seg.duration
                    else:
                        z_g_tf_ += w1s*seg.duration
        for _ in range(end_rounds):
            if high_alt_strategy == "full turns":
                z_g_tf_ += w3*horiz_path_.segments[-1].duration
            else:
                for seg in horiz_path_.segments[-1].segments:
                    if seg.x0["chi_A_dot"] != 0:
                        z_g_tf_ += w3*seg.duration
                    else:
                        z_g_tf_ += w3s*seg.duration
        z_g_tf_ += w1*horiz_path_.segments[start_rounds].duration
        z_g_tf_ += 0.5*(w20+w21)*horiz_path_.segments[start_rounds + 1].duration
        z_g_tf_ += w3*horiz_path_.segments[start_rounds + 2].duration
        return z_g_tf_

    # State: six horizontal airspeeds and matching vertical speeds.
    v_a_xy_1_straight = v_a_xy_1 = v_a_xy_20 = airspeed1
    v_a_xy_21 = v_a_xy_3 = v_a_xy_3_straight = airspeed2
    w_A_g_1_straight = w_A_g_1 = w_A_g_20 = 0
    w_A_g_21 = w_A_g_3 = w_A_g_3_straight = 0
    start_rounds = 0
    end_rounds = 0

    logger.push("trochoid_CSC3D_const_thrust_v2")

    max_settle = 30
    n_passes = 0
    while True:
        n_passes += 1
        logger.push("thrust_iteration")

        # Outer fixed point: freeze the geometry, bisect T exactly on the
        # resulting pure z_g_tf(T), then update the frozen state. Repeats until
        # the geometry stops moving.
        T_current = ac.T_min + 0.5*(ac.T_max - ac.T_min)
        horiz_path = None
        n_settle = 0
        while n_settle < max_settle:
            n_settle += 1
            # Freeze geometry from the current state.
            if turn_n_max is not None:
                turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                    turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1),
                    vt.gamma_A_from_V_A_w_A_g(airspeed1, w_A_g_1), 0, 9.81)
                turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                    turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3),
                    vt.gamma_A_from_V_A_w_A_g(airspeed2, w_A_g_3), 0, 9.81)
                turn_rate = (turn_rate1, turn_rate2)
            airspeeds = (
                v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
                v_a_xy_3_straight)
            horiz_path = update_horiz_path(airspeeds, turn_rate)
            if horiz_path is None:
                logger.pop(success=False, iterations=n_settle)
                logger.pop(success=False, iterations=n_passes)
                return None
            straight_segment_duration = \
                horiz_path.segments[start_rounds + 1].duration
            V_A_xy_dot_2 = (v_a_xy_21-v_a_xy_20)/straight_segment_duration

            def final_alt_from_thrust(T):
                """Map from thrust to final altitude on fixed horizontal path geometry."""
                w1s = utils.clip_abs(
                    ac.vert_speed(V_A=airspeed1, T=T, gamma_A_dot=0), airspeed1)
                w1 = utils.clip_abs(
                    ac.vert_speed(
                        V_A=airspeed1, T=T, chi_A_dot=turn_rate1,
                        gamma_A_dot=0), airspeed1)
                w20 = utils.clip_abs(
                    ac.vert_speed(
                        V_A=airspeed1, T=T, V_A_dot=V_A_xy_dot_2,
                        gamma_A_dot=0), airspeed1)
                w21 = utils.clip_abs(
                    ac.vert_speed(
                        V_A=airspeed2, T=T, V_A_dot=V_A_xy_dot_2,
                        gamma_A_dot=0), airspeed2)
                w3 = utils.clip_abs(
                    ac.vert_speed(
                        V_A=airspeed2, T=T, chi_A_dot=turn_rate2,
                        gamma_A_dot=0), airspeed2)
                w3s = utils.clip_abs(
                    ac.vert_speed(V_A=airspeed2, T=T, gamma_A_dot=0), airspeed2)
                return int_final_altitude(
                    horiz_path, w1s, w1, w20, w21, w3, w3s)

            # Exact bisection on the frozen (pure, monotone) map.
            T_lower = ac.T_min
            T_upper = ac.T_max
            T_current = T_lower + 0.5*(T_upper-T_lower)
            residual = final_alt_from_thrust(T_current) - goal_z
            while (T_upper - T_lower) > T_eps:
                if residual < 0:
                    T_upper = T_current
                else:
                    T_lower = T_current
                T_current = T_lower + 0.5*(T_upper-T_lower)
                residual = final_alt_from_thrust(T_current) - goal_z
                logger.add_value(
                    "z_g_tf_residual",
                    abs(residual),
                    T_current=T_current,
                    T_bracket=T_upper - T_lower)

            # Update the vertical speeds and airspeeds with the new thrust
            w_A_g_1_straight = utils.clip_abs(
                ac.vert_speed(V_A=airspeed1, T=T_current, gamma_A_dot=0),
                airspeed1)
            w_A_g_1 = utils.clip_abs(
                ac.vert_speed(
                    V_A=airspeed1,
                    T=T_current,
                    chi_A_dot=turn_rate1,
                    gamma_A_dot=0), airspeed1)
            w_A_g_20 = utils.clip_abs(
                ac.vert_speed(
                    V_A=airspeed1,
                    T=T_current,
                    V_A_dot=V_A_xy_dot_2,
                    gamma_A_dot=0), airspeed1)
            w_A_g_21 = utils.clip_abs(
                ac.vert_speed(
                    V_A=airspeed2,
                    T=T_current,
                    V_A_dot=V_A_xy_dot_2,
                    gamma_A_dot=0), airspeed2)
            w_A_g_3 = utils.clip_abs(
                ac.vert_speed(
                    V_A=airspeed2,
                    T=T_current,
                    chi_A_dot=turn_rate2,
                    gamma_A_dot=0), airspeed2)
            w_A_g_3_straight = utils.clip_abs(
                ac.vert_speed(V_A=airspeed2, T=T_current, gamma_A_dot=0),
                airspeed2)

            new_airspeeds = (
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g_1_straight),
                    wind_speed),
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g_1), wind_speed),
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed1, w_A_g_20), wind_speed),
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g_21), wind_speed),
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g_3), wind_speed),
                utils.clip(
                    vt.V_A_xy_from_V_A_w_A_g(airspeed2, w_A_g_3_straight),
                    wind_speed))
            (
                v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
                v_a_xy_3_straight) = new_airspeeds

            # Stop once the frozen geometry no longer moves.
            if all(abs(a - b) < speed_eps
                   for a, b in zip(airspeeds, new_airspeeds)):
                break

        # Residual on the settled state (pure map value at the solved T).
        z_g_tf = int_final_altitude(
            horiz_path, w_A_g_1_straight, w_A_g_1, w_A_g_20, w_A_g_21, w_A_g_3,
            w_A_g_3_straight)
        logger.pop(success=True, iterations=n_settle)

        altitude_reached = abs(z_g_tf - goal_z) <= z_eps
        if not altitude_reached or thrust_constraint is not None and not \
                thrust_constraint(T_current):
            if goal_z < start_z:
                start_rounds += 1
            else:
                end_rounds += 1
            continue
        else:
            break

    # Post-convergence consistency pass: re-build the horizontal path based on
    # final converged vertical speed to ensure satisfaction of load factor
    # constraint with sufficient accuracy.
    if turn_n_max is not None:
        turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1),
            vt.gamma_A_from_V_A_w_A_g(airspeed1, w_A_g_1), 0, 9.81)
        turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3),
            vt.gamma_A_from_V_A_w_A_g(airspeed2, w_A_g_3), 0, 9.81)
        turn_rate = (turn_rate1, turn_rate2)
        airspeeds = (
            v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
            v_a_xy_3_straight)
        horiz_path = update_horiz_path(airspeeds, turn_rate)
        if horiz_path is None:
            logger.pop(success=False)
            return None

    logger.pop(
        success=altitude_reached,
        start_rounds=start_rounds,
        end_rounds=end_rounds,
        n_passes=n_passes)

    # Construct vertical profile
    vert_prof_segments = []
    if start_rounds:
        start_round = horiz_path.segments[0]
        if high_alt_strategy == "full turns":
            start_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=start_z,
                w_A_g=w_A_g_1,
                tf=start_round.duration,
            )
        else:
            start_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1,
                        tf=start_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1_straight,
                        tf=start_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_1,
                        tf=start_round.segments[2].duration,
                    ),
                ])
    if end_rounds:
        end_round = horiz_path.segments[-1]

        if high_alt_strategy == "full turns":
            end_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=start_z,
                w_A_g=w_A_g_3,
                tf=end_round.duration,
            )
        else:
            end_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3_straight,
                        tf=end_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=start_z,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[2].duration,
                    ),
                ])
    csc_horiz_path = horiz_path.segments[start_rounds:start_rounds + 3]
    csc_vert_prof = Trajectory(
        [
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_1, tf=csc_horiz_path[0].duration),
            LinearVertSpeedProfile.init_from(
                z_g_t0=start_z,
                w_A_g_t0=w_A_g_20,
                w_A_g_tf=w_A_g_21,
                tf=csc_horiz_path[1].duration),
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_3, tf=csc_horiz_path[2].duration),
        ])
    vert_prof_segments = []
    for _ in range(start_rounds):
        vert_prof_segments.append(copy.deepcopy(start_round_vert_prof))
    vert_prof_segments.append(csc_vert_prof)
    for _ in range(end_rounds):
        vert_prof_segments.append(copy.deepcopy(end_round_vert_prof))
    vert_profile = Trajectory(vert_prof_segments)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path


def get_trochoid_path_3D_longitudinal_transition_alt(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        turn_n_max,
        wind_speed,
        wind_dir,
        path_type,
        ac: ExtDubinsAirplane,
        transition_strategy="rapid_descent_fixed_length",
        V_A_eps=1e-3,
        gamma_A_eps=1e-5,
        z_eps=1e-2,
        damping_factor=0.02,
        high_alt_strategy="full turns",
        **kwargs):
    """Calculate 3D Trochoid-Dubins paths with longitudinal transition over the straight segment.

    Limitations:
        - Currently considers only CSC horizontal paths (no CCC paths).
        - Only low-altitude and high-altidude cases are considered (no medium-altitude case).

    NOTE: This has been deprecated in favor of the more stable and efficient variant
    implemented in `get_trochoid_path_3D_longitudinal_transition` that does not require damping.

    NOTE: The binary search over gamma_A is limited by the minimal step size gamma_A_eps. However, the resulting resolution may
    also be affected by the minimal airspeed step size (V_A_eps) of the inner-loop function to calculate the longitduinal maneuver over the
    straight path segment (LongitudinalManeuver.solve_rapid_descent_fixed_length).

    TODO: Standardize notation for segment-wise indexed variables. Currently it's a mix of different
          indexing conventions that come from the history and evolution of the algorithm

    """
    if isinstance(turn_rate, tuple) or isinstance(turn_rate, list):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        turn_rate2 = turn_rate

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")

    if high_alt_strategy == "full turns":
        start_end_round_func = get_full_turn
    elif high_alt_strategy == "periodic paths":
        start_end_round_func = get_trochoid_periodic_path
    else:
        raise ValueError(f"Unknown high_alt_strategy: {high_alt_strategy}")

    # NOTE: We will have up to 4 vertical speeds and corresponding horizontal
    # airspeeds:
    # - straight on start/end round (periodic path): w_A_g_straight
    # - initial turn of CSC path or start round turn: w_A_g_1
    # - straight segment of CSC path: w_A_g_2
    # - final turn of CSC path or end round turn: w_A_g_3
    # This gives even more (5) horizontal airspeeds:
    # - straight on start round:v_a_xy_1_straight
    # - initial turn of CSC path or start round turn: v_a_xy_1
    # - start of straight segment of CSC path: v_a_xy_20
    # - end of straight segment of CSC path: v_a_xy_21
    # - final turn of CSC path or end round turn: v_a_xy_3
    # - straight on end round: v_a_xy_3_straight
    v_a_xy_1_straight = airspeed
    v_a_xy_1 = airspeed
    v_a_xy_20 = airspeed
    v_a_xy_21 = airspeed
    v_a_xy_3 = airspeed
    v_a_xy_3_straight = airspeed
    w_A_g_1 = 0
    w_A_g_20 = 0
    w_A_g_21 = 0
    w_A_g_3 = 0
    w_A_g_3_straight = 0
    end_rounds = 0

    if turn_n_max is not None:
        gamma_A_1 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1)
        turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1), gamma_A_1,
            0, 9.81)

    T_1 = ac.thrust_required(
        V_A=vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1),
        w_A_g=w_A_g_1,
        chi_A_dot=turn_rate1)

    # Iteratively adjust longitudinal maneuver and/or add end rounds until final altitude is reached
    T_min_2 = ac.T_min  # Minimal thrust over longitudinal transition segment
    T_max_2 = T_1  # Maximal thrust over longitudinal transition segment
    gamma_A_21_min = -math.radians(60)

    logger.push("start_end_rounds_iteration")

    stage = "add end rounds"
    outer_iterations = 0
    max_outer_iterations = 200
    while True:
        logger.push("final_airspeed_iteration")

        outer_iterations += 1
        if outer_iterations > max_outer_iterations:
            logger.pop(success=False)
            logger.pop(success=False)
            return None

        # Reset horizontal and vertical airspeed components after the longitudinal transition
        w_A_g_21 = 0
        w_A_g_3 = 0
        w_A_g_3_straight = 0
        v_a_xy_21 = airspeed
        v_a_xy_3 = airspeed
        v_a_xy_3_straight = airspeed

        z_g_tf_values = logger.track_values("z_g_tf")
        while not utils.converged(z_g_tf_values, abs_tol=z_eps):
            # Update turn rate if n_max is given
            gamma_A_2 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3)
            if turn_n_max is not None:
                turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                    turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3),
                    gamma_A_2, 0, ac.g)
            turn_rate = (turn_rate1, turn_rate2)

            # Update horizontal path with new airspeeds unless we are in the
            # "finalize final altitude" stage. Here, we cannot risk bifurcation
            # of the function z_g_tf(gamma_A_21) through updating airspeeds
            # which may cause the horizontal path geometry to change.
            airspeeds = (
                v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
                v_a_xy_3_straight)
            if not stage == "finalize final altitude":
                logger.push("get_trochoid_path_2D")
                horiz_path = get_trochoid_path_2D(
                    start_x, start_y, start_dir, goal_x, goal_y, goal_dir,
                    airspeeds, turn_rate, wind_speed, wind_dir, path_type, 0,
                    end_rounds, start_end_round_func)
                logger.pop(success=True)
            if horiz_path is None:
                logger.pop(success=False)
                logger.pop(success=False)
                return None

            straight_segment_length = horiz_path.segments[1].xf[
                "l_xy"] - horiz_path.segments[1].x0["l_xy"]

            z_g_20 = start_z + w_A_g_1*horiz_path.segments[0].duration

            if transition_strategy == "rapid_descent_fixed_length":

                # Calculate thrust delta from turn to straight segment and use it
                # for accelerating over the straight segment
                T_max_2 = T_1

                # First stage: final airspeed is calculated iteratively based on
                # T_max at start and T_min at end of rapid descent transition
                # governed by gamma_A_min constraint.
                # if inner_stage == "free final airspeed":
                if not z_g_tf_values:
                    logger.push("solve_rapid_descent_fixed_length")
                    lon_man = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                        ac,
                        v_a_xy_20,
                        w_A_g_20,
                        z_g_20,
                        horiz_path.segments[1].chi,
                        wind_speed,
                        wind_dir,
                        straight_segment_length,
                        T_min=T_min_2,
                        T_max=T_max_2,
                        gamma_A_min=gamma_A_21_min,
                        V_A_eps=V_A_eps)
                    logger.pop(success=True)
                # Second stage: final airspeed from previous iteration is used
                # with new segment length that reflects changes in the horizontal
                # path due final airspeed from last iteration / first stage.
                else:
                    logger.push(
                        "solve_rapid_descent_fixed_length_fixed_airspeed")
                    lon_man = LongitudinalManeuver.solve_rapid_descent_fixed_length_fixed_airspeed(
                        ac,
                        v_a_xy_20,
                        w_A_g_20,
                        z_g_20,
                        horiz_path.segments[1].chi,
                        wind_speed,
                        wind_dir,
                        straight_segment_length,
                        T_min=T_min_2,
                        gamma_A_min=gamma_A_21_min,
                        V_A_eps=V_A_eps,
                        V_A_xy_tf=v_a_xy_21)
                    logger.pop(success=True)

                # Replace straight segment of horizontal path
                new_straight_segment = LinearSpeedStraightFlight.init_from(
                    x_g_t0=horiz_path.segments[1].x_g_t0,
                    y_g_t0=horiz_path.segments[1].y_g_t0,
                    l_xy_tf=straight_segment_length,
                    V_A_xy_t0=lon_man.V_A_xy_t0,
                    V_A_xy_tf=lon_man.V_A_xy_tf,
                    chi=horiz_path.segments[1].chi,
                    V_W_xy=wind_speed,
                    chi_W=wind_dir,
                    t0=horiz_path.segments[0].tf)
                horiz_path.segments[1] = new_straight_segment
                horiz_path.alignSegments(align_initial_states=False)

            else:
                raise ValueError(
                    f"Transition strategy not supported: {transition_strategy}")

            # Update vertical speeds after the longitudinal transition maneuver
            # to maintain final vertical speed
            w_A_g_21 = lon_man.w_A_g_tf

            w_A_g_3 = lon_man.w_A_g_tf
            w_A_g_3_straight = lon_man.w_A_g_tf

            # Update horizontal airspeeds
            v_a_xy_21 = lon_man.V_A_xy_tf
            v_a_xy_3 = vt.V_A_xy_from_V_A_w_A_g(lon_man.V_A_tf, w_A_g_3)
            v_a_xy_3_straight = vt.V_A_xy_from_V_A_w_A_g(
                lon_man.V_A_tf, w_A_g_3_straight)

            v_a_xy_21 = utils.clip(v_a_xy_21, wind_speed)
            v_a_xy_3 = utils.clip(v_a_xy_3, wind_speed)
            v_a_xy_3_straight = utils.clip(v_a_xy_3_straight, wind_speed)

            # Update final altitude
            z_g_21 = lon_man.z_g_tf
            z_g_31 = z_g_21 + w_A_g_3*horiz_path.segments[2].duration
            z_g_tf = z_g_31
            for _ in range(end_rounds):
                if high_alt_strategy == "full turns":
                    z_g_tf += w_A_g_3*horiz_path.segments[-1].duration
                elif high_alt_strategy == "periodic paths":
                    for seg in horiz_path.segments[-1].segments:
                        if seg.x0["chi_A_dot"] != 0:
                            z_g_tf += w_A_g_3*seg.duration
                        else:
                            z_g_tf += w_A_g_3_straight*seg.duration
                else:
                    raise ValueError(
                        f"High altitude strategy not supported: {high_alt_strategy}"
                    )
            # DEBUG_OUTPUT = True
            if DEBUG_OUTPUT:
                print(
                    f"z_g_tf iteration {len(z_g_tf_values)}: {z_g_20=:.3f} {z_g_21=:.3f} {z_g_tf=:.3f} | "
                    f"{v_a_xy_21=:.3f} {w_A_g_21=:.3f} | "
                    f"{v_a_xy_3=:.3f} {w_A_g_3=:.3f} | "
                    f"{straight_segment_length=:.3f} | "
                    f"{gamma_A_21_min=:.6f} {gamma_A_2=:.6f} gamma_A_diff={gamma_A_21_min-gamma_A_2:.6f}"
                )
            z_g_tf_values.append(z_g_tf)

        logger.pop(iterations=z_g_tf_values.n_iterations, success=True)

        # Early exit for free final altitude problems?
        if goal_z is None:
            logger.pop(success=True)
            stage = "done"
            break

        if stage == "add end rounds":
            # print(f"stage: add end rounds -> {end_rounds=}")
            if z_g_tf is None:
                stage = "done"
                break
            if z_g_tf < goal_z:
                # Too high, add end rounds to lose more altitude
                end_rounds += 1
                # print(
                #     f"add end rounds: {z_g_tf=:.3f} < {goal_z=:.3f} | {end_rounds=}"
                # )
            else:
                # print(
                #     f"add end rounds finished: {z_g_tf=:.3f} >= {goal_z=:.3f} | {end_rounds=} | {gamma_A_21_min=:.3f}"
                # )
                gamma_A_21_min_upper = 0
                gamma_A_21_min_lower = lon_man.gamma_A_tf
                gamma_A_damping_term = damping_factor*(
                    gamma_A_21_min_upper-gamma_A_21_min_lower)
                gamma_A_21_min_lower -= gamma_A_damping_term
                gamma_A_21_min = lon_man.gamma_A_tf
                gamma_A_21_min_values = []
                stage = "optimize final altitude"

                logger.pop(end_rounds=end_rounds, success=True)
                logger.push("optimize_final_altitude")

        elif stage == "optimize final altitude" or stage == "finalize final altitude":
            # DEBUG_OUTPUT = True
            gamma_A_damping_term = damping_factor*(
                gamma_A_21_min_upper-gamma_A_21_min_lower)
            if z_g_tf < goal_z:
                if DEBUG_OUTPUT:
                    print(
                        f"optimize final altitude: {z_g_20=:.3f} {z_g_21=:.3f} {z_g_tf=:.3f} < {goal_z=:.3f} | {v_a_xy_20=:.3f} {v_a_xy_21=:.3f} | {w_A_g_20=:.3f} {w_A_g_21=:.3f} {w_A_g_3=:.3f} | "
                        f"{gamma_A_21_min_lower=:.7f} <= {gamma_A_21_min=:.7f} <= {gamma_A_21_min_upper=:.7f}"
                        f" -> high, decrease gamma_A_21_min")
                # Too high, decrease gamma_A_21_min
                gamma_A_21_min_upper = min(
                    gamma_A_21_min_upper, gamma_A_21_min + gamma_A_damping_term)
                gamma_A_21_min = gamma_A_21_min_lower + 0.5*(
                    gamma_A_21_min_upper-gamma_A_21_min_lower)
            else:
                if DEBUG_OUTPUT:
                    print(
                        f"optimize final altitude: {z_g_20=:.3f} {z_g_21=:.3f} {z_g_tf=:.3f} > {goal_z=:.3f} | {v_a_xy_20=:.3f} {v_a_xy_21=:.3f} | {w_A_g_20=:.3f} {w_A_g_21=:.3f} {w_A_g_3=:.3f} | "
                        f"{gamma_A_21_min_lower=:.6f} <= {gamma_A_21_min=:.6f} <= {gamma_A_21_min_upper=:.6f}"
                        f" -> low, increase gamma_A_21_min")
                # Too low, increase gamma_A_21_min
                gamma_A_21_min_lower = max(
                    gamma_A_21_min_lower, gamma_A_21_min - gamma_A_damping_term)
                gamma_A_21_min = gamma_A_21_min + 0.5*(
                    gamma_A_21_min_upper-gamma_A_21_min)
            gamma_A_21_min_values.append(gamma_A_21_min)

            if abs(z_g_tf - goal_z) < z_eps:
                stage = "done"
            if abs(gamma_A_21_min_upper - gamma_A_21_min_lower) <= gamma_A_eps:
                stage = "finalize final altitude"
                gamma_A_21_min_upper = 0
                gamma_A_21_min_lower = lon_man.gamma_A_tf
        else:
            raise ValueError("Invalid stage")

        # Running another iteration may overshoot the solution
        if stage == "done":
            logger.pop(iterations=len(gamma_A_21_min_values), success=True)
            break

    # Construct vertical profile
    vert_prof_segments = []
    if end_rounds:
        end_round = horiz_path.segments[-1]
        if high_alt_strategy == "full turns":
            end_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=z_g_31,
                w_A_g=w_A_g_3,
                tf=end_round.duration,
            )
        elif high_alt_strategy == "periodic paths":
            # Note that initial states, i.e. start altitude and time, will be
            # aligned auomatically later when we construct the final vertical profile
            end_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3_straight,
                        tf=end_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[2].duration,
                    ),
                ])
        else:
            raise ValueError(
                f"High altitude strategy not supported: {high_alt_strategy}")
    csc_horiz_path = horiz_path.segments[0:3]
    csc_vert_prof = Trajectory(
        [
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_1, tf=csc_horiz_path[0].duration),
            # ConstVertSpeedProfile(
            #     z_g_t0=start_z, w_A_g=w_A_g_2, tf=csc_horiz_path[1].duration),
            LinearVertSpeedProfile.init_from(
                z_g_t0=z_g_20,
                w_A_g_t0=w_A_g_20,
                w_A_g_tf=w_A_g_21,
                tf=csc_horiz_path[1].duration),
            ConstVertSpeedProfile(
                z_g_t0=z_g_21, w_A_g=w_A_g_3, tf=csc_horiz_path[2].duration),
        ])
    vert_prof_segments = []
    vert_prof_segments.append(csc_vert_prof)
    for _ in range(end_rounds):
        vert_prof_segments.append(copy.deepcopy(end_round_vert_prof))
    vert_profile = Trajectory(vert_prof_segments, align_initial_states=True)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path


def get_trochoid_path_3D_longitudinal_transition(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        turn_n_max,
        wind_speed,
        wind_dir,
        path_type,
        ac: ExtDubinsAirplane,
        transition_strategy="rapid_descent_fixed_length",
        speed_eps=1e-3,
        gamma_A_eps=1e-5,
        z_eps=1e-2,
        high_alt_strategy="full turns",
        **kwargs):
    """Calculate 3D Trochoid-Dubins paths with longitudinal transition over the
    straight segment.

    Newer version of :func:`get_trochoid_path_3D_longitudinal_transition_alt` that
    updates the horizontal path on each iteration of the final flight path angle.
    Here, the horizontal path computation is moved to an outer loop. For each
    fixed horizontal path, the reachable max/min flight path angl eis computed.
    This avoids bifurcation of the bi-section method when the horizontal path
    geometry changes in reaction to changes in vertical speed and turn rates
    (i.e. z_g_tf is non-monotonic in gamma_A_21_min)
    """
    if isinstance(turn_rate, tuple) or isinstance(turn_rate, list):
        turn_rate1 = turn_rate[0]
        turn_rate2 = turn_rate[-1]
    else:
        turn_rate1 = turn_rate
        # Used by the nested build_horizontal_path() closure below.
        turn_rate2 = turn_rate  # noqa: F841

    if goal_dir is None:
        raise NotImplementedError("CS 3D path not implemented")

    if high_alt_strategy == "full turns":
        start_end_round_func = get_full_turn
    elif high_alt_strategy == "periodic paths":
        start_end_round_func = get_trochoid_periodic_path
    else:
        raise ValueError(f"Unknown high_alt_strategy: {high_alt_strategy}")

    # NOTE: See v1 for the naming conventions of the segment-wise vertical
    # speeds and horizontal airspeeds.
    v_a_xy_1_straight = airspeed
    v_a_xy_1 = airspeed
    v_a_xy_20 = airspeed
    v_a_xy_21 = airspeed
    v_a_xy_3 = airspeed
    v_a_xy_3_straight = airspeed
    w_A_g_1 = 0
    w_A_g_20 = 0
    w_A_g_21 = 0
    w_A_g_3 = 0
    w_A_g_3_straight = 0
    end_rounds = 0

    if turn_n_max is not None:
        gamma_A_1 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1)
        turn_rate1 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
            turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1), gamma_A_1,
            0, 9.81)

    T_1 = ac.thrust_required(
        V_A=vt.V_A_from_V_A_xy_w_A_g(v_a_xy_1, w_A_g_1),
        w_A_g=w_A_g_1,
        chi_A_dot=turn_rate1)

    T_min_2 = ac.T_min
    T_max_2 = T_1

    # Two-level structure:
    #   OUTER loop: (re)build the horizontal path from the current airspeeds
    #               and search for the gamma_A_21_min that hits goal_z on that
    #               fixed geometry. The geometry changes only as fast as the
    #               airspeeds converge, so a handful of outer passes suffice.
    #   INNER search: on the FIXED geometry, gamma_A_21_min -> z_g_tf is
    #               monotonic, so a bracketed line search converges reliably.
    # Used by the nested closures below.
    gamma_A_21_min = -math.radians(60)  # noqa: F841

    def build_horizontal_path(v_a_xy_21, v_a_xy_3, v_a_xy_3_straight, w_A_g_3):
        """Build the horizontal path for the current airspeeds and final-turn
        descent rate. Returns ``None`` if infeasible."""
        gamma_A_2 = vt.gamma_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3)
        if turn_n_max is not None:
            turn_rate2 = fpm.chi_A_dot_abs_from_n_gamma_A_dot(
                turn_n_max, vt.V_A_from_V_A_xy_w_A_g(v_a_xy_3, w_A_g_3),
                gamma_A_2, 0, ac.g)
        airspeeds = (
            v_a_xy_1_straight, v_a_xy_1, v_a_xy_20, v_a_xy_21, v_a_xy_3,
            v_a_xy_3_straight)
        return get_trochoid_path_2D(
            start_x, start_y, start_dir, goal_x, goal_y, goal_dir, airspeeds,
            (turn_rate1, turn_rate2), wind_speed, wind_dir, path_type, 0,
            end_rounds, start_end_round_func)

    def solve_longitudinal_maneuver_fixed_horiz_path(
            horiz_path, gamma_A_21_min, V_A_xy_tf=None):
        """Iteratively calculate the longitudinal maneuver on a fixed horizontal
        path geometry. This is a fixed-point iteration yielding final airspeed
        and vertical speed values to be re-used in the outer iteration.
        """
        logger.push("final_airspeed_iteration")

        straight_segment_length = horiz_path.segments[1].xf[
            "l_xy"] - horiz_path.segments[1].x0["l_xy"]
        z_g_20 = start_z + w_A_g_1*horiz_path.segments[0].duration

        # Reset horizontal and vertical airspeed components after the
        # longitudinal transition
        w_A_g_21 = 0
        w_A_g_3 = 0
        w_A_g_3_straight = 0
        v_a_xy_21 = airspeed
        v_a_xy_3 = airspeed
        v_a_xy_3_straight = airspeed

        z_g_tf_values = logger.track_values("z_g_tf")

        # First iteration: final airspeed is calculated iteratively
        # based on T_max at start and T_min at end of the rapid descent
        # transition, governed by the gamma_A_min constraint. Later
        # iterations reuse the previous final airspeed.
        if V_A_xy_tf is None:
            lon_man = LongitudinalManeuver.solve_rapid_descent_fixed_length(
                ac,
                v_a_xy_20,
                w_A_g_20,
                z_g_20,
                horiz_path.segments[1].chi,
                wind_speed,
                wind_dir,
                straight_segment_length,
                T_min=T_min_2,
                T_max=T_max_2,
                gamma_A_min=gamma_A_21_min,
                V_A_eps=speed_eps)
            V_A_xy_tf = lon_man.V_A_xy_tf

        while not utils.converged(z_g_tf_values, abs_tol=z_eps):
            if transition_strategy == "rapid_descent_fixed_length":
                lon_man = LongitudinalManeuver.solve_rapid_descent_fixed_length_fixed_airspeed(
                    ac,
                    v_a_xy_20,
                    w_A_g_20,
                    z_g_20,
                    horiz_path.segments[1].chi,
                    wind_speed,
                    wind_dir,
                    straight_segment_length,
                    T_min=T_min_2,
                    gamma_A_min=gamma_A_21_min,
                    V_A_eps=speed_eps,
                    V_A_xy_tf=V_A_xy_tf)

                # Replace straight segment of horizontal path
                new_straight_segment = LinearSpeedStraightFlight.init_from(
                    x_g_t0=horiz_path.segments[1].x_g_t0,
                    y_g_t0=horiz_path.segments[1].y_g_t0,
                    l_xy_tf=straight_segment_length,
                    V_A_xy_t0=lon_man.V_A_xy_t0,
                    V_A_xy_tf=lon_man.V_A_xy_tf,
                    chi=horiz_path.segments[1].chi,
                    V_W_xy=wind_speed,
                    chi_W=wind_dir,
                    t0=horiz_path.segments[0].tf)
                horiz_path.segments[1] = new_straight_segment
                horiz_path.alignSegments(align_initial_states=False)
            else:
                raise ValueError(
                    f"Transition strategy not supported: {transition_strategy}")

            # Update vertical speeds after the longitudinal transition maneuver
            w_A_g_21 = lon_man.w_A_g_tf
            w_A_g_3 = lon_man.w_A_g_tf
            w_A_g_3_straight = lon_man.w_A_g_tf

            # Update horizontal airspeeds
            v_a_xy_21 = utils.clip(lon_man.V_A_xy_tf, wind_speed)
            v_a_xy_3 = utils.clip(lon_man.V_A_xy_tf, wind_speed)
            v_a_xy_3_straight = utils.clip(
                vt.V_A_xy_from_V_A_w_A_g(lon_man.V_A_tf, w_A_g_3_straight),
                wind_speed)

            # Update final altitude
            z_g_21 = lon_man.z_g_tf
            z_g_31 = z_g_21 + w_A_g_3*horiz_path.segments[2].duration
            z_g_tf = z_g_31
            for _ in range(end_rounds):
                if high_alt_strategy == "full turns":
                    z_g_tf += w_A_g_3*horiz_path.segments[-1].duration
                elif high_alt_strategy == "periodic paths":
                    for seg in horiz_path.segments[-1].segments:
                        if seg.x0["chi_A_dot"] != 0:
                            z_g_tf += w_A_g_3*seg.duration
                        else:
                            z_g_tf += w_A_g_3_straight*seg.duration
                else:
                    raise ValueError(
                        f"High altitude strategy not supported: {high_alt_strategy}"
                    )
            z_g_tf_values.append(z_g_tf)

        logger.pop(iterations=z_g_tf_values.n_iterations, success=True)
        return dict(
            horiz_path=horiz_path,
            z_g_20=z_g_20,
            z_g_21=z_g_21,
            z_g_31=z_g_31,
            z_g_tf=z_g_tf,
            w_A_g_21=w_A_g_21,
            w_A_g_3=w_A_g_3,
            w_A_g_3_straight=w_A_g_3_straight,
            v_a_xy_21=v_a_xy_21,
            v_a_xy_3=v_a_xy_3,
            v_a_xy_3_straight=v_a_xy_3_straight)

    def bisect_gamma_A_for_final_alt(
            horiz_path,
            gamma_A_21_min=-math.radians(60),
            gamma_A_21_max=0.0,
            V_A_xy_tf=None):
        """Find the gamma_A_21_min that makes z_g_tf == goal_z on the given
        fixed geometry.

        ``z_g_tf`` is monotonic in gamma_A_21_min on a fixed geometry (steeper
        descent -> higher z_g_tf), so a bracketed bisection is well posed.
        Returns ``(state, gamma)``, or ``(None, None)`` if infeasible.
        """
        state = solve_longitudinal_maneuver_fixed_horiz_path(
            horiz_path, gamma_A_21_min, V_A_xy_tf=V_A_xy_tf)
        if state is None:
            return None
        if goal_z is None or abs(state["z_g_tf"] - goal_z) < z_eps:
            return state
        # If even the steepest descent does not reach goal_z, the caller must
        # add an end round (geometry change).
        if state["z_g_tf"] < goal_z:
            return state
        logger.push("bisect_gamma_A_for_final_alt")
        iterations = 0
        while abs(gamma_A_21_max - gamma_A_21_min) > gamma_A_eps:
            iterations += 1
            gamma_mid = 0.5*(gamma_A_21_min+gamma_A_21_max)
            state = solve_longitudinal_maneuver_fixed_horiz_path(
                horiz_path, gamma_mid, V_A_xy_tf=V_A_xy_tf)
            if state is None:
                logger.pop(success=False)
                return None
            if abs(state["z_g_tf"] - goal_z) < z_eps:
                logger.pop(success=True, iterations=iterations)
                return state
            # Trace the bisection: altitude residual and bracket width
            logger.add_value(
                "z_g_tf_residual",
                abs(state["z_g_tf"] - goal_z),
                gamma_A_21_min=gamma_mid,
                bracket=gamma_A_21_max - gamma_A_21_min)
            if state["z_g_tf"] > goal_z:
                # Descended too far: need a shallower descent (gamma -> 0).
                gamma_A_21_min = gamma_mid
            else:
                # Not descended enough: need a steeper descent (gamma -> -60).
                gamma_A_21_max = gamma_mid
        logger.pop(success=True, iterations=iterations)
        return state

    logger.push("start_end_rounds_iteration")
    state = None
    w_A_g_3 = 0.0
    outer_iterations = 0
    max_outer_iterations = 200
    max_end_rounds = 50
    V_A_xy_tf_pinned = None
    # History of the last two outer states (v_a_xy_21, v_a_xy_3) for 2-cycle
    # detection. The outer fixed-point map can settle into a period-2 orbit
    # (the geometry reacts to the airspeeds, which react back to the geometry);
    # plain Picard iteration never satisfies its convergence test there and
    # would exhaust the iteration budget.
    prev_states = []
    while True:
        outer_iterations += 1
        if outer_iterations > max_outer_iterations:
            logger.pop(success=False, outer_iterations=outer_iterations)
            return None
        if end_rounds > max_end_rounds:
            logger.pop(
                success=False, end_rounds=end_rounds, reason="max end rounds")
            return None

        # Build the horizontal path for the current airspeeds
        logger.push("horiz_path")
        horiz_path = build_horizontal_path(
            v_a_xy_21, v_a_xy_3, v_a_xy_3_straight, w_A_g_3)
        if horiz_path is None:
            logger.pop(success=False)
            logger.pop(success=False, outer_iterations=outer_iterations)
            return None
        logger.pop(success=True)

        if goal_z is not None:
            # Search final gamma_A on the fixed geometry
            state = bisect_gamma_A_for_final_alt(
                horiz_path, V_A_xy_tf=V_A_xy_tf_pinned)
            if state is None:
                logger.pop(success=False, outer_iterations=outer_iterations)
                return None
            if state["z_g_tf"] < goal_z - z_eps:
                # Cannot reach goal_z even at the steepest descent: add an end round and rebuild the geometry.
                end_rounds += 1
                continue
        else:
            state = solve_longitudinal_maneuver_fixed_horiz_path(
                horiz_path, -math.radians(60))

        # if goal_z is None:
        #     break

        # Update the airspeeds for the next geometry pass.
        v_a_xy_21_new = state["v_a_xy_21"]
        v_a_xy_3_new = state["v_a_xy_3"]
        w_A_g_3_new = state["w_A_g_3"]

        # Check whether the geometry has converged.
        converged = (
            abs(v_a_xy_21_new - v_a_xy_21) < speed_eps
            and abs(v_a_xy_3_new - v_a_xy_3) < speed_eps
            and abs(w_A_g_3_new - w_A_g_3) < speed_eps)
        if converged:
            # Re-compute final horizontal path to ensure satisfaction of load
            # factor constraint to sufficient accuracy
            logger.push("horiz_path")
            converged_path = build_horizontal_path(
                v_a_xy_21_new, v_a_xy_3_new, state["v_a_xy_3_straight"],
                w_A_g_3_new)
            logger.pop(success=converged_path is not None)
            if converged_path is not None:
                horiz_path = converged_path
                state = {**state, "horiz_path": converged_path}
            break

        # detect 2-cycle and pin final airspeed to force convergence
        new_state = (v_a_xy_21_new, v_a_xy_3_new)
        if V_A_xy_tf_pinned is None and len(prev_states) >= 2 and all(
                abs(new_state[i] - prev_states[-2][i]) < speed_eps
                for i in range(2)):
            V_A_xy_tf_pinned = v_a_xy_3_new
            prev_states.clear()
        else:
            prev_states.append(new_state)
            if len(prev_states) > 2:
                prev_states.pop(0)

        v_a_xy_21 = v_a_xy_21_new
        v_a_xy_3 = v_a_xy_3_new
        v_a_xy_3_straight = state["v_a_xy_3_straight"]
        w_A_g_3 = state["w_A_g_3"]

    logger.pop(
        success=True, end_rounds=end_rounds, outer_iterations=outer_iterations)

    # Construct vertical profile
    horiz_path = state["horiz_path"]
    z_g_20 = state["z_g_20"]
    z_g_21 = state["z_g_21"]
    z_g_31 = state["z_g_31"]
    w_A_g_21 = state["w_A_g_21"]
    w_A_g_3 = state["w_A_g_3"]
    w_A_g_3_straight = state["w_A_g_3_straight"]

    vert_prof_segments = []
    if end_rounds:
        end_round = horiz_path.segments[-1]
        if high_alt_strategy == "full turns":
            end_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=z_g_31,
                w_A_g=w_A_g_3,
                tf=end_round.duration,
            )
        elif high_alt_strategy == "periodic paths":
            # Note that initial states, i.e. start altitude and time, will be
            # aligned automatically later when we construct the final vertical
            # profile
            end_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3_straight,
                        tf=end_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[2].duration,
                    ),
                ])
        else:
            raise ValueError(
                f"High altitude strategy not supported: {high_alt_strategy}")
    csc_horiz_path = horiz_path.segments[0:3]
    csc_vert_prof = Trajectory(
        [
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_1, tf=csc_horiz_path[0].duration),
            LinearVertSpeedProfile.init_from(
                z_g_t0=z_g_20,
                w_A_g_t0=w_A_g_20,
                w_A_g_tf=w_A_g_21,
                tf=csc_horiz_path[1].duration),
            ConstVertSpeedProfile(
                z_g_t0=z_g_21, w_A_g=w_A_g_3, tf=csc_horiz_path[2].duration),
        ])
    vert_prof_segments = []
    vert_prof_segments.append(csc_vert_prof)
    for _ in range(end_rounds):
        vert_prof_segments.append(copy.deepcopy(end_round_vert_prof))
    vert_profile = Trajectory(vert_prof_segments, align_initial_states=True)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path

    logger.pop(success=True, end_rounds=end_rounds)

    # Construct vertical profile
    horiz_path = state["horiz_path"]
    z_g_20 = state["z_g_20"]
    z_g_21 = state["z_g_21"]
    z_g_31 = state["z_g_31"]
    w_A_g_21 = state["w_A_g_21"]
    w_A_g_3 = state["w_A_g_3"]
    w_A_g_3_straight = state["w_A_g_3_straight"]

    vert_prof_segments = []
    if end_rounds:
        end_round = horiz_path.segments[-1]
        if high_alt_strategy == "full turns":
            end_round_vert_prof = ConstVertSpeedProfile(
                z_g_t0=z_g_31,
                w_A_g=w_A_g_3,
                tf=end_round.duration,
            )
        elif high_alt_strategy == "periodic paths":
            # Note that initial states, i.e. start altitude and time, will be
            # aligned automatically later when we construct the final vertical
            # profile
            end_round_vert_prof = Trajectory(
                [
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[0].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3_straight,
                        tf=end_round.segments[1].duration,
                    ),
                    ConstVertSpeedProfile(
                        z_g_t0=z_g_31,
                        w_A_g=w_A_g_3,
                        tf=end_round.segments[2].duration,
                    ),
                ])
        else:
            raise ValueError(
                f"High altitude strategy not supported: {high_alt_strategy}")
    csc_horiz_path = horiz_path.segments[0:3]
    csc_vert_prof = Trajectory(
        [
            ConstVertSpeedProfile(
                z_g_t0=start_z, w_A_g=w_A_g_1, tf=csc_horiz_path[0].duration),
            LinearVertSpeedProfile.init_from(
                z_g_t0=z_g_20,
                w_A_g_t0=w_A_g_20,
                w_A_g_tf=w_A_g_21,
                tf=csc_horiz_path[1].duration),
            ConstVertSpeedProfile(
                z_g_t0=z_g_21, w_A_g=w_A_g_3, tf=csc_horiz_path[2].duration),
        ])
    vert_prof_segments = []
    vert_prof_segments.append(csc_vert_prof)
    for _ in range(end_rounds):
        vert_prof_segments.append(copy.deepcopy(end_round_vert_prof))
    vert_profile = Trajectory(vert_prof_segments, align_initial_states=True)
    lifted_path = LiftedSegment(
        horiz_profile=horiz_path, vert_profile=vert_profile)

    return lifted_path


def get_trochoid_paths(
    start_x,
    start_y,
    start_z,
    start_dir,
    goal_x,
    goal_y,
    goal_z,
    goal_dir,
    airspeed,
    turn_rate,
    wind_speed,
    wind_dir,
    path_types=["LSL", "RSR", "LSR", "RSL"],
    strategy=None,
    **kwargs,
):
    result_paths = {}
    for path_type in path_types:
        trochoid_path = get_trochoid_path(
            start_x,
            start_y,
            start_z,
            start_dir,
            goal_x,
            goal_y,
            goal_z,
            goal_dir,
            airspeed,
            turn_rate,
            wind_speed,
            wind_dir,
            path_type,
            strategy,
            **kwargs,
        )
        if trochoid_path:
            result_paths[path_type] = trochoid_path
    return result_paths


def get_shortest_trochoid_path(
    start_x,
    start_y,
    start_z,
    start_dir,
    goal_x,
    goal_y,
    goal_z,
    goal_dir,
    airspeed,
    turn_rate,
    wind_speed,
    wind_dir,
    strategy="const. vertical speed",
    **kwargs,
):
    paths = get_trochoid_paths(
        start_x,
        start_y,
        start_z,
        start_dir,
        goal_x,
        goal_y,
        goal_z,
        goal_dir,
        airspeed,
        turn_rate,
        wind_speed,
        wind_dir,
        strategy=strategy,
        **kwargs)
    if not paths:
        return None
    paths = list(paths.values())
    paths.sort(key=lambda path: path.tf)
    return paths[0]

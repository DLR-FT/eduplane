# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A circular turn segment in constant wind:
  - constant airspeed (only horizontal component considered here)
  - constant wind.
"""

from dataclasses import field

import numpy
import scipy.optimize
from scipy.special import ellipe, ellipeinc

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.utils import utils
from eduplane.utils.dataclass_greedy_init import dataclass_greedy_init

from ..discrete_segment import DiscreteSegment
from ..segment_base import HorizProfile


def x_g_M_from_x_g(x_g, chi, radius, turn_dir):
    """Calculate the turn midpoint x-coordinate from a given
    position and flight path azimuth."""
    return x_g - radius*numpy.sin(chi)*turn_dir


def y_g_M_from_y_g(y_g, chi, radius, turn_dir):
    """Calculate the turn midpoint y-coordinate from a given
    position and flight path azimuth."""
    return y_g + radius*numpy.cos(chi)*turn_dir


def t_full_turn(V_A_xy, V_W_xy, radius):
    """Calculate time for a full turn / orbit time."""
    k = V_W_xy/V_A_xy
    k_squared = k**2
    return (4*radius*ellipe(k_squared))/(V_A_xy*(1-k_squared))


@dataclass_greedy_init
class CircularTurn(HorizProfile):
    """Circular turn at constant airspeed horizontal trajectory profile."""

    # Initial/final values
    chi_t0: float
    chi_tf: float

    # Constants
    V_A_xy: float
    radius: float
    turn_dir: float

    # Optional initial/final values
    x_g_t0: float = field(default=0)
    y_g_t0: float = field(default=0)
    t0: float = field(default=0)
    l_xy_t0: float = field(default=0)

    # Optional constants
    chi_W: float = field(default=0)
    V_W_xy: float = field(default=0)
    n_full_turns: int = field(default=0)

    # Derived initial/final values
    V_K_xy_t0: float = field(default=None)
    chi_A_t0: float = field(default=None)
    chi_dot_t0: float = field(default=None)
    tau_t0: float = field(init=False)
    tf: float = field(default=None)

    # Derived constants
    u_W_g: float = field(init=False)
    v_W_g: float = field(init=False)
    x_g_M: float = field(init=False)
    y_g_M: float = field(init=False)
    t_full_turn: float = field(init=False)

    # Predefined constants
    V_A_xy_dot: float = field(init=False, default=0)

    def __post_init__(self):
        """Initialize from airspeed, radius, initial and final flight path azimuth."""
        # Make sure turn dir is -1 or 1
        self.turn_dir = self.turn_dir/abs(self.turn_dir)
        # Calculate time for full turn/orbit
        self.t_full_turn = t_full_turn(self.V_A_xy, self.V_W_xy, self.radius)
        # Unwrap chi_tf according to turn rate, i.e. chi_tf<=chi_t0 for ccw and
        # chi_tf>chi_t0 for cw turns.
        self.chi_tf = self.chi_t0 + utils.angular_diff(
            self.chi_t0, self.chi_tf, self.turn_dir)
        self.tau_t0 = self.tau_from_chi(self.chi_t0)
        tau_tf = self.tau_from_chi(self.chi_tf)
        self.tf = self.t0 + (
            tau_tf -
            self.tau_t0)*self.turn_dir + self.n_full_turns*self.t_full_turn
        # Calculate derived constants
        self.u_W_g = vt.u_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        self.v_W_g = vt.v_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        self.x_g_M = x_g_M_from_x_g(
            self.x_g_t0, self.chi_t0, self.radius, self.turn_dir)
        self.y_g_M = y_g_M_from_y_g(
            self.y_g_t0, self.chi_t0, self.radius, self.turn_dir)

    def x_g_from_chi(self, chi):
        # return self.x_g_M + self.radius*numpy.sin(chi)
        return self.x_g_M + self.radius*numpy.sin(chi)*self.turn_dir

    def y_g_from_chi(self, chi):
        # return self.y_g_M - self.radius*numpy.cos(chi)
        return self.y_g_M - self.radius*numpy.cos(chi)*self.turn_dir

    def tau_from_chi(self, chi):
        n_full_turns = abs(chi - self.chi_t0) // (2*numpy.pi)
        k = self.V_W_xy/self.V_A_xy
        k_squared = k**2
        C = self.radius/(self.V_A_xy*(1-k_squared))
        theta = chi - self.chi_W
        E = ellipeinc(theta, k_squared)
        offset_term = k*numpy.sin(theta)
        return C*(E-offset_term) + n_full_turns*self.t_full_turn

    def chi(self, t):
        dt = (t - self.t0) % self.t_full_turn
        n_full_turns = (t - self.t0) // self.t_full_turn
        k = self.V_W_xy/self.V_A_xy
        k_squared = k**2
        # Scaling factor for the elliptic integral term
        C = self.radius/(self.V_A_xy*(1-k_squared))

        # Forward function: tau(chi)
        def tau_from_track(chi):
            theta = chi - self.chi_W
            E = ellipeinc(theta, k_squared)
            offset_term = k*numpy.sin(theta)
            return C*(E-offset_term) - self.tau_t0

        chi_guess = self.chi_t0 + (self.V_A_xy/self.radius)*dt*self.turn_dir

        chi_at_t = scipy.optimize.newton(
            lambda _chi: tau_from_track(_chi) - self.turn_dir*dt, x0=chi_guess)
        chi_at_t += n_full_turns*2*numpy.pi*self.turn_dir
        return chi_at_t

    def y_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return self.y_g_from_chi(chi)

    def x_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return self.x_g_from_chi(chi)

    def V_K_xy(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return vt.V_K_xy_from_V_A_xy_chi(
            self.V_A_xy, chi, self.V_W_xy, self.chi_W)

    def chi_dot(self, t, chi=None):
        V_K_xy = self.V_K_xy(t, chi)
        return V_K_xy/self.radius*self.turn_dir

    def chi_A_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return km.chi_A_dot_from_chi_dot_V_A_xy_dot.eval_if(
            mask=self.V_W_xy > 0,
            fill_value=self.chi_dot(t, chi),
            V_A_xy=self.V_A_xy,
            V_A_xy_dot=self.V_A_xy_dot,
            chi=chi,
            chi_dot=self.chi_dot(t, chi),
            V_W_xy=self.V_W_xy,
            chi_W=self.chi_W)

    def x_a(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return self.x_g(t, chi) - self.V_W_xy*numpy.cos(self.chi_W)*t

    def y_a(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return self.y_g(t, chi) - self.V_W_xy*numpy.sin(self.chi_W)*t

    def u_K_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        V_K_xy = self.V_K_xy(t, chi)
        return V_K_xy*numpy.cos(chi)

    def v_K_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        V_K_xy = self.V_K_xy(t, chi)
        return V_K_xy*numpy.sin(chi)

    def u_K_g_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        V_K_xy = self.V_K_xy(t, chi)
        V_K_xy_dot = self.V_K_xy_dot(t, chi)
        chi_dot = self.chi_dot(t, chi)
        return V_K_xy_dot*numpy.cos(chi) - V_K_xy*chi_dot*numpy.sin(chi)

    def v_K_g_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        V_K_xy = self.V_K_xy(t, chi)
        V_K_xy_dot = self.V_K_xy_dot(t, chi)
        chi_dot = self.chi_dot(t, chi)
        return V_K_xy_dot*numpy.sin(chi) + V_K_xy*chi_dot*numpy.cos(chi)

    def u_A_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        chi_A = self.chi_A(t, chi)
        return self.V_A_xy*numpy.cos(chi_A)

    def v_A_g(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        chi_A = self.chi_A(t, chi)
        return self.V_A_xy*numpy.sin(chi_A)

    def u_A_g_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        chi_A = self.chi_A(t, chi)
        chi_A_dot = self.chi_A_dot(t, chi)
        return -self.V_A_xy*chi_A_dot*numpy.sin(chi_A)

    def v_A_g_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        chi_A = self.chi_A(t, chi)
        chi_A_dot = self.chi_A_dot(t, chi)
        return self.V_A_xy*chi_A_dot*numpy.cos(chi_A)

    def chi_A(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return vt.chi_A_from_V_A_xy_chi(
            self.V_A_xy, chi, self.V_W_xy, self.chi_W)

    def l_xy(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        return self.radius*(chi - self.chi_t0)*self.turn_dir

    def V_K_xy_dot(self, t, chi=None):
        if chi is None:
            chi = self.chi(t)
        chi_A = self.chi_A(t, chi)
        chi_A_dot = self.chi_A_dot(t, chi)
        return km.V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            self.V_A_xy, self.V_A_xy_dot, chi_A, chi_A_dot, self.V_W_xy,
            self.chi_W)

    def shift(self, new_t0, **new_initial_values):
        if self.contains_valid_value(new_initial_values, "x_g"):
            self.x_g_t0 = new_initial_values["x_g"]
            self.x_g_M = x_g_M_from_x_g(
                self.x_g_t0, self.chi_t0, self.radius, self.turn_dir)
        if self.contains_valid_value(new_initial_values, "y_g"):
            self.y_g_t0 = new_initial_values["y_g"]
            self.y_g_M = y_g_M_from_y_g(
                self.y_g_t0, self.chi_t0, self.radius, self.turn_dir)
        if self.contains_valid_value(new_initial_values, "l_xy"):
            self.l_xy_t0 = new_initial_values["l_xy"]
        super().shift(new_t0)
        return self

    def evaluate(self, t):
        self.check_parameter(t)
        values = self.get_empty_dict()
        values.update(
            {
                # Constants
                "V_A_xy": self.V_A_xy,
                "V_A_xy_dot": self.V_A_xy_dot,
                "u_W_g": self.u_W_g,
                "v_W_g": self.v_W_g,
                "V_W_xy": self.V_W_xy,
                "chi_W": self.chi_W,
                # Parametric states
                "x_g": self.x_g(t),
                "y_g": self.y_g(t),
                "u_K_g": self.u_K_g(t),
                "v_K_g": self.v_K_g(t),
                "u_K_g_dot": self.u_K_g_dot(t),
                "v_K_g_dot": self.v_K_g_dot(t),
                "V_K_xy": self.V_K_xy(t),
                "V_K_xy_dot": self.V_K_xy_dot(t),
                "u_A_g": self.u_A_g(t),
                "v_A_g": self.v_A_g(t),
                "u_A_g_dot": self.u_A_g_dot(t),
                "v_A_g_dot": self.v_A_g_dot(t),
                "chi_A": self.chi_A(t),
                "chi_A_dot": self.chi_A_dot(t),
                "chi": self.chi(t),
                "chi_dot": self.chi_dot(t),
                "l_xy": self.l_xy(t),
            })
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        # Constants
        df["V_A_xy"] = self.V_A_xy
        df["V_A_xy_dot"] = self.V_A_xy_dot
        df["u_W_g"] = self.u_W_g
        df["v_W_g"] = self.v_W_g
        df["V_W_xy"] = self.V_W_xy
        df["chi_W"] = self.chi_W
        # Parametric states
        df["x_g"] = self.x_g(t_index)
        df["y_g"] = self.y_g(t_index)
        df["u_K_g"] = self.u_K_g(t_index)
        df["v_K_g"] = self.v_K_g(t_index)
        df["u_K_g_dot"] = self.u_K_g_dot(t_index)
        df["v_K_g_dot"] = self.v_K_g_dot(t_index)
        df["V_K_xy"] = self.V_K_xy(t_index)
        df["V_K_xy_dot"] = self.V_K_xy_dot(t_index)
        df["u_A_g"] = self.u_A_g(t_index)
        df["v_A_g"] = self.v_A_g(t_index)
        df["u_A_g_dot"] = self.u_A_g_dot(t_index)
        df["v_A_g_dot"] = self.v_A_g_dot(t_index)
        df["chi"] = self.chi(t_index)
        df["chi_dot"] = self.chi_dot(t_index)
        df["chi_A"] = self.chi_A(t_index)
        df["chi_A_dot"] = self.chi_A_dot(t_index)
        df["l_xy"] = self.l_xy(t_index)
        return DiscreteSegment(df=df)

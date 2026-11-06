# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A steady (quasi-stationary) turn segment in constant wind:
  - constant airspeed (only horizontal component considered here)
  - constant turn rate
  - constant wind.

These assumptions result in a horizontal flight path profile in the shape of a
trochoid.
"""

from dataclasses import field

import numpy
from scipy.special import ellipeinc

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import HorizProfile


def x_g_M_from_x_g(x_g, V_A_xy, chi_A, chi_A_dot):
    """Calculate the instantaneous turn midpoint x-coordinate from a given
    position and air-path azimuth."""
    return x_g - V_A_xy/chi_A_dot*numpy.sin(chi_A)


def y_g_M_from_y_g(y_g, V_A_xy, chi_A, chi_A_dot):
    """Calculate the instantaneous turn midpoint y-coordinate from a given
    position and air-path azimuth."""
    return y_g + V_A_xy/chi_A_dot*numpy.cos(chi_A)


def l_xy_from_t(t, V_A_xy, chi_A_t0, chi_A_dot, V_W_xy, chi_W):
    """Calculate horizontal distance along the trochoid."""
    R_A = V_A_xy/chi_A_dot
    R_W = V_W_xy/chi_A_dot
    k2 = (4*R_A*R_W)/(R_A + R_W)**2
    phi = 0.5*(chi_A_dot*t + chi_A_t0 - chi_W)
    phi_t0 = 0.5*(chi_A_t0-chi_W)
    ellip_int = ellipeinc(phi, k2)
    ellip_int_t0 = ellipeinc(phi_t0, k2)
    return 2*(R_A+R_W)*(ellip_int-ellip_int_t0)


@dataclass_greedy_init
class SteadyTurn(HorizProfile):
    """Steady turn at constant airspeed horizontal trajectory profile."""

    # Initial/final values

    # Constants
    V_A_xy: float

    # Optional initial/final values
    chi_t0: float
    chi_A_t0: float = field(default=None)
    x_g_t0: float = field(default=0)
    y_g_t0: float = field(default=0)
    t0: float = field(default=0)
    l_xy_t0: float = field(default=0)
    chi_tf: float = field(default=None)
    chi_A_tf: float = field(default=None)
    x_g_tf: float = field(default=None)
    y_g_tf: float = field(default=None)
    tf: float = field(default=None)

    # Optional constants
    chi_A_dot: float = field(default=0)
    chi_W: float = field(default=0)
    V_W_xy: float = field(default=0)

    # Derived initial/final values
    x_g_M_t0: float = field(default=None, init=False)
    y_g_M_t0: float = field(default=None, init=False)
    V_K_xy_t0: float = field(default=None)
    chi_dot_t0: float = field(default=None)

    # Derived constants
    u_W_g: float = field(init=False)
    v_W_g: float = field(init=False)

    # Predefined constants
    V_A_xy_dot: float = field(init=False, default=0)

    @init_field("x_g_tf")
    def x_g_tf_from_x_g_t0(
            V_A_xy, chi_A_t0, chi_A_dot, V_W_xy, chi_W, x_g_t0, t0, tf):
        """Initialize field 'x_g_tf'."""
        dt = tf - t0
        x_g_M_t0 = x_g_M_from_x_g(x_g_t0, V_A_xy, chi_A_t0, chi_A_dot)
        return (
            x_g_M_t0 + V_A_xy/chi_A_dot*numpy.sin(chi_A_dot*dt + chi_A_t0) +
            V_W_xy*numpy.cos(chi_W)*dt)

    @init_field("y_g_tf")
    def y_g_tf_from_y_g_t0(
            V_A_xy, chi_A_t0, chi_A_dot, V_W_xy, chi_W, y_g_t0, t0, tf):
        """Initialize field 'y_g_tf'."""
        dt = tf - t0
        y_g_M_t0 = y_g_M_from_y_g(y_g_t0, V_A_xy, chi_A_t0, chi_A_dot)
        return (
            y_g_M_t0 - V_A_xy/chi_A_dot*numpy.cos(chi_A_dot*dt + chi_A_t0) +
            V_W_xy*numpy.sin(chi_W)*dt)

    @init_field("x_g_t0")
    def x_g_t0_from_x_g_tf(
            V_A_xy, chi_A_tf, chi_A_dot, V_W_xy, chi_W, x_g_tf, t0, tf):
        """Initialize field 'x_g_t0'."""
        dt = t0 - tf
        x_g_M_tf = x_g_M_from_x_g(x_g_tf, V_A_xy, chi_A_tf, chi_A_dot)
        return (
            x_g_M_tf + V_A_xy/chi_A_dot*numpy.sin(chi_A_dot*dt + chi_A_tf) +
            V_W_xy*numpy.cos(chi_W)*dt)

    @init_field("y_g_t0")
    def y_g_t0_from_y_g_tf(
            V_A_xy, chi_A_tf, chi_A_dot, V_W_xy, chi_W, y_g_tf, t0, tf):
        """Initialize field 'y_g_t0'."""
        dt = t0 - tf
        y_g_M_tf = y_g_M_from_y_g(y_g_tf, V_A_xy, chi_A_tf, chi_A_dot)
        return (
            y_g_M_tf - V_A_xy/chi_A_dot*numpy.cos(chi_A_dot*dt + chi_A_tf) +
            V_W_xy*numpy.sin(chi_W)*dt)

    @init_field("chi_t0")
    def chi_t0_from_chi_A_t0(V_A_xy, chi_A_t0, V_W_xy, chi_W):
        """Initialize field 'chi_t0'."""
        return vt.chi_from_V_A_xy_chi_A(V_A_xy, chi_A_t0, V_W_xy, chi_W)

    @init_field("chi_A_t0")
    def chi_A_t0_from_chi_t0(V_A_xy, chi_t0, V_W_xy, chi_W):
        """Initialize field 'chi_A_t0'."""
        return vt.chi_A_from_V_A_xy_chi(V_A_xy, chi_t0, V_W_xy, chi_W)

    @init_field("chi_t0")
    def chi_t0_from_chi_tf(V_A_xy, chi_tf, chi_A_dot, V_W_xy, chi_W, t0, tf):
        """Initialize field 'chi_t0'."""
        chi_A_tf = vt.chi_A_from_V_A_xy_chi(V_A_xy, chi_tf, V_W_xy, chi_W)
        chi_A_t0 = chi_A_tf - chi_A_dot*(tf-t0)
        return vt.chi_from_V_A_xy_chi_A(V_A_xy, chi_A_t0, V_W_xy, chi_W)

    @init_field("chi_t0")
    def chi_t0_from_chi_A_tf(
            V_A_xy, chi_A_tf, chi_A_dot, V_W_xy, chi_W, t0, tf):
        """Initialize field 'chi_A_t0'."""
        chi_A_t0 = chi_A_tf - chi_A_dot*(tf-t0)
        return vt.chi_from_V_A_xy_chi_A(V_A_xy, chi_A_t0, V_W_xy, chi_W)

    @init_field("chi_tf")
    def chi_tf_from_chi_A_tf(V_A_xy, chi_A_tf, V_W_xy, chi_W):
        """Initialize field 'chi_tf'."""
        return vt.chi_from_V_A_xy_chi_A(V_A_xy, chi_A_tf, V_W_xy, chi_W)

    @init_field("chi_A_tf")
    def chi_A_tf_from_chi_tf(V_A_xy, chi_tf, V_W_xy, chi_W):
        """Initialize field 'chi_A_tf'."""
        return vt.chi_A_from_V_A_xy_chi(V_A_xy, chi_tf, V_W_xy, chi_W)

    @init_field("V_A_xy")
    def V_A_xy_from_V_K_xy_t0(V_K_xy_t0, chi_t0, V_W_xy, chi_W):
        """Initialize field 'V_A_xy'."""
        return vt.V_A_xy_from_V_K_xy_chi(V_K_xy_t0, chi_t0, V_W_xy, chi_W)

    @init_field("chi_A_dot")
    def chi_A_dot_from_chi_dot_t0(V_A_xy, chi_t0, chi_dot_t0, V_W_xy, chi_W):
        if V_W_xy > 0:
            return km.chi_A_dot_from_chi_dot_V_A_xy_dot(
                V_A_xy, 0, chi_t0, chi_dot_t0, V_W_xy, chi_W)
        else:
            return chi_dot_t0

    def __post_init__(self):
        """Initialize from airspeed, initial flight path azimuth, turn rate,
        i.e. air-path azimuth rate, and final time."""
        # Calculate derived initial/final values
        if self.chi_A_t0 is None:
            self.chi_A_t0 = vt.chi_A_from_V_A_xy_chi(
                self.V_A_xy, self.chi_t0, self.V_W_xy, self.chi_W)
        if self.chi_t0 is None:
            self.chi_t0 = vt.chi_from_V_A_xy_chi_A(
                self.V_A_xy, self.chi_A_t0, self.V_W_xy, self.chi_W)
        if self.x_g_M_t0 is None:
            self.x_g_M_t0 = x_g_M_from_x_g(
                self.x_g_t0, self.V_A_xy, self.chi_A_t0, self.chi_A_dot)
        if self.y_g_M_t0 is None:
            self.y_g_M_t0 = y_g_M_from_y_g(
                self.y_g_t0, self.V_A_xy, self.chi_A_t0, self.chi_A_dot)
        if self.V_K_xy_t0 is None:
            self.V_K_xy_t0 = vt.V_K_xy_from_V_A_xy_chi(
                self.V_A_xy, self.chi_t0, self.V_W_xy, self.chi_W)
        if self.chi_dot_t0 is None:
            if self.V_W_xy > 0:
                self.chi_dot_t0 = km.chi_dot_from_chi_A_dot_V_A_xy_dot(
                    self.V_A_xy, self.V_A_xy_dot, self.chi_t0, self.chi_A_dot,
                    self.V_W_xy, self.chi_W)
            else:
                self.chi_dot_t0 = self.chi_A_dot
        if self.chi_A_tf is None:
            self.chi_A_tf = self.chi_A_t0 + self.chi_A_dot*(self.tf - self.t0)
        if self.chi_tf is None:
            self.chi_tf = vt.chi_from_V_A_xy_chi_A(
                self.V_A_xy, self.chi_A_tf, self.V_W_xy, self.chi_W)
        if self.x_g_tf is None:
            self.x_g_tf = self.x_g_tf_from_x_g_t0(
                self.V_A_xy, self.chi_A_t0, self.chi_A_dot, self.V_W_xy,
                self.chi_W, self.x_g_t0, self.t0, self.tf)
        if self.y_g_tf is None:
            self.y_g_tf = self.y_g_tf_from_y_g_t0(
                self.V_A_xy, self.chi_A_t0, self.chi_A_dot, self.V_W_xy,
                self.chi_W, self.y_g_t0, self.t0, self.tf)
        # Calculate derived constants
        self.u_W_g = vt.u_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        self.v_W_g = vt.v_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)

    def x_g(self, t):
        dt = t - self.t0
        return (
            self.V_A_xy/self.chi_A_dot*
            numpy.sin(self.chi_A_dot*dt + self.chi_A_t0) +
            self.V_W_xy*numpy.cos(self.chi_W)*dt + self.x_g_M_t0)

    def y_g(self, t):
        dt = t - self.t0
        return (
            -self.V_A_xy/self.chi_A_dot*
            numpy.cos(self.chi_A_dot*dt + self.chi_A_t0) +
            self.V_W_xy*numpy.sin(self.chi_W)*dt + self.y_g_M_t0)

    def x_a(self, t):
        dt = t - self.t0
        return (
            self.V_A_xy/self.chi_A_dot*
            numpy.sin(self.chi_A_dot*dt + self.chi_A_t0))

    def y_a(self, t):
        dt = t - self.t0
        return (
            -self.V_A_xy/self.chi_A_dot*
            numpy.cos(self.chi_A_dot*dt + self.chi_A_t0))

    def u_K_g(self, t):
        return self.u_A_g(t) + self.u_W_g

    def v_K_g(self, t):
        return self.v_A_g(t) + self.v_W_g

    def u_K_g_dot(self, t):
        V_K_xy = self.V_K_xy(t)
        V_K_xy_dot = self.V_K_xy_dot(t)
        chi = self.chi(t)
        chi_dot = self.chi_dot(t)
        return V_K_xy_dot*numpy.cos(chi) - V_K_xy*chi_dot*numpy.sin(chi)

    def v_K_g_dot(self, t):
        V_K_xy = self.V_K_xy(t)
        V_K_xy_dot = self.V_K_xy_dot(t)
        chi = self.chi(t)
        chi_dot = self.chi_dot(t)
        return V_K_xy_dot*numpy.sin(chi) + V_K_xy*chi_dot*numpy.cos(chi)

    def u_A_g(self, t):
        dt = t - self.t0
        return self.V_A_xy*numpy.cos(self.chi_A_dot*dt + self.chi_A_t0)

    def v_A_g(self, t):
        dt = t - self.t0
        return self.V_A_xy*numpy.sin(self.chi_A_dot*dt + self.chi_A_t0)

    def u_A_g_dot(self, t):
        dt = t - self.t0
        return -self.V_A_xy*self.chi_A_dot*numpy.sin(
            self.chi_A_dot*dt + self.chi_A_t0)

    def v_A_g_dot(self, t):
        dt = t - self.t0
        return self.V_A_xy*self.chi_A_dot*numpy.cos(
            self.chi_A_dot*dt + self.chi_A_t0)

    def chi_A(self, t):
        dt = t - self.t0
        return self.chi_A_t0 + self.chi_A_dot*dt

    def chi(self, t):
        chi_A = self.chi_A(t)
        return vt.chi_from_V_A_xy_chi_A(
            self.V_A_xy, chi_A, self.V_W_xy, self.chi_W)

    def chi_dot(self, t):
        return km.chi_dot_from_chi_A_dot_V_A_xy_dot.eval_if(
            mask=self.V_W_xy > 0,
            fill_value=self.chi_A_dot,
            V_A_xy=self.V_A_xy,
            V_A_xy_dot=0,
            chi=self.chi(t),
            chi_A_dot=self.chi_A_dot,
            V_W_xy=self.V_W_xy,
            chi_W=self.chi_W)

    def l_xy(self, t):
        dt = t - self.t0
        return self.l_xy_t0 + l_xy_from_t(
            dt, self.V_A_xy, self.chi_A_t0, self.chi_A_dot, self.V_W_xy,
            self.chi_W)

    def V_K_xy(self, t):
        chi = self.chi(t)
        return vt.V_K_xy_from_V_A_xy_chi(
            self.V_A_xy, chi, self.V_W_xy, self.chi_W)

    def V_K_xy_dot(self, t):
        return km.V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            self.V_A_xy, 0, self.chi_A(t), self.chi_A_dot, self.V_W_xy,
            self.chi_W)

    def shift(self, new_t0, **new_initial_values):
        if self.contains_valid_value(new_initial_values, "x_g"):
            self.x_g_t0 = new_initial_values["x_g"]
            self.x_g_M_t0 = x_g_M_from_x_g(
                self.x_g_t0, self.V_A_xy, self.chi_A_t0, self.chi_A_dot)
        if self.contains_valid_value(new_initial_values, "y_g"):
            self.y_g_t0 = new_initial_values["y_g"]
            self.y_g_M_t0 = y_g_M_from_y_g(
                self.y_g_t0, self.V_A_xy, self.chi_A_t0, self.chi_A_dot)
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
                "chi_A_dot": self.chi_A_dot,
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
        df["chi_A_dot"] = self.chi_A_dot
        df["l_xy"] = self.l_xy(t_index)
        return DiscreteSegment(df=df)

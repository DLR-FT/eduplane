# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A straight flight horizontal trajectory segment."""

from dataclasses import field

from eduplane.model import velocity_triangle as vt
from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import HorizProfile


def l_xy_tf_from_tf(t0, tf, l_xy_t0, V_K_xy):
    """Calculate final horizontal distance from initial/final time and velocity."""
    return l_xy_t0 + V_K_xy*(tf-t0)


def tf_from_l_xy_tf(t0, l_xy_tf, l_xy_t0, V_K_xy):
    """Calculate final time from initial/final horizontal distance and velocity."""
    return t0 + (l_xy_tf-l_xy_t0)/V_K_xy


@dataclass_greedy_init
class ConstSpeedStraightFlight(HorizProfile):

    # Initial/final values
    tf: float

    # Constants
    V_A_xy: float
    chi: float

    # Optional initial/final values
    t0: float = field(default=0)
    x_g_t0: float = field(default=0)
    y_g_t0: float = field(default=0)
    l_xy_t0: float = field(default=0)

    # Optional constants
    V_W_xy: float = field(default=0)
    chi_W: float = field(default=0)

    # Derived initial/final values
    l_xy_tf: float = field(default=None)

    # Derived constants
    V_K_xy: float = field(default=None)
    chi_A: float = field(default=None)
    u_K_g: float = field(default=None)
    v_K_g: float = field(default=None)
    u_A_g: float = field(default=None)
    v_A_g: float = field(default=None)
    u_W_g: float = field(default=None)
    v_W_g: float = field(default=None)

    # Predefined constants
    u_K_g_dot: float = field(init=False, default=0)
    v_K_g_dot: float = field(init=False, default=0)
    V_K_xy_dot: float = field(init=False, default=0)
    chi_dot: float = field(init=False, default=0)
    u_A_g_dot: float = field(init=False, default=0)
    v_A_g_dot: float = field(init=False, default=0)
    V_A_xy_dot: float = field(init=False, default=0)
    chi_A_dot: float = field(init=False, default=0)

    def __post_init__(self):
        """Initialize from airspeed, flight path azimuth and final time."""
        # Calculate derived constants
        if self.chi_A is None:
            self.chi_A = vt.chi_A_from_V_A_xy_chi(
                self.V_A_xy, self.chi, self.V_W_xy, self.chi_W)
        if self.V_K_xy is None:
            self.V_K_xy = vt.V_K_xy_from_V_A_xy_chi_A(
                self.V_A_xy, self.chi_A, self.V_W_xy, self.chi_W)
        self.u_K_g = vt.u_K_g_from_V_K_xy(self.V_K_xy, self.chi)
        self.v_K_g = vt.v_K_g_from_V_K_xy(self.V_K_xy, self.chi)
        self.u_A_g = vt.u_A_g_from_V_A_xy(self.V_A_xy, self.chi_A)
        self.v_A_g = vt.v_A_g_from_V_A_xy(self.V_A_xy, self.chi_A)
        self.u_W_g = vt.u_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        self.v_W_g = vt.v_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        # Calculate derived initial/final values
        if self.l_xy_tf is None:
            self.l_xy_tf = l_xy_tf_from_tf(
                self.t0, self.tf, self.l_xy_t0, self.V_K_xy)

    @init_field("tf")
    def tf_from_l_xy_tf(l_xy_t0, l_xy_tf, V_K_xy, t0):
        return tf_from_l_xy_tf(l_xy_t0, l_xy_tf, t0, V_K_xy)

    @init_field("chi")
    def chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W):
        return vt.chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W)

    @init_field("chi")
    def chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W):
        return vt.chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W)

    @init_field("V_A_xy")
    def V_A_xy_from_V_K_xy(V_K_xy, chi, V_W_xy, chi_W):
        return vt.V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W)

    @init_field("V_K_xy")
    def V_K_xy_from_V_A_xy(V_A_xy, chi, V_W_xy, chi_W):
        return vt.V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W)

    def x_g(self, t, **states):
        return self.x_g_t0 + self.u_K_g*(t - self.t0)

    def y_g(self, t, **states):
        return self.y_g_t0 + self.v_K_g*(t - self.t0)

    def l_xy(self, t, **states):
        return self.l_xy_t0 + self.V_K_xy*(t - self.t0)

    def shift(self, new_t0, **new_initial_values):
        if self.contains_valid_value(new_initial_values, "x_g"):
            self.x_g_t0 = new_initial_values["x_g"]
        if self.contains_valid_value(new_initial_values, "y_g"):
            self.y_g_t0 = new_initial_values["y_g"]
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
                "u_K_g": self.u_K_g,
                "v_K_g": self.v_K_g,
                "u_K_g_dot": self.u_K_g_dot,
                "v_K_g_dot": self.v_K_g_dot,
                "V_K_xy": self.V_K_xy,
                "V_K_xy_dot": self.V_K_xy_dot,
                "chi": self.chi,
                "chi_dot": self.chi_dot,
                "u_A_g": self.u_A_g,
                "v_A_g": self.v_A_g,
                "u_A_g_dot": self.u_A_g_dot,
                "v_A_g_dot": self.v_A_g_dot,
                "V_A_xy": self.V_A_xy,
                "V_A_xy_dot": self.V_A_xy_dot,
                "chi_A": self.chi_A,
                "chi_A_dot": self.chi_A_dot,
                "u_W_g": self.u_W_g,
                "v_W_g": self.v_W_g,
                "V_W_xy": self.V_W_xy,
                "chi_W": self.chi_W,
                # Parametric states
                "x_g": self.x_g(t),
                "y_g": self.y_g(t),
                # Parameters
                "l_xy": self.l_xy(t),
            })
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        # Constants
        df["u_K_g"] = self.u_K_g
        df["v_K_g"] = self.v_K_g
        df["u_K_g_dot"] = self.u_K_g_dot
        df["v_K_g_dot"] = self.v_K_g_dot
        df["V_K_xy"] = self.V_K_xy
        df["V_K_xy_dot"] = self.V_K_xy_dot
        df["chi"] = self.chi
        df["chi_dot"] = self.chi_dot
        df["u_A_g"] = self.u_A_g
        df["v_A_g"] = self.v_A_g
        df["u_A_g_dot"] = self.u_A_g_dot
        df["v_A_g_dot"] = self.v_A_g_dot
        df["V_A_xy"] = self.V_A_xy
        df["V_A_xy_dot"] = self.V_A_xy_dot
        df["chi_A"] = self.chi_A
        df["chi_A_dot"] = self.chi_A_dot
        df["u_W_g"] = self.u_W_g
        df["v_W_g"] = self.v_W_g
        df["V_W_xy"] = self.V_W_xy
        df["chi_W"] = self.chi_W
        df["x_g"] = self.x_g(t_index)
        df["y_g"] = self.y_g(t_index)
        df["l_xy"] = self.l_xy(t_index)
        return DiscreteSegment(df=df)

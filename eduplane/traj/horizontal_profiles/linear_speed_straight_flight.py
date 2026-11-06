# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A horizontal trajectory profile for straight flight with linear airspeed."""

from dataclasses import field

import numpy
import scipy.optimize

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import HorizProfile


def u_A_g_dot_from_V_A_xy_dot_chi_A_dot(V_A_xy, V_A_xy_dot, chi_A, chi_A_dot):
    """Acceleration assumming constant airspeed rate and air path azimuth rate."""
    return V_A_xy_dot*numpy.cos(chi_A) - V_A_xy*chi_A_dot*numpy.sin(chi_A)


def v_A_g_dot_from_V_A_xy_dot_chi_A_dot(V_A_xy, V_A_xy_dot, chi_A, chi_A_dot):
    """Acceleration assumming constant airspeed rate and air path azimuth rate."""
    return V_A_xy_dot*numpy.sin(chi_A) + V_A_xy*chi_A_dot*numpy.cos(chi_A)


def l_A_xy_from_chi(t, chi, V_A_xy_dot, V_A_xy_t0, V_W_xy, chi_W, eps=1e-9):
    """Air-relative distance travelled in direction of the segment's
    ground track, i.e. distance travelled along segment neglecting
    any wind drift.

    This solution is based on standard form integral of sqrt(u²-a²)du.
    """
    V_W_xy_par = V_W_xy*numpy.sin(chi_W - chi)
    sqrt_term1 = numpy.sqrt(V_A_xy_t0**2 - V_W_xy_par**2)
    sqrt_term2 = numpy.sqrt((V_A_xy_t0 + V_A_xy_dot*t)**2 - V_W_xy_par**2)
    ln_arg_denom = V_A_xy_t0 + sqrt_term1
    ln_arg_num = V_A_xy_dot*t + V_A_xy_t0 + sqrt_term2
    if abs(V_A_xy_dot) <= eps:
        return V_A_xy_t0*t
    else:
        return 1/(2*V_A_xy_dot)*(
            (V_A_xy_dot*t + V_A_xy_t0)*sqrt_term2 - V_A_xy_t0*sqrt_term1 -
            V_W_xy_par**2*numpy.log(ln_arg_num/ln_arg_denom))


def l_W_xy_from_chi(t, chi, V_W_xy, chi_W):
    """Distance travelled along segment due to wind drift."""
    V_W_xy_parallel = V_W_xy*numpy.cos(chi - chi_W)
    dist_wind_parallel = V_W_xy_parallel*t
    return dist_wind_parallel


def l_xy_from_chi(t, chi, V_A_xy_dot, V_A_xy_t0, V_W_xy, chi_W):
    """Distance travelled along segment due to airspeed and wind drift."""
    l_A_xy = l_A_xy_from_chi(t, chi, V_A_xy_dot, V_A_xy_t0, V_W_xy, chi_W)
    l_W_xy = l_W_xy_from_chi(t, chi, V_W_xy, chi_W)
    return l_A_xy + l_W_xy


def tf_from_l_xy_tf(
        l_xy_t0,
        l_xy_tf,
        V_A_xy_t0,
        V_A_xy_dot,
        chi,
        V_W_xy,
        chi_W,
        t0=0,
        eps=1e-9):
    """Calculate final time from segment length."""
    d_l_xy = l_xy_tf - l_xy_t0
    if V_W_xy < eps:
        if abs(V_A_xy_dot) < eps:
            tf = d_l_xy/V_A_xy_t0
        else:
            po2 = V_A_xy_t0/V_A_xy_dot
            q = -2*d_l_xy/V_A_xy_dot
            radicand = po2**2 - q
            if radicand < 0:
                raise ValueError("Infinite segment time with speed < 0!")
            if V_A_xy_dot > 0:  # -> V_A_xy_dot > eps
                tf = -po2 + numpy.sqrt(radicand)
            else:  # -> V_A_xy_dot < -eps
                tf = -po2 - numpy.sqrt(radicand)
    else:
        tf = scipy.optimize.newton(
            lambda tf: l_A_xy_from_chi(
                tf, chi, V_A_xy_dot, V_A_xy_t0, V_W_xy, chi_W, eps) +
            l_W_xy_from_chi(tf, chi, V_W_xy, chi_W) - d_l_xy,
            x0=d_l_xy/V_A_xy_t0,
            tol=eps)
    return t0 + tf


def tf_from_l_xy_tf_V_A_xy_tf(
        l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W, eps=1e-12):
    """Calculate final time from segment length."""
    d_l_xy = l_xy_tf - l_xy_t0
    if V_W_xy < eps:
        # No wind: the ground distance is the mean airspeed times the
        # duration, i.e. d_l_xy = tf*(V_A_xy_t0 + V_A_xy_tf)/2.
        V_A_xy_mean = 0.5*(V_A_xy_t0+V_A_xy_tf)
        if abs(V_A_xy_mean) < eps:
            raise ValueError("Infinite segment time with zero mean speed!")
        tf = d_l_xy/V_A_xy_mean
    else:
        # With wind, solve
        #   h(t) = l_A_xy(t, (V_A_xy_tf-V_A_xy_t0)/t, ...) + l_W_xy(t, ...)
        #          - d_l_xy = 0
        # for the duration t. Unlike the equivalent residual in V_A_xy_dot
        # (which is singular at V_A_xy_dot=0), h(t) is monotonic in t, so a
        # bracketed root find is guaranteed to converge.
        def h(t):
            return l_A_xy_from_chi(
                t, chi, (V_A_xy_tf-V_A_xy_t0)/t, V_A_xy_t0, V_W_xy, chi_W,
                eps) + l_W_xy_from_chi(t, chi, V_W_xy, chi_W) - d_l_xy

        V_K_xy_min = vt.V_K_xy_from_V_A_xy_chi(
            min(V_A_xy_t0, V_A_xy_tf), chi, V_W_xy, chi_W)
        V_K_xy_max = vt.V_K_xy_from_V_A_xy_chi(
            max(V_A_xy_t0, V_A_xy_tf), chi, V_W_xy, chi_W)
        t_lo = d_l_xy/V_K_xy_max
        t_hi = d_l_xy/V_K_xy_min

        # The constant-airspeed bounds ignore the wind-drift term and can be
        # too tight (or even reversed) when the airspeed varies, so expand the
        # bracket outward until h(t) changes sign.
        f_lo = h(t_lo)
        f_hi = h(t_hi)
        for _ in range(200):
            if f_lo < 0 < f_hi:
                break
            if not (f_lo < 0):
                t_lo *= 0.5
                f_lo = h(t_lo)
            if not (f_hi > 0):
                t_hi *= 2.0
                f_hi = h(t_hi)
        else:
            raise ValueError(
                "tf_from_l_xy_tf_V_A_xy_tf: failed to bracket segment time!")
        tf = scipy.optimize.brentq(h, t_lo, t_hi, xtol=eps, maxiter=200)
    return tf


def V_A_xy_dot_from_l_xy_tf(
        l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W, eps=1e-9):
    """Calculate required airspeed rate-of-change to reach a given final
    airspeed within a given segment length.
    """
    if abs(V_A_xy_tf - V_A_xy_t0) < eps:
        return 0.0
    tf = tf_from_l_xy_tf_V_A_xy_tf(
        l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W, eps)
    return (V_A_xy_tf-V_A_xy_t0)/tf


@dataclass_greedy_init
class LinearSpeedStraightFlight(HorizProfile):
    """Straight flight with linear airspeed horizontal trajectory profile."""

    # Initial/final values
    tf: float
    V_A_xy_t0: float

    # Constants
    chi: float

    # Optional initial/final values
    x_g_t0: float = field(default=0)
    y_g_t0: float = field(default=0)
    t0: float = field(default=0)
    l_xy_t0: float = field(default=0)

    # Optional constants
    V_A_xy_dot: float = field(default=0)
    chi_W: float = field(default=0)
    V_W_xy: float = field(default=0)
    eps: float = field(default=1e-6)

    # Derived initial/final values
    chi_A_t0: float = field(default=None)
    V_K_xy_t0: float = field(default=None)
    V_K_xy_tf: float = field(default=None)
    V_A_xy_tf: float = field(default=None)
    l_xy_tf: float = field(default=None)

    # Derived constants
    u_W_g: float = field(default=None)
    v_W_g: float = field(default=None)

    # Predefined constants
    chi_dot: float = field(init=False, default=0)

    def __post_init__(self):
        """Initialize from airspeed, airspeed rate, flight path azimuth and
        final time."""
        # Calculate derived initial/final values
        if self.chi_A_t0 is None:
            self.chi_A_t0 = vt.chi_A_from_V_A_xy_chi(
                self.V_A_xy_t0, self.chi, self.V_W_xy, self.chi_W)
        if self.V_A_xy_tf is None:
            self.V_A_xy_tf = self.V_A_xy_t0 + self.V_A_xy_dot*(
                self.tf - self.t0)
        if self.V_K_xy_t0 is None:
            self.V_K_xy_t0 = vt.V_K_xy_from_V_A_xy_chi(
                self.V_A_xy_t0, self.chi, self.V_W_xy, self.chi_W)
        if self.V_K_xy_tf is None:
            self.V_K_xy_tf = vt.V_K_xy_from_V_A_xy_chi(
                self.V_A_xy_tf, self.chi, self.V_W_xy, self.chi_W)
        if self.l_xy_tf is None:
            dt = self.tf - self.t0
            self.l_xy_tf = l_xy_from_chi(
                dt, self.chi, self.V_A_xy_dot, self.V_A_xy_t0, self.V_W_xy,
                self.chi_W)
        # Calculate derived constants
        self.u_W_g = vt.u_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)
        self.v_W_g = vt.v_W_g_from_V_W_xy(self.V_W_xy, self.chi_W)

    @init_field("V_A_xy_dot")
    def V_A_xy_dot_from_V_A_xy_t0_tf(t0, tf, V_A_xy_t0, V_A_xy_tf):
        return (V_A_xy_tf-V_A_xy_t0)/(tf-t0)

    @init_field("V_A_xy_dot")
    def V_A_xy_dot_from_l_xy_tf(
            l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W):
        return V_A_xy_dot_from_l_xy_tf(
            l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W)

    @init_field("tf")
    def tf_from_l_xy_tf(
            l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_dot, chi, V_W_xy, chi_W, t0,
            eps):
        return tf_from_l_xy_tf(
            l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_dot, chi, V_W_xy, chi_W, t0,
            eps)

    @init_field("chi")
    def chi_from_V_A_xy_chi_A(V_A_xy_t0, chi_A_t0, V_W_xy, chi_W):
        return vt.chi_from_V_A_xy_chi_A(V_A_xy_t0, chi_A_t0, V_W_xy, chi_W)

    @init_field("V_A_xy_t0")
    def V_A_xy_t0_from_V_K_xy_t0(V_K_xy_t0, chi, V_W_xy, chi_W):
        return vt.V_A_xy_from_V_K_xy_chi(V_K_xy_t0, chi, V_W_xy, chi_W)

    @init_field("V_A_xy_tf")
    def V_A_xy_tf_from_V_K_xy_tf(V_K_xy_tf, chi, V_W_xy, chi_W):
        return vt.V_A_xy_from_V_K_xy_chi_A(V_K_xy_tf, chi, V_W_xy, chi_W)

    def x_g(self, t):
        dt = t - self.t0
        l_A_xy = l_A_xy_from_chi(
            dt, self.chi, self.V_A_xy_dot, self.V_A_xy_t0, self.V_W_xy,
            self.chi_W, self.eps)
        l_W_xy = l_W_xy_from_chi(dt, self.chi, self.V_W_xy, self.chi_W)
        # print("###", self.x_g_t0, l_A_xy, l_W_xy)
        return self.x_g_t0 + (l_A_xy+l_W_xy)*numpy.cos(self.chi)

    def y_g(self, t):
        dt = t - self.t0
        l_A_xy = l_A_xy_from_chi(
            dt, self.chi, self.V_A_xy_dot, self.V_A_xy_t0, self.V_W_xy,
            self.chi_W, self.eps)
        l_W_xy = l_W_xy_from_chi(dt, self.chi, self.V_W_xy, self.chi_W)
        return self.y_g_t0 + (l_A_xy+l_W_xy)*numpy.sin(self.chi)

    def l_xy(self, t):
        dt = t - self.t0
        l_A_xy = l_A_xy_from_chi(
            dt, self.chi, self.V_A_xy_dot, self.V_A_xy_t0, self.V_W_xy,
            self.chi_W, self.eps)
        l_W_xy = l_W_xy_from_chi(dt, self.chi, self.V_W_xy, self.chi_W)
        return self.l_xy_t0 + l_A_xy + l_W_xy

    def V_A_xy(self, t):
        dt = t - self.t0
        return self.V_A_xy_t0 + self.V_A_xy_dot*dt

    def chi_A(self, t):
        V_A_xy = self.V_A_xy(t)
        return vt.chi_A_from_V_A_xy_chi(
            V_A_xy, self.chi, self.V_W_xy, self.chi_W)

    def chi_A_dot(self, t):
        V_A_xy = self.V_A_xy(t)
        return km.chi_A_dot_from_chi_dot_V_A_xy_dot.eval_if(
            mask=self.V_W_xy > 0,
            fill_value=self.chi_dot,
            V_A_xy=V_A_xy,
            V_A_xy_dot=self.V_A_xy_dot,
            chi=self.chi,
            chi_dot=self.chi_dot,
            V_W_xy=self.V_W_xy,
            chi_W=self.chi_W)

    def u_A_g(self, t):
        V_A_xy = self.V_A_xy(t)
        chi_A = self.chi_A(t)
        return V_A_xy*numpy.cos(chi_A)

    def v_A_g(self, t):
        V_A_xy = self.V_A_xy(t)
        chi_A = self.chi_A(t)
        return V_A_xy*numpy.sin(chi_A)

    def V_K_xy(self, t):
        V_A_xy = self.V_A_xy(t)
        return vt.V_K_xy_from_V_A_xy_chi(
            V_A_xy, self.chi, self.V_W_xy, self.chi_W)

    def u_K_g(self, t):
        V_K_xy = self.V_K_xy(t)
        return V_K_xy*numpy.cos(self.chi)

    def v_K_g(self, t):
        V_K_xy = self.V_K_xy(t)
        return V_K_xy*numpy.sin(self.chi)

    def V_K_xy_dot(self, t):
        V_A_xy = self.V_A_xy(t)
        chi_A = self.chi_A(t)
        return km.V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, self.V_A_xy_dot, chi_A, 0, self.V_W_xy, self.chi_W)

    def u_K_g_dot(self, t):
        V_K_xy_dot = self.V_K_xy_dot(t)
        return V_K_xy_dot*numpy.cos(self.chi)

    def v_K_g_dot(self, t):
        V_K_xy_dot = self.V_K_xy_dot(t)
        return V_K_xy_dot*numpy.sin(self.chi)

    def u_A_g_dot(self, t):
        V_A_xy = self.V_A_xy(t)
        chi_A = self.chi_A(t)
        chi_A_dot = self.chi_A_dot(t)
        return u_A_g_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, self.V_A_xy_dot, chi_A, chi_A_dot)

    def v_A_g_dot(self, t):
        V_A_xy = self.V_A_xy(t)
        chi_A = self.chi_A(t)
        chi_A_dot = self.chi_A_dot(t)
        return v_A_g_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, self.V_A_xy_dot, chi_A, chi_A_dot)

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
                "chi": self.chi,
                "chi_dot": self.chi_dot,
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
                "V_A_xy": self.V_A_xy(t),
                "chi_A": self.chi_A(t),
                "chi_A_dot": self.chi_A_dot(t),
                "l_xy": self.l_xy(t),
            })
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        # Constants
        df["chi"] = self.chi
        df["chi_dot"] = self.chi_dot
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
        df["V_A_xy"] = self.V_A_xy(t_index)
        df["chi_A"] = self.chi_A(t_index)
        df["chi_A_dot"] = self.chi_A_dot(t_index)
        df["l_xy"] = self.l_xy(t_index)
        return DiscreteSegment(df=df)

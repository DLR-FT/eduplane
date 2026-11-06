# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Vertical profile with linear vertical speed.
"""

from dataclasses import field

from eduplane.model import velocity_triangle as vt
from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import VertProfile


@dataclass_greedy_init
class LinearVertSpeedProfile(VertProfile):

    # Initial/final values
    tf: float

    # Optional intial/final values
    t0: float = field(default=0)
    z_g_t0: float = field(default=0)
    w_A_g_t0: float = field(default=0)
    z_g_tf: float = field(default=None)
    w_A_g_tf: float = field(default=None)
    w_K_g_t0: float = field(default=None)
    w_K_g_tf: float = field(default=None)

    # Optional constants
    w_W_g: float = field(default=0)
    w_A_g_dot: float = field(default=None)
    w_K_g_dot: float = field(default=None)

    @init_field("w_A_g_dot")
    def w_A_g_dot_from_w_K_g_dot(w_K_g_dot):
        return w_K_g_dot

    @init_field("w_A_g_dot")
    def w_A_g_dot_from_tf(w_A_g_t0, w_A_g_tf, tf):
        return (w_A_g_tf-w_A_g_t0)/tf

    @init_field("w_K_g_dot")
    def w_K_g_dot_from_w_A_g_dot(w_A_g_dot):
        return w_A_g_dot

    @init_field("w_K_g_t0")
    def w_K_g_t0_from_w_A_g_t0_w_W_g(w_A_g_t0, w_W_g):
        return vt.w_K_g_from_w_A_g_w_W_g(w_A_g_t0, w_W_g)

    @init_field("w_K_g_tf")
    def w_K_g_tf_from_w_A_g_tf_w_W_g(w_A_g_tf, w_W_g):
        return vt.w_K_g_from_w_A_g_w_W_g(w_A_g_tf, w_W_g)

    @init_field("w_A_g_t0")
    def w_A_g_t0_from_w_K_g_t0_w_W_g(w_K_g_t0, w_W_g):
        return vt.w_A_g_from_w_K_g_w_W_g(w_K_g_t0, w_W_g)

    @init_field("w_A_g_tf")
    def w_A_g_tf_from_w_K_g_t0_w_W_g(w_K_g_tf, w_W_g):
        return vt.w_A_g_from_w_K_g_w_W_g(w_K_g_tf, w_W_g)

    @init_field("z_g_t0")
    def z_g_t0_from_z_g_tf(z_g_tf, w_A_g_t0, w_A_g_dot, w_W_g, tf):
        return z_g_tf - (w_A_g_t0+w_W_g)*tf - 0.5*w_A_g_dot*tf**2

    # @classmethod
    # def from_w_K_g(cls, tf, t0=0, z_g_t0=0, w_K_g_t0=0, w_K_g_dot=0, w_W_g=0):
    #     """Alternative intialization from initial vertical velocity and
    #     acceleration components, w_K_g and w_K_g_dot.
    #     """
    #     w_A_g_dot = w_K_g_dot
    #     w_A_g_t0 = vt.w_A_g_from_w_K_g_w_W_g(w_K_g_t0, w_W_g)
    #     return cls(
    #         tf,
    #         t0,
    #         z_g_t0,
    #         w_A_g_t0,
    #         w_A_g_dot=w_A_g_dot,
    #         w_W_g=w_W_g,
    #         w_K_g_t0=w_K_g_t0,
    #         w_K_g_dot=w_K_g_dot)

    def __post_init__(self):
        """Initialize from initial vertical airspeed component and constant
        air-relative acceleration acceleration, w_A_g_t0 and w_A_g_dot."""
        # Calculate derived constants if not user-specified
        if self.w_A_g_tf is None:
            self.w_A_g_tf = self.w_A_g(self.tf)
        if self.w_K_g_t0 is None:
            self.w_K_g_t0 = vt.w_K_g_from_w_A_g_w_W_g(self.w_A_g_t0, self.w_W_g)
        if self.w_K_g_tf is None:
            self.w_K_g_tf = vt.w_K_g_from_w_A_g_w_W_g(self.w_A_g_tf, self.w_W_g)
        if self.w_K_g_dot is None:
            self.w_K_g_dot = self.w_A_g_dot
        if self.z_g_tf is None:
            self.z_g_tf = self.z_g(self.tf)
        if self.w_A_g_tf is None:
            self.w_A_g_tf = self.w_A_g(self.tf)

    def z_g(self, t):
        """Calculate vertical position."""
        dt = (t - self.t0)
        return self.z_g_t0 + (
            self.w_A_g_t0 + self.w_W_g)*dt + 0.5*self.w_A_g_dot*dt**2

    def w_A_g(self, t):
        """Calculate vertical airspeed component."""
        dt = (t - self.t0)
        return self.w_A_g_t0 + self.w_A_g_dot*dt

    def w_K_g(self, t):
        """Calculate vertical airspeed component."""
        return self.w_A_g(t) + self.w_W_g

    def shift(self, new_t0, **new_initial_values):
        """Shift time paramter interval and initial values."""
        if self.contains_valid_value(new_initial_values, "z_g"):
            self.z_g_t0 = new_initial_values["z_g"]
        super().shift(new_t0)
        return self

    def evaluate(self, t):
        """Evaluate parametric states at given time parameter value."""
        self.check_parameter(t)
        values = self.get_empty_dict()
        values["w_A_g"] = self.w_A_g(t)
        values["w_A_g_dot"] = self.w_A_g_dot
        values["w_K_g"] = self.w_K_g(t)
        values["w_K_g_dot"] = self.w_K_g_dot
        values["w_W_g"] = self.w_W_g
        values["z_g"] = self.z_g(t)
        values["h_g"] = self.h_g(t)
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize within time parameter interval [t0, tf]."""
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        df["w_A_g"] = self.w_A_g(t_index)
        df["w_A_g_dot"] = self.w_A_g_dot
        df["w_K_g"] = self.w_K_g(t_index)
        df["w_K_g_dot"] = self.w_K_g_dot
        df["w_W_g"] = self.w_W_g
        df["z_g"] = self.z_g(t_index)
        df["h_g"] = self.h_g(t_index)
        return DiscreteSegment(df)

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Vertical profile with constant vertical speed.
"""

from dataclasses import field

from eduplane.model import velocity_triangle as vt
from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import VertProfile


@dataclass_greedy_init
class ConstVertSpeedProfile(VertProfile):

    # Initial/final values
    tf: float

    # Optional intial/final values
    t0: float = field(default=0)

    # Optional constants
    w_A_g: float = field(default=0)
    w_W_g: float = field(default=0)

    # Derived constants
    # (values calculated on initialization if not user-specified)
    w_K_g: float = field(default=None)
    z_g_t0: float = field(default=None)
    z_g_tf: float = field(default=None)

    # Fixed constants
    w_A_g_dot: float = field(init=False, default=0)
    w_K_g_dot: float = field(init=False, default=0)

    # @classmethod
    # def from_w_K_g(cls, tf, t0=0, z_g_t0=0, w_K_g=0, w_W_g=0):
    #     """Alternative intialization from vertical velocity component, w_K_g.
    #     """
    #     w_A_g = vt.w_A_g_from_w_K_g_w_W_g(w_K_g, w_W_g)
    #     return cls(tf, t0, z_g_t0, w_A_g, w_W_g, w_K_g)

    @init_field("w_A_g")
    def w_A_g_from_w_K_g_w_W_g(w_K_g, w_W_g):
        return vt.w_A_g_from_w_K_g_w_W_g(w_K_g, w_W_g)

    @init_field("w_K_g")
    def w_K_g_from_w_A_g_w_W_g(w_A_g, w_W_g):
        return vt.w_K_g_from_w_A_g_w_W_g(w_A_g, w_W_g)

    @init_field("w_K_g")
    def w_K_g_from_tf(z_g_t0, z_g_tf, t0, tf):
        if tf > 0:
            return (z_g_tf-z_g_t0)/(tf-t0)
        else:
            return 0

    @init_field("z_g_t0")
    def z_g_t0_from_z_g_tf(z_g_tf, w_K_g, t0, tf):
        return z_g_tf - w_K_g*(tf-t0)

    def __post_init__(self):
        """Initialize from vertical airspeed component, w_A_g."""
        # Calculate derived constants if not user-specified
        if self.w_K_g is None and self.w_A_g is not None:
            self.w_K_g = vt.w_K_g_from_w_A_g_w_W_g(self.w_A_g, self.w_W_g)
        if self.w_A_g is None and self.w_K_g is not None:
            self.w_A_g = vt.w_A_g_from_w_K_g_w_W_g(self.w_K_g, self.w_W_g)
        if self.z_g_tf is None:
            self.z_g_tf = self.z_g(self.tf)

    def z_g(self, t):
        """Calculate vertical position."""
        return self.z_g_t0 + self.w_K_g*(t - self.t0)

    def shift(self, new_t0, **new_initial_values):
        """Shift time paramter interval and initial values."""
        if self.contains_valid_value(new_initial_values, "z_g"):
            self.z_g_t0 = new_initial_values["z_g"]
            self.z_g_tf = self.z_g(self.tf)
        super().shift(new_t0)
        return self

    def evaluate(self, t):
        """Evaluate parametric states at given time parameter value."""
        self.check_parameter(t)
        values = self.get_empty_dict()
        values["w_A_g"] = self.w_A_g
        values["w_A_g_dot"] = self.w_A_g_dot
        values["w_K_g"] = self.w_K_g
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
        df["w_A_g"] = self.w_A_g
        df["w_A_g_dot"] = self.w_A_g_dot
        df["w_K_g"] = self.w_K_g
        df["w_K_g_dot"] = self.w_K_g_dot
        df["w_W_g"] = self.w_W_g
        df["z_g"] = self.z_g(t_index)
        df["h_g"] = self.h_g(t_index)
        return DiscreteSegment(df)

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""Base classes for representing trajectory segments."""

import math
from dataclasses import asdict, astuple

import numpy
import pandas


class SegmentBase(object):

    default_t_res = 0.1

    def __iter__(self):
        return iter(asdict(self))

    def items(self):
        return iter(asdict(self).items())

    def keys(self):
        return iter(asdict(self))

    def values(self):
        return iter(astuple(self))

    @property
    def duration(self):
        if self.tf is not None:
            return self.tf - self.t0
        else:
            return None

    @property
    def x0(self):
        return self.evaluate(self.t0)

    @property
    def xf(self):
        if self.tf is not None:
            return self.evaluate(self.tf)
        else:
            return None

    def get_time_index(self, t0=None, tf=None, t_res=None):
        if t_res is None:
            t_res = self.default_t_res
        if t0 is None:
            try:
                t0 = self.t0
            except AttributeError:
                t0 = 0
        if tf is None:
            try:
                tf = self.tf
            except AttributeError:
                tf = t0
        num_points = int(math.ceil((tf-t0)/t_res + 1))
        if num_points > 1e6:
            raise ValueError(
                f"Time index for interval [{t0}, {tf}]"
                f" at t_res={t_res} exceeds maximum length")
        return pandas.Index(numpy.linspace(t0, tf, num_points), name="t")

    def check_parameter(self, t):
        if t < self.t0 or t > self.tf:
            raise ValueError(
                f"Time parameter {t} not within segment's "
                f"parameter range [{self.t0, self.tf}]")

    def contains_valid_value(self, params, req_param):
        return req_param in params and not numpy.isnan(params[req_param])

    def update_missing(self, values, new_values):
        for k, v in new_values.items():
            if k not in values or values[k] is None or numpy.isnan(values[k]):
                values[k] = v

    def calculate_length(self, df, l_t0=None, l_tf=None):
        delta_x_g_squared = df["x_g"].diff().fillna(0).pow(2)
        delta_y_g_squared = df["y_g"].diff().fillna(0).pow(2)
        delta_z_g_squared = df["z_g"].diff().fillna(0).pow(2)
        delta_l_squared = delta_x_g_squared + delta_y_g_squared + delta_z_g_squared
        delta_l = delta_l_squared.pow(0.5)
        if l_t0 is not None:
            df["l"] = l_t0 + delta_l.cumsum()
            # df["l_xy"] = self.x0["l_xy"] + delta_l_xy.cumsum()
        else:
            df["l"] = l_tf - delta_l[::-1].cumsum().shift(1, fill_value=0)[::-1]
            # df["l"] = self.xf["l"] + delta_l.cumsum()[::-1]
            # df["l_xy"] = self.xf["l_xy"] - delta_l_xy[::-1].cumsum()

    def get_empty_dict(self):
        return dict(zip(self.states, [numpy.nan]*len(self.states)))

    def get_empty_dataframe(self, t_index):
        return pandas.DataFrame(index=t_index, columns=self.states, dtype=float)

    def evaluate(self, t):
        return {}

    def shift(self, new_t0):
        if self.tf is not None:
            self.tf = new_t0 + self.duration
        self.t0 = new_t0
        return self

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize the trajectory segment.

        This is a most basic and generic implementation that assumes all states
        to have parametric representations, i.e. no integration is required.

        Note that the non-vectorized 'evaluate' function is used. Better
        performance may be achieved by specializing this function and evaluate
        parametric states with vectorized operations.
        """
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        return pandas.DataFrame(
            [self.evaluate(t) for t in t_index], index=t_index)


class HorizProfile(SegmentBase):
    states = (
        "u_K_g",
        "v_K_g",
        "u_K_g_dot",
        "v_K_g_dot",
        "V_K_xy",
        "V_K_xy_dot",
        "chi",
        "chi_dot",
        "u_A_g",
        "v_A_g",
        "u_A_g_dot",
        "v_A_g_dot",
        "V_A_xy",
        "V_A_xy_dot",
        "chi_A",
        "chi_A_dot",
        "u_W_g",
        "v_W_g",
        "V_W_xy",
        "chi_W",
        "x_g",
        "y_g",
        "l_xy",
    )


class VertProfile(SegmentBase):
    states = (
        "w_K_g",
        "w_K_g_dot",
        "w_A_g",
        "w_A_g_dot",
        "z_g",
        "h_g",
        "w_W_g",
    )

    def h_g(self, t):
        return -self.z_g(t)


class AirspeedProfile(SegmentBase):
    states = (
        "V_A",
        "V_A_dot",
        "V_A_dot2",
    )


class LiftedSegmentBase(SegmentBase):
    states = (
        HorizProfile.states + VertProfile.states + AirspeedProfile.states + (
            "gamma_A", "gamma_A_dot", "gamma", "gamma_dot", "V_K", "V_K_dot",
            "V_W", "gamma_W", "l"))

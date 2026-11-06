# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A lifted trajectory segment, i.e. a horizontal segment combined with a
vertical profile to form a 3D trajectory."""

from dataclasses import dataclass, field

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt

from .discrete_segment import DiscreteSegment
from .segment_base import HorizProfile, LiftedSegmentBase, VertProfile


@dataclass
class LiftedSegment(LiftedSegmentBase):
    horiz_profile: HorizProfile
    vert_profile: VertProfile

    # Initial/final state values
    l_t0: float = field(default=0)

    @property
    def t0(self):
        return max(self.horiz_profile.t0, self.vert_profile.t0)

    @property
    def tf(self):
        return min(self.horiz_profile.tf, self.vert_profile.tf)

    @property
    def x0(self):
        x0 = self.get_empty_dict()
        x0.update(self.horiz_profile.x0)
        x0.update(self.vert_profile.x0)
        self.calculate_coupled_states(x0)
        x0["l"] = self.l_t0
        return x0

    def shift(self, new_t0, **new_initial_values):
        self.horiz_profile.shift(new_t0, **new_initial_values)
        self.vert_profile.shift(new_t0, **new_initial_values)
        if self.contains_valid_value(new_initial_values, "l"):
            self.l_t0 = new_initial_values["l"]
        return self

    @staticmethod
    def calculate_coupled_states(values):
        values["V_A"] = vt.V_A_from_V_A_xy_w_A_g(
            values["V_A_xy"], values["w_A_g"])
        values["V_A_dot"] = km.V_A_dot_from_V_A_xy_dot_w_A_g_dot(
            values["V_A_xy"], values["V_A_xy_dot"], values["w_A_g"],
            values["w_A_g_dot"])
        values["gamma_A"] = vt.gamma_A_from_V_A_w_A_g(
            values["V_A"], values["w_A_g"])
        values["gamma_A_dot"] = km.gamma_A_dot_from_V_A_xy_dot_w_A_g_dot(
            values["V_A"], values["V_A_xy_dot"], values["w_A_g_dot"],
            values["gamma_A"])

        values["V_K"] = vt.V_K_from_V_K_xy_w_K_g(
            values["V_K_xy"], values["w_K_g"])
        values["V_K_dot"] = km.V_K_dot_from_V_K_xy_dot_w_K_g_dot(
            values["V_K_xy"], values["V_K_xy_dot"], values["w_K_g"],
            values["w_K_g_dot"])
        values["gamma"] = vt.gamma_from_V_K_w_K_g.eval_if(
            values, mask=values["V_K"] > 0, fill_value=0)
        values["gamma_dot"] = km.gamma_dot_from_V_K_xy_dot_w_K_g_dot(
            values["V_K"], values["V_K_xy_dot"], values["w_K_g_dot"],
            values["gamma"])

        values["V_W"] = vt.V_W_from_V_W_xy_w_W_g(
            values["V_W_xy"], values["w_W_g"])
        values["gamma_W"] = vt.gamma_W_from_V_W_w_W_g.eval_if(
            values, mask=values["V_W"] > 0, fill_value=0)

        # NOTE: Lifted states with quadratic airspeed are currenlty not
        # supported as this would require extension of the horizontally and
        # vertically decoupled kinematic model. Thus, V_A_dot2 is currently
        # used only for segments which are re-parametrized with a quadratic
        # airspeed profile.
        values["V_A_dot2"] = 0
        return values

    def evaluate(self, t):
        """Evaluate the segment at the given time parameter value."""
        values = self.get_empty_dict()
        values.update(self.horiz_profile.evaluate(t))
        values.update(self.vert_profile.evaluate(t))
        self.calculate_coupled_states(values)
        return values

    def integrate(self, t_index=None, t_res=None, t0=None, tf=None):
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        df.update(self.horiz_profile.discretize(t_index).df)
        df.update(self.vert_profile.discretize(t_index).df)
        self.calculate_coupled_states(df)
        self.calculate_length(df, l_t0=self.x0["l"])
        return df

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize the segment at the given time parameter values,
        resolution or number of points."""
        return DiscreteSegment(self.integrate(t_index, t_res, t0, tf))

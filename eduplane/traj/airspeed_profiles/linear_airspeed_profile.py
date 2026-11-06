# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A linear airspeed profile."""

from dataclasses import field

import numpy

from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import AirspeedProfile


@dataclass_greedy_init
class LinearAirspeedProfile(AirspeedProfile):

    # Initial/final values
    V_A_t0: float
    tf: float

    # Optional initial/final values
    t0: float = field(default=0)

    # Optional constants
    V_A_dot: float = field(default=None)

    # Derived constants
    V_A_tf: float = field(default=None)

    # Pre-defined constants
    V_A_dot2: float = field(init=False, default=0)

    def __post_init__(self):
        """Initialize from final time, initial airspeed and airspeed rate."""
        if self.V_A_tf is None:
            self.V_A_tf = self.V_A(self.tf)

    @init_field("V_A_dot")
    def V_A_dot_from_tf(V_A_t0, V_A_tf, tf):
        # if (V_A_tf-V_A_t0)/(0.5*(V_A_dot_t0+V_A_dot_tf)) < 0:
        #     print(
        #         "####", "V_A_t0=", V_A_t0, "V_A_tf=", V_A_tf, "V_A_dot_t0",
        #         V_A_dot_t0, "V_A_dot_tf", V_A_dot_tf)
        return (V_A_tf-V_A_t0)/tf

    @init_field("tf")
    def tf_from_V_A_dot(V_A_t0, V_A_tf, V_A_dot):
        # if (V_A_tf-V_A_t0)/(0.5*(V_A_dot_t0+V_A_dot_tf)) < 0:
        #     print(
        #         "####", "V_A_t0=", V_A_t0, "V_A_tf=", V_A_tf, "V_A_dot_t0",
        #         V_A_dot_t0, "V_A_dot_tf", V_A_dot_tf)
        return (V_A_tf-V_A_t0)/V_A_dot

    @init_field("V_A_t0")
    def V_A_t0_from_V_A_tf(V_A_tf, V_A_dot, tf):
        return V_A_tf - V_A_dot*tf

    def V_A(self, t):
        """Calculate airspeed magnitude."""
        return self.V_A_t0 + self.V_A_dot*(
            numpy.clip(t, self.t0, self.tf) - self.t0)

    def shift(self, new_t0, **new_initial_values):
        """Shift time paramter interval and initial values."""
        super().shift(new_t0)
        return self

    def stretch(self, new_dt):
        """
        Stretch profile to new parameter range [t0, t0+dt], keeping constant
        the initial time as well as initial and final airspeed.

        Returns a new instance of QuadraticAirspeedProfile.
        """
        new_V_A_dot = (self.V_A_tf - self.V_A_t0)/new_dt
        return LinearAirspeedProfile(
            V_A_t0=self.V_A_t0,
            V_A_tf=self.V_A_tf,
            t0=self.t0,
            tf=self.t0 + new_dt,
            V_A_dot=new_V_A_dot)

    def evaluate(self, t):
        """Evaluate parametric states at given time parameter value."""
        self.check_parameter(t)
        values = self.get_empty_dict()
        values["V_A_dot2"] = self.V_A_dot2
        values["V_A_dot"] = self.V_A_dot if t <= self.tf else 0
        values["V_A"] = self.V_A(t)
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None, x0=None):
        """Discretize within time parameter interval [t0, tf]."""
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        df["V_A_dot2"] = self.V_A_dot2
        # Set V_A_dot to self.V_A_dot where index <= self.tf
        df["V_A_dot"] = 0
        df.loc[df.index <= self.tf, "V_A_dot"] = self.V_A_dot
        df["V_A"] = self.V_A(t_index)
        return DiscreteSegment(df)

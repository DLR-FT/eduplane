# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A quadratic airspeed profile."""

from dataclasses import field

from eduplane.utils.dataclass_greedy_init import (
    dataclass_greedy_init, init_field)

from ..discrete_segment import DiscreteSegment
from ..segment_base import AirspeedProfile


@dataclass_greedy_init
class QuadraticAirspeedProfile(AirspeedProfile):

    # Initial/final values
    V_A_t0: float
    tf: float

    # Optional initial/final values
    t0: float = field(default=0)

    # Optional constants
    V_A_dot2: float = field(default=0)
    eps: float = field(default=1e-9)

    # Derived constants
    V_A_tf: float = field(default=None)
    V_A_dot_t0: float = field(default=None)
    V_A_dot_tf: float = field(default=None)

    def __post_init__(self):
        """Initialize from final time, initial airspeed and derivatives."""
        if self.V_A_tf is None:
            self.V_A_tf = self.V_A(self.tf)
        if self.V_A_dot_tf is None:
            self.V_A_dot_tf = self.V_A_dot(self.tf)

    @init_field("V_A_dot2")
    def V_A_dot2_from_V_A_dot_tf(V_A_dot_t0, V_A_dot_tf, tf):
        if tf > 0:
            return (V_A_dot_tf-V_A_dot_t0)/tf
        else:
            return 0

    @init_field("tf")
    def tf_from_V_A_tf_V_A_dot_tf(V_A_t0, V_A_tf, V_A_dot_t0, V_A_dot_tf):
        # if (V_A_tf-V_A_t0)/(0.5*(V_A_dot_t0+V_A_dot_tf)) < 0:
        #     print(
        #         "####", "V_A_t0=", V_A_t0, "V_A_tf=", V_A_tf, "V_A_dot_t0",
        #         V_A_dot_t0, "V_A_dot_tf", V_A_dot_tf)
        if abs(V_A_dot_t0 + V_A_dot_tf) > 0:
            return (V_A_tf-V_A_t0)/(0.5*(V_A_dot_t0+V_A_dot_tf))
        else:
            return 0

    @init_field("V_A_dot_tf")
    def V_A_dot_tf_from_V_A_dot_t0(V_A_dot_t0, V_A_t0, V_A_tf, tf):
        if tf > 0:
            return 2*(V_A_tf-V_A_t0)/tf - V_A_dot_t0
        else:
            return 0

    @init_field("V_A_dot_t0")
    def V_A_dot_t0_from_V_A_dot_tf(V_A_dot_tf, V_A_t0, V_A_tf, tf):
        if tf > 0:
            return 2*(V_A_tf-V_A_t0)/tf - V_A_dot_tf
        else:
            return 0

    def V_A_dot(self, t):
        """Calculate airspeed magnitude."""
        dt = t - self.t0
        return self.V_A_dot_t0 + self.V_A_dot2*dt

    def V_A(self, t):
        """Calculate airspeed magnitude."""
        dt = t - self.t0
        return self.V_A_t0 + self.V_A_dot_t0*dt + 0.5*self.V_A_dot2*dt**2

    def shift(self, new_t0, **new_initial_values):
        """Shift time paramter interval and initial values."""
        super().shift(new_t0)
        return self

    def stretch(self, new_dt):
        """
        Stretch profile to new parameter range [t0, t0+dt], keeping constant
        the initial time, initial and final airspeed and either initial or
        final airspeed rate, depending on whichever one's absolute value is
        smaller.

        Returns a new instance of QuadraticAirspeedProfile.

        Formulas to calculate initial/final airspeed rate from final/initial
        airspeed rate depending on new parameter range dt:

        V_A_tf = V_A_t0 + V_A_dot_t0*dt + 0.5*V_A_dot2*dt^2
        V_A_dot2 = (V_A_dot_tf-V_A_dot_t0)/dt
        -> V_A_tf = V_A_t0 + V_A_dot_t0*dt + 0.5*(V_A_dot_tf - V_A_dot_t0)*dt
        -> V_A_dot_tf = 2*(V_A_tf - V_A_t0 - 0.5*V_A_dot_t0*dt) / dt

        V_A_t0 = V_A_tf - V_A_dot_tf*dt + 0.5*V_A_dot2*dt^2
        V_A_dot2 = (V_A_dot_tf - V_A_dot_t0)/dt
        -> V_A_t0 = V_A_tf - V_A_dot_tf*dt + 0.5*(V_A_dot_tf - V_A_dot_t0)*dt
        -> V_A_dot_t0 = 2*(V_A_tf - V_A_t0 - 0.5*V_A_dot_tf*dt) / dt

        """
        # Keep either V_A_dot_t0 or V_A_dot_tf constant, depending on which
        # one has a smaller absolute value.
        if abs(self.V_A_dot_t0) <= abs(self.V_A_dot_tf):
            new_V_A_dot_t0 = self.V_A_dot_t0
            new_V_A_dot_tf = 2*(
                self.V_A_tf - self.V_A_t0 - 0.5*self.V_A_dot_t0*new_dt)/new_dt
        else:
            new_V_A_dot_tf = self.V_A_dot_tf
            new_V_A_dot_t0 = 2*(
                self.V_A_tf - self.V_A_t0 - 0.5*self.V_A_dot_tf*new_dt)/new_dt
        new_V_A_dot2 = (new_V_A_dot_tf-new_V_A_dot_t0)/new_dt
        return QuadraticAirspeedProfile(
            t0=self.t0,
            tf=self.t0 + new_dt,
            V_A_t0=self.V_A_t0,
            V_A_tf=self.V_A_tf,
            V_A_dot_t0=new_V_A_dot_t0,
            V_A_dot_tf=new_V_A_dot_tf,
            V_A_dot2=new_V_A_dot2)

    def evaluate(self, t):
        """Evaluate parametric states at given time parameter value."""
        self.check_parameter(t)
        values = self.get_empty_dict()
        values["V_A_dot2"] = self.V_A_dot2
        values["V_A_dot"] = self.V_A_dot(t)
        values["V_A"] = self.V_A(t)
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None, x0=None):
        """Discretize within time parameter interval [t0, tf]."""
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        df = self.get_empty_dataframe(t_index)
        df["V_A_dot2"] = self.V_A_dot2
        df["V_A_dot"] = self.V_A_dot(t_index)
        df["V_A"] = self.V_A(t_index)
        return DiscreteSegment(df)

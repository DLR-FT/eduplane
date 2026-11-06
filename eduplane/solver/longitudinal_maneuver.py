# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""Single-segment transitional longitudinal maneuvers, i.e., trajectory
segments with varying airspeed and vertical speed.
"""

import math
from dataclasses import asdict, dataclass, field

import eduplane.model.kinematic_model as km
import eduplane.model.velocity_triangle as vt
from eduplane.model.ext_dubins_airplane import ExtDubinsAirplane
from eduplane.traj.horizontal_profiles.linear_speed_straight_flight import (
    LinearSpeedStraightFlight, tf_from_l_xy_tf_V_A_xy_tf)
from eduplane.traj.lifted_segment import LiftedSegment
from eduplane.traj.vertical_profiles.const_vert_speed_profile import (
    ConstVertSpeedProfile)
from eduplane.traj.vertical_profiles.linear_vert_speed_profile import (
    LinearVertSpeedProfile)
from eduplane.utils import utils


@dataclass
class LongitudinalManeuver(object):

    ac: ExtDubinsAirplane
    z_g_t0: float
    V_A_xy_t0: float
    w_A_g_t0: float
    chi: float
    V_W_xy: float
    chi_W: float
    t0: float = field(default=0)
    chi_A_dot_abs: float = field(default=0)
    V_A_xy_tf: float = field(default=None)
    w_A_g_tf: float = field(default=None)
    tf: float = field(default=None)

    def __post_init__(self):
        if self.V_A_xy_tf is None:
            self.V_A_xy_tf = self.V_A_xy_t0
        if self.w_A_g_tf is None:
            self.w_A_g_tf = self.w_A_g_t0
        if self.tf is None:
            self.tf = self.t0

    @property
    def z_g_tf(self):
        return self.z_g_t0 + 0.5*(self.w_A_g_t0 + self.w_A_g_tf)*self.duration

    @property
    def V_A_t0(self):
        return vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_t0, self.w_A_g_t0)

    @property
    def V_A_tf(self):
        return vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf)

    @property
    def w_A_g_dot(self):
        if self.duration > 0:
            return (self.w_A_g_tf - self.w_A_g_t0)/self.duration
        else:
            return 0

    @property
    def V_A_xy_dot(self):
        if self.duration > 0:
            return (self.V_A_xy_tf - self.V_A_xy_t0)/self.duration
        else:
            return 0

    @property
    def gamma_A_t0(self):
        return vt.gamma_A_from_V_A_w_A_g(self.V_A_t0, self.w_A_g_t0)

    @property
    def gamma_A_tf(self):
        return vt.gamma_A_from_V_A_w_A_g(self.V_A_tf, self.w_A_g_tf)

    @property
    def duration(self):
        return self.tf - self.t0

    @property
    def V_A_dot_t0(self):
        return km.V_A_dot_from_V_A_xy_dot_w_A_g_dot(
            self.V_A_xy_t0,
            self.V_A_xy_dot,
            self.w_A_g_t0,
            self.w_A_g_dot,
        )

    @property
    def V_A_dot_tf(self):
        return km.V_A_dot_from_V_A_xy_dot_w_A_g_dot(
            self.V_A_xy_tf,
            self.V_A_xy_dot,
            self.w_A_g_tf,
            self.w_A_g_dot,
        )

    @property
    def T_s_t0(self):
        return self.ac.thrust_required(
            V_A=self.V_A_t0,
            V_A_dot=0,
            w_A_g=self.w_A_g_t0,
            w_A_g_dot=0,
            chi_A_dot=self.chi_A_dot_abs)

    @property
    def T_s_tf(self):
        return self.ac.thrust_required(
            V_A=self.V_A_tf,
            V_A_dot=0,
            w_A_g=self.w_A_g_tf,
            w_A_g_dot=0,
            chi_A_dot=self.chi_A_dot_abs)

    @property
    def T_t0(self):
        return self.ac.thrust_required(
            V_A=self.V_A_t0,
            V_A_dot=self.V_A_dot_t0,
            w_A_g=self.w_A_g_t0,
            w_A_g_dot=self.w_A_g_dot,
            chi_A_dot=self.chi_A_dot_abs)

    @property
    def T_tf(self):
        return self.ac.thrust_required(
            V_A=self.V_A_tf,
            V_A_dot=self.V_A_dot_tf,
            w_A_g=self.w_A_g_tf,
            w_A_g_dot=self.w_A_g_dot,
            chi_A_dot=self.chi_A_dot_abs)

    def getRecord(self, prefix=""):
        return {
            f"{prefix}{key}": value
            for key, value in asdict(self).items() if value is not None
        }

    def getVerticalProfile(self):
        if self.w_A_g_dot is not None and self.w_A_g_dot != 0:
            return LinearVertSpeedProfile(
                z_g_t0=self.z_g_t0,
                z_g_tf=self.z_g_tf,
                w_A_g_t0=self.w_A_g_t0,
                w_A_g_dot=self.w_A_g_dot,
                t0=self.t0,
                tf=self.tf)
        else:
            return ConstVertSpeedProfile(
                z_g_t0=self.z_g_t0,
                z_g_tf=self.z_g_tf,
                w_A_g=self.w_A_g_t0,
                t0=self.t0,
                tf=self.tf)

    def getLiftedSegment(self, x_g_t0, y_g_t0):
        # Construct trajectory with initial final steady-state segments
        horiz_profile = LinearSpeedStraightFlight(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=self.tf,
            V_A_xy_t0=self.V_A_xy_t0,
            V_A_xy_dot=self.V_A_xy_dot,
            chi=self.chi,
            V_W_xy=self.V_W_xy,
            chi_W=self.chi_W)
        vert_profile = self.getVerticalProfile()
        return LiftedSegment(
            horiz_profile=horiz_profile,
            vert_profile=vert_profile,
        )

    @classmethod
    def solve_rapid_climb_fixed_length_fixed_airspeed(
            cls, ac: ExtDubinsAirplane, V_A_xy_t0, w_A_g_t0, z_g_t0, chi,
            V_W_xy, chi_W, xy_dist):

        self = cls(
            z_g_t0=z_g_t0,
            V_A_xy_t0=V_A_xy_t0,
            w_A_g_t0=w_A_g_t0,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W,
            ac=ac,
        )

        # Calculate max. climb rate for current goal airspeed
        w_A_g_climb, V_A_climb = ac.descent_rate_min(
            V_A=vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf))
        # print(f"{w_A_g_climb=}, {V_A_climb=}, {vt.V_A_xy_from_V_A_w_A_g(
        #         V_A_climb, w_A_g_climb)}")
        # Update goal horizontal and vertical airspeed components
        self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(V_A_climb, w_A_g_climb)
        self.w_A_g_tf = w_A_g_climb
        # Update final time
        self.tf = tf_from_l_xy_tf_V_A_xy_tf(
            0, xy_dist, self.V_A_xy_t0, self.V_A_xy_tf, self.chi, self.V_W_xy,
            self.chi_W)

        return self

    @classmethod
    def solve_rapid_climb_fixed_length(
            cls,
            ac: ExtDubinsAirplane,
            V_A_xy_t0,
            w_A_g_t0,
            z_g_t0,
            chi,
            V_W_xy,
            chi_W,
            xy_dist,
            T_dec=0,
            gamma_A_min=None):

        self = cls(
            z_g_t0=z_g_t0,
            V_A_xy_t0=V_A_xy_t0,
            w_A_g_t0=w_A_g_t0,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W,
            ac=ac,
        )

        V_A_tf_min = self.V_A_t0
        V_A_tf_max = self.ac.V_A_max
        while True:
            # Calculate max. climb rate for current goal airspeed
            w_A_g_climb, V_A_climb = ac.descent_rate_min(
                V_A=vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf))
            # print(f"{w_A_g_climb=}, {V_A_climb=}, {vt.V_A_xy_from_V_A_w_A_g(
            #      V_A_climb, w_A_g_climb)}")
            # Update goal horizontal and vertical airspeed components
            self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(V_A_climb, w_A_g_climb)
            self.w_A_g_tf = w_A_g_climb
            # Update final time
            self.tf = tf_from_l_xy_tf_V_A_xy_tf(
                0, xy_dist, self.V_A_xy_t0, self.V_A_xy_tf, self.chi,
                self.V_W_xy, self.chi_W)

            # Break if thrust is smooth enough
            # if abs(self.T_s_t0 - self.T_t0) < delta_T or abs(V_A_tf_max-V_A_tf_min) < 0.1:
            if abs(V_A_tf_max - V_A_tf_min) < 0.1:
                break
            # Adapt goal airspeed to reduce or increase thrust at t0
            if self.T_t0 + T_dec > self.T_s_t0:
                V_A_tf_max = self.V_A_tf
            else:
                V_A_tf_min = self.V_A_tf
            self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(
                V_A_tf_min + 0.5*(V_A_tf_max-V_A_tf_min), self.w_A_g_tf)

        return self

    @classmethod
    def solve_rapid_descent_fixed_length_fixed_airspeed(
            cls,
            ac: ExtDubinsAirplane,
            V_A_xy_t0,
            w_A_g_t0,
            z_g_t0,
            chi,
            V_W_xy,
            chi_W,
            xy_dist,
            T_min=None,
            gamma_A_min=None,
            V_A_xy_tf=None,
            V_A_eps=1e-3):

        if T_min is None:
            T_min = ac.T_min

        self = cls(
            z_g_t0=z_g_t0,
            V_A_xy_t0=V_A_xy_t0,
            V_A_xy_tf=V_A_xy_tf,
            w_A_g_t0=w_A_g_t0,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W,
            ac=ac,
        )

        # Cap final airspeed at the terminal-dive airspeed for T_min: above
        # this airspeed no steady vertical speed exists (the aircraft
        # decelerates even in a vertical dive).
        V_A_terminal = ac.terminal_dive_airspeed(T_min=T_min)

        V_A_tf_values = []
        while not utils.converged(V_A_tf_values, abs_tol=V_A_eps):
            V_A_tf = vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf)

            V_A_dot = min(
                0.0, (V_A_tf - self.V_A_t0)/self.tf) if self.tf > 0 else 0

            # Calculate max. descent rate for current goal airspeed, capped at
            # the terminal-dive airspeed (above it no steady vertical speed
            # exists).
            V_A_descent = min(V_A_tf, V_A_terminal)

            try:
                w_A_g_descent = ac.vert_speed(
                    V_A=V_A_descent, T=T_min, V_A_dot=V_A_dot)
            except ValueError:
                # No equilibrium exists for this airspeed rate (e.g. near the
                # terminal-dive airspeed while accelerating). Fall back to the
                # steady (V_A_dot=0) descent rate.
                w_A_g_descent = ac.vert_speed(V_A=V_A_descent, T=T_min)
            # w_A_g_descent, V_A_descent = ac.descent_rate_max(
            #     V_A=V_A_tf, T_min=T_min)
            # print(f"{w_A_g_descent=}, {V_A_descent=}, {vt.V_A_xy_from_V_A_w_A_g(
            #         V_A_descent, w_A_g_descent)}")

            # Enforce max. descent angle constraint
            if gamma_A_min is not None:
                w_A_g_descent_max = vt.w_A_g_from_V_A_gamma_A(
                    V_A_tf, gamma_A_min)
                w_A_g_descent = min(w_A_g_descent, w_A_g_descent_max)

            # # Update goal horizontal and vertical airspeed components
            if V_A_xy_tf is None:
                self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(V_A_tf, w_A_g_descent)
            self.w_A_g_tf = w_A_g_descent
            # Update final time
            self.tf = tf_from_l_xy_tf_V_A_xy_tf(
                0, xy_dist, self.V_A_xy_t0, self.V_A_xy_tf, self.chi,
                self.V_W_xy, self.chi_W)

            # For constant airspeed case, no iteration is required as V_A_tf
            # was already known above based on state at t0
            if V_A_xy_tf is None:
                break
            else:
                V_A_tf_values.append(V_A_tf)

        return self

    @classmethod
    def solve_rapid_descent_fixed_length(
            cls,
            ac: ExtDubinsAirplane,
            V_A_xy_t0,
            w_A_g_t0,
            z_g_t0,
            chi,
            V_W_xy,
            chi_W,
            xy_dist,
            T_min=None,
            T_max=None,
            gamma_A_min=-math.radians(60),
            V_A_xy_tf=None,
            V_A_eps=1e-3):
        """Iteratively calculate final airspeed and descent rate over a fixed-length segment
        """
        self = cls(
            z_g_t0=z_g_t0,
            V_A_xy_t0=V_A_xy_t0,
            w_A_g_t0=w_A_g_t0,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W,
            ac=ac,
        )

        if V_A_xy_tf is not None:
            self.V_A_xy_tf = V_A_xy_tf

        if T_min is None:
            T_min = self.ac.T_min

        if T_max is None:
            T_max = self.T_s_t0

        V_A_tf_min = self.V_A_t0
        V_A_tf_max = self.ac.V_A_max
        # Cap final airspeed at the terminal-dive airspeed for T_min: above
        # this airspeed no steady vertical speed exists (the aircraft
        # decelerates even in a vertical dive).
        V_A_terminal = ac.terminal_dive_airspeed(T_min=T_min)
        V_A_tf_values = [0]
        while True:
            V_A_tf = vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf)
            V_A_dot = min(
                0.0, (V_A_tf - self.V_A_t0)/self.tf) if self.tf > 0 else 0

            # Calculate max. descent rate for current goal airspeed
            V_A_descent = min(
                vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf),
                V_A_terminal)
            try:
                w_A_g_descent = ac.vert_speed(
                    V_A=V_A_descent, T=T_min, V_A_dot=V_A_dot)
            except ValueError:
                # No equilibrium exists for this airspeed rate (e.g. near the
                # terminal-dive airspeed while accelerating). Fall back to the
                # steady (V_A_dot=0) descent rate.
                w_A_g_descent = ac.vert_speed(V_A=V_A_descent, T=T_min)
            # w_A_g_descent, V_A_descent = ac.descent_rate_max(
            #     V_A=vt.V_A_from_V_A_xy_w_A_g(self.V_A_xy_tf, self.w_A_g_tf), T_min=T_min)
            # print(f"{w_A_g_descent=}, {V_A_descent=}, {vt.V_A_xy_from_V_A_w_A_g(
            #      V_A_descent, w_A_g_descent)}")

            # Update goal horizontal and vertical airspeed components
            # Enforce max. climb angle constraint and max. airspeed constraints
            if V_A_xy_tf is None:
                self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(
                    V_A_descent, w_A_g_descent)
                V_A_xy_tf_min = vt.V_A_xy_from_V_A_gamma_A(
                    V_A_tf_min, gamma_A_min)
                V_A_xy_tf_max = V_A_tf_max
                self.V_A_xy_tf = max(self.V_A_xy_tf, V_A_xy_tf_min)
                self.V_A_xy_tf = min(self.V_A_xy_tf, V_A_xy_tf_max)

            # Enforce max. descent angle constraint
            self.w_A_g_tf = w_A_g_descent
            w_A_g_tf_max = vt.w_A_g_from_V_A_xy_gamma_A(
                self.V_A_xy_tf - V_A_eps, gamma_A_min)
            self.w_A_g_tf = min(self.w_A_g_tf, w_A_g_tf_max)

            # Update final time
            self.tf = tf_from_l_xy_tf_V_A_xy_tf(
                0, xy_dist, self.V_A_xy_t0, self.V_A_xy_tf, self.chi,
                self.V_W_xy, self.chi_W)

            # Break if final airspeed converged
            if abs(V_A_tf_max - V_A_tf_min) < V_A_eps:
                break
            if utils.converged(V_A_tf_values, V_A_eps):
                break

            # Adapt final airspeed to reduce or increase thrust at t0
            if self.T_t0 > T_max:
                # Thrust at t0 too high, reduce final airspeed
                V_A_tf_max = self.V_A_tf
            else:
                # Thrust at t0 too low, increase final airspeed
                V_A_tf_min = self.V_A_tf
            # print(f"{self.T_tf=:.3f} {T_min=:.3f}")
            # print(f"{V_A_tf_min=:.3f} <= {self.V_A_xy_tf=:.3f} <= {V_A_tf_max=:.3f}")
            self.V_A_xy_tf = vt.V_A_xy_from_V_A_w_A_g(
                V_A_tf_min + 0.5*(V_A_tf_max-V_A_tf_min), self.w_A_g_tf)
            V_A_tf_values.append(self.V_A_tf)

        return self

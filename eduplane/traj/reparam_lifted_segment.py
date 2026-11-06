# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
A trajectory segment that is re-parametrized based on an airspeed profile and
lifted based on a vertical profile.

The parametrization of both the vertical profile and the airspeed profile are
preserved, i.e. the vertical speed and airspeed of the resulting trajectory
will be identical to the input profiles. The horizontal profile is
re-parametrized based on the resulting horizontal airspeed component.

The general scheme of the combined lifting and re-parametrization algorithm is:

- Initialize the reparametrized and lifted state at t_i
- For each time step (t_i, t_i+1):

  - calculate linear approximation of the corresponding horizontal distance
    travelled based on the horizontal velocity component at t_i
  - calculate the corresponding time step (tau_i, tau_i+1) in the original
    parametrization of the horizontal path (tau) by solving the integral of
    the first-order taylor series for t (with the original segment's values)
  - calculate the reparametrized and lifted state at t_i+1

This numerical integration and re-parametrization can be performed forward or
backwards direction. For forward integration, the profiles initial times are
aligned and the resulting final state and time are computed. For backward
integration, the profiles final times are aligned and the resulting initial
state and time are computed. By default, the initial time and length are set
to zero for both forward and backward integration.

Alignment of the input profiles is not enforced:

- if the given airspeed and vertical profiles are too short, the lifted segment
  will be truncated, i.e. not cover the whole horizontal path.
- if the horizontal path is too short, the airspeed and vertical profile will
  be truncated, i.e. final airspeed and altitude may not be reached.
- an iterative procedure is given to calculate fitting constant or linear
  airspeed and vertical profiles
"""

# TODO: For long trajectories, accuracy of absolute coordinates may degrade.
# This could probably be improvied by integrating integrating segment-wise and
# keeping segment start and end points fixed.

from dataclasses import dataclass, field

import numpy
import pandas

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.traj.reparam_segment import dl_xy_from_dt, dt_from_dl_xy
from eduplane.utils import utils

from .discrete_segment import DiscreteSegment
from .segment_base import (
    AirspeedProfile, HorizProfile, LiftedSegmentBase, VertProfile)


@dataclass
class ReparamLiftedSegment(LiftedSegmentBase):
    horiz_profile: HorizProfile
    vert_profile: VertProfile
    airspeed_profile: AirspeedProfile

    # Initial/final values
    t0_reparam: float = field(init=False, default=None)
    tf_reparam: float = field(init=False, default=None)
    _x0: dict = field(init=False, default_factory=lambda: {"l": None})
    _xf: dict = field(init=False, default_factory=lambda: {"l": None})

    # Constants
    t_res: float = LiftedSegmentBase.default_t_res
    align_left: bool = True

    # Init-only variables

    @property
    def x0(self):
        return self._x0

    @property
    def xf(self):
        return self._xf

    @property
    def t0(self):
        return self.t0_reparam

    @property
    def tf(self):
        return self.tf_reparam

    @property
    def l_t0(self):
        return self._x0["l"] if self._x0["l"] is not None else 0

    @property
    def l_tf(self):
        return self._xf["l"] if self._xf["l"] is not None else 0

    @classmethod
    def align_airspeed_and_vertical_profiles(
            cls, original_segment, vertical_func, airspeed_func, t_res=1e-1):
        tf = [original_segment.tf]
        aligned = False
        while not utils.converged(tf, abs_tol=t_res**2) or not aligned:
            reparam_lifted_segment = cls(
                original_segment,
                vertical_func(tf[-1] - original_segment.t0),
                airspeed_func(tf[-1] - original_segment.t0),
                t_res=t_res)

            aligned = reparam_lifted_segment.check_alignment(t_res**2)

            tf.append(
                reparam_lifted_segment.tf_reparam +
                (original_segment.tf - reparam_lifted_segment.xf["t_orig"]))
            # print(f"  --> new tf={tf[-1]}")
        return reparam_lifted_segment

    @staticmethod
    def apply_airspeed_and_vertical_profile(
            x, vert_prof, airspeed_prof, full_state=True):
        """For a state or an array of states of a horizontal profile, apply an
        airspeed and a vertical profile.

        The horizontal profile geometry remains unchanged and also the
        vertical speed and airspeed profiles over time remain unchanged.
        In order to achieve this, the two profiles must be applied in
        combination, i.e. the step of lifting and re-parametrization cannot
        be performed sequentially.
        """
        V_K_xy_orig = x["V_K_xy"]
        chi_dot_orig = x["chi_dot"]
        # Assign airspeed profile states.
        for state in AirspeedProfile.states:
            x[state] = airspeed_prof[state]
        # Assign vertical profile states
        for state in VertProfile.states:
            x[state] = vert_prof[state]

        # Calculate airspeed-dependent states while keeping chi from horizontal
        # profile and states of vertical profile (w_K_g, w_K_g_dot, w_A_g,
        # w_A_g_dot)
        x["gamma_A"] = vt.gamma_A_from_V_A_w_A_g(x["V_A"], x["w_A_g"])
        x["V_A_xy"] = vt.V_A_xy_from_V_A_gamma_A(x["V_A"], x["gamma_A"])
        x["V_K_xy"] = vt.V_K_xy_from_V_A_xy_chi(
            x["V_A_xy"], x["chi"], x["V_W_xy"], x["chi_W"])
        x["chi_A"] = vt.chi_A_from_V_A_xy_chi(
            x["V_A_xy"], x["chi"], x["V_W_xy"], x["chi_W"])

        # The curvature of the trajectory's projection into the horizontal
        # plane remains constant, so we can calculate values for chi_dot
        # based on the ratio between original and new horizontal velocity
        # component.
        #
        # Note that for the angular velocity in the horizontal plane (i.e.
        # chi_dot), the horizontal velocity component V_K_xy is used, while for
        # the angular velocity in the vertical plane the velocity magnitude
        # V_K would be used, as the instantaneous vertical plane is rotated
        # with the velocity vector.
        #
        # chi_dot_orig = V_K_xy_orig*kappa_xy
        # -> chi_dot = V_K_xy*kappa_xy = V_K_xy*chi_dot_orig/V_K_xy_orig
        # gamma_dot_orig = V_K_orig*kappa_z
        # -> gamma_dot = V_K*kappa_z = V_K*gamma_dot_orig/V_K_orig
        x["chi_dot"] = x["V_K_xy"]*chi_dot_orig/V_K_xy_orig
        # From chi_dot and V_A_xy_dot, we can calculate other derivatives of
        # states in the horizontal plane.
        x["V_A_xy_dot"] = km.V_A_xy_dot_from_V_A_dot_w_A_g_dot(
            x["V_A"], x["V_A_dot"], x["w_A_g"], x["w_A_g_dot"])
        x["chi_A_dot"] = km.chi_A_dot_from_chi_dot_V_A_xy_dot.eval_if(
            mask=x["V_W_xy"] > 0,
            fill_value=x["chi_dot"],
            V_A_xy=x["V_A_xy"],
            V_A_xy_dot=x["V_A_xy_dot"],
            chi=x["chi"],
            chi_dot=x["chi_dot"],
            V_W_xy=x["V_W_xy"],
            chi_W=x["chi_W"])
        x["V_K_xy_dot"] = km.V_K_xy_dot_from_chi_dot_chi_A_dot.eval_if(
            mask=x["V_W_xy"] > 0 and numpy.abs(x["chi_W"] - x["chi_A"]) > 0,
            fill_value=x["V_A_xy_dot"],
            V_K_xy=x["V_K_xy"],
            chi_dot=x["chi_dot"],
            chi_A=x["chi_A"],
            chi_A_dot=x["chi_A_dot"],
            V_W_xy=x["V_W_xy"],
            chi_W=x["chi_W"])

        if full_state:
            # Calculate total wind speed and wind vector inclination angle
            x["V_W"] = vt.V_W_from_V_W_xy_w_W_g(x["V_W_xy"], x["w_W_g"])
            x["gamma_W"] = vt.gamma_W_from_V_W_w_W_g.eval_if(
                x, mask=x["V_W"] > 0, fill_value=0)

            # Calculate remaining speed components and velocity
            x["u_K_g"] = vt.u_K_g_from_V_K_xy(x["V_K_xy"], x["chi"])
            x["v_K_g"] = vt.v_K_g_from_V_K_xy(x["V_K_xy"], x["chi"])
            x["u_A_g"] = vt.u_A_g_from_V_A_xy(x["V_A_xy"], x["chi_A"])
            x["v_A_g"] = vt.v_A_g_from_V_A_xy(x["V_A_xy"], x["chi_A"])
            x["V_K"] = vt.V_K_from_V_K_xy_w_K_g(x["V_K_xy"], x["w_K_g"])

            # Now we can get gamma_dot, gamma_A_dot and finally V_K_dot is obtained
            # from the 3D velocity triangle solution
            x["gamma"] = vt.gamma_from_V_K_w_K_g(x["V_K"], x["w_K_g"])
            x["gamma_dot"] = km.gamma_dot_from_V_K_xy_dot_w_K_g_dot(
                x["V_K"], x["V_K_xy_dot"], x["w_K_g_dot"], x["gamma"])
            x["V_K_dot"] = km.V_K_dot_from_V_A_dot(
                x["V_A"], x["V_A_dot"], x["chi"], x["chi_dot"], x["gamma"],
                x["gamma_dot"], x["V_W"], x["chi_W"], x["gamma_W"])
            # x["gamma_A_dot"] = km.gamma_A_dot_from_V_K_dot_V_A_dot_gamma_dot(
            #     x["V_K"], x["V_K_dot"], x["gamma"], x["gamma_dot"], x["V_A"],
            #     x["V_A_dot"], x["V_W"], x["gamma_W"])
            x["gamma_A_dot"] = km.gamma_A_dot_from_V_A_dot_w_A_g_dot(
                x["V_A"], x["V_A_dot"], x["w_A_g_dot"], x["gamma_A"])

            # Calculate derivatives
            x["u_A_g_dot"] = km.u_A_g_dot_from_V_A_dot(
                x["V_A"], x["V_A_dot"], x["chi_A"], x["chi_A_dot"],
                x["gamma_A"], x["gamma_A_dot"])
            x["v_A_g_dot"] = km.v_A_g_dot_from_V_A_dot(
                x["V_A"], x["V_A_dot"], x["chi_A"], x["chi_A_dot"],
                x["gamma_A"], x["gamma_A_dot"])
            x["u_K_g_dot"] = x["u_A_g_dot"]
            x["v_K_g_dot"] = x["v_A_g_dot"]

        return x

    def __post_init__(self):
        if self.align_left:
            # Make all profiles start at horizontal path t0 / tau_0.
            # This avoids shifting the horizontal path such that t_orig relates
            # to the horizontal profile's parametrization.
            self.t0_reparam = self.horiz_profile.t0
            self.airspeed_profile.shift(self.t0)
            self.vert_profile.shift(self.t0)
            self._x0.update(
                self.apply_airspeed_and_vertical_profile(
                    self.horiz_profile.x0,
                    self.vert_profile.x0,
                    self.airspeed_profile.x0,
                    full_state=True))
            self._x0["l_xy"] = self.horiz_profile.x0["l_xy"]
            self._x0["t_orig"] = self.horiz_profile.t0
            df, _ = self.integrate(full_state=False)
            # Set initial and final time and states
            self.tf_reparam = df.index[-1]
            if self._x0["l"] is None:
                self._x0["l"] = 0
            self._xf = df.iloc[-1].to_dict()
            self._xf = self.apply_airspeed_and_vertical_profile(
                self._xf, self._xf, self._xf, full_state=True)
        else:
            # Make all profiles end at horizontal path tf / tau_f.
            self.tf_reparam = self.horiz_profile.tf
            self.airspeed_profile.shift(
                self.tf - self.airspeed_profile.duration)
            self.vert_profile.shift(self.tf - self.vert_profile.duration)
            self._xf.update(
                self.apply_airspeed_and_vertical_profile(
                    self.horiz_profile.xf,
                    self.vert_profile.xf,
                    self.airspeed_profile.xf,
                    full_state=True))
            self._xf["l_xy"] = self.horiz_profile.xf["l_xy"]
            self._xf["t_orig"] = self.horiz_profile.tf
            df, _ = self.integrate_backwards(full_state=False)
            self._x0 = df.iloc[0].to_dict()
            self._x0 = self.apply_airspeed_and_vertical_profile(
                self._x0, self._x0, self._x0, full_state=True)
            # If no final length is given, set final length to absolute length, i.e.
            # length starts at l_t0=0
            if self.xf["l"] is None:
                self._x0["l"] = 0
                self._xf["l"] = df["l"].iloc[-1] - df["l"].iloc[0]
            self.t0_reparam = df.index[0]
            self.shift(new_t0=0)

    def check_alignment(self, t_eps=1e-1):
        """Check if the airspeed profile and the original segment are aligned
        such that the re-parametrized semgent's final state aligns with both
        profiles' final states.
        """
        return abs(self.vert_profile.tf - self.tf_reparam) <= t_eps and abs(
            self.airspeed_profile.tf - self.tf_reparam) <= t_eps and abs(
                self._xf["t_orig"] - self.horiz_profile.tf) <= t_eps

    def shift(self, new_t0, **new_initial_values):
        changed_initial_values = {
            k: v
            for k, v in new_initial_values.items()
            if k in self._x0 and self._x0[k] != v
        }
        if self.align_left:
            duration = self.duration
            self.t0_reparam = new_t0
            self.tf_reparam = new_t0 + duration
            if self.contains_valid_value(changed_initial_values, "l"):
                if "l" in self._x0 and "l" in self._xf:
                    length = self._xf["l"] - self._x0["l"]
                    self._x0["l"] = changed_initial_values["l"]
                    self._xf["l"] = self._x0["l"] + length
                else:
                    # Length has not (yet) been calculated
                    self._x0["l"] = changed_initial_values["l"]
            # Make sure all profiles start at same t0 and pass on any other
            # initial values
            self.horiz_profile.shift(self.t0_reparam, **changed_initial_values)
            self.vert_profile.shift(self.t0_reparam, **changed_initial_values)
            self.airspeed_profile.shift(
                self.t0_reparam, **changed_initial_values)
        else:
            duration = self.duration
            self.tf_reparam = new_t0 + duration
            self.t0_reparam = new_t0
            if self.contains_valid_value(changed_initial_values, "l"):
                if "l" in self._x0 and "l" in self._xf:
                    length = self._xf["l"] - self._x0["l"]
                    self._x0["l"] = changed_initial_values["l"]
                    self._xf["l"] = self._x0["l"] + length
                else:
                    # Length has not (yet) been calculated
                    self._x0["l"] = changed_initial_values["l"]
            # Make sure all profiles start at same tf and pass on any other
            # initial values
            self.horiz_profile.shift(
                self.tf_reparam - self.horiz_profile.duration,
                **changed_initial_values)
            # NOTE: Airspeed and vertical profile durations are equal or longer
            # than this segment's duration. If durations are equal, we need to
            # numerically sanitize the result of the subtraction below to avoid
            # time ranges that do not include [t0_reparam, tf_reparam].
            self.airspeed_profile.shift(
                min(
                    self.tf_reparam - self.airspeed_profile.duration,
                    self.t0_reparam), **changed_initial_values)
            self.vert_profile.shift(
                min(
                    self.tf_reparam - self.vert_profile.duration,
                    self.t0_reparam), **changed_initial_values)
        return self

    def evaluate_reparam(self, t, t_xy, full_state=True):
        xy = self.horiz_profile.evaluate(t_xy)
        if not t >= self.vert_profile.t0 and t <= self.vert_profile.tf:
            raise ValueError("t must be in the range of the vertical profile")
        z = self.vert_profile.evaluate(t)
        va = self.airspeed_profile.evaluate(t)
        return self.apply_airspeed_and_vertical_profile(xy, z, va, full_state)

    def evaluate(self, t):
        return {}

    def integrate(self, t_index=None, t_eps=1e-12, full_state=True):
        """Perform lifting of horizontal profile and - at the same time -
        re-parametrization based on lifted states.

        NOTE: The algorithm is essentially the same as in
        ReparamSegment.integrate, however, re-parametrization is done based on
        the horizontal velocity component and horizontal profile distances.
        """
        # Create time index if none is given
        if t_index is None:
            t_index = self.get_time_index(
                self.t0, self.t0 +
                min(self.vert_profile.duration, self.airspeed_profile.duration),
                self.t_res)

        # For forward integration, t_index must start at left-aligned t0
        if t_index[0] != self.t0:
            raise ValueError(
                "Forward integration requires to start at left-aligned t0 value"
            )

        tau_array = [self.horiz_profile.t0]
        xy_tau1 = self.horiz_profile.x0
        x_array = [self._x0]

        # Handle case of zero-length segments (t_index has length 1)
        if len(t_index) == 1:
            return pandas.DataFrame(x_array, index=t_index), t_index

        # Evaluate original and new time parameter values and states
        for t1, t2 in zip(t_index, t_index[1:]):
            # print(f"t1={t1}, t2={t2}")
            # TODO: Use higher-order approx. with derivatives of V_K
            dt = t2 - t1
            # TODO: Implement this equation in terms of an smath function
            # Approximate horizontal segment length dl_xy:
            dl_xy = dl_xy_from_dt(
                dt, x_array[-1]["V_K_xy"], x_array[-1]["V_K_xy_dot"])
            # Approximate corresponding interval in original parametrization
            # dtau:
            dtau = dt_from_dl_xy(
                dl_xy, xy_tau1["V_K_xy"], xy_tau1["V_K_xy_dot"])
            # print(
            #     f"\nt1={t1}, t2={t2}, tau1={tau_array[-1]}, tau2={tau_array[-1]+dtau} "
            #     f"dt={dt}, dtau={dtau}, tf={self.vert_profile.tf}, tau_f={self.horiz_profile.tf} "
            #     f"dl_xy={dl_xy}")

            # If this timestep would overshoot the horizontal profile's final
            # time, make it smaller and recalculate the target t2 value.
            if tau_array[-1] + dtau >= self.horiz_profile.tf - t_eps:
                # print(
                #     f"Time step would exceed the horizontal profile final time tau_f={self.horiz_profile.tf}"
                # )
                dtau = self.horiz_profile.tf - tau_array[-1]
                dl_xy = dl_xy_from_dt(
                    dtau, xy_tau1["V_K_xy"], xy_tau1["V_K_xy_dot"])
                dt_new = dt_from_dl_xy(
                    dl_xy, x_array[-1]["V_K_xy"], x_array[-1]["V_K_xy_dot"])
                # NOTE: As rounding errors may occur, the new t2 value may
                # have to be clipped to the valid range.
                assert (t1 + dt_new <= t2 + 1e-6*(t2-t1))
                t2 = min(t2, t1 + dt_new)
                tau_array.append(self.horiz_profile.tf)
                x_array.append(
                    self.evaluate_reparam(t2, tau_array[-1], full_state))
                break

            # Calculate next reparametrized and original state at t2/t2_xy
            # (i.e. states at next iteration's t1)
            tau_array.append(tau_array[-1] + dtau)
            x_array.append(self.evaluate_reparam(t2, tau_array[-1], full_state))
            xy_tau1 = self.horiz_profile.evaluate(tau_array[-1])

        # Note that the time index is clipped to the last t2 loop value which
        # may be smaller than t_index[-1] if the end of the horizontal profile
        # was reached.
        t_index_reparam = numpy.unique(numpy.clip(t_index, t_index[0], t2))
        df = pandas.DataFrame(x_array, index=t_index_reparam)
        df["t_orig"] = tau_array

        self.calculate_length(df, l_t0=self.l_t0)

        return df, t_index_reparam

    def integrate_backwards(self, t_index=None, t_eps=1e-12, full_state=True):
        """Perform lifting of horizontal profile and - at the same time -
        re-parametrization based on lifted states starting from a final state.
        """
        # Create time index if none is given
        if t_index is None:
            t_index = self.get_time_index(
                max(self.vert_profile.t0, self.airspeed_profile.t0),
                self.vert_profile.tf, self.t_res)
            assert (self.vert_profile.tf == self.airspeed_profile.tf)

        # For backward integration, t_index must end at right-aligned tf
        if t_index[-1] != self.tf:
            raise ValueError(
                "Backward integration requires to start at right-aligned tf value"
            )

        tau_array = [self.horiz_profile.tf]
        x_array = [self._xf]
        xy_tau2 = self.horiz_profile.xf

        # Handle case of zero-length segments (t_index has length 1)
        if len(t_index) == 1:
            return pandas.DataFrame(x_array, index=t_index), t_index

        # Evaluate original and new time parameter values and states
        for t1, t2 in zip(reversed(t_index[:-1]), reversed(t_index)):
            # print(f"t1={t1}, t2={t2}")
            dt = t2 - t1
            dl_xy = dl_xy_from_dt(
                dt, x_array[-1]["V_K_xy"], -x_array[-1]["V_K_xy_dot"])
            # Approximate corresponding interval in original parametrization
            # dtau:
            dtau = dt_from_dl_xy(
                dl_xy, xy_tau2["V_K_xy"], -xy_tau2["V_K_xy_dot"])
            # print(
            #     f"\nt1={t1}, t2={t2}, tau1={tau_array[-1]}, tau2={tau_array[-1]+dtau} "
            #     f"dt={dt}, dtau={dtau}, tf={self.vert_profile.tf}, tau_f={self.horiz_profile.tf} "
            #     f"dl_xy={dl_xy}")

            # If this timestep would overshoot the horizontal profile's initial
            # time, make it smaller and recalculate the target t1 value.
            if tau_array[-1] - dtau <= self.horiz_profile.t0 + t_eps:
                # print(
                #     f"Time step would exceed the horizontal profile initial time tau_f={self.horiz_profile.t0}"
                # )
                dtau = tau_array[-1] - self.horiz_profile.t0
                dl_xy = dl_xy_from_dt(
                    dtau, xy_tau2["V_K_xy"], -xy_tau2["V_K_xy_dot"])
                dt_new = dt_from_dl_xy(
                    dl_xy, x_array[-1]["V_K_xy"], -x_array[-1]["V_K_xy_dot"])
                # NOTE: As rounding errors may occur, the new t2 value may
                # have to be clipped to the valid range.
                assert (t1 + dt_new <= t2 + 1e-6*(t2-t1))
                t1 = max(t1, t2 - dt_new)
                tau_array.append(self.horiz_profile.t0)
                x_array.append(
                    self.evaluate_reparam(t1, tau_array[-1], full_state))
                break

            # Calculate next reparametrized and original state at t2/t2_xy
            # (i.e. states at next iteration's t1)
            tau_array.append(tau_array[-1] - dtau)
            x_array.append(self.evaluate_reparam(t1, tau_array[-1], full_state))
            xy_tau2 = self.horiz_profile.evaluate(tau_array[-1])

        # Note that the time index is clipped to the last t1 loop value which
        # may be greater than t_index[-1] if the end of the horizontal profile
        # was reached.
        t_index_reparam = numpy.unique(numpy.clip(t_index, t1, t_index[-1]))
        # t_index_reparam = t_index
        df = pandas.DataFrame(list(reversed(x_array)), index=t_index_reparam)
        df["t_orig"] = list(reversed(tau_array))

        self.calculate_length(df, l_tf=self.l_tf)
        # print("l_t0", self.l_t0)
        # print("df l", df["l"])

        return df, t_index_reparam

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize the segment based on the given initial and final times and
        the time resolution *or* an already discretized time index.

        Depending on wether the segment is integrated forwards or backwards,
        the initial or the final time (initial final element of the time_index
        if given) will be aligned with the vertical profile's and airspeed profile's
        initial or final time.
        """
        # Get default t_res value if none is given
        if t_res is None:
            t_res = self.t_res
        # Create time index if none is given
        if t_index is None:
            if t0 is None:
                t0 = self.t0
            if tf is None:
                tf = self.tf
            t_index = self.get_time_index(t0, tf, t_res)
        # Handle case: left-aligned integration
        if self.align_left:
            # Make sure the index starts at t0
            if t_index[0] != self.t0_reparam:
                t_index_integrate = self.get_time_index(
                    self.t0_reparam, t_index[-1], t_res)
            else:
                t_index_integrate = t_index
            df, t_index_reparam = self.integrate(t_index_integrate)
            if t_index_integrate is not t_index:
                df = df.reindex(t_index).interpolate(method="linear")
        # Handle case: right-aligned integration
        else:
            # Make sure the index ends at tf
            if t_index[-1] != self.tf_reparam:
                t_index_integrate = self.get_time_index(
                    t_index[0], self.tf_reparam, t_res)
            else:
                t_index_integrate = t_index
            df, t_index_reparam = self.integrate_backwards(t_index_integrate)
            if t_index_integrate is not t_index:
                df = df.reindex(t_index).interpolate(method="linear")
        return DiscreteSegment(df)

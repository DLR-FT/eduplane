# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
A reparametrized trajectory segment composed of a fixed lifted segment geometry
and an airspeed profile.

NOTE: Enforcing a new airspeed profile on a fixed geometry *will* alter the
vertical speed profile, i.e. vertical speed is not preserved. Check out
`ReparamLiftedSegment` for a re-parametrization preserving both vertical speed
and airspeed profile given as inputs while altering the horizontal airspeed
component.

NOTE: The alignment of input profiles is not and cannot be enforced:

- If the given airspeed profile is too short, the lifted segment will be
  truncated, i.e. not cover the whole horizontal path.
- If the horizontal path is too short, the airspeed profile will be
  truncated, i.e. its final airspeed may not be reached.
- An iterative procedure is given to calculate fitting constant or linear
  airspeed profiles, see method `apply_airspeed_profile`.

The reparametrization result is restricted to a user-specified time grid.

Since the corresponding time grid in respect to the original parametrization
is not known a-priori, the reparametrization requires the original segment to
be parametrically evaluable. Also, the airspeed profile must be parametrically
evaluable.

The reparametrization is based on building first-order taylor series
approximations of the velocity in order to 'convert' from old to new
parametrization (i.e. time intervals).

The general scheme of the reparametrization algorithm is:

- Initialize the reparametrized state at t_i
- For each time step (t_i, t_i+1):

  - calculate the corresponding distance by integration of the first-order
    taylor series of the velocity at t_i
  - calculate the corresponding time step (t_i,orig, t_i+1,orig) in the
    original parametrization by solving the integral of the first-order
    taylor series for t (with the original segment's values)
  - calculate the reparametrized state at t_i+1

"""

from dataclasses import InitVar, dataclass, field

import numpy
import pandas

from eduplane.model import kinematic_model as km
from eduplane.model import velocity_triangle as vt
from eduplane.traj.airspeed_profiles.linear_airspeed_profile import (
    LinearAirspeedProfile)
from eduplane.utils import utils

from .discrete_segment import DiscreteSegment
from .segment_base import AirspeedProfile, LiftedSegmentBase


def dl_xy_from_dt(dt, V_K_xy, V_K_xy_dot):
    return dt*(V_K_xy + 0.5*dt*V_K_xy_dot)


def dt_from_dl_xy(dl_xy, V_K_xy, V_K_xy_dot, V_K_xy_eps=1e-9):
    if abs(V_K_xy_dot) > V_K_xy_eps:
        po2 = V_K_xy/V_K_xy_dot
        q = -2*dl_xy/V_K_xy_dot
        radicand = po2**2 - q
        if radicand < 0:
            raise ValueError("Infinite segment time with V_K_xy < 0")
        elif V_K_xy_dot > 0:
            dt = -po2 + numpy.sqrt(radicand)
        else:
            dt = -po2 - numpy.sqrt(radicand)
    else:
        dt = dl_xy/V_K_xy
    # assert (dt >= 1e-12)
    return dt


@dataclass
class ReparamSegment(LiftedSegmentBase):
    original_segment: LiftedSegmentBase
    airspeed_profile: AirspeedProfile

    # Final time and state
    tf_reparam: float = field(init=False, default=None)
    xf: dict = field(init=False, default=None)

    # Constants
    t_res: float = LiftedSegmentBase.default_t_res

    # Init-only variables
    init_final_time_and_state: InitVar[bool] = True

    @property
    def x0(self):
        return self.apply_airspeed_profile(
            self.original_segment.x0, self.airspeed_profile.x0)

    @property
    def t0(self):
        return self.airspeed_profile.t0

    @property
    def tf(self):
        return self.tf_reparam

    def __post_init__(self, init_final_time_and_state):
        self.original_segment.shift(self.airspeed_profile.t0)
        if init_final_time_and_state:
            self.calculate_final_time_and_state()

    @classmethod
    def align_linear_airspeed(
            cls, original_segment, V_A_t0, V_A_tf, t_res=1e-1):
        tf = [original_segment.tf]
        aligned = False
        while not utils.converged(tf, abs_tol=t_res**2) or not aligned:
            # print(f"iteration #{len(tf)} tf={tf[-1]}")
            reparam_segment = cls(
                original_segment,
                LinearAirspeedProfile.init_from(
                    V_A_t0=V_A_t0,
                    V_A_tf=V_A_tf,
                    tf=tf[-1] - original_segment.t0),
                t_res=t_res)
            aligned = reparam_segment.check_alignment(t_eps=t_res**2)
            tf.append(
                reparam_segment.tf_reparam +
                (original_segment.tf - reparam_segment.xf["t_orig"]))
            # print(
            #     f" --> aligned? {aligned} converged? {utils.converged(tf, abs_tol=t_res**2)} new tf={tf[-1]}"
            # )
        return reparam_segment

    @staticmethod
    def apply_airspeed_profile(x, airspeed_prof, full_state=True):
        """For a state or an array of states of a lifted segment, apply a new
        airspeed profile. States defining the trajectory's geometry (e.g.
        chi, gamma, l, l_xy) remain unchanged while all states that depend on
        the airspeed or other air-relative quantities (e.g. speed components,
        accelerations) are updated.
        """
        V_K_orig = x["V_K"]
        V_K_xy_orig = x["V_K_xy"]
        gamma_dot_orig = x["gamma_dot"]
        chi_dot_orig = x["chi_dot"]
        # Assign airspeed profile states.
        for state in AirspeedProfile.states:
            x[state] = airspeed_prof[state]
        # Calculate airspeed-dependent states while keeping gamma and chi.
        x["V_K"] = vt.V_K_from_V_A_chi_gamma(
            x["V_A"], x["chi"], x["gamma"], x["V_W"], x["chi_W"], x["gamma_W"])
        x["V_K_xy"] = vt.V_K_xy_from_V_K_gamma(x["V_K"], x["gamma"])

        # The curvature of the trajectory remains constant, so we can calculate
        # values for gamma_dot and chi_dot from original and new velocity.
        #
        # Note that for the angular velocity in the horizontal plane (i.e.
        # chi_dot), the horizontal velocity component V_K_xy is used, while for
        # the angular velocity in the vertical plane the velocity magnitude
        # V_K is used, as the instantaneous vertical plane is rotated with the
        # velocity vector.
        #
        # chi_dot_orig = V_K_xy_orig*kappa_xy
        # -> chi_dot = V_K_xy*kappa_xy = V_K_xy*chi_dot_orig/V_K_xy_orig
        # gamma_dot_orig = V_K_orig*kappa_z
        # -> gamma_dot = V_K*kappa_z = V_K*gamma_dot_orig/V_K_orig
        x["chi_dot"] = x["V_K_xy"]*chi_dot_orig/V_K_xy_orig
        x["gamma_dot"] = x["V_K"]*gamma_dot_orig/V_K_orig

        # V_K_dot is obtained from the 3D velocity triangle solution
        x["V_K_dot"] = km.V_K_dot_from_V_A_dot(
            x["V_A"], x["V_A_dot"], x["chi"], x["chi_dot"], x["gamma"],
            x["gamma_dot"], x["V_W"], x["chi_W"], x["gamma_W"])

        if full_state:
            x["w_K_g"] = vt.w_K_g_from_V_K_gamma(x["V_K"], x["gamma"])
            x["w_A_g"] = vt.w_A_g_from_w_K_g_w_W_g(x["w_K_g"], x["w_W_g"])
            x["gamma_A"] = vt.gamma_A_from_V_A_w_A_g(x["V_A"], x["w_A_g"])
            x["V_A_xy"] = vt.V_A_xy_from_V_A_gamma_A(x["V_A"], x["gamma_A"])
            x["chi_A"] = vt.chi_A_from_V_A_xy_chi(
                x["V_A_xy"], x["chi"], x["V_W_xy"], x["chi_W"])
            x["u_A_g"] = vt.u_A_g_from_V_A_xy(x["V_A_xy"], x["chi_A"])
            x["v_A_g"] = vt.v_A_g_from_V_A_xy(x["V_A_xy"], x["chi_A"])
            x["V_K_xy_dot"] = km.V_K_xy_dot_from_V_K_dot_gamma_dot(
                x["V_K"], x["V_K_dot"], x["gamma"], x["gamma_dot"])
            x["w_K_g_dot"] = km.w_K_g_dot_from_V_K_dot_gamma_dot(
                x["V_K"], x["V_K_dot"], x["gamma"], x["gamma_dot"])
            x["u_K_g"] = vt.u_K_g_from_V_K_xy(x["V_K_xy"], x["chi"])
            x["v_K_g"] = vt.v_K_g_from_V_K_xy(x["V_K_xy"], x["chi"])

            # TODO: Reconsider if this is the best/most efficient/accurate way
            # to go from velocity components and path geometry to air-relative
            # quantities. There may be more than one approach.
            x["V_A_xy_dot"] = km.V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                x["V_K_xy"], x["V_K_xy_dot"], x["chi"], x["chi_dot"], x["V_W"],
                x["chi_W"])
            x["gamma_A_dot"] = km.gamma_A_dot_from_V_K_dot_V_A_dot_gamma_dot(
                x["V_K"], x["V_K_dot"], x["gamma"], x["gamma_dot"], x["V_A"],
                x["V_A_dot"], x["V_W"], x["gamma_W"])
            x["chi_A_dot"] = km.chi_A_dot_from_chi_dot_V_A_xy_dot.eval_if(
                mask=x["V_W_xy"] > 0,
                fill_value=x["chi_dot"],
                V_A_xy=x["V_A_xy"],
                V_A_xy_dot=x["V_A_xy_dot"],
                chi=x["chi"],
                chi_dot=x["chi_dot"],
                V_W_xy=x["V_W_xy"],
                chi_W=x["chi_W"])
            x["w_A_g_dot"] = km.w_A_g_dot_from_gamma_A_dot(
                x["V_A"], x["V_A_dot"], x["gamma_A"], x["gamma_A_dot"])

            # Calculate derivatives
            x["u_A_g_dot"] = km.u_A_g_dot_from_V_A_dot(
                x["V_A"], x["V_A_dot"], x["chi_A"], x["chi_A_dot"],
                x["gamma_A"], x["gamma_A_dot"])
            x["v_A_g_dot"] = km.v_A_g_dot_from_V_A_dot(
                x["V_A"], x["V_A_dot"], x["chi_A"], x["chi_A_dot"],
                x["gamma_A"], x["gamma_A_dot"])
            x["w_A_g_dot"] = km.w_A_g_dot_from_gamma_A_dot(
                x["V_A"], x["V_A_dot"], x["gamma_A"], x["gamma_A_dot"])
            x["u_K_g_dot"] = x["u_A_g_dot"]
            x["v_K_g_dot"] = x["v_A_g_dot"]
            x["w_K_g_dot"] = x["w_A_g_dot"]

        return x

    def check_alignment(self, t_eps=1e-1):
        """Check if the airspeed profile and the original segment are aligned
        such that the re-parametrized semgent's final state aligns with both
        profiles' final states.
        """
        # if abs(self.airspeed_profile.tf - self.tf) > t_eps:
        #     print("airspeed profile not aligned")
        # if abs(self.xf["t_orig"] - self.original_segment.tf) > t_eps:
        #     print("original segment not aligned")
        return abs(self.airspeed_profile.tf - self.tf_reparam) <= t_eps and abs(
            self.xf["t_orig"] - self.original_segment.tf) <= t_eps

    def shift(self, new_t0, **new_initial_values):
        changed_initial_values = {
            k: v
            for k, v in new_initial_values.items()
            if k in self.x0 and self.x0[k] != v
        }
        if new_t0 != self.t0 or changed_initial_values:
            self.original_segment.shift(new_t0, **changed_initial_values)
            self.airspeed_profile.shift(new_t0, **changed_initial_values)
            if self.tf_reparam is not None:
                self.calculate_final_time_and_state()
        return self

    def evaluate_reparam(self, t, t_orig, full_state=True):
        """Evaluate original trajectory and airspeed profile at given original
        and new time parameter values obtained from re-parametrization. The
        new airspeed is applied to the original trajectory's state and the
        result is returned.

        Note that the new time parameter t is clamped to the airspeed profile's
        time domain. Essentially, the last airspeed is returned for all times
        greater than the airspeed profile's final time. This is required to
        completely re-parametrize the original trajectory as the resulting
        final time cannot be known a-priori.
        """
        x_orig = self.original_segment.evaluate(t_orig)
        airspeed_prof = self.airspeed_profile.evaluate(
            min(t, self.airspeed_profile.tf))
        return self.apply_airspeed_profile(x_orig, airspeed_prof, full_state)

    def evaluate(self, t):
        return {}

    def calculate_final_time_and_state(self):
        df, _ = self.integrate(full_state=False)
        self.tf_reparam = df.index[-1]
        self.xf = df.iloc[-1].to_dict()

    def integrate(
            self,
            t_index=None,
            t_res=None,
            t0=None,
            tf=None,
            t_eps=1e-12,
            full_state=True):
        # Initialize start state and time, corresponding original state time
        # and loop values for old time parametrization.
        if t0 is None:
            t0 = self.t0
            t_orig_array = [t0]
            x_array = [self.x0]
            x_t1_orig = self.original_segment.x0
        else:
            t_orig_array = [t0]
            x0 = self.evaluate_reparam(t0, t_orig_array[-1])
            x_array = [x0]
            x_t1_orig = self.original_segment.evaluate(t_orig_array[-1])

        # Initialize time index
        if t_index is None:
            if t_res is None:
                t_res = self.t_res
            if tf is None:
                tf = self.airspeed_profile.tf
            t_index = self.get_time_index(t0, tf, t_res).to_list()

        # Evaluate original and new time parameter values and states
        for t1, t2 in zip(t_index, t_index[1:]):
            # print(f"t1={t1}, t2={t2}")
            dt = t2 - t1
            # Approximate segment length dl_xy:
            ds = dl_xy_from_dt(dt, x_array[-1]["V_K"], x_array[-1]["V_K_dot"])
            # Approximate corresponding interval in original parametrization
            # dt_orig:
            dt_orig = dt_from_dl_xy(ds, x_t1_orig["V_K"], x_t1_orig["V_K_dot"])

            # If this timestep would overshoot the original segment's final
            # time, make it smaller and recalculate the target t2 value.
            # print(
            #     f"Overshoot? {t_orig_array[-1]} + {dt_orig} = {t_orig_array[-1] + dt_orig}  >= {self.original_segment.tf}"
            # )
            if t_orig_array[-1] + dt_orig >= self.original_segment.tf - t_eps:
                dt_orig = self.original_segment.tf - t_orig_array[-1]
                ds_orig = dl_xy_from_dt(
                    dt_orig, x_t1_orig["V_K"], x_t1_orig["V_K_dot"])
                dt_new = dt_from_dl_xy(
                    ds_orig, x_array[-1]["V_K"], x_array[-1]["V_K_dot"])
                # NOTE: As rounding errors may occur, the new t2 value may
                # have to be clipped to the valid range.
                assert (t1 + dt_new <= t2 + 1e-6*(t2-t1))
                # print(
                #     f"Correcting t2 from {t2} to {t1 + dt_new}, difference: {t2-(t1 + dt_new)}"
                # )
                t2 = min(t2, t1 + dt_new)
                t_orig_array.append(self.original_segment.tf)
                x_array.append(
                    self.evaluate_reparam(t2, t_orig_array[-1], full_state))
                break

            # Since we handle cases of overshoots within accuracy of t_eps
            # above, we can clip the original segment's time step to the
            # valid range.
            # dt_orig = min(dt_orig, self.original_segment.tf - t_orig_array[-1])

            # Calculate next reparametrized and original state at t2/t2_orig
            # (i.e. states at next iteration's t1)
            t_orig_array.append(t_orig_array[-1] + dt_orig)
            x_array.append(
                self.evaluate_reparam(t2, t_orig_array[-1], full_state))
            x_t1_orig = self.original_segment.evaluate(t_orig_array[-1])

        # Create result dataframe, add original time parameter values and
        # calculate length
        t_index_reparam = numpy.unique(numpy.clip(t_index, t0, t2))
        df = pandas.DataFrame(x_array, index=t_index_reparam)
        df["t_orig"] = t_orig_array
        self.original_segment.calculate_length(df, l_t0=self.x0["l"])

        return df, t_index_reparam

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        df, t_index_reparam = self.integrate(t_index, t_res, t0, tf)
        # If the discretization is required to have a given final time, this
        # may have to be enforced due to limited accuracy alignment between
        # original segment and airspeed profile.
        tf_required = t_index[-1] if t_index is not None else tf
        if tf_required is not None:
            if t_res is None:
                t_res = self.t_res
            if abs(t_index_reparam[-1] - tf_required) > t_res**2:
                raise ValueError(
                    f"Failed to re-parametrize with final time tf={tf_required}."
                )
            t_index_reparam[-1] = tf_required
            df = df.set_index(t_index_reparam)
        return DiscreteSegment(df)

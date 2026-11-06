# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""A discrete trajectory segment."""

from dataclasses import dataclass, field

import numpy
import pandas
from scipy import integrate

from eduplane.utils import utils

from .segment_base import SegmentBase


@dataclass
class DiscreteSegment(SegmentBase):
    df: pandas.DataFrame = field(default_factory=pandas.DataFrame, repr=False)
    states: tuple = field(init=False)

    def __post_init__(self):
        if self.df.empty:
            self.df = pandas.DataFrame(index=pandas.Index([0], name="t"))
        self.states = self.df.columns

    @property
    def t0(self):
        return self.df.index[0]

    @property
    def tf(self):
        return self.df.index[-1]

    def shift(self, new_t0, **new_initial_values):
        """Shift the segment to start at a new initial time.
        Optionally, other initial values may be shifted as well
        (e.g. position or trajectory length)."""
        new_index = self.df.index.to_numpy(copy=True)
        new_index += new_t0 - new_index[0]
        self.df["new_t"] = new_index
        self.df = self.df.set_index("new_t")
        for col, col_t0 in new_initial_values.items():
            # TODO: Consider putting list of states to be shifted into base class
            if col in ("x_g", "y_g", "z_g", "l", "l_xy"):
                if not numpy.isnan(col_t0):
                    col = col.strip("_t0")
                    if col in self.df.columns:
                        self.df[col] += (col_t0 - self.df[col].iloc[0])
        return self

    def interpolate(self, t_array):
        """Interpolate the discrete segment at the given time index values."""
        new_index = pandas.Index(t_array)
        df = self.df.reindex(
            self.df.index.append(new_index).unique().sort_values())
        df = df.interpolate(method="index")
        df = df.loc[new_index]
        return df

    def evaluate(self, t):
        """Evaluate the segment at the given time parameter value.
        This is done by interpolating the dataframe."""
        values = self.get_empty_dict()
        values.update(self.interpolate([t]).loc[t].to_dict())
        return values

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize the segment at the given time parameter values,
        resolution or number of points.
        Interpolation is used for any new time parameter
        value in the resulting index."""
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        return DiscreteSegment(self.interpolate(t_index))

    def diff_state(self, state, unwrap_angle=False):
        """Numerically differentiate the given state.
        Implementation is based on the `numpy.gradient` function:
        "Second order accurate central differences in the interior points and
        either first or second order accurate one-sides (forward or backwards)
        differences at the boundaries"
        (https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.gradient.html)

        Optionally, discontinuities of magnitude 2pi from angular wrap-arounds
        can be unwrapped prior to differentiation.
        """
        if unwrap_angle:
            state_np = numpy.unwrap(self.df[state].to_numpy())
        else:
            state_np = self.df[state].to_numpy()
        if len(self.df) >= 2:
            derivative = numpy.gradient(
                state_np,
                self.df.index.to_series().values)
        else:
            derivative = 0.0
        return pandas.Series(derivative, index=self.df.index)

    def integrate_state(self, state, initial_value=0, wrap_angle=False):
        """Numerically integrate the given state.

        Implementation is implemented using scipy.integrate.cumtrapz.

        Optionally, when integrating angular values, the result can be wrapped
        to intervals [-pi, pi] or [0, 2pi] by specifying "signed" or "unsigned"
        angle wrapping.
        """
        num_int = integrate.cumulative_trapezoid(
            y=self.df[state].to_numpy(),
            x=self.df[state].index.to_numpy(),
            initial=0) + initial_value
        if wrap_angle == "signed":
            num_int = utils.angle_wrap_signed(num_int)
        elif wrap_angle == "unsigned":
            num_int = utils.angle_wrap_unsigned(num_int)
        return num_int

    def _check_diff(
            self,
            state,
            derivative,
            unwrap_angle=False,
            raise_err=False,
            warn=True,
            abs_err_threshold=1e-2):
        """Check numerically differentiated state against its given derivative.
        """
        num_diff = self.diff_state(state, unwrap_angle)
        err = (num_diff - self.df[derivative]).abs()
        err_max = err[err.idxmax()]
        # Use max. timestep for consevative error approx.
        dt_max = self.df.index.to_series().diff().max()
        # First order taylor series error term
        first_order_error = 0.5*self.df[derivative].abs().max()*dt_max
        # Second-order error term
        second_order_error = 0.5*self.df[derivative].diff().abs().max()*dt_max
        if err_max > first_order_error + second_order_error + abs_err_threshold:
            # # DEBUG:
            # from matplotlib import pyplot
            # self.df[state + "_err"] = err
            # self.df[state + "_num_diff"] = num_diff
            # self.df[[state, derivative, state + "_num_diff",
            #          state + "_err"]].plot()
            # self.df[[state + "_err"]].plot()
            # pyplot.show()
            err_str = (
                f"Max. difference between numerical diffentiation of {state}"
                f" and {derivative} is {err_max} which exceeds error margin of"
                f" {first_order_error + second_order_error + abs_err_threshold}"
            )
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def _check_int(
            self,
            state,
            derivative,
            continuity=0,
            wrap_angle=False,
            raise_err=False,
            warn=True,
            abs_err_threshold=1e-2):
        """Check state against result from numerical integration of its
        derivative.
        """
        if continuity < 0:
            return
        num_int = self.integrate_state(derivative, self.df[state].iloc[0])
        # Note that intead of wrapping the integrated angle, we can just unwrap
        # the reference integral state. This way, we don't need the user to
        # specify if it's a signed or unsigned angle and we can keep the
        # same signature as in _check_diff().
        if wrap_angle:
            ref_int = numpy.unwrap(self.df[state].to_numpy())
        else:
            ref_int = self.df[state].to_numpy()
        err = pandas.Series(num_int - ref_int, index=self.df.index).abs()
        # Since we're using integration, it makes sense to only look at error
        # steps, not the total error accumulating over time.
        err = err.diff().fillna(0).abs()
        err_max = err[err.idxmax()]
        # Use max. timestep for conservative error approx.
        dt_max = self.df.index.to_series().diff().max()
        # (Conservative) first order taylor series error term
        first_order_error = 0.5*self.df[derivative].abs().max()*dt_max
        # Second-order error term
        second_order_error = 0.5*self.df[derivative].diff().abs().max()*dt_max
        # If the given state is not continuously differentiable, i.e. the
        # derivative may have "jumps", we have to expect the integrated state
        # to have errors in the magnitude |derivative|_max * dt_max
        discont_error = 0
        if continuity == 0:
            discont_error = 2*first_order_error
        if err_max > first_order_error + second_order_error + discont_error + abs_err_threshold:
            # # DEBUG
            # from matplotlib import pyplot
            # self.df[state + "_err"] = err
            # self.df[state + "_num_int"] = num_int
            # self.df[[state, derivative, state + "_num_int",
            #          state + "_err"]].plot()
            # # self.df[[state + "_err"]].plot()
            # # self.df[[state, derivative]].plot()
            # pyplot.show()
            err_str = (
                f"Max. difference between numerical integration of {derivative}"
                f" and {state} is {err_max} which exceeds error margin of"
                f" {first_order_error+second_order_error+discont_error+abs_err_threshold}"
                f" at t={err.idxmax()}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def _check_sum(
            self,
            lhs_states,
            rhs_states,
            raise_err=False,
            warn=True,
            abs_err_threshold=1e-6):
        err = (
            self.df[lhs_states].sum(axis="columns") -
            self.df[rhs_states].sum(axis="columns")).abs()
        err_max = err[err.idxmax()]
        if err_max > abs_err_threshold:
            # # DEBUG plot
            # import matplotlib.pyplot as plt
            # self.df["err"] = err
            # self.df[["err"] + lhs_states + rhs_states].plot()
            # plt.show()
            err_str = (
                f"Max. absolute difference between sum of {lhs_states} and"
                f"  sum of {rhs_states} is {err_max} which exceeds error"
                f" margin of {abs_err_threshold}"
                f" at t={err.idxmax()}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def _check_mag(
            self,
            components,
            magnitude,
            raise_err=False,
            warn=True,
            abs_err_threshold=1e-6):
        err = (
            self.df[components].pow(2).sum(axis="columns").pow(0.5) -
            self.df[magnitude]).abs()
        err_max = err[err.idxmax()]
        if err_max > abs_err_threshold:
            err_str = (
                f"Max. absolute difference between euclidean norm of vector"
                f" components {components} and vector magnitude {magnitude}"
                f" exceeds error margin of {abs_err_threshold}: {err_max}"
                f" at t={err.idxmax()}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def _check_polar(
            self,
            vec,
            polar_vec,
            left_handed=False,
            raise_err=False,
            warn=True,
            abs_err_threshold=1e-6):
        err_x = (
            self.df[polar_vec[0]]*self.df[polar_vec[1]].apply(numpy.cos) -
            self.df[vec[0]]).abs()
        err_y = (
            self.df[polar_vec[0]]*self.df[polar_vec[1]].apply(numpy.sin) -
            self.df[vec[1]]*(-1 if left_handed else 1)).abs()
        err_x_max = err_x[err_x.idxmax()]
        err_y_max = err_y[err_y.idxmax()]
        if err_x_max > abs_err_threshold or err_y_max > abs_err_threshold:
            err_str = (
                f"Max. absolute difference between component-wise vector {vec}"
                f" and polar representation {polar_vec} is ({err_x_max},"
                f" {err_y_max}) which exceeds error margin of"
                f" {abs_err_threshold}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def check_valid(self, raise_err=False, warn=True, dt_min=1e-12):
        # Check for NaN values
        nans = utils.getNaNs(self.df, exclude=["t_orig"])
        if not nans.empty:
            err_str = (
                f"Found {len(nans)} NaN values "
                f"in dataframe columns {nans.columns.to_list()} "
                f"at parameter values {nans.index.to_list()}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)
        # Check for zero / close-to-zero time parameter intervals
        t_values = self.df.index.to_series()
        t_diff = t_values.diff()
        if t_diff.min() < dt_min:
            err_str = (
                f"Encountered zero or close-to-zero time parameter "
                f"intervals: {t_values.loc[t_diff < dt_min].to_numpy()}")
            if warn:
                print(err_str)
            if raise_err:
                raise ValueError(err_str)

    def check_kinematic_consistency(
            self,
            horizontal_continuity=0,
            vertical_continuity=0,
            horizontal_airpath_continuity=0,
            airspeed_continuity=0,
            abs_err_threshold=1e-6,
            raise_err=False,
            warn=True):
        """Check kinematic consistency of discretized trajectory by numerically
        integrating state derivatives and comparing the result with the
        corresponding given states.

        Limited continuity is handled separately for horizontal profile,
        vertical profile and airspeed profile. The corresponding argument
        specify the expected parametric continuity (0: no "jumps", 1: )
        - 0: continuous, i.e. no "jumps", but possibly not continuously differentiable
        - 1: once continuously differentiable, i.e. first derivative may be discontinuous
        - 2: twice continuously differentiable, i.e. second derivative may be discontinuous

        The absolute error threshold is applied to checks involving numerical
        integration or differentiation only.
        """

        # Basic check for NaN and close-to-zero time parameter intervals
        # =====================================================================
        self.check_valid(raise_err, warn)

        # Check consistency of integrals and derivatives by numerically
        # integrating derivatives
        # =====================================================================

        # 1) Horizontal projection states
        if "x_g" in self.states and "u_K_g" in self.states:
            self._check_int(
                "x_g", "u_K_g", min(horizontal_continuity, airspeed_continuity),
                False, raise_err, warn, abs_err_threshold)
        if "y_g" in self.states and "v_K_g" in self.states:
            self._check_int(
                "y_g", "v_K_g", min(horizontal_continuity, airspeed_continuity),
                False, raise_err, warn, abs_err_threshold)
        if "l_xy" in self.states and "V_K_xy" in self.states:
            self._check_int(
                "l_xy", "V_K_xy", horizontal_continuity, False, raise_err, warn,
                abs_err_threshold)
        if "u_K_g" in self.states and "u_K_g_dot" in self.states:
            self._check_int(
                "u_K_g", "u_K_g_dot",
                min(horizontal_continuity, airspeed_continuity) - 1, False,
                raise_err, warn, abs_err_threshold)
        if "v_K_g" in self.states and "v_K_g_dot" in self.states:
            self._check_int(
                "v_K_g", "v_K_g_dot",
                min(horizontal_continuity, airspeed_continuity) - 1, False,
                raise_err, warn, abs_err_threshold)
        # NOTE: V_K_xy may jump if w_K_g jumps (C0 vertical profile)
        if "V_K_xy" in self.states and "V_K_xy_dot" in self.states:
            self._check_int(
                "V_K_xy", "V_K_xy_dot",
                min(horizontal_continuity - 1, vertical_continuity - 1), False,
                raise_err, warn, abs_err_threshold)
        # Note that tangent angles or azimuths of speed vectors are continuous
        # but only continuously differentiable if the path is C2 or higher.
        # For C1 (once continuously differentiable) paths, the tangent is
        # (C0-)continuous i.e. not necessarily differentiable.
        if "chi" in self.states and "chi_dot" in self.states:
            self._check_int(
                "chi", "chi_dot", horizontal_continuity - 1, True, raise_err,
                warn, abs_err_threshold)

        # 2) Vertical profile states
        if "z_g" in self.states and "w_K_g" in self.states:
            self._check_int(
                "z_g", "w_K_g", vertical_continuity, False, raise_err, warn,
                abs_err_threshold)
        if "w_K_g" in self.states and "w_K_g_dot" in self.states:
            self._check_int(
                "w_K_g", "w_K_g_dot", vertical_continuity - 1, False, raise_err,
                warn, abs_err_threshold)

        # 3) Coupled position states
        if "l" in self.states and "V_K" in self.states:
            self._check_int(
                "l", "V_K", min(horizontal_continuity, vertical_continuity),
                False, raise_err, warn, abs_err_threshold)
        if "V_K" in self.states and "V_K_dot" in self.states:
            self._check_int(
                "V_K", "V_K_dot",
                min(horizontal_continuity, vertical_continuity) - 1, False,
                raise_err, warn, abs_err_threshold)

        # 4) Airpath and airspeed profile
        if "V_A" in self.states and "V_A_dot" in self.states:
            self._check_int(
                "V_A", "V_A_dot", airspeed_continuity, False, raise_err, warn,
                abs_err_threshold)
        # NOTE: V_A_xy may jump if w_A_g jumps (C0 vertical profile)
        if "V_A_xy" in self.states and "V_A_xy_dot" in self.states:
            self._check_int(
                "V_A_xy", "V_A_xy_dot",
                min(airspeed_continuity, vertical_continuity - 1), False,
                raise_err, warn, abs_err_threshold)
        if "u_A_g" in self.states and "u_A_g_dot" in self.states:
            self._check_int(
                "u_A_g", "u_A_g_dot", horizontal_airpath_continuity - 1, False,
                raise_err, warn, abs_err_threshold)
        if "v_A_g" in self.states and "v_A_g_dot" in self.states:
            self._check_int(
                "v_A_g", "v_A_g_dot", horizontal_airpath_continuity - 1, False,
                raise_err, warn, abs_err_threshold)
        if "chi_A" in self.states and "chi_A_dot" in self.states:
            self._check_int(
                "chi_A", "chi_A_dot", horizontal_airpath_continuity - 1, True,
                raise_err, warn, abs_err_threshold)
        if "w_A_g" in self.states and "w_A_g_dot" in self.states:
            self._check_int(
                "w_A_g", "w_A_g_dot",
                min(airspeed_continuity, vertical_continuity - 1), False,
                raise_err, warn, abs_err_threshold)
        if "gamma_A" in self.states and "gamma_A_dot" in self.states:
            self._check_int(
                "gamma_A", "gamma_A_dot",
                min(airspeed_continuity, vertical_continuity - 1), True,
                raise_err, warn, abs_err_threshold)

        # Check speed vector, magnitude and tangent angle consistency
        # =====================================================================
        # Check velocity equals airspeed + wind speed
        if all(x in self.states for x in ["u_K_g", "u_A_g", "u_W_g"]):
            self._check_sum(
                ["u_K_g"], ["u_A_g", "u_W_g"], raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["v_K_g", "v_A_g", "v_W_g"]):
            self._check_sum(
                ["v_K_g"], ["v_A_g", "v_W_g"], raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["w_K_g", "w_A_g", "w_W_g"]):
            self._check_sum(
                ["w_K_g"], ["w_A_g", "w_W_g"], raise_err, warn,
                abs_err_threshold)
        # Check vector magnitudes
        if all(x in self.states for x in ["u_K_g", "v_K_g", "V_K_xy"]):
            self._check_mag(
                ["u_K_g", "v_K_g"], "V_K_xy", raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_K_g", "v_K_g", "w_K_g", "V_K"]):
            self._check_mag(
                ["u_K_g", "v_K_g", "w_K_g"], "V_K", raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_A_g", "v_A_g", "V_A_xy"]):
            self._check_mag(
                ["u_A_g", "v_A_g"], "V_A_xy", raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_A_g", "v_A_g", "w_A_g", "V_A"]):
            self._check_mag(
                ["u_A_g", "v_A_g", "w_A_g"], "V_A", raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_W_g", "v_W_g", "V_W_xy"]):
            self._check_mag(
                ["u_W_g", "v_W_g"], "V_W_xy", raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_W_g", "v_W_g", "w_W_g", "V_W"]):
            self._check_mag(
                ["u_W_g", "v_W_g", "w_W_g"], "V_W", raise_err, warn,
                abs_err_threshold)
        # Check vector components and polar representation
        if all(x in self.states for x in ["u_A_g", "v_A_g", "V_A_xy", "chi_A"]):
            self._check_polar(
                ["u_A_g", "v_A_g"], ["V_A_xy", "chi_A"], False, raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["V_A_xy", "w_A_g", "V_A", "gamma_A"]):
            self._check_polar(
                ["V_A_xy", "w_A_g"], ["V_A", "gamma_A"], True, raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_K_g", "v_K_g", "V_K_xy", "chi"]):
            self._check_polar(
                ["u_K_g", "v_K_g"], ["V_K_xy", "chi"], False, raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["V_K_xy", "w_K_g", "V_K", "gamma"]):
            self._check_polar(
                ["V_K_xy", "w_K_g"], ["V_K", "gamma"], True, raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["u_W_g", "v_W_g", "V_W_xy", "chi_W"]):
            self._check_polar(
                ["u_W_g", "v_W_g"], ["V_W_xy", "chi_W"], False, raise_err, warn,
                abs_err_threshold)
        if all(x in self.states for x in ["V_W_xy", "w_W_g", "V_W", "gamma_W"]):
            self._check_polar(
                ["V_W_xy", "w_W_g"], ["V_W", "gamma_W"], True, raise_err, warn,
                abs_err_threshold)

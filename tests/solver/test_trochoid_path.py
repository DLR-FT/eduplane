# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.trochoid_path.

Run from top-level package directory:

   python3 -m unittest tests.solver.test_trochoid_path

NOTE: Run benchmarks with disabled CPU turbo boost:
        echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
        sudo cpupower frequency-set -g performance
    Revert with
        echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
        sudo cpupower frequency-set -g powersave
    Monitor/verfify with
        sudo cpupower frequency-info
        watch -n 1 grep \"cpu MHz\" /proc/cpuinfo
"""

import inspect
import math
import os
import random
import statistics
import subprocess
import traceback
import unittest
from collections.abc import Iterable

import numpy
import pandas
from tqdm import tqdm

from eduplane.solver.dubins_path import get_dubins_path_3D
from eduplane.solver.trochoid_path import *
from eduplane.utils import logger, plotting

from . import test_tools
from .test_tools import Sample2D as Sample
from .test_tools import Sample3D

DEBUG_PLOTS = False
# DEBUG_PLOTS = True

PRINT_TRACES_REPORT = False

NUM_CASES = int(os.getenv("NUM_TEST_CASES", default=100))
NUM_CASES_BENCHMARK = NUM_CASES

NUM_CASES_CSC_BASIC = NUM_CASES
NUM_CASES_CSC_RANDOM = NUM_CASES
NUM_CASES_CSC_BENCHMARK = NUM_CASES_BENCHMARK

NUM_CASES_CSCA_BASIC = NUM_CASES
NUM_CASES_CSCA_RANDOM = NUM_CASES
NUM_CASES_CSCA_BENCHMARK = NUM_CASES_BENCHMARK

NUM_CASES_CSC3D_CONST_VS_BASIC = NUM_CASES
NUM_CASES_CSC3D_CONST_VS_RANDOM = NUM_CASES
NUM_CASES_CSC3D_CONST_VS_BENCHMARK = NUM_CASES_BENCHMARK
NUM_CASES_CSC3D_CONST_VS_CONVERGENCE = NUM_CASES

NUM_CASES_CSC3D_CONST_THRUST_BASIC = NUM_CASES
NUM_CASES_CSC3D_CONST_THRUST_RANDOM = NUM_CASES
NUM_CASES_CSC3D_CONST_THRUST_BENCHMARK = NUM_CASES_BENCHMARK
NUM_CASES_CSC3D_CONST_THRUST_CONVERGENCE = NUM_CASES

NUM_CASES_CSC3D_RAPID_DESCENT_BASIC = NUM_CASES
NUM_CASES_CSC3D_RAPID_DESCENT_RANDOM = NUM_CASES
NUM_CASES_CSC3D_RAPID_DESCENT_BENCHMARK = NUM_CASES_BENCHMARK

NUM_CASES_SOLVER_COMPARISON = 10


def plot_csc(csc, t_a_roots):
    """Debug plot for investigating issues with CSC solvers"""
    from eduplane.traj.horizontal_profiles import SteadyTurn
    seg1 = SteadyTurn(
        x_g_t0=csc.x_t0,
        y_g_t0=csc.y_t0,
        V_A_xy=csc.v_a_1,
        chi_A_dot=csc.omega1,
        chi_t0=csc.chi_t0,
        V_W_xy=csc.v_w,
        chi_W=csc.psi_w,
        tf=2*csc.t_2pi2,
    )
    seg3_helper = SteadyTurn(
        x_g_t0=csc.x_tf,
        y_g_t0=csc.y_tf,
        V_A_xy=csc.v_a_2,
        chi_A_dot=csc.omega1,
        chi_t0=csc.chi_tf,
        V_W_xy=csc.v_w,
        chi_W=csc.psi_w,
        tf=csc.t_2pi2,
    )
    seg3 = SteadyTurn(
        x_g_t0=csc.x_tf - 2*(seg3_helper.xf["x_g"] - csc.x_tf),
        y_g_t0=csc.y_tf - 2*(seg3_helper.xf["y_g"] - csc.y_tf),
        V_A_xy=csc.v_a_2,
        chi_A_dot=csc.omega1,
        chi_t0=csc.chi_tf,
        V_W_xy=csc.v_w,
        chi_W=csc.psi_w,
        tf=2*csc.t_2pi2,
    )
    seg2_candidates = []
    for t_a in t_a_roots:
        seg2 = ConstSpeedStraightFlight.init_from(
            x_g_t0=seg1.x_g(t_a),
            y_g_t0=seg1.y_g(t_a),
            l_xy_tf=csc.v_a_1/abs(csc.omega1)*20,
            V_A_xy=csc.v_a_1,
            chi=seg1.chi(t_a),
            V_W_xy=csc.v_w,
            chi_W=csc.psi_w,
        )
        seg2_candidates.append(seg2)
    seg1_discr = seg1.discretize()
    seg3_discr = seg3.discretize()
    seg2_discr_df = [seg2.discretize().df for seg2 in seg2_candidates]
    plotting.plot_topview(
        [seg1_discr.df, seg3_discr.df] + seg2_discr_df,
        wind_speed=csc.v_w,
        wind_dir=csc.psi_w,
        POIs=[
            dict(
                x_g=csc.x_t0,
                y_g=csc.y_t0,
                chi=csc.chi_t0,
                label="start",
                style="go",
                marker=(3, 0, math.degrees(csc.chi_t0))),
            dict(
                x_g=csc.x_tf,
                y_g=csc.y_tf,
                chi=csc.chi_tf,
                label="goal",
                style="go",
                marker=(3, 0, math.degrees(csc.chi_tf))),
        ])
    plotting.show()


class TrochoidPathTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)
        logger.reset()
        logger.enable_rt_traces()

    def report_vert_speed_slack(self, discr_paths, vert_speed_max, eps=1e-6):
        vertical_speed_slacks = [
            vert_speed_max - abs(w_A_g) for path in discr_paths
            for w_A_g in path.df["w_A_g"]
        ]
        vert_speed_violations = sum(
            slack < -eps for slack in vertical_speed_slacks)
        print(
            f"Vertical speed constraint (|w_A_g| <= {vert_speed_max} [m/s]) slack"
            f"(min/median/max): {min(vertical_speed_slacks):.6g} / "
            f"{statistics.median(vertical_speed_slacks):.6g} / "
            f"{max(vertical_speed_slacks):.6g}; "
            f"violating samples: {vert_speed_violations}/{len(vertical_speed_slacks)}"
        )
        self.assertEqual(vert_speed_violations, 0)

    def report_load_factor_slack(
            self,
            discr_paths,
            load_factor_max,
            turns_only=False,
            aircraft=None,
            eps=1e-6):
        load_factor_slacks = []
        for path in discr_paths:
            if aircraft is not None:
                aircraft.fly_trajectory(path)
            else:
                path.df["n"] = fpm.n_from_chi_A_dot_gamma_A_dot(
                    path.df["V_A"], path.df["chi_A_dot"], path.df["gamma_A"],
                    path.df["gamma_A_dot"], 9.81)
            samples = path.df
            if turns_only:
                samples = samples[samples["chi_A_dot"].abs() > 1e-3]
            load_factor_slacks.extend(
                load_factor_max - abs(n) for n in samples["n"])
        constraint_name = f"Load factor constraint ({load_factor_max})"
        if turns_only:
            constraint_name += " during turns"
        violations = sum(slack < -eps for slack in load_factor_slacks)
        print(
            f"Max. load factor constraint (n <= {load_factor_max}) slack"
            f"(min/median/max): {min(load_factor_slacks):.6g} / "
            f"{statistics.median(load_factor_slacks):.6g} / {max(load_factor_slacks):.6g}; "
            f"violating samples: {violations}/{len(load_factor_slacks)}")
        self.assertEqual(violations, 0)

    def report_thrust_slack(
            self, discr_paths, ac, thrust_min, thrust_max, eps=1e-6):
        thrust_min = ac.T_min if thrust_min is None else thrust_min
        thrust_max_is_T_t0 = thrust_max == "T_t0"
        if thrust_max is None:
            thrust_max = ac.T_max

        min_slacks = []
        max_slacks = []
        for path in discr_paths:
            ac.fly_trajectory(path)
            T = path.df["T"]
            min_slacks.extend(T - thrust_min)
            if thrust_max_is_T_t0:
                max_slacks.extend(T.iloc[0] - T)
            else:
                max_slacks.extend(thrust_max - T)

        for name, slacks in [
            (f"T >= {thrust_min}", min_slacks),
            (f"T <= {thrust_max}", max_slacks),
        ]:
            violations = sum(slack < -eps for slack in slacks)
            print(
                f"Thrust constraint ({name}) slack (min/median/max): "
                f"{min(slacks):.6g} / {statistics.median(slacks):.6g} / "
                f"{max(slacks):.6g}; "
                f"violating samples: {violations}/{len(slacks)}")
            self.assertEqual(violations, 0)

    @staticmethod
    def _filter_solver_kwargs(solver_func, kwargs):
        """Drop keyword arguments a solver does not accept.

        The comparison passes a superset of arguments (e.g. both ``turn_rate``
        for the trochoid solvers and ``turn_radius`` for the Dubins solver).
        Solvers with a ``**kwargs`` catch-all receive everything; others only
        the parameters they declare.
        """
        signature = inspect.signature(solver_func)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values())
        if accepts_var_kwargs:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items() if key in signature.parameters
        }

    def run_solver_comparison(self, test_cases, solvers, desc, load_factor_max):
        """Compare several 3D solvers on the same test cases by final flight time.

        Each solver is called on every test case with the shared load-factor
        bound ``turn_n_max=load_factor_max`` so that the Dubins and trochoid
        paths are compared on equal terms. For every case the final flight time
        ``t_f`` of each solver's path is collected; cases where a solver returns
        ``None`` (infeasible) are recorded as missing.

        ``solvers`` is a list of ``(label, solver_func, extra_kwargs)`` tuples.
        ``extra_kwargs`` are merged into the per-case arguments (e.g.
        ``high_alt_strategy``, ``ac``, ``thrust_constraint``).

        Returns ``(flight_times, discr_paths)`` where ``flight_times`` is a wide
        dataframe with one row per test case and one column per solver, and
        ``discr_paths`` maps each solver label to the list of its discretized
        paths (for constraint checks).
        """
        labels = [label for label, _, _ in solvers]
        rows = []
        discr_paths = {label: [] for label in labels}
        print(f"Running solver comparison: {desc}")
        with tqdm(total=len(test_cases), desc=desc) as pbar:
            for i, test_case in enumerate(test_cases):
                pbar.update()
                row = {"case": i}
                for label, solver_func, extra_kwargs in solvers:
                    kwargs = dict(test_case)
                    kwargs["turn_n_max"] = load_factor_max
                    # Provide both turn_rate (trochoid) and turn_radius
                    # (Dubins); each solver picks up what it accepts.
                    kwargs["turn_radius"] = (
                        test_case["airspeed"]/test_case["turn_rate"])
                    kwargs.update(extra_kwargs)
                    kwargs = self._filter_solver_kwargs(solver_func, kwargs)
                    logger.push(solver_func.__name__)
                    try:
                        path = solver_func(**kwargs)
                        logger.pop(success=path is not None)
                    except Exception:
                        logger.unwind(error=True)
                        path = None
                    if path is not None:
                        row[label] = path.tf
                        discr_paths[label].append(path.discretize())
                    else:
                        row[label] = numpy.nan
                rows.append(row)
        if PRINT_TRACES_REPORT:
            logger.printReport()
        return pandas.DataFrame(rows), discr_paths

    def report_flight_time_comparison(
            self,
            flight_times,
            discr_paths,
            labels,
            desc,
            load_factor_max,
            vert_speed_max=None,
            aircraft=None):
        """Report and statistically compare final flight times of several solvers.

        ``flight_times`` is the wide dataframe returned by
        :meth:`run_solver_comparison` and ``discr_paths`` the matching mapping
        of solver label to discretized paths. Prints a summary of the
        flight-time distribution per solver, the pairwise differences on the
        cases solved by both solvers, and asserts that the shared load-factor
        (and, if given, vertical-speed) constraints are obeyed by every solver.
        """
        print(f"\n=== Flight time comparison: {desc} ===")
        print(f"Test cases: {len(flight_times)}")
        summary = pandas.DataFrame(
            {
                "solved": flight_times[labels].notna().sum(),
                "min": flight_times[labels].min(),
                "median": flight_times[labels].median(),
                "mean": flight_times[labels].mean(),
                "max": flight_times[labels].max(),
            })
        print("Final flight time t_f [s] per solver:")
        print(summary.to_string())

        # Pairwise comparison on the cases solved by both solvers.
        print("\nPairwise flight time differences [s] (common cases only):")
        for i, label_a in enumerate(labels):
            for label_b in labels[i + 1:]:
                common = flight_times[[label_a, label_b]].dropna()
                if common.empty:
                    print(f"  {label_a} vs {label_b}: no common cases")
                    continue
                diff = common[label_b] - common[label_a]
                print(
                    f"  {label_b} - {label_a}: n={len(common)}, "
                    f"min/median/mean/max = {diff.min():.6g} / "
                    f"{diff.median():.6g} / {diff.mean():.6g} / "
                    f"{diff.max():.6g}")

        # Constraint checks: every solver must obey the shared load-factor
        # bound (and the vertical-speed bound, if given).
        for label in labels:
            print(f"\nConstraint checks for '{label}':")
            self.report_load_factor_slack(
                discr_paths[label],
                load_factor_max,
                turns_only=True,
                aircraft=aircraft)
            if vert_speed_max is not None:
                self.report_vert_speed_slack(discr_paths[label], vert_speed_max)

    # def random(self, min_value, max_value):
    #     return min_value + (max_value-min_value)*random.random()

    def run_test_case(
            self,
            test_cases,
            test_func,
            desc,
            horizontal_continuity=1,
            vertical_continuity=1,
            horiz_airpath_continuity=0,
            airspeed_continuity=1,
            eps_xyz=1e-1,
            eps_chi=1e-3):
        discr_paths = []
        print(f"Running test suite: {desc}")
        with tqdm(total=len(test_cases), desc=desc) as pbar:
            for test_case in test_cases:
                pbar.update()
                logger.push(test_func.__name__)

                path = test_func(**test_case)

                # LSR/RSL Dubins paths should exist for all cases k >= 4, for k < 4 they may exist or not.
                # For trochoid paths, a conservative sufficient condition for existance can be derived by adding the
                # wind drift: D >= 2R+W with the max. turn radius R. D is the distance between the instantaneous
                # midpoints of the trochoids at start and goal and W is the maximum wind drift over a CSC path.
                # Given only the distance from start to goal D_sg, we have to add 2*R which gives: D_sg >= 4R+W
                if isinstance(test_case["airspeed"], Iterable):
                    airspeed_min = min(test_case["airspeed"])
                    airspeed_max = max(test_case["airspeed"])
                else:
                    airspeed_min = airspeed_max = test_case["airspeed"]
                if isinstance(test_case["turn_rate"], Iterable):
                    turn_rate_min = min(test_case["turn_rate"])
                    turn_rate_max = max(test_case["turn_rate"])
                else:
                    turn_rate_min = turn_rate_max = test_case["turn_rate"]
                turn_radius_max = (
                    airspeed_max + test_case["wind_speed"])/turn_rate_min
                D_sg = math.sqrt(
                    (test_case["goal_x"] - test_case["start_x"])**2 +
                    (test_case["goal_y"] - test_case["start_y"])**2)
                D_W_max = (
                    4*math.pi/turn_rate_min + D_sg/
                    (airspeed_min + test_case["wind_speed"])
                )*test_case["wind_speed"]
                if path is None and test_case["path_type"] in [
                        "LSR", "RSL"
                ] and D_sg <= 4*turn_radius_max + D_W_max:
                    logger.pop(path_type=test_case["path_type"], success=False)
                    continue

                # print(f"{D_sg=} <= {4*turn_radius_max} + {D_W_max=}")
                self.assertIsNotNone(
                    path,
                    msg=
                    f"Failed to plan {test_case['path_type']} path. Full test case:\n    {test_case}"
                )
                path_discr = path.discretize()
                try:
                    path_discr.check_kinematic_consistency(
                        horizontal_continuity,
                        vertical_continuity,
                        horiz_airpath_continuity,
                        airspeed_continuity,
                        raise_err=True,
                        abs_err_threshold=1e-6)
                    # Check initial and final state
                    self.assertAlmostEqual(
                        path_discr.x0["x_g"], test_case["start_x"])
                    self.assertAlmostEqual(
                        path_discr.xf["x_g"],
                        test_case["goal_x"],
                        delta=eps_xyz)
                    self.assertAlmostEqual(
                        path_discr.x0["y_g"], test_case["start_y"])
                    self.assertAlmostEqual(
                        path_discr.xf["y_g"],
                        test_case["goal_y"],
                        delta=eps_xyz)
                    if "start_z" in test_case:
                        self.assertAlmostEqual(
                            path_discr.x0["z_g"],
                            test_case["start_z"],
                            delta=eps_xyz)
                    if "goal_z" in test_case and test_case["goal_z"] is not None:
                        self.assertAlmostEqual(
                            path_discr.xf["z_g"],
                            test_case["goal_z"],
                            delta=eps_xyz)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.x0["chi"], test_case["start_dir"])),
                        0.0,
                        delta=eps_chi)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.xf["chi"], test_case["goal_dir"])),
                        0.0,
                        delta=eps_chi)
                except (AssertionError, ValueError) as err:
                    if DEBUG_PLOTS:
                        print("Test case failed with assertion error:", err)
                        if "ac" in test_case:
                            test_case["ac"].fly_trajectory(path_discr)
                            timelines = [
                                "V_A", "V_A_xy", "V_A_xy_dot", "w_A_g",
                                "gamma_A", "T", "n"
                            ]
                        else:
                            timelines = [
                                "V_A", "V_A_xy", "V_A_xy_dot", "w_A_g",
                                "gamma_A"
                            ]
                        POIs = test_tools.get_start_goal_POIs(test_case)
                        plotting.plot_ref_trajectory(
                            paths=[path_discr.df], POIs=POIs)
                        plotting.plot_timeline(
                            trajectories=[path_discr.df],
                            subplots=timelines,
                            columns=2)
                        plotting.show()
                    traceback.print_exc()
                    raise

                discr_paths.append(path_discr)
                logger.pop(path_type=test_case["path_type"], success=True)
        if PRINT_TRACES_REPORT:
            logger.printReport()
        return discr_paths

    def test_csc_basic(self):
        """Test CSC 2D Trochoid paths with basic test set"""
        turn_radius = 100
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        turn_rate = airspeed/turn_radius
        turn_radius_max = (airspeed+wind_speed)/turn_rate
        k_values = [0, 2, 4, 6, 8]
        path_types = ["LSL", "RSR", "LSR", "RSL"]

        test_cases = test_tools.get_test_cases_2D(
            turn_rate,
            airspeed,
            wind_speed,
            wind_dir,
            xy_dist=[k*turn_radius_max for k in k_values],
            path_types=path_types)

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSC_BASIC)

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_2D,
            "Testing CSC 2D Trochoid paths - basic test set")

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()

    def test_csc_random(self):
        """Test CSC 2D Trochoid paths with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8

        test_cases = test_tools.get_test_cases_2D_randomized(
            num_test_cases=NUM_CASES_CSC_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            path_types=path_types,
        )

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_2D,
            "Testing CSC 2D Trochoid paths - randomized test set")

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()
        return

    def test_csc_benchmark(self):
        """Run benchmark for CSC 2D Trochoid paths with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        # path_types = ["LSL", "RSR"]
        # path_types = ["LSR", "RSL"]
        k_min = 0
        k_max = 8

        test_cases = test_tools.get_test_cases_2D_randomized(
            num_test_cases=NUM_CASES_CSC_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            path_types=path_types,
        )

        discr_paths = test_tools.run_benchmark(
            test_cases, get_trochoid_path_2D,
            "Benchmark for CSC 2D Trochoid paths - randomized test set")

    def test_csca_basic(self):
        """Test CSCA 2D Trochoid paths with basic test set"""
        turn_radius_1 = 50
        turn_radius_2 = 80
        airspeed_1 = 15
        airspeed_2 = 25
        wind_speed = 5
        wind_dir = -math.pi*0.25
        turn_rate_1 = airspeed_1/turn_radius_1
        turn_rate_2 = airspeed_2/turn_radius_2
        turn_radius_max = max(turn_radius_1, turn_radius_2)
        k_values = [0, 2, 4, 6, 8]
        path_types = ["LSL", "RSR", "LSR", "RSL"]

        test_cases = test_tools.get_test_cases_2D(
            (turn_rate_1, turn_rate_2),
            (airspeed_1, airspeed_2),
            wind_speed,
            wind_dir,
            xy_dist=[k*turn_radius_max for k in k_values],
            path_types=path_types,
        )

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSCA_BASIC)

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_2D,
            "Testing CSCA 2D Trochoid paths - basic test set")

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()

    def test_csca_random(self):
        """Test CSCA 2D Trochoid paths with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8

        test_cases = test_tools.get_test_cases_2D_randomized(
            num_test_cases=NUM_CASES_CSCA_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            path_types=path_types,
        )
        # Add second airspeed and turn rate to all test cases
        for test_case in test_cases:
            other_test_case = random.sample(test_cases, 1)[0]
            test_case["airspeed"] = (
                test_case["airspeed"],
                utils.listify(other_test_case["airspeed"])[0])
            test_case["turn_rate"] = (
                test_case["turn_rate"],
                utils.listify(other_test_case["turn_rate"])[0])

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_2D,
            "Testing CSCA 2D Trochoid paths - randomized test set")

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()
        return

    def test_csca_benchmark(self):
        """Run benchmark for CSCA 2D Trochoid paths with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8

        test_cases = test_tools.get_test_cases_2D_randomized(
            num_test_cases=NUM_CASES_CSCA_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            path_types=path_types,
        )

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_2D,
            "Benchmark for CSCA 2D Trochoid paths - randomized test set")

    def test_csc3d_const_vert_speed_basic(self):
        """Run tests for CSC3D trochoid paths with constant vertical speed with basic test set"""
        turn_radius_1 = 50
        turn_radius_2 = 80
        airspeed_1 = 15
        airspeed_2 = 25
        wind_speed = 5
        wind_dir = -math.pi*0.25
        turn_rate_1 = airspeed_1/turn_radius_1
        turn_rate_2 = airspeed_2/turn_radius_2
        turn_radius_max = max(turn_radius_1, turn_radius_2)
        k_values = [0, 2, 4, 6, 8]
        l_values = [0, 1, 2]
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D(
            (turn_rate_1, turn_rate_2), (airspeed_1, airspeed_2),
            wind_speed,
            wind_dir,
            xy_dist=[k*turn_radius_max for k in k_values],
            z_dist=[l*turn_radius_max for l in l_values],
            path_types=path_types)

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSC3D_CONST_VS_BASIC)

        # DEBUG TEST CASE:
        # test_cases = [test_cases[150]]

        # Unconstrained vertical speed
        # =====================================================================
        # Add parameters turn_n_max, high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["vert_speed_constraint"] = None

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with unconstrained constant vertical speed - basic test set"
        )
        # Plot the results
        if DEBUG_PLOTS:
            plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in plot_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in plot_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                ],
                columns=1)
            plotting.show()

        # Constrained vertical speed and periodic paths
        # =====================================================================
        # Add vertical speed cosntraints parameters high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(w_A_g) < 5

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with constrained vertical speed, periodic paths - basic test set"
        )
        # Plot the results
        if DEBUG_PLOTS:
            plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in plot_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in plot_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                ],
                columns=1)
            plotting.show()

        # Constrained vertical speed and full turns
        # =====================================================================
        # Change high-altitude strategy to use full turns instead of periodic paths
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with constrained vertical speed, full turns - basic test set"
        )

        # Plot the results
        if DEBUG_PLOTS:
            plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in plot_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in plot_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                ],
                columns=1)
            plotting.show()

    def test_csc3d_const_vert_speed_random(self):
        """Run tests for CSC3D trochoid paths with constant vertical speed with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        # wind_speed_max = 0
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        vert_speed_max = 4
        load_factor_max = 4
        eps = 1e-12

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Unconstrained vertical speed
        # =====================================================================
        # Add parameters turn_n_max, high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["vert_speed_constraint"] = None

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with unconstrained constant vertical speed - randomized test set"
        )
        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True)
        # Plot the results
        if DEBUG_PLOTS:
            # plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g"
                ],
                columns=1)
            plotting.show()

        # Constrained vertical speed and periodic paths
        # =====================================================================
        # Add vertical speed cosntraints parameters high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with constrained vertical speed, periodic paths - randomized test set"
        )

        # Check constraints
        self.report_vert_speed_slack(discr_paths, vert_speed_max)
        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True)

        # Plot the results
        if DEBUG_PLOTS:
            # plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "n"
                ],
                columns=1)
            plotting.show()

        # Constrained vertical speed and full turns
        # =====================================================================
        # Change high-altitude strategy to use full turns instead of periodic paths
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with constrained vertical speed, full turns - randomized test set"
        )

        # Check constraints
        self.report_vert_speed_slack(discr_paths, vert_speed_max)
        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True)

        # Plot the results
        if DEBUG_PLOTS:
            # plot_paths = discr_paths[10:11]
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                ],
                columns=1)
            plotting.show()

    def test_solver_comparison_const_vert_speed(self):
        """Compare the 3D Dubins and trochoid solvers by final flight time on the
        same randomized test cases.

        Both solvers are run with the same load-factor bound (``turn_n_max``)
        and the same vertical-speed constraint, using periodic paths for the
        trochoid solver. For every test case the final flight time ``t_f`` of
        each solver is collected and compared statistically.
        """
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        vert_speed_max = 4
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_SOLVER_COMPARISON,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        vert_speed_constraint = lambda w_A_g: abs(w_A_g) < vert_speed_max
        solvers = [
            (
                "dubins",
                get_dubins_path_3D,
                dict(vert_speed_constraint=vert_speed_constraint),
            ),
            (
                "trochoid",
                get_trochoid_path_3D_const_vert_speed,
                dict(
                    high_alt_strategy="periodic paths",
                    vert_speed_constraint=vert_speed_constraint),
            ),
        ]
        labels = [label for label, _, _ in solvers]

        flight_times, discr_paths = self.run_solver_comparison(
            test_cases, solvers,
            "Solver comparison: 3D Dubins vs trochoid, constrained vertical speed, periodic paths",
            load_factor_max)

        self.report_flight_time_comparison(
            flight_times,
            discr_paths,
            labels,
            "3D Dubins vs trochoid, constrained vertical speed, periodic paths",
            load_factor_max,
            vert_speed_max=vert_speed_max)

        if DEBUG_PLOTS:
            plotting.plot_stats(
                [
                    flight_times[[label]].rename(columns={label: "t_f"})
                    for label in labels
                ],
                "t_f",
                labels,
                title="Final flight time by solver")
            plotting.show()

    def test_solver_comparison_const_thrust(self):
        """Compare the 3D trochoid constant-thrust and constant-vertical-speed
        solvers by final flight time on the same randomized test cases.

        Both solvers use periodic paths and the same load-factor bound. The
        constant-thrust solver additionally obeys the thrust constraint.
        """
        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        thrust_min = 0
        thrust_max = 30
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_SOLVER_COMPARISON,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        solvers = [
            (
                "const. vert. speed",
                get_trochoid_path_3D_const_vert_speed,
                dict(
                    high_alt_strategy="periodic paths",
                    vert_speed_constraint=None),
            ),
            (
                "const. thrust",
                get_trochoid_path_3D_const_thrust,
                dict(
                    high_alt_strategy="periodic paths",
                    ac=ac,
                    thrust_constraint=lambda T: T > thrust_min and T <
                    thrust_max),
            ),
        ]
        labels = [label for label, _, _ in solvers]

        flight_times, discr_paths = self.run_solver_comparison(
            test_cases, solvers,
            "Solver comparison: 3D trochoid constant thrust vs constant vertical speed, periodic paths",
            load_factor_max)

        self.report_flight_time_comparison(
            flight_times,
            discr_paths,
            labels,
            "3D trochoid constant thrust vs constant vertical speed, periodic paths",
            load_factor_max,
            aircraft=ac)

        if DEBUG_PLOTS:
            plotting.plot_stats(
                [
                    flight_times[[label]].rename(columns={label: "t_f"})
                    for label in labels
                ],
                "t_f",
                labels,
                title="Final flight time by solver")
            plotting.show()

    def test_solver_comparison_rapid_descent(self):
        """Compare the 3D trochoid rapid-descent and constant-vertical-speed
        solvers by final flight time on the same randomized descent test cases.

        Only descent cases (``goal_z > start_z``) are analyzed, since the rapid
        descent transition is only defined for descents. Both solvers use
        periodic paths and the same load-factor bound.
        """
        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_SOLVER_COMPARISON*10,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Filter for descent cases
        test_cases = [tc for tc in test_cases if tc["goal_z"] > tc["start_z"]]
        test_cases = random.sample(test_cases, NUM_CASES_SOLVER_COMPARISON)

        solvers = [
            (
                "const. vert. speed",
                get_trochoid_path_3D_const_vert_speed,
                dict(
                    high_alt_strategy="periodic paths",
                    vert_speed_constraint=None),
            ),
            (
                "rapid descent",
                get_trochoid_path_3D_longitudinal_transition,
                dict(
                    high_alt_strategy="periodic paths",
                    ac=ac,
                    transition_strategy="rapid_descent_fixed_length"),
            ),
        ]
        labels = [label for label, _, _ in solvers]

        flight_times, discr_paths = self.run_solver_comparison(
            test_cases, solvers,
            "Solver comparison: 3D trochoid rapid descent vs constant vertical speed, periodic paths, descent cases",
            load_factor_max)

        self.report_flight_time_comparison(
            flight_times,
            discr_paths,
            labels,
            "3D trochoid rapid descent vs constant vertical speed, periodic paths, descent cases",
            load_factor_max,
            aircraft=ac)

        if DEBUG_PLOTS:
            plotting.plot_stats(
                [
                    flight_times[[label]].rename(columns={label: "t_f"})
                    for label in labels
                ],
                "t_f",
                labels,
                title="Final flight time by solver")
            plotting.show()

    def test_csc3d_const_vert_speed_benchmark(self):
        """Run tests for CSC3D trochoid paths with constant vertical speed with randomized test set"""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        # wind_speed_max = 0
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        vert_speed_max = 4
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Unconstrained vertical speed
        # =====================================================================
        # Add parameters turn_n_max, high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["vert_speed_constraint"] = None

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Testing CSC3D trochoid paths with unconstrained constant vertical speed - randomized test set"
        )

        # Constrained vertical speed and periodic paths
        # =====================================================================
        # Add vertical speed cosntraints parameters high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Benchmark for CSC3D trochoid paths with constrained vertical speed, periodic paths - randomized test set"
        )

        # Constrained vertical speed and full turns
        # =====================================================================
        # Change high-altitude strategy to use full turns instead of periodic paths
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Benchmark for CSC3D trochoid paths with constrained vertical speed, full turns - randomized test set"
        )

    def test_csc3d_const_vert_speed_convergence(self):
        """Collect, plot and analyze runtime and value traces for the 3D trochoid solver with randomized test set and constrained vertical speed."""
        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        # wind_speed_max = 0
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        vert_speed_max = 4
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_CONVERGENCE,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Constrained vertical speed and periodic paths
        # =====================================================================
        # Add parameters turn_n_max, high_alt_strategy, vert_speed_constraint
        # Add vertical speed cosntraints parameters high_alt_strategy, vert_speed_constraint
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        rt_traces, value_traces, table = test_tools.collect_traces(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Collect traces for CSC 3D trochoid paths - randomized test set with vertical speed constraint"
        )

        # Print traces
        print("=== TRACES SUMMARY ===")
        print(table)
        print("=== RT TRACES ===")
        print(rt_traces)
        print("=== VALUE TRACES ===")
        print(value_traces)

        # # Assess traces
        # self.assertTrue((rt_traces["success"] == True).all())

        if DEBUG_PLOTS:
            # Plot residual of w_A_g for each span_id
            merged = value_traces.merge(table, on="trace_id", how="left")
            filtered_by_span = merged[merged["span"] ==
                                      "trochoid_CSC3D_const_vert_speed"]
            filtered_by_variable = filtered_by_span[filtered_by_span["variable"]
                                                    == "w_A_g_residual"]

            fig, ax = plotting.plot_iteration_values(
                filtered_by_variable, title="w_A_g fixed-point residual")
            plotting.show()

            fig, ax = plotting.plot_iteration_histogram(
                rt_traces,
                span="trochoid_CSC3D_const_vert_speed",
                iterations_counter="iterations",
                # iterations_counter="n_passes",      # n_passes includes added start/end rounds
                title="w_A_g iterations")
            plotting.show()

        # Constrained vertical speed and full turns
        # =====================================================================
        # Change high-altitude strategy to use full turns instead of periodic paths
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        rt_traces, value_traces, table = test_tools.collect_traces(
            test_cases, get_trochoid_path_3D_const_vert_speed,
            "Collect traces for CSC 3D trochoid paths with constrained vertical speed, full turns - randomized test set"
        )

        # Print traces
        print("=== TRACES SUMMARY ===")
        print(table)
        print("=== RT TRACES ===")
        print(rt_traces)
        print("=== VALUE TRACES ===")
        print(value_traces)

        # # Assess traces
        # self.assertTrue((rt_traces["success"] == True).all())

        if DEBUG_PLOTS:
            # Plot residual of w_A_g for each span_id
            merged = value_traces.merge(table, on="trace_id", how="left")
            filtered_by_span = merged[merged["span"] ==
                                      "trochoid_CSC3D_const_vert_speed"]
            filtered_by_variable = filtered_by_span[filtered_by_span["variable"]
                                                    == "w_A_g_residual"]

            fig, ax = plotting.plot_iteration_values(
                filtered_by_variable, title="w_A_g fixed-point residual")
            plotting.show()

            fig, ax = plotting.plot_iteration_histogram(
                rt_traces,
                span="trochoid_CSC3D_const_vert_speed",
                iterations_counter="iterations",
                # iterations_counter="n_passes",      # n_passes includes added start/end rounds
                title="w_A_g iterations")
            plotting.show()

    def test_csc3d_const_thrust_basic(self):
        """Run tests for CSC3D trochoid paths with constant thrust with basic test set"""
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)
        ac = test_tools.get_aircraft_model()
        turn_radius = 80
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        turn_rate = airspeed/turn_radius
        k_values = [0, 2, 4, 6, 8]
        l_values = [0, 1, 2]
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D(
            turn_rate,
            airspeed,
            wind_speed,
            wind_dir,
            xy_dist=[k*turn_radius for k in k_values],
            z_dist=[l*turn_radius for l in l_values],
            path_types=path_types)

        # Select a reasonable subset of test cases
        test_cases = random.sample(
            test_cases, NUM_CASES_CSC3D_CONST_THRUST_BASIC)

        # Constrained thrust with periodic paths
        # =====================================================================
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["thrust_constraint"] = lambda T: T > 0 and T < 30

        discr_paths = self.run_test_case(
            test_cases,
            get_trochoid_path_3D_const_thrust,
            "Test CSC3D trochoid paths with constant thrust and periodic paths - basic test set",
            vertical_continuity=0,
            airspeed_continuity=0)

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Constrained thrust with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases,
            get_trochoid_path_3D_const_thrust,
            "Test CSC3D trochoid paths with constant thrust and full turns - basic test set",
            vertical_continuity=0,
            airspeed_continuity=0)

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

    def test_csc3d_const_thrust_random(self):
        """Run tests for CSC3D trochoid paths with constant thrust with random test set"""
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)
        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_THRUST_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # DEBUG individual test case
        # global DEBUG_PLOTS
        # DEBUG_PLOTS = True
        # test_cases = [test_cases[454]]
        # get_trochoid_path_3D_const_thrust = get_trochoid_path_3D_const_thrust_v2

        # Constrained thrust with periodic paths
        # =====================================================================
        thrust_min = 0
        thrust_max = 30
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case[
                "thrust_constraint"] = lambda T: T > thrust_min and T < thrust_max

        discr_paths = self.run_test_case(
            test_cases,
            get_trochoid_path_3D_const_thrust,
            "Test CSC3D trochoid paths with constant thrust and periodic paths - randomized test set",
            vertical_continuity=0,
            airspeed_continuity=0)
        self.report_thrust_slack(discr_paths, ac, thrust_min, thrust_max)
        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True, aircraft=ac)

        # Plot the results
        if DEBUG_PLOTS:
            for p in discr_paths:
                ac.fly_trajectory(p)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "chi_A_dot",
                    "n",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Constrained thrust with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases,
            get_trochoid_path_3D_const_thrust,
            "Test CSC3D trochoid paths with constant thrust and full turns - randomized test set",
            vertical_continuity=0,
            airspeed_continuity=0)
        self.report_thrust_slack(discr_paths, ac, thrust_min, thrust_max)
        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True, aircraft=ac)

        # Plot the results
        if DEBUG_PLOTS:
            for p in discr_paths:
                ac.fly_trajectory(p)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

    def test_csc3d_const_thrust_benchmark(self):
        """Run tests for CSC3D trochoid paths with constant thrust with random test set"""

        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_THRUST_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Constrained thrust with periodic paths
        # =====================================================================
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["thrust_constraint"] = lambda T: T > 0 and T < 30

        discr_paths = test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_thrust,
            "Benchmark for CSC3D trochoid paths with constant thrust and periodic paths - randomized test set"
        )

        # Constrained thrust with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_thrust,
            "Benchmark for CSC3D trochoid paths with constant thrust and full turns - randomized test set"
        )

    def test_csc3d_const_thrust_convergence(self):
        """Collect, plot and analyze runtime and value traces for the CSC3D trochoid solver with constant thrust with random test set"""

        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_THRUST_CONVERGENCE,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Constrained thrust with periodic paths
        # =====================================================================
        T_min = 0
        T_max = 30
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["thrust_constraint"] = lambda T: T > T_min and T < T_max

        rt_traces, value_traces, table = test_tools.collect_traces(
            test_cases, get_trochoid_path_3D_const_thrust,
            "Collect traces for CSC 3D trochoid paths with constant thrust, full turns - randomized test set"
        )

        # Print traces
        print("=== TRACES SUMMARY ===")
        print(table)
        print("=== RT TRACES ===")
        print(rt_traces)
        print("=== VALUE TRACES ===")
        print(value_traces)

        if DEBUG_PLOTS:
            # Plot residual of w_A_g for each span_id
            merged = value_traces.merge(table, on="trace_id", how="left")
            outer_spans = merged[merged["span"] ==
                                 "trochoid_CSC3D_const_thrust"]
            inner_spans = merged[merged["span"] ==
                                 "trochoid_CSC3D_const_thrust.thrust_iteration"]
            final_alt_residuals = inner_spans[inner_spans["variable"] ==
                                              "z_g_tf_residual"]
            # thrust_iteration = inner_spans[inner_spans["variable"] ==
            #                                "T_bracket"]

            # print(thrust_iteration)
            # fig, ax = plotting.plot_iteration_values(
            #     thrust_iteration,
            #     ylabel="\Delta T",
            #     title="thrust bracket",
            #     median=False)
            # plotting.show()

            fig, ax = plotting.plot_iteration_values(
                final_alt_residuals, ylabel="z_g", title="z_g_tf residual")
            plotting.show()

            fig, ax = plotting.plot_iteration_histogram(
                rt_traces,
                span="trochoid_CSC3D_const_thrust.thrust_iteration",
                iterations_counter="iterations",
                # iterations_counter="n_passes",      # n_passes includes added start/end rounds
                title="z_g_tf iterations")
            plotting.show()

        return

        # Constrained thrust with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_const_thrust,
            "Benchmark for CSC3D trochoid paths with constant thrust and full turns - randomized test set"
        )

    def test_csc3d_rapid_descent_basic(self):
        """Run tests for CSC3D trochoid paths with rapid descent transition with basic test set."""
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)
        ac = test_tools.get_aircraft_model()

        turn_radius = 80
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        turn_rate = airspeed/turn_radius
        k_values = [0, 2, 4, 6, 8]
        l_values = [0.5, 1]
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D(
            turn_rate,
            airspeed,
            wind_speed,
            wind_dir,
            xy_dist=[k*turn_radius for k in k_values],
            z_dist=[l*turn_radius for l in l_values],
            path_types=path_types)

        # Filter for descent cases
        test_cases = [tc for tc in test_cases if tc["goal_z"] > tc["start_z"]]

        # Select a reasonable subset of test cases
        test_cases = random.sample(
            test_cases, NUM_CASES_CSC3D_RAPID_DESCENT_BASIC)

        # # Select a specific case for debugging
        # global DEBUG_PLOTS
        # DEBUG_PLOTS = True
        # test_cases = [test_cases[418]]

        # Fixed final altitude, with periodic paths
        # =====================================================================
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["transition_strategy"] = "rapid_descent_fixed_length"

        discr_paths = self.run_test_case(
            test_cases,
            get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and periodic paths - basic test set",
        )

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Fixed final altitude, with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and full turns - basic test set"
        )

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Free final altitude
        # =====================================================================
        for test_case in test_cases:
            test_case["goal_z"] = None

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and free final altitude - basic test set"
        )

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

    def test_csc3d_rapid_descent_random(self):
        """Run tests for CSC3D trochoid paths with rapid descent transition with randomized test set."""
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)
        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        thrust_min = 0
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_RAPID_DESCENT_RANDOM*10,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Filter for descent cases
        test_cases = [tc for tc in test_cases if tc["goal_z"] > tc["start_z"]]

        # Select a reasonable subset of test cases
        test_cases = random.sample(
            test_cases, NUM_CASES_CSC3D_RAPID_DESCENT_RANDOM)

        # # Select a specific case for debugging
        # global DEBUG_PLOTS
        # DEBUG_PLOTS = True
        # test_cases = [
        #     test_cases[5],
        # ]

        # Fixed final altitude with periodic paths
        # =====================================================================
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["transition_strategy"] = "rapid_descent_fixed_length"

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and periodic paths - randomized test set"
        )
        self.report_load_factor_slack(
            discr_paths, 4, turns_only=True, aircraft=ac)
        self.report_thrust_slack(discr_paths, ac, thrust_min, None)

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Fixed final altitude with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and full turns - randomized test set"
        )
        self.report_load_factor_slack(
            discr_paths, 4, turns_only=True, aircraft=ac)
        self.report_thrust_slack(discr_paths, ac, thrust_min, None)

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

        # Free final altitude
        # =====================================================================
        for test_case in test_cases:
            test_case["goal_z"] = None

        discr_paths = self.run_test_case(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Test CSC3D trochoid paths with rapid descent transition and free final altitude - randomized test set"
        )
        self.report_load_factor_slack(
            discr_paths, 4, turns_only=True, aircraft=ac)
        self.report_thrust_slack(discr_paths, ac, thrust_min, None)

        # Plot the results
        if DEBUG_PLOTS:
            for discr_path in discr_paths:
                ac.fly_trajectory(discr_path)
            plotting.plot_ref_trajectory(paths=[p.df for p in discr_paths])
            plotting.plot_timeline(
                trajectories=[p.df for p in discr_paths],
                subplots=[
                    # "V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
                    "V_A",
                    "V_A_xy",
                    "w_A_g",
                    "T"
                ],
                columns=1)
            plotting.show()

    def test_csc3d_rapid_descent_benchmark(self):
        """Run benchmark for CSC3D trochoid paths with rapid descent transition with randomized test set."""
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)
        ac = test_tools.get_aircraft_model()

        turn_radius_min = 50
        turn_radius_max = 200
        airspeed_min = 15
        airspeed_max = 25
        wind_speed_min = 0
        wind_speed_max = 10
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_min = 0
        k_max = 8
        l_min = -1
        l_max = 1
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_RAPID_DESCENT_BENCHMARK*10,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            z_dist_bounds=[l_min*turn_radius_max, l_max*turn_radius_max],
            path_types=path_types,
        )

        # Filter for descent cases
        test_cases = [tc for tc in test_cases if tc["goal_z"] > tc["start_z"]]

        # Select a reasonable subset of test cases
        test_cases = random.sample(
            test_cases, NUM_CASES_CSC3D_RAPID_DESCENT_BENCHMARK)

        # Select a specific case for debugging
        # global DEBUG_PLOTS
        # DEBUG_PLOTS = True
        # test_cases = test_cases[96:97]

        # Rapid descent with periodic paths
        # =====================================================================
        for test_case in test_cases:
            test_case["turn_n_max"] = load_factor_max
            test_case["high_alt_strategy"] = "periodic paths"
            test_case["ac"] = ac
            test_case["transition_strategy"] = "rapid_descent_fixed_length"

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Benchmark of CSC3D trochoid paths with rapid descent transition, periodic paths - randomized test set"
        )

        # Rapid descent with full turns
        # =====================================================================
        for test_case in test_cases:
            test_case["high_alt_strategy"] = "full turns"
            test_case["smooth_thrust"] = False

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Benchmark of CSC3D trochoid paths with rapid descent transition, full turns - randomized test set"
        )

        # Rapid descent with free final altitude
        # =====================================================================
        for test_case in test_cases:
            test_case["goal_z"] = None

        test_tools.run_benchmark(
            test_cases, get_trochoid_path_3D_longitudinal_transition,
            "Benchmark of CSC3D trochoid paths with rapid descent transition, free final altitutde - randomized test set"
        )

    def test_cs_basic(self):
        turn_radius = 100
        airspeed = 15
        wind_speed = 5
        wind_dir = math.pi*0.25
        turn_rate = airspeed/turn_radius
        # Create a start and a set of goal points
        start = Sample(50, 100, math.radians(45))
        k = 10
        goals = [
            Sample(k*turn_radius, 0, None),
            Sample(-k*turn_radius, 0, None),
            Sample(0, -k*turn_radius, None),
            Sample(0, k*turn_radius, None)
        ]
        path_types = ["LS", "RS"]
        # Test computation of all the path types
        discr_paths = []
        for goal in goals:
            for path_type in path_types:
                path = get_trochoid_path_2D(
                    start.x_g, start.y_g, start.chi, goal.x_g, goal.y_g, None,
                    airspeed, turn_rate, wind_speed, wind_dir, path_type)

                self.assertIsNotNone(
                    path,
                    msg=f"Failed to plan {path_type} path"
                    f" from {start} to {goal}")
                path_discr = path.discretize()
                path_discr.check_kinematic_consistency(
                    1, 1, 0, 1, raise_err=True, abs_err_threshold=1e-6)
                discr_paths.append(path_discr)

        # Plot the results
        if DEBUG_PLOTS:
            # if True:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()
        return

    def test_trochoid_path_csca_beautiful_plot(self):
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=10,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=800,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=35)
        ac = test_tools.get_aircraft_model()

        # Horizontal CSCA path
        # =========================================
        airspeed_1 = 15
        airspeed_2 = 20
        # airspeed_2_mps = 20
        wind_speed = 5
        # wind_speed = 0
        wind_dir = -math.pi*0.25
        turn_n_max = 2.5
        turn_rate_1 = airspeed_1/50
        turn_rate_2 = airspeed_2/100

        print(
            f"turn_rate_1: {math.degrees(turn_rate_1):.3f} deg/s, turn_rate_2: {math.degrees(turn_rate_2):.3f} deg/s"
        )

        # Create a start and a set of goal points
        start = Sample(0, 0, math.radians(-30))
        goal = Sample(400, 200, math.radians(180))

        path_types = ["LSL", "RSR", "RSL", "LSR"]

        # Test computation of all the path types
        discr_paths = []
        for path_type in path_types:
            path = get_trochoid_path_2D(
                start_x=start.x_g,
                start_y=start.y_g,
                start_dir=start.chi,
                goal_x=goal.x_g,
                goal_y=goal.y_g,
                goal_dir=goal.chi,
                airspeed=(airspeed_1, airspeed_2),
                turn_rate=(turn_rate_1, turn_rate_2),
                wind_speed=wind_speed,
                wind_dir=wind_dir,
                path_type=path_type)
            self.assertIsNotNone(
                path,
                msg=f"Failed to plan {path_type} path "
                f"from {start} to {goal}")
            path_discr = path.discretize()
            path_discr.check_kinematic_consistency(
                1, 1, 0, 1, raise_err=True, abs_err_threshold=1e-6)
            # Check initial and final state
            self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
            self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=0)
            self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g)
            self.assertAlmostEqual(path_discr.xf["y_g"], goal.y_g, places=0)
            self.assertAlmostEqual(
                abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
                0.0,
                places=2)
            self.assertAlmostEqual(
                abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
                0.0,
                places=2)

            # Store all paths for plotting
            discr_paths.append(path_discr)

        # Plot the results
        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=start.x_g,
                y_g=start.y_g,
                # chi=start.chi,
                label="start",
                s=40,
                color="tab:cyan",
                # style="go",
                marker=(3, 0, math.degrees(start.chi + math.pi))),
            dict(
                x_g=goal.x_g,
                y_g=goal.y_g,
                # chi=goal.chi,
                label="goal",
                s=40,
                color="tab:pink",
                # style="gx",
                marker=(3, 0, math.degrees(goal.chi)))
        ]

        fig, ax = plotting.plot_topview(
            trajs=[p.df for p in discr_paths],
            POIs=POIs,
            s_col="V_A_xy",
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            title=False)
        # ax["T"].set_ylim(-1, 5)

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("trochoid_path_csca.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0", "trochoid_path_csca.pdf",
                "trochoid_path_csca.cropped.pdf"
            ],
            check=True)
        if DEBUG_PLOTS:
            plotting.show()

    def test_trochoid_path_3D_const_vert_speed_beautiful_plot(self):

        airspeed_1 = 20
        airspeed_2 = 20
        # airspeed_2_mps = 20
        wind_speed = 5
        # wind_speed = 0
        wind_dir = math.pi*0.25
        turn_n_max = 1.5
        # Create a start and a set of goal points
        start = Sample3D(0, 0, 0, math.radians(-30))
        goal = Sample3D(200, 200, -75, math.radians(-30))

        path_type = "LSL"

        discr_paths = []

        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=50)

        ac = test_tools.get_aircraft_model()

        # Path will full turns
        # ====================
        path = get_trochoid_path_3D_const_vert_speed(
            start_x=start.x_g,
            start_y=start.y_g,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g,
            goal_y=goal.y_g,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=(airspeed_1, airspeed_2),
            # turn_rate=(turn_rate_1_radps, turn_rate_2_radps),
            # turn_n_max=None,
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            vert_speed_constraint=lambda w: abs(w) < 2,
            high_alt_strategy="full turns")
        self.assertIsNotNone(
            path,
            msg=f"Failed to plan {path_type} path "
            f"from {start} to {goal}")
        path_discr = path.discretize()
        path_discr.check_kinematic_consistency(
            1, 1, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
        self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=2)
        self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g)
        self.assertAlmostEqual(path_discr.xf["y_g"], goal.y_g, places=2)
        self.assertAlmostEqual(path_discr.x0["z_g"], start.z_g)
        self.assertAlmostEqual(path_discr.xf["z_g"], goal.z_g, places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=2)
        ac.fly_trajectory(path_discr)
        discr_paths.append(path_discr)

        # Path with periodic path
        # ====================
        periodic_path_y_offset = -200
        path = get_trochoid_path_3D_const_vert_speed(
            start_x=start.x_g,
            start_y=start.y_g + periodic_path_y_offset,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g,
            goal_y=goal.y_g + periodic_path_y_offset,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=(airspeed_1, airspeed_2),
            # turn_rate=(turn_rate_1_radps, turn_rate_2_radps),
            # turn_n_max=None,
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            vert_speed_constraint=lambda w: abs(w) < 2,
            high_alt_strategy="periodic paths")
        self.assertIsNotNone(
            path,
            msg=f"Failed to plan {path_type} path "
            f"from {start} to {goal}")
        path_discr = path.discretize()
        path_discr.check_kinematic_consistency(
            1, 1, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
        self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=2)
        self.assertAlmostEqual(
            path_discr.x0["y_g"], start.y_g + periodic_path_y_offset)
        self.assertAlmostEqual(
            path_discr.xf["y_g"], goal.y_g + periodic_path_y_offset, places=2)
        self.assertAlmostEqual(path_discr.x0["z_g"], start.z_g)
        self.assertAlmostEqual(path_discr.xf["z_g"], goal.z_g, places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=2)
        ac.fly_trajectory(path_discr)
        discr_paths.append(path_discr)

        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=(start.x_g, start.x_g),
                y_g=(start.y_g, start.y_g + periodic_path_y_offset),
                z_g=(start.z_g, start.z_g),
                # chi=start.chi,
                label="start",
                # style="go",
                s=10,
                marker=(3, 0, math.degrees(start.chi) + 180),
                color="tab:cyan",
            ),
            dict(
                x_g=(goal.x_g, goal.x_g),
                y_g=(goal.y_g, goal.y_g + periodic_path_y_offset),
                z_g=(goal.z_g, goal.z_g),
                # chi=goal.chi,
                label="goal",
                # style="gx",
                s=10,
                marker=(3, 0, math.degrees(goal.chi) + 180),
                color="tab:pink",
            ),
        ]

        fig, ax = plotting.plot_ref_trajectory(
            paths={
                "continuous turns": discr_paths[0].df,
                "periodic paths": discr_paths[1].df,
            },
            POIs=POIs,
            s_col=None,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            timelines=["n", "T"])
        ax["T"].set_ylim(10, 18)

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("trochoid_path_csc3d_const_vert_speed.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0",
                "trochoid_path_csc3d_const_vert_speed.pdf",
                "trochoid_path_csc3d_const_vert_speed.cropped.pdf"
            ],
            check=True)
        if DEBUG_PLOTS:
            plotting.show()

    def test_trochoid_path_3D_const_thrust_varspeed_beautiful_plot(self):

        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=10,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=800,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=35)
        ac = test_tools.get_aircraft_model()

        # v_a_cruise = ac.airspeed_opt(gamma_A=0)
        # v_a_descent = ac.airspeed_opt(gamma_A=math.radians(10))
        # print(f"{v_a_cruise=}, {v_a_descent=}")
        # plotting.plot_thrust_over_airspeed(
        #     ac,
        #     gamma_A=[
        #         -math.radians(20),
        #         -math.radians(15),
        #         -math.radians(10),
        #         -math.radians(5),
        #         0,
        #         math.radians(10),
        #         math.radians(15),
        #         math.radians(20),
        #     ])
        # plotting.show()
        # return

        # Horizontal CSC path
        # =========================================
        airspeed_1 = 25
        airspeed_2 = 20
        # airspeed_2_mps = 20
        wind_speed = 5
        # wind_speed = 0
        wind_dir = -math.pi*0.25 + math.pi
        turn_n_max = 2.5

        # Create a start and a set of goal points
        start = Sample3D(0, 0, -25, math.radians(-30))
        goal = Sample3D(120, 180, 0, math.radians(-90))

        path_type = "RSL"

        # Test computation of all the path types
        discr_paths = []
        path = get_trochoid_path_3D_const_thrust(
            start_x=start.x_g,
            start_y=start.y_g,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g,
            goal_y=goal.y_g,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=(airspeed_1, airspeed_2),
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            ac=ac,
            thrust_constraint=lambda T: T > 0 and T < 25,
            high_alt_strategy="periodic paths")
        self.assertIsNotNone(
            path,
            msg=f"Failed to plan {path_type} path "
            f"from {start} to {goal}")
        path_discr = path.discretize()
        path_discr.check_kinematic_consistency(
            1, 0, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
        self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=0)
        self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g)
        self.assertAlmostEqual(path_discr.xf["y_g"], goal.y_g, places=0)
        self.assertAlmostEqual(path_discr.x0["z_g"], start.z_g)
        self.assertAlmostEqual(path_discr.xf["z_g"], goal.z_g, places=0)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=2)

        # Store all paths for plotting
        discr_paths.append(path_discr)

        # Plot the results
        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=start.x_g,
                y_g=start.y_g,
                z_g=start.z_g,
                # chi=start.chi,
                label="start",
                s=10,
                color="tab:cyan",
                # style="go",
                marker=(3, 0, math.degrees(start.chi + math.pi))),
            dict(
                x_g=goal.x_g,
                y_g=goal.y_g,
                z_g=goal.z_g,
                # chi=goal.chi,
                label="goal",
                s=10,
                color="tab:pink",
                # style="gx",
                marker=(3, 0, math.degrees(goal.chi + math.pi)))
        ]

        ac.fly_trajectory(path_discr)
        fig, ax = plotting.plot_ref_trajectory(
            paths=path_discr.df,
            POIs=POIs,
            s_col=None,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            timelines=["T", "V_A", "w_A_g"])
        ax["T"].set_ylim(-1, 2)
        ax["3D view"].view_init(elev=-20, azim=290, roll=180)

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("trochoid_path_csc3d_const_thrust_varspeed.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0",
                "trochoid_path_csc3d_const_thrust_varspeed.pdf",
                "trochoid_path_csc3d_const_thrust_varspeed.cropped.pdf"
            ],
            check=True)
        if DEBUG_PLOTS:
            plotting.show()

    def test_trochoid_path_3D_const_thrust_beautiful_plot(self):

        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=10,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=800,
        #     n_min=0.2,
        #     n_max=2.0,
        #     mu_A_max=math.radians(60),
        #     V_A_max=35)
        ac = test_tools.get_aircraft_model()

        # v_a_cruise = ac.airspeed_opt(gamma_A=0)
        # v_a_descent = ac.airspeed_opt(gamma_A=math.radians(10))
        # print(f"{v_a_cruise=}, {v_a_descent=}")
        # plotting.plot_thrust_over_airspeed(
        #     ac,
        #     gamma_A=[
        #         -math.radians(20),
        #         -math.radians(15),
        #         -math.radians(10),
        #         -math.radians(5),
        #         0,
        #         math.radians(10),
        #         math.radians(15),
        #         math.radians(20),
        #     ])
        # plotting.show()
        # return

        # Horizontal CSC path
        # =========================================
        airspeed_1 = 20
        airspeed_2 = 20
        # airspeed_2_mps = 20
        wind_speed = 5
        # wind_speed = 0
        wind_dir = -math.pi*0.25
        turn_n_max = 2.5

        # Create a start and a set of goal points
        start = Sample3D(0, 0, 0, math.radians(-30))
        goal = Sample3D(100, 100, -50, math.radians(-90))

        path_type = "RSL"

        # Test computation of all the path types
        discr_paths = []
        path = get_trochoid_path_3D_const_thrust(
            start_x=start.x_g,
            start_y=start.y_g,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g,
            goal_y=goal.y_g,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=(airspeed_1, airspeed_2),
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            ac=ac,
            thrust_constraint=lambda T: T > 0 and T < 25,
            high_alt_strategy="periodic paths")
        self.assertIsNotNone(
            path,
            msg=f"Failed to plan {path_type} path "
            f"from {start} to {goal}")
        path_discr = path.discretize()
        path_discr.check_kinematic_consistency(
            1, 0, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
        self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=0)
        self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g)
        self.assertAlmostEqual(path_discr.xf["y_g"], goal.y_g, places=0)
        self.assertAlmostEqual(path_discr.x0["z_g"], start.z_g)
        self.assertAlmostEqual(path_discr.xf["z_g"], goal.z_g, places=0)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=2)

        # Store all paths for plotting
        discr_paths.append(path_discr)

        # Plot the results
        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=start.x_g,
                y_g=start.y_g,
                z_g=start.z_g,
                # chi=start.chi,
                label="start",
                s=10,
                color="tab:cyan",
                # style="go",
                marker=(3, 0, math.degrees(start.chi + math.pi))),
            dict(
                x_g=goal.x_g,
                y_g=goal.y_g,
                z_g=goal.z_g,
                # chi=goal.chi,
                label="goal",
                s=10,
                color="tab:pink",
                # style="gx",
                marker=(3, 0, math.degrees(goal.chi + math.pi)))
        ]

        ac.fly_trajectory(path_discr)
        fig, ax = plotting.plot_ref_trajectory(
            paths=path_discr.df,
            POIs=POIs,
            s_col=None,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            timelines=["n", "T", "w_A_g"])
        ax["T"].set_ylim(-1, 30)

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("trochoid_path_csc3d_const_thrust.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0",
                "trochoid_path_csc3d_const_thrust.pdf",
                "trochoid_path_csc3d_const_thrust.cropped.pdf"
            ],
            check=True)
        if DEBUG_PLOTS:
            plotting.show()

    def test_trochoid_path_3D_rapid_descent_beautiful_plot(self):
        # ac = ExtDubinsAirplane(
        #     g=9.81,
        #     m=4,
        #     rho=1.225,
        #     S=0.98,
        #     C_D0=0.032,
        #     C_L0=0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=5000,
        #     n_min=0.2,
        #     n_max=1.2,
        #     mu_A_max=math.radians(60),
        #     V_A_max=35)
        ac = test_tools.get_aircraft_model()

        airspeed = 20
        # airspeed_2_mps = 20
        wind_speed = 5
        # wind_speed = 0
        wind_dir = -math.pi*0.25
        turn_n_max = 2.5

        # Create a start and a set of goal points
        start = Sample3D(0, 0, -100, math.radians(-30))
        goal = Sample3D(0, 100, 0, math.radians(-90))

        path_type = "RSL"

        discr_paths = []

        x_offset = 50
        y_offset = 10
        path = get_trochoid_path_3D_longitudinal_transition(
            start_x=start.x_g + x_offset,
            start_y=start.y_g + y_offset,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g + x_offset,
            goal_y=goal.y_g + y_offset,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=airspeed,
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            ac=ac,
            thrust_constraint=lambda T: T > 0 and T < 30,
            transition_strategy="rapid_descent_fixed_length",
            # high_alt_strategy="periodic paths",
            high_alt_strategy="full turns",
            smooth_thrust=False)
        self.assertIsNotNone(
            path,
            msg=f"Failed to plan {path_type} path "
            f"from {start} to {goal}")
        path_discr = path.discretize()
        ac.fly_trajectory(path_discr)
        path_discr.check_kinematic_consistency(
            1, 1, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g + x_offset)
        self.assertAlmostEqual(
            path_discr.xf["x_g"], goal.x_g + x_offset, places=0)
        self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g + y_offset)
        self.assertAlmostEqual(
            path_discr.xf["y_g"], goal.y_g + y_offset, places=0)
        self.assertAlmostEqual(path_discr.x0["z_g"], start.z_g)
        self.assertAlmostEqual(path_discr.xf["z_g"], goal.z_g, places=0)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=2)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=2)

        discr_paths.append(path_discr)

        # Plan constant thrust path for comparison
        shifted_initial_turn = LiftedSegment(
            copy.deepcopy(path.horiz_profile.segments[0]),
            copy.deepcopy(path.vert_profile.segments[0])).shift(
                0, x_g=-x_offset, y_g=-y_offset)
        # discr_paths.append(shifted_initial_turn.discretize())
        # ac.fly_trajectory(discr_paths[-1])
        path_const_thrust = get_trochoid_path_3D_const_thrust(
            start_x=shifted_initial_turn.xf["x_g"],
            start_y=shifted_initial_turn.xf["y_g"],
            start_z=shifted_initial_turn.xf["z_g"],
            start_dir=shifted_initial_turn.xf["chi"],
            goal_x=goal.x_g - x_offset,
            goal_y=goal.y_g - y_offset,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=airspeed,
            turn_rate=None,
            turn_n_max=turn_n_max,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type="LSL",
            ac=ac,
            thrust_constraint=lambda T: T > 0 and T < 15,
            high_alt_strategy="full turns",
        )
        path_const_thrust = Trajectory(
            [shifted_initial_turn, path_const_thrust])
        path_const_thrust_discr = path_const_thrust.discretize()
        ac.fly_trajectory(path_const_thrust_discr)
        # path_const_thrust_discr.df[["u_K_g", "v_K_g", "x_g", "y_g",
        #                             "chi"]].plot()
        # plotting.show()
        path_const_thrust_discr.check_kinematic_consistency(
            1, 0, 0, 1, raise_err=True, abs_err_threshold=1e-6)
        discr_paths.append(path_const_thrust_discr)

        # Plot the results
        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=start.x_g - x_offset,
                y_g=start.y_g - y_offset,
                z_g=start.z_g,
                # chi=start.chi,
                label="start",
                s=10,
                color="tab:cyan",
                # style="go",
                marker=(3, 0, math.degrees(start.chi + math.pi))),
            dict(
                x_g=goal.x_g - x_offset,
                y_g=goal.y_g - y_offset,
                z_g=goal.z_g,
                # chi=goal.chi,
                label="goal",
                s=10,
                color="tab:pink",
                # style="gx",
                marker=(3, 0, math.degrees(goal.chi + math.pi))),
            dict(
                x_g=start.x_g + x_offset,
                y_g=start.y_g + y_offset,
                z_g=start.z_g,
                # chi=start.chi,
                s=10,
                color="tab:cyan",
                # style="go",
                marker=(3, 0, math.degrees(start.chi + math.pi))),
            dict(
                x_g=goal.x_g + x_offset,
                y_g=goal.y_g + y_offset,
                z_g=goal.z_g,
                # chi=goal.chi,
                s=10,
                color="tab:pink",
                # style="gx",
                marker=(3, 0, math.degrees(goal.chi + math.pi)))
        ]

        fig, ax = plotting.plot_ref_trajectory(
            paths={
                "const. thrust": discr_paths[1].df,
                "rapid descent": discr_paths[0].df,
            },
            # paths=[p.df for p in discr_paths],
            POIs=POIs,
            s_col=None,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            timelines=["n", "T", "V_A", "w_A_g"])
        # ax["T"].set_ylim(-1, 5)

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("trochoid_path_csc3d_rapid_descent.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0",
                "trochoid_path_csc3d_rapid_descent.pdf",
                "trochoid_path_csc3d_rapid_descent.cropped.pdf"
            ],
            check=True)
        if DEBUG_PLOTS:
            plotting.show()

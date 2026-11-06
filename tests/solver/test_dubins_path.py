# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.dubins_path.

Run from top-level package directory:

   python3 -m unittest tests.solver.test_dubins_path

"""

import math
import os
import random
import statistics
import subprocess
import traceback
import unittest

from tqdm import tqdm

from eduplane.solver.dubins_path import *
from eduplane.utils import plotting, utils

from . import test_tools
from .test_tools import Sample3D

DEBUG_PLOTS = False
# DEBUG_PLOTS = True

PRINT_TRACES_REPORT = False

NUM_CASES = int(os.getenv("NUM_TEST_CASES", default=100))

NUM_CASES_CSC_BASIC = NUM_CASES
NUM_CASES_CSC_RANDOM = NUM_CASES
NUM_CASES_CSC_BENCHMARK = NUM_CASES
NUM_CASES_CSC_CONVERGENCE = NUM_CASES

NUM_CASES_CSC3D_BASIC = NUM_CASES
NUM_CASES_CSC3D_RANDOM = NUM_CASES
NUM_CASES_CSC3D_BENCHMARK = NUM_CASES
NUM_CASES_CSC3D_CONVERGENCE = NUM_CASES

NUM_CASES_CSC3D_CONST_VS_BASIC = NUM_CASES
NUM_CASES_CSC3D_CONST_VS_RANDOM = NUM_CASES
NUM_CASES_CSC3D_CONST_VS_BENCHMARK = NUM_CASES
NUM_CASES_CSC3D_CONST_VS_CONVERGENCE = NUM_CASES


class DubinsPathTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)
        logger.reset()
        logger.enable_rt_traces()

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

    def run_test_case(self, test_cases, test_func, desc):
        discr_paths = []
        print(f"Running test suite: {desc}")
        with tqdm(total=len(test_cases), desc=desc) as pbar:
            for test_case in test_cases:
                pbar.update()
                path = test_func(**test_case)
                # LSR/RSL paths should exist for all cases k >= 4, for k < 4 they may exist or not.
                # NOTE: The sufficient condition for existance is D>=2R where R is the turn radius and D is the
                # distance between the midpoints of the tangential circles.
                # Given just the distance between start and goal D_sg that gives D_sg >= 4R.
                D_sg = math.sqrt(
                    (test_case["goal_x"] - test_case["start_x"])**2 +
                    (test_case["goal_y"] - test_case["start_y"])**2)
                if test_case["path_type"] in [
                        "LSR", "RSL"
                ] and D_sg <= 4*test_case["turn_radius"]:
                    if path is None:
                        continue

                self.assertIsNotNone(
                    path,
                    msg=
                    f"Failed to plan {test_case['path_type']} path. Full test case:\n    {test_case}"
                )
                try:
                    path_discr = path.discretize()
                    path_discr.check_kinematic_consistency(
                        1, 1, 0, 1, raise_err=True)
                    # Check initial and final state
                    self.assertAlmostEqual(
                        path_discr.x0["x_g"], test_case["start_x"])
                    self.assertAlmostEqual(
                        path_discr.xf["x_g"], test_case["goal_x"], places=3)
                    self.assertAlmostEqual(
                        path_discr.x0["y_g"], test_case["start_y"])
                    self.assertAlmostEqual(
                        path_discr.xf["y_g"], test_case["goal_y"], places=3)
                    if "start_z" in test_case:
                        self.assertAlmostEqual(
                            path_discr.x0["z_g"],
                            test_case["start_z"],
                            places=3)
                    if "goal_z" in test_case:
                        self.assertAlmostEqual(
                            path_discr.xf["z_g"], test_case["goal_z"], places=3)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.x0["chi"], test_case["start_dir"])),
                        0.0,
                        places=3)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.xf["chi"], test_case["goal_dir"])),
                        0.0,
                        places=3)
                    discr_paths.append(path_discr)
                except (AssertionError, ValueError) as err:
                    if DEBUG_PLOTS:
                        print("Test case failed with assertion error:", err)
                        if "z_g" in path_discr:
                            if "ac" in test_case:
                                test_case["ac"].fly_trajectory(path_discr)
                            POIs = test_tools.get_start_goal_POIs(test_case)
                            plotting.plot_ref_trajectory(
                                paths=[path_discr.df], POIs=POIs)
                            plotting.plot_timeline(
                                trajectories=[path_discr.df],
                                subplots=[
                                    "V_A", "V_A_xy", "V_A_xy_dot", "w_A_g",
                                    "gamma_A", "T", "n"
                                ],
                                columns=2)
                        else:
                            plotting.plot_topview([path_discr.df])
                            plotting.plot_timeline(
                                trajectories=[path_discr.df],
                                subplots=[
                                    "x_g", "y_g", "u_K_g", "v_K_g", "chi",
                                    "chi_A"
                                ],
                                columns=2)
                            plotting.show()
                    traceback.print_exc()
                    raise

        if PRINT_TRACES_REPORT:
            logger.printReport()
        return discr_paths

    def test_csc_basic(self):
        """Test 2D Dubins solver with basic test cases"""
        turn_radius = 100
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_values = [0, 2, 4, 6, 8]

        test_cases = test_tools.get_test_cases_2D(
            turn_rate=airspeed/turn_radius,
            airspeed=airspeed,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            xy_dist=[k*turn_radius for k in k_values],
            path_types=["LSL", "RSR", "LSR", "RSL"],
            angles=[0,
                    math.radians(90),
                    math.radians(180),
                    math.radians(270)])

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSC_BASIC)

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_2D,
            "Testing CSC 2D Dubins paths - basic test set")

        # Plot the results
        if DEBUG_PLOTS:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()

    def test_csc_random(self):
        """Test 2D Dubins solver with randomized test cases"""
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
            path_types=["LSL", "RSR", "LSR", "RSL"],
        )

        # test_cases = test_cases[24:25]

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_2D,
            "Testing CSC 2D Dubins paths - randomized test set")

        # Plot the results
        if DEBUG_PLOTS:
            plotting.plot_topview([p.df for p in discr_paths])
            # plotting.plot_timeline(
            #     trajectories=[p.df for p in discr_paths],
            #     subplots=["V_A_xy", "chi_A_dot"],
            #     columns=1)
            plotting.show()

    def test_csc_benchmark(self):
        """Run benchmark for 2D Dubins solver with randomized test cases"""
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
            num_test_cases=NUM_CASES_CSC_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=[k_min*turn_radius_min, k_max*turn_radius_max],
            path_types=path_types,
        )

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]

        test_tools.run_benchmark(
            test_cases, get_dubins_path_2D, "Benchmark for CSC 2D Dubins paths")

    def test_csc3d_basic(self):
        """Test the 3D Dubins solver with basic test set."""
        turn_radius = 100
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_values = [0, 2, 4, 6, 8]
        l_values = [0, 0.5, 1]
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D(
            turn_rate=airspeed/turn_radius,
            airspeed=airspeed,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            xy_dist=[k*turn_radius for k in k_values],
            z_dist=[l*turn_radius for l in l_values],
            path_types=path_types)

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSC3D_BASIC)

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_3D,
            "Testing CSC 3D Dubins Airplane paths - basic test set")

        # Plot the results
        if DEBUG_PLOTS:
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

    def test_csc3d_randomized(self):
        """Test the 3D Dubins solver with randomized test set."""
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
            num_test_cases=NUM_CASES_CSC3D_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=(k_min*turn_radius_min, k_max*turn_radius_max),
            z_dist_bounds=(l_min*turn_radius_max, l_max*turn_radius_max),
            path_types=path_types)

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_3D,
            "Testing CSC 3D Dubins Airplane paths - randomized test set")

        # Plot the results
        if DEBUG_PLOTS:
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

    def test_csc3d_benchmark(self):
        """Run benchmark for 3D Dubins solver with randomized test set."""
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
            num_test_cases=NUM_CASES_CSC3D_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=(k_min*turn_radius_min, k_max*turn_radius_max),
            z_dist_bounds=(l_min*turn_radius_max, l_max*turn_radius_max),
            path_types=path_types)

        # Replace field turn_rate with turn_radius
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max

        test_tools.run_benchmark(
            test_cases, get_dubins_path_3D,
            "Benchmark for CSC 3D Dubins Airplane paths")

    def test_csc3d_constrained_vert_speed_basic(self):
        """Test the 3D Dubins solver with basic test set and constrained vertical speed."""
        turn_radius = 100
        airspeed = 15
        wind_speed = 5
        wind_dir = -math.pi*0.25
        path_types = ["LSL", "RSR", "LSR", "RSL"]
        k_values = [0, 2, 4, 6, 8]
        l_values = [0, 1, 2]
        load_factor_max = 4
        vs_constraint = lambda w_A_g: abs(w_A_g) < 2

        test_cases = test_tools.get_test_cases_3D(
            turn_rate=airspeed/turn_radius,
            airspeed=airspeed,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            xy_dist=[k*turn_radius for k in k_values],
            z_dist=[l*turn_radius for l in l_values],
            path_types=path_types)

        # Select a reasonable subset of test cases
        test_cases = random.sample(test_cases, NUM_CASES_CSC3D_CONST_VS_BASIC)

        # Replace field turn_rate with turn_radius and add vertical speed constraint
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max
            test_case["vert_speed_constraint"] = vs_constraint

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_3D,
            "Testing CSC 3D Dubins Airplane paths - basic test set with vertical speed constraint"
        )

        # Plot the results
        if DEBUG_PLOTS:
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

    def test_csc3d_constrained_vert_speed_randomized(self):
        """Test the 3D Dubins solver with basic test set and constrained vertical speed."""
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
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_RANDOM,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=(k_min*turn_radius_min, k_max*turn_radius_max),
            z_dist_bounds=(l_min*turn_radius_max, l_max*turn_radius_max),
            path_types=path_types)

        # Select test case for debugging
        # global DEBUG_PLOTS
        # DEBUG_PLOTS = True
        # test_cases = [test_cases[176]]

        # Replace field turn_rate with turn_radius and add vertical speed constraint
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            # test_case["turn_radius"] = None
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        discr_paths = self.run_test_case(
            test_cases, get_dubins_path_3D,
            "Testing CSC 3D Dubins Airplane paths - randomized test set with vertical speed constraint"
        )

        self.report_load_factor_slack(
            discr_paths, load_factor_max, turns_only=True)

        vertical_speed_slacks = [
            vert_speed_max - abs(w_A_g) for path in discr_paths
            for w_A_g in path.df["w_A_g"]
        ]
        violations = sum(slack < 0 for slack in vertical_speed_slacks)
        print(
            "Vertical speed constraint slack [m/s] "
            f"(min/median/max): {min(vertical_speed_slacks):.6g} / "
            f"{statistics.median(vertical_speed_slacks):.6g} / "
            f"{max(vertical_speed_slacks):.6g}; "
            f"violating samples: {violations}/{len(vertical_speed_slacks)}")
        self.assertEqual(violations, 0)

        # Plot the results
        if DEBUG_PLOTS:
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

    def test_csc3d_constrained_vert_speed_benchmark(self):
        """Run benchmark for the 3D Dubins solver with randomized test set and constrained vertical speed."""
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
        vert_speed_max = 2
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=(k_min*turn_radius_min, k_max*turn_radius_max),
            z_dist_bounds=(l_min*turn_radius_max, l_max*turn_radius_max),
            path_types=path_types)

        # Replace field turn_rate with turn_radius and add vertical speed constraint
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        test_tools.run_benchmark(
            test_cases, get_dubins_path_3D,
            "Benchmark for CSC 3D Dubins Airplane paths - randomized test set with vertical speed constraint"
        )

    def test_csc3d_constrained_vert_speed_convergence(self):
        """Collect, plot and analyze runtime and value traces for the 3D Dubins solver with randomized test set and constrained vertical speed."""
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
        vert_speed_max = 2
        load_factor_max = 4

        test_cases = test_tools.get_test_cases_3D_randomized(
            num_test_cases=NUM_CASES_CSC3D_CONST_VS_BENCHMARK,
            turn_rate_bounds=(
                airspeed_min/turn_radius_max, airspeed_max/turn_radius_min),
            airspeed_bounds=(airspeed_min, airspeed_max),
            wind_speed_bounds=(wind_speed_min, wind_speed_max),
            xy_dist_bounds=(k_min*turn_radius_min, k_max*turn_radius_max),
            z_dist_bounds=(l_min*turn_radius_max, l_max*turn_radius_max),
            path_types=path_types)

        # Replace field turn_rate with turn_radius and add vertical speed constraint
        for test_case in test_cases:
            test_case[
                "turn_radius"] = test_case["airspeed"]/test_case["turn_rate"]
            del test_case["turn_rate"]
            test_case["turn_n_max"] = load_factor_max
            test_case["vert_speed_constraint"] = lambda w_A_g: abs(
                w_A_g) < vert_speed_max

        rt_traces, value_traces, table = test_tools.collect_traces(
            test_cases, get_dubins_path_3D,
            "Collect traces for CSC 3D Dubins Airplane paths - randomized test set with vertical speed constraint"
        )

        # Print traces
        print("=== TRACES SUMMARY ===")
        print(table)
        print("=== RT TRACES ===")
        print(rt_traces)
        print("=== VALUE TRACES ===")
        print(value_traces)

        # Check if reported iterations equals number of recorded values
        # for each successful iteration span
        n_residuals = value_traces[value_traces["variable"] == "w_A_g_residual"] \
            .groupby("span_id").size()
        converged = rt_traces[rt_traces["success"]]
        reported = converged.set_index("span_id")["iterations"]
        recorded = n_residuals.reindex(reported.index).fillna(0).astype(int)
        self.assertTrue(
            reported.astype(int).equals(recorded),
            msg="Recorded residual samples do not match reported iterations:\n"
            f"{reported[reported.astype(int) != recorded].to_string()}")

        if DEBUG_PLOTS:
            # Plot residual of w_A_g for each span_id
            merged = value_traces.merge(table, on="trace_id", how="left")
            filtered_by_span = merged[merged["span"] == "dubins_csc3D"]
            filtered_by_variable = filtered_by_span[filtered_by_span["variable"]
                                                    == "w_A_g_residual"]

            # NOTE: One run solves this fixed point once per start/end round that
            # the vertical-speed constraint forces onto it, and every solve
            # restarts "step" at zero -- so the overlay must be grouped by
            # "span_id" (one curve per solve), not by "trace_id" (one curve per
            # run). plot_iteration_values() groups by "span_id" by default and
            # shifts the abscissa to start at iteration 1, since htrace records
            # one residual per *update* (index 0 is already after the first one).
            fig, ax = plotting.plot_iteration_values(
                filtered_by_variable,
                title="w_A_g fixed-point residual",
                ylabel=plotting.get_label("w_A_g_residual"))
            plotting.show()

            fig, ax = plotting.plot_iteration_histogram(
                rt_traces,
                span="dubins_csc3D",
                iterations_counter="iterations",
                title="w_A_g iterations")
            plotting.show()

    def test_csc3d_constrained_vert_speed_beautiful_plot(self):
        """Test the 3D solver with 3D cases (i.e. goal altitude != start altitude)."""
        # Start with testing the 3D solver with horizontal paths
        turn_radius_m = 100
        xy_dist = turn_radius_m*4
        airspeed_mps = 20
        wind_speed = 10
        # wind_speed = 0
        wind_dir = math.pi*0.25
        # turn_rate_1_radps = airspeed_mps/turn_radius_m
        # turn_rate_2_radps = airspeed_mps/turn_radius_m
        # Create a start and a set of goal points
        start = Sample3D(0, 200, 0, 0)
        goal = Sample3D(200, 300, -100, math.pi)
        path_type = "LSR"
        vs_constraint = lambda w_A_g: abs(w_A_g) < 2

        path = get_dubins_path_3D(
            start_x=start.x_g,
            start_y=start.y_g,
            start_z=start.z_g,
            start_dir=start.chi,
            goal_x=goal.x_g,
            goal_y=goal.y_g,
            goal_z=goal.z_g,
            goal_dir=goal.chi,
            airspeed=airspeed_mps,
            turn_radius=turn_radius_m,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            path_type=path_type,
            vert_speed_constraint=vs_constraint)
        self.assertIsNotNone(
            path, msg=f"Failed to plan {path_type} path from {start} to {goal}")
        path_discr = path.discretize()
        path_discr.check_kinematic_consistency(1, 1, 0, 1, raise_err=True)
        # Check initial and final state
        self.assertAlmostEqual(path_discr.x0["x_g"], start.x_g)
        self.assertAlmostEqual(path_discr.xf["x_g"], goal.x_g, places=3)
        self.assertAlmostEqual(path_discr.x0["y_g"], start.y_g)
        self.assertAlmostEqual(path_discr.xf["y_g"], goal.y_g, places=3)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.x0["chi"], start.chi)),
            0.0,
            places=3)
        self.assertAlmostEqual(
            abs(utils.angular_diff(path_discr.xf["chi"], goal.chi)),
            0.0,
            places=3)
        # Plot the results
        plotting.configure(mode="latex", pageformat="IEEE_single_column")
        POIs = [
            dict(
                x_g=start.x_g,
                y_g=start.y_g,
                z_g=start.z_g,
                # chi=start.chi,
                label="start",
                # style="go",
                s=10,
                marker=(3, 0, math.degrees(start.chi)),
                color="tab:cyan",
            ),
            dict(
                x_g=goal.x_g,
                y_g=goal.y_g,
                z_g=goal.z_g,
                # chi=goal.chi,
                label="goal",
                # style="gx",
                s=10,
                marker=(3, 0, math.degrees(goal.chi)),
                color="tab:pink",
            ),
        ]

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
        ac.fly_trajectory(path_discr)
        fig, ax = plotting.plot_ref_trajectory(
            paths=path_discr.df,
            POIs=POIs,
            s_col=None,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            timelines=["n", "T"])

        # Fine-tune and save
        fig.canvas.draw()
        fig.savefig("dubins_path_csc3d_constrained_vert_speed.pdf")
        subprocess.run(
            [
                "pdfcrop", "--margins", "0 0 0 0",
                "dubins_path_csc3d_constrained_vert_speed.pdf",
                "dubins_path_csc3d_constrained_vert_speed.cropped.pdf"
            ],
            check=True)
        # NOTE: trim margins with pdfcrop:
        #       pdfcrop --margins '1 1 1 1' dubins_path_csc3d_constrained_vert_speed.pdf out_trimmed.pdfs
        # timelines=["V_A", "w_A_g"])
        # plotting.plot_timeline(
        #     trajectories=path_discr.df,
        #     subplots=[
        #         #"V_A", "V_A_dot", "z_g", "w_A_g", "w_A_g_dot", "chi_A_dot"
        #         # "V_A",
        #         # "V_A_xy",
        #         # "w_A_g",
        #         # "gamma_A",
        #         "T",
        #         "n"
        #     ],
        #     columns=1)
        # plotting.show()
        # plotting.show(block=True)

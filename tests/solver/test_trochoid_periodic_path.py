# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.trochoid_path.

Run from top-level package directory:

   python3 -m unittest tests.solver.test_trochoid_periodic_path

"""

import math
import random
import unittest

from tqdm import tqdm

from eduplane.solver.trochoid_periodic_path import *
from eduplane.utils import logger

from .test_tools import Sample2D as Sample

DEBUG_OUTPUT = False
DEBUG_PLOTS = False
# DEBUG_PLOTS = True


def debug_plots(samples, discr_paths, turn_radius_m):
    """Plot results from test case for debugging purposes."""
    # Plot the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    # Plot all samples
    for sample in samples:
        if sample.chi is None:
            ax.scatter([sample.y_g], [sample.x_g], marker='x')
        else:
            arrow = [sample.y_g, sample.x_g] + [
                turn_radius_m/10*math.sin(sample.chi),
                turn_radius_m/10*math.cos(sample.chi)
            ]
            ax.arrow(
                *arrow,
                fc="r",
                ec="r",
                head_width=turn_radius_m/10,
                head_length=turn_radius_m/10)
    # Plot all paths
    for path_discr in discr_paths:
        ax.plot(path_discr.df["y_g"], path_discr.df["x_g"], linewidth=0.5)
    # ax.set_xlim((-(k + 2)*turn_radius_m, (k+2)*turn_radius_m))
    # ax.set_ylim((-(k + 2)*turn_radius_m, (k+2)*turn_radius_m))
    ax.autoscale()
    ax.set_aspect("equal")
    plt.show()


class TrochoidPeriodicPathTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)
        logger.reset()

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_full_turn_basic(self):
        turn_radius_m = 100
        airspeed_mps = 15
        # wind_speed_vec = Vec2D(5,0)
        # wind_speed_vec = Vec2D(0,0)
        # wind_speed_vec = (4, -2)
        wind_speed = 5
        # wind_speed = 0
        wind_dir = -math.pi*0.25
        turn_rate_radps = -airspeed_mps/turn_radius_m
        turn_radius_max = (airspeed_mps+wind_speed)/turn_rate_radps
        # Create a start and a set of goal points
        # k_values = [0, 1, 1.9, 2.0, 2.1, 3, 3.9, 4, 4.1, 5, 6, 7, 8]

        path_types = ["LSL", "RSR"]

        def get_samples(k=1):
            return [
                Sample(k*turn_radius_m, 0, 0),
                Sample(k*turn_radius_m, 0, math.radians(90)),
                Sample(k*turn_radius_m, 0, math.radians(180)),
                Sample(k*turn_radius_m, 0, math.radians(270)),
                # Sample(-k*turn_radius_m, 0, 0),
                # Sample(-k*turn_radius_m, 0, math.radians(90)),
                # Sample(-k*turn_radius_m, 0, math.radians(180)),
                # Sample(-k*turn_radius_m, 0, math.radians(270)),
                # Sample(0, -k*turn_radius_m, 0),
                # Sample(0, -k*turn_radius_m, math.radians(90)),
                # Sample(0, -k*turn_radius_m, math.radians(180)),
                # Sample(0, -k*turn_radius_m, math.radians(270)),
                # Sample(0, k*turn_radius_m, 0),
                # Sample(0, k*turn_radius_m, math.radians(90)),
                # Sample(0, k*turn_radius_m, math.radians(180)),
                # Sample(0, k*turn_radius_m, math.radians(270)),
            ]

        samples = get_samples()

        discr_paths = []
        with tqdm(total=len(samples)*len(path_types),
                  desc="Testing full turn periodic paths - basic") as pbar:
            for sample in samples:
                for path_type in path_types:
                    pbar.update()
                    path = get_trochoid_periodic_path(
                        sample.x_g, sample.y_g, sample.chi, airspeed_mps,
                        airspeed_mps, turn_rate_radps, wind_speed, wind_dir)

                    self.assertIsNotNone(
                        path,
                        msg=
                        f"Failed to plan periodic {path_type} path from {sample}"
                    )

                    path_discr = path.discretize()
                    path_discr.check_kinematic_consistency(raise_err=True)
                    # Check initial and final state
                    self.assertAlmostEqual(path_discr.x0["x_g"], sample.x_g)
                    self.assertAlmostEqual(
                        path_discr.xf["x_g"], sample.x_g, places=3)
                    self.assertAlmostEqual(path_discr.x0["y_g"], sample.y_g)
                    self.assertAlmostEqual(
                        path_discr.xf["y_g"], sample.y_g, places=3)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.x0["chi"], sample.chi)),
                        0.0,
                        places=3)
                    self.assertAlmostEqual(
                        abs(
                            utils.angular_diff(
                                path_discr.xf["chi"], sample.chi)),
                        0.0,
                        places=3)
                    discr_paths.append(path_discr)

        # Plot the results
        if DEBUG_PLOTS:
            debug_plots(samples, discr_paths, turn_radius_m)

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.horizontal_profiles.steady_turn.

Run from top-level package directory:

   python3 -m unittest tests.traj.horizontal_profiles.test_steady_turn

"""

import math
import unittest

import numpy

from eduplane.traj.horizontal_profiles.steady_turn import *
from eduplane.utils import utils


class SteadyTurnTest(unittest.TestCase):

    def test_l_xy(self):
        # Case 1: 2-minute turn, no wind
        seg = SteadyTurn.init_from(
            x_g_t0=0,
            y_g_t0=0,
            tf=120,
            V_A_xy=20,
            chi_A_t0=0,
            chi_A_dot=numpy.pi/60,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(seg.evaluate(0)["l_xy"], 0)
        self.assertAlmostEqual(
            seg.evaluate(30)["l_xy"], 0.5*math.pi*seg.V_A_xy/seg.chi_A_dot)
        self.assertAlmostEqual(
            seg.evaluate(60)["l_xy"], math.pi*seg.V_A_xy/seg.chi_A_dot)
        self.assertAlmostEqual(
            seg.evaluate(90)["l_xy"], 1.5*math.pi*seg.V_A_xy/seg.chi_A_dot)
        self.assertAlmostEqual(
            seg.evaluate(120)["l_xy"], 2*math.pi*seg.V_A_xy/seg.chi_A_dot)
        # Case 1: 2-minute turn with wind
        chi_W = math.pi/2
        seg = SteadyTurn.init_from(
            x_g_t0=0,
            y_g_t0=0,
            tf=240,
            V_A_xy=20,
            chi_A_t0=vt.chi_A_from_V_A_xy_chi(
                V_A_xy=20, chi=math.pi, V_W_xy=10, chi_W=chi_W),
            chi_A_dot=numpy.pi/60,
            V_W_xy=10,
            chi_W=chi_W)
        df = seg.discretize().df
        df["d_x_g"] = df["x_g"].diff().fillna(0)
        df["d_y_g"] = df["y_g"].diff().fillna(0)
        df["l_xy_num"] = df.apply(
            lambda row: numpy.sqrt(row.d_x_g**2 + row.d_y_g**2),
            axis=1).cumsum()
        df["l_xy_error"] = (df["l_xy"] - df["l_xy_num"]).abs()
        self.assertLessEqual(df["l_xy_error"].max(), 0.1)

    def test_segment_interface(self):

        # Implements a fully parametric horizontal trajectory segment
        seg = SteadyTurn(
            x_g_t0=0,
            y_g_t0=0,
            tf=60,
            V_A_xy=20,
            chi_t0=0,
            chi_A_dot=numpy.pi/60,
            V_W_xy=0,
            chi_W=0)
        values = seg.evaluate(0)
        self.assertTupleEqual(tuple(values.keys()), HorizProfile.states)
        for k, v in values.items():
            self.assertIsNotNone(v, msg=f"Value of {k} is None")
        for k, v in values.items():
            self.assertFalse(math.isnan(v), msg=f"Value of {k} is NaN")
        df = seg.discretize().df
        self.assertDictEqual(df.iloc[0].to_dict(), seg.x0)
        self.assertDictEqual(df.iloc[-1].to_dict(), seg.xf)
        # Initial time and states can be shifted
        seg.shift(10, x_g=10, y_g=20)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 70)
        self.assertAlmostEqual(seg.x0["x_g"], 10)
        self.assertAlmostEqual(seg.x0["y_g"], 20)

    def test_init(self):
        # Initialize from V_A_xy, chi_A_dot and chi_t0 (default)
        seg1 = SteadyTurn(
            x_g_t0=0,
            y_g_t0=0,
            tf=60,
            V_A_xy=20,
            chi_t0=0,
            chi_A_dot=numpy.pi/60,
            V_W_xy=5,
            chi_W=0)
        # Initialize from V_A_xy, chi_A_dot and chi_A_t0
        seg2 = SteadyTurn.init_from(
            x_g_t0=0,
            y_g_t0=0,
            tf=60,
            V_A_xy=20,
            chi_A_t0=0,
            chi_A_dot=numpy.pi/60,
            V_W_xy=5,
            chi_W=0)
        # Initialize from V_K_xy_t0, chi_A_dot and chi_t0
        seg3 = SteadyTurn.init_from(
            x_g_t0=0,
            y_g_t0=0,
            tf=60,
            V_K_xy_t0=25,
            chi_t0=0,
            chi_A_dot=numpy.pi/60,
            V_W_xy=5,
            chi_W=0)
        # Initialize from V_K_xy_t0, chi_dot_t0 and chi_t0
        seg3 = SteadyTurn.init_from(
            x_g_t0=0,
            y_g_t0=0,
            tf=60,
            V_A_xy=20,
            chi_t0=0,
            chi_dot_t0=numpy.pi/60*(20/25),
            V_W_xy=5,
            chi_W=0)
        # Initialize from x_g_tf, y_g_tf, chi_A_tf
        seg4 = SteadyTurn.init_from(
            x_g_tf=seg1.xf["x_g"],
            y_g_tf=seg1.xf["y_g"],
            tf=60,
            V_A_xy=20,
            chi_A_tf=seg1.xf["chi_A"],
            chi_A_dot=numpy.pi/60,
            V_W_xy=5,
            chi_W=0)

        # # Pretty-print seg1, seg2, seg3
        # print("seg1:")
        # for k, v in seg1.items():
        #     print(f"  {k}: {v}")
        # print("seg2:")
        # for k, v in seg2.items():
        #     print(f"  {k}: {v}")
        # print("seg3:")
        # for k, v in seg3.items():
        #     print(f"  {k}: {v}")
        # print("seg4:")
        # for k, v in seg4.items():
        #     print(f"  {k}: {v}")

        def almostEqualIncludingAngles(v1, v2, msg=""):
            try:
                self.assertAlmostEqual(v1, v2)
            except AssertionError:
                angle_diff = utils.angular_diff(v1, v2)
                self.assertAlmostEqual(
                    angle_diff,
                    0,
                    msg=msg + f" (angular diff.: {math.degrees(angle_diff)}°)")

        for (k1, v1), (k2, v2) in zip(seg1.items(), seg2.items()):
            almostEqualIncludingAngles(v1, v2, msg=f"fields: {k1}, {k2}")
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg3.items()):
            almostEqualIncludingAngles(v1, v2, msg=f"fields: {k1}, {k2}")
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg4.items()):
            almostEqualIncludingAngles(v1, v2, msg=f"fields: {k1}, {k2}")

        # Examples with real-world numbers
        seg1_rw = SteadyTurn(
            x_g_t0=10,
            y_g_t0=20,
            tf=30,
            V_A_xy=20,
            chi_t0=math.radians(45),
            chi_A_dot=numpy.pi/60,
            V_W_xy=5,
            chi_W=math.radians(30))
        # Initialize same turn from x_g_tf, y_g_tf, chi_A_tf
        seg2_rw = SteadyTurn.init_from(
            x_g_tf=seg1_rw.x_g_tf,
            y_g_tf=seg1_rw.y_g_tf,
            tf=seg1_rw.tf,
            V_A_xy=seg1_rw.V_A_xy,
            chi_tf=seg1_rw.chi_tf,
            chi_A_dot=seg1_rw.chi_A_dot,
            V_W_xy=seg1_rw.V_W_xy,
            chi_W=seg1_rw.chi_W)
        for (k1, v1), (k2, v2) in zip(seg1_rw.items(), seg2_rw.items()):
            almostEqualIncludingAngles(v1, v2, msg=f"fields: {k1}, {k2}")

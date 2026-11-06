# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.model.kinematic_model.

Run from top-level package directory:

   python3 -m unittest tests.model.test_kinematic_model

"""

import math
import random
import unittest

import numpy
import sympy

from eduplane.model import velocity_triangle as vt
from eduplane.model.kinematic_model import *
from eduplane.utils import smath
from eduplane.utils.smath import is_zero

# Symbol objects (named), shared by the symbolic equivalence tests.
_SYMBOLS = smath.create_symbols(symbols, "name")


class KinematicModelTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_chi_A_dot_from_chi_dot_V_A_xy_dot(self):
        # TODO: The function should be defined piecewise for V_W_xy > 0 and
        #       V_W_xy = 0, however, sympy.piecewise is useless with the numpy
        #       backend (i.e. it doesn't avoid any ZeroDivisionErrors).
        #       Once the following issues are solved, we could add a piecewise
        #       implementation and test it like below:
        #         - https://github.com/sympy/sympy/issues/15014
        #         - https://github.com/sympy/sympy/issues/14789
        # # V_W_xy = 0
        # chi_A_dot = chi_A_dot_from_chi_dot_V_A_xy_dot(
        #     V_A_xy=10, V_A_xy_dot=1, chi=0, chi_dot=0, V_W_xy=0, chi_W=0)
        # self.assertAlmostEqual(chi_A_dot, 0)
        # V_W_xy != 0 and chi = chi_W
        chi_A_dot = chi_A_dot_from_chi_dot_V_A_xy_dot(
            V_A_xy=10, V_A_xy_dot=1, chi=0, chi_dot=0, V_W_xy=5, chi_W=0)
        self.assertAlmostEqual(chi_A_dot, 0)
        # V_W_xy != 0 and chi != chi_W
        chi_A_dot = chi_A_dot_from_chi_dot_V_A_xy_dot(
            V_A_xy=10,
            V_A_xy_dot=1,
            chi=0,
            chi_dot=0,
            V_W_xy=5,
            chi_W=math.pi/2)
        self.assertNotEqual(chi_A_dot, 0)

    def test_horizontal_projection_simple(self):
        return
        # Simple case:
        # 45° heading, tail-wind, accelerated straight flight.
        V_A_xy = 20
        V_A_xy_dot = 5
        chi_A = 0.25*math.pi
        chi_A_dot = 0
        V_K_xy = 30
        chi = chi_A
        V_W_xy = 10
        chi_W = chi_A

        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(V_K_xy_dot, V_A_xy_dot)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(chi_dot, 0)
        self.assertAlmostEqual(chi_dot_alt, 0)
        chi_D_dot = chi_D_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        chi_D_dot_alt = chi_D_dot_from_chi_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(chi_D_dot, 0)
        self.assertAlmostEqual(chi_D_dot_alt, 0)
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)

        # Simple case:
        # 45° heading, tail-wind, turning flight.
        # Expectation: At this extremum (i.e. completely collapsed velocity
        # triangle with chi=chi_A and chi_D=0), airspeed rate and acceleration
        # as well as path and air-path azimuth rates are identical
        V_A_xy_dot = 0
        chi_A_dot = math.pi/60
        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(V_K_xy_dot, V_A_xy_dot)
        chi_D_dot = chi_D_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        chi_D_dot_alt = chi_D_dot_from_chi_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(chi_D_dot, chi_D_dot_alt)
        # self.assertAlmostEqual(chi_D_dot, chi_A_dot)
        # self.assertAlmostEqual(chi_D_dot_alt, chi_A_dot)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        self.assertAlmostEqual(chi_dot, chi_dot_alt)
        # self.assertAlmostEqual(chi_dot, chi_A_dot)
        # self.assertAlmostEqual(chi_dot_alt, chi_A_dot)
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)

        # Simple case:
        # 45° heading, cross-wind at 135°, accelerated straight flight.
        # Expectation: Although no turn, the path azimuth rate will be
        # negative. As the velocity increases with airspeed, chi will
        # converge towards chi_A. The acceleration will be less than the
        # airspeed rate but with same sign.
        V_A_xy = 20
        V_A_xy_dot = 5
        chi_A = 0.25*math.pi
        chi_A_dot = 0
        V_W_xy = 20
        chi_W = chi_A + 0.5*math.pi
        V_K_xy = math.sqrt(V_A_xy**2 + V_W_xy**2)
        chi = chi_A + 0.25*math.pi
        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(V_K_xy_dot, V_A_xy_dot)
        self.assertGreater(V_K_xy_dot, 0)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(chi_dot, 0)
        self.assertLess(chi_dot_alt, 0)
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)
        # When decreasing airspeed in this situation, path azimuth will be
        # positive and acceleration will be greater than airspeed rate.
        V_A_xy_dot = -5
        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertGreater(V_K_xy_dot, V_A_xy_dot)
        self.assertLess(V_K_xy_dot, 0)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        self.assertGreater(chi_dot, 0)
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertGreater(chi_dot_alt, 0)
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)
        # Simple case:
        # 45° heading, cross-wind at 135°, turning into wind.
        # Expectation: Although zero airspeed rate, there will be a positive
        # acceleration due to picking up wind speed components (i.e. flattening
        # the triangle). The wind speed components 'dampens' the path azimuth
        # rate, i.e it is less than the air-path azimuth rate.
        V_A_xy_dot = 0
        chi_A_dot = math.pi/30
        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertGreater(V_K_xy_dot, V_A_xy_dot)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(chi_dot, chi_A_dot)
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(chi_dot_alt, chi_A_dot)
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)
        # When turning against the wind, deceleration occurs, but still, the
        # path azimuth rate's magnitude is less than the air-path azimuth rate.
        chi_A_dot = -math.pi/30
        V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
            V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(V_K_xy_dot, V_A_xy_dot)
        chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
            V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(abs(chi_dot), abs(chi_A_dot))
        chi_dot_alt = chi_dot_from_chi_A_dot_V_K_xy_dot(
            V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
        self.assertLess(abs(chi_dot_alt), abs(chi_A_dot))
        # Check round-trip calculations.
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_K_xy_dot(
                V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            chi_A_dot_from_chi_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
        self.assertAlmostEqual(
            V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)

    def test_horizontal_projection_random(self):
        for i in range(1000):
            # Generate random horizontal velocity triangle with derivatives
            # for airspeed: V_A_xy_dot and chi_A_dot
            u_A_g = self.random(1, 50)
            v_A_g = self.random(1, 50)
            v_A_g = self.random(1, 50)
            # Note that equations used below don't currently work for
            # the zero-wind case (see docstrings)
            u_W_g = self.random(.1, u_A_g)
            v_W_g = self.random(.1, v_A_g)
            u_K_g = u_A_g + u_W_g
            v_K_g = v_A_g + v_W_g
            V_A_xy = math.sqrt(u_A_g**2 + v_A_g**2)
            V_A_xy_dot = self.random(-10, 10)
            V_K_xy = math.sqrt(u_K_g**2 + v_K_g**2)
            V_W_xy = math.sqrt(u_W_g**2 + v_W_g**2)
            chi_A = math.atan2(v_A_g, u_A_g)
            chi_A_dot = self.random(-math.pi/60, math.pi/60)
            chi = math.atan2(v_K_g, u_K_g)
            chi_W = math.atan2(v_W_g, u_W_g)
            chi_D = chi_A - chi
            # print(
            #     f"V_A_xy = ({u_A_g}, {v_A_g}) = {V_A_xy} @ {math.degrees(chi_A)}"
            # )
            # print(
            #     f"V_K_xy = ({u_K_g}, {v_K_g}) = {V_K_xy} @ {math.degrees(chi)}"
            # )
            # print(
            #     f"V_W_xy = ({u_W_g}, {v_W_g}) = {V_W_xy} @ {math.degrees(chi_W)}"
            # )
            # print(
            #     f"V_A_xy_dot = {V_A_xy_dot}, chi_A_dot =  {math.degrees(chi_A_dot)}"
            # )

            # We don't have a ground truth for other derivatives, but we can
            # test round-trip calculations and consistency of alternate
            # equations.
            V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
                V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
            chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)

            # chi_D_dot
            chi_D_dot = chi_A_dot - chi_dot
            self.assertAlmostEqual(
                chi_D_dot_from_chi_A_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                chi_D_dot)
            self.assertAlmostEqual(
                chi_D_dot_from_chi_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_D_dot)

            # chi_dot
            self.assertAlmostEqual(
                chi_dot_from_chi_A_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                chi_dot)
            self.assertAlmostEqual(
                chi_dot_from_chi_A_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W), chi_dot)

            # chi_A_dot
            self.assertAlmostEqual(
                chi_A_dot_from_chi_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
            self.assertAlmostEqual(
                chi_A_dot_from_chi_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W),
                chi_A_dot)

            # V_A_xy_dot
            self.assertAlmostEqual(
                V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                    V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W),
                V_A_xy_dot)
            self.assertAlmostEqual(
                V_A_xy_dot_from_chi_A_dot_chi_dot(
                    V_A_xy, chi_A_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)

            # V_K_xy_dot
            self.assertAlmostEqual(
                V_K_xy_dot_from_chi_dot_chi_A_dot(
                    V_K_xy, chi_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                V_K_xy_dot)

    def test_3D_velocity_triangle_random(self):
        for i in range(1000):
            # Generate random velocity triangle
            u_A_g = self.random(1, 50)
            v_A_g = self.random(1, 50)
            w_A_g = self.random(1, 50)
            u_W_g = self.random(0, u_A_g)
            v_W_g = self.random(0, v_A_g)
            w_W_g = self.random(0, w_A_g)
            u_K_g = u_A_g + u_W_g
            v_K_g = v_A_g + v_W_g
            w_K_g = w_A_g + w_W_g
            V_A_xy = math.sqrt(u_A_g**2 + v_A_g**2)
            V_A = math.sqrt(V_A_xy**2 + w_A_g**2)
            V_K_xy = math.sqrt(u_K_g**2 + v_K_g**2)
            V_K = math.sqrt(V_K_xy**2 + w_K_g**2)
            V_W_xy = math.sqrt(u_W_g**2 + v_W_g**2)
            V_W = math.sqrt(V_W_xy**2 + w_W_g**2)
            chi_A = math.atan2(v_A_g, u_A_g)
            chi = math.atan2(v_K_g, u_K_g)
            chi_W = math.atan2(v_W_g, u_W_g)
            chi_D = chi_A - chi

            gamma_A = vt.gamma_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g)
            gamma = vt.gamma_from_V_K_xy_w_K_g(V_K_xy, w_K_g)
            gamma_W = vt.gamma_W_from_V_W_xy_w_W_g(
                V_W_xy, w_W_g) if V_W_xy > 0 else 0

            V_A_dot = self.random(-10, 10)
            chi_A_dot = self.random(-math.pi/60, math.pi/60)
            gamma_A_dot = self.random(-math.pi/30, math.pi/30)

            # print(
            #     f"V_A_xy = ({u_A_g}, {v_A_g}) = {V_A_xy} @ {math.degrees(chi_A)}"
            # )
            # print(
            #     f"V_K_xy = ({u_K_g}, {v_K_g}) = {V_K_xy} @ {math.degrees(chi)}"
            # )
            # print(
            #     f"V_W_xy = ({u_W_g}, {v_W_g}) = {V_W_xy} @ {math.degrees(chi_W)}"
            # )
            # print(
            #     f"V_A_xy_dot = {V_A_xy_dot}, chi_A_dot =  {math.degrees(chi_A_dot)}"
            # )

            # We don't have a ground truth for other derivatives, but we can
            # test round-trip calculations and consistency of alternate
            # equations.
            V_A_xy_dot = V_A_xy_dot_from_V_A_dot_gamma_A_dot(
                V_A, V_A_dot, gamma_A, gamma_A_dot)
            V_K_xy_dot = V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
                V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W)
            chi_dot = chi_dot_from_chi_A_dot_V_A_xy_dot(
                V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W)
            w_A_g_dot = w_A_g_dot_from_V_A_dot_V_A_xy_dot(
                V_A, V_A_dot, V_A_xy, V_A_xy_dot)
            w_K_g_dot = w_A_g_dot
            gamma_dot = gamma_dot_from_V_K_xy_dot_w_K_g_dot(
                V_K, V_K_xy_dot, w_K_g_dot, gamma)
            V_K_dot = V_K_dot_from_V_K_xy_dot_w_K_g_dot(
                V_K_xy, V_K_xy_dot, w_K_g, w_K_g_dot)

            # chi_D_dot
            chi_D_dot = chi_A_dot - chi_dot
            self.assertAlmostEqual(
                chi_D_dot_from_chi_A_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                chi_D_dot)
            self.assertAlmostEqual(
                chi_D_dot_from_chi_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_D_dot)

            # chi_dot
            self.assertAlmostEqual(
                chi_dot_from_chi_A_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                chi_dot)
            self.assertAlmostEqual(
                chi_dot_from_chi_A_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W), chi_dot)

            # chi_A_dot
            self.assertAlmostEqual(
                chi_A_dot_from_chi_dot_V_A_xy_dot(
                    V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W), chi_A_dot)
            self.assertAlmostEqual(
                chi_A_dot_from_chi_dot_V_K_xy_dot(
                    V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W),
                chi_A_dot)

            # V_A_xy_dot
            self.assertAlmostEqual(
                V_A_xy_dot_from_V_A_dot_gamma_A_dot(
                    V_A, V_A_dot, gamma_A, gamma_A_dot), V_A_xy_dot)
            self.assertAlmostEqual(
                V_A_xy_dot_from_V_A_dot_w_A_g_dot(
                    V_A, V_A_dot, w_A_g, w_A_g_dot), V_A_xy_dot)
            self.assertAlmostEqual(
                V_A_xy_dot_from_V_K_xy_dot_chi_dot(
                    V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W),
                V_A_xy_dot)
            self.assertAlmostEqual(
                V_A_xy_dot_from_chi_A_dot_chi_dot(
                    V_A_xy, chi_A_dot, chi, chi_dot, V_W_xy, chi_W), V_A_xy_dot)
            self.assertAlmostEqual(
                V_A_xy_dot_from_w_A_g_dot_gamma_A_dot(
                    V_A, w_A_g_dot, gamma_A, gamma_A_dot), V_A_xy_dot)

            # V_A_dot
            self.assertAlmostEqual(
                V_A_dot_from_V_A_xy_dot_w_A_g_dot(
                    V_A_xy, V_A_xy_dot, w_A_g, w_A_g_dot), V_A_dot)
            self.assertAlmostEqual(
                V_A_dot_from_V_K_dot(
                    V_K, V_K_dot, chi_A, chi_A_dot, gamma_A, gamma_A_dot, V_W,
                    chi_W, gamma_W), V_A_dot)

            # w_A_g_dot
            self.assertAlmostEqual(
                w_A_g_dot_from_gamma_A_dot(V_A, V_A_dot, gamma_A, gamma_A_dot),
                w_A_g_dot)
            self.assertAlmostEqual(
                w_A_g_dot_from_V_A_dot_V_A_xy_dot(
                    V_A, V_A_dot, V_A_xy, V_A_xy_dot), w_A_g_dot)

            # V_K_dot
            self.assertAlmostEqual(
                V_K_dot_from_V_K_xy_dot_w_K_g_dot(
                    V_K_xy, V_K_xy_dot, w_K_g, w_K_g_dot), V_K_dot)
            self.assertAlmostEqual(
                V_K_dot_from_V_A_dot(
                    V_A, V_A_dot, chi, chi_dot, gamma, gamma_dot, V_W, chi_W,
                    gamma_W), V_K_dot)

            # V_K_xy_dot
            self.assertAlmostEqual(
                V_K_xy_dot_from_V_K_dot_w_K_g_dot(
                    V_K, V_K_dot, w_K_g, w_K_g_dot), V_K_xy_dot)
            self.assertAlmostEqual(
                V_K_xy_dot_from_V_K_dot_gamma_dot(
                    V_K, V_K_dot, gamma, gamma_dot), V_K_xy_dot)
            self.assertAlmostEqual(
                V_K_xy_dot_from_w_K_g_dot_gamma_dot(
                    V_K, w_K_g_dot, gamma, gamma_dot), V_K_xy_dot)
            self.assertAlmostEqual(
                V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
                    V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                V_K_xy_dot)
            self.assertAlmostEqual(
                V_K_xy_dot_from_chi_dot_chi_A_dot(
                    V_K_xy, chi_dot, chi_A, chi_A_dot, V_W_xy, chi_W),
                V_K_xy_dot)

            # w_K_g_dot
            self.assertAlmostEqual(
                w_K_g_dot_from_V_K_dot_gamma_dot(
                    V_K, V_K_dot, gamma, gamma_dot), w_K_g_dot)
            self.assertAlmostEqual(
                w_K_g_dot_from_V_K_dot_V_K_xy_dot(
                    V_K, V_K_dot, V_K_xy, V_K_xy_dot), w_K_g_dot)
            self.assertAlmostEqual(
                w_K_g_dot_from_V_K_xy_dot_gamma_dot(
                    V_K, V_K_xy_dot, gamma, gamma_dot), w_K_g_dot)

            # gamma_dot
            self.assertAlmostEqual(
                gamma_dot_from_V_K_xy_dot_w_K_g_dot(
                    V_K, V_K_xy_dot, w_K_g_dot, gamma), gamma_dot)
            self.assertAlmostEqual(
                gamma_dot_from_V_K_dot_V_K_xy_dot(
                    V_K, V_K_dot, V_K_xy_dot, gamma), gamma_dot)
            self.assertAlmostEqual(
                gamma_dot_from_V_K_dot_w_K_g_dot(
                    V_K, V_K_dot, w_K_g_dot, gamma), gamma_dot)

            # gamma_A_dot
            self.assertAlmostEqual(
                gamma_A_dot_from_V_A_dot_w_A_g_dot(
                    V_A, V_A_dot, w_A_g_dot, gamma_A), gamma_A_dot)
            self.assertAlmostEqual(
                gamma_A_dot_from_V_A_xy_dot_w_A_g_dot(
                    V_A, V_A_xy_dot, w_A_g_dot, gamma_A), gamma_A_dot)
            self.assertAlmostEqual(
                gamma_A_dot_from_V_K_dot_V_A_dot_gamma_dot(
                    V_K, V_K_dot, gamma, gamma_dot, V_A, V_A_dot, V_W, gamma_W),
                gamma_A_dot)

    def test_check_dimensions(self):
        """Symbolic dimensions check.
        """
        checked = [
            # Equations of motion
            u_A_g_from_V_A,
            v_A_g_from_V_A,
            w_A_g_from_V_A,
            u_A_g_dot_from_V_A_dot,
            v_A_g_dot_from_V_A_dot,
            w_A_g_dot_from_gamma_A_dot,
            # Horizontal velocity triangle derivative equations
            chi_D_dot_from_chi_A_dot_V_K_xy_dot,
            chi_D_dot_from_chi_dot_V_A_xy_dot,
            chi_A_dot_from_chi_dot_V_A_xy_dot,
            chi_A_dot_from_chi_dot_V_K_xy_dot,
            chi_dot_from_chi_A_dot_V_K_xy_dot,
            chi_dot_from_chi_A_dot_V_A_xy_dot,
            V_A_xy_dot_from_V_K_xy_dot_chi_dot,
            V_A_xy_dot_from_chi_A_dot_chi_dot,
            V_K_xy_dot_from_V_A_xy_dot_chi_A_dot,
            V_K_xy_dot_from_chi_dot_chi_A_dot,
            V_K_dot_from_V_A_dot,
            # Airspeed rate and components
            V_A_dot_from_V_A_xy_dot_w_A_g_dot,
            V_A_xy_dot_from_V_A_dot_w_A_g_dot,
            w_A_g_dot_from_V_A_dot_V_A_xy_dot,
            V_A_xy_dot_from_w_A_g_dot_gamma_A_dot,
            w_A_g_dot_from_V_A_xy_dot_gamma_A_dot,
        ]
        for fn in checked:
            # ``check_dimensions`` probes each function with all arguments set
            # to 1, which lands outside the domain of some functions (e.g.
            # ``sqrt`` of a negative, division by zero). Those yield ``nan``
            # plus a RuntimeWarning, but the *dimensional* check still passes,
            # so we silence the spurious numpy warnings here.
            with numpy.errstate(invalid="ignore", divide="ignore"):
                self.assertTrue(
                    fn.check_dimensions(),
                    f"check_dimensions failed for {fn.__name__}")

    def test_chi_A_dot_from_chi_dot_V_K_xy_dot_equivalence(self):
        """``chi_A_dot = chi_D_dot + chi_dot`` derived symbolically.
        """
        syms = _SYMBOLS
        chi_A_dot = syms["chi_A_dot"]
        chi_dot = syms["chi_dot"]

        # chi_A_dot = chi_D_dot_from_chi_A_dot_V_K_xy_dot.rhs + chi_dot
        check = sympy.Eq(
            chi_A_dot, chi_D_dot_from_chi_A_dot_V_K_xy_dot.eq.rhs + chi_dot)
        check = sympy.Eq(chi_A_dot, sympy.solve(check, chi_A_dot)[0])
        self.assertTrue(
            is_zero(check.rhs - chi_A_dot_from_chi_dot_V_K_xy_dot.eq.rhs))

    def test_V_A_xy_dot_from_chi_A_dot_chi_dot_equivalence(self):
        """``V_A_xy_dot`` derived from ``chi_A_dot_from_chi_dot_V_A_xy_dot``.
        """
        syms = _SYMBOLS
        V_A_xy_dot = syms["V_A_xy_dot"]

        check = sympy.Eq(
            V_A_xy_dot,
            sympy.solve(chi_A_dot_from_chi_dot_V_A_xy_dot.eq, V_A_xy_dot)[0])
        self.assertTrue(
            is_zero(check.rhs - V_A_xy_dot_from_chi_A_dot_chi_dot.eq.rhs))

    def test_V_K_xy_dot_from_chi_dot_chi_A_dot_equivalence(self):
        """``V_K_xy_dot`` derived from ``chi_dot_from_chi_A_dot_V_K_xy_dot``.
        """
        syms = _SYMBOLS
        V_K_xy_dot = syms["V_K_xy_dot"]

        check = sympy.Eq(
            V_K_xy_dot,
            sympy.solve(chi_dot_from_chi_A_dot_V_K_xy_dot.eq, V_K_xy_dot)[0])
        self.assertTrue(
            is_zero(check.rhs - V_K_xy_dot_from_chi_dot_chi_A_dot.eq.rhs))

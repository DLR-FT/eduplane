# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.model.velocity_triangle.

Run from top-level package directory:

   python3 -m unittest tests.model.test_velocity_triangle

"""

import math
import random
import unittest

import sympy

from eduplane.model.velocity_triangle import *
from eduplane.utils import smath
from eduplane.utils.smath import is_zero, solve_equiv

# Symbol objects (named), shared by the symbolic equivalence tests.
_SYMBOLS = smath.create_symbols(symbols, "name")


class VelocityTriangleTests(unittest.TestCase):

    def setUp(self):
        random.seed(123456789)

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    def test_chi(self):
        pass

    def test_roundtrip_gamma_A_w_A_g(self):
        random.seed(1234567890)

        for i in range(1000):
            V_A = self.random(10, 40)
            V_A_dot = 0  # self.random(-5, 5)
            gamma_A = self.random(-math.pi/3, math.pi/3)
            gamma_A_dot = 0  # self.random(-math.pi/3, math.pi/3)
            g = 10
            mu_A = 0  # self.random(-math.pi/3, math.pi/3)

            mapping = {
                "V_A": V_A,
                "V_A_dot": V_A_dot,
                "gamma_A": gamma_A,
                "gamma_A_dot": gamma_A_dot,
                "g": g,
                "mu_A": mu_A,
            }
            mapping["V_A_xy"] = math.cos(mapping["gamma_A"])*mapping["V_A"]

            w_A_g1 = w_A_g_from_V_A_gamma_A.eval_if(**mapping)
            w_A_g2 = w_A_g_from_V_A_xy_gamma_A.eval_if(**mapping)
            self.assertAlmostEqual(w_A_g1, w_A_g2)
            mapping["w_A_g"] = w_A_g1

            gamma_A1 = gamma_A_from_V_A_w_A_g.eval_if(**mapping)
            gamma_A2 = gamma_A_from_V_A_xy_w_A_g.eval_if(**mapping)
            self.assertAlmostEqual(gamma_A1, mapping["gamma_A"])
            self.assertAlmostEqual(gamma_A2, mapping["gamma_A"])

    def test_gamma_W(self):
        # Note that gamma_W is positive for negative (upwards pointing)
        # vertical components of the wind velocity vector.
        self.assertAlmostEqual(
            gamma_W_from_V_W_w_W_g(V_W=10*math.sqrt(2), w_W_g=-10), math.pi/4)
        self.assertAlmostEqual(
            gamma_W_from_V_W_w_W_g(V_W=10, w_W_g=-10), math.pi/2)
        # Note that gamma_W is undefined for V_W=0
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(gamma_W_from_V_W_w_W_g(V_W=0, w_W_g=10), 0.0)

    def test_horizontal_projection_simple(self):
        # Simple case:
        # 45° heading, tail-wind
        V_A_xy = 20
        chi_A = 0.25*math.pi
        V_K_xy = 30
        chi = chi_A
        V_W_xy = 10
        chi_W = chi_A
        chi_D = 0
        self.assertAlmostEqual(
            V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), V_A_xy)
        self.assertAlmostEqual(
            V_A_xy_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), V_A_xy)
        # V_K_xy
        self.assertAlmostEqual(
            V_K_xy_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), V_K_xy)
        self.assertAlmostEqual(
            V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), V_K_xy)
        # V_W_xy
        self.assertAlmostEqual(
            V_W_xy_from_V_A_xy_V_K_xy(V_A_xy, chi_A, V_K_xy, chi), V_W_xy)
        # chi_A
        self.assertAlmostEqual(
            chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_A)
        self.assertAlmostEqual(
            chi_A_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), chi_A)
        # chi
        self.assertAlmostEqual(
            chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), chi)
        self.assertAlmostEqual(
            chi_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi)
        # chi_D
        self.assertAlmostEqual(
            chi_D_from_V_A_xy_V_K_xy(V_A_xy, V_K_xy, V_W_xy, chi_A, chi_W),
            chi_D)
        self.assertAlmostEqual(
            chi_D_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_D)
        self.assertAlmostEqual(
            chi_D_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi_D)
        # Simple case:
        # 45° heading, 135° cross-wind
        V_A_xy = 20
        chi_A = 0.25*math.pi
        V_W_xy = 20
        chi_W = chi_A + 0.5*math.pi
        V_K_xy = math.sqrt(V_A_xy**2 + V_W_xy**2)
        chi = chi_A + 0.25*math.pi
        chi_D = chi_A - chi
        self.assertAlmostEqual(
            V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), V_A_xy)
        self.assertAlmostEqual(
            V_A_xy_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), V_A_xy)
        # V_K_xy
        self.assertAlmostEqual(
            V_K_xy_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), V_K_xy)
        self.assertAlmostEqual(
            V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), V_K_xy)
        # V_W_xy
        self.assertAlmostEqual(
            V_W_xy_from_V_A_xy_V_K_xy(V_A_xy, chi_A, V_K_xy, chi), V_W_xy)
        # chi_A
        self.assertAlmostEqual(
            chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_A)
        self.assertAlmostEqual(
            chi_A_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), chi_A)
        # chi
        self.assertAlmostEqual(
            chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), chi)
        self.assertAlmostEqual(
            chi_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi)
        # chi_D
        self.assertAlmostEqual(
            chi_D_from_V_A_xy_V_K_xy(V_A_xy, V_K_xy, V_W_xy, chi_A, chi_W),
            chi_D)
        self.assertAlmostEqual(
            chi_D_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_D)
        self.assertAlmostEqual(
            chi_D_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi_D)

    def test_horizontal_projection_random(self):
        for i in range(1000):
            # Generate random horizontal velocity triangle
            u_A_g = self.random(1, 50)
            v_A_g = self.random(1, 50)
            u_W_g = self.random(0, u_A_g)
            v_W_g = self.random(0, v_A_g)
            u_K_g = u_A_g + u_W_g
            v_K_g = v_A_g + v_W_g
            V_A_xy = math.sqrt(u_A_g**2 + v_A_g**2)
            V_K_xy = math.sqrt(u_K_g**2 + v_K_g**2)
            V_W_xy = math.sqrt(u_W_g**2 + v_W_g**2)
            chi_A = math.atan2(v_A_g, u_A_g)
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
            # V_A_xy
            self.assertAlmostEqual(
                V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), V_A_xy)
            self.assertAlmostEqual(
                V_A_xy_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), V_A_xy)
            # V_K_xy
            self.assertAlmostEqual(
                V_K_xy_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), V_K_xy)
            self.assertAlmostEqual(
                V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), V_K_xy)
            # V_W_xy
            self.assertAlmostEqual(
                V_W_xy_from_V_A_xy_V_K_xy(V_A_xy, chi_A, V_K_xy, chi), V_W_xy)
            # chi_A
            self.assertAlmostEqual(
                chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_A)
            self.assertAlmostEqual(
                chi_A_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), chi_A)
            # chi
            self.assertAlmostEqual(
                chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), chi)
            self.assertAlmostEqual(
                chi_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi)
            # chi_D
            self.assertAlmostEqual(
                chi_D_from_V_A_xy_V_K_xy(V_A_xy, V_K_xy, V_W_xy, chi_A, chi_W),
                chi_D)
            self.assertAlmostEqual(
                chi_D_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_D)
            self.assertAlmostEqual(
                chi_D_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi_D)

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

            gamma_A = gamma_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g)
            gamma = gamma_from_V_K_xy_w_K_g(V_K_xy, w_K_g)
            gamma_W = gamma_W_from_V_W_xy_w_W_g(
                V_W_xy, w_W_g) if V_W_xy > 0 else 0

            # print(
            #     f"V_A_xy = ({u_A_g}, {v_A_g}) = {V_A_xy} @ {math.degrees(chi_A)}"
            # )
            # print(
            #     f"V_K_xy = ({u_K_g}, {v_K_g}) = {V_K_xy} @ {math.degrees(chi)}"
            # )
            # print(
            #     f"V_W_xy = ({u_W_g}, {v_W_g}) = {V_W_xy} @ {math.degrees(chi_W)}"
            # )
            # V_A_xy
            self.assertAlmostEqual(
                V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), V_A_xy)
            self.assertAlmostEqual(
                V_A_xy_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), V_A_xy)
            # V_K_xy
            self.assertAlmostEqual(
                V_K_xy_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), V_K_xy)
            self.assertAlmostEqual(
                V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), V_K_xy)
            # V_W_xy
            self.assertAlmostEqual(
                V_W_xy_from_V_A_xy_V_K_xy(V_A_xy, chi_A, V_K_xy, chi), V_W_xy)
            # chi_A
            self.assertAlmostEqual(
                chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_A)
            self.assertAlmostEqual(
                chi_A_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W), chi_A)
            # chi
            self.assertAlmostEqual(
                chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W), chi)
            self.assertAlmostEqual(
                chi_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi)
            # chi_D
            self.assertAlmostEqual(
                chi_D_from_V_A_xy_V_K_xy(V_A_xy, V_K_xy, V_W_xy, chi_A, chi_W),
                chi_D)
            self.assertAlmostEqual(
                chi_D_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W), chi_D)
            self.assertAlmostEqual(
                chi_D_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W), chi_D)
            # V_A
            self.assertAlmostEqual(
                V_A_from_V_A_xy_gamma_A(V_A_xy, gamma_A), V_A)
            self.assertAlmostEqual(V_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g), V_A)
            self.assertAlmostEqual(
                V_A_from_V_K_chi_A_gamma_A(
                    V_K, chi_A, gamma_A, V_W, chi_W, gamma_W), V_A)
            # V_K
            self.assertAlmostEqual(V_K_from_V_K_xy_gamma(V_K_xy, gamma), V_K)
            self.assertAlmostEqual(V_K_from_V_K_xy_w_K_g(V_K_xy, w_K_g), V_K)
            self.assertAlmostEqual(
                V_K_from_V_A_chi_gamma(V_A, chi, gamma, V_W, chi_W, gamma_W),
                V_K)
            # gamma
            self.assertAlmostEqual(
                gamma_from_V_K_V_A_gamma_A(V_K, V_A, gamma_A, w_W_g), gamma)
            self.assertAlmostEqual(gamma_from_V_K_w_K_g(V_K, w_K_g), gamma)
            self.assertAlmostEqual(
                gamma_from_V_K_xy_w_K_g(V_K_xy, w_K_g), gamma)
            # gamma_A
            self.assertAlmostEqual(
                gamma_A_from_V_A_V_K_gamma(V_A, V_K, gamma, w_W_g), gamma_A)
            self.assertAlmostEqual(gamma_A_from_V_A_w_A_g(V_A, w_A_g), gamma_A)
            self.assertAlmostEqual(
                gamma_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g), gamma_A)

    def test_check_dimensions(self):
        """Symbolic dimensions check.
        """
        checked = [
            # Horizontal speed / velocity magnitudes
            V_A_xy_from_V_K_xy_chi,
            V_K_xy_from_V_A_xy_chi_A,
            V_W_xy_from_V_A_xy_V_K_xy,
            V_A_xy_from_V_K_xy_chi_A,
            V_K_xy_from_V_A_xy_chi,
            # Flight path azimuth deviation (drift angle)
            chi_D_from_V_A_xy_chi,
            chi_D_from_V_K_xy_chi_A,
            chi_D_from_V_A_xy_V_K_xy,
            # Flight path and air-path azimuth angles
            chi_from_V_A_xy_chi_A,
            chi_from_V_K_xy_chi_A,
            chi_A_from_V_A_xy_chi,
            chi_A_from_V_K_xy_chi,
            # Vertical / horizontal components and air-path inclination angle
            gamma_A_from_V_A_xy_w_A_g,
            gamma_A_from_V_A_w_A_g,
            w_A_g_from_V_A_gamma_A,
            w_A_g_from_V_A_xy_gamma_A,
            V_A_xy_from_V_A_gamma_A,
            V_A_xy_from_w_A_g_gamma_A,
            V_A_from_V_A_xy_gamma_A,
        ]
        for fn in checked:
            self.assertTrue(
                fn.check_dimensions(),
                f"check_dimensions failed for {fn.__name__}")

    def test_gamma_A_chain_consistency(self):
        """Symbolic consistency of the gamma_A / w_A_g / V_A_xy equations.
        """
        syms = _SYMBOLS
        gamma_A = syms["gamma_A"]
        V_A = syms["V_A"]
        V_A_xy = syms["V_A_xy"]
        w_A_g = syms["w_A_g"]

        # gamma_A_from_V_A_w_A_g  ==  w_A_g_from_V_A_gamma_A
        #   gamma_A = asin(-w_A_g/V_A)  ->  w_A_g = -sin(gamma_A)*V_A
        self.assertTrue(
            solve_equiv(
                gamma_A_from_V_A_w_A_g.eq, w_A_g, w_A_g_from_V_A_gamma_A))

        # w_A_g_from_V_A_gamma_A  ==  w_A_g_from_V_A_xy_gamma_A
        #   -sin(gamma_A)*V_A == -tan(gamma_A)*V_A_xy  (V_A_xy = V_A*cos(gamma_A))
        lhs = w_A_g_from_V_A_gamma_A.eq.rhs
        rhs = w_A_g_from_V_A_xy_gamma_A.eq.rhs.subs(
            V_A_xy, V_A*sympy.cos(gamma_A))
        self.assertTrue(is_zero(lhs - rhs))

        # w_A_g_from_V_A_xy_gamma_A  ==  V_A_xy_from_w_A_g_gamma_A
        #   V_A_xy = -w_A_g/tan(gamma_A)
        self.assertTrue(
            solve_equiv(
                w_A_g_from_V_A_xy_gamma_A.eq, V_A_xy,
                V_A_xy_from_w_A_g_gamma_A))

        # V_A_xy_from_V_A_gamma_A  ==  V_A_xy_from_w_A_g_gamma_A
        #   cos(gamma_A)*V_A == -w_A_g/tan(gamma_A)  (w_A_g = -sin(gamma_A)*V_A)
        lhs = V_A_xy_from_V_A_gamma_A.eq.rhs
        rhs = V_A_xy_from_w_A_g_gamma_A.eq.rhs.subs(
            w_A_g, -sympy.sin(gamma_A)*V_A)
        self.assertTrue(is_zero(lhs - rhs))

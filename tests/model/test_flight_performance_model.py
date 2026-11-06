# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.model.flight_performance_model.

Run from top-level package directory:

   python3 -m unittest tests.model.test_flight_performance_model

"""

import math
import random
import unittest

import numpy
import sympy
from sympy import cos, sin, sqrt

from eduplane.model import kinematic_model as km
from eduplane.model.flight_performance_model import *
from eduplane.utils import smath
from eduplane.utils.smath import is_zero, positive_branch

# Symbol objects (named) and definition objects, shared by the symbolic tests.
_SYMBOLS = smath.create_symbols(symbols, "name")
_DEFINITIONS = smath.get_definitions(symbols)


class FlightPerformanceModelTest(unittest.TestCase):

    def random(self, min_value, max_value):
        return min_value + (max_value-min_value)*random.random()

    # -------------------------------------------------------------------------
    # Round-trip consistency checks with simple examples and randomized inputs
    # -------------------------------------------------------------------------

    def test_n_from_X(self):
        random.seed(1234567890)

        for i in range(100):
            V_A = self.random(10, 40)
            gamma_A = self.random(-math.pi/3, math.pi/3)
            gamma_A_dot = self.random(-math.pi/3, math.pi/3)
            g = 10
            chi_A_dot = 0
            mu_A = 0

            mapping = {
                "V_A": V_A,
                "gamma_A": gamma_A,
                "gamma_A_dot": gamma_A_dot,
                "g": g,
                "chi_A_dot": chi_A_dot,
                "mu_A": mu_A
            }

            n1 = n_from_chi_A_dot_gamma_A_dot(
                V_A, chi_A_dot, gamma_A, gamma_A_dot, g)
            n2 = n_from_forces_z_a2(V_A, gamma_A, gamma_A_dot, mu_A, g)
            self.assertAlmostEqual(n1, n2)

            n3 = n_from_forces_z_a2.eval_if(**mapping)
            n4 = n_from_chi_A_dot_gamma_A_dot.eval_if(**mapping)
            self.assertAlmostEqual(n2, n3)
            self.assertAlmostEqual(n3, n4)

    def test_roundtrip_n_gamma_A_dot_chi_A_dot(self):
        from . import kinematic_model as km

        random.seed(1234567890)

        for i in range(100):
            V_A = self.random(10, 40)
            gamma_A = self.random(-math.pi/3, math.pi/3)
            gamma_A_dot = self.random(-math.pi/3, math.pi/3)
            g = 10
            mu_A = self.random(-math.pi/3, math.pi/3)

            mapping = {
                "V_A": V_A,
                "gamma_A": gamma_A,
                "gamma_A_dot": gamma_A_dot,
                "g": g,
                "mu_A": mu_A
            }

            mapping["chi_A_dot"] = km.chi_A_dot_from_mu_A.eval_if(**mapping)
            mapping["n"] = n_from_chi_A_dot_gamma_A_dot.eval_if(**mapping)

            gamma_A_dot1 = km.gamma_A_dot_from_mu_A.eval_if(**mapping)
            gamma_A_dot2 = gamma_A_dot_from_n_chi_A_dot.eval_if(**mapping)
            self.assertAlmostEqual(gamma_A_dot1, mapping["gamma_A_dot"])
            self.assertAlmostEqual(gamma_A_dot2, mapping["gamma_A_dot"])

            chi_A_dot_abs1 = chi_A_dot_abs_from_n_gamma_A_dot.eval_if(**mapping)
            self.assertAlmostEqual(chi_A_dot_abs1, abs(mapping["chi_A_dot"]))

    def test_roundtrip_T_n_gamma_A_w_A_g(self):
        from . import kinematic_model as km
        from . import velocity_triangle as vt

        random.seed(1234567890)

        for i in range(100):
            V_A = self.random(10, 40)
            V_A_dot = self.random(-5, 5)
            gamma_A = self.random(-math.pi/3, math.pi/3)
            gamma_A_dot = self.random(-math.pi/3, math.pi/3)
            g = 10
            mu_A = 0  # self.random(-math.pi/3, math.pi/3)
            C_D0 = self.random(0.1, 0.4)
            C_L0 = self.random(0.1, 0.4)
            m = self.random(1, 20)
            k = self.random(0.1, 0.5)
            S = self.random(1, 10)
            rho = self.random(1, 1.5)

            mapping = {
                "V_A": V_A,
                "V_A_dot": V_A_dot,
                "gamma_A": gamma_A,
                "gamma_A_dot": gamma_A_dot,
                "g": g,
                "mu_A": mu_A,
                "C_D0": C_D0,
                "C_L0": C_L0,
                "m": m,
                "k": k,
                "S_": S,
                "rho": rho,
            }
            mapping["chi_A_dot"] = km.chi_A_dot_from_mu_A.eval_if(**mapping)
            mapping["w_A_g"] = vt.w_A_g_from_V_A_gamma_A.eval_if(**mapping)
            mapping["n"] = n_from_chi_A_dot_gamma_A_dot.eval_if(**mapping)

            gamma_A1 = vt.gamma_A_from_V_A_w_A_g.eval_if(**mapping)
            self.assertAlmostEqual(gamma_A1, mapping["gamma_A"])

            T1 = T_from_n_gamma_A.eval_if(**mapping)
            T2 = T_from_n_w_A_g.eval_if(**mapping)
            self.assertAlmostEqual(T1, T2)
            mapping["T"] = T1

            # Round-trip calculations from thrust
            gamma_A2 = gamma_A_from_T.eval_if(**mapping)
            self.assertAlmostEqual(gamma_A2, mapping["gamma_A"])
            w_A_g = w_A_g_from_T.eval_if(**mapping)
            self.assertAlmostEqual(w_A_g, mapping["w_A_g"])
            n_push = n_push_from_T_gamma_A.eval_if(**mapping)
            n_pull = n_pull_from_T_w_A_g.eval_if(**mapping)
            # NOTE: It's hard to decide whether we're pushing or pulling with
            # a zero-drag lift coefficient != 0 but it's gotta be one of the two.
            self.assertTrue(
                abs(mapping["n"] - n_push) < 1e-9
                or abs(mapping["n"] - n_pull) < 1e-9)

            V_A_dot2 = V_A_dot_from_T.eval_if(**mapping)
            self.assertAlmostEqual(V_A_dot2, mapping["V_A_dot"])

        # Again with zero airspeed rate to check round-trip airspeed
        # calculation from thrust:
        for i in range(100):
            V_A = self.random(10, 40)
            V_A_dot = 0
            gamma_A = self.random(-math.pi/3, math.pi/3)
            gamma_A_dot = 0
            g = 10
            mu_A = 0
            C_D0 = self.random(0.1, 0.4)
            C_L0 = self.random(0.1, 0.4)
            m = self.random(1, 20)
            k = self.random(0.1, 0.5)
            S = self.random(1, 10)
            rho = self.random(1, 1.5)

            mapping = {
                "V_A": V_A,
                "V_A_dot": V_A_dot,
                "gamma_A": gamma_A,
                "gamma_A_dot": gamma_A_dot,
                "g": g,
                "mu_A": mu_A,
                "C_D0": C_D0,
                "C_L0": C_L0,
                "m": m,
                "k": k,
                "S_": S,
                "rho": rho,
            }
            mapping["chi_A_dot"] = km.chi_A_dot_from_mu_A.eval_if(**mapping)
            mapping["w_A_g"] = vt.w_A_g_from_V_A_gamma_A.eval_if(**mapping)
            mapping["n"] = n_from_chi_A_dot_gamma_A_dot.eval_if(**mapping)

            gamma_A1 = vt.gamma_A_from_V_A_w_A_g.eval_if(**mapping)
            self.assertAlmostEqual(gamma_A1, mapping["gamma_A"])

            T1 = T_from_n_gamma_A.eval_if(**mapping)
            T2 = T_from_n_w_A_g.eval_if(**mapping)
            self.assertAlmostEqual(T1, T2)
            mapping["T"] = T1

            # Round-trip calculations from thrust
            V_A2 = V_A_from_T_gamma_A.eval_if(**mapping)
            self.assertAlmostEqual(V_A2, mapping["V_A"])

    def test_T_s_from_gamma_A(self):

        args = {
            "m": 4,
            "S_": 0.98,
            "C_D0": 0.032,
            "C_L0": 0.0,
            "k": 0.06,
            "rho": 1.225,
            "g": 9.81
        }

        # Straight level flight
        args["V_A"] = 15
        args["V_A_dot"] = 0
        args["gamma_A"] = 0
        args["chi_A_dot"] = 0
        T_s = T_s_from_gamma_A.eval_if(args)
        T_from_n = T_from_n_gamma_A.eval_if(args, n=1)
        self.assertAlmostEqual(T_s, T_from_n)

        # Straight climbing flight
        args["V_A"] = 15
        args["V_A_dot"] = 0
        args["gamma_A"] = 0.5
        args["chi_A_dot"] = 0
        T_s = T_s_from_gamma_A.eval_if(args)
        T_from_n = T_from_n_gamma_A.eval_if(
            args, n=n_s_from_V_A_gamma_A.eval_if(args))
        self.assertAlmostEqual(T_s, T_from_n)

        # Turning and climbing flight
        args["V_A"] = 15
        args["V_A_dot"] = 0
        args["gamma_A"] = 0.5
        args["chi_A_dot"] = 0.1
        T_s = T_s_from_gamma_A.eval_if(args)
        T_from_n = T_from_n_gamma_A.eval_if(
            args, n=n_s_from_V_A_gamma_A.eval_if(args))
        self.assertAlmostEqual(T_s, T_from_n)

    # ------------------------------------------------------------------
    # Symbolic dimensionality and equivalence checks
    # ------------------------------------------------------------------

    def test_check_dimensions(self):
        """Flight-performance functions pass a dimensional check.

        NOTE: ``gamma_A_dot_from_n_chi_A_dot`` is excluded as it produces a
        false negative result. Seems to be a problem with the piecewise
        implementation as checking both piecewise expressions by themselves
        works fine.
        """
        checked = [
            forces_x_a,
            forces_y_a1,
            forces_z_a2,
            n_from_forces_z_a2,
            n_from_forces_y_a1,
            n_from_chi_A_dot_gamma_A_dot,
            chi_A_dot_abs_from_n_gamma_A_dot,
            mu_A_from_n_chi_A_dot,
            mu_A_from_n_gamma_A_dot,
            C_L_from_chi_A_dot_gamma_A_dot,
            C_L_from_n,
            n_from_C_L,
            chi_A_dot_abs_from_C_L_gamma_A_dot,
            gamma_A_dot_from_C_L_chi_A_dot,
            V_A_from_C_L_n,
            V_A_from_C_L_gamma_A,
            D_from_n,
            T_from_n_gamma_A,
            T_s_from_gamma_A,
            n_pull_from_T_gamma_A,
            n_push_from_T_gamma_A,
            gamma_A_from_T,
            w_A_g_from_T,
            V_A_dot_from_T,
            V_A_from_T_gamma_A,
        ]
        for fn in checked:
            # ``check_dimensions`` probes each function with all arguments set
            # to 1, which lands outside the domain of some functions (e.g.
            # ``sqrt`` of a negative, ``arccos`` of |x|>1). Those yield ``nan``
            # plus a RuntimeWarning, but the *dimensional* check still passes,
            # so we silence the spurious numpy warnings here.
            with numpy.errstate(invalid="ignore", divide="ignore"):
                self.assertTrue(
                    fn.check_dimensions(),
                    f"check_dimensions failed for {fn.__name__}")

    def test_load_factor_from_forces(self):
        """``n`` derived from the force equilibrium equations.
        """
        syms = _SYMBOLS
        L, W, n, m, g, mu_A = [
            syms[k] for k in ("L", "W", "n", "m", "g", "mu_A")
        ]
        L_z, L_xy = syms["L_z"], syms["L_xy"]

        # Check n_from_forces_z_a2 equivalent to forces_z_a2
        #   L_z = W*cos(gamma_A) + m*gamma_A_dot*V_A, L_z = L*cos(mu_A), n = L/W
        eq = forces_z_a2.eq.subs(L_z, L*cos(mu_A)).subs(L, n*W).subs(W, m*g)
        n_sol = sympy.solve(eq, n)[0]
        self.assertTrue(is_zero(n_sol - n_from_forces_z_a2.eq.rhs))

        # Check n_from_forces_y_a1 equivalent to forces_y_a1
        eq = forces_y_a1.eq.subs(L_xy, L*sin(mu_A)).subs(L, n*W).subs(W, m*g)
        n_sol = sympy.solve(eq, n)[0]
        self.assertTrue(is_zero(n_sol - n_from_forces_y_a1.eq.rhs))

    def test_rates_from_load_factor_equalized(self):
        """``chi_A_dot`` / ``gamma_A_dot`` / ``mu_A`` from load factor.
        """
        syms = _SYMBOLS
        chi_A_dot = syms["chi_A_dot"]
        gamma_A_dot = syms["gamma_A_dot"]
        mu_A = syms["mu_A"]
        g = syms["g"]
        gamma_A = syms["gamma_A"]
        V_A = syms["V_A"]

        # Reference equation: n_from_forces_y_a1 == n_from_forces_z_a2
        n_equalized = sympy.Eq(
            0, n_from_forces_y_a1.eq.rhs - n_from_forces_z_a2.eq.rhs)

        # chi_A_dot_from_mu_A: solve n_equalized for chi_A_dot
        sols = sympy.solve(n_equalized, chi_A_dot)
        self.assertTrue(
            any(is_zero(s - km.chi_A_dot_from_mu_A.eq.rhs) for s in sols))

        # gamma_A_dot_from_mu_A: solve n_equalized for gamma_A_dot
        sols = sympy.solve(n_equalized, gamma_A_dot)
        self.assertTrue(
            any(is_zero(s - km.gamma_A_dot_from_mu_A.eq.rhs) for s in sols))

        # mu_A_from_chi_A_dot_gamma_A_dot: derive tan(mu_A) from n_equalized
        lhs = n_from_forces_y_a1.eq.rhs
        rhs = n_from_forces_z_a2.eq.rhs
        tan_mu = sympy.simplify(
            (lhs*g*sin(mu_A)*cos(mu_A))/cos(mu_A)/
            (g*cos(gamma_A) + gamma_A_dot*V_A))
        self.assertTrue(
            is_zero(
                sympy.atan(tan_mu) - km.mu_A_from_chi_A_dot_gamma_A_dot.eq.rhs))

    def test_mu_A_from_load_factor(self):
        """``mu_A`` from ``chi_A_dot`` and ``gamma_A_dot``.
        """
        syms = _SYMBOLS
        mu_A = syms["mu_A"]

        # Check mu_A_from_n_chi_A_dot is equivalent to from n_from_forces_y_a1
        sols = sympy.solve(n_from_forces_y_a1.eq, mu_A)
        self.assertTrue(
            any(is_zero(s - mu_A_from_n_chi_A_dot.eq.rhs) for s in sols))

        # Check mu_A_from_n_gamma_A_dot is equivalent to n_from_forces_z_a2
        sols = sympy.solve(n_from_forces_z_a2.eq, mu_A)
        self.assertTrue(
            any(is_zero(s - mu_A_from_n_gamma_A_dot.eq.rhs) for s in sols))

    def test_n_from_chi_A_dot_gamma_A_dot_derivation(self):
        """``n_from_chi_A_dot_gamma_A_dot`` derived from total lift (squared)."""
        syms = _SYMBOLS
        L, W, n, m, g = [syms[k] for k in ("L", "W", "n", "m", "g")]
        L_z, L_xy = syms["L_z"], syms["L_xy"]

        L_from = sympy.Eq(L, sqrt(L_xy**2 + L_z**2))
        L_from = L_from.subs(forces_y_a1.lhs, forces_y_a1.rhs)
        L_from = L_from.subs(forces_z_a2.lhs, forces_z_a2.rhs)
        eq = sympy.Eq(L_from.lhs**2, L_from.rhs**2)
        eq = sympy.Eq(eq.lhs/W**2, eq.rhs/W**2)
        eq = eq.subs(L/W, n).subs(W, m*g)
        n_pos = sympy.solve(eq, n)[1]  # positive solution

        # Compare squared with the positive branch of the Piecewise function
        n_pos_branch = positive_branch(n_from_chi_A_dot_gamma_A_dot.eq.rhs)
        self.assertTrue(is_zero(n_pos**2 - n_pos_branch**2))

    def test_rates_from_n(self):
        """``chi_A_dot`` / ``gamma_A_dot`` from the load-factor equation."""
        syms = _SYMBOLS
        n = syms["n"]
        chi_A_dot = syms["chi_A_dot"]
        gamma_A_dot = syms["gamma_A_dot"]

        n_pos_branch = positive_branch(n_from_chi_A_dot_gamma_A_dot.eq.rhs)

        # chi_A_dot_abs_from_n_gamma_A_dot: solve n = sqrt(...)/g for chi_A_dot
        eq = sympy.Eq(n, n_pos_branch)
        sols = sympy.solve(eq, chi_A_dot)
        self.assertTrue(
            any(
                is_zero(s - chi_A_dot_abs_from_n_gamma_A_dot.eq.rhs)
                for s in sols))

        # gamma_A_dot_from_n_chi_A_dot: solve n = sqrt(...)/g for gamma_A_dot
        eq = sympy.Eq(n, n_pos_branch)
        sols = sympy.solve(eq, gamma_A_dot)
        gamma_A_dot_pos = positive_branch(gamma_A_dot_from_n_chi_A_dot.eq.rhs)
        self.assertTrue(any(is_zero(s - gamma_A_dot_pos) for s in sols))

    def test_C_L_from_n_derivation(self):
        """``C_L_from_n`` derived from the ``C_L`` definition."""
        syms = _SYMBOLS
        L, W, n, m, g, C_L = [syms[k] for k in ("L", "W", "n", "m", "g", "C_L")]
        defs = _DEFINITIONS
        C_L_eq = defs["C_L"].eq

        eq = C_L_eq.subs(L, n*W).subs(W, m*g)
        C_L_sol = sympy.solve(eq, C_L)[0]
        self.assertTrue(is_zero(C_L_sol - C_L_from_n.eq.rhs))

    def test_V_A_from_C_L_n(self):
        """``V_A_from_C_L_n`` vs ``C_L_from_n``."""
        syms = _SYMBOLS
        V_A = syms["V_A"]

        sols = sympy.solve(C_L_from_n.eq, V_A)
        self.assertTrue(
            any(is_zero(s**2 - V_A_from_C_L_n.eq.rhs**2) for s in sols))

    def test_C_L_from_chi_A_dot_gamma_A_dot(self):
        """``C_L_from_chi_A_dot_gamma_A_dot`` (squared) vs load factor."""
        syms = _SYMBOLS
        n = syms["n"]
        V_A = syms["V_A"]
        m = syms["m"]
        rho = syms["rho"]
        S_ = syms["S_"]
        g = syms["g"]

        n_pos_branch = positive_branch(n_from_chi_A_dot_gamma_A_dot.eq.rhs)
        # C_L = 2*m/(rho*V_A^2*S_) * n*g  (since n = sqrt(radicand)/g)
        lhs = C_L_from_chi_A_dot_gamma_A_dot.eq.rhs**2
        rhs = (2*m/(rho*V_A**2*S_)*(n_pos_branch*g))**2
        self.assertTrue(is_zero(lhs - rhs))

    def test_chi_A_dot_abs_from_C_L_gamma_A_dot(self):
        """``chi_A_dot_abs_from_C_L_gamma_A_dot`` (squared) vs load factor."""
        syms = _SYMBOLS
        C_L = syms["C_L"]

        C_L_expr = sympy.solve(n_from_C_L.eq, C_L)[0]
        lhs = chi_A_dot_abs_from_C_L_gamma_A_dot.eq.rhs.subs(C_L, C_L_expr)
        rhs = chi_A_dot_abs_from_n_gamma_A_dot.eq.rhs
        self.assertTrue(is_zero(lhs**2 - rhs**2))

    def test_gamma_A_dot_from_C_L_chi_A_dot(self):
        """``gamma_A_dot_from_C_L_chi_A_dot`` vs load factor."""
        syms = _SYMBOLS
        C_L = syms["C_L"]

        C_L_expr = sympy.solve(n_from_C_L.eq, C_L)[0]
        lhs = gamma_A_dot_from_C_L_chi_A_dot.eq.rhs.subs(C_L, C_L_expr)
        rhs = positive_branch(gamma_A_dot_from_n_chi_A_dot.eq.rhs)
        self.assertTrue(is_zero(lhs - rhs))

    def test_T_from_n_gamma_A_derivation(self):
        """``T_from_n_gamma_A`` derived from the ``forces_x_a`` equation."""
        syms = _SYMBOLS
        L, W, n, m, g = [syms[k] for k in ("L", "W", "n", "m", "g")]
        D, C_D, C_L = syms["D"], syms["C_D"], syms["C_L"]
        defs = _DEFINITIONS

        eq = forces_x_a.eq
        eq = eq.subs(D, defs["D"].rhs)
        eq = eq.subs(C_D, defs["C_D"].rhs)
        eq = eq.subs(C_L, defs["C_L"].rhs)
        eq = eq.subs(L, n*W).subs(W, m*g)
        eq = sympy.simplify(eq)
        self.assertTrue(is_zero(eq.rhs - T_from_n_gamma_A.eq.rhs))

    def test_n_pull_push_from_T_gamma_A(self):
        """``n_pull`` / ``n_push`` from ``T_from_n_gamma_A`` (solve quadratic).

        Solves ``T_from_n_gamma_A`` for ``n`` (after clearing denominators) and
        checks the two sign branches against ``n_pull_from_T_gamma_A`` and
        ``n_push_from_T_gamma_A``.
        """
        syms = _SYMBOLS
        T = syms["T"]
        n = syms["n"]
        V_A = syms["V_A"]
        V_A_dot = syms["V_A_dot"]
        gamma_A = syms["gamma_A"]
        C_D0 = syms["C_D0"]
        C_L0 = syms["C_L0"]
        m = syms["m"]
        g = syms["g"]
        k = syms["k"]
        S_ = syms["S_"]
        rho = syms["rho"]
        W = syms["W"]

        # Rewrite the thrust equation as a quadratic in n.
        eq = T_from_n_gamma_A.eq.subs(m*g, W)
        eq = sympy.Eq(eq.lhs*2*V_A**2*rho*S_, eq.rhs*2*V_A**2*rho*S_)
        eq = sympy.expand(eq)
        eq = sympy.expand(sympy.Eq(eq.lhs/(4*k*W**2), eq.rhs/(4*k*W**2)))
        sols = sympy.solve(eq, n)
        sols = [s.subs(W, m*g) for s in sols]

        # Each solution corresponds to exactly one of the pull/push branches.
        self.assertTrue(
            any(is_zero(s - n_pull_from_T_gamma_A.eq.rhs) for s in sols))
        self.assertTrue(
            any(is_zero(s - n_push_from_T_gamma_A.eq.rhs) for s in sols))

    def test_gamma_A_from_T(self):
        """``gamma_A_from_T`` consistent with ``T_from_n_gamma_A``.
        """
        syms = _SYMBOLS
        T = syms["T"]
        n = syms["n"]
        V_A = syms["V_A"]
        V_A_dot = syms["V_A_dot"]
        gamma_A = syms["gamma_A"]
        m = syms["m"]
        g = syms["g"]
        W = syms["W"]
        D = syms["D"]
        C_D = syms["C_D"]
        C_L = syms["C_L"]
        L = syms["L"]
        defs = _DEFINITIONS

        # Extract the inner expression of gamma_A_from_T, i.e. T/(g*m), the
        # counter_forces_term.
        asin_arg = gamma_A_from_T.eq.rhs.args[0]
        max_ = [a for a in asin_arg.atoms(sympy.Max)][0]
        inner = max_.args[1]

        # sin(gamma_A) from T_from_n_gamma_A
        sin_gamma = sympy.solve(T_from_n_gamma_A.eq, sympy.sin(gamma_A))[0]
        sin_gamma = sin_gamma.subs(D, defs["D"].rhs)
        sin_gamma = sin_gamma.subs(C_D, defs["C_D"].rhs)
        sin_gamma = sin_gamma.subs(C_L, defs["C_L"].rhs)
        sin_gamma = sin_gamma.subs(L, n*W).subs(W, m*g)

        self.assertTrue(is_zero(sin_gamma - inner))

    def test_w_A_g_from_T(self):
        """``w_A_g_from_T`` consistent with ``T_from_n_w_A_g``."""
        syms = _SYMBOLS
        w_A_g = syms["w_A_g"]

        sols = sympy.solve(T_from_n_w_A_g.eq, w_A_g)
        self.assertTrue(any(is_zero(s - w_A_g_from_T.eq.rhs) for s in sols))

    def test_V_A_dot_from_T(self):
        """``V_A_dot_from_T`` consistent with ``T_from_n_gamma_A``."""
        syms = _SYMBOLS
        V_A_dot = syms["V_A_dot"]

        sols = sympy.solve(T_from_n_gamma_A.eq, V_A_dot)
        self.assertTrue(any(is_zero(s - V_A_dot_from_T.eq.rhs) for s in sols))

    def test_V_A_from_T_gamma_A(self):
        """``V_A_from_T_gamma_A`` is the inverse of ``T_from_n_gamma_A``.

        Substitutes the ``V_A`` expression back into the thrust equation and
        checks the round-trip recovers ``T``.
        """
        syms = _SYMBOLS
        T = syms["T"]
        V_A = syms["V_A"]

        tgt = V_A_from_T_gamma_A.eq.rhs  # V_A as a function of T
        T_expr = T_from_n_gamma_A.eq.rhs  # T as a function of V_A
        T_roundtrip = T_expr.subs(V_A, tgt)
        self.assertTrue(is_zero(T_roundtrip - T))

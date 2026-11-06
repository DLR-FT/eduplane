# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.model.ext_dubins_airplane.

Run from top-level package directory:

   python3 -m unittest tests.model.test_ext_dubins_airplane

"""

import math
import unittest

from eduplane.model.ext_dubins_airplane import *


def plot_thrust_over_airspeed(
        ac, gamma_A, V_A_range=numpy.linspace(0, 50), chi_A_dot=0):
    import pandas
    from matplotlib import pyplot as plt
    df = pandas.DataFrame(
        {
            "V_A":
            V_A_range,
            "T": [
                utils.error2constant(fpm.T_s_from_gamma_A, constant=numpy.nan)(
                    V_A, gamma_A, chi_A_dot, ac.C_D0, ac.C_L0, ac.m, ac.g, ac.k,
                    ac.S, ac.rho) for V_A in V_A_range
            ]
        })
    df["P"] = df["T"]*V_A_range
    ax = df.plot(y=["T", "P"], x="V_A")
    ax.axhline(ac.P_max)
    ax.axhline(ac.T_max)
    plt.show()


class ExtDubinsAirplaneTests(unittest.TestCase):

    def __test_ref_state(
            self, ac: ExtDubinsAirplane, ref_V_A, ref_n, ref_mu_A,
            ref_chi_A_dot, ref_gamma_A, ref_gamma_A_dot, ref_w_A_g_dot,
            ref_V_A_dot):
        """Given a reference state, test the different strategies to calculate
        state variables."""

        # Load factor
        n1 = ac.load_factor(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            mu_A=ref_mu_A,
            gamma_A_dot=ref_gamma_A_dot)
        n2 = ac.load_factor(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            gamma_A_dot=ref_gamma_A_dot,
            chi_A_dot=ref_chi_A_dot)

        if abs(ref_mu_A) > 0:
            n3 = ac.load_factor(
                V_A=ref_V_A,
                gamma_A=ref_gamma_A,
                gamma_A_dot=ref_gamma_A_dot,
                mu_A=ref_mu_A)
        else:
            n3 = n2
        self.assertAlmostEqual(n1, n2)
        self.assertAlmostEqual(n2, n3)
        if ref_n is not None:
            self.assertAlmostEqual(n1, ref_n)
            self.assertAlmostEqual(n2, ref_n)
            self.assertAlmostEqual(n3, ref_n)
        else:
            ref_n = n1

        # Air-path inclination rate
        gamma_A_dot1 = ac.air_path_inclination_rate(
            V_A=ref_V_A,
            V_A_dot=ref_V_A_dot,
            gamma_A=ref_gamma_A,
            w_A_g_dot=ref_w_A_g_dot)
        if ref_n is not None:
            gamma_A_dot2 = ac.air_path_inclination_rate(
                V_A=ref_V_A,
                V_A_dot=ref_V_A_dot,
                gamma_A=ref_gamma_A,
                n=ref_n,
                chi_A_dot=ref_chi_A_dot)
        else:
            gamma_A_dot2 = gamma_A_dot1
        if abs(ref_mu_A) > 0:
            gamma_A_dot3 = ac.air_path_inclination_rate(
                V_A=ref_V_A,
                V_A_dot=ref_V_A_dot,
                gamma_A=ref_gamma_A,
                mu_A=ref_mu_A,
                chi_A_dot=ref_chi_A_dot)
        else:
            gamma_A_dot3 = gamma_A_dot2

        if ref_gamma_A_dot is not None:
            self.assertAlmostEqual(gamma_A_dot1, ref_gamma_A_dot)
            self.assertAlmostEqual(gamma_A_dot2, ref_gamma_A_dot)
            self.assertAlmostEqual(gamma_A_dot3, ref_gamma_A_dot)
        else:
            ref_gamma_A_dot = gamma_A_dot1
        self.assertAlmostEqual(gamma_A_dot1, gamma_A_dot2)
        self.assertAlmostEqual(gamma_A_dot2, gamma_A_dot3)

        # Roll angle
        mu_A1 = ac.roll_angle(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            n=ref_n,
            gamma_A_dot=ref_gamma_A_dot)
        mu_A2 = ac.roll_angle(
            V_A=ref_V_A, gamma_A=ref_gamma_A, n=ref_n, chi_A_dot=ref_chi_A_dot)

        self.assertAlmostEqual(mu_A1, mu_A2)
        if ref_mu_A:
            self.assertAlmostEqual(mu_A1, ref_mu_A)
            self.assertAlmostEqual(mu_A2, ref_mu_A)
        else:
            ref_mu_A = mu_A1

        # Turn rate
        chi_A_dot1 = ac.turn_rate(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            gamma_A_dot=ref_gamma_A_dot,
            mu_A=ref_mu_A)
        chi_A_dot2 = ac.turn_rate(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            gamma_A_dot=ref_gamma_A_dot,
            n=ref_n)
        self.assertAlmostEqual(chi_A_dot1, chi_A_dot2)
        if ref_chi_A_dot:
            self.assertAlmostEqual(chi_A_dot1, ref_chi_A_dot)
            self.assertAlmostEqual(chi_A_dot2, ref_chi_A_dot)
        else:
            ref_chi_A_dot = chi_A_dot1

        # Vertical acceleration
        w_A_g_dot1 = ac.vert_acc(
            V_A=ref_V_A,
            gamma_A=ref_gamma_A,
            n=ref_n,
            chi_A_dot=ref_chi_A_dot,
            V_A_dot=0)
        if ref_w_A_g_dot:
            self.assertAlmostEqual(w_A_g_dot1, ref_w_A_g_dot)

    def test_ref_states(self):

        ac = ExtDubinsAirplane(
            m=20,
            S=2,
            C_D0=0.028,
            C_L0=0.0,
            k=0.04,
            T_min=0,
            T_max=50,
            P_max=500,
            mu_A_max=math.radians(45),
            n_min=0,
            n_max=2,
            C_L_min=-1.2,
            C_L_max=1.4,
            V_A_max=50,
            g=9.81,
            rho=1.225)

        # 2g level turn at 60 deg bank and 20 m/s
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_n = 2
        ref_mu_A = math.radians(60)
        ref_chi_A_dot = ac.g*math.tan(ref_mu_A)/ref_V_A
        ref_gamma_A = 0
        ref_gamma_A_dot = 0
        ref_w_A_g_dot = 0

        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

        # Turn at 60 deg bank, 20 m/s airspeed and 30 deg flight path angle.
        # Note that the load factor scales with cos(gamma_A).
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_mu_A = math.radians(60)
        ref_gamma_A = math.radians(30)
        ref_gamma_A_dot = 0
        ref_w_A_g_dot = 0
        ref_n = 2*math.cos(ref_gamma_A)
        ref_chi_A_dot = ac.g*math.tan(ref_mu_A)/ref_V_A

        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

        # 2g pull-up at 0 deg bank and 0 deg flight path angle.
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_mu_A = math.radians(0)
        ref_gamma_A = math.radians(0)
        ref_gamma_A_dot = ac.g/ref_V_A
        ref_w_A_g_dot = -ac.g
        ref_n = 2
        ref_chi_A_dot = 0

        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

        # 3g pull-up at 0 deg bank and 20 deg flight path angle.
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_mu_A = math.radians(0)
        ref_gamma_A = math.radians(0)
        ref_gamma_A_dot = 2*ac.g/ref_V_A
        ref_w_A_g_dot = -2*ac.g*cos(ref_gamma_A)
        ref_n = cos(ref_gamma_A) + 2
        ref_chi_A_dot = 0

        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

        # Vertically accelerated turn:
        # 2g turn + 1g pull-up at 20 m/s airspeed and 0 deg flight path angle
        # Note that a smaller bank angle (45° as opposed to 60°) is required
        # for the horizontal 2g acceleration (for turning) as opposed to
        # quasi-steady flight as the total lift force is higher
        # (see also 'mu_A_from_chi_A_dot_gamma_A_dot').
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_mu_A = math.radians(45)
        ref_gamma_A = math.radians(0)
        ref_gamma_A_dot = ac.g/ref_V_A
        ref_w_A_g_dot = -ac.g
        ref_chi_A_dot = 2*ac.g/ref_V_A
        ref_n = None

        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

        # Vertically accelerated turn:
        # 2g turn + 1g pull-up at 20 m/s airspeed and 30 deg flight path angle
        ref_V_A = 20
        ref_V_A_dot = 0
        ref_gamma_A = math.radians(30)
        ref_gamma_A_dot = ac.g/ref_V_A
        ref_w_A_g_dot = km.w_A_g_dot_from_gamma_A_dot(
            ref_V_A, ref_V_A_dot, ref_gamma_A, ref_gamma_A_dot)
        ref_chi_A_dot = 2*ac.g/ref_V_A
        ref_mu_A = km.mu_A_from_chi_A_dot_gamma_A_dot(
            ref_V_A, ref_chi_A_dot, ref_gamma_A, ref_gamma_A_dot, ac.g)
        ref_n = None
        self.__test_ref_state(
            ac, ref_V_A, ref_n, ref_mu_A, ref_chi_A_dot, ref_gamma_A,
            ref_gamma_A_dot, ref_w_A_g_dot, ref_V_A_dot)

    def test_climb_rate_max_from_thrust(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.028,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.04,
            T_min=0,
            T_max=30,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        # Constants
        chi_A_dot = 0
        mu_A = 0
        n = 1

        V_A_min_n1 = ac.airspeed_bounds(n=n)[0]

        climb_rate_n1, climb_airspeed_n1 = ac.climb_rate_max(n=n)
        self.assertGreater(climb_rate_n1, 0)
        self.assertGreaterEqual(climb_airspeed_n1, V_A_min_n1)
        self.assertLessEqual(climb_airspeed_n1, ac.V_A_max)
        descent_rate_n1, descent_airspeed_n1 = ac.descent_rate_min(
            T_max=ac.T_min, n=n)
        self.assertGreater(descent_rate_n1, 0)
        # self.assertGreaterEqual(descent_airspeed_n1, V_A_min_n1)
        # self.assertLessEqual(descent_airspeed_n1, ac.V_A_max)

        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed_n1, w_A_g=-climb_rate_n1, n=n)
        descent_thrust = ac.thrust_required(
            V_A=descent_airspeed_n1, w_A_g=descent_rate_n1, n=n)

        self.assertAlmostEqual(climb_thrust, ac.T_max)
        self.assertAlmostEqual(descent_thrust, ac.T_min, places=3)

        # Above example used assumed a fixed load factor, however,
        # at steady climbing/descending flight the load factor scales
        # with the flight path angle (i.e. with cos(gamma_A)).
        # We can set n=None to invoke an iterative calculation of load factor
        # and climb rate. In the given example, the load factor is smaller
        # than assumed, so the iterated climb rate is slightly larger.
        # We can do the iteration manually:
        gamma_A_at_climb_rate_n1 = vt.gamma_A_from_V_A_w_A_g(
            climb_airspeed_n1, -climb_rate_n1)
        load_factor_at_climb_rate_n1 = ac.load_factor(
            V_A=climb_airspeed_n1,
            w_A_g=-climb_rate_n1,
            chi_A_dot=0,
            w_A_g_dot=0)
        self.assertAlmostEqual(
            load_factor_at_climb_rate_n1, cos(gamma_A_at_climb_rate_n1))
        gamma_A_at_descent_rate_n1 = vt.gamma_A_from_V_A_w_A_g(
            descent_airspeed_n1, descent_rate_n1)
        load_factor_at_descent_rate_n1 = ac.load_factor(
            V_A=descent_airspeed_n1,
            w_A_g=descent_rate_n1,
            chi_A_dot=0,
            w_A_g_dot=0)
        self.assertAlmostEqual(
            load_factor_at_descent_rate_n1, cos(gamma_A_at_descent_rate_n1))
        climb_rate_iterated, climb_airspeed_iterated = ac.climb_rate_max(
            n=load_factor_at_climb_rate_n1)
        self.assertGreater(climb_rate_iterated, climb_rate_n1)
        descent_rate_iterated, descent_airspeed_iterated = ac.descent_rate_min(
            T_max=ac.T_min, n=load_factor_at_descent_rate_n1)
        self.assertGreater(climb_rate_iterated, climb_rate_n1)
        # ..or just leave default value n=None:
        climb_rate, climb_airspeed = ac.climb_rate_max()
        self.assertGreater(climb_rate, climb_rate_n1)
        # Note that the one manual iteration did not give us the final result,
        # but it should be pretty close already.
        self.assertAlmostEqual(climb_rate, climb_rate_iterated, places=2)
        self.assertGreater(climb_rate, 0)
        descent_rate, descent_airspeed = ac.descent_rate_min(T_max=ac.T_min)
        self.assertLess(descent_rate, descent_rate_n1)
        # Note that the one manual iteration did not give us the final result,
        # but it should be pretty close already.
        self.assertAlmostEqual(descent_rate, descent_rate_iterated, places=2)
        self.assertGreater(descent_rate, 0)

        # From these values, we should be able to get back the min/max thrust
        # values as a 'round-trip' calculation
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed, w_A_g=-climb_rate, w_A_g_dot=0)
        self.assertAlmostEqual(climb_thrust, ac.T_max, places=3)
        descent_thrust = ac.thrust_required(
            V_A=descent_airspeed, w_A_g=descent_rate, w_A_g_dot=0)
        self.assertAlmostEqual(descent_thrust, ac.T_min, places=3)

        # Special case: n=0
        chi_A_dot = 0
        mu_A = 0
        n = 0

        climb_rate_n0, climb_airspeed_n0 = ac.climb_rate_max(n=n)
        self.assertGreater(climb_rate_n0, 0)
        self.assertLessEqual(climb_airspeed_n0, ac.V_A_max)
        descent_rate_n0, descent_airspeed_n0 = ac.descent_rate_min(
            T_max=ac.T_min, n=n)
        self.assertGreater(descent_rate_n0, 0)
        self.assertLessEqual(descent_airspeed_n0, ac.V_A_max)

        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed_n0, w_A_g=-climb_rate_n0, n=n)
        descent_thrust = ac.thrust_required(
            V_A=descent_airspeed_n0, w_A_g=descent_rate_n0, n=n)

        self.assertAlmostEqual(climb_thrust, ac.T_max)
        self.assertAlmostEqual(descent_thrust, ac.T_min)

    def test_climb_rate_max_from_airspeed(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.028,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.04,
            T_min=0,
            T_max=30,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=50)

        # Constants
        chi_A_dot = 0
        mu_A = 0
        n = 1

        # Minimum level flight airspeed
        V_A_min_level = ac.airspeed_bounds(n=n, gamma_A=0)[0]

        # Make sure the min. airspeed is not constrained due to max. thrust
        T_V_A_min_level = ac.thrust_required(V_A=V_A_min_level, gamma_A=0, n=n)
        self.assertLess(T_V_A_min_level, ac.T_max)

        # The thrust reserve should allow us to climb
        climb_rate_level = ac.climb_rate_max(V_A=V_A_min_level, n=n)[0]
        self.assertGreater(climb_rate_level, 0)
        descent_rate_level = ac.descent_rate_min(V_A=V_A_min_level, n=n)[0]
        self.assertLess(descent_rate_level, 0)

        # Maximum thrust level flight airspeed
        # Increase power limit to only make thrust constraint effective
        ac.P_max = 1e5
        V_A_max_level = ac.airspeed(T=ac.T_max, n=n, gamma_A=0)
        # Make sure the max. airspeed is constrained by thrust
        self.assertLess(V_A_max_level, ac.V_A_max)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_max_level, w_A_g=0), ac.T_max)

        # Without any thrust reserve, no climb possible
        climb_rate_level = ac.climb_rate_max(V_A=V_A_max_level, n=n)[0]
        self.assertAlmostEqual(climb_rate_level, 0, places=3)
        descent_rate_level = ac.descent_rate_min(V_A=V_A_max_level, n=n)[0]
        self.assertAlmostEqual(descent_rate_level, 0, places=3)

        # Slightly decreasing airspeed should give us some thrust reserve
        # and climb ability
        climb_rate_level = ac.climb_rate_max(V_A=V_A_max_level*0.9, n=n)[0]
        self.assertGreater(climb_rate_level, 0)
        descent_rate_level = ac.descent_rate_min(V_A=V_A_max_level*0.9, n=n)[0]
        self.assertLess(descent_rate_level, 0)

        # Maximum power level flight airspeed
        # Increase thrust limit to only make power constraint effective
        ac.P_max = 500
        ac.T_max = 1e3
        V_A_max_level = ac.airspeed(P=ac.P_max, n=n, gamma_A=0)
        # Make sure the max. airspeed is constrained by power
        self.assertLess(V_A_max_level, ac.V_A_max)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_max_level, w_A_g=0), ac.P_max)

        # Without any power reserve, no climb possible
        climb_rate_level = ac.climb_rate_max(V_A=V_A_max_level, n=n)[0]
        self.assertAlmostEqual(climb_rate_level, 0, places=3)
        descent_rate_level = ac.descent_rate_min(V_A=V_A_max_level, n=n)[0]
        self.assertAlmostEqual(descent_rate_level, 0, places=3)

        # Slightly decreasing airspeed should give us some power reserve
        # and climb ability
        climb_rate_level = ac.climb_rate_max(V_A=V_A_max_level*0.9, n=n)[0]
        self.assertGreater(climb_rate_level, 0)
        descent_rate_level = ac.descent_rate_min(V_A=V_A_max_level*0.9, n=n)[0]
        self.assertLess(descent_rate_level, 0)

    def test_descent_rate_max_from_thrust(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.028,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.04,
            T_min=0,
            T_max=30,
            P_max=250,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=35)

        # Find max. descent rate at at min. thrust
        descent_rate, descent_airspeed = ac.descent_rate_max()
        self.assertGreater(descent_rate, 0)
        # Round-trip airspeed calculation
        V_A_descent = ac.airspeed(w_A_g=descent_rate, T=ac.T_min)
        self.assertAlmostEqual(descent_airspeed, V_A_descent, places=3)
        # Round-trip thrust calculation
        descent_thrust = ac.thrust_required(
            V_A=descent_airspeed, w_A_g=descent_rate)
        self.assertAlmostEqual(descent_thrust, ac.T_min, places=3)
        # Min. climb rate equals max. descent rate
        climb_rate, climb_airspeed = ac.climb_rate_min()
        self.assertAlmostEqual(descent_rate, -climb_rate)
        self.assertAlmostEqual(descent_airspeed, climb_airspeed)

        # Find min. climb rate at max. thrust and max. airspeed < dive speed
        ac.V_A_max = 30
        T = ac.T_max
        climb_rate, climb_airspeed = ac.climb_rate_min(T_min=T)
        self.assertGreater(climb_rate, 0)
        # Round-trip thrust calculation
        climb_thrust = ac.thrust_required(V_A=climb_airspeed, w_A_g=-climb_rate)
        self.assertAlmostEqual(climb_thrust, T)
        # Max. descent rate equals min. climb rate
        descent_rate, descent_airspeed = ac.descent_rate_max(T_min=T)
        self.assertAlmostEqual(descent_rate, -climb_rate)
        self.assertAlmostEqual(descent_airspeed, climb_airspeed)

        # Special case: n=0 at full thrust
        n = 0
        T = ac.T_max
        climb_rate_n0, climb_airspeed_n0 = ac.climb_rate_min(T_min=T, n=n)
        self.assertGreater(climb_rate_n0, 0)
        descent_rate_n0, descent_airspeed_n0 = ac.descent_rate_max(T_min=T, n=n)
        self.assertLess(descent_rate_n0, 0)
        # Round-trip thrust calculation
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed_n0, w_A_g=-climb_rate_n0, n=n)
        descent_thrust = ac.thrust_required(
            V_A=descent_airspeed_n0, w_A_g=descent_rate_n0, n=n)
        self.assertAlmostEqual(climb_thrust, T)
        self.assertAlmostEqual(descent_thrust, T)

    def test_terminal_dive_airspeed(self):
        # Model where the terminal-dive velocity lies below V_A_max, so a
        # real root exists within vert_speed's search interval.
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=1000,
            n_min=0.2,
            n_max=4.0,
            mu_A_max=math.radians(60),
            V_A_max=50)

        # Terminal-dive airspeed at min. thrust is below the structural limit.
        V_A_term = ac.terminal_dive_airspeed(T_min=ac.T_min)
        self.assertLess(V_A_term, ac.V_A_max)

        # Consistency with vert_speed: at the terminal-dive airspeed a steady
        # vertical-speed equilibrium still exists, i.e. vert_speed must not
        # raise (this was the invariant that motivated the fix).
        w_A_g = ac.vert_speed(V_A=V_A_term, T=ac.T_min)
        self.assertIsNotNone(w_A_g)

        # Just above the terminal-dive airspeed no equilibrium exists, so
        # vert_speed raises (excess thrust has no sign change).
        with self.assertRaises(ValueError):
            ac.vert_speed(V_A=V_A_term + 0.5, T=ac.T_min)

        # Clamping: when the terminal-dive velocity exceeds V_A_max, the
        # result is clamped to V_A_max.
        ac.V_A_max = 20
        self.assertEqual(ac.terminal_dive_airspeed(T_min=ac.T_min), ac.V_A_max)

    def test_load_factor(self):
        # Note that the below round-trip checks from load factor to thrust and
        # back are only valid at n=1 and with a symmetric drag polar
        # (with C_L0=0).
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Climb at 100% thrust and n=1
        climb_rate, climb_airspeed = ac.climb_rate_max(n=1)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed, w_A_g=-climb_rate, n=1)
        # climb_gamma_A = vt.gamma_A_from_V_A_w_A_g(climb_airspeed, -climb_rate)
        self.assertAlmostEqual(climb_thrust, ac.T_max)
        # With no thrust reserve, we cannot accelerate vertically
        min_load_factor, max_load_factor = ac.load_factor(
            V_A=climb_airspeed, T=climb_thrust, w_A_g=-climb_rate, V_A_dot=0)
        self.assertAlmostEqual(min_load_factor, -1)
        self.assertAlmostEqual(max_load_factor, 1)

        # Climb at 80% thrust and n=1
        climb_rate, climb_airspeed = ac.climb_rate_max(n=1, T_max=ac.T_max*0.8)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed, w_A_g=-climb_rate, n=1)
        # climb_gamma_A = vt.gamma_A_from_V_A_w_A_g(climb_airspeed, -climb_rate)
        self.assertAlmostEqual(climb_thrust, ac.T_max*0.8)
        min_load_factor, max_load_factor = ac.load_factor(
            V_A=climb_airspeed, T=climb_thrust, w_A_g=-climb_rate, V_A_dot=0)
        self.assertAlmostEqual(min_load_factor, -1)
        self.assertAlmostEqual(max_load_factor, 1)

        # Climb at 80% thrust and n=1.5
        climb_rate, climb_airspeed = ac.climb_rate_max(
            n=1.5, T_max=ac.T_max*0.8)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed, w_A_g=-climb_rate, n=1.5)
        # climb_gamma_A = vt.gamma_A_from_V_A_w_A_g(climb_airspeed, -climb_rate)
        self.assertAlmostEqual(climb_thrust, ac.T_max*0.8)
        min_load_factor, max_load_factor = ac.load_factor(
            V_A=climb_airspeed, T=climb_thrust, w_A_g=-climb_rate, V_A_dot=0)
        self.assertAlmostEqual(min_load_factor, -1.5)
        self.assertAlmostEqual(max_load_factor, 1.5)

        # Special case: n=0
        # Note that we have to slightly increase the thrust value for load
        # factor calculation to avoid numerical issues. The thrust for zero
        # load factor is equal to the drag force at zero lift, hence, any
        # numerical deviation will result in an infeasible flight state.
        climb_rate_n0, climb_airspeed_n0 = ac.climb_rate_max(
            n=0, T_max=ac.T_max*0.8)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed_n0, w_A_g=-climb_rate_n0, n=0)
        self.assertAlmostEqual(climb_thrust, ac.T_max*0.8)
        min_load_factor, max_load_factor = ac.load_factor(
            V_A=climb_airspeed_n0,
            T=climb_thrust*(1+1e-9),
            w_A_g=-climb_rate_n0,
            V_A_dot=0)
        self.assertAlmostEqual(min_load_factor, 0, places=3)
        self.assertAlmostEqual(max_load_factor, 0, places=3)

        # # Real-world test case #1
        # T = 28.129
        # V_A = 8.182
        # w_A_g = -5.214
        # w_A_g_dot = -28.542
        # chi_A_dot = 0
        # gamma_A = 0.691
        # gamma_A_dot = 5.681
        # n_from_w_A_g_dot = ac.load_factor(
        #     V_A=V_A, w_A_g=w_A_g, chi_A_dot=chi_A_dot, w_A_g_dot=w_A_g_dot)
        # n_from_gamma_A_dot = ac.load_factor(
        #     V_A=V_A,
        #     gamma_A=gamma_A,
        #     chi_A_dot=chi_A_dot,
        #     gamma_A_dot=gamma_A_dot)
        # w_A_g_dot_from_gamma_A_dot = km.w_A_g_dot_from_V_A_dot(
        #     V_A, 0, gamma_A, gamma_A_dot)
        # self.assertAlmostEqual(w_A_g_dot, w_A_g_dot_from_gamma_A_dot, places=2)
        # self.assertAlmostEqual(n_from_w_A_g_dot, n_from_gamma_A_dot)

    def test_airspeed_max(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Max. airspeed at level flight - constrained by power
        V_A_max = ac.airspeed_max(gamma_A=0)
        T_at_V_A_max = ac.thrust_required(V_A=V_A_max, gamma_A=0)
        P_at_V_A_max = ac.power_required(V_A=V_A_max, gamma_A=0)
        self.assertAlmostEqual(P_at_V_A_max, ac.P_max, places=1)
        self.assertLess(T_at_V_A_max, ac.T_max)

        # Max. airspeed at climbing flight - constrained by thrust
        gamma_A_climb = 0.15*math.pi
        V_A_max = ac.airspeed_max(gamma_A=gamma_A_climb)
        T_at_V_A_max = ac.thrust_required(V_A=V_A_max, gamma_A=gamma_A_climb)
        P_at_V_A_max = ac.power_required(V_A=V_A_max, gamma_A=gamma_A_climb)
        self.assertAlmostEqual(T_at_V_A_max, ac.T_max)
        self.assertLess(P_at_V_A_max, ac.P_max)

    def test_airspeed(self):
        ac = ExtDubinsAirplane(
            m=6,
            S=0.98,
            C_D0=0.032,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Max. airspeed calculation
        # =========================

        # Max. level airspeed at max. thrust - bounded by V_max
        ac.V_A_max = 30
        V_A_level_T_max = ac.airspeed(gamma_A=0, n=1, T=ac.T_max)
        self.assertAlmostEqual(V_A_level_T_max, ac.V_A_max)

        # Max. level/climb/descent airspeed at max. thrust - not bounded by V_max
        ac.V_A_max = 100
        V_A_level_T_max = ac.airspeed(gamma_A=0, n=1, T=ac.T_max)
        self.assertGreater(V_A_level_T_max, 30)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_level_T_max, gamma_A=0, n=1), ac.T_max)
        V_A_climb_T_max = ac.airspeed(gamma_A=0.1*math.pi, T=ac.T_max)
        self.assertLess(V_A_climb_T_max, V_A_level_T_max)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_climb_T_max, gamma_A=0.1*math.pi),
            ac.T_max)
        V_A_descent_T_max = ac.airspeed(gamma_A=-0.1*math.pi, T=ac.T_max)
        self.assertGreater(V_A_descent_T_max, V_A_level_T_max)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_descent_T_max, gamma_A=-0.1*math.pi),
            ac.T_max)

        # Special case: n=0
        ac.V_A_max = 100
        V_A_T_max_n0 = ac.airspeed(gamma_A=0, n=0, T=ac.T_max)
        self.assertGreater(V_A_T_max_n0, V_A_level_T_max)

        # Max. level/climb/descent airspeed at max. power - not bounded by V_max
        ac.V_A_max = 100
        V_A_level_P_max = ac.airspeed(gamma_A=0, n=1, P=ac.P_max)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_level_P_max, gamma_A=0, n=1),
            ac.P_max,
            places=3)
        V_A_climb_P_max = ac.airspeed(gamma_A=0.1*math.pi, P=ac.P_max)
        self.assertLess(V_A_climb_P_max, V_A_level_P_max)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_climb_P_max, gamma_A=0.1*math.pi),
            ac.P_max)
        V_A_descent_P_max = ac.airspeed(gamma_A=-0.1*math.pi, P=ac.P_max)
        self.assertGreater(V_A_descent_P_max, V_A_level_P_max)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_descent_P_max, gamma_A=-0.1*math.pi),
            ac.P_max)

        # Level accelerating/decelerating flight at max. power
        V_A_level_acc_P_max = ac.airspeed(
            gamma_A=0, n=1, P=ac.P_max, V_A_dot=0.5)
        V_A_level_dec_P_max = ac.airspeed(
            gamma_A=0, n=1, P=ac.P_max, V_A_dot=-0.5)
        self.assertAlmostEqual(
            ac.power_required(
                V_A=V_A_level_acc_P_max, gamma_A=0, n=1, V_A_dot=0.5),
            ac.P_max,
            places=3)
        self.assertAlmostEqual(
            ac.power_required(
                V_A=V_A_level_dec_P_max, gamma_A=0, n=1, V_A_dot=-0.5),
            ac.P_max,
            places=3)
        self.assertLess(V_A_level_acc_P_max, V_A_level_P_max)
        self.assertGreater(V_A_level_dec_P_max, V_A_level_P_max)

        # Min. airspeed calculation
        # =========================

        # Find thrust setting for minimum feasible airspeed
        ac.C_L_max = 4
        V_A_min_level = ac.airspeed_min(gamma_A=0)
        T_V_A_min_level = ac.thrust_required(V_A=V_A_min_level, gamma_A=0)

        # from eduplane.utils import plotting
        # plotting.plot_thrust_over_airspeed(ac, gamma_A=0)
        # plotting.show()

        # Reproduce V_A_min_level via thrust setting
        V_A_min_level_from_T = ac.airspeed(
            gamma_A=0, T=T_V_A_min_level, V_A_i0="min")
        self.assertAlmostEqual(V_A_min_level, V_A_min_level_from_T)

        # Reproduce V_A_min_level via power setting
        V_A_min_level_from_P = ac.airspeed(
            gamma_A=0, P=T_V_A_min_level*V_A_min_level, V_A_i0="min")
        self.assertAlmostEqual(V_A_min_level, V_A_min_level_from_P)

        # # Min. level/climb/descent airspeed at max. thrust
        # V_A_min_level_T_max = ac.airspeed(gamma_A=0, n=1, T=ac.T_max)
        # self.assertLess(V_A_min_level_T_max, V_A_level_T_max)
        # self.assertAlmostEqual(
        #     ac.thrust_required(V_A=V_A_min_level_T_max, gamma_A=0, n=1), ac.T_max)
        # V_A_min_climb_T_max = ac.airspeed(gamma_A=0.1*math.pi, T=ac.T_max)
        # self.assertLess(V_A_min_climb_T_max, V_A_climb_T_max)
        # self.assertAlmostEqual(
        #     ac.thrust_required(V_A=V_A_climb_T_max, gamma_A=0.1*math.pi), ac.T_max)
        # V_A_min_descent_T_max = ac.airspeed(gamma_A=-0.1*math.pi, T=ac.T_max)
        # self.assertLess(V_A_min_descent_T_max, V_A_descent_T_max)
        # self.assertAlmostEqual(
        #     ac.thrust_required(V_A=V_A_min_descent_T_max, gamma_A=-0.1*math.pi),
        #     ac.T_max)

        # Real-world and debugging test cases
        # ================================

        # Test case from climb_angle_max
        gamma_A_climb = 0.588400406350374

        # plot_thrust_over_airspeed(ac, gamma_A_climb)
        V_A_climb_P_max = ac.airspeed(gamma_A=gamma_A_climb, P=ac.P_max)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_climb_P_max, gamma_A=gamma_A_climb),
            ac.P_max,
            places=3)

    def test_airspeed_opt(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Min. thrust airspeed at level straight flight
        V_A_opt = ac.airspeed_opt(n=1, gamma_A=0)
        T_V_A_opt = ac.thrust_required(V_A=V_A_opt, gamma_A=0, n=1)
        T_V_A_opt_left = ac.thrust_required(V_A=V_A_opt*0.99, gamma_A=0, n=1)
        T_V_A_opt_right = ac.thrust_required(V_A=V_A_opt*1.01, gamma_A=0, n=1)
        self.assertLess(T_V_A_opt, T_V_A_opt_left)
        self.assertLess(T_V_A_opt, T_V_A_opt_right)
        # Note that if we leave out the load factor, quasi-steady flight state
        # is assumed (i.e. n=1 at level flight).
        V_A_opt_steady = ac.airspeed_opt(gamma_A=0)
        self.assertAlmostEqual(V_A_opt, V_A_opt_steady)

        # Min. thrust airspeed for descending flight at constant gamma_A
        gamma_A_descent = math.radians(-10)
        n_descent = math.cos(gamma_A_descent)
        V_A_opt_descent = ac.airspeed_opt(n=n_descent, gamma_A=gamma_A_descent)
        T_V_A_opt = ac.thrust_required(
            V_A=V_A_opt_descent, gamma_A=gamma_A_descent, n=n_descent)
        # NOTE: Not a true minimum, but minimum admissable airspeed, i.e.
        # smaller airspeeds would violate min. thrust/max. drag. Hence, thrust
        # only increases towards higher airspeeds in this case.
        # However, we can check if the min. thrust is reached at this airspeed.
        T_V_A_opt_right = ac.thrust_required(
            V_A=V_A_opt_descent*1.01, gamma_A=gamma_A_descent, n=n_descent)
        self.assertLess(T_V_A_opt, T_V_A_opt_right)
        self.assertAlmostEqual(T_V_A_opt, ac.T_min, places=3)
        # Note that if we leave out the load factor, quasi-steady flight state
        # is assumed.
        V_A_opt_descent_steady = ac.airspeed_opt(gamma_A=gamma_A_descent)
        self.assertAlmostEqual(V_A_opt_descent, V_A_opt_descent_steady)

        # Min. thrust airspeed for descending flight at constant vert. speed
        # and with fixed load factor
        w_A_g_descent = vt.w_A_g_from_V_A_gamma_A(
            V_A_opt_descent, gamma_A_descent)
        n_descent = fpm.n_s_from_V_A_w_A_g(
            V_A_opt_descent, w_A_g_descent, 0, ac.g)
        V_A_opt_descent = ac.airspeed_opt(n=n_descent, w_A_g=w_A_g_descent)
        T_V_A_opt = ac.thrust_required(
            V_A=V_A_opt_descent, w_A_g=w_A_g_descent, n=n_descent)
        # NOTE: Not a true minimum, but minimum admissable airspeed, i.e.
        # smaller airspeeds would violate min. thrust/max. drag. Hence, thrust
        # only increases towards higher airspeeds in this case.
        # However, we can check if the min. thrust is reached at this airspeed.
        T_V_A_opt_right = ac.thrust_required(
            V_A=V_A_opt_descent*1.01, w_A_g=w_A_g_descent, n=n_descent)
        self.assertLess(T_V_A_opt, T_V_A_opt_right)
        self.assertAlmostEqual(T_V_A_opt, ac.T_min, places=2)
        # Note that if we leave out the load factor, quasi-steady flight state
        # is assumed. In contrast, to the airspeed optimization for constant
        # gamma_A, the air-path inclination angle and load factor will be free
        # in this optimization. Hence, we cannot assume to obtain the same
        # result as above with a fixed load factor.
        V_A_opt_descent_steady = ac.airspeed_opt(w_A_g=w_A_g_descent)
        n_descent_steady = fpm.n_s_from_V_A_w_A_g(
            V_A_opt_descent_steady, w_A_g_descent, 0, ac.g)
        T_V_A_opt_steady = ac.thrust_required(
            V_A=V_A_opt_descent_steady, w_A_g=w_A_g_descent, n=n_descent_steady)
        # NOTE: Not a true minimum, but minimum admissable airspeed, i.e.
        # smaller airspeeds would violate min. thrust/max. drag. Hence, thrust
        # only increases towards higher airspeeds in this case.
        # However, we can check if the min. thrust is reached at this airspeed.
        T_V_A_opt_steady_right = ac.thrust_required(
            V_A=V_A_opt_descent_steady*1.01,
            w_A_g=w_A_g_descent,
            n=n_descent_steady)
        self.assertLess(T_V_A_opt_steady, T_V_A_opt_steady_right)
        self.assertAlmostEqual(T_V_A_opt_steady, ac.T_min, places=2)

        # Max. thrust airspeed at level straight flight
        V_A_opt = ac.airspeed_opt(n=1, gamma_A=0, opt_goal="max thrust")
        T_V_A_opt = ac.thrust_required(V_A=V_A_opt, gamma_A=0, n=1)
        T_V_A_opt_left = ac.thrust_required(V_A=V_A_opt*0.99, gamma_A=0, n=1)
        T_V_A_opt_right = ac.thrust_required(V_A=V_A_opt*1.01, gamma_A=0, n=1)
        # plot_thrust_over_airspeed(ac, gamma_A=0)
        # NOTE: Not a true thrust maximum, but maximum admissable airspeed
        self.assertGreater(T_V_A_opt, T_V_A_opt_left)
        # Note that if we leave out the load factor, quasi-steady flight state
        # is assumed (i.e. n=1 at level flight).
        V_A_opt_steady = ac.airspeed_opt(gamma_A=0, opt_goal="max thrust")
        self.assertAlmostEqual(V_A_opt, V_A_opt_steady)

        # Optimize for power

        # Min. power airspeed at level straight flight
        V_A_opt = ac.airspeed_opt(n=1, gamma_A=0, opt_goal="min power")
        P_V_A_opt = ac.power_required(V_A=V_A_opt, gamma_A=0, n=1)
        P_V_A_opt_left = ac.power_required(V_A=V_A_opt*0.99, gamma_A=0, n=1)
        P_V_A_opt_right = ac.power_required(V_A=V_A_opt*1.01, gamma_A=0, n=1)
        # print(f"V_A_opt={V_A_opt} P_V_A_opt={P_V_A_opt}")
        # plot_thrust_over_airspeed(ac, gamma_A=0)
        # NOTE: Not a true power minimum, but minimum admissable airspeed
        self.assertLess(P_V_A_opt, P_V_A_opt_right)
        # Note that if we leave out the load factor, quasi-steady flight state
        # is assumed (i.e. n=1 at level flight).
        V_A_opt_steady = ac.airspeed_opt(gamma_A=0, opt_goal="min power")
        self.assertAlmostEqual(V_A_opt, V_A_opt_steady)

    def test_airspeed_bounds(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.05,
            C_L0=0.0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=30,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # from eduplane.utils import plotting
        # plotting.plot_thrust_over_airspeed(
        #     ac,
        #     gamma_A=[math.radians(30), 0,
        #              math.radians(-10),
        #              math.radians(-30)])
        # plotting.show()

        # === Test calculating global airspeed bounds =========================

        # Level flight - min. airspeed bounded by C_L_max, max. airspeed bounded by power
        V_A_min, V_A_max = ac.airspeed_bounds(gamma_A=0)
        C_L_at_V_A_min = ac.lift_coefficient(V_A=V_A_min, gamma_A=0)
        T_at_V_A_max = ac.thrust_required(V_A=V_A_max, gamma_A=0)
        P_at_V_A_max = ac.power_required(V_A=V_A_max, gamma_A=0)
        self.assertAlmostEqual(C_L_at_V_A_min, ac.C_L_max)
        self.assertAlmostEqual(P_at_V_A_max, ac.P_max)
        # Increase max. power to see max. thrust constraint become active
        ac.P_max = 1000
        V_A_min, V_A_max = ac.airspeed_bounds(gamma_A=0)
        T_at_V_A_max = ac.thrust_required(V_A=V_A_max, gamma_A=0)
        self.assertAlmostEqual(T_at_V_A_max, ac.T_max)

        # Slow, climbing flight - min. *and* max airspeed bounded by T_max
        # (to make this limit effective, we have to construct a case on the
        # negative arc of the T(V_A) curve by increasing max. lift coefficient
        # and reducing max. thrust)
        ac.C_L_max = 4.0
        ac.T_max = 25
        V_A_min, V_A_max = ac.airspeed_bounds(gamma_A=math.radians(30))
        T_V_A_min = ac.thrust_required(V_A=V_A_min, gamma_A=math.radians(30))
        T_V_A_max = ac.thrust_required(V_A=V_A_min, gamma_A=math.radians(30))
        self.assertAlmostEqual(T_V_A_min, ac.T_max)
        self.assertAlmostEqual(T_V_A_max, ac.T_max)

        # === Test calculating airspeed bounds reachable from ref. airspeed ===
        ac.T_max = 30
        ac.P_max = 500
        ac.C_L_max = 1.2

        # Slow, climbing flight (negative arc / left side of T(V_A) curve)
        # - min. airspeed bounded by C_L_max, max. airspeed bounded by min. thrust
        # (again to make this case work, we have to increase C_L_max)
        ac.C_L_max = 4.0
        V_A_min, V_A_max = ac.airspeed_bounds(
            gamma_A=math.radians(-10), V_A_ref=4.5)
        C_L_at_V_A_min = ac.lift_coefficient(
            V_A=V_A_min, gamma_A=math.radians(-10))
        T_at_V_A_max = ac.thrust_required(
            V_A=V_A_max, gamma_A=math.radians(-10))
        self.assertAlmostEqual(C_L_at_V_A_min, ac.C_L_max)
        self.assertAlmostEqual(T_at_V_A_max, ac.T_min, places=3)

        # Fast, climbing flight (positive arc / right side of T(V_A) curve)
        # - min. airspeed bounded by min. thrust, max. airspeed bounded by max. thrust
        ac.C_L_max = 1.2
        V_A_min, V_A_max = ac.airspeed_bounds(
            gamma_A=math.radians(-10), V_A_ref=15)
        T_at_V_A_min = ac.thrust_required(
            V_A=V_A_min, gamma_A=math.radians(-10))
        T_at_V_A_max = ac.thrust_required(
            V_A=V_A_max, gamma_A=math.radians(-10))
        P_at_V_A_max = ac.power_required(V_A=V_A_max, gamma_A=math.radians(-10))
        self.assertAlmostEqual(P_at_V_A_max, ac.P_max, places=3)
        self.assertAlmostEqual(T_at_V_A_min, ac.T_min, places=3)

        # At level flight, the T(V_A) curve does not hit the min. thrust constraint,
        # hence, we get global bounds with V_A_min limited by max. lift coeff.
        V_A_min, V_A_max = ac.airspeed_bounds(
            gamma_A=math.radians(0), V_A_ref=15)
        C_L_at_V_A_min = ac.lift_coefficient(V_A=V_A_min, gamma_A=0)
        T_at_V_A_max = ac.thrust_required(V_A=V_A_max, gamma_A=0)
        P_at_V_A_max = ac.power_required(V_A=V_A_max, gamma_A=0)
        self.assertAlmostEqual(C_L_at_V_A_min, ac.C_L_max)
        self.assertAlmostEqual(P_at_V_A_max, ac.P_max, places=3)

    def test_airspeed_min(self):
        # ac = ExtDubinsAirplane(
        #     m=4,
        #     S=0.98,
        #     C_D0=0.05,
        #     C_L0=0.0,
        #     C_L_min=-1.2,
        #     C_L_max=1.2*4,
        #     k=0.06,
        #     T_min=0,
        #     T_max=30,
        #     P_max=500,
        #     n_min=-0.5,
        #     n_max=2,
        #     mu_A_max=math.radians(60),
        #     V_A_max=30)

        # # from eduplane.utils import plotting
        # # plotting.plot_thrust_over_airspeed(
        # #     ac,
        # #     gamma_A=[math.radians(30), 0,
        # #              math.radians(-10),
        # #              math.radians(-30)])
        # # plotting.show()

        # # Min. airspeed at level flight - bounded by C_L_max
        # V_A_min = ac.airspeed_min(gamma_A=0)
        # C_L_at_V_A_min = ac.lift_coefficient(V_A=V_A_min, gamma_A=0)
        # self.assertAlmostEqual(C_L_at_V_A_min, ac.C_L_max)

        # # Min. airspeed at descending flight - bounded by T_min
        # V_A_min_descent = ac.airspeed_min(gamma_A=math.radians(-10))
        # T_V_A_min_descent = ac.thrust_required(
        #     V_A=V_A_min_descent, gamma_A=math.radians(-10))
        # self.assertAlmostEqual(T_V_A_min_descent, ac.T_min)

        # # Min. airspeed at slow, climbing flight - bounded by T_max
        # # (to make this limit effective, we have to construct a case on the
        # # negative arc of the thrust-over-airspeed curve by increasing max.
        # # lift coefficient and reducing max. thrust)
        # ac.C_L_max = 4.0
        # ac.T_max = 25
        # V_A_min_climb = ac.airspeed_min(gamma_A=math.radians(30))
        # T_V_A_min_climb = ac.thrust_required(V_A=V_A_min_climb, gamma_A=math.radians(30))
        # self.assertAlmostEqual(T_V_A_min_climb, ac.T_max)

        # # Min. airspeed at slow, climbing flight - bounded by P_max
        # # (as with max. thrust limit, this becomes effective with high max.
        # # lift coefficient and low max. power limit, resetting max. thrust
        # # to its original value to limit airspeed by max. power)
        # ac.T_max = 30
        # ac.P_max = 100
        # V_A_min_climb = ac.airspeed_min(gamma_A=math.radians(30))
        # P_V_A_min_climb = ac.power_required(V_A=V_A_min_climb, gamma_A=math.radians(30))
        # self.assertAlmostEqual(P_V_A_min_climb, ac.P_max)

        # Real-world test case
        # ====================
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0,
            k=0.06,
            T_min=0,
            T_max=28,
            P_max=500,
            n_min=-1.0,
            n_max=5.0,
            C_L_min=-1.2,
            C_L_max=3.0,
            mu_A_max=1.0471975511965976,
            V_A_max=50,
            g=9.81,
            rho=1.225,
            eps=1e-09)

        # print("T at 10m/s", ac.thrust_required(V_A=10, w_A_g=-5.722216638394428))
        # print(
        #     "V_A at T=25.409", ac.airspeed(T=25.409, w_A_g=-5.722216638394428))
        # print("V_A_min at T=25.409", fpm.V_A_min_from_T_gamma_A()

        # from eduplane.utils import plotting
        # V_A_range = numpy.linspace(5, 50, 100)
        # T_from_V_A = fpm.T_from_n_gamma_A(
        #     n=1,
        #     V_A=V_A_range,
        #     V_A_dot=0,
        #     gamma_A=0,
        #     C_D0=ac.C_D0,
        #     C_L0=ac.C_L0,
        #     m=ac.m,
        #     g=ac.g,
        #     k=ac.k,
        #     S_=ac.S,
        #     rho=ac.rho)
        # df = pandas.DataFrame(dict(T=T_from_V_A, V_A=V_A_range))
        # df.plot(x="V_A", y="T")
        # plotting.show()

        # # T_range = numpy.linspace(20, 30, 100)
        # T_range = T_from_V_A
        # V_A_from_T = fpm.V_A_from_T_gamma_A(
        #     T=T_range,
        #     n=1,
        #     gamma_A=0,
        #     V_A_dot=0,
        #     C_D0=ac.C_D0,
        #     C_L0=ac.C_L0,
        #     m=ac.m,
        #     g=ac.g,
        #     k=ac.k,
        #     S_=ac.S,
        #     rho=ac.rho)
        # V_A_min_from_T = fpm.V_A_min_from_T_gamma_A(
        #     T=T_range,
        #     n=1,
        #     gamma_A=0,
        #     V_A_dot=0,
        #     C_D0=ac.C_D0,
        #     C_L0=ac.C_L0,
        #     m=ac.m,
        #     g=ac.g,
        #     k=ac.k,
        #     S_=ac.S,
        #     rho=ac.rho)
        # df = pandas.DataFrame(
        #     dict(
        #         T=T_range,
        #         V_A=V_A_range,
        #         V_A_from_T=V_A_from_T,
        #         V_A_min_from_T=V_A_min_from_T))
        # # df.plot(x="V_A_from_T", y="T")
        # # df.plot(x="V_A_min_from_T", y="T")
        # df.plot(x="T", y=["V_A", "V_A_from_T", "V_A_min_from_T"])
        # plotting.show()

        # V_A_min = ac.airspeed_min(
        #     w_A_g=-5.722216638394428, chi_A_dot=0, T_min=0)
        # print(f"{V_A_min=}")

    def test_airspeed_rate(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.2,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Climb at 80% thrust
        n_ref = 1.5
        T_ref = ac.T_max*0.8
        climb_rate, climb_airspeed = ac.climb_rate_max(n=n_ref, T_max=T_ref)
        climb_gamma_A = vt.gamma_A_from_V_A_w_A_g(climb_airspeed, -climb_rate)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed, w_A_g=-climb_rate, n=n_ref)
        self.assertAlmostEqual(climb_thrust, T_ref)

        n_max = ac.load_factor(
            V_A=climb_airspeed, T=T_ref, w_A_g=-climb_rate, V_A_dot=0)[1]
        self.assertAlmostEqual(n_max, n_ref)

        # At max. load factor, there should be no excess thrust for airspeed rate
        V_A_dot_n_max = ac.airspeed_rate(
            V_A=climb_airspeed, w_A_g=-climb_rate, T=T_ref, n=n_max)

        self.assertAlmostEqual(V_A_dot_n_max, 0)

        # At reduced load factor, there should be some excess thrust for airspeed rate
        V_A_dot_n_red = ac.airspeed_rate(
            V_A=climb_airspeed, w_A_g=-climb_rate, T=T_ref, n=n_max*0.8)
        self.assertGreater(V_A_dot_n_red, 0)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed,
            V_A_dot=V_A_dot_n_red,
            w_A_g=-climb_rate,
            n=n_max*0.8)
        self.assertAlmostEqual(climb_thrust, T_ref)

        # Further increasing load factor should require to provide excess thrust with neg. airspeed rate
        V_A_dot_n_inc = ac.airspeed_rate(
            V_A=climb_airspeed, w_A_g=-climb_rate, T=T_ref, n=n_max*1.2)
        self.assertLess(V_A_dot_n_inc, 0)
        climb_thrust = ac.thrust_required(
            V_A=climb_airspeed,
            V_A_dot=V_A_dot_n_inc,
            w_A_g=-climb_rate,
            n=n_max*1.2)
        self.assertAlmostEqual(climb_thrust, T_ref)

    def test_vert_speed(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.2,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Steady climb at 45°
        V_A = 15
        gamma_A = math.radians(45)
        n = 1*cos(gamma_A)
        T = ac.thrust_required(V_A=V_A, gamma_A=gamma_A)
        w_A_g = ac.vert_speed(V_A=V_A, T=T, n=n)
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A))
        # Same result providing angular rates instead of load factor
        w_A_g = ac.vert_speed(V_A=V_A, T=T, chi_A_dot=0, gamma_A_dot=0)
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A))

        # Increase thrust at steady-state load factor -> accelerate
        T *= 1.2
        V_A_dot = ac.airspeed_rate(V_A=V_A, gamma_A=gamma_A, T=T, n=n)
        w_A_g = ac.vert_speed(V_A=V_A, T=T, n=n, V_A_dot=V_A_dot)
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A))
        # Same result providing accelerations instead of load factor

        w_A_g = ac.vert_speed(
            V_A=V_A, T=T, V_A_dot=V_A_dot, chi_A_dot=0, gamma_A_dot=0)
        print("gamma_A input: ", gamma_A)
        print("gamma_A result:", vt.gamma_A_from_V_A_w_A_g(V_A, w_A_g))
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A))

        # Increase thrust again, this time for a negative vertical acceleration
        T *= 1.05
        n = ac.load_factor(V_A=V_A, gamma_A=gamma_A, T=T, V_A_dot=V_A_dot)[1]
        w_A_g_dot = ac.vert_acc(
            V_A=V_A, V_A_dot=V_A_dot, gamma_A=gamma_A, chi_A_dot=0, n=n)
        T_check = ac.thrust_required(
            V_A=V_A, gamma_A=gamma_A, V_A_dot=V_A_dot, w_A_g_dot=w_A_g_dot)
        w_A_g = ac.vert_speed(V_A=V_A, T=T, n=n, V_A_dot=V_A_dot)
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A))
        # Same result providing accelerations instead of load factor
        w_A_g = ac.vert_speed(
            V_A=V_A,
            T=T,
            V_A_dot=V_A_dot,
            chi_A_dot=0,
            w_A_g_dot=w_A_g_dot,
            gamma_A_lower=math.radians(-80),
            gamma_A_upper=math.radians(80))
        self.assertAlmostEqual(w_A_g, -V_A*math.sin(gamma_A), places=6)

    def test_vert_acc(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.2,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Steady climb at 45°
        V_A_ref = 15
        gamma_A_ref = math.radians(45)
        n_ref = 1*cos(gamma_A_ref)
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(gamma_A_dot, 0)
        vert_acc = ac.vert_acc(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(vert_acc, 0)

        # Climb at 30° with increasing airspeed
        # (airspeed rate partially in vert. direction)
        V_A_ref = 15
        gamma_A_ref = math.radians(30)
        n_ref = 1*cos(gamma_A_ref)
        V_A_dot_ref = 5
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(gamma_A_dot, 0)
        vert_acc = ac.vert_acc(
            V_A=V_A_ref,
            gamma_A=gamma_A_ref,
            n=n_ref,
            chi_A_dot=0,
            V_A_dot=V_A_dot_ref)
        self.assertAlmostEqual(vert_acc, -V_A_dot_ref*sin(gamma_A_ref))

        # 2g Pull-up at 0°
        # (all of pull-up acceleration in vert. direction)
        V_A_ref = 15
        gamma_A_ref = 0
        n_ref = 2
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(gamma_A_dot, ac.g/V_A_ref)
        vert_acc = ac.vert_acc(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(vert_acc, -ac.g)

        # 2g Pull-up at 60°
        # (airspeed rate partially in vert. direction)
        V_A_ref = 15
        gamma_A_ref = math.radians(60)
        n_ref = 1*cos(gamma_A_ref) + 1
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(gamma_A_dot, ac.g/V_A_ref)
        vert_acc = ac.vert_acc(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(vert_acc, -ac.g*cos(gamma_A_ref))

        # 0g Push-down at 0°
        V_A_ref = 15
        gamma_A_ref = math.radians(0)
        n_ref = 0
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(gamma_A_dot, -ac.g/V_A_ref)
        vert_acc = ac.vert_acc(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        self.assertAlmostEqual(vert_acc, ac.g)

    def test_thrust(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            C_L0=0.2,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)
        # 2g pull-up at gamma_A=60°
        V_A_ref = 15
        gamma_A_ref = math.radians(60)
        n_ref = 1*cos(gamma_A_ref) + 1
        gamma_A_dot = ac.air_path_inclination_rate(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        w_A_g_dot = ac.vert_acc(
            V_A=V_A_ref, gamma_A=gamma_A_ref, n=n_ref, chi_A_dot=0, V_A_dot=0)
        # Thrust from n
        T1 = ac.thrust_required(
            V_A=V_A_ref, V_A_dot=0, gamma_A=gamma_A_ref, n=n_ref)
        # Thrust from gamma_A_dot
        T2 = ac.thrust_required(
            V_A=V_A_ref,
            V_A_dot=0,
            gamma_A=gamma_A_ref,
            gamma_A_dot=gamma_A_dot)
        # Thrust from w_A_g_dot
        T3 = ac.thrust_required(
            V_A=V_A_ref, V_A_dot=0, gamma_A=gamma_A_ref, w_A_g_dot=w_A_g_dot)
        self.assertAlmostEqual(T1, T2)
        self.assertAlmostEqual(T1, T3)

    def test_climb_angle_min(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            # C_L0=0.2,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=500,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Get min. sustainable angle and corresponding
        # max. airspeed at max. thrust
        gamma_A_min, V_A_gamma_A_min = ac.climb_angle_min(T=ac.T_max)
        # Should be climbing
        self.assertGreater(gamma_A_min, 0)
        self.assertLess(gamma_A_min, math.radians(45))
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_gamma_A_min, gamma_A=gamma_A_min),
            ac.T_max)
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_min, T=ac.T_max), V_A_gamma_A_min)

        # Get max. sustainable "dive" angle and corresponding
        # max. airspeed at zero thrust
        gamma_A_min, V_A_gamma_A_min = ac.climb_angle_min(T=ac.T_min)
        # Should be descending
        self.assertLess(gamma_A_min, 0)
        self.assertGreater(gamma_A_min, -0.5*math.pi)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_gamma_A_min, gamma_A=gamma_A_min),
            ac.T_min)
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_min, T=ac.T_min), V_A_gamma_A_min)

        # Get max. sustainable "dive" angle and corresponding
        # max. airspeed at zero thrust while pulling-up
        gamma_A_min, V_A_gamma_A_min = ac.climb_angle_min(T=ac.T_min, n=1.6)
        self.assertLess(gamma_A_min, 0)
        self.assertGreater(gamma_A_min, -0.5*math.pi)
        self.assertAlmostEqual(V_A_gamma_A_min, ac.V_A_max)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_gamma_A_min, gamma_A=gamma_A_min, n=1.6),
            ac.T_min)

        # TODO: Add test with power instead of thrust

    def test_climb_angle_max(self):
        ac = ExtDubinsAirplane(
            m=4,
            S=0.98,
            C_D0=0.032,
            # C_L0=0.2,
            C_L0=0,
            C_L_min=-1.2,
            C_L_max=1.2,
            k=0.06,
            T_min=0,
            T_max=25,
            P_max=5000,
            n_min=-0.5,
            n_max=2,
            mu_A_max=math.radians(60),
            V_A_max=30)

        # Get max. sustainable climb angle and corresponding
        # airspeed at max. thrust
        gamma_A_max, V_A_gamma_A_max = ac.climb_angle_max()
        # Should be climbing
        self.assertGreater(gamma_A_max, 0)
        self.assertLess(gamma_A_max, math.radians(45))
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_gamma_A_max, gamma_A=gamma_A_max),
            ac.T_max)
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_max, T=ac.T_max),
            V_A_gamma_A_max,
            places=1)

        # Get max. sustainable angle and corresponding
        # airspeed at max. power. The P_max constraint is "activated" here
        # by decreasing it below the value required for max. climb angle at
        # T_max as calculated above.
        P_max_reduced = 0.9*(V_A_gamma_A_max*ac.T_max)
        gamma_A_max, V_A_gamma_A_max = ac.climb_angle_max(P_max=P_max_reduced)

        # Should be climbing
        self.assertGreater(gamma_A_max, 0)
        # self.assertLess(gamma_A_max, math.radians(45))
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_max, P=P_max_reduced),
            V_A_gamma_A_max,
            places=3)
        self.assertAlmostEqual(
            ac.power_required(V_A=V_A_gamma_A_max, gamma_A=gamma_A_max),
            P_max_reduced,
            places=3)

        # Get max. climb angle and corresponding airspeed at zero thrust,
        # i.e. min. glide-path angle
        gamma_A_min, V_A_gamma_A_min = ac.climb_angle_max(T_max=ac.T_min)
        # Should be descending
        self.assertLess(gamma_A_min, 0)
        self.assertGreater(gamma_A_min, -0.5*math.pi)
        self.assertAlmostEqual(
            ac.thrust_required(V_A=V_A_gamma_A_min, gamma_A=gamma_A_min),
            ac.T_min)
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_min, T=ac.T_min),
            V_A_gamma_A_min,
            places=3)

        # Get min. glide-path angle while pulling up
        n_pull = 1.6
        gamma_A_min_pull, V_A_gamma_A_min_pull = ac.climb_angle_min(
            T=ac.T_min, n=n_pull)
        self.assertLess(gamma_A_min_pull, 0)
        self.assertGreater(gamma_A_min_pull, -0.5*math.pi)
        self.assertLess(gamma_A_min_pull, gamma_A_min)
        self.assertAlmostEqual(
            ac.thrust_required(
                V_A=V_A_gamma_A_min_pull, gamma_A=gamma_A_min_pull, n=n_pull),
            ac.T_min)
        self.assertAlmostEqual(
            ac.airspeed(gamma_A=gamma_A_min_pull, T=ac.T_min, n=n_pull),
            V_A_gamma_A_min_pull)

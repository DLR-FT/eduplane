# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.horizontal_profiles.linear_speed_straight_flight.

Run from top-level package directory:

   python3 -m unittest tests.traj.horizontal_profiles.test_linear_speed_straight_flight

"""

import math
import unittest

from eduplane.traj.horizontal_profiles.linear_speed_straight_flight import *


class LinearSpeedStraightFlightTest(unittest.TestCase):

    def test_tf_from_l_xy_tf(self):
        # 100 m with 10 m/s airspeed -> 10 s
        tf_no_wind_no_acc = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=10,
            V_A_xy_dot=0,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(tf_no_wind_no_acc, 10.0)
        # 100 m with 5-15 m/s airspeed (10 m/s mean) -> 10 s
        tf_no_wind_acc = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=5,
            V_A_xy_dot=1,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(tf_no_wind_acc, 10.0)
        # 100 m with 4-14 m/s airspeed (9 m/s mean) + 1 m/s tailwind -> 10 s
        tf_tailwind_acc = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=4,
            V_A_xy_dot=1,
            chi=0,
            V_W_xy=1,
            chi_W=0)
        self.assertAlmostEqual(tf_tailwind_acc, 10.0)
        # 100 m with 6-16 m/s airspeed (11 m/s mean) - 1 m/s headwind -> 10 s
        tf_headwind_acc = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=6,
            V_A_xy_dot=1,
            chi=math.pi,
            V_W_xy=1,
            chi_W=0)
        self.assertAlmostEqual(tf_headwind_acc, 10.0)
        # 100 m with 16-6 m/s airspeed (11 m/s mean) - 1 m/s headwind -> 10 s
        tf_headwind_acc = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=16,
            V_A_xy_dot=-1,
            chi=math.pi,
            V_W_xy=1,
            chi_W=0)
        self.assertAlmostEqual(tf_headwind_acc, 10.0)
        # Real-world case: consistency of both function variants
        tf_test1 = tf_from_l_xy_tf_V_A_xy_tf(
            l_xy_t0=0,
            l_xy_tf=50,
            V_A_xy_t0=15,
            V_A_xy_tf=35,
            chi=0,
            V_W_xy=5,
            chi_W=math.radians(45))
        V_A_xy_dot = (35-15)/tf_test1
        tf_test2 = tf_from_l_xy_tf(
            l_xy_t0=0,
            l_xy_tf=50,
            V_A_xy_t0=15,
            V_A_xy_dot=V_A_xy_dot,
            chi=0,
            V_W_xy=5,
            chi_W=math.radians(45))
        self.assertAlmostEqual(tf_test1, tf_test2)

    def test_tf_from_l_xy_tf_V_A_xy_tf_no_wind(self):
        # No wind and varying airspeed: d_l_xy = tf*(V_A_xy_t0+V_A_xy_tf)/2
        # 100 m with 5-15 m/s airspeed (10 m/s mean) -> 10 s
        tf_acc = tf_from_l_xy_tf_V_A_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=5,
            V_A_xy_tf=15,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(tf_acc, 10.0)
        # 100 m with 15-5 m/s airspeed (10 m/s mean) -> 10 s
        tf_dec = tf_from_l_xy_tf_V_A_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=15,
            V_A_xy_tf=5,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(tf_dec, 10.0)
        # Constant airspeed is the degenerate case
        tf_const = tf_from_l_xy_tf_V_A_xy_tf(
            l_xy_t0=0,
            l_xy_tf=100,
            V_A_xy_t0=10,
            V_A_xy_tf=10,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        self.assertAlmostEqual(tf_const, 10.0)

    def test_tf_from_l_xy_tf_V_A_xy_tf_with_wind(self):
        # With wind, the returned duration must reproduce the segment length
        # when fed back through tf_from_l_xy_tf.
        cases = [
            # (l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W)
            (50, 15, 35, 0, 5, math.radians(45)),
            (100, 6, 16, math.pi, 1, 0),
            (100, 16, 6, math.pi, 1, 0),
        ]
        for l_xy_tf, V_A_xy_t0, V_A_xy_tf, chi, V_W_xy, chi_W in cases:
            tf = tf_from_l_xy_tf_V_A_xy_tf(
                l_xy_t0=0,
                l_xy_tf=l_xy_tf,
                V_A_xy_t0=V_A_xy_t0,
                V_A_xy_tf=V_A_xy_tf,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            self.assertGreater(tf, 0)
            # Consistency with the V_A_xy_dot-parameterized variant
            V_A_xy_dot = (V_A_xy_tf-V_A_xy_t0)/tf
            tf_check = tf_from_l_xy_tf(
                l_xy_t0=0,
                l_xy_tf=l_xy_tf,
                V_A_xy_t0=V_A_xy_t0,
                V_A_xy_dot=V_A_xy_dot,
                chi=chi,
                V_W_xy=V_W_xy,
                chi_W=chi_W)
            self.assertAlmostEqual(tf, tf_check, places=6)

    def test_interface(self):
        # Implements a fully parametric horizontal trajectory segment:
        # 100 m with 5-15 m/s airspeed (10 m/s mean) -> 10 s
        seg = LinearSpeedStraightFlight(
            x_g_t0=0,
            y_g_t0=0,
            tf=10,
            V_A_xy_t0=5,
            V_A_xy_dot=1,
            chi=0,
            V_W_xy=0,
            chi_W=0)
        values = seg.evaluate(0)
        self.assertTupleEqual(tuple(values.keys()), HorizProfile.states)
        for k, v in values.items():
            self.assertFalse(math.isnan(v), msg=f"{k} is NaN")

        # Check initial and final values
        self.assertAlmostEqual(seg.t0, 0)
        self.assertAlmostEqual(seg.tf, 10)
        self.assertAlmostEqual(seg.x0["x_g"], 0)
        self.assertAlmostEqual(seg.x0["y_g"], 0)
        self.assertAlmostEqual(seg.xf["x_g"], 100)
        self.assertAlmostEqual(seg.xf["y_g"], 0)

        # Check result of discretize vs. evaluate at t0, tf
        df = seg.discretize().df
        self.assertListEqual(
            list(df.iloc[0].to_dict().keys()), list(seg.x0.keys()))
        self.assertDictEqual(df.iloc[0].to_dict(), seg.x0)
        self.assertDictEqual(df.iloc[-1].to_dict(), seg.xf)

        # Initial time and states can be shifted
        seg.shift(10, x_g=10, y_g=20)
        self.assertAlmostEqual(seg.t0, 10)
        self.assertAlmostEqual(seg.tf, 20)
        self.assertAlmostEqual(seg.x0["x_g"], 10)
        self.assertAlmostEqual(seg.x0["y_g"], 20)
        self.assertAlmostEqual(seg.xf["x_g"], 110)
        self.assertAlmostEqual(seg.xf["y_g"], 20)

    def test_init(self):
        # Linear speed straight flight segment:
        # 100 m with 5-15 m/s airspeed (10 m/s mean) -> 10 s

        # Initialize from V_A_xy_t0 and V_A_xy_dot (default):
        seg1 = LinearSpeedStraightFlight(
            tf=10, V_A_xy_t0=5, V_A_xy_dot=1, chi=0, V_W_xy=0, chi_W=0)
        # Initialize from V_A_xy_t0 and V_A_xy_tf:
        seg2 = LinearSpeedStraightFlight.init_from(
            tf=10, V_A_xy_t0=5, V_A_xy_tf=15, chi=0, V_W_xy=0, chi_W=0)
        # Initialize from V_A_xy_t0, chi_A_t0 and V_A_xy_dot
        seg3 = LinearSpeedStraightFlight.init_from(
            l_xy_tf=100,
            V_A_xy_t0=5,
            V_A_xy_dot=1,
            chi_A_t0=0,
            V_W_xy=0,
            chi_W=0)
        # Initialize from V_A_xy_t0, V_A_xy_tf and chi_A_t0
        seg4 = LinearSpeedStraightFlight.init_from(
            l_xy_tf=100,
            V_A_xy_t0=5,
            V_A_xy_tf=15,
            chi_A_t0=0,
            V_W_xy=0,
            chi_W=0)
        # Initialize from V_K_xy_t0, V_K_xy_tf and chi
        seg5 = LinearSpeedStraightFlight.init_from(
            l_xy_tf=100, V_K_xy_t0=5, V_K_xy_tf=15, chi=0, V_W_xy=0, chi_W=0)
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg2.items()):
            self.assertAlmostEqual(v1, v2, msg=f"fields: {k1}, {k2}")
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg3.items()):
            self.assertAlmostEqual(v1, v2, msg=f"fields: {k1}, {k2}")
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg4.items()):
            self.assertAlmostEqual(v1, v2, msg=f"fields: {k1}, {k2}")
        for (k1, v1), (k2, v2) in zip(seg1.items(), seg5.items()):
            self.assertAlmostEqual(v1, v2, msg=f"fields: {k1}, {k2}")

        # Real-world case
        seg6 = LinearSpeedStraightFlight.init_from(
            x_g_t0=100,
            y_g_t0=100,
            l_xy_tf=302.88381268363815,
            V_A_xy_t0=25,
            V_A_xy_tf=25.11036739934959,
            chi=-0.9979106771768852,
            V_W_xy=5,
            chi_W=0.7853981633974483,
            t0=0)
        self.assertAlmostEqual(seg6.l_xy_tf, 302.88381268363815)
        self.assertAlmostEqual(
            math.sqrt((seg6.xf["x_g"] - 100)**2 + (seg6.xf["y_g"] - 100)**2),
            302.88381268363815)
        # Log from failed attempt (final length is < 302.88)
        # Greedy initialization of LinearSpeedStraightFlight: 1st run
        # Greedy initialization of LinearSpeedStraightFlight: 2nd run with default values {'x_g_t0': 0, 'y_g_t0': 0, 't0': 0, 'l_xy_t0': 0, 'chi_W': 0, 'V_W_xy': 0, 'eps': 1e-06}
        # Could not initialize field tf from init method tf_from_l_xy_tf with signature (l_xy_t0, l_xy_tf, V_A_xy_t0, V_A_xy_dot, chi, V_W_xy, chi_W, eps): missing a required argument: 'V_A_xy_dot'
        # Could not initialize field V_A_xy_dot from init method V_A_xy_dot_from_V_A_xy_t0_tf with signature (t0, tf, V_A_xy_t0, V_A_xy_tf): missing a required argument: 'tf'
        # Initialized field V_A_xy_dot to value 0.008570146819480944 from {'l_xy_t0': 0, 'l_xy_tf': np.float64(302.88381268363815), 'V_A_xy_t0': 25, 'V_A_xy_tf': np.float64(25.11036739934959), 'chi': np.float64(-0.9979106771768852), 'V_W_xy': 5, 'chi_W': 0.7853981633974483} using V_A_xy_dot_from_l_xy_tf
        # Initialized field tf to value 12.878111799066087 from {'l_xy_t0': 0, 'l_xy_tf': np.float64(302.88381268363815), 'V_A_xy_t0': 25, 'V_A_xy_dot': np.float64(0.008570146819480944), 'chi': np.float64(-0.9979106771768852), 'V_W_xy': 5, 'chi_W': 0.7853981633974483, 'eps': 1e-06} using tf_from_l_xy_tf

    def test_no_wind(self):
        # Linear segment going from (-100, -100) to (100, 100) in 10 seconds
        # with average speed V_A_xy_avg=l_xy_tf/10 and initial speed
        # V_A_xy_t0=V_A_xy_avg-5
        x_g_t0 = -100
        y_g_t0 = -100
        l_xy_tf = math.sqrt(2*200**2)
        tf = 10
        V_A_xy_avg = l_xy_tf/tf
        V_A_xy_t0 = V_A_xy_avg - 5
        V_A_xy_tf = V_A_xy_avg + 5
        V_A_xy_dot = (V_A_xy_tf-V_A_xy_t0)/tf
        chi = math.radians(45)
        # Initialize from initial/final airspeed
        lin_seg_from_V_A_xy_t0_tf = LinearSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy_t0=V_A_xy_t0,
            V_A_xy_tf=V_A_xy_tf,
            chi=chi,
            V_W_xy=0,
            chi_W=0)
        # Initialize from initial airspeed and airspeed rate
        lin_seg_from_V_A_xy_dot = LinearSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy_t0=V_A_xy_t0,
            V_A_xy_dot=V_A_xy_dot,
            chi=chi,
            V_W_xy=0,
            chi_W=0)
        for lin_seg in [lin_seg_from_V_A_xy_t0_tf, lin_seg_from_V_A_xy_dot]:
            self.assertAlmostEqual(lin_seg.tf, tf)
            self.assertAlmostEqual(lin_seg.l_xy_tf, l_xy_tf)
            self.assertAlmostEqual(lin_seg.chi, chi)
            self.assertAlmostEqual(lin_seg.V_A_xy_dot, V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy_dot(t=0), V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy_dot(t=tf), V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy(t=0), V_A_xy_t0)
            self.assertAlmostEqual(lin_seg.V_K_xy(t=tf), V_A_xy_tf)
            self.assertAlmostEqual(lin_seg.V_A_xy(t=0), V_A_xy_t0)
            self.assertAlmostEqual(lin_seg.V_A_xy(t=tf), V_A_xy_tf)
            self.assertAlmostEqual(lin_seg.chi_A(t=0), chi)
            self.assertAlmostEqual(lin_seg.x_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.y_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.x_g(t=tf), 100)
            self.assertAlmostEqual(lin_seg.y_g(t=tf), 100)

    def test_with_wind(self):
        # Linear segment going from (-100, -100) to (100, 100) in 10 seconds
        # with average speed V_A_xy_avg=l_xy_tf/10 and initial speed
        # V_A_xy_t0=V_A_xy_avg-5 with head wind V_W_xy=7
        x_g_t0 = -100
        y_g_t0 = -100
        l_xy_tf = math.sqrt(2*200**2)
        tf = 10
        V_K_xy_avg = l_xy_tf/tf
        V_A_xy_avg = V_K_xy_avg + 7
        V_A_xy_t0 = V_A_xy_avg - 5
        V_A_xy_tf = V_A_xy_avg + 5
        V_A_xy_dot = (V_A_xy_tf-V_A_xy_t0)/tf
        chi = math.radians(45)
        V_W_xy = 7
        chi_W = chi + math.pi
        # Initialize from initial/final airspeed
        lin_seg_from_V_A_xy_t0_tf = LinearSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy_t0=V_A_xy_t0,
            V_A_xy_tf=V_A_xy_tf,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        # Initialize from initial airspeed and airspeed rate
        lin_seg_from_V_A_xy_dot = LinearSpeedStraightFlight.init_from(
            x_g_t0=x_g_t0,
            y_g_t0=y_g_t0,
            tf=tf,
            V_A_xy_t0=V_A_xy_t0,
            V_A_xy_dot=V_A_xy_dot,
            chi=chi,
            V_W_xy=V_W_xy,
            chi_W=chi_W)
        for lin_seg in [lin_seg_from_V_A_xy_t0_tf, lin_seg_from_V_A_xy_dot]:
            self.assertAlmostEqual(lin_seg.tf, tf)
            self.assertAlmostEqual(lin_seg.l_xy_tf, l_xy_tf)
            self.assertAlmostEqual(lin_seg.chi, chi)
            self.assertAlmostEqual(lin_seg.V_A_xy_dot, V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy_dot(t=0), V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy_dot(t=tf), V_A_xy_dot)
            self.assertAlmostEqual(lin_seg.V_K_xy(t=0), V_A_xy_t0 - V_W_xy)
            self.assertAlmostEqual(lin_seg.V_K_xy(t=tf), V_A_xy_tf - V_W_xy)
            self.assertAlmostEqual(lin_seg.V_A_xy(t=0), V_A_xy_t0)
            self.assertAlmostEqual(lin_seg.V_A_xy(t=tf), V_A_xy_tf)
            self.assertAlmostEqual(lin_seg.chi_A(t=0), chi)
            self.assertAlmostEqual(lin_seg.x_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.y_g(t=0), -100)
            self.assertAlmostEqual(lin_seg.x_g(t=tf), 100)
            self.assertAlmostEqual(lin_seg.y_g(t=tf), 100)

    def test_realworld_examples(self):
        # "Real-world" example with cross-wind component
        # chi=-39.81539389815609°
        # chi_W=135.61903014790633
        # => ca. 175° difference, i.e. head-wind conditions
        # Approximate solution derived manually from plot: V_A_dot=-0.606635
        lin_seg = LinearSpeedStraightFlight.init_from(
            l_xy_tf=341.0183036974966,
            V_A_xy_t0=33.02144276779774,
            V_A_xy_tf=15.53606797109165,
            chi=-0.6949097165012835,
            V_W_xy=12.465841243242233,
            chi_W=2.3669986044424176)
        self.assertAlmostEqual(
            lin_seg.x0["V_A_xy"], 33.02144276779774, places=3)
        self.assertAlmostEqual(
            lin_seg.xf["V_A_xy"], 15.53606797109165, places=3)

        # Another "real-world" example
        lin_seg = LinearSpeedStraightFlight.init_from(
            l_xy_tf=641.6370930840534,
            V_A_xy_t0=19.086751817062147,
            V_A_xy_tf=19.75205572747579,
            chi=-2.551817570681724,
            V_W_xy=14.998231714891412,
            chi_W=0.7753556710008676)
        self.assertAlmostEqual(
            lin_seg.x0["V_A_xy"], 19.086751817062147, places=3)
        self.assertAlmostEqual(
            lin_seg.xf["V_A_xy"], 19.75205572747579, places=3)

    def test_kinematic_consistency(self):

        # "Real-world" example from trochoid path solver tests
        # traj = Trajectory(segments=[SteadyTurn(chi_t0=0.14878748865176306, tf=3.7276002709633755, V_A_xy=21.117468438102286, x_g_t0=-233.8286180776584, y_g_t0=80.1260411099924, t0=0, l_xy_t0=0, chi_A_dot=-0.10566240704382544, chi_W=0.7853981633974483, V_W_xy=5, x_g_M_t0=-232.31679216577004, y_g_M_t0=-119.72615638756844, V_K_xy_t0=24.92781191771101, chi_A_t0=0.007564575690109765, chi_dot_t0=-0.08862024712307509, u_W_g=3.5355339059327378, v_W_g=3.5355339059327373, V_A_xy_dot=0), LinearSpeedStraightFlight(tf=13.93549314968347, V_A_xy_t0=21.117468438102286, chi=-0.189116181909351, x_g_t0=-143.8380468115753, y_g_t0=78.58297237785541, t0=3.7276002709633755, l_xy_t0=90.43260898005396, V_A_xy_dot=-0.689775854848547, chi_W=0.7853981633974483, V_W_xy=5, eps=1e-06, chi_A_t0=-0.38630264143709636, V_K_xy_t0=23.516096824550136, V_K_xy_tf=16.26245971306279, V_A_xy_tf=14.076310122161502, l_xy_tf=203.18019257301756, u_W_g=3.5355339059327378, v_W_g=3.5355339059327373, chi_dot=0), SteadyTurn(chi_t0=-0.189116181909351, tf=17.415055950339752, V_A_xy=14.076310122161502, x_g_t0=55.71959906470602, y_g_t0=40.38694344387827, t0=13.93549314968347, l_xy_t0=293.61280155307156, chi_A_dot=0.051661147696380576, chi_W=0.7853981633974483, V_W_xy=5, x_g_M_t0=183.33442705919407, y_g_M_t0=281.12837490933725, V_K_xy_t0=16.26245971306279, chi_A_t0=-0.4874294969120353, chi_dot_t0=0.0427414180250685, u_W_g=3.5355339059327378, v_W_g=3.5355339059327373, V_A_xy_dot=0)])
        seg = LinearSpeedStraightFlight(
            tf=13.93549314968347,
            V_A_xy_t0=21.117468438102286,
            chi=-0.189116181909351,
            x_g_t0=-143.8380468115753,
            y_g_t0=78.58297237785541,
            t0=3.7276002709633755,
            l_xy_t0=90.43260898005396,
            V_A_xy_dot=-0.689775854848547,
            chi_W=0.7853981633974483,
            V_W_xy=5,
            eps=1e-06,
            chi_A_t0=-0.38630264143709636,
            V_K_xy_t0=23.516096824550136,
            V_K_xy_tf=16.26245971306279,
            V_A_xy_tf=14.076310122161502,
            l_xy_tf=203.18019257301756,
            u_W_g=3.5355339059327378,
            v_W_g=3.5355339059327373)
        seg_discr = seg.discretize()
        seg_discr.check_kinematic_consistency(
            horizontal_continuity=2, vertical_continuity=2, raise_err=True)
        # import matplotlib.pyplot as plt
        # seg_discr.df[["V_A_xy", "V_A_xy_dot", "V_K_xy", "V_K_xy_dot"]].plot()
        # plt.show()

    # def test_time_diff(self):

    #     north = LinearSpeedStraightFlight.init_from(
    #         l_xy_tf=100, V_A_xy_t0=10, V_A_xy_dot=0, chi=0, V_W_xy=0, chi_W=0)
    #     north_acc = LinearSpeedStraightFlight.init_from(
    #         l_xy_tf=100, V_A_xy_t0=10, V_A_xy_dot=1, chi=0, V_W_xy=0, chi_W=0)
    #     print("north", north.tf)
    #     print("north_acc", north_acc.tf)

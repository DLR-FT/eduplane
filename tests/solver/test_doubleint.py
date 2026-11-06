# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.solver.doubleint.

Run from top-level package directory:

   python3 -m unittest tests.solver.test_doubleint

"""

import unittest

import pandas

from eduplane.solver.doubleint import *

DEBUG_PLOTS = False


# Integrate double int segment
def integrate_segment(t, x, v, u, t_stop, t_step=0.1, endpoint=True):
    """Helper function to integrate a single segment of a double-integrator
    trajectory."""
    t0 = t[-1]
    x0 = x[-1]
    v0 = v[-1]
    u_cmd = u[-1]
    if t_stop > t0:
        t_range = numpy.arange(t0 + t_step, t_stop, t_step)
        for t_i in t_range:
            dt = t_i - t0
            t.append(t_i)
            x.append(x0 + v0*dt + 0.5*u_cmd*dt**2)
            v.append(v0 + u_cmd*dt)
            u.append(u_cmd)
        if endpoint and t[-1] != t_stop:
            dt = t_stop - t0
            t.append(t_stop)
            x.append(x0 + v0*dt + 0.5*u_cmd*dt**2)
            v.append(v0 + u_cmd*dt)
            u.append(u_cmd)


# Integrate multi-segment trajectory
def integrate(problem, solution, t_step=None, t0=0, n_points=500):
    """Helper function to integrate a double-integrator trajectory."""
    if not t_step:
        t_step = max(solution.tf/n_points, 0.001)
    timeline = [t0]
    u_timeline = [solution.u0]
    v_timeline = [problem.v0]
    x_timeline = [problem.x0]
    # First segment
    if solution.t1 > 0:
        integrate_segment(
            timeline, x_timeline, v_timeline, u_timeline, solution.t1, t_step)
    # Middle segment
    if solution.t2 > solution.t1:
        u_timeline[-1] = 0
        integrate_segment(
            timeline, x_timeline, v_timeline, u_timeline, solution.t2, t_step)
    # Last segment
    if solution.tf > solution.t2:
        u_timeline[-1] = solution.u2
        integrate_segment(
            timeline, x_timeline, v_timeline, u_timeline, solution.tf, t_step)
    df = pandas.DataFrame(
        {
            "t": timeline,
            "x": x_timeline,
            "v": v_timeline,
            "u": u_timeline
        })
    df = df.set_index("t")
    return df


def discretize(problem, solution, t_array, t0=None, align_u=False):
    """Discretize a double integrator trajectory on a given time grid.

    The time grid may be larger than the double int. solution in which case
    the profile is padded with zero velociy and acceleration segments depending
    on the the 'align' parameter. Care has to taken that the boundary constraints
    at the padded end of the trajectory are zero.
    """
    # Find t0, tf
    if t0 is None:
        t0 = t_array.values[0]
    tf = t0 + solution.tf
    # Note that index value on the grid (t0_idx and tf_idx) may not exactly
    # match t0 and tf. The error is accounted for with t0_eps and tf_eps.
    t0_idx = t_array.get_loc(t0, method="backfill")  # exact match or next
    tf_idx = t_array.get_loc(tf, method="pad")  # exact match or previous
    t0_eps = t_array[t0_idx] - t0
    tf_eps = t_array[tf_idx] - tf
    # Check if alignment will produce valid results
    if t0 != t_array.values[0] and problem.v0 != 0:
        raise ValueError(
            "Right alignment of vertical profile "
            "with starting velocity != 0")
    if tf != t_array.values[-1] and problem.vg != 0:
        raise ValueError(
            "Left alignment of vertical profile "
            "with final velocity != 0")
    # Integrate segments first to get border points
    borders = integrate(problem, solution, 1e6, 0)
    # print("borders\n{}".format(borders))
    timeline = []
    u_timeline = []
    v_timeline = []
    x_timeline = []
    for t in t_array[:t0_idx]:
        # print("front padding at t={}".format(t))
        timeline.append(t)
        u_timeline.append(0)
        v_timeline.append(0)
        x_timeline.append(borders.iloc[0].x)
    for t in t_array[t0_idx:tf_idx + 1]:
        seg_t0 = borders.index.asof(t - t0)
        x0 = borders.loc[seg_t0].x
        v0 = borders.loc[seg_t0].v
        u_cmd = borders.loc[seg_t0].u
        dt = t - t0 - seg_t0
        timeline.append(t)
        x_timeline.append(x0 + v0*dt + 0.5*u_cmd*dt**2)
        v_timeline.append(v0 + u_cmd*dt)
        u_timeline.append(u_cmd)
    for t in t_array[tf_idx + 1:]:
        # print("back padding at t={}".format(t))
        timeline.append(t)
        u_timeline.append(0)
        v_timeline.append(0)
        x_timeline.append(borders.iloc[-1].x)
    df = pandas.DataFrame(
        {
            "t": timeline,
            "x": x_timeline,
            "v": v_timeline,
            "u": u_timeline
        })
    # print("index: ",len(t_array), "timeline: ",len(timeline))
    # This will align the input 'u' with gamma/w which may be helpful to avoid
    # spikes in numerically derived quantities
    if align_u:
        df["u"] = df["u"].shift(1)
        df["u"].iloc[0] = 0
    df = df.set_index("t")
    return df


class DoubleIntTests(unittest.TestCase):

    def test_doubleint_topt(self):
        # No saturation:
        # From x0=-10, v0=-4 to xg=10, vg=4 in 20 seconds
        problem = FreeFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_topt(problem)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, 2)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -4)

        # With saturation:
        # From x0=-10, v0=-4 to xg=10, vg=4 in 20 seconds
        problem = FreeFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            v_min=-5,
            v_max=5,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_topt(problem)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, -5)
        self.assertLessEqual(solution.v1, 5)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, 2)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -4)

        # With saturation and different constraints for first and last segment:
        # From x0=-10, v0=-5 to xg=10, vg=5
        problem = FreeFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            v_min=-5,
            v_max=5,
            u0_min=-4,
            u0_max=2,
            u2_min=-2,
            u2_max=1)
        solutions = doubleint_topt(problem)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, -5)
        self.assertLessEqual(solution.v1, 5)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, 2)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -2)

        # Test special case: all negative acceleration bounds -> just maintain
        # constant vertical speed
        problem = FreeFinalTimeProblem(
            x0=-29.595606691438608,
            v0=-6.221820758620558,
            xg=-119.93908865165672,
            vg=-6.221820758620558,
            v_max=0.6798970895318142,
            v_min=-6.2218207586205585,
            u0_max=-0.3447045872455626,
            u0_min=-3.3008816099382603,
            u2_max=-0.3447045872455626,
            u2_min=-3.3008816099382603)
        solutions = doubleint_topt(problem)
        self.assertGreaterEqual(solution.v1, -5)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Another special case: initial airspeed violates constraints
        problem = FreeFinalTimeProblem(
            x0=39.9603131718545,
            v0=5.073441024855529,
            xg=54.55817482443385,
            vg=-4.725725987618894,
            v_max=2.93578569046999,
            v_min=-9.455455847030183,
            u0_max=1.795973150010571,
            u0_min=-2.0720183780491266,
            u2_max=1.5555320841370985,
            u2_min=-2.2288381791772682)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # One more..
        problem = FreeFinalTimeProblem(
            x0=39.9603131718545,
            v0=5.073441024855529,
            xg=54.55817482443385,
            vg=-4.725725987618894,
            v_max=2.93578569046999,
            v_min=-9.455455847030183,
            u0_max=1.795973150010571,
            u0_min=-2.0720183780491266,
            u2_max=1.5555320841370985,
            u2_min=-2.2288381791772682)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Another variant of this special case: initial airspeed violates constraints,
        # however, a solution would exist with zero-length first segment, such that
        # the feasible speed range is not reached and the violating speed value is
        # maintained unti t2.
        problem = FreeFinalTimeProblem(
            x0=43.655065898871555,
            v0=5.081846175853661,
            xg=51.99605554511534,
            vg=1.5975928459037778,
            v_max=4.794633188152802,
            v_min=-11.184177034727186,
            u0_max=1.375893981766147,
            u0_min=-2.1173007665406898,
            u2_max=1.627703133376958,
            u2_min=-1.9841972889460828)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # One more (boy, this is tricky..)
        problem = FreeFinalTimeProblem(
            x0=48.868803963690134,
            v0=1.4445809690810414,
            xg=44.65800222671123,
            vg=7.401358253894093,
            v_max=2.8771942434824433,
            v_min=-10.513472344867692,
            u0_max=1.8079676343393796,
            u0_min=-1.9879366808898797,
            u2_max=1.1243816113941072,
            u2_min=-2.2319501961918484)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Special case with non-symmetric constraints:
        # Segment #3 has all negativ acceleration bounds, such that we would
        # have to accelerate in first segment to increase from initial to
        # final speed.
        problem = FreeFinalTimeProblem(
            x0=83.53702183369572,
            v0=-5.794078588476582,
            xg=14.87968244059391,
            vg=-4.352946736470519,
            v_max=1.800035565944111,
            v_min=-9.411188582530778,
            u0_max=0.13965684200485498,
            u0_min=-2.4306752032388363,
            u2_max=-0.17814893022333494,
            u2_min=-2.2840758401842125)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Another one like that...
        problem = FreeFinalTimeProblem(
            x0=92.0485696188865,
            v0=-0.05430607912581298,
            xg=51.84730952746097,
            vg=-2.6642871247185,
            v_max=0.9106952455532686,
            v_min=-7.366278703330069,
            u0_max=0.72256255715938,
            u0_min=-1.9620468425460562,
            u2_max=-0.3477658120406597,
            u2_min=-2.179468376959244)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # One more..
        problem = FreeFinalTimeProblem(
            x0=81.67790974990199,
            v0=-5.50953024569227,
            xg=45.92297294353411,
            vg=-3.9909395913582126,
            v_max=2.0213588079814717,
            v_min=-9.708787663008204,
            u0_max=0.5046401930101213,
            u0_min=-2.299410239802356,
            u2_max=-0.022788481536480634,
            u2_min=-2.2121142270605247)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Test case where the target state was not reached
        problem = FreeFinalTimeProblem(
            x0=21.073734647921803,
            v0=-3.4172975221895965,
            xg=13.323800274712616,
            vg=1.7120589971838243,
            v_max=0.7976441515041996,
            v_min=-6.884561843839691,
            u0_max=1.3434274713914927,
            u0_min=-2.364463415998721,
            u2_max=1.457008480379611,
            u2_min=-1.9813160244370092)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)

        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Special case: final speed violates constraint
        problem = FreeFinalTimeProblem(
            x0=2.484431796907227,
            v0=-6.904565257024043,
            xg=48.26570540998474,
            vg=6.893013111564119,
            v_max=3.103025970463779,
            v_min=-9.842919812858574,
            u0_max=1.1393836365246974,
            u0_min=-2.4922592136336505,
            u2_max=0.9286756424306041,
            u2_min=-2.1730507408334607)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)

        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Special case: Asymmetric constraints require different inputs
        # than "regular" time-optimal inputs.
        # This may happen when start and goal position are relatively close
        # compared to initial/final velocity difference and accelerations.
        problem = FreeFinalTimeProblem(
            x0=48.38926124630181,
            v0=-3.884637605573936,
            xg=56.44844080906538,
            vg=5.51127186188037,
            v_max=9.930247037229563,
            v_min=-9.492951623542526,
            u0_max=0.8375768674483665,
            # u0_max=1.1955407150486523,
            u0_min=-2.0270805871226845,
            u2_max=1.1955407150486523,
            u2_min=-2.1207006158788513)
        solutions = doubleint_topt(problem)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        # Another "real-world" special case
        problem = FreeFinalTimeProblem(
            x0=-60.370491880107174,
            v0=0,
            xg=-95.55187949545731,
            vg=-9.630280646186472,
            v_max=9.653509871957212,
            v_min=-10.685461474045347,
            u0_max=8.989502781722635,
            u0_min=-3.30119402843335,
            u2_max=8.022168275209403,
            u2_min=-1.965418937606159)
        solutions = doubleint_topt(problem, verbose=3)
        self.assertGreater(len(solutions), 0)
        dt1 = solutions[0].t1
        dt2 = solutions[0].t2 - solutions[0].t1
        dt3 = solutions[0].tf - solutions[0].t2
        self.assertAlmostEqual(
            problem.v0 + dt1*solutions[0].u0 + dt3*solutions[0].u2, problem.vg)
        self.assertAlmostEqual(
            problem.x0 + dt1*(problem.v0 + 0.5*solutions[0].u0*dt1) +
            dt2*solutions[0].v1 + dt3*
            (solutions[0].v1 + 0.5*solutions[0].u2*dt3), problem.xg)
        # solution = solutions[0]
        # discr_solution = integrate(problem, solution)
        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

    def test_doubleint_fixed(self):
        # No saturation:
        # From x0=-10, v0=-5 to xg=10, vg=5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, 2)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -4)
        # return

        # For fixed-time, saturation will necessarily result in no solution:
        # From x0=-10, v0=-5 to xg=10, vg=5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-2.5,
            v_max=2.5,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_fixed(problem, verbose=INFO)

        # if DEBUG_PLOTS:
        #     solution = solutions[0]
        #     discr_solution = integrate(problem, solution)
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertEqual(len(solutions), 0)

        # No saturation and different constraints for first and last segment:
        # From x0=-10, v0=-5 to xg=10, vg=-5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-2,
            u2_max=1)
        solutions = doubleint_fixed(problem)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, 2)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -2)

        # Same sign accelerations, i.e. accelerate towards goal state,
        # stay constant for a while to reach final time, finish accelerating.
        # From x0=-10, v0=-5 to xg=10, vg=-5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=5,
            xg=-20,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-2,
            u2_max=1)
        solutions = doubleint_fixed(problem)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()
        # return

        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, 5)
        self.assertAlmostEqual(discr_solution.iloc[0].u, -4)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, -20)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, -2)

        # "real-world" test case:
        problem = FixedFinalTimeProblem(
            x0=-341.63816130680874,
            v0=1.319030259330269,
            xg=-309.99785502212956,
            vg=-4.554010929291562,
            tf=59.4158374026035,
            v_max=10.770444218749985,
            v_min=-11.20539636484804,
            u0_max=9.29781409315697,
            u0_min=-2.886719665509498,
            u2_max=8.882531849086174,
            u2_min=-2.4363711891361692)

        # # NOTE: There exists a time-optimal solution!
        # topt_problem = FreeFinalTimeProblem(
        #     **{k: v
        #        for k, v in problem.asdict()().items() if k != "tf"})
        # topt_problem = topt_problem._replace(v_max=0.62)
        # topt_solutions = doubleint_topt(topt_problem, verbose=INFO)

        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[0].u, solution.u0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, solution.u2)

    def test_doubleint_fixed_special_cases(self):

        # =====================================================================
        # Another "real-world" test case
        # NOTE: There exists a time-optimal solution with tf ~= 8s, but no
        # fixed-time solution for tf ~= 40s. Solutions start to exist again
        # at around tf >= 122s.
        problem = FixedFinalTimeProblem(
            x0=-192.39559980440825,
            v0=22.94692142223775,
            xg=0.0,
            vg=2.678965157774523,
            tf=44.38802438584392,
            v_max=28.279771060738877,
            v_min=-7.450389096532,
            u0_max=5.923385671206013,
            u0_min=-0.42145273999443267,
            u2_max=9.255564944981073,
            u2_min=-5.50709301372847)

        # NOTE: There exists a time-optimal solution!
        topt_problem = FreeFinalTimeProblem(
            **{
                k: v
                for k, v in problem.asdict().items() if k != "tf"
            })
        topt_solutions = doubleint_topt(topt_problem, verbose=INFO)
        self.assertGreater(len(topt_solutions), 0)

        discr_solutions = [
            integrate(topt_problem, topt_solution)
            for topt_solution in topt_solutions
        ]
        if DEBUG_PLOTS:
            from eduplane.utils import plotting
            plotting.plot_doubleint_multi(
                [topt_problem]*len(discr_solutions), discr_solutions)
            plotting.show()

        for topt_solution in topt_solutions:
            discr_solution = integrate(topt_problem, topt_solution)
            if DEBUG_PLOTS:
                from eduplane.utils import plotting
                plotting.plot_doubleint(topt_problem, discr_solution)
                plotting.show()
            self.assertGreaterEqual(topt_solution.v1, problem.v_min)
            self.assertLessEqual(topt_solution.v1, problem.v_max)
            self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
            self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
            if topt_solution.t1 > 0:
                self.assertAlmostEqual(
                    discr_solution.iloc[0].u, topt_solution.u0)
            self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
            self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
            if topt_solution.tf > topt_solution.t2:
                self.assertAlmostEqual(
                    discr_solution.iloc[-1].u, topt_solution.u2)

        # No solution exists
        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 0)

        # =====================================================================
        # Another "real-world" test case
        # NOTE: There exists a time-optimal solution with tf ~= 8s, but no
        # fixed-time solution for tf ~= 40s. Solutions start to exist again
        # at around tf >= 122s.
        problem = FixedFinalTimeProblem(
            x0=-192.39559980440825,
            v0=22.94692142223775,
            xg=0.0,
            vg=2.678965157774523,
            tf=44.38802438584392,
            v_max=28.279771060738877,
            v_min=-7.450389096532,
            u0_max=5.923385671206013,
            u0_min=-0.42145273999443267,
            u2_max=9.255564944981073,
            u2_min=-5.50709301372847)

        # NOTE: There exists a time-optimal solution!
        topt_problem = FreeFinalTimeProblem(
            **{
                k: v
                for k, v in problem.asdict().items() if k != "tf"
            })
        topt_solutions = doubleint_topt(topt_problem, verbose=INFO)
        self.assertGreater(len(topt_solutions), 0)

        for topt_solution in topt_solutions:
            discr_solution = integrate(topt_problem, topt_solution)
            if DEBUG_PLOTS:
                # print("#### PLOTTING TIME-OPT SOLUTION", topt_solution)
                # discr_solution = integrate(topt_problem, topt_solutions[0])
                # import matplotlib.pyplot as plt
                # discr_solution.plot()
                import matplotlib.pyplot as plt
                discr_solution.plot()
                plt.show()
            self.assertGreaterEqual(topt_solution.v1, problem.v_min)
            self.assertLessEqual(topt_solution.v1, problem.v_max)
            self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
            self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
            if topt_solution.t1 > 0:
                self.assertAlmostEqual(
                    discr_solution.iloc[0].u, topt_solution.u0)
            self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
            self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
            if topt_solution.tf > topt_solution.t2:
                self.assertAlmostEqual(
                    discr_solution.iloc[-1].u, topt_solution.u2)

        # No solution exists
        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 0)

        # =====================================================================
        # Another "real-world" test case
        # NOTE: this one failed without the tolerance checks for v1 against v_min/v_max,
        # there is a fixed-time solution for tf >= 60.56946499745869
        problem = FixedFinalTimeProblem(
            x0=-150,
            v0=-0.0,
            xg=-100,
            vg=-0.0,
            # tf=211.38849462596178,
            tf=60.56946499745869,
            v_max=0.8259829696103534,
            v_min=-6.093997554022512,
            u0_max=19.610190000000003,
            u0_min=-28.542764438446035,
            u2_max=19.610190000000003,
            u2_min=-28.542764438446035)

        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 1)

        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min - 1e-3)
        self.assertLessEqual(solution.v1, problem.v_max + 1e-3)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[0].u, solution.u0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, solution.u2)

        # =====================================================================
        # Another "real-world test case"
        # NOTE: this one failed without the tolerance checks for v1 against v_min/v_max,
        # there is a fixed-time solution for tf >= 60.56946499745869
        problem = FixedFinalTimeProblem(
            x0=-100,
            v0=-0.0,
            xg=-150,
            vg=-0.0,
            tf=9.913965699937076,
            v_max=0.7845078910712124,
            v_min=-5.534576665372364,
            u0_max=-2.7557574039500015,
            u0_min=-6.133237245947127,
            u2_max=6.478207106527825,
            u2_min=-4.986748084287777)
        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 1)

        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[0].u, solution.u0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, solution.u2)

        # =====================================================================
        # Another "real-world test case"
        # This one produced an invalid solution without the tolerance checks on
        # denominator (u0**2 + u2**2) in the doubleint_fixed solver.
        problem = FixedFinalTimeProblem(
            x0=-200,
            v0=-0.0,
            xg=-100,
            vg=-0.0,
            tf=30,
            v_max=2.5798072569951658,
            v_min=-6.281306269465928,
            u0_max=7.848000000000001,
            u0_min=-19.59057,
            u2_max=7.848,
            u2_min=-19.59057)
        solutions = doubleint_fixed(problem, verbose=INFO)
        self.assertEqual(len(solutions), 0)
        # Valid solution exists for tf >= 38.992
        solutions = doubleint_topt(problem, verbose=INFO)
        self.assertEqual(len(solutions), 1)

        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[0].u, solution.u0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)
        self.assertAlmostEqual(discr_solution.iloc[-1].u, solution.u2)

    def test_doubleint_fixed_min_acc(self):
        # No saturation:
        # From x0=-10, v0=-5 to xg=10, vg=5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_fixed_min_acc(problem, verbose=INFO)
        self.assertGreater(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        if DEBUG_PLOTS:
            from eduplane.utils import plotting
            plotting.plot_doubleint(problem, discr_solution)
            plotting.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)

        # For fixed-time, saturation will necessarily result in no solution:
        # From x0=-10, v0=-5 to xg=10, vg=5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-2.5,
            v_max=2.5,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_fixed_min_acc(problem, verbose=INFO)
        self.assertGreater(len(solutions), 0)

        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        if DEBUG_PLOTS:
            from eduplane.utils import plotting
            plotting.plot_doubleint(problem, discr_solution)
            plotting.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)

        # No saturation and different constraints for first and last segment:
        # From x0=-10, v0=-5 to xg=10, vg=-5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=-5,
            xg=10,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-2,
            u2_max=1)
        solutions = doubleint_fixed_min_acc(problem)
        self.assertGreater(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        if DEBUG_PLOTS:
            from eduplane.utils import plotting
            plotting.plot_doubleint(problem, discr_solution)
            plotting.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, -5)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, 10)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)

        # Same sign accelerations, i.e. accelerate towards goal state,
        # stay constant for a while to reach final time, finish accelerating.
        # From x0=-10, v0=-5 to xg=10, vg=-5 in 15 seconds
        problem = FixedFinalTimeProblem(
            x0=-10,
            v0=5,
            xg=-20,
            vg=-5,
            tf=15,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-2,
            u2_max=1)
        solutions = doubleint_fixed_min_acc(problem)
        self.assertGreater(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()
        # return

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, -10)
        self.assertAlmostEqual(discr_solution.iloc[0].v, 5)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, -20)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, -5)

        # "real-world" test case:
        problem = FixedFinalTimeProblem(
            x0=-341.63816130680874,
            v0=1.319030259330269,
            xg=-309.99785502212956,
            vg=-4.554010929291562,
            tf=59.4158374026035,
            v_max=10.770444218749985,
            v_min=-11.20539636484804,
            u0_max=9.29781409315697,
            u0_min=-2.886719665509498,
            u2_max=8.882531849086174,
            u2_min=-2.4363711891361692)

        solutions = doubleint_fixed_min_acc(problem, verbose=INFO)
        self.assertGreater(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        # if DEBUG_PLOTS:
        #     import matplotlib.pyplot as plt
        #     discr_solution.plot()
        #     plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)

        problem = FixedFinalTimeProblem(
            x0=-100,
            v0=-0.0,
            xg=-200,
            vg=-0.0,
            tf=30,
            v_max=2.5798072569951658,
            v_min=-9.695434031682657,
            u0_max=6.870208660415992,
            u0_min=-18.999123223666366,
            u2_max=6.870208660415992,
            u2_min=-18.999123223666366)

        solutions = doubleint_fixed_min_acc(problem, verbose=INFO)
        self.assertGreater(len(solutions), 1)
        solution = solutions[0]
        discr_solution = integrate(problem, solution)

        if DEBUG_PLOTS:
            # if True:
            import matplotlib.pyplot as plt
            discr_solution.plot()
            plt.show()

        self.assertGreaterEqual(solution.v1, problem.v_min)
        self.assertLessEqual(solution.v1, problem.v_max)
        self.assertAlmostEqual(discr_solution.iloc[0].x, problem.x0)
        self.assertAlmostEqual(discr_solution.iloc[0].v, problem.v0)
        self.assertAlmostEqual(discr_solution.iloc[-1].x, problem.xg)
        self.assertAlmostEqual(discr_solution.iloc[-1].v, problem.vg)

    def test_doubleint_free(self):
        # No saturation, case 1:
        # From v0=-4 to v1=10+4 in t1=7+2 with u0=2, from v1=14 to vg=10
        # in tf-t1=1 with u2=-4
        problem = FreeFinalPositionProblem(
            x0=10,
            v0=-4,
            u0=2,
            vg=10,
            tf=10,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_free(problem)
        self.assertEqual(solutions[0].u0, 2)
        self.assertEqual(solutions[0].u2, -4)
        self.assertEqual(solutions[0].t1, 9)
        self.assertEqual(solutions[0].t2, 9)
        self.assertEqual(solutions[0].tf, 10)
        self.assertEqual(solutions[0].v1, 14)

        # No saturation, case 2:
        # From v0=-4 to v1=4 in t1=4 with u0=2, from v1=4 to vg=-8 in tf-t1=3
        # with u2=-4
        problem = FreeFinalPositionProblem(
            x0=10,
            v0=-4,
            u0=2,
            vg=-8,
            tf=7,
            v_min=-100,
            v_max=100,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_free(problem)
        self.assertEqual(solutions[0].u0, 2)
        self.assertEqual(solutions[0].u2, -4)
        self.assertEqual(solutions[0].t1, 4)
        self.assertEqual(solutions[0].t2, 4)
        self.assertEqual(solutions[0].tf, 7)
        self.assertEqual(solutions[0].v1, 4)

        # With saturation, case 1:
        # From v0=-4 to v1=10+2 in t1=7+1 with u0=2, sat. at v1=12 for t2-t1=2,
        # from v1=12 to vg=10 in tf-t1=0.5 with u2=-4
        problem = FreeFinalPositionProblem(
            x0=10,
            v0=-4,
            u0=2,
            vg=10,
            tf=10.5,
            v_min=-100,
            v_max=12,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_free(problem)
        self.assertEqual(solutions[0].u0, 2)
        self.assertEqual(solutions[0].u2, -4)
        self.assertEqual(solutions[0].t1, 8)
        self.assertEqual(solutions[0].t2, 10)
        self.assertEqual(solutions[0].tf, 10.5)
        self.assertEqual(solutions[0].v1, 12)

        # With saturation, case 2:
        # From v0=-4 to v1=2 in t1=3 with u0=2, at v1=2 for t2-t1=2, from v1=2
        # to vg=-8 in tf-t1=2.5 with u2=-4
        problem = FreeFinalPositionProblem(
            x0=10,
            v0=-4,
            u0=2,
            vg=-8,
            tf=7.5,
            v_min=-100,
            v_max=2,
            u0_min=-4,
            u0_max=2,
            u2_min=-4,
            u2_max=2)
        solutions = doubleint_free(problem)
        self.assertEqual(solutions[0].u0, 2)
        self.assertEqual(solutions[0].u2, -4)
        self.assertEqual(solutions[0].t1, 3)
        self.assertEqual(solutions[0].t2, 5)
        self.assertEqual(solutions[0].tf, 7.5)
        self.assertEqual(solutions[0].v1, 2)

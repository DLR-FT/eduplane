# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.utils.htrace.

Run from top-level package directory:

   python3 -m unittest tests.utils.test_htrace

"""

import builtins
import time
import unittest
from unittest.mock import patch

import numpy

from eduplane.utils import utils
from eduplane.utils.htrace import *
from eduplane.utils.htrace import HTraceLogger


class HTraceTest(unittest.TestCase):

    def test_fold_traces(self):
        result = fold_traces(
            [
                {
                    "a": 1,
                    "duration": 1
                },
                {
                    "b": 2,
                    "duration": 2
                },
                {
                    "a": 1,
                    "duration": 3
                },
                {
                    "a": 2,
                    "duration": 4
                },
            ])
        # # for k, v in result.items():
        # #     print(f"{k}: {v}")
        # for t in result:
        #     print(f"{t}")
        self.assertAlmostEqual(result[0]["acc. time"], 3 + 1)
        self.assertAlmostEqual(result[0]["a"], 1)
        self.assertAlmostEqual(result[0]["count"], 2)
        self.assertAlmostEqual(result[1]["acc. time"], 4)
        self.assertAlmostEqual(result[1]["a"], 2)
        self.assertAlmostEqual(result[1]["count"], 1)
        self.assertAlmostEqual(result[2]["acc. time"], 2)
        self.assertAlmostEqual(result[2]["b"], 2)
        self.assertAlmostEqual(result[2]["count"], 1)

    def test_htrace_push_pop(self):
        logger = HTraceLogger(rt_traces=True)

        logger.push("outer")
        time.sleep(0.2)
        logger.push("inner")
        time.sleep(0.2)
        logger.pop(inner_result=True, iterations=10, type="B")
        time.sleep(0.2)
        logger.push("inner")
        time.sleep(0.2)
        logger.pop(inner_result=False, iterations=20, type="A")
        logger.push("inner")
        time.sleep(0.2)
        logger.pop(inner_result=True)
        logger.pop(outer_result=True, some_metric=2, iterations=5)
        logger.push("outer")
        logger.pop(outer_result=True, some_metric=4)
        logger.push("outer")
        logger.pop(outer_result=True, some_metric=5)

        with patch.object(builtins, "print") as mock_print:
            logger.printReport()
        output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
        print(output)
        self.assertIn("outer_result: 100.0% true, 0.0% false, n=3", output)
        self.assertIn("some_metric: 2.000 < 3.667 | 4.000 < 5.000, n=3", output)
        self.assertIn("iterations: 5.000 < 5.000 | 5.000 < 5.000, n=1", output)
        self.assertIn("inner_result: 66.7% true, 33.3% false, n=3", output)
        self.assertIn(
            "iterations: 10.000 < 15.000 | 15.000 < 20.000, n=2", output)

    def test_htrace_scoped_span(self):
        logger = HTraceLogger(rt_traces=True)

        with logger.span("outer") as outer1:
            time.sleep(0.2)
            with logger.span("inner") as inner1:
                time.sleep(0.2)
                inner1.note(inner_result=True, iterations=10, type="B")
            time.sleep(0.2)
            with logger.span("inner") as inner2:
                time.sleep(0.2)
                inner2.note(inner_result=False, iterations=20, type="A")
            with logger.span("inner") as inner3:
                time.sleep(0.2)
                inner3.note(inner_result=True)
            outer1.note(outer_result=True, some_metric=2, iterations=5)
        with logger.span("outer") as outer2:
            outer2.note(outer_result=True, some_metric=4)
        with logger.span("outer") as outer3:
            outer3.note(outer_result=True, some_metric=5)

        with patch.object(builtins, "print") as mock_print:
            logger.printReport()
        output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
        print(output)
        self.assertIn("outer_result: 100.0% true, 0.0% false, n=3", output)
        self.assertIn("some_metric: 2.000 < 3.667 | 4.000 < 5.000, n=3", output)
        self.assertIn("iterations: 5.000 < 5.000 | 5.000 < 5.000, n=1", output)
        self.assertIn("inner_result: 66.7% true, 33.3% false, n=3", output)
        self.assertIn(
            "iterations: 10.000 < 15.000 | 15.000 < 20.000, n=2", output)


class HTraceValueTracesTest(unittest.TestCase):
    """Tests for the iteration value tracing API."""

    # -- disabled no-op behavior -------------------------------------------

    def test_disabled_is_noop(self):
        logger = HTraceLogger()  # disabled by default

        with logger.span("s") as s:
            s.add_value("v", 1.0)
            wrapped = s.wrap_fn(lambda x: x*2, name="f")
            wrapped(3)
            cb = s.callback("v")
            cb(numpy.array([1.0]))

        conv = logger.track_values("v")
        values = [0.0]
        while not utils.converged(values, abs_tol=1e-6):
            values.append(values[-1] + 1.0)
            if len(values) > 3:
                break

        self.assertTrue(logger.get_rt_traces().empty)
        self.assertTrue(logger.get_value_traces().empty)
        self.assertEqual(logger.get_rt_stats().shape[0], 0)

        # Test that wrapped functions are identical when disabled
        def f(x):
            return x + 1

        with logger.span("s") as s:
            wrapped = s.wrap_fn(f, name="f")
        self.assertIs(wrapped, f)

    # -- activation layers --------------------------------------------------

    def test_rt_traces_only(self):
        logger = HTraceLogger()
        logger.enable_rt_traces()
        self.assertTrue(logger.enabled_rt_traces)
        self.assertFalse(logger.enabled_value_traces)

        with logger.span("outer"):
            for _ in range(5):
                with logger.span("inner") as s:
                    s.add_value("v", 1.0)
                    s.wrap_fn(lambda x: x, name="f")(1.0)

        # Timing/metadata recorded, but no iteration values.
        self.assertEqual(len(logger.get_rt_traces()), 5 + 1)
        self.assertTrue(logger.get_value_traces().empty)

    def test_value_traces_only(self):
        logger = HTraceLogger()
        logger.enable_value_traces()
        self.assertFalse(logger.enabled_rt_traces)
        self.assertTrue(logger.enabled_value_traces)

        with logger.span("outer"):
            for _ in range(5):
                with logger.span("inner") as s:
                    s.add_value("v", 1.0)
                    s.wrap_fn(lambda x: x, name="f")(1.0)

        # Span exists to hold samples, but no timing metadata is recorded.
        self.assertTrue(logger.get_rt_traces().empty)
        value_traces = logger.get_value_traces()
        print(value_traces)
        self.assertEqual(len(value_traces), 10)
        self.assertEqual(
            list(value_traces.loc[value_traces["variable"] == "v"]["value"]),
            [1.0]*5)
        self.assertEqual(
            list(value_traces.loc[value_traces["variable"] == "f"]["value"]),
            [1.0]*5)

    def test_enable_enables_both(self):
        logger = HTraceLogger()
        logger.enable()
        self.assertTrue(logger.enabled_rt_traces)
        self.assertTrue(logger.enabled_value_traces)
        logger.disable()
        self.assertFalse(logger.enabled_rt_traces)
        self.assertFalse(logger.enabled_value_traces)

    # -- span + sample basics ----------------------------------------------

    def test_rt_and_value_traces(self):
        logger = HTraceLogger(rt_traces=True, value_traces=True)
        with logger.span("outer", path_type="LSR") as s:
            s.add_value("w_A_g", 0.0)
            with logger.span("inner") as si:
                si.add_value("t_a", 1.0)
                si.add_value("t_a", 2.0)

        rt_traces = logger.get_rt_traces()
        self.assertEqual(len(rt_traces), 2)
        self.assertIn("outer", rt_traces["span"].values)
        self.assertIn("outer.inner", rt_traces["span"].values)
        self.assertEqual(rt_traces["duration"].isna().sum(), 0)
        self.assertGreater(rt_traces["duration"].sum(), 0)
        self.assertGreater(rt_traces["self_duration"].sum(), 0)

        value_traces = logger.get_value_traces()
        self.assertEqual(len(value_traces), 3)
        self.assertIn("w_A_g", set(value_traces["variable"]))
        self.assertEqual(
            list(value_traces[value_traces["variable"] == "t_a"]["value"]),
            [1.0, 2.0])

    def test_span_note_metadata(self):
        logger = HTraceLogger(rt_traces=True)
        with logger.span("s") as s:
            s.note(converged=False)
        spans = logger.get_rt_traces()
        self.assertEqual(spans.iloc[0]["converged"], False)

    def test_exception_safety(self):
        logger = HTraceLogger(rt_traces=True)
        with self.assertRaises(ValueError):
            with logger.span("s") as s:
                s.add_value("v", 1.0)
                raise ValueError("boom")
        # Span still closed and recorded despite the exception.
        self.assertEqual(len(logger.get_rt_traces()), 1)

    def test_span_unwind(self):
        logger = HTraceLogger(rt_traces=True)
        # Simulate an unbalanced push (e.g. an exception through a solver's
        # push/pop pair): open two spans without popping them.
        logger.push("outer")
        logger.push("inner")
        self.assertEqual(logger.unwind(error=True), 2)
        # Stack must be empty: a subsequent span must be top-level, not nested.
        with logger.span("next") as s:
            pass
        spans = logger.get_rt_traces()
        self.assertTrue("next" in spans["span"].values)

    # -- wrap_fn / callback -------------------------------------------------

    def test_wrap_fn(self):
        logger = HTraceLogger(value_traces=True)

        def square(x):
            return x*x

        with logger.span("newton") as s:
            f = s.wrap_fn(square, name="csc.f")
            f(2.0)
            f(3.0)

        tr = logger.get_value_traces()
        print(tr)
        evals = tr[(tr["span"] == "newton") & (tr["variable"] == "csc.f")]
        self.assertEqual(list(evals["value"]), [4.0, 9.0])
        self.assertEqual(list(evals["step"]), [0, 1])

    def test_callback(self):
        logger = HTraceLogger(value_traces=True)

        with logger.span("minimize") as s:
            cb = s.callback("V_A")
            cb(numpy.array([1.0]))
            cb(numpy.array([2.0]))

        it = logger.get_value_traces()
        acc = it[(it["span"] == "minimize") & (it["variable"] == "V_A")]
        self.assertEqual(list(acc["value"]), [1.0, 2.0])

    # -- track_values -------------------------------------------------------

    def test_track_values(self):
        logger = HTraceLogger(rt_traces=True, value_traces=True)

        logger.push("dubins_csc3D")
        values = logger.track_values("w_A_g", [1.0])
        while not utils.converged(values, abs_tol=1e-3):
            values.append(0.5*values[-1])
        logger.pop(iterations=len(values))

        spans = logger.get_rt_traces()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans.iloc[0]["span"], "dubins_csc3D")

        it = logger.get_value_traces()
        trace = it[it["variable"] == "w_A_g"]
        self.assertEqual(list(trace["value"]), list(values))

    def test_track_values_residual(self):
        logger = HTraceLogger(value_traces=True)

        logger.push("loop")
        values = logger.track_values("v", [1.0])
        values.append(0.5)
        values.append(0.25)
        logger.pop()

        it = logger.get_value_traces()
        residuals = it[it["variable"] == "v_residual"]
        self.assertEqual(list(residuals["value"]), [0.5, 0.25])

    def test_track_values_disabled(self):
        logger = HTraceLogger()  # disabled
        values = logger.track_values("v", [1.0])
        values.append(0.5)
        # Behaves like a plain list, records nothing.
        self.assertEqual(list(values), [1.0, 0.5])
        self.assertTrue(logger.get_value_traces().empty)

    # -- get_rt_stats ------------------------------------------------------

    def test_get_rt_stats(self):
        logger = HTraceLogger(rt_traces=True)

        with logger.span("outer", path_type="LSR") as s:
            s.add_value("v", 1.0)
        with logger.span("outer", path_type="RSL") as s:
            s.add_value("v", 2.0)

        summary = logger.get_rt_stats()
        self.assertEqual(summary["count"].iloc[0], 2)

    # -- wrap_fn with arg_name ----------------------------------------------

    def test_wrap_fn_records_arg_and_signed_value(self):
        logger = HTraceLogger(value_traces=True)

        with logger.span("newton"):
            f = logger.wrap_fn(lambda x: x**2 - 2, name="f", arg_name="t_a")
            for x in (0.5, 1.0, 1.5, 1.8):
                f(x)

        values = logger.get_value_traces()
        wide = values.pivot_table(
            index="step", columns="variable", values="value")
        self.assertEqual(list(wide["t_a"]), [0.5, 1.0, 1.5, 1.8])
        # The value is recorded *signed*: the sign tells which side of the root
        # the iterate is on, and the magnitude is recoverable downstream.
        self.assertEqual(
            list(wide["f"]), [0.25 - 2, 1.0 - 2, 2.25 - 2, 3.24 - 2])
        self.assertTrue(any(v < 0 for v in wide["f"]))
        self.assertTrue(any(v > 0 for v in wide["f"]))

    def test_wrap_fn_without_arg_name_records_only_value(self):
        logger = HTraceLogger(value_traces=True)

        with logger.span("obj"):
            logger.wrap_fn(lambda x: 2*x, name="obj.f")(3.0)

        values = logger.get_value_traces()
        self.assertEqual(list(values["variable"]), ["obj.f"])
        self.assertEqual(list(values["value"]), [6.0])

    def test_wrap_fn_disabled_returns_fn_unchanged(self):
        logger = HTraceLogger()  # disabled

        def fn(x):
            return x

        self.assertIs(logger.wrap_fn(fn, name="f", arg_name="t"), fn)

    def test_span_id_separates_invocations(self):
        """Equal span paths must still be distinguishable per invocation."""
        logger = HTraceLogger(rt_traces=True, value_traces=True)

        for case in (0, 1):
            logger.trace_id = case
            for guess in (0, 1):
                with logger.span("newton", guess=guess) as s:
                    s.add_value("f_residual", 1.0/(guess+1))

        rt = logger.get_rt_traces()
        values = logger.get_value_traces()
        # 4 root solves, each with its own unique span id.
        self.assertEqual(rt["span_id"].nunique(), 4)
        self.assertEqual(values["span_id"].nunique(), 4)
        self.assertEqual(len(rt), 4)
        # The span path and trace_id repeat, span_id does not.
        self.assertEqual(sorted(rt["trace_id"].unique()), [0, 1])
        # Each invocation restarts its own step counter at zero.
        self.assertEqual(sorted(values["step"].unique()), [0])

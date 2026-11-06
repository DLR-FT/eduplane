# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.utils.smath.

Run from top-level package directory:

   python3 -m unittest tests.utils.test_smath

"""

import math
import sys
import unittest

from eduplane.utils.smath import *


class SMathTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        if "-v" in sys.argv:
            self.verbose = True
        else:
            self.verbose = False
        super().__init__(*args, **kwargs)

    def test_smath_func_generator(self):
        # pylint: disable=function-redefined, no-member
        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "hyp": {
                "name": "C",
                "dimension": "length",
                "unit": "m"
            }
        }

        # Wrapped functions can be called with positional or keyword arguments
        @smath_func(symbols)
        def hyp(a, b):
            return sqrt(a**2 + 2*b**2)

        self.assertEqual(hyp(1, 2), 3)
        self.assertNotEqual(hyp(2, 1), 3)
        self.assertEqual(hyp(a=1, b=2), 3)
        self.assertEqual(hyp(b=2, a=1), 3)

        # The order of the wrapped function's positional args is preserved
        @smath_func(symbols)
        def hyp(b, a):
            return sqrt(a**2 + 2*b**2)

        self.assertEqual(hyp(2, 1), 3)
        self.assertNotEqual(hyp(1, 2), 3)
        self.assertEqual(hyp(a=1, b=2), 3)
        self.assertEqual(hyp(b=2, a=1), 3)

        # Wrapped function can be vectorized (use numpy as math lib)
        @smath_func(symbols, backend="numpy")
        def hyp(a, b):
            return sqrt(a**2 + 2*b**2)

        self.assertTrue(
            numpy.array_equal(
                hyp(numpy.asarray([1, 1]), numpy.asarray([2, 2])),
                numpy.asarray([3, 3])))

        # Using the attached 'with_units' function,
        # the return value will have a physical unit attached.
        @smath_func(symbols)
        def hyp(a, b):
            return sqrt(a**2 + 2*b**2)

        self.assertEqual(hyp.with_units(1, 2), 3*unyt.m)
        self.assertEqual(hyp.with_units(1, 2).units, unyt.m)
        self.assertTrue(hyp.check_dimensions())

        # The 'with_units' function also checks input argument units (if any)
        # and will try to convert to unit specified in symbols list.
        self.assertEqual(hyp.with_units(b=2*unyt.m, a=1*unyt.m), 3*unyt.m)
        self.assertEqual(hyp.with_units(b=2*unyt.m, a=100*unyt.cm), 3*unyt.m)
        with self.assertRaises(unyt.exceptions.UnitConversionError):
            hyp.with_units(b=2*unyt.m, a=1*unyt.kg)

        # The 'with_units' function also performs dimensional analysis for
        # the whole function and result value.
        symbols2 = {**symbols}
        symbols2["b"] = {"name": "B", "dimension": "area", "unit": "m**2"}

        @smath_func(symbols2)
        def hyp(a, b):
            return sqrt(a**2 + 2*b**2)

        with self.assertRaises(unyt.exceptions.UnitOperationError):
            hyp.with_units(1, 2)

    def test_smath_func_piecewise(self):
        # pylint: disable=function-redefined, no-member
        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "c": {
                "name": "C",
                "dimension": "area",
                "unit": "m^2"
            }
        }

        # Wrapped functions can be called with positional or keyword arguments
        @smath_func(symbols, returns="c", backend="math")
        def func_piecewise(a, b):
            return Piecewise(
                (a*b, a >= 0),
                (-a*b, a < 0),
            )

        self.assertEqual(func_piecewise(2, 2), 4)
        self.assertEqual(func_piecewise(-2, 2), 4)
        self.assertEqual(func_piecewise(2, -2), -4)

        # Test same function with numpy backend
        @smath_func(symbols, returns="c", backend="numpy")
        def func_piecewise(a, b):
            return Piecewise(
                (a*b, a >= 0),
                (-a*b, a < 0),
            )

        self.assertEqual(func_piecewise(2, 2), 4)
        self.assertEqual(func_piecewise(-2, 2), 4)
        self.assertEqual(func_piecewise(2, -2), -4)

        # With the math backend, only the desired expression is evaluated,
        # which may avoid division-by-zero errors and warnings.
        @smath_func(symbols, returns="c", backend="math")
        def func_piecewise(a):
            return Piecewise(
                (1/a, a > 0),
                (0, a <= 0),
            )

        numpy.seterr(all='ignore')
        self.assertEqual(func_piecewise(2), 1/2)
        self.assertEqual(func_piecewise(1), 1)
        self.assertEqual(func_piecewise(0), 0)

        # Note that with the Numpy backend, all function pieces are evaluated,
        # which means that division-by-zero errors/warnings may be produced
        # before the correct output is selected
        # (see e.g. https://github.com/sympy/sympy/issues/15014):
        @smath_func(symbols, returns="c", backend="numpy")
        def func_piecewise(a):
            return Piecewise(
                (1/a, a > 0),
                (0, a <= 0),
            )

        numpy.seterr(all='ignore')
        self.assertEqual(func_piecewise(2), 1/2)
        self.assertEqual(func_piecewise(1), 1)
        # TODO: Works when wrapping into array, but passing a scalar
        # results in a ZeroDivisionError.
        # Related:
        # - https://stackoverflow.com/questions/38040846/error-with-sympy-lambdify-for-piecewise-functions-and-numpy-module
        # - https://github.com/sympy/sympy/issues/15014
        # - https://github.com/sympy/sympy/issues/14789
        # self.assertEqual(func_piecewise(0), 0)
        self.assertEqual(func_piecewise(numpy.array(0)), 0)

    def test_smath_eval_if(self):
        import pandas

        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "c": {
                "name": "C",
                "dimension": "area",
                "unit": "m^2"
            }
        }

        # Wrapped functions can be called with positional or keyword arguments
        @smath_func(symbols, backend="numpy")
        def c(a, b):
            return 1*a + 10*b

        # Eval function from mapping
        self.assertEqual(c.eval_if({"a": 1, "b": 2, "other": 3}), 21)
        # Eval function from keyword argument(s)
        self.assertEqual(c.eval_if(a=1, b=2, other=3, another=4), 21)
        # Eval function from mapping and keyword argument(s)
        self.assertEqual(c.eval_if({"a": 1}, b=2, other=3, another=4), 21)
        # Eval function from dataframe and keyword argument
        self.assertListEqual(
            c.eval_if(pandas.DataFrame({"a": [1, 2]}), b=2).to_list(), [21, 22])
        # Eval function from masked dataframe (note that an numpy.ndarray is
        # returned when applying masks)
        self.assertListEqual(
            c.eval_if(
                pandas.DataFrame({"a": [1, 2]}),
                mask=[True, False],
                fill_value=0,
                b=2).tolist(), [21.0, 0.0])

    def test_smath_func_runtime_overhead(self):
        # pylint: disable=function-redefined, no-member

        import math
        from timeit import repeat

        import numpy

        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "hyp": {
                "name": "C",
                "dimension": "length",
                "unit": "m"
            }
        }

        def hyp_math(a, b):
            return math.sqrt(a**2 + b**2)

        def hyp_numpy(a, b):
            return numpy.sqrt(a**2 + b**2)

        # Time a callable and return the *minimum* seconds-per-call over
        # several repeats. Using best-of (min) rather than a single run makes
        # the benchmark robust against scheduler jitter and GC pauses.
        def bench(callable_, number=200000, repeat_=7):
            return min(repeat(callable_, number=number, repeat=repeat_))/number

        # smath_func returns the lambdified function directly (metadata is
        # attached as attributes, not via an extra call layer), so invocation
        # cost should be ~= native lambdify. We assert a generous 2x bound to
        # catch a regression while allowing headroom for noise.
        benchmark_hyp_math = bench(lambda: hyp_math(1, 2))
        benchmark_hyp_numpy = bench(lambda: hyp_numpy(1, 2))

        # Benchmark backend "math" vs. native math function
        @smath_func(symbols, backend="math")
        def hyp(a, b):
            return sqrt(a**2 + b**2)

        benchmark_hyp_smath = bench(lambda: hyp(1, 2))
        self.assertLess(benchmark_hyp_smath, benchmark_hyp_math*2.0)
        if self.verbose:
            print(
                f"\nBenchmark math vs. smath with math backend: {benchmark_hyp_math} {benchmark_hyp_smath}"
            )

        # Benchmark backend "numpy" vs native numpy function
        @smath_func(symbols, backend="numpy")
        def hyp(a, b):  # noqa: F811 (intentional redefinition per backend)
            return sqrt(a**2 + b**2)

        benchmark_hyp_smath = bench(lambda: hyp(1, 2))
        self.assertLess(benchmark_hyp_smath, benchmark_hyp_numpy*2.0)
        if self.verbose:
            print(
                f"Benchmark numpy vs. smath with numpy backend: {benchmark_hyp_numpy} {benchmark_hyp_smath}"
            )

        # Benchmark "with_units" function.
        # Parsing units before calling 'with_units' function will
        # reduce the runtime overhead a bit .. still about factor 100.
        parse_units(symbols)
        benchmark_hyp_numpy = bench(
            lambda: hyp_numpy(1, 2), number=10000, repeat_=7)
        benchmark_hyp_check_units = bench(
            lambda: hyp.with_units(1, 2), number=10000, repeat_=7)
        self.assertTrue(benchmark_hyp_check_units < benchmark_hyp_numpy*130)
        if self.verbose:
            print(
                f"Benchmark numpy vs. smath with numpy backend and unit checking: {benchmark_hyp_numpy} {benchmark_hyp_check_units}"
            )

        # Benchmark "eval_if" function.
        import pandas
        df = pandas.DataFrame({"a": range(100), "b": range(100, 200)})
        benchmark_hyp_numpy = bench(
            lambda: hyp_numpy(df.a, df.b), number=1000, repeat_=7)
        benchmark_hyp_eval_if_every = bench(
            lambda: hyp.eval_if(df), number=1000, repeat_=7)
        benchmark_hyp_eval_if_half = bench(
            lambda: hyp.eval_if(df, df.a >= 50), number=1000, repeat_=7)
        self.assertTrue(benchmark_hyp_eval_if_every < benchmark_hyp_numpy*1.5)
        self.assertTrue(benchmark_hyp_eval_if_half < benchmark_hyp_numpy*5)
        if self.verbose:
            print(
                f"Benchmark numpy vs. smath eval_if: {benchmark_hyp_numpy} {benchmark_hyp_eval_if_every}"
            )
            print(
                f"Benchmark numpy vs. smath eval_if with mask: {benchmark_hyp_numpy} {benchmark_hyp_eval_if_half}"
            )

    def test_get_dimension(self):
        dim_expr = "time"
        self.assertEqual(get_dimension(dim_expr), unyt.dimensions.time)

        dim_expr = "length/time"
        self.assertEqual(
            get_dimension(dim_expr),
            unyt.dimensions.length/unyt.dimensions.time)

    def test_smath_func_from_expr(self):
        # pylint: disable=function-redefined, no-member

        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "hyp": {
                "name": "C",
                "dimension": "length",
                "unit": "m"
            }
        }

        # Create symbols and an expression
        from types import SimpleNamespace
        __symbols = create_symbols(symbols, "name")
        s = SimpleNamespace(**__symbols)
        expr = sqrt(s.a**2 + 2*s.b**2)

        # smath functions from expressions can only be called with keyword arguments
        # as the order of positional arguments would have to depend on the expression
        hyp = smath_func.from_expr("hyp", expr, symbols)
        self.assertEqual(hyp(a=1, b=2), 3)
        self.assertEqual(hyp(b=2, a=1), 3)
        self.assertNotEqual(hyp(a=2, b=1), 3)

        # Wrapped function can be vectorized (use numpy as math lib)
        hyp = smath_func.from_expr("hyp", expr, symbols, backend="numpy")
        self.assertTrue(
            numpy.array_equal(
                hyp(a=numpy.asarray([1, 1]), b=numpy.asarray([2, 2])),
                numpy.asarray([3, 3])))

        # Using the attached 'with_units' function,
        # the return value will have a physical unit attached.
        hyp = smath_func.from_expr("hyp", expr, symbols)
        self.assertEqual(hyp.with_units(a=1, b=2).units, unyt.m)
        self.assertEqual(hyp.with_units(b=2, a=1).units, unyt.m)
        self.assertEqual(hyp.with_units(a=1, b=2), 3*unyt.m)
        self.assertEqual(hyp.with_units(b=2, a=1), 3*unyt.m)
        self.assertNotEqual(hyp.with_units(a=2, b=1), 3*unyt.m)

        # We can check dimensions only with 'check_dimensions'.
        # For this, the equation has to be evaluated. By default,
        # a '1' is passed for each argument, but other values
        # may be provided.
        self.assertTrue(hyp.check_dimensions())
        self.assertTrue(hyp.check_dimensions(b=4))

        # The 'with_units' function also checks input argument units (if any)
        # and will try to convert to unit specified in symbols list.
        self.assertEqual(hyp.with_units(b=2*unyt.m, a=1*unyt.m), 3*unyt.m)
        self.assertEqual(hyp.with_units(b=2*unyt.m, a=100*unyt.cm), 3*unyt.m)
        with self.assertRaises(unyt.exceptions.UnitConversionError):
            hyp.with_units(b=2*unyt.m, a=1*unyt.kg)

        # The 'with_units' function also performs dimensional analysis for
        # the whole function and result value.
        symbols2 = {**symbols}
        symbols2["b"] = {"name": "B", "dimension": "area", "unit": "m**2"}
        hyp = smath_func.from_expr("hyp", expr, symbols2)
        with self.assertRaises(unyt.exceptions.UnitOperationError):
            hyp.with_units(a=1, b=2)

    def test_check_dimensions_dimensionless(self):
        # Check processing of dimensionless quantities, specifically angles
        symbols = {
            "a": {
                "name": "A",
                "dimension": "dimensionless",
                "unit": None
            },
            "b": {
                "name": "B",
                "dimension": "dimensionless",
                "unit": None
            }
        }

        @smath_func(symbols, returns="b", backend="numpy")
        def calc_b(b):
            return asin(b)

        self.assertTrue(calc_b.check_dimensions())

    def test_rename_symbols(self):

        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "hyp": {
                "name": "C",
                "dimension": "length",
                "unit": "m"
            }
        }

        # Create symbols and an expression
        from types import SimpleNamespace
        __symbols = create_symbols(symbols, "name")
        s = SimpleNamespace(**__symbols)
        expr = sqrt(s.a**2 + s.b**2)
        self.assertEqual(s.a.name, "A")
        self.assertEqual(s.b.name, "B")
        self.assertEqual(s.hyp.name, "C")
        for symb in get_symbols(expr, symbols):
            self.assertIn(symb.name, ["A", "B", "C"])

        # Rename symbols in expression to their IDs
        name2id_map = get_name_id_map(symbols, name_key="name")
        self.assertEqual(list(name2id_map.keys()), ["A", "B", "C"])
        self.assertEqual(list(name2id_map.values()), ["a", "b", "hyp"])

        expr = rename_symbols(expr, symbols, name2id_map)
        for symb in get_symbols(expr, symbols):
            self.assertIn(symb.name, ["a", "b", "hyp"])

    def test_get_definitions(self):

        symbols = {
            "a": {
                "name": "A",
                "dimension": "length",
                "unit": "m"
            },
            "b": {
                "name": "B",
                "dimension": "length",
                "unit": "m"
            },
            "hyp": {
                "name": "C",
                "dimension": "length",
                "unit": "m",
                "definition": "sqrt(a**2+b**2)"
            }
        }

        defs = get_definitions(symbols)
        self.assertEqual(defs["hyp"](a=1, b=1), math.sqrt(2))

        # Also useful (e.g. in a notebook environment):
        # Provide a name prefix and inject into local/global scope
        defs = get_definitions(symbols, name_prefix="calc_")
        globals().update(**defs)
        # pylint: disable=undefined-variable
        self.assertEqual(calc_hyp(a=1, b=1), math.sqrt(2))

    def test_remove_dynamic_symbols(self):

        original = config["enable_dynamic_symbols"]
        try:
            config["enable_dynamic_symbols"] = True
            symbols = {
                "a": {
                    "name": "A",
                    "dimension": "dimensionless",
                    "unit": None
                },
                "b": {
                    "name": "B",
                    "dimension": "dimensionless",
                    "unit": None
                }
            }

            @smath_func(symbols, returns="b", backend="numpy")
            def calc_b(b):
                return asin(b)

            rhs_expr = calc_b.eq.rhs
            free_symb_names = [s.name for s in rhs_expr.free_symbols]
            dyn_symb_names = [s.name for s in find_dynamicsymbols(rhs_expr)]
            # b is instantiated as dynamic symbol B(t)
            self.assertIn("t", free_symb_names)
            self.assertIn("B", dyn_symb_names)
            # After replacing dynamic symbols with constant symbols of same name,
            # we will see "B" as free symbol
            rhs_expr = remove_dynamic_symbols(
                rhs_expr, symbols, name_key="name")
            free_symb_names = [s.name for s in rhs_expr.free_symbols]
            dyn_symb_names = [s.name for s in find_dynamicsymbols(rhs_expr)]
            self.assertIn("B", free_symb_names)
            self.assertEqual(dyn_symb_names, [])
        finally:
            config["enable_dynamic_symbols"] = original

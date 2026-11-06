# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import division, print_function

import dataclasses
import functools
import inspect
import math
from dataclasses import dataclass

import numpy


@dataclass
class BetterDataClass:

    def asdict(self):
        return dataclasses.asdict(self)


def converged(values, abs_tol=1e-6, max_iter=1e3, unique_values=True):
    """Helper function to check if an iteratively calculated variable has
    converged.
    If unique_values=True, values are checked for uniqueness and a ValueError
    is raised on re-occurring values. This may be used to avoid infinite
    loops. Note that the first value is not considered in the uniqueness check,
    as it may be a first guess for the optimization procedure and as such it
    must be considered an input, not an output of the algorithm.
    """
    values_count = len(values)
    if values_count > 1 and abs(values[-1] - values[-2]) < abs_tol:
        return True
    if values_count > max_iter:
        raise ValueError(f"Max. number of iterations reached: {max_iter}")
    if (unique_values and values_count > 1
            and len(numpy.unique(values[1:])) != len(values) - 1):
        # non_unique_values = [value for value in values[1:] if values[1:].count(value) > 1]
        # raise ValueError(f"Values not unique: {non_unique_values}")
        raise ValueError("Values not unique")
    return False


def getListPattern(arg, list_types=(tuple, list)):
    """Extract the nesting pattern of a list of lists.

    Examples:
    [1] -> []
    [[1,2], [3,4]] -> [[], []]
    [[[1], [2]], [3,4]] -> [[[], []], []]
    """
    if not isinstance(arg, list_types):
        return None
    nested_lists = [p for p in arg if isinstance(p, list_types)]
    if not nested_lists:
        return []
    else:
        return [getListPattern(nl) for nl in nested_lists]


def listify(arg, pattern=[], repeat=1, list_types=(tuple, list)):
    """'Listify' arbitrary objects, i.e. return a list (or nested list based
    on the given nesting pattern) containing the given object.
    If the given object is a list or nested list, its structure is tried to
    be matched against the desired nesting pattern.

    This can be used to 'listify' function parameters and to allow both single
    objects, lists of objects or nested lists of objects to be passed to a
    the function. After calling listify, the argument safely be assumed to have
    the structure of the given pattern and no further logic is required in the
    function implementation.

    Examples:
    listify("one", [])            -> ["one"]
    listify(["one"], [])          -> ["one"]
    listify(["one"], [[], []])    -> [["one"], ["one"]]
    listify(["one", "two"], [[], []])    -> [["one", "two"], ["one", "two"]]
    """
    if not isinstance(arg, list_types):
        return listify(
            [arg for _ in range(repeat)],
            pattern=pattern,
            list_types=list_types)
    else:
        arg_pattern = getListPattern(arg, list_types)
        pattern = getListPattern(pattern, list_types)
        if arg_pattern == pattern:
            return arg
        else:
            return [
                listify(arg, pattern=p, list_types=list_types) for p in pattern
            ]


def error2constant(func, constant=numpy.nan, verbose=False):
    """Decorator to make a function that calculates a numerical value safely
    callable, i.e. to return a constant on any runtime exceptions.

    This is especially useful when numerically optimizing functions of limited
    domain using e.g. scipy.optimize.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # args_dict = inspect.signature(func).bind(*args, **kwargs).arguments
        # print(f"Calling {func.__name__} with arguments {args_dict}")
        try:
            return func(*args, **kwargs)
        except Exception as err:
            args_dict = inspect.signature(func).bind(*args, **kwargs).arguments
            if verbose:
                print(
                    f"An error of type {type(err)} was raised inside {func.__name__} "
                    f"when called with arguments {args_dict}"
                    f": {err}")
            return constant

    return wrapper


def getNaNs(df, exclude=[]):
    """Reduce a dataframe to its NaN values."""
    # Drop excluded columns
    df = df.drop(exclude, axis=1, errors="ignore")
    # Drop columns without NaNs
    df = df.loc[df.isnull().any(axis=1)]
    # Drop all non-NaN values
    df = df.loc[:, df.isnull().any(axis=0)]
    return df


def angular_diff(a: float, b: float, turn_dir: float | None = None) -> float:
    """
    Return signed angular difference from angle a to b (b - a) in radians.
    - turn_dir=None: returns value in [-π, π] (smallest angular difference).
    - turn_dir=1: returns smallest positive/clockwise rotation angle in range [0, 2π).
    - turn_dir=-1: returns smallest negative/counter-clockwise rotation angle in (-2π, 0].
    """
    raw = b - a
    if turn_dir is None:
        # Normalize to (-π, π]
        return angle_wrap_signed(raw)
    if turn_dir < 0:
        # CCW rotation (-2pi, 0]
        return -angle_wrap_unsigned(-raw)
    else:
        # CW rotation [0, -2π)
        return angle_wrap_unsigned(raw)


def angle_wrap_signed(a):
    return (a + math.pi) % (2*math.pi) - math.pi


def angle_wrap_unsigned(a):
    return a % (2*math.pi)


def clip(value, min_value=None, max_value=None):
    return numpy.clip(value, min_value, max_value)


def clip_abs(value, max_abs_value=None):
    abs_diff = abs(value) - abs(
        max_abs_value if max_abs_value is not None else value)
    if abs_diff > 0:
        return value - math.copysign(abs_diff, value)
    else:
        return value


def bounded_value(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def bounded_abs_value(value, max_abs_value):
    return min(max(value, -max_abs_value), max_abs_value)

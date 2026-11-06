# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Hierarchical runtime tracing, trace generation, analysis, and reporting.

This module is intended to instruments iterative and nested algorithms so that
*what happened inside them* can be analyzed after a run -- not just the
runtimes, but also concrete runtime values like residuals and others.

Concept model
-------------

Three concepts compose the trace:

**Span** -- a named, *timed*, *nestable* region of code. A span represents one
invocation of a logical unit of work (e.g. ``dubins_csc3D``,
``csc_newton``, or ``minimize_V_A``). Spans nest to mirror the call stack; each
span has:

* ``name`` -- the label given at ``push``/``span`` (leaf name), and
* ``path`` -- the fully-qualified dot-joined name reflecting nesting
  (e.g. ``outer.inner``).

A span also accumulates:

* ``duration`` -- total wall-clock time spent inside the span, and
* ``self_duration`` -- time spent *excluding* nested child spans (i.e. the
  span's own work, separate from its children),
* scalar ``metadata`` -- key/value facts about this invocation (e.g.
  ``path_type="LSR"``, ``converged=False``), and
* zero or more **value traces**.

When a span is closed via :meth:`Span.close` its ``duration``/``self_duration``
is finalized and the span becomes part of the logging trace.

**Value trace** -- the sequence of scalar *values* recorded for one iteration
variable inside a span, representing the steps of an iterative procedure.
Value traces are stored per span as a simple ``Dict[str, List[float]]``
mapping an iteration-variable name to its values in emission order (the list
index is the step). Naming conventions such as ``*_residual`` for residuals
and ``*.f`` for objective-function evaluations are helpful to analyze traces.

Activation
----------

Tracing is **optional and off by default**, and split into two independent
layers:

* **Runtime traces** (:meth:`HTraceLogger.enable_rt_traces`) -- span timing and
  scalar metadata, i.e. the classic ``push``/``pop`` behaviour.
* **Value traces** (:meth:`HTraceLogger.enable_value_traces`) -- iteration
  values recorded via ``add_value``/``track_values``/``wrap_fn``/``callback``.

:meth:`HTraceLogger.enable` turns on both; :meth:`HTraceLogger.disable` turns
both off. While a layer is disabled its instrumentation calls are cheap no-ops
(one branch), and ``wrap_fn`` returns the wrapped function *unchanged*. Spans
are created whenever either layer is active (value traces need a span to hold
their values). Use :meth:`HTraceLogger.tracing_enabled` for a scoped toggle.
"""

from __future__ import annotations

import itertools
import numbers
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy
import pandas

__all__ = [
    "HTraceLogger",
    "Span",
    "ValueTrace",
    "fold_traces",
    "freeze_trace",
    "summarize_folded_traces",
    "analyze_stats",
]

# ---------------------------------------------------------------------------
# Pure, side-effect-free helpers (fold/analyze/freeze a list of span traces).
# ---------------------------------------------------------------------------


def freeze_trace(value, exclude_keys=("duration", )):
    """Return a hashable representation for value (handles lists/dicts/sets)."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                (k, freeze_trace(v)) for k, v in value.items()
                if k not in exclude_keys))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_trace(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(freeze_trace(v) for v in value))
    return value


def fold_traces(
    traces: Iterable[Dict[str, Any]], exclude_keys=("duration", )
) -> List[Dict[str, Any]]:
    groups = defaultdict(lambda: {"acc. time": 0.0, "count": 0})
    for trace in traces:
        key = freeze_trace(trace, exclude_keys)
        trace_group = groups[key]
        trace_group["acc. time"] += trace["duration"]
        trace_group["count"] += 1
        trace_group.update(
            {
                k: v
                for k, v in trace.items() if k not in exclude_keys
            })
    return [groups[k] for k in sorted(groups.keys())]


def summarize_folded_traces(folded_traces):
    result = {"acc. time": 0.0, "count": 0}
    for trace in folded_traces:
        result["acc. time"] += trace["acc. time"]
        result["count"] += trace["count"]
    return result


def analyze_stats(
    folded_traces, exclude_keys=("acc. time", "count", "duration")):
    pivot = {}
    for trace in folded_traces:
        for k, v in trace.items():
            if k not in exclude_keys:
                if k in pivot:
                    pivot[k].append(v)
                else:
                    pivot[k] = [v]
    bool_stats = {}
    num_stats = {}
    str_stats = {}
    for k, v in pivot.items():
        if isinstance(v[0], bool):
            bool_stats[k] = {
                "n": len(v),
                "true": v.count(True)/len(v)*100,
                "false": v.count(False)/len(v)*100,
            }
        elif isinstance(v[0], numbers.Number):
            num_stats[k] = {
                "n": len(v),
                "min": min(v),
                "max": max(v),
                "mean": float(numpy.mean(v)),
                "median": float(numpy.median(v)),
            }
        else:
            str_stats[k] = {"n": len(v)}
            for s in v:
                str_stats[k][str(s)] = v.count(s)/len(v)*100
    return bool_stats, num_stats, str_stats


def _to_scalar(value) -> float:
    """Coerce a value to a scalar float (arrays -> ||.||)."""
    arr = numpy.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size == 1 else float(numpy.linalg.norm(arr))


# Columns of the get_rt_stats() DataFrame (kept stable for downstream use).
_RT_STATS_COLUMNS = [
    "span", "acc_time", "count", "metric", "n", "min", "max", "mean", "median",
    "true_pct", "false_pct", "distribution"
]

# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class Span:
    """A named, timed, nestable region of code.

    A span is both the in-flight stack entry and the finalized trace: it
    accumulates ``duration``/``self_duration`` when :meth:`close` is called.
    ``active`` is ``False`` for the no-op span yielded while tracing is
    disabled, in which case every method is a cheap no-op.

    Iteration variables are stored as ``values: Dict[str, List[float]]``
    (variable name -> values in emission order).
    """

    def __init__(
            self,
            logger: "HTraceLogger",
            name: str,
            path: str,
            meta: Optional[Dict[str, Any]] = None,
            trace_id: Any = None,
            active: bool = True):
        self._logger = logger
        self.name = name
        self.path = path
        self.meta = dict(meta or {})
        self.trace_id = trace_id
        self.span_id = next(self._logger._span_ids)
        self.active = active
        self.start = time.perf_counter()
        self.child_time = 0.0
        self.duration: Optional[float] = None
        self.self_duration: Optional[float] = None
        self.values: Dict[str, List[float]] = defaultdict(list)

    def add_value(self, variable, value, **extra) -> None:
        """Append one value to the named iteration variable.

        ``extra`` key/value pairs are recorded as sibling variables (aligned by
        index), e.g. ``s.add_value("z_g_tf_residual", r, bracket=w)``.
        """
        if not self.active or not self._logger.enabled_value_traces:
            return
        self.values[str(variable)].append(_to_scalar(value))
        for k, v in extra.items():
            self.values[str(k)].append(_to_scalar(v))

    def note(self, **meta) -> None:
        """Attach scalar span metadata (e.g. ``converged=False``)."""
        if self.active:
            self.meta.update(meta)

    def wrap_fn(
            self,
            fn: Callable,
            name: str,
            arg_name: Optional[str] = None,
            **meta) -> Callable:
        """Wrap an objective/residual ``f(x)`` so every call is recorded.

        ``name`` identifies the recorded variable; by convention use ``*.f``
        for objective functions and ``*_residual`` for residuals. Intended for
        one-call-per-step solvers (``newton``, ``brentq``,
        ``minimize_scalar``). Returns ``fn`` unchanged when disabled.

        ``arg_name`` optionally also records the *first argument* ``x`` under
        that name, giving a paired (iterate, function value) sequence per root
        solve; both arrays are aligned by call index.
        """
        if not self.active or not self._logger.enabled_value_traces:
            return fn

        def wrapped(*args, **kwargs):
            y = fn(*args, **kwargs)
            self.add_value(name, y, **meta)
            if arg_name is not None and args:
                self.add_value(arg_name, args[0])
            return y

        return wrapped

    def callback(self, name, **meta) -> Callable:
        """Return a scipy-``minimize``-style callback recording accepted steps.

        ``minimize`` calls ``callback(xk)`` after each accepted iterate (no
        finite-difference probes), so this records only the descent path. This
        is :meth:`wrap_fn` specialized to record the first argument instead of
        the return value.
        """
        return self.wrap_fn(lambda xk, *args, **kwargs: xk, name=name, **meta)

    def close(self, **meta) -> None:
        """Finalize the span: freeze its duration and attach ``meta``."""
        if not self.active:
            return
        self.duration = time.perf_counter() - self.start
        self.self_duration = self.duration - self.child_time
        self.meta.update(meta)


class ValueTrace:
    """A list of iteration values that mirrors appends into a span.

    Uses composition (holds a ``values`` list) rather than subclassing ``list``
    so it is never accidentally treated as one. It supports ``len``, indexing
    and iteration, so it can be passed directly to ``utils.converged``.

    Each appended value is recorded on the span under ``variable``; when
    ``residual`` is set, the absolute difference to the previous value is
    recorded as ``<variable>_residual`` (the same quantity ``utils.converged``
    tests). Initial values are recorded without a residual.

    Intended usage::

        logger.push("dubins_csc3D")
        w_A_g_values = logger.track_values("w_A_g", [0])
        while not utils.converged(w_A_g_values, abs_tol=t_precision):
            ...
            w_A_g_values.append(new_w_A_g)
        logger.pop(iterations=len(w_A_g_values))
    """

    def __init__(self, span, variable, initial=(), residual=True):
        self._span = span
        self._variable = variable
        self._residual = residual
        self.values = list(initial)
        # Number of appends, i.e. loop iterations performed. Unlike
        # ``len(values)`` this excludes the initial guess, so it matches the
        # number of residual points.
        self.n_iterations = 0
        if span is not None:
            for value in self.values:
                span.add_value(variable, value)

    def append(self, value) -> None:
        self.values.append(value)
        self.n_iterations += 1
        if self._span is None:
            return
        self._span.add_value(self._variable, value)
        if self._residual and len(self.values) > 1:
            self._span.add_value(
                f"{self._variable}_residual",
                abs(self.values[-1] - self.values[-2]))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return f"ValueTrace({self._variable}, {self.values!r})"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class HTraceLogger:
    """Hierarchical tracer with iteration sampling and pandas export."""

    def __init__(
            self,
            rt_traces: bool = False,
            value_traces: bool = False,
            delim: str = "."):
        self.delim = delim
        self._rt_traces = rt_traces
        self._value_traces = value_traces
        self._local = threading.local()
        self._spans: List[Span] = []
        self._span_ids = itertools.count()
        # Identifier attached to spans/value traces opened from now on. Set this
        # before each solver invocation (e.g. to a test-case index) so that
        # convergence curves from different cases can be told apart.
        self.trace_id: Any = None

    # -- activation ---------------------------------------------------------

    @property
    def enabled_rt_traces(self) -> bool:
        """Whether span timing/metadata (the classic push/pop traces) is on."""
        return self._rt_traces

    @property
    def enabled_value_traces(self) -> bool:
        """Whether iteration-value collection is on."""
        return self._value_traces

    @property
    def enabled(self) -> bool:
        """Whether any tracing is active (i.e. spans are created)."""
        return self._rt_traces or self._value_traces

    def enable(self) -> None:
        """Enable both runtime and value traces."""
        self._rt_traces = True
        self._value_traces = True

    def disable(self) -> None:
        """Disable all tracing."""
        self._rt_traces = False
        self._value_traces = False

    def enable_rt_traces(self) -> None:
        """Enable span timing and scalar metadata (classic push/pop traces)."""
        self._rt_traces = True

    def enable_value_traces(self) -> None:
        """Enable collection of iteration values."""
        self._value_traces = True

    @contextmanager
    def tracing_enabled(self):
        prev = (self._rt_traces, self._value_traces)
        self._rt_traces = self._value_traces = True
        try:
            yield self
        finally:
            self._rt_traces, self._value_traces = prev

    # -- span stack ---------------------------------------------------------

    def _get_stack(self) -> List[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def push(self, span_name: str, **meta) -> Span:
        """Open a span on the (thread-local) stack and start timing it.

        Returns an inactive (no-op) span while tracing is disabled. Use
        :meth:`span` for the scoped equivalent.
        """
        if not self.enabled:
            return Span(self, span_name, span_name, active=False)
        stack = self._get_stack()
        parent_path = stack[-1].path if stack else ""
        path = (
            f"{parent_path}{self.delim}{span_name}"
            if parent_path else span_name)
        span = Span(
            self, name=span_name, path=path, meta=meta, trace_id=self.trace_id)
        stack.append(span)
        return span

    def _pop_span(self, span: Span, **kwargs) -> Span:
        stack = self._get_stack()
        span.close(**kwargs)
        self._spans.append(span)
        if stack and stack[-1] is span:
            stack.pop()
        if stack:
            stack[-1].child_time += span.duration
        return span

    def pop(self, **kwargs):
        """Close the innermost open span, attaching ``kwargs`` as metadata.

        No-op while tracing is disabled. Prefer the context-manager
        :meth:`span` so spans close even on exceptions.
        """
        stack = self._get_stack()
        if not stack:
            return
        self._pop_span(stack[-1], **kwargs)

    def unwind(self, **kwargs) -> int:
        """Close *all* open spans on the stack, innermost first.

        Required after an exception propagated through an unbalanced
        ``push``/``pop`` pair (e.g. an early ``return`` or raised error inside a
        solver): otherwise the leftover spans leak into the next invocation and
        corrupt its span nesting. Returns the number of spans closed.
        """
        stack = self._get_stack()
        closed = 0
        while stack:
            self._pop_span(stack[-1], **kwargs)
            closed += 1
        return closed

    # -- scoped span object (Q2) --------------------------------------------

    @contextmanager
    def span(self, name: str, **meta):
        """Scoped span. No-op while tracing is disabled."""
        span = self.push(name, **meta)
        try:
            yield span
        finally:
            if span.active:
                self._pop_span(span)

    # -- iteration sampling (requirement 1) ----------------------------------

    def add_value(self, variable, value, **extra) -> None:
        """Record one value trace under the current (top) span, if any."""
        if not self._value_traces:
            return
        stack = self._get_stack()
        if stack:
            stack[-1].add_value(variable, value, **extra)

    # -- logged convergence predicate (Q3) ----------------------------------

    def track_values(
            self,
            variable: str,
            initial=(),
            residual: bool = True) -> ValueTrace:
        """Return a :class:`ValueTrace` mirroring appends into the current span.

        Intended for ``while not utils.converged(values, ...)`` loops: replace
        the plain values list with the returned trace and pass it to
        ``utils.converged`` directly. Falls back to a plain list wrapper when
        no span is open (i.e. tracing is disabled).
        """
        stack = self._get_stack()
        span = stack[-1] if stack else None
        return ValueTrace(span, variable, initial=initial, residual=residual)

    def wrap_fn(
            self,
            fn: Callable,
            name: str,
            arg_name: Optional[str] = None,
            **meta) -> Callable:
        """Wrap an objective ``f(x)`` to record each call on the current span.

        Convenience wrapper around :meth:`Span.wrap_fn` for use inside an
        already-open ``logger.push``/``pop`` span. ``arg_name`` optionally also
        records the first argument (see :meth:`Span.wrap_fn`).
        Returns ``fn`` unchanged when tracing is disabled or
        no span is open.
        """
        if not self._value_traces:
            return fn
        stack = self._get_stack()
        if not stack:
            return fn
        return stack[-1].wrap_fn(fn, name=name, arg_name=arg_name, **meta)

    # -- pandas export (requirement 3) --------------------------------------

    def get_rt_traces(self) -> pandas.DataFrame:
        """Runtime-trace view: one row per span. Empty if rt traces are off."""
        rows = []
        if self._rt_traces:
            for span in self._spans:
                row = {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "span": span.path,
                    "duration": span.duration,
                    "self_duration": span.self_duration,
                }
                row.update(span.meta)
                rows.append(row)
        return pandas.DataFrame(rows)

    def get_value_traces(self) -> pandas.DataFrame:
        """Value-trace view: one row per recorded value."""
        rows = []
        for span in self._spans:
            for variable, values in span.values.items():
                for step, value in enumerate(values):
                    rows.append(
                        {
                            "trace_id": span.trace_id,
                            "span_id": span.span_id,
                            "span": span.path,
                            "variable": variable,
                            "value": value,
                            "step": step,
                        })
        return pandas.DataFrame(rows)

    def get_rt_stats(self) -> pandas.DataFrame:
        """Folded runtime-trace statistics as a tidy DataFrame.

        One row per ``(span, metric)``. The span-level ``acc_time`` and
        ``count`` are repeated on each row. Numeric metrics fill
        ``min``/``max``/``mean``/``median``; boolean metrics fill
        ``true_pct``/``false_pct``; string metrics fill ``distribution``.
        Empty if rt traces are off.
        """
        rows = []
        if self._rt_traces:
            grouped = defaultdict(list)
            for span in self._spans:
                grouped[span.path].append(
                    {
                        "duration": span.duration,
                        **span.meta
                    })
            for span, traces in grouped.items():
                folded = fold_traces(traces)
                summary = summarize_folded_traces(folded)
                base = {
                    "span": span,
                    "acc_time": summary["acc. time"],
                    "count": summary["count"],
                }
                bool_stats, num_stats, str_stats = analyze_stats(traces)
                for metric, v in bool_stats.items():
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "n": v["n"],
                            "true_pct": v["true"],
                            "false_pct": v["false"],
                        })
                for metric, v in num_stats.items():
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "n": v["n"],
                            "min": v["min"],
                            "max": v["max"],
                            "mean": v["mean"],
                            "median": v["median"],
                        })
                for metric, v in str_stats.items():
                    distribution = ", ".join(
                        f"{pct:.1f}% {key}" for key, pct in v.items()
                        if key != "n")
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "n": v["n"],
                            "distribution": distribution,
                        })
        return pandas.DataFrame(rows, columns=_RT_STATS_COLUMNS)

    # -- reporting ---------------------------------------------------------

    def printReport(self, summarize=True, stats=True, detailed=False):
        """Print a human-readable report of the runtime traces.

        Built on :meth:`get_rt_stats`; use that method directly to obtain the
        same evaluation as a DataFrame.
        """
        if not self._rt_traces or not self._spans:
            print("No traces logged to report")
            return
        print("Report over all traces:")
        df = self.get_rt_stats()
        for span, group in df.groupby("span", sort=False):
            acc_time = group["acc_time"].iloc[0]
            count = group["count"].iloc[0]
            if summarize:
                print(f"  {span}: acc. time={acc_time:.6f}, count={count}")
            else:
                print(f"  {span}")
            if stats:
                for _, row in group.iterrows():
                    metric = row["metric"]
                    if pandas.notna(row["true_pct"]):
                        print(
                            f"    {metric}: {row['true_pct']:.1f}% true, "
                            f"{row['false_pct']:.1f}% false, n={row['n']}")
                    elif pandas.notna(row["mean"]):
                        print(
                            f"    {metric}: {row['min']:.3f} < "
                            f"{row['mean']:.3f} | {row['median']:.3f} < "
                            f"{row['max']:.3f}, n={row['n']}")
                    else:
                        print(
                            f"    {metric}: {row['distribution']}, "
                            f"n={row['n']}")
            if detailed:
                for t in fold_traces([{"duration": s.duration, **s.meta}
                                      for s in self._spans if s.path == span]):
                    print(f"    {t}")

    def reset(self) -> None:
        self._spans = []
        self._local.stack = []

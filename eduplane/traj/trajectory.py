# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import List

import numpy
import pandas

from .discrete_segment import DiscreteSegment
from .segment_base import SegmentBase


@dataclass(init=False)
class Trajectory(SegmentBase):
    segments: List[SegmentBase] = field(default_factory=list)

    skip_zero_length_segments: bool = False

    t_res: float = 1e-6

    align_initial_states: bool = True

    @property
    def x0(self):
        if self.segments:
            return self.segments[0].x0
        else:
            return super().x0

    @property
    def xf(self):
        if self.segments:
            return self.segments[-1].xf
        else:
            return super().x0

    def __init__(
            self,
            segments=None,
            t0=0,
            x0=None,
            align_initial_states=True,
            skip_zero_length_segments=False):
        self.align_initial_states = align_initial_states
        if skip_zero_length_segments and segments is not None:
            self.segments = [
                seg for seg in segments if seg.tf > seg.t0 + self.t_res
            ]
        else:
            self.segments = segments
        if not self.segments and x0:
            self.segments = [
                DiscreteSegment(
                    pandas.DataFrame(data=x0, index=[float(t0)], dtype=float))
            ]
        else:
            self.alignSegments(align_initial_states)

    def __iter__(self):
        return self.segments.__iter__()

    def __len__(self):
        return len(self.segments)

    @property
    def t0(self):
        return self.segments[0].t0

    @property
    def tf(self):
        return self.segments[-1].tf

    @property
    def empty(self):
        return len(self.segments) == 0

    def alignSegments(self, align_initial_states=True):
        """Align all segments including initial values."""
        if len(self.segments) <= 1:
            return
        for seg1, seg2 in zip(self.segments, self.segments[1:]):
            if align_initial_states:
                seg2.shift(seg1.tf, **seg1.xf)
            else:
                seg2.shift(seg1.tf)

    def append(self, new_segment):
        """Append a new segment."""
        if new_segment.tf is None:
            raise ValueError("Cannot append segment with unknown final time")
        if (self.skip_zero_length_segments
                and new_segment.tf <= new_segment.t0 + self.t_res):
            return
        if self.segments:
            last_segment = self.segments[-1]
            new_segment.shift(last_segment.tf, **last_segment.xf)
        self.segments.append(new_segment)

    def shift(self, new_t0, **new_initial_values):
        self.segments[0].shift(new_t0, **new_initial_values)
        self.alignSegments(align_initial_states=bool(new_initial_values))
        return self

    def evaluate(self, t):
        # Find last segment which contains t
        seg = next(
            seg for seg in reversed(self.segments)
            if t <= seg.tf and t >= seg.t0)
        return seg.evaluate(t)

    def discretize(self, t_index=None, t_res=None, t0=None, tf=None):
        """Discretize the segment at the given time parameter values,
        resolution or number of points."""
        if t_index is None:
            t_index = self.get_time_index(t0, tf, t_res)
        dfs = []
        # print(
        #     f"Discretizing trajectory of {len(self.segments)} segments with t_index: {t_index}"
        # )
        # for idx, seg in enumerate(self.segments):
        #     print(f"  seg #{idx}: t->[{seg.t0:.16f}, {seg.tf:.16f}]")
        next_x0 = self.x0
        next_t0 = t_index[0]
        for seg in self.segments:
            # For discretizing each segment, the index is clipped to the
            # segment bounds. Note that these irregular parameter values are
            # dropped later on, however, the exact segment bound values must
            # be used as initial values for the subsequent segment.
            # For this to work, there cannot be any gaps in between segment
            # parameter intervals. This is ensured here independently of
            # aligning initial states.
            if not next_t0 == seg.t0:
                seg.shift(next_t0)
            if not self.align_initial_states:
                next_x0 = seg.x0
            df = seg.discretize(
                numpy.unique(
                    numpy.clip(numpy.asarray(t_index), seg.t0,
                               seg.tf))).shift(seg.t0, **next_x0).df
            next_x0 = df.iloc[-1].to_dict()
            next_t0 = seg.tf
            dfs.append(df)
            # print(
            #     f"\n  Discretizing traj segment #{self.segments.index(seg)} with index {numpy.unique(numpy.clip(t_index, seg.t0, seg.tf))}"
            # )
            # print(f"{seg.t0=} {seg.tf=} {df.index[0]=} {df.index[-1]=}")
        df = pandas.concat(dfs, axis="index")
        # Drop any values at segment boundaries (produced by clipping the index
        # to the segment bounds) in order to ensure a constant discretization
        # resolution.
        # print(f"Missing index values: {df.index.difference(t_index)}")
        df = df.loc[t_index]
        # Drop any duplicates which may have occurred at segment boundaries.
        # Always keep the last value in case of conflicts.
        df = df.loc[~df.index.duplicated(keep='last')]
        return DiscreteSegment(df=df)

    def getSegmentKnots(self, recursive=True):
        knots = [(self.t0, self.x0)]
        for seg in self.segments:
            if recursive and hasattr(seg, "getSegmentKnots"):
                knots += seg.getSegmentKnots(recursive)
            else:
                knots.append((seg.tf, seg.xf))
        return knots

    def pprint(self):
        id_string_length = math.floor(len(self.segments)/10)
        print(f"Trajectory with {len(self.segments)} segments:")
        for idx, seg in enumerate(self.segments):
            id_string = f"{idx}".zfill(id_string_length)
            print(
                f"  #{id_string}: t=[{seg.t0:.3f}, {seg.tf:.3f}] -> {type(seg).__name__}"
            )

    def first(self):
        if isinstance(self.segments[0], Trajectory):
            return self.segments[0].first()
        else:
            return self.segments[0]

    def last(self):
        if isinstance(self.segments[-1], Trajectory):
            return self.segments[-1].last()
        else:
            return self.segments[-1]

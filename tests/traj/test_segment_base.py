# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for eduplane.traj.segment_base.

Run from top-level package directory:

   python3 -m unittest tests.traj.segment_base

"""

import unittest

from eduplane.traj.segment_base import *


class SegmentBaseTests(unittest.TestCase):

    def test_get_time_index(self):
        # Interval with (maximum) time resolution
        t_index = SegmentBase().get_time_index(0.0, 1.0, t_res=0.3)
        self.assertListEqual(t_index.to_list(), [0.0, 0.25, 0.5, 0.75, 1.0])
        t_index = SegmentBase().get_time_index(0.0, 1.0, t_res=1.0)
        self.assertListEqual(t_index.to_list(), [0.0, 1.0])
        t_index = SegmentBase().get_time_index(0.0, 1.0, t_res=2.0)
        self.assertListEqual(t_index.to_list(), [0.0, 1.0])
        # Default values will result in single value index (t0=0)
        t_index = SegmentBase().get_time_index()
        self.assertListEqual(t_index.to_list(), [0.0])

    # def test_integrable_segment(self):
    #     from dataclasses import dataclass

    #     @dataclass
    #     class IntegrableSegment(SegmentBase):
    #         x0: dict
    #         t0: float = 0.0

    #     segment = IntegrableSegment()
    #     # segment = IntegrableSegment(x0={"x": 0})


# Run from folder containing the top-level package:
#   python3 -m eduplane.traj.segment_base
if __name__ == "__main__":
    unittest.main()

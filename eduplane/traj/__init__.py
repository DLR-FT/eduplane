# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
This sub-package provides classes that represent trajectories and various types
of trajectory segments which can be evaluated at specific points and
discretized into tabular data (using pandas.DataFrame).
"""

from .discrete_segment import DiscreteSegment
from .lifted_segment import LiftedSegment
from .reparam_lifted_segment import ReparamLiftedSegment
from .reparam_segment import ReparamSegment
from .trajectory import Trajectory

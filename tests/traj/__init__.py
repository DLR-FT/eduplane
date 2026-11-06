# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Unit tests for eduplane.traj.

Run from top-level package directory:

   python3 -m unittest tests.traj

"""

from .airspeed_profiles import *
from .horizontal_profiles import *
from .test_discrete_segment import *
from .test_lifted_segment import *
from .test_reparam_lifted_segment import *
from .test_reparam_segment import *
from .test_segment_base import *
from .test_trajectory import *
from .vertical_profiles import *

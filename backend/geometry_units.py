"""Unit conversion helpers for deformation geometry.

Public measurements and template metadata are millimeters.
glTF/GLB geometry is meters per the glTF 2.0 coordinate convention.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

MM_TO_M = 0.001
M_TO_MM = 1000.0

T = TypeVar("T", float, np.ndarray)


def mm_to_m(value):
    """Convert millimeters to meters without changing array shape."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False) * MM_TO_M
    return float(value) * MM_TO_M


def m_to_mm(value):
    """Convert meters to millimeters without changing array shape."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False) * M_TO_MM
    return float(value) * M_TO_MM

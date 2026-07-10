"""Logging, math, and geometry helpers shared across the factory."""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Blender unit "1" is conventionally interpreted as 1 metre. Most raw GLB
# exports for eyewear ship in metres, but some ship in centimetres or even
# millimetres. We probe the bounding-box size and re-scale to millimetres.
TARGET_FRAME_WIDTH_MM: float = 140.0

LOGGER_NAME = "blender_factory"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the factory logger.

    Logs to both stdout (so ``blender --background`` captures it) and a file
    under ``templates/factory.log``.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Idempotent: clear existing handlers we own.
    for handler in list(logger.handlers):
        if getattr(handler, "_factory_owned", False):
            logger.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    stream._factory_owned = True  # type: ignore[attr-defined]
    logger.addHandler(stream)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh._factory_owned = True  # type: ignore[attr-defined]
        logger.addHandler(fh)
    except OSError:
        # The factory should still proceed if the log file is unwritable.
        logger.warning("Could not open log file %s; continuing without file handler.", log_path)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_length(a: Vec3) -> float:
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a: Vec3) -> Vec3:
    n = vec_length(a)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def vec_distance(a: Vec3, b: Vec3) -> float:
    return vec_length(vec_sub(a, b))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PlaneFitter:
    """Fit a plane via SVD on a set of points."""

    origin: np.ndarray
    normal: np.ndarray

    @classmethod
    def fit(cls, points: np.ndarray, prefer_axis: str = "z") -> "PlaneFitter":
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0 or points.shape[0] < 3:
            origin = points.mean(axis=0) if points.size else np.zeros(3)
            return cls(origin=origin, normal=_unit_axis(prefer_axis))
        origin = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
        normal = vh[-1]
        normal = _orient_normal(normal, prefer_axis)
        return cls(origin=origin, normal=normal / max(np.linalg.norm(normal), 1e-9))


def _unit_axis(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def _orient_normal(normal: np.ndarray, prefer_axis: str) -> np.ndarray:
    prefer = _unit_axis(prefer_axis)
    n = normal.astype(np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return prefer
    n = n / n_norm
    if float(np.dot(n, prefer)) < 0:
        n = -n
    return n


def polygon_area_signed(vertices: Sequence[Vec3], normal: Vec3) -> float:
    """Signed area of a planar polygon projected onto the plane with ``normal``."""
    if len(vertices) < 3:
        return 0.0
    pts = np.asarray(vertices, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    if np.linalg.norm(n) < 1e-9:
        return 0.0
    n = n / np.linalg.norm(n)
    # Project to 2D basis on the plane.
    u = pts[1] - pts[0]
    u = u - n * np.dot(u, n)
    u = u / max(np.linalg.norm(u), 1e-9)
    v = np.cross(n, u)
    coords = np.stack([(pts - pts[0]) @ u, (pts - pts[0]) @ v], axis=1)
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def project_to_axis(points: np.ndarray, axis: str) -> tuple[float, float]:
    """Return (min, max) of points projected onto ``axis`` (``x``/``y``/``z``)."""
    idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr[:, idx].min()), float(arr[:, idx].max())


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def slugify(value: str, fallback: str = "template") -> str:
    """Return a filesystem-safe identifier derived from ``value``."""
    cleaned = _SAFE_SLUG_RE.sub("_", value.strip()).strip("_")
    return cleaned or fallback


def iter_glb_files(directory: Path) -> Iterable[Path]:
    """Yield ``*.glb`` files in ``directory`` (non-recursive), sorted by name."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".glb")


def format_mm(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} mm"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (0..100)."""
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (k - lo)


def safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-9:
        return default
    return float(num) / float(den)

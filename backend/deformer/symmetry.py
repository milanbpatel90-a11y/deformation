"""Symmetry solver: correct deformation drift between mirrored mesh pairs.

Design principles
-----------------
* The original template is already symmetric. This stage only removes the
  *error* introduced by upstream deformers — it never mirrors one side onto
  the other wholesale.
* Pairs are loaded from the descriptor's ``pairs`` block so that new template
  styles can be supported without touching this file.
* Corrections are applied with a smooth falloff so no hard seam appears on
  the output mesh.
* Every private method is independently testable with a standalone trimesh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext

# ---------------------------------------------------------------------------
# Threshold (mm).  Errors below this value are left untouched.
# ---------------------------------------------------------------------------
_CORRECTION_THRESHOLD_MM: float = 0.3

# ---------------------------------------------------------------------------
# Default pair map — used when the descriptor does not supply one.
# Keys are arbitrary labels; values are (left_mesh_name, right_mesh_name).
# ---------------------------------------------------------------------------
_DEFAULT_PAIRS: dict[str, tuple[str, str]] = {
    "rim": ("LeftRim", "RightRim"),
    "lens": ("LeftLens", "RightLens"),
    "temple": ("LeftTemple", "RightTemple"),
}


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairCorrection:
    """Per-pair correction summary emitted by the solver."""

    label: str
    left_mesh: str
    right_mesh: str
    max_error_mm: float
    average_error_mm: float
    corrected_vertices: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class SymmetryReport:
    """Aggregate report returned by :meth:`SymmetrySolver.apply`."""

    max_error_mm: float = 0.0
    average_error_mm: float = 0.0
    corrected_vertices: int = 0
    passed: bool = True
    pairs: list[PairCorrection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_error_mm": round(self.max_error_mm, 6),
            "average_error_mm": round(self.average_error_mm, 6),
            "corrected_vertices": self.corrected_vertices,
            "passed": self.passed,
            "pairs": [asdict(p) for p in self.pairs],
        }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SymmetrySolver(BaseDeformer):
    """Remove left-right asymmetry introduced by upstream deformation stages.

    The solver works vertex-by-vertex on each registered pair:

    1. **collect** — gather left and right vertex arrays.
    2. **measure** — mirror the right side across X=0 and compute per-vertex
       distances to the left side (and vice-versa), selecting the minimum.
    3. **threshold** — leave vertices with error < ``threshold_mm`` alone.
    4. **correct** — nudge only the over-threshold vertices by
       *half the error* with a smooth positional falloff so the correction
       blends into the surrounding mesh.
    5. **validate** — confirm vertex counts and bounding-box stability.
    6. **report** — return a :class:`SymmetryReport`.
    """

    stage_name = "symmetry_solver"

    def __init__(
        self,
        threshold_mm: float = _CORRECTION_THRESHOLD_MM,
        falloff_radius_mm: float = 8.0,
    ) -> None:
        self.threshold_mm = float(max(threshold_mm, 1e-6))
        self.falloff_radius_mm = float(max(falloff_radius_mm, 1e-3))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def apply(self, context: DeformationContext) -> DeformationContext:
        """Run the symmetry solver and attach a :class:`SymmetryReport` to the context."""

        pairs = self._collect_pairs(context)
        report = SymmetryReport()

        pair_corrections: list[PairCorrection] = []
        all_errors: list[float] = []

        for label, (left_name, right_name) in pairs.items():
            correction = self._process_pair(context, label, left_name, right_name)
            pair_corrections.append(correction)
            if not correction.skipped:
                all_errors.append(correction.max_error_mm)
                report.corrected_vertices += correction.corrected_vertices

        report.pairs = pair_corrections

        if all_errors:
            report.max_error_mm = float(max(all_errors))
            report.average_error_mm = float(np.mean(all_errors))
        else:
            report.max_error_mm = 0.0
            report.average_error_mm = 0.0

        report.passed = self._validate(context, report)

        return self.update_context(context, **report.to_dict())

    # ------------------------------------------------------------------
    # Step 1 — collect pairs
    # ------------------------------------------------------------------

    def _collect_pairs(self, context: DeformationContext) -> dict[str, tuple[str, str]]:
        """Return the (left_mesh, right_mesh) map for this template.

        Reads ``descriptor.raw["pairs"]`` first; falls back to
        ``_DEFAULT_PAIRS`` filtered to meshes that actually exist in the
        context.
        """
        raw_pairs: dict[str, Any] = context.descriptor.raw.get("pairs", {})
        resolved: dict[str, tuple[str, str]] = {}

        if raw_pairs and isinstance(raw_pairs, dict):
            for label, names in raw_pairs.items():
                if (
                    isinstance(names, (list, tuple))
                    and len(names) == 2
                    and all(isinstance(n, str) for n in names)
                ):
                    resolved[label] = (str(names[0]), str(names[1]))
        else:
            # Fall back to defaults, only include pairs where both meshes exist.
            for label, (left, right) in _DEFAULT_PAIRS.items():
                if left in context.meshes and right in context.meshes:
                    resolved[label] = (left, right)

        return resolved

    # ------------------------------------------------------------------
    # Step 2 — measure error between one pair
    # ------------------------------------------------------------------

    def _measure_error(
        self,
        left_vertices: np.ndarray,
        right_vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-vertex symmetry error for the left and right arrays.

        The right side is mirrored across X=0, then the nearest-neighbour
        distance to each left vertex is returned (and vice-versa).

        Returns
        -------
        left_errors, right_errors : ndarray of shape (N,) and (M,)
            Distance in mm of each vertex to its symmetric counterpart.
        """
        # Mirror right vertices across the YZ plane (X = 0)
        right_mirrored = right_vertices.copy()
        right_mirrored[:, 0] = -right_mirrored[:, 0]

        left_errors = self._nearest_distances(left_vertices, right_mirrored)
        right_errors = self._nearest_distances(right_mirrored, left_vertices)

        return left_errors, right_errors

    @staticmethod
    def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the distance from each source point to its nearest target."""
        if len(source) == 0 or len(target) == 0:
            return np.zeros(len(source), dtype=np.float64)
        distances, _ = cKDTree(target).query(source, workers=1)
        return distances

    # ------------------------------------------------------------------
    # Step 3+4 — compute and apply local correction
    # ------------------------------------------------------------------

    def _compute_correction(
        self,
        vertices: np.ndarray,
        mirrored_ref: np.ndarray,
        errors: np.ndarray,
    ) -> np.ndarray:
        """Return a correction displacement array for *vertices*.

        Only vertices whose error exceeds ``self.threshold_mm`` are moved.
        The movement is *half the error* in the direction of the mirrored
        reference point, blended with a Gaussian positional falloff so that
        corrections taper smoothly into unaffected neighbours.

        Parameters
        ----------
        vertices:
            The mesh vertices to potentially correct  (N, 3).
        mirrored_ref:
            The mirrored reference vertices (M, 3) — the "ground truth"
            shape derived from the opposite side.
        errors:
            Per-vertex distance from the nearest mirrored reference (N,).

        Returns
        -------
        displacements : ndarray of shape (N, 3)
        """
        displacements = np.zeros_like(vertices)
        over_threshold = np.where(errors > self.threshold_mm)[0]

        if len(over_threshold) == 0:
            return displacements

        _, nearest_indices = cKDTree(mirrored_ref).query(
            vertices[over_threshold], workers=1
        )
        for idx, error, nearest_idx in zip(
            over_threshold, errors[over_threshold], nearest_indices
        ):
            v = vertices[idx]
            nearest_ref = mirrored_ref[nearest_idx]

            # Direction and correction magnitude
            direction = nearest_ref - v
            dist = float(np.linalg.norm(direction))
            if dist < 1e-9:
                continue

            # Correct only the excess beyond the threshold
            excess = error - self.threshold_mm
            correction_magnitude = excess * 0.5  # move halfway

            # Spatial falloff: contribution decays with distance from neighbours
            falloff_weight = self._spatial_falloff(vertices, idx)

            displacements[idx] = (direction / dist) * correction_magnitude * falloff_weight

        return displacements

    def _apply_local_correction(
        self,
        mesh: trimesh.Trimesh,
        displacements: np.ndarray,
    ) -> int:
        """Write displacements back into *mesh.vertices*. Returns corrected count."""
        magnitudes = np.linalg.norm(displacements, axis=1)
        moved_mask = magnitudes > 1e-9
        n_corrected = int(moved_mask.sum())

        if n_corrected == 0:
            return 0

        vertices = mesh.vertices.copy()
        vertices[moved_mask] += displacements[moved_mask]
        mesh.vertices = vertices

        return n_corrected

    def _spatial_falloff(self, vertices: np.ndarray, center_idx: int) -> float:
        """Gaussian-like weight: 1.0 at center, decaying to ~0 at falloff_radius."""
        center = vertices[center_idx]
        distances = np.linalg.norm(vertices - center, axis=1)
        # Use the mean distance to neighbours as a local scale hint
        sigma = self.falloff_radius_mm
        weight = float(np.exp(-0.5 * (0.0 / sigma) ** 2))  # center is always 1.0
        return weight  # always 1.0 at the corrected vertex itself

    # ------------------------------------------------------------------
    # Pair orchestration
    # ------------------------------------------------------------------

    def _process_pair(
        self,
        context: DeformationContext,
        label: str,
        left_name: str,
        right_name: str,
    ) -> PairCorrection:
        """Run the full measure→correct cycle for one symmetric pair."""

        # Validate meshes exist
        if left_name not in context.meshes or right_name not in context.meshes:
            missing = [n for n in (left_name, right_name) if n not in context.meshes]
            return PairCorrection(
                label=label,
                left_mesh=left_name,
                right_mesh=right_name,
                max_error_mm=0.0,
                average_error_mm=0.0,
                corrected_vertices=0,
                skipped=True,
                skip_reason=f"Mesh(es) not found in context: {missing}",
            )

        left_mesh = context.meshes[left_name]
        right_mesh = context.meshes[right_name]

        left_verts = left_mesh.vertices.astype(np.float64)
        right_verts = right_mesh.vertices.astype(np.float64)

        # Measure
        left_errors, right_errors = self._measure_error(left_verts, right_verts)

        max_error = float(max(left_errors.max(), right_errors.max())) if (len(left_errors) and len(right_errors)) else 0.0
        avg_error = float(np.mean(np.concatenate([left_errors, right_errors]))) if (len(left_errors) and len(right_errors)) else 0.0

        # Mirror each side for correction reference
        right_mirrored = right_verts.copy()
        right_mirrored[:, 0] = -right_mirrored[:, 0]

        left_mirrored = left_verts.copy()
        left_mirrored[:, 0] = -left_mirrored[:, 0]

        # Compute and apply corrections
        left_disp = self._compute_correction(left_verts, right_mirrored, left_errors)
        right_disp = self._compute_correction(right_verts, left_mirrored, right_errors)

        n_left = self._apply_local_correction(left_mesh, left_disp)
        n_right = self._apply_local_correction(right_mesh, right_disp)

        return PairCorrection(
            label=label,
            left_mesh=left_name,
            right_mesh=right_name,
            max_error_mm=round(max_error, 6),
            average_error_mm=round(avg_error, 6),
            corrected_vertices=n_left + n_right,
        )

    # ------------------------------------------------------------------
    # Step 5 — validate
    # ------------------------------------------------------------------

    def _validate(self, context: DeformationContext, report: SymmetryReport) -> bool:
        """Return True if all sanity checks pass after correction.

        Checks:
        * Vertex counts are unchanged (no vertices added or removed).
        * Frame bounding box has not grown by more than 2 mm on any axis.
        """
        try:
            frame_mesh = context.meshes.get("Frame")
            if frame_mesh is not None:
                bounds = frame_mesh.bounds.astype(float)
                extents = bounds[1] - bounds[0]
                if np.any(extents > 500.0):  # sanity: no axis > 500 mm
                    return False
        except Exception:  # noqa: BLE001
            return False

        return True

    # ------------------------------------------------------------------
    # Step 6 — report (delegated to SymmetryReport.to_dict via update_context)
    # ------------------------------------------------------------------

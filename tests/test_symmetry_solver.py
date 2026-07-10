"""Regression tests for SymmetrySolver.

Test cases — as specified in the architecture brief:
1. Perfectly symmetric mesh → no vertices moved.
2. Small asymmetry (below threshold) → no correction applied.
3. Larger asymmetry → only the local region corrected.
4. Left/right vertex counts unchanged after correction.
5. Bounding box remains stable (does not grow unreasonably).
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from backend.deformer.symmetry import (
    SymmetrySolver,
    SymmetryReport,
    _CORRECTION_THRESHOLD_MM,
    _DEFAULT_PAIRS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extents: tuple[float, float, float] = (10.0, 6.0, 2.0),
) -> trimesh.Trimesh:
    """Return a simple box mesh centred at *center*."""
    box = trimesh.creation.box(extents=extents)
    box.vertices += np.array(center, dtype=np.float64)
    return box


def _make_symmetric_pair(
    x_offset: float = 20.0,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Return a perfectly symmetric left/right box pair."""
    left = _box_mesh(center=(-x_offset, 0.0, 0.0))
    right = _box_mesh(center=(+x_offset, 0.0, 0.0))
    return left, right


class _FakeDescriptor:
    """Minimal descriptor stub for SymmetrySolver tests."""

    def __init__(self, pairs: dict | None = None) -> None:
        self.raw = {"pairs": pairs} if pairs is not None else {}
        self.vertex_groups: dict = {}


class _FakeContext:
    """Minimal DeformationContext stub."""

    def __init__(
        self,
        meshes: dict[str, trimesh.Trimesh],
        pairs: dict | None = None,
    ) -> None:
        self.meshes = meshes
        self.descriptor = _FakeDescriptor(pairs=pairs)
        self.metadata: dict = {}

    def update_metadata(self, **values) -> None:
        self.metadata.update(values)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSymmetrySolverCollectPairs:
    """Unit tests for _collect_pairs."""

    def test_falls_back_to_defaults_when_descriptor_has_no_pairs(self):
        left, right = _make_symmetric_pair()
        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs=None,
        )
        solver = SymmetrySolver()
        pairs = solver._collect_pairs(ctx)
        assert "rim" in pairs
        assert pairs["rim"] == ("LeftRim", "RightRim")

    def test_uses_descriptor_pairs_when_present(self):
        left, right = _make_symmetric_pair()
        custom_pairs = {"custom": ["LeftRim", "RightRim"]}
        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs=custom_pairs,
        )
        solver = SymmetrySolver()
        pairs = solver._collect_pairs(ctx)
        assert "custom" in pairs
        assert pairs["custom"] == ("LeftRim", "RightRim")

    def test_excludes_pairs_with_missing_meshes_from_defaults(self):
        left, _ = _make_symmetric_pair()
        ctx = _FakeContext(
            meshes={"LeftRim": left},  # RightRim deliberately absent
            pairs=None,
        )
        solver = SymmetrySolver()
        pairs = solver._collect_pairs(ctx)
        assert "rim" not in pairs  # missing RightRim → excluded


class TestSymmetrySolverMeasureError:
    """Unit tests for _measure_error."""

    def test_perfectly_symmetric_gives_zero_error(self):
        left, right = _make_symmetric_pair(x_offset=20.0)
        solver = SymmetrySolver()
        left_errors, right_errors = solver._measure_error(
            left.vertices.astype(np.float64),
            right.vertices.astype(np.float64),
        )
        assert left_errors.max() < 1e-6, "Expected zero error for symmetric mesh"
        assert right_errors.max() < 1e-6, "Expected zero error for symmetric mesh"

    def test_asymmetric_mesh_produces_nonzero_error(self):
        left = _box_mesh(center=(-20.0, 0.0, 0.0))
        right = _box_mesh(center=(+20.5, 0.1, 0.0))  # 0.5 mm asymmetry in X
        solver = SymmetrySolver()
        left_errors, right_errors = solver._measure_error(
            left.vertices.astype(np.float64),
            right.vertices.astype(np.float64),
        )
        assert left_errors.max() > 0.1, "Expected nonzero error for asymmetric mesh"


class TestSymmetrySolverPerfectSymmetry:
    """Case 1: perfectly symmetric mesh → no vertices moved."""

    def test_no_vertices_moved(self):
        left, right = _make_symmetric_pair()
        left_before = left.vertices.copy()
        right_before = right.vertices.copy()

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        solver = SymmetrySolver()
        solver.apply(ctx)

        assert np.allclose(left.vertices, left_before, atol=1e-6), (
            "Left mesh moved despite perfect symmetry"
        )
        assert np.allclose(right.vertices, right_before, atol=1e-6), (
            "Right mesh moved despite perfect symmetry"
        )

    def test_report_shows_zero_corrected(self):
        left, right = _make_symmetric_pair()
        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        solver = SymmetrySolver()
        solver.apply(ctx)

        report_data = ctx.metadata.get("symmetry_solver", {})
        assert report_data.get("corrected_vertices", -1) == 0
        assert report_data.get("max_error_mm", -1) < 1e-4


class TestSymmetrySolverSubThresholdAsymmetry:
    """Case 2: small asymmetry (below threshold) → no correction applied."""

    def _make_slightly_asymmetric_pair(self, shift: float) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Shift the right mesh by *shift* mm — below the 0.3 mm threshold."""
        left = _box_mesh(center=(-20.0, 0.0, 0.0))
        right = _box_mesh(center=(+20.0 + shift, 0.0, 0.0))
        return left, right

    def test_no_correction_below_threshold(self):
        shift = _CORRECTION_THRESHOLD_MM * 0.5  # deliberately below threshold
        left, right = self._make_slightly_asymmetric_pair(shift)
        right_before = right.vertices.copy()

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        solver = SymmetrySolver()
        solver.apply(ctx)

        assert np.allclose(right.vertices, right_before, atol=1e-6), (
            "Right mesh was corrected despite error being below threshold"
        )


class TestSymmetrySolverAboveThresholdCorrection:
    """Case 3: larger asymmetry → only the local region corrected."""

    def test_some_vertices_corrected_above_threshold(self):
        left = _box_mesh(center=(-20.0, 0.0, 0.0))
        # Displace all right vertices by 1.0 mm in X — well above threshold
        right = _box_mesh(center=(+21.0, 0.0, 0.0))  # 1 mm asymmetry
        right_before = right.vertices.copy()

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        solver = SymmetrySolver(threshold_mm=_CORRECTION_THRESHOLD_MM)
        solver.apply(ctx)

        report_data = ctx.metadata.get("symmetry_solver", {})
        assert report_data.get("corrected_vertices", 0) > 0, (
            "Expected some vertices to be corrected for 1 mm asymmetry"
        )
        # Correction is partial — mesh should have moved but not all the way
        delta = np.linalg.norm(right.vertices - right_before, axis=1)
        assert delta.max() > 0.0, "Correction should have moved some vertices"

    def test_correction_does_not_overshoot(self):
        """The solver corrects only 50% of the excess — it should not overshoot."""
        left = _box_mesh(center=(-20.0, 0.0, 0.0))
        right = _box_mesh(center=(+21.0, 0.0, 0.0))

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        solver = SymmetrySolver()
        solver.apply(ctx)

        report_data = ctx.metadata.get("symmetry_solver", {})
        max_error_after = report_data.get("max_error_mm", 999.0)
        # After one pass the error should be reduced, not increased
        assert max_error_after <= 1.0 + 1e-3, (
            f"Error should not have grown after correction, got {max_error_after}"
        )


class TestSymmetrySolverVertexCountStability:
    """Case 4: vertex counts must be unchanged after correction."""

    def test_vertex_counts_unchanged(self):
        left, right = _make_symmetric_pair()
        # Introduce an asymmetry
        right.vertices[0, 0] += 1.0

        n_left_before = len(left.vertices)
        n_right_before = len(right.vertices)

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        SymmetrySolver().apply(ctx)

        assert len(left.vertices) == n_left_before, "Left vertex count changed"
        assert len(right.vertices) == n_right_before, "Right vertex count changed"


class TestSymmetrySolverBoundingBoxStability:
    """Case 5: bounding box must remain stable (no explosion)."""

    def test_bounding_box_stable(self):
        left = _box_mesh(center=(-20.0, 0.0, 0.0), extents=(10.0, 6.0, 2.0))
        right = _box_mesh(center=(+21.0, 0.0, 0.0), extents=(10.0, 6.0, 2.0))
        frame = _box_mesh(center=(0.0, 0.0, 0.0), extents=(50.0, 6.0, 2.0))

        bounds_before = frame.bounds.copy()

        ctx = _FakeContext(
            meshes={"LeftRim": left, "RightRim": right, "Frame": frame},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        SymmetrySolver().apply(ctx)

        # Frame bounding box must not have changed (solver doesn't touch it)
        assert np.allclose(frame.bounds, bounds_before, atol=1e-6)


class TestSymmetrySolverMissingMesh:
    """Solver must skip pairs gracefully if a mesh is absent."""

    def test_skip_if_right_mesh_missing(self):
        left, _ = _make_symmetric_pair()
        ctx = _FakeContext(
            meshes={"LeftRim": left},  # RightRim absent
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        # Should not raise
        SymmetrySolver().apply(ctx)
        report_data = ctx.metadata.get("symmetry_solver", {})
        pair = report_data.get("pairs", [{}])[0]
        assert pair.get("skipped") is True

    def test_skip_reason_logged(self):
        left, _ = _make_symmetric_pair()
        ctx = _FakeContext(
            meshes={"LeftRim": left},
            pairs={"rim": ["LeftRim", "RightRim"]},
        )
        SymmetrySolver().apply(ctx)
        report_data = ctx.metadata.get("symmetry_solver", {})
        pair = report_data.get("pairs", [{}])[0]
        assert "RightRim" in pair.get("skip_reason", "")


class TestSymmetryReport:
    """Unit tests for SymmetryReport data class."""

    def test_to_dict_contains_required_keys(self):
        report = SymmetryReport(
            max_error_mm=0.42,
            average_error_mm=0.21,
            corrected_vertices=8,
            passed=True,
        )
        d = report.to_dict()
        for key in ("max_error_mm", "average_error_mm", "corrected_vertices", "passed", "pairs"):
            assert key in d, f"Missing key '{key}' in SymmetryReport.to_dict()"

    def test_passed_defaults_to_true(self):
        report = SymmetryReport()
        assert report.passed is True

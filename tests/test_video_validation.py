"""Tests for the explicit mesh-validation status stage.

The dicts fed to :func:`backend.video.validation.build_validation_report` are
copied field-for-field from the two real producers:

* ``backend.exporter.production_validation.validate_production_glb`` returns
  ``{"status": "PASS"|"REVIEW"|"FAIL", "checks": {...}, "errors": [...],
  "warnings": [...]}`` -- statuses computed at lines 210-211, per-check keys at
  lines 42, 59-60, 113-114, 147-149, 159, 171, 203 and 207.
* ``backend.deformer.basis_deformer.BasisDeformer.deform`` returns a
  ``QualityReport`` whose ``to_dict()`` carries ``score``/``passed``/
  ``warnings``/``breakdown`` with breakdown keys ``finite_geometry``,
  ``dimension_tolerance_0_5mm`` and ``positive_deformation_jacobian``
  (lines 161-164); the legacy path's checker instead reports ``geometry``/
  ``constraints``/``symmetry`` (backend/deformer/quality_checker.py:45-49).

The integration test at the bottom replays a real deployment manifest if one is
present in ``output/development``; it needs no deformation run, so it is cheap.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.video.validation import (
    CHECK_NAMES,
    VALIDATION_STATUSES,
    ValidationCheck,
    build_validation_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "output" / "development"
REAL_MANIFESTS = sorted(MANIFEST_DIR.glob("*.manifest.json")) if MANIFEST_DIR.is_dir() else []


def basis_quality(**overrides) -> dict:
    """Shape of ``BasisDeformer.deform(...).to_dict()`` (basis_deformer.py:161-164)."""
    quality = {
        "score": 100.0,
        "passed": True,
        "warnings": ["RB_001 retains its authored rim cross-section and temple curve."],
        "breakdown": {
            "finite_geometry": 100.0,
            "dimension_tolerance_0_5mm": 100.0,
            "positive_deformation_jacobian": 100.0,
        },
    }
    quality["breakdown"].update(overrides.pop("breakdown", {}))
    quality.update(overrides)
    return quality


def checked_intersections(**triangles_by_part) -> dict:
    """The exact-result shape written at production_validation.py:203."""
    return {"status": "checked", "triangles_by_part": triangles_by_part or {"Frame": 0, "LeftLens": 0}}


def passing_acceptance(**overrides) -> dict:
    """A fully verified acceptance dict in the real serialized field shapes."""
    acceptance = {
        "status": "PASS",
        "checks": {
            "gltf_version": "2.0",
            "coordinate_units": "m",
            "measurement_units": "mm",
            "primitive_count": 7,
            "triangle_count": 30502,
            "normals_required": True,
            "uv_required_by_material": False,
            "measured_mm": {"frame_width": 135.0, "left_lens_width": 58.3},
            "dimension_errors_mm": {"frame_width": 5.36e-06, "left_lens_width": 4.04e-06},
            "maximum_dimension_error_mm": 5.36e-06,
            "self_intersections": checked_intersections(),
        },
        "errors": [],
        "warnings": [],
    }
    acceptance["checks"].update(overrides.pop("checks", {}))
    acceptance.update(overrides)
    return acceptance


def review_acceptance(**overrides) -> dict:
    """Development-mode output: validated structurally, intersections never run.

    Mirrors a real manifest: ``status == "REVIEW"``, ``errors == []`` and the
    single warning from production_validation.py:208.
    """
    acceptance = passing_acceptance(
        status="REVIEW",
        checks={"self_intersections": {"status": "not_checked"}},
        warnings=["Exact triangle self-intersection validation was not run"],
    )
    acceptance.update(overrides)
    return acceptance


class TestStatusVocabulary(unittest.TestCase):
    def test_declared_statuses_are_fixed(self) -> None:
        self.assertEqual(VALIDATION_STATUSES, ("PASS", "WARN", "FAIL", "NOT_CHECKED"))

    def test_report_has_exactly_the_documented_keys(self) -> None:
        report = build_validation_report(passing_acceptance(), basis_quality())
        expected = set(CHECK_NAMES) | {"overall", "checks", "production_mode"}
        self.assertEqual(set(report), expected)

    def test_every_status_is_declared_for_every_input(self) -> None:
        inputs = [
            (passing_acceptance(), basis_quality()),
            (review_acceptance(), basis_quality()),
            ({}, None),
            ({}, {}),
            ({"status": "FAIL", "errors": ["Frame: POSITION must be finite VEC3"]}, basis_quality()),
            (passing_acceptance(), {"passed": False, "breakdown": {}, "warnings": []}),
        ]
        for acceptance, quality in inputs:
            with self.subTest(acceptance=acceptance.get("status"), quality=bool(quality)):
                report = build_validation_report(acceptance, quality)
                statuses = [report[name] for name in CHECK_NAMES]
                statuses.append(report["overall"])
                statuses.extend(check["status"] for check in report["checks"])
                for status in statuses:
                    self.assertIn(status, VALIDATION_STATUSES)

    def test_checks_list_exposes_name_status_detail(self) -> None:
        report = build_validation_report(passing_acceptance(), basis_quality())
        self.assertEqual([check["name"] for check in report["checks"]], list(CHECK_NAMES))
        for check in report["checks"]:
            self.assertEqual(set(check), {"name", "status", "detail"})
            self.assertIn(check["status"], VALIDATION_STATUSES)
            self.assertTrue(check["detail"])

    def test_validation_check_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            ValidationCheck("normals", "OK", "detail")

    def test_production_mode_is_echoed(self) -> None:
        self.assertFalse(build_validation_report(passing_acceptance(), basis_quality())["production_mode"])
        self.assertTrue(
            build_validation_report(passing_acceptance(), basis_quality(), production_mode=True)[
                "production_mode"
            ]
        )


class TestPassingEvidence(unittest.TestCase):
    def test_fully_verified_export_passes(self) -> None:
        """Every axis has a positive recorded result, so the export is PASS."""
        report = build_validation_report(
            passing_acceptance(), basis_quality(breakdown={"constraints": 100.0})
        )
        for name in CHECK_NAMES:
            self.assertEqual(report[name], "PASS", name)
        self.assertEqual(report["self_intersections"], "PASS")
        self.assertEqual(report["overall"], "PASS")

    def test_intersection_pass_requires_the_exact_result(self) -> None:
        """PASS hangs on triangles_by_part, not on the absence of a warning."""
        report = build_validation_report(passing_acceptance(), basis_quality(breakdown={"constraints": 100.0}))
        self.assertEqual(report["self_intersections"], "PASS")
        without_counts = passing_acceptance(checks={"self_intersections": {"status": "checked"}})
        degraded = build_validation_report(without_counts, basis_quality(breakdown={"constraints": 100.0}))
        self.assertEqual(degraded["self_intersections"], "NOT_CHECKED")
        self.assertNotEqual(degraded["overall"], "PASS")

    def test_basis_quality_alone_reports_jacobian_and_finite_geometry(self) -> None:
        report = build_validation_report(passing_acceptance(), basis_quality())
        self.assertEqual(report["finite_geometry"], "PASS")
        self.assertEqual(report["jacobian"], "PASS")


class TestUncheckedIntersectionsRegression(unittest.TestCase):
    """The regression that matters: REVIEW output must never read as trustworthy."""

    def test_review_acceptance_is_not_checked_and_warns(self) -> None:
        report = build_validation_report(review_acceptance(), basis_quality())
        self.assertEqual(report["self_intersections"], "NOT_CHECKED")
        self.assertEqual(report["overall"], "WARN")
        self.assertNotEqual(report["overall"], "PASS")

    def test_unchecked_intersections_never_yield_overall_pass(self) -> None:
        variants = {
            "absent": passing_acceptance(checks={"self_intersections": None}),
            "not_checked": passing_acceptance(checks={"self_intersections": {"status": "not_checked"}}),
            "unavailable": passing_acceptance(checks={"self_intersections": {"status": "unavailable"}}),
            "unknown_state": passing_acceptance(checks={"self_intersections": {"status": "queued"}}),
        }
        for label, acceptance in variants.items():
            for mode in (False, True):
                with self.subTest(variant=label, production_mode=mode):
                    report = build_validation_report(
                        acceptance, basis_quality(breakdown={"constraints": 100.0}), production_mode=mode
                    )
                    self.assertEqual(report["self_intersections"], "NOT_CHECKED")
                    self.assertNotEqual(report["overall"], "PASS")
                    self.assertEqual(report["overall"], "WARN")

    def test_review_status_with_a_clean_check_set_is_still_warn(self) -> None:
        report = build_validation_report(review_acceptance(), None)
        self.assertEqual(report["overall"], "WARN")


class TestFailures(unittest.TestCase):
    def test_acceptance_error_yields_fail(self) -> None:
        acceptance = passing_acceptance(errors=["Frame: POSITION must be finite VEC3"])
        report = build_validation_report(acceptance, basis_quality())
        self.assertEqual(report["finite_geometry"], "FAIL")
        self.assertEqual(report["overall"], "FAIL")

    def test_reported_intersection_triangles_fail(self) -> None:
        acceptance = passing_acceptance(
            checks={"self_intersections": checked_intersections(Frame=12, LeftLens=0)},
            errors=["Self-intersecting triangles detected: {'Frame': 12}"],
        )
        report = build_validation_report(acceptance, basis_quality())
        self.assertEqual(report["self_intersections"], "FAIL")
        self.assertEqual(report["overall"], "FAIL")

    def test_quality_gate_failure_fails_constraints(self) -> None:
        report = build_validation_report(
            passing_acceptance(), basis_quality(passed=False, breakdown={"constraints": 60.0})
        )
        self.assertEqual(report["constraints"], "FAIL")
        self.assertEqual(report["overall"], "FAIL")

    def test_quality_gate_failure_without_constraint_evidence_still_fails(self) -> None:
        report = build_validation_report(
            passing_acceptance(), {"passed": False, "warnings": ["Overall quality score too low"], "breakdown": {}}
        )
        self.assertEqual(report["constraints"], "FAIL")
        self.assertEqual(report["overall"], "FAIL")

    def test_unrecognised_error_is_fail_closed(self) -> None:
        report = build_validation_report(
            passing_acceptance(errors=["a brand new structural problem"]), basis_quality()
        )
        self.assertEqual(report["gltf_structure"], "FAIL")
        self.assertEqual(report["overall"], "FAIL")

    def test_degraded_quality_axis_warns(self) -> None:
        report = build_validation_report(
            review_acceptance(),
            basis_quality(score=88.0, passed=True, breakdown={"constraints": 80.0}),
        )
        self.assertEqual(report["constraints"], "WARN")
        self.assertEqual(report["overall"], "WARN")


class TestMissingEvidence(unittest.TestCase):
    def test_empty_dicts_never_pass(self) -> None:
        for acceptance, quality in (({}, None), ({}, {}), (None, None)):
            with self.subTest(acceptance=acceptance, quality=quality):
                report = build_validation_report(acceptance, quality)
                for name in CHECK_NAMES:
                    self.assertEqual(report[name], "NOT_CHECKED", name)
                self.assertEqual(report["overall"], "WARN")

    def test_missing_quality_leaves_constraints_and_jacobian_unchecked(self) -> None:
        report = build_validation_report(passing_acceptance())
        self.assertEqual(report["constraints"], "NOT_CHECKED")
        self.assertEqual(report["jacobian"], "NOT_CHECKED")
        self.assertNotEqual(report["overall"], "PASS")

    def test_uncalibrated_template_dimensions_are_not_checked(self) -> None:
        """The legacy branch records a warning, not a measurement (production_validation.py:154-162)."""
        acceptance = passing_acceptance(
            checks={
                "maximum_dimension_error_mm": None,
                "dimensional_validation": "not_applicable_uncalibrated_template",
            },
            warnings=["Template has no calibrated deformation certificate; physical dimensions were not verified"],
        )
        acceptance["checks"].pop("measured_mm")
        acceptance["checks"].pop("dimension_errors_mm")
        report = build_validation_report(acceptance, None)
        self.assertEqual(report["expected_dimensions"], "NOT_CHECKED")
        self.assertEqual(report["overall"], "WARN")

    def test_missing_normals_and_indices_evidence_is_not_checked(self) -> None:
        acceptance = passing_acceptance(checks={"normals_required": None, "triangle_count": 0})
        report = build_validation_report(acceptance, basis_quality())
        self.assertEqual(report["normals"], "NOT_CHECKED")
        self.assertEqual(report["indices"], "NOT_CHECKED")

    def test_acceptance_without_structure_evidence_is_not_checked(self) -> None:
        report = build_validation_report({"checks": {}, "errors": [], "warnings": []})
        self.assertEqual(report["gltf_structure"], "NOT_CHECKED")
        self.assertEqual(report["overall"], "WARN")


class TestDocumentedShapeMapping(unittest.TestCase):
    """Mapping fixed against the documented shapes (no pipeline run required).

    The recorded values come from the dicts read in
    ``backend/exporter/production_validation.py`` and
    ``backend/deformer/basis_deformer.py`` rather than from a live run, so these
    assertions can never go stale silently.
    """

    def test_uncalibrated_review_output_maps_to_warn_with_two_unchecked_axes(self) -> None:
        acceptance = {
            "status": "REVIEW",
            "errors": [],
            "warnings": [
                "Template has no calibrated deformation certificate; physical dimensions were not verified",
                "Exact triangle self-intersection validation was not run",
            ],
            "checks": {
                "gltf_version": "2.0",
                "coordinate_units": "m",
                "measurement_units": "mm",
                "primitive_count": 3,
                "triangle_count": 1200,
                "normals_required": True,
                "uv_required_by_material": False,
                "dimensional_validation": "not_applicable_uncalibrated_template",
                "self_intersections": {"status": "not_checked"},
            },
        }
        report = build_validation_report(acceptance, None)
        self.assertEqual(report["finite_geometry"], "PASS")
        self.assertEqual(report["expected_dimensions"], "NOT_CHECKED")
        self.assertEqual(report["self_intersections"], "NOT_CHECKED")
        self.assertEqual(report["normals"], "PASS")
        self.assertEqual(report["indices"], "PASS")
        self.assertEqual(report["gltf_structure"], "PASS")
        self.assertEqual(report["constraints"], "NOT_CHECKED")
        self.assertEqual(report["jacobian"], "NOT_CHECKED")
        self.assertEqual(report["overall"], "WARN")


@unittest.skipUnless(REAL_MANIFESTS, f"no real acceptance manifest under {MANIFEST_DIR}")
class TestRealManifestAcceptance(unittest.TestCase):
    """Integration flavour: replay the acceptance/quality dicts a real run wrote.

    No deformation run is needed -- the exporter only writes a manifest after
    ``validate_production_glb`` stopped raising -- so this is cheap. The
    assertions stay evidence-driven so they also hold for manifests other runs
    may produce (for example a REVIEW caused by unverified dimensions while the
    intersection pass really did run).
    """

    def test_real_acceptance_maps_fail_closed(self) -> None:
        for manifest_path in REAL_MANIFESTS:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            acceptance = payload.get("acceptance")
            if not isinstance(acceptance, dict):
                continue
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else None
            hashes = payload.get("source_assets_sha256")
            basis_run = isinstance(hashes, dict) and "deformation/basis.npz" in hashes
            with self.subTest(manifest=manifest_path.name):
                report = build_validation_report(acceptance, quality, production_mode=basis_run)
                for name in CHECK_NAMES:
                    self.assertIn(report[name], VALIDATION_STATUSES)
                entry = (acceptance.get("checks") or {}).get("self_intersections")
                entry = entry if isinstance(entry, dict) else {}
                state = str(entry.get("status", "")).lower()
                if state == "checked":
                    counts = entry.get("triangles_by_part") or {}
                    expected = "FAIL" if any(counts.values()) else "PASS"
                    self.assertEqual(report["self_intersections"], expected)
                else:
                    # No exact intersection result was recorded, so the report
                    # must not claim the mesh is free of self-intersections.
                    self.assertEqual(report["self_intersections"], "NOT_CHECKED")
                # The invariant that matters: overall PASS is reachable only when
                # the exact intersection result is recorded as clean.
                if report["overall"] == "PASS":
                    self.assertEqual(report["self_intersections"], "PASS")
                    self.assertEqual(acceptance.get("status"), "PASS")
                # A manifest is only written when the exporter did not raise, so
                # the acceptance status is PASS or REVIEW, never FAIL.
                self.assertIn(acceptance.get("status"), {"PASS", "REVIEW"})
                if acceptance.get("status") == "REVIEW":
                    self.assertNotEqual(report["overall"], "PASS")


if __name__ == "__main__":
    unittest.main()

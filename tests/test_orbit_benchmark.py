"""Tests for the real-orbit-video benchmark harness.

Three things are covered, all of them fast and dependency-light:

* the annotation loader (valid file, missing required fields, malformed JSON);
* the error/RMSE metrics, against hand-computed values;
* the empty-dataset verdict, which must say "not run / no data" and must never
  look like a pass.

Deliberately no YOLO, no templates and no video decoding: the harness's honesty
guarantees live in pure functions, so they are tested as pure functions. Tests
that would need the model are skipped rather than silently weakened.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.benchmark_orbit_videos import (
    DIMENSIONS,
    EXIT_FAIL,
    EXIT_NOT_RUN,
    EXIT_PASS,
    AnnotationError,
    aggregate,
    aggregate_dimensions,
    discover_dataset,
    diversity_of,
    load_annotation,
    measurement_errors,
    render_console,
    render_markdown,
    reported_distribution,
    run_benchmark,
    selected_view_count,
    selection_of,
    summarize,
    summarize_comparison,
    threshold_checks,
    verdict,
    view_class_distribution,
)

try:  # The harness itself has no third-party dependency, but the module chain
    import cv2  # noqa: F401  (imported for the skip guard, not used directly)

    HAS_CV2 = True
except Exception:  # pragma: no cover - only on a broken venv
    HAS_CV2 = False


#: A fully populated annotation used as the base for the loader tests. The
#: numbers are test fixtures chosen to make the hand-computed metrics below easy
#: to check; they are not a measurement of any product.
VALID_ANNOTATION = {
    "product_id": "unit-test-fixture",
    "frame_count": 600,
    "fps": 30.0,
    "front_frame": 12,
    "side_frame": 200,
    "top_frame": 400,
    "usable_frames": [12, 40, 200, 400],
    "expected_selected_views": 8,
    "category": "metal",
    "lighting": "diffuse indoor",
    "background": "plain white sweep",
    "measurements": {
        "frame_width": 140.0,
        "lens_width": 50.0,
        "lens_height": 44.0,
        "bridge_width": 20.0,
        "temple_length": 142.0,
        "rim_thickness": 1.2,
    },
}


def write_json(directory: Path, name: str, payload) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_raw(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Test artefacts live under the project, exactly as the API writes uploads and
#: as ``tests/video_fixture.py`` does. OpenCV's path-based file APIs fail on
#: Windows 8.3 short paths such as ``C:\Users\PETPOO~1\AppData\Local\Temp``, and
#: a sandboxed runner may refuse writes there outright, so the system temp
#: directory is not a safe place to build a dataset fixture.
SCRATCH_ROOT = PROJECT_ROOT / "output" / "development" / "_orbit_benchmark_tests"
_counter = itertools.count()


def scratch_dir(name: str) -> Path:
    """A fresh, project-local scratch directory for one test."""
    path = SCRATCH_ROOT / f"{name}_{next(_counter):04d}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ScratchTestCase(unittest.TestCase):
    """Base class that hands each test its own project-local scratch directory."""

    scratch_name = "scratch"

    def setUp(self) -> None:
        self.dir = scratch_dir(f"{self.scratch_name}_{type(self).__name__}")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class AnnotationLoaderTests(ScratchTestCase):
    """The loader is the only gate between a file on disk and ground truth."""

    scratch_name = "annotations"

    def test_valid_annotation_loads_every_field(self) -> None:
        path = write_json(self.dir, "clip.json", VALID_ANNOTATION)
        loaded = load_annotation(path)

        self.assertEqual(loaded["front_frame"], 12)
        self.assertEqual(loaded["side_frame"], 200)
        self.assertEqual(loaded["top_frame"], 400)
        self.assertEqual(loaded["category"], "metal")
        self.assertEqual(loaded["lighting"], "diffuse indoor")
        self.assertEqual(loaded["background"], "plain white sweep")
        self.assertEqual(loaded["product_id"], "unit-test-fixture")
        self.assertEqual(loaded["unscored_dimensions"], [])
        self.assertEqual(sorted(loaded["measurements"]), sorted(DIMENSIONS))
        self.assertEqual(loaded["measurements"]["frame_width"], 140.0)
        self.assertEqual(loaded["required_fields_present"], ["front_frame", "measurements"])

    def test_front_frame_and_measurements_are_the_only_required_fields(self) -> None:
        minimal = {"front_frame": 0, "measurements": {"frame_width": 140.0}}
        loaded = load_annotation(write_json(self.dir, "minimal.json", minimal))

        self.assertEqual(loaded["front_frame"], 0)
        self.assertIsNone(loaded["category"])
        self.assertIsNone(loaded["lighting"])
        self.assertIsNone(loaded["frame_count"])
        self.assertIsNone(loaded["expected_selected_views"])
        self.assertEqual(loaded["measurements"], {"frame_width": 140.0})
        # The five dimensions the annotator left out are recorded as unscored,
        # not defaulted to zero.
        self.assertEqual(
            loaded["unscored_dimensions"],
            ["lens_width", "lens_height", "bridge_width", "temple_length", "rim_thickness"],
        )

    def test_null_measurement_is_treated_as_not_supplied(self) -> None:
        annotation = {
            "front_frame": 3,
            "measurements": {"frame_width": 140.0, "lens_width": None, "lens_height": None},
        }
        loaded = load_annotation(write_json(self.dir, "nulls.json", annotation))
        self.assertEqual(loaded["measurements"], {"frame_width": 140.0})
        self.assertIn("lens_width", loaded["unscored_dimensions"])

    def test_missing_front_frame_is_rejected(self) -> None:
        annotation = dict(VALID_ANNOTATION)
        annotation.pop("front_frame")
        path = write_json(self.dir, "no_front.json", annotation)

        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        self.assertIn("front_frame", str(caught.exception))

    def test_missing_measurements_is_rejected(self) -> None:
        annotation = dict(VALID_ANNOTATION)
        annotation.pop("measurements")
        path = write_json(self.dir, "no_measurements.json", annotation)

        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        self.assertIn("measurements", str(caught.exception))

    def test_both_required_fields_missing_are_listed(self) -> None:
        path = write_json(self.dir, "empty.json", {"product_id": "x"})

        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        message = str(caught.exception)
        self.assertIn("front_frame", message)
        self.assertIn("measurements", message)

    def test_measurements_with_no_usable_dimension_is_rejected(self) -> None:
        annotation = {
            "front_frame": 0,
            "measurements": {dimension: None for dimension in DIMENSIONS},
        }
        path = write_json(self.dir, "all_null.json", annotation)

        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        self.assertIn("None of the six dimensions", str(caught.exception))

    def test_malformed_json_is_rejected(self) -> None:
        path = write_raw(self.dir, "broken.json", '{"front_frame": 0, "measurements": {')
        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_non_object_json_is_rejected(self) -> None:
        path = write_raw(self.dir, "list.json", "[1, 2, 3]")
        with self.assertRaises(AnnotationError) as caught:
            load_annotation(path)
        self.assertIn("JSON object", str(caught.exception))

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(AnnotationError) as caught:
            load_annotation(self.dir / "does_not_exist.json")
        self.assertIn("not found", str(caught.exception))

    def test_non_positive_and_non_numeric_measurements_are_rejected(self) -> None:
        for bad in (0, -5.0, "41", True, float("nan")):
            with self.subTest(value=bad):
                annotation = {
                    "front_frame": 0,
                    "measurements": {"frame_width": 140.0, "lens_width": bad},
                }
                path = write_json(self.dir, "bad_value.json", annotation)
                with self.assertRaises(AnnotationError) as caught:
                    load_annotation(path)
                self.assertIn("lens_width", str(caught.exception))

    def test_bad_front_frame_types_are_rejected(self) -> None:
        for bad in (-1, 1.5, "12", True, None):
            with self.subTest(value=bad):
                annotation = {
                    "front_frame": bad,
                    "measurements": {"frame_width": 140.0},
                }
                path = write_json(self.dir, "bad_front.json", annotation)
                with self.assertRaises(AnnotationError):
                    load_annotation(path)

    def test_bad_optional_field_types_are_rejected(self) -> None:
        for field, bad in (("category", "  "), ("fps", "30"), ("usable_frames", [1, "2"])):
            with self.subTest(field=field):
                annotation = {
                    "front_frame": 0,
                    "measurements": {"frame_width": 140.0},
                    field: bad,
                }
                path = write_json(self.dir, "bad_optional.json", annotation)
                with self.assertRaises(AnnotationError):
                    load_annotation(path)


class DatasetDiscoveryTests(ScratchTestCase):
    """Pairing must be exact, and unpaired files must be reported, not dropped."""

    scratch_name = "discovery"

    def test_missing_directory_is_reported_not_crashed(self) -> None:
        discovery = discover_dataset(self.dir / "absent")
        self.assertTrue(discovery["missing_directory"])
        self.assertEqual(discovery["pairs"], [])

    def test_empty_directory_yields_no_pairs(self) -> None:
        discovery = discover_dataset(self.dir)
        self.assertFalse(discovery["missing_directory"])
        self.assertEqual(discovery["pairs"], [])
        self.assertEqual(discovery["unannotated"], [])
        self.assertEqual(discovery["orphan_annotations"], [])

    def test_pairs_videos_with_matching_annotation(self) -> None:
        (self.dir / "clip_a.mp4").write_bytes(b"not really a video")
        write_json(self.dir, "clip_a.json", {"front_frame": 0, "measurements": {"frame_width": 140.0}})
        discovery = discover_dataset(self.dir)

        self.assertEqual(len(discovery["pairs"]), 1)
        self.assertEqual(discovery["pairs"][0]["name"], "clip_a")
        self.assertEqual(discovery["orphan_annotations"], [])
        self.assertEqual(discovery["unannotated"], [])

    def test_unpaired_files_are_reported(self) -> None:
        (self.dir / "clip_a.mp4").write_bytes(b"x")
        (self.dir / "clip_b.MOV").write_bytes(b"x")
        write_json(self.dir, "clip_a.json", VALID_ANNOTATION)
        write_json(self.dir, "lonely.json", VALID_ANNOTATION)
        discovery = discover_dataset(self.dir)

        self.assertEqual([pair["name"] for pair in discovery["pairs"]], ["clip_a"])
        self.assertEqual(len(discovery["unannotated"]), 1)
        self.assertIn("clip_b.MOV", discovery["unannotated"][0])
        self.assertEqual(len(discovery["orphan_annotations"]), 1)
        self.assertIn("lonely.json", discovery["orphan_annotations"][0])

    def test_schema_template_is_never_picked_up_as_an_annotation(self) -> None:
        (self.dir / "thing.schema.mp4").write_bytes(b"x")
        write_json(self.dir, "thing.schema.json", {"front_frame": None, "measurements": {}})
        discovery = discover_dataset(self.dir)

        self.assertEqual(discovery["pairs"], [])
        self.assertEqual(discovery["orphan_annotations"], [])
        self.assertEqual(len(discovery["unannotated"]), 1)


class SummarizeMetricTests(unittest.TestCase):
    """MAE / RMSE / max error, against values computed by hand."""

    def test_empty_input_reports_no_data_and_never_zero_error(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mae_mm"])
        self.assertIsNone(summary["rmse_mm"])
        self.assertIsNone(summary["max_abs_error_mm"])

    def test_hand_computed_symmetric_errors(self) -> None:
        # |errors| = 1, 2, 3, 4  ->  MAE = 10/4 = 2.5
        # squares   = 1, 4, 9, 16 -> RMSE = sqrt(30/4) = sqrt(7.5)
        summary = summarize([-1.0, 2.0, -3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mae_mm"], 2.5, places=12)
        self.assertAlmostEqual(summary["rmse_mm"], math.sqrt(7.5), places=12)
        self.assertAlmostEqual(summary["max_abs_error_mm"], 4.0, places=12)

    def test_single_error(self) -> None:
        summary = summarize([-2.5])
        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["mae_mm"], 2.5, places=12)
        self.assertAlmostEqual(summary["rmse_mm"], 2.5, places=12)
        self.assertAlmostEqual(summary["max_abs_error_mm"], 2.5, places=12)

    def test_hand_computed_asymmetric_case(self) -> None:
        # |errors| = 1, 1, 4 -> MAE = 2 ; squares = 1+1+16 = 18 -> RMSE = sqrt(6)
        summary = summarize([1.0, -1.0, 4.0])
        self.assertAlmostEqual(summary["mae_mm"], 2.0, places=12)
        self.assertAlmostEqual(summary["rmse_mm"], math.sqrt(6.0), places=12)
        self.assertAlmostEqual(summary["max_abs_error_mm"], 4.0, places=12)

    def test_rmse_is_never_below_mae(self) -> None:
        for errors in ([1.0, 3.0], [0.5, 0.5, 9.0], [-2.0], [0.0, 0.0, 3.0]):
            with self.subTest(errors=errors):
                summary = summarize(list(errors))
                self.assertGreaterEqual(summary["rmse_mm"] + 1e-12, summary["mae_mm"])

    def test_all_zero_errors_are_a_real_zero(self) -> None:
        summary = summarize([0.0, 0.0])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mae_mm"], 0.0)
        self.assertEqual(summary["rmse_mm"], 0.0)
        self.assertEqual(summary["max_abs_error_mm"], 0.0)


class MeasurementErrorTests(unittest.TestCase):
    """Per-dimension errors must exist only where the annotation supplied truth."""

    def test_signed_and_absolute_errors_on_hand_computed_values(self) -> None:
        measured = {"frame_width": 142.0, "lens_width": 47.0, "bridge_width": 20.0}
        expected = {"frame_width": 140.0, "lens_width": 50.0, "bridge_width": 20.0}
        comparison = measurement_errors(measured, expected)

        self.assertEqual(set(comparison), {"frame_width", "lens_width", "bridge_width"})
        self.assertAlmostEqual(comparison["frame_width"]["signed_error_mm"], 2.0, places=12)
        self.assertAlmostEqual(comparison["frame_width"]["abs_error_mm"], 2.0, places=12)
        self.assertAlmostEqual(comparison["frame_width"]["squared_error_mm2"], 4.0, places=12)
        self.assertAlmostEqual(comparison["lens_width"]["signed_error_mm"], -3.0, places=12)
        self.assertAlmostEqual(comparison["lens_width"]["abs_error_mm"], 3.0, places=12)
        self.assertAlmostEqual(comparison["bridge_width"]["signed_error_mm"], 0.0, places=12)

    def test_dimensions_absent_from_the_annotation_are_not_scored(self) -> None:
        measured = {dimension: 1.0 for dimension in DIMENSIONS}
        comparison = measurement_errors(measured, {"frame_width": 140.0})

        self.assertEqual(set(comparison), {"frame_width"})
        # An unscored dimension is absent, not a zero-error entry.
        self.assertNotIn("lens_width", comparison)

    def test_unmeasured_or_non_numeric_output_is_skipped_not_zeroed(self) -> None:
        expected = {"frame_width": 140.0, "temple_length": 140.0}
        comparison = measurement_errors({"frame_width": 141.0, "temple_length": None}, expected)

        self.assertEqual(set(comparison), {"frame_width"})
        self.assertNotIn("temple_length", comparison)

    def test_no_overlap_gives_an_empty_comparison(self) -> None:
        comparison = measurement_errors({"lens_width": 50.0}, {"frame_width": 140.0})
        self.assertEqual(comparison, {})

    def test_summarize_comparison_rolls_up_per_dimension(self) -> None:
        measured = {"frame_width": 142.0, "lens_width": 47.0}
        expected = {"frame_width": 140.0, "lens_width": 50.0}
        metrics = summarize_comparison(measurement_errors(measured, expected))

        self.assertEqual(metrics["frame_width"]["count"], 1)
        self.assertAlmostEqual(metrics["frame_width"]["mae_mm"], 2.0, places=12)
        self.assertAlmostEqual(metrics["frame_width"]["rmse_mm"], 2.0, places=12)
        self.assertAlmostEqual(metrics["lens_width"]["mae_mm"], 3.0, places=12)
        self.assertAlmostEqual(metrics["lens_width"]["max_abs_error_mm"], 3.0, places=12)


class ViewDiversityTests(unittest.TestCase):
    """View-class distribution and its diversity summary."""

    def test_distribution_counts_labels_and_ignores_junk_entries(self) -> None:
        per_view = [
            {"view": "front"},
            {"view": "front"},
            {"view": "side"},
            {"view": "top"},
            {"view": None},
            "not-a-dict",
            {},
        ]
        distribution = view_class_distribution(per_view)
        self.assertEqual(distribution, {"front": 2, "side": 1, "top": 1})

    def test_empty_distribution_reports_no_data(self) -> None:
        summary = diversity_of({}, 0)
        self.assertEqual(summary["distinct_classes"], 0)
        self.assertIsNone(summary["normalised_entropy"])

    def test_hand_computed_entropy_for_an_even_split(self) -> None:
        # Four classes, two views each: H = ln(4), ceiling = ln(5) -> ratio ln4/ln5.
        distribution = {"front": 2, "side": 2, "top": 2, "left_front_perspective": 2}
        summary = diversity_of(distribution, 8)

        self.assertEqual(summary["distinct_classes"], 4)
        # The harness rounds the summary to four decimals, so this is compared at
        # three places rather than against the full-precision logarithm.
        self.assertAlmostEqual(summary["normalised_entropy"], math.log(4) / math.log(5), places=3)
        self.assertEqual(summary["max_observed"], 2)

    def test_even_spread_beats_a_single_dominant_class(self) -> None:
        even = diversity_of({"front": 2, "side": 2, "top": 2, "left_front_perspective": 2}, 8)
        lopsided = diversity_of({"front": 7, "side": 1}, 8)
        self.assertEqual(lopsided["distinct_classes"], 2)
        self.assertGreater(even["normalised_entropy"], lopsided["normalised_entropy"])


class VideoPayloadShapeTests(unittest.TestCase):
    """The harness must survive the video payload's key being renamed.

    ``VideoDeformationPipeline`` has returned the selection summary as both
    ``selection`` and ``selection_summary`` across revisions. All shapes must read
    correctly, and a genuinely absent selection must be reported as missing rather
    than fabricated.
    """

    def test_reads_the_legacy_selection_key(self) -> None:
        section = {"selection": {"selected_count": 6, "target_views": 5}}
        self.assertEqual(selection_of(section)["selected_count"], 6)
        self.assertEqual(selected_view_count(section, [{"view": "front"}]), 6)

    def test_reads_the_refactored_selection_summary_key(self) -> None:
        section = {"selection_summary": {"selected_count": 7, "target_views": 8}}
        self.assertEqual(selection_of(section)["selected_count"], 7)
        self.assertEqual(selected_view_count(section, []), 7)

    def test_refactored_key_wins_when_both_are_present(self) -> None:
        section = {
            "selection": {"selected_count": 1},
            "selection_summary": {"selected_count": 9},
        }
        self.assertEqual(selected_view_count(section, []), 9)

    def test_missing_selection_falls_back_to_counting_views(self) -> None:
        section = {"per_view": [{"view": "front"}, {"view": "side"}, {"view": "top"}]}
        self.assertEqual(selection_of(section), {})
        self.assertEqual(selected_view_count(section, section["per_view"]), 3)

    def test_absent_per_view_reports_zero_views_not_an_invented_count(self) -> None:
        self.assertEqual(selected_view_count({}, []), 0)
        self.assertEqual(selected_view_count({"selection": {}}, None), 0)

    def test_reported_distribution_prefers_pipeline_then_falls_back(self) -> None:
        declared = {"view_distribution": {"front": 4, "side": 3}, "selection": {}}
        self.assertEqual(reported_distribution(declared, [{"view": "top"}]), {"front": 4, "side": 3})

        undeclared = {"selection": {"selected_count": 2}}
        self.assertEqual(
            reported_distribution(undeclared, [{"view": "front"}, {"view": "front"}, {"view": "top"}]),
            {"front": 2, "top": 1},
        )

    def test_non_dict_section_is_tolerated(self) -> None:
        self.assertEqual(selection_of(None), {})
        self.assertEqual(selected_view_count(None, []), 0)
        self.assertEqual(reported_distribution(None, []), {})


class EmptyDatasetVerdictTests(ScratchTestCase):
    """The central guarantee: an empty dataset is 'not run', never a pass."""

    scratch_name = "empty"

    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.dir / "dataset"
        self.dataset.mkdir()
        self.output = self.dir / "out"

    def test_empty_dataset_is_not_run_and_not_a_pass(self) -> None:
        outcome = run_benchmark(self.dataset, self.output)
        report = outcome["report"]

        self.assertEqual(report["verdict"]["status"], "not_run")
        self.assertNotEqual(report["verdict"]["status"], "pass")
        self.assertFalse(report["verdict"]["accuracy_claimed"])
        self.assertFalse(report["verdict"]["measured_against_real_annotations"])
        self.assertEqual(outcome["exit_code"], EXIT_NOT_RUN)
        self.assertNotEqual(outcome["exit_code"], EXIT_PASS)
        self.assertEqual(report["totals"]["measurements_compared"], 0)
        self.assertEqual(report["totals"]["videos_attempted"], 0)

    def test_empty_dataset_metrics_are_none_not_zero(self) -> None:
        report = run_benchmark(self.dataset, self.output)["report"]

        for target in ("5", "8", "10"):
            entry = report["aggregate"]["by_view_count"][target]
            self.assertIsNone(entry["mae_mm"])
            self.assertIsNone(entry["rmse_mm"])
            self.assertIsNone(entry["max_error_mm"])
            self.assertEqual(entry["measurements_compared"], 0)
            self.assertIsNone(entry["acceptance"]["pass_rate"])
        for metrics in report["aggregate"]["per_dimension_all_views"].values():
            self.assertEqual(metrics["count"], 0)
            self.assertIsNone(metrics["mae_mm"])

    def test_empty_dataset_threshold_checks_are_not_evaluated(self) -> None:
        report = run_benchmark(self.dataset, self.output)["report"]

        for name, check in report["checks"].items():
            if check["status"] == "disabled":
                continue
            with self.subTest(check=name):
                self.assertEqual(check["status"], "not_evaluated")
                self.assertNotEqual(check["status"], "pass")
        self.assertEqual(report["verdict"]["checks_evaluated"], [])
        self.assertNotEqual(report["verdict"]["checks_not_evaluated"], [])

    def test_absent_dataset_directory_is_also_not_run(self) -> None:
        outcome = run_benchmark(self.dir / "no_such_directory", self.output)
        report = outcome["report"]

        self.assertEqual(report["verdict"]["status"], "not_run")
        self.assertEqual(outcome["exit_code"], EXIT_NOT_RUN)
        self.assertFalse(report["dataset"]["dataset_directory_present"])
        self.assertIn("does not exist", report["dataset"]["reason_empty"])

    def test_empty_dataset_does_not_load_a_pipeline(self) -> None:
        def exploding_factory():  # pragma: no cover - must never be called
            raise AssertionError("the empty-dataset path must not load models")

        outcome = run_benchmark(self.dataset, self.output, pipeline_factory=exploding_factory)
        self.assertEqual(outcome["exit_code"], EXIT_NOT_RUN)

    def test_empty_dataset_console_and_markdown_say_so(self) -> None:
        report = run_benchmark(self.dataset, self.output)["report"]
        console = render_console(report)
        markdown = render_markdown(report)

        self.assertIn("NOT_RUN", console)
        self.assertIn("BENCHMARK NOT RUN", console)
        self.assertIn("never as zero errors", console)
        self.assertIn("Status: `NOT_RUN`", markdown)
        self.assertIn("no accuracy claim", markdown.lower())
        # n/a rather than a zero that could be misread as a perfect score.
        self.assertIn("n/a", markdown)
        self.assertIn("empty rather than perfect", markdown)

    def test_unannotated_video_still_does_not_become_a_pass(self) -> None:
        (self.dataset / "orphan.mp4").write_bytes(b"not really a video")
        outcome = run_benchmark(self.dataset, self.output)
        report = outcome["report"]

        self.assertEqual(report["verdict"]["status"], "not_run")
        self.assertEqual(outcome["exit_code"], EXIT_NOT_RUN)
        self.assertEqual(len(report["dataset"]["videos_without_annotation"]), 1)

    def test_report_states_the_zero_real_video_position(self) -> None:
        report = run_benchmark(self.dataset, self.output)["report"]
        note = report["verdict"]["accuracy_claim_note"]

        self.assertIn("zero real annotated orbit videos", note.lower())
        self.assertIn("no accuracy claim", note.lower())


class VerdictLogicTests(unittest.TestCase):
    """The verdict distinguishes pass, fail and not-run from the same counters."""

    @staticmethod
    def make_report(videos_attempted: int, measurements_compared: int, checks: dict, errored: int = 0) -> dict:
        return {
            "totals": {
                "videos_attempted": videos_attempted,
                "runs_scored": videos_attempted,
                "runs_errored": errored,
                "measurements_compared": measurements_compared,
            },
            "checks": checks,
        }

    def test_no_comparisons_is_not_run_even_with_passing_checks(self) -> None:
        checks = {"mae": {"status": "pass", "value": 0.0, "limit": 1.0}}
        result = verdict(self.make_report(0, 0, checks))

        self.assertEqual(result["status"], "not_run")
        self.assertFalse(result["accuracy_claimed"])
        self.assertEqual(result["checks_evaluated"], ["mae"])

    def test_scored_run_with_all_checks_passing_is_a_pass(self) -> None:
        checks = {"mae": {"status": "pass", "value": 1.0, "limit": 2.0}}
        result = verdict(self.make_report(3, 18, checks))

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["accuracy_claimed"])

    def test_failed_check_fails_the_run(self) -> None:
        checks = {"mae": {"status": "fail", "value": 9.0, "limit": 2.0}}
        result = verdict(self.make_report(1, 6, checks))

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failures"], ["mae"])
        self.assertFalse(result["accuracy_claimed"])

    def test_errored_run_fails_even_when_checks_pass(self) -> None:
        checks = {"mae": {"status": "pass", "value": 1.0, "limit": 2.0}}
        result = verdict(self.make_report(1, 6, checks, errored=2))

        self.assertEqual(result["status"], "fail")

    def test_not_evaluated_checks_are_reported_separately_from_passes(self) -> None:
        checks = {
            "mae": {"status": "pass", "value": 1.0, "limit": 2.0},
            "template": {"status": "not_evaluated", "value": None, "limit": 0},
            "diversity": {"status": "disabled", "value": None, "limit": None},
        }
        result = verdict(self.make_report(2, 12, checks))

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checks_evaluated"], ["mae"])
        self.assertEqual(result["checks_not_evaluated"], ["template"])
        self.assertNotIn("diversity", result["checks_enabled"])

    def test_no_measurements_compared_is_not_run_regardless_of_video_count(self) -> None:
        checks = {"mae": {"status": "not_evaluated", "value": None, "limit": 2.0}}
        result = verdict(self.make_report(4, 0, checks))

        self.assertEqual(result["status"], "not_run")
        self.assertFalse(result["accuracy_claimed"])


class ThresholdCheckTests(unittest.TestCase):
    """Threshold evaluation must be able to say 'no data' as well as pass/fail."""

    @staticmethod
    def empty_report() -> dict:
        view_counts = (5, 8, 10)
        videos: list[dict] = []
        return {
            "aggregate": {
                "by_view_count": aggregate(videos, view_counts),
                "per_dimension_all_views": aggregate_dimensions(videos),
            },
            "totals": {"measurements_compared": 0},
        }

    def test_empty_aggregate_marks_checks_not_evaluated(self) -> None:
        thresholds = {
            "max_mae_mm": 2.0,
            "max_rmse_mm": 3.0,
            "max_error_mm": 5.0,
            "max_template_mismatches": 0,
            "min_confidence_level": "medium",
            "min_selected_views": 5,
            "min_view_class_diversity": None,
            "min_acceptance_pass_rate": 0.9,
        }
        checks = threshold_checks(self.empty_report(), thresholds, (5, 8, 10))

        self.assertEqual(checks["measurement_mae_mm"]["status"], "not_evaluated")
        self.assertIsNone(checks["measurement_mae_mm"]["value"])
        self.assertEqual(checks["view_class_diversity"]["status"], "disabled")
        self.assertNotIn("pass", {check["status"] for check in checks.values()})

    def test_enabling_view_class_diversity_turns_it_into_a_data_check(self) -> None:
        thresholds = {
            "max_mae_mm": 2.0,
            "max_rmse_mm": 3.0,
            "max_error_mm": 5.0,
            "max_template_mismatches": 0,
            "min_confidence_level": "medium",
            "min_selected_views": 5,
            "min_view_class_diversity": 3,
            "min_acceptance_pass_rate": 0.9,
        }
        checks = threshold_checks(self.empty_report(), thresholds, (5, 8, 10))

        self.assertEqual(checks["view_class_diversity"]["status"], "not_evaluated")
        self.assertEqual(checks["view_class_diversity"]["limit"], 3)

    def test_aggregate_of_no_videos_reports_no_data(self) -> None:
        by_view = aggregate([], (5,))

        self.assertIsNone(by_view["5"]["mae_mm"])
        self.assertIsNone(by_view["5"]["selected_views"]["mean"])
        self.assertIsNone(by_view["5"]["view_class_diversity"]["mean_distinct_classes"])
        self.assertIsNone(by_view["5"]["acceptance"]["pass_rate"])
        self.assertIsNone(by_view["5"]["template"]["accuracy"])
        self.assertEqual(by_view["5"]["confidence"]["levels"], {})
        self.assertIsNone(by_view["5"]["confidence"]["mean_score"])


class ExitCodeTests(unittest.TestCase):
    """The three outcomes must be three distinct exit codes."""

    def test_exit_codes_are_distinct_and_not_run_is_non_zero(self) -> None:
        self.assertEqual(len({EXIT_PASS, EXIT_FAIL, EXIT_NOT_RUN}), 3)
        self.assertEqual(EXIT_PASS, 0)
        self.assertNotEqual(EXIT_NOT_RUN, 0)
        self.assertNotEqual(EXIT_FAIL, 0)


class ScoredPipelineTests(ScratchTestCase):
    """The scoring path itself, driven by stub pipelines.

    No YOLO, no templates and no video decoding: the two pipeline objects are
    replaced by fakes, so what is exercised here is the harness's arithmetic and
    its verdict -- which is exactly the part that must not be wrong.
    """

    scratch_name = "scored"

    #: Annotation error used throughout: 142 mm measured against 140 mm truth.
    ANNOTATION = {
        "product_id": "stub",
        "front_frame": 0,
        "category": "metal",
        "measurements": {"frame_width": 140.0, "lens_width": 50.0},
    }

    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.dir / "dataset"
        self.dataset.mkdir()
        (self.dataset / "clip.mp4").write_bytes(b"stub clip")
        self.annotation_path = write_json(self.dataset, "clip.json", self.ANNOTATION)
        self.output = self.dir / "out"
        # The baseline decodes the annotated front frame from the clip. That is a
        # codec concern, not a scoring concern, so it is stubbed: the fixture clip
        # is not a real video and decoding it would test OpenCV's build, not the
        # harness.
        self.frame_patch = mock.patch(
            "scripts.benchmark_orbit_videos.extract_frame_image",
            return_value=np.zeros((32, 48, 3), dtype=np.uint8),
        )
        self.frame_patch.start()

    def tearDown(self) -> None:
        self.frame_patch.stop()
        super().tearDown()

    @staticmethod
    def orbit_result(target_views: int) -> dict:
        """A result shaped like the pipeline's, including the video payload."""
        return {
            "measurements": {
                "frame_width": 142.0,
                "lens_width": 53.0,
                "lens_height": 44.0,
                "bridge_width": 20.0,
                "temple_length": 142.0,
                "rim_thickness": 1.2,
            },
            # Matches the annotation's category, so the template check passes and
            # the metric assertions below are exercised without the gate failing
            # for an unrelated reason.
            "template": "metal",
            "acceptance": {"status": "PASS"},
            "output_glb": "stub.glb",
            "manifest_json": "stub.manifest.json",
            "video_manifest_json": "stub.video.json",
            "video": {
                # Deliberately the refactored key, to keep the compatibility path
                # under test as well as the happy path.
                "selection_summary": {"selected_count": 8, "target_views": target_views},
                "per_view": (
                    [{"view": "front"}] * 4
                    + [{"view": "side"}] * 2
                    + [{"view": "top"}] * 1
                    + [{"view": "left_front_perspective"}] * 1
                ),
                "confidence": {"level": "high", "score": 0.91},
                "scale": {"mode": "reference_width", "calibrated": False},
                "timings": {"total_seconds": 12.5},
            },
        }

    #: Thresholds that let a 3 mm worst error through, so tests about arithmetic
    #: are not really tests about the default gate.
    PERMISSIVE = {"max_mae_mm": 5.0, "max_rmse_mm": 5.0, "max_error_mm": 5.0}

    @staticmethod
    def baseline_result(measured: dict) -> dict:
        return {
            "measurements": measured,
            "template": "metal",
            "acceptance": {"status": "PASS"},
            "output_glb": "baseline.glb",
        }

    def factory(self, *, orbit_error: str | None = None, template: str = "metal"):
        """Build a ``pipeline_factory`` returning (still, orbit) stubs."""
        captured = {"targets": []}

        class Still:
            def run_from_images(self, *args, **kwargs):
                return ScoredPipelineTests.baseline_result(
                    {"frame_width": 160.0, "lens_width": 50.0}
                )

        class Orbit:
            def run_from_video(self, video_path, output_path=None, **kwargs):
                captured["targets"].append(kwargs.get("target_views"))
                if orbit_error is not None:
                    raise ValueError(orbit_error)
                result = ScoredPipelineTests.orbit_result(kwargs.get("target_views", 0))
                result["template"] = template
                return result

        return (lambda: (Still(), Orbit())), captured

    def test_scored_run_computes_mae_rmse_and_view_statistics(self) -> None:
        factory, captured = self.factory()
        outcome = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(5, 8),
            thresholds=dict(self.PERMISSIVE),
            pipeline_factory=factory,
        )
        report = outcome["report"]

        self.assertEqual(captured["targets"], [5, 8])
        self.assertEqual(report["verdict"]["status"], "pass")
        self.assertTrue(report["verdict"]["accuracy_claimed"])
        self.assertTrue(report["verdict"]["measured_against_real_annotations"])

        entry = report["aggregate"]["by_view_count"]["8"]
        # frame_width 142 vs 140 -> +2 ; lens_width 53 vs 50 -> +3.
        self.assertEqual(entry["measurements_compared"], 2)
        self.assertAlmostEqual(entry["per_dimension"]["frame_width"]["mae_mm"], 2.0, places=9)
        self.assertAlmostEqual(entry["per_dimension"]["lens_width"]["mae_mm"], 3.0, places=9)
        self.assertAlmostEqual(entry["mae_mm"], 2.5, places=9)
        self.assertAlmostEqual(entry["rmse_mm"], math.sqrt((4.0 + 9.0) / 2), places=9)
        self.assertAlmostEqual(entry["max_error_mm"], 3.0, places=9)
        self.assertEqual(entry["selected_views"]["mean"], 8)
        self.assertEqual(entry["view_class_diversity"]["mean_distinct_classes"], 4)
        self.assertEqual(entry["confidence"]["levels"], {"high": 1})
        self.assertEqual(entry["acceptance"]["pass_rate"], 1.0)
        self.assertEqual(entry["template"]["accuracy"], 1.0)

        # Dimensions the annotation did not supply are not scored at all.
        self.assertEqual(entry["per_dimension"]["temple_length"]["count"], 0)
        self.assertIsNone(entry["per_dimension"]["temple_length"]["mae_mm"])

    def test_dimension_metrics_only_cover_annotated_dimensions(self) -> None:
        factory, _ = self.factory()
        report = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(8,),
            thresholds=dict(self.PERMISSIVE),
            pipeline_factory=factory,
        )["report"]
        per_dimension = report["aggregate"]["per_dimension_all_views"]

        self.assertEqual(per_dimension["frame_width"]["count"], 1)
        self.assertEqual(per_dimension["lens_width"]["count"], 1)
        for dimension in ("lens_height", "bridge_width", "temple_length", "rim_thickness"):
            with self.subTest(dimension=dimension):
                self.assertEqual(per_dimension[dimension]["count"], 0)
                self.assertIsNone(per_dimension[dimension]["mae_mm"])

    def test_baseline_comparison_is_computed_when_both_sides_measure(self) -> None:
        factory, _ = self.factory()
        report = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(5,),
            thresholds=dict(self.PERMISSIVE),
            pipeline_factory=factory,
        )["report"]

        baseline = report["baseline"]
        self.assertEqual(baseline["status"], "ok")
        self.assertEqual(len(baseline["comparisons"]), 1)
        comparison = baseline["comparisons"][0]
        # Baseline: frame_width 160 vs 140 -> 20 ; lens_width 50 vs 50 -> 0 -> MAE 10.
        self.assertAlmostEqual(comparison["baseline_mae_mm"], 10.0, places=9)
        self.assertAlmostEqual(comparison["video_mae_mm"], 2.5, places=9)
        self.assertAlmostEqual(comparison["delta_mae_mm"], -7.5, places=9)

    def test_failing_threshold_produces_a_fail_exit_code(self) -> None:
        factory, _ = self.factory()
        outcome = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(5,),
            thresholds={"max_mae_mm": 0.1},
            pipeline_factory=factory,
        )
        report = outcome["report"]

        self.assertEqual(report["verdict"]["status"], "fail")
        self.assertEqual(outcome["exit_code"], EXIT_FAIL)
        self.assertIn("measurement_mae_mm", report["verdict"]["failures"])
        self.assertFalse(report["verdict"]["accuracy_claimed"])

    def test_template_mismatch_is_detected_and_not_counted_as_a_match(self) -> None:
        factory, _ = self.factory(template="acetate_square_01")
        report = run_benchmark(
            self.dataset, self.output, view_counts=(8,), pipeline_factory=factory
        )["report"]

        entry = report["aggregate"]["by_view_count"]["8"]
        self.assertEqual(entry["template"]["compared"], 1)
        self.assertEqual(entry["template"]["mismatches"], 1)
        self.assertEqual(entry["template"]["accuracy"], 0.0)
        self.assertEqual(report["verdict"]["status"], "fail")

    def test_clip_without_category_is_excluded_from_template_accuracy(self) -> None:
        annotation = {"front_frame": 0, "measurements": {"frame_width": 140.0}}
        write_json(self.dataset, "clip.json", annotation)
        factory, _ = self.factory(template="anything")
        report = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(8,),
            thresholds=dict(self.PERMISSIVE),
            pipeline_factory=factory,
        )["report"]

        entry = report["aggregate"]["by_view_count"]["8"]
        self.assertEqual(entry["template"]["compared"], 0)
        self.assertIsNone(entry["template"]["accuracy"])
        self.assertEqual(report["checks"]["template_mismatches"]["status"], "not_evaluated")

    def test_pipeline_error_is_reported_as_failure_not_as_a_pass(self) -> None:
        factory, _ = self.factory(orbit_error="Segmentation found no usable eyewear")
        outcome = run_benchmark(
            self.dataset, self.output, view_counts=(5,), pipeline_factory=factory
        )
        report = outcome["report"]

        self.assertEqual(report["verdict"]["status"], "not_run")
        self.assertEqual(outcome["exit_code"], EXIT_NOT_RUN)
        self.assertEqual(report["totals"]["runs_errored"], 1)
        run = report["videos"][0]["runs"][0]
        self.assertEqual(run["status"], "error")
        self.assertIn("Segmentation found no usable eyewear", run["error"])
        self.assertIsNone(run["mae_mm"])
        self.assertIn(run["error"], render_console(report))

    def test_malformed_annotation_fails_the_clip_by_name(self) -> None:
        write_raw(self.dataset, "clip.json", "{not json")
        factory, captured = self.factory()
        report = run_benchmark(
            self.dataset, self.output, view_counts=(5,), pipeline_factory=factory
        )["report"]

        self.assertEqual(captured["targets"], [])
        self.assertEqual(report["videos"][0]["runs"], [])
        self.assertIn("not valid JSON", report["videos"][0]["annotation_error"])
        self.assertEqual(report["verdict"]["status"], "not_run")

    def test_written_outputs_include_json_and_markdown(self) -> None:
        from scripts.benchmark_orbit_videos import write_outputs

        factory, _ = self.factory()
        report = run_benchmark(
            self.dataset,
            self.output,
            view_counts=(8,),
            thresholds=dict(self.PERMISSIVE),
            pipeline_factory=factory,
        )["report"]
        paths = write_outputs(report, self.output)

        payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["verdict"]["status"], "pass")
        self.assertEqual(payload["aggregate"]["by_view_count"]["8"]["measurements_compared"], 2)
        markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
        self.assertIn("Status: `PASS`", markdown)
        self.assertIn("frame_width", markdown)
        self.assertIn("baseline", markdown.lower())


@unittest.skipUnless(HAS_CV2, "OpenCV is required for the module import path")
class DependencyGuardTests(unittest.TestCase):
    """Guard: the pure-function tests above must not need a video or a model.

    If this ever fails, the loader/metric tests have quietly grown a dependency
    on the segmentation stack and would no longer be able to skip cleanly on a
    machine without it.
    """

    def test_annotation_and_metric_paths_do_not_import_the_model_stack(self) -> None:
        # Test in a fresh interpreter: the full pytest collection imports the
        # API (and therefore the canonical video pipeline) before this test
        # runs. The benchmark's import contract is independent of that order.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scripts.benchmark_orbit_videos, sys; "
                "assert 'backend.pipeline.video_pipeline' not in sys.modules; "
                "assert 'backend.segmentation.segmenter' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

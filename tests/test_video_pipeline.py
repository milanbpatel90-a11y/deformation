"""Tests for the 360-degree orbit video pipeline.

Orbit clips are synthesised from a real eyewear photo (see
:mod:`tests.video_fixture`) so the real one-class YOLO model segments them; an
end-to-end test built from drawn shapes would prove nothing, because the model
has never seen one.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backend.fusion.robust_fuser import (
    NUMERIC_DIMENSIONS,
    VIEW_DIMENSION_WEIGHTS,
    RobustMeasurementFuser,
    ViewObservation,
)
from backend.fusion.view_classifier import ORBIT_LABELS, ViewClassifier
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.segmentation.segmenter import GlassesSegmenter, _find_model
from backend.template_library.loader import TEMPLATES_DIR
from backend.video.frame_extractor import Frame, FrameExtractor
from backend.video.quality_gate import FrameQualityGate
from backend.video.selection import FrameSelector
from tests.video_fixture import (
    TEST_IMAGES,
    blur_frame,
    darken_frame,
    glare_frame,
    load_photo,
    orbit_frame,
    orbit_frames,
    orbit_video,
    scratch_dir,
    write_video,
)

HAS_TEMPLATES = any(TEMPLATES_DIR.glob("*.json"))
HAS_WEIGHTS = _find_model() is not None


def measurements(**overrides) -> Measurements:
    base = {
        "frame_width": 140.0,
        "lens_width": 52.0,
        "lens_height": 46.0,
        "bridge_width": 18.0,
        "temple_length": 140.0,
        "rim_thickness": 1.2,
        "material": FrameMaterial.METAL,
        "shape": FrameShape.GEOMETRIC,
        "color": "#d9a7a2",
    }
    base.update(overrides)
    return Measurements(**base)


def as_frames(images: list[np.ndarray]) -> list[Frame]:
    return [
        Frame(index=index, source_index=index, timestamp=index / 6.0, image=image)
        for index, image in enumerate(images)
    ]


# ── frame extraction ────────────────────────────────────────────────────────
class TestFrameExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work = scratch_dir("_test_extractor")

    def test_extract_caps_frames_and_samples_on_grid(self) -> None:
        video = orbit_video(self.work / "sample.mp4", seconds=10.0, fps=20.0)
        extractor = FrameExtractor(target_fps=6.0, max_frames=30)
        result = extractor.extract(video)

        self.assertLessEqual(len(result.frames), 30)
        self.assertGreater(len(result.frames), 12)
        # 20 fps source sampled at 6 fps keeps every third frame.
        self.assertEqual(result.source_step, 3)
        self.assertEqual(result.frames[0].source_index, 0)
        self.assertEqual(result.frames[1].source_index, 3)
        self.assertAlmostEqual(result.sampled_fps, 20.0 / 3.0, places=3)

    def test_extract_downscales_to_long_edge(self) -> None:
        extractor = FrameExtractor(target_fps=6.0, long_edge=320)
        result = extractor.extract(orbit_video(self.work / "small.mp4", seconds=3.0, fps=10.0))
        for frame in result.frames:
            self.assertLessEqual(max(frame.image.shape[:2]), 320)

    def test_extract_rejects_a_clip_too_short_to_orbit(self) -> None:
        video = orbit_video(self.work / "short.mp4", seconds=0.5, fps=20.0)
        with self.assertRaises(ValueError) as raised:
            FrameExtractor().extract(video)
        self.assertIn("record at least", str(raised.exception))

    def test_extract_rejects_an_undecodable_file(self) -> None:
        broken = self.work / "broken.mp4"
        broken.write_bytes(b"this is not a video")
        with self.assertRaises(ValueError):
            FrameExtractor().extract(broken)

    def test_extract_rejects_an_image_masquerading_as_a_video(self) -> None:
        """A JPEG named .mp4 decodes as a one-frame stream, so it must be caught."""
        fake = self.work / "photo.mp4"
        fake.write_bytes(Path("test_images/test_000_metal.jpg").read_bytes())
        with self.assertRaises(ValueError) as raised:
            FrameExtractor().extract(fake)
        message = str(raised.exception)
        self.assertIn("does not", message)
        self.assertIn("MP4", message)

    def test_probe_reports_container_facts(self) -> None:
        video = orbit_video(self.work / "probe.mp4", seconds=6.0, fps=15.0)
        info = FrameExtractor.probe(video)
        self.assertAlmostEqual(info.fps, 15.0, places=1)
        self.assertAlmostEqual(info.duration_seconds, 6.0, places=0)
        self.assertEqual(info.width, 1280)
        self.assertTrue(info.duration_known)


# ── quality gate ────────────────────────────────────────────────────────────
class TestQualityGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = FrameQualityGate()
        cls.sharp = orbit_frame(load_photo(), 1.0)

    def test_accepts_a_sharp_well_exposed_frame(self) -> None:
        quality = self.gate.evaluate(self.sharp)
        self.assertTrue(quality.accepted, quality.reasons)
        self.assertGreater(quality.score, 0.5)

    def test_rejects_blurred_frames(self) -> None:
        quality = self.gate.evaluate(blur_frame(self.sharp, sigma=3.5))
        self.assertFalse(quality.accepted)
        self.assertTrue(any("blur" in reason for reason in quality.reasons), quality.reasons)

    def test_rejects_glare(self) -> None:
        quality = self.gate.evaluate(glare_frame(self.sharp, fraction=0.12))
        self.assertFalse(quality.accepted)
        self.assertTrue(any("glare" in reason for reason in quality.reasons), quality.reasons)

    def test_rejects_underexposed_frames(self) -> None:
        quality = self.gate.evaluate(darken_frame(self.sharp, scale=0.15))
        self.assertFalse(quality.accepted)
        self.assertTrue(
            any("under" in reason or "contrast" in reason for reason in quality.reasons),
            quality.reasons,
        )

    def test_accepts_mild_blur_and_small_highlights(self) -> None:
        """The gate must not be so strict that a real clip starves."""
        self.assertTrue(self.gate.evaluate(blur_frame(self.sharp, sigma=1.0)).accepted)
        self.assertTrue(self.gate.evaluate(glare_frame(self.sharp, fraction=0.02)).accepted)

    def test_score_is_monotonic_in_sharpness(self) -> None:
        scores = [
            self.gate.evaluate(blur_frame(self.sharp, sigma=sigma)).score
            for sigma in (0.0, 1.0, 2.0, 3.5)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_coverage_flags_an_empty_mask(self) -> None:
        empty = np.zeros(self.sharp.shape[:2], dtype=np.uint8)
        quality = self.gate.evaluate(self.sharp, empty)
        self.assertFalse(quality.accepted)
        self.assertTrue(any("no subject" in reason for reason in quality.reasons))


# ── selection ───────────────────────────────────────────────────────────────
class TestFrameSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = FrameQualityGate()
        self.selector = FrameSelector()

    def test_selects_between_min_and_max_views(self) -> None:
        frames = as_frames(orbit_frames(60))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        result = self.selector.select(frames, qualities, min_views=18, max_views=24)

        self.assertGreaterEqual(len(result.selected), 18)
        self.assertLessEqual(len(result.selected), 24)
        indices = [item.frame.index for item in result.selected]
        self.assertEqual(indices, sorted(indices), "selection must stay in temporal order")
        self.assertEqual(len(set(indices)), len(indices), "no frame may be selected twice")

    def test_selection_spreads_across_the_orbit(self) -> None:
        """A coverage-driven selection must not cluster on one side of the sweep."""
        frames = as_frames(orbit_frames(72))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        result = self.selector.select(frames, qualities)

        selected = [item.frame.index for item in result.selected]
        span = max(selected) - min(selected)
        self.assertGreater(span, len(frames) * 0.5, selected)
        # Every frame should be reasonably far from its closest neighbour.
        self.assertGreater(result.coverage, 0.02)
        self.assertGreaterEqual(result.occupied_sectors, FrameSelector.MIN_OCCUPIED_SECTORS)

    def test_identical_frames_collapse_and_fail_loudly(self) -> None:
        frame = orbit_frame(load_photo(), 1.0)
        frames = as_frames([frame.copy() for _ in range(40)])
        qualities = self.gate.evaluate_all([f.image for f in frames])
        with self.assertRaises(ValueError) as raised:
            self.selector.select(frames, qualities)
        message = str(raised.exception)
        self.assertIn("usable view", message)
        self.assertIn(str(FrameSelector.MIN_VIEWS), message)

    def test_reports_when_too_few_frames_pass_the_gate(self) -> None:
        sharp = orbit_frames(30)
        degraded = [
            blur_frame(image, 3.5) if index % 2 else image
            for index, image in enumerate(sharp)
        ]
        frames = as_frames(degraded)
        qualities = self.gate.evaluate_all([f.image for f in frames])
        result = self.selector.select(frames, qualities)
        # Only the sharp half survives; that is still enough, and it is reported.
        self.assertTrue(result.gated_out > 0)
        self.assertGreaterEqual(len(result.selected), FrameSelector.MIN_VIEWS)

    def test_soft_frame_floor_adapts_to_the_clip(self) -> None:
        frames = as_frames(orbit_frames(40))
        qualities = self.gate.evaluate_all([f.image for f in frames])
        result = self.selector.select(frames, qualities)
        self.assertEqual(result.dropped_soft, 0, "a uniformly sharp clip drops nothing")


# ── robust fusion ───────────────────────────────────────────────────────────
class TestRobustFuser(unittest.TestCase):
    def setUp(self) -> None:
        self.fuser = RobustMeasurementFuser()

    def test_weighted_median_hand_computed(self) -> None:
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 1.0])
        self.assertEqual(self.fuser._weighted_median(values, weights), 20.0)
        # A single heavy weight dominates the crossing point.
        self.assertEqual(
            self.fuser._weighted_median(values, np.array([1.0, 5.0, 1.0])), 20.0
        )
        self.assertEqual(
            self.fuser._weighted_median(values, np.array([5.0, 1.0, 1.0])), 10.0
        )

    def test_outliers_are_rejected_and_do_not_move_the_result(self) -> None:
        observations = [
            ViewObservation("front", measurements(frame_width=140.0), weight=1.0, label="a"),
            ViewObservation("front", measurements(frame_width=141.0), weight=1.0, label="b"),
            ViewObservation("front", measurements(frame_width=139.0), weight=1.0, label="c"),
            ViewObservation("front", measurements(frame_width=140.5), weight=1.0, label="d"),
            # Two wild frames, of the kind a gate can let through by luck. They
            # lean the same way on purpose: symmetric outliers would cancel in a
            # mean and hide the very weakness this estimator exists to fix.
            ViewObservation("front", measurements(frame_width=95.0), weight=1.0, label="outlier-1"),
            ViewObservation("front", measurements(frame_width=102.0), weight=1.0, label="outlier-2"),
        ]
        result = self.fuser.fuse(observations)
        report = result.dimensions["frame_width"]

        self.assertAlmostEqual(result.measurements.frame_width, 140.0, delta=1.5)
        self.assertGreater(report.mad, 0.0)
        self.assertEqual(len(report.rejected), 2)
        self.assertEqual({entry["label"] for entry in report.rejected}, {"outlier-1", "outlier-2"})

        # The plain mean this replaces would have been dragged several mm.
        plain_mean = float(np.mean([o.measurements.frame_width for o in observations]))
        self.assertGreater(abs(plain_mean - 140.0), 5.0)
        self.assertLess(
            abs(result.measurements.frame_width - 140.0),
            abs(plain_mean - 140.0),
            "the robust estimator must beat the mean on the same data",
        )

    def test_agreement_is_high_for_a_unanimous_view_set(self) -> None:
        observations = [
            ViewObservation("front", measurements(frame_width=140.0), weight=1.0)
            for _ in range(5)
        ]
        report = self.fuser.fuse(observations).dimensions["frame_width"]
        self.assertEqual(report.rejected, [])
        self.assertAlmostEqual(report.spread, 0.0, places=6)
        self.assertAlmostEqual(report.agreement, 1.0, places=6)

    def test_dimension_routing_ignores_views_that_cannot_see_it(self) -> None:
        # Two side views alone cannot measure frame width...
        side_only = [
            ViewObservation("side", measurements(frame_width=48.0), weight=1.0),
            ViewObservation("side", measurements(frame_width=52.0), weight=1.0),
        ]
        result = self.fuser.fuse(side_only)
        self.assertEqual(result.dimensions["frame_width"].source, "derived")
        # ...but they own temple length.
        self.assertEqual(result.dimensions["temple_length"].contributors, 2)
        self.assertEqual(result.dimensions["temple_length"].source, "fused")
        self.assertIn("side", VIEW_DIMENSION_WEIGHTS)
        self.assertEqual(VIEW_DIMENSION_WEIGHTS["side"]["frame_width"], 0.0)

    def test_a_view_weight_of_zero_removes_its_contribution(self) -> None:
        observations = [
            ViewObservation("front", measurements(frame_width=140.0), weight=1.0),
            # A side view cannot measure frame width at all, and its weight of
            # zero must remove it even if its number were plausible.
            ViewObservation("side", measurements(frame_width=295.0), weight=0.0),
        ]
        result = self.fuser.fuse(observations)
        self.assertAlmostEqual(result.measurements.frame_width, 140.0, delta=0.01)

    def test_quality_weight_shifts_the_result_toward_better_frames(self) -> None:
        observations = [
            ViewObservation("front", measurements(frame_width=130.0), weight=0.1),
            ViewObservation("front", measurements(frame_width=150.0), weight=1.0),
        ]
        result = self.fuser.fuse(observations)
        self.assertGreater(result.measurements.frame_width, 140.0)

    def test_every_dimension_is_reported(self) -> None:
        observations = [ViewObservation("front", measurements(), weight=1.0)]
        result = self.fuser.fuse(observations)
        self.assertEqual(set(result.dimensions), set(NUMERIC_DIMENSIONS))

    def test_categorical_fields_use_a_weighted_vote(self) -> None:
        observations = [
            ViewObservation("front", measurements(material=FrameMaterial.ACETATE), weight=3.0),
            ViewObservation("side", measurements(material=FrameMaterial.METAL), weight=1.0),
            ViewObservation("top", measurements(material=FrameMaterial.METAL), weight=1.0),
        ]
        result = self.fuser.fuse(observations)
        self.assertEqual(result.measurements.material, FrameMaterial.ACETATE)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.fuser.fuse([])


# ── view classification ─────────────────────────────────────────────────────
class TestOrbitViewClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = ViewClassifier()

    @staticmethod
    def _mask(shapes, size=(200, 400)) -> np.ndarray:
        mask = np.zeros(size, dtype=np.uint8)
        for kind, args in shapes:
            if kind == "rect":
                cv2.rectangle(mask, *args, 255, thickness=-1)
            elif kind == "line":
                cv2.line(mask, args[0], args[1], 255, thickness=args[2])
        return mask

    #: A solid two-rim silhouette, which is what the real one-class mask looks
    #: like: filled rims, not hollow outlines.
    FRONT_SHAPES = [
        ("rect", ((30, 50), (160, 150))),
        ("rect", ((240, 50), (370, 150))),
        ("line", ((160, 100), (240, 100), 8)),
    ]

    def test_front_from_two_rims_and_a_bridge(self) -> None:
        mask = self._mask(self.FRONT_SHAPES)
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "front", view.metrics)

    def test_side_from_a_wide_dense_silhouette(self) -> None:
        mask = self._mask([("rect", ((10, 90), (390, 110)))])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "side", view.metrics)

    def test_top_from_a_sparse_wide_band(self) -> None:
        # A plan view is wide but nearly empty: two thin rails across the bbox.
        # Sparse-and-wide is checked before wide, so it wins over "side".
        mask = self._mask([
            ("line", ((10, 80), (390, 80), 2)),
            ("line", ((10, 120), (390, 120), 2)),
        ])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "top", view.metrics)

    def test_left_perspective_from_a_left_heavy_silhouette(self) -> None:
        # The left half carries clearly more mask area, which by the documented
        # convention means the left side is turned towards the camera.
        mask = self._mask([
            ("rect", ((30, 60), (220, 140))),
            ("rect", ((240, 60), (285, 140))),
        ])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "left_front_perspective", view.metrics)
        self.assertLess(view.metrics["lateral_skew"], 0.0)

    def test_right_perspective_from_a_right_heavy_silhouette(self) -> None:
        mask = self._mask([
            ("rect", ((115, 60), (160, 140))),
            ("rect", ((180, 60), (370, 140))),
        ])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "right_front_perspective", view.metrics)
        self.assertGreater(view.metrics["lateral_skew"], 0.0)

    def test_near_symmetric_silhouette_is_not_called_a_perspective(self) -> None:
        """A tiny area difference is not enough to claim a side."""
        mask = self._mask([
            ("rect", ((30, 50), (196, 150))),
            ("rect", ((204, 50), (370, 150))),
            ("line", ((196, 100), (204, 100), 8)),
        ])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertEqual(view.label, "front", view.metrics)
        self.assertLess(abs(view.metrics["lateral_skew"]), ViewClassifier.PERSPECTIVE_SKEW)

    def test_rear_is_declared_unavailable(self) -> None:
        """The one-class mask carries no front/rear evidence, so rear is refused."""
        self.assertFalse(ViewClassifier.REAR_AVAILABLE)
        self.assertNotIn("rear", ORBIT_LABELS)

        mask = self._mask(self.FRONT_SHAPES)
        image = np.zeros((200, 400, 3), np.uint8)
        view = self.classifier.classify_orbit_view(image, mask)
        self.assertEqual(view.label, "front")
        # Still reported explicitly, so a caller is never left guessing whether
        # "rear" was absent because it was unavailable or because it was missed.
        self.assertEqual(view.metrics["rear_available"], 0.0)
        self.assertNotEqual(view.label, "rear")

    def test_every_label_is_in_the_documented_vocabulary(self) -> None:
        mask = self._mask([("rect", ((10, 90), (390, 110)))])
        view = self.classifier.classify_orbit_view(np.zeros((200, 400, 3), np.uint8), mask)
        self.assertIn(view.label, ORBIT_LABELS)

    def test_empty_mask_is_handled(self) -> None:
        view = self.classifier.classify_orbit_view(
            np.zeros((50, 50, 3), np.uint8), np.zeros((50, 50), np.uint8)
        )
        self.assertEqual(view.confidence, 0.0)

    def test_legacy_still_image_labels_are_unchanged(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        front = np.zeros((100, 100), np.uint8)
        front[40:60, 20:80] = 255
        self.assertEqual(self.classifier.classify_view(image, front), "front")
        side = np.zeros((100, 100), np.uint8)
        side[45:55, 10:90] = 255
        self.assertEqual(self.classifier.classify_view(image, side), "side")


# ── segmentation model selection ────────────────────────────────────────────
class TestSegmentationModel(unittest.TestCase):
    @unittest.skipUnless(HAS_WEIGHTS, "no YOLO weights available")
    def test_segmenter_loads_a_single_class_model(self) -> None:
        segmenter = GlassesSegmenter()
        self.assertEqual(segmenter.class_count, 1, segmenter.model_type)
        self.assertEqual(segmenter.model_type, f"yolo:{_find_model().name}")

    @unittest.skipUnless(HAS_WEIGHTS, "no YOLO weights available")
    def test_segmenter_finds_eyewear_in_a_real_photo(self) -> None:
        segmenter = GlassesSegmenter()
        image = cv2.imread(str(Path("test_images") / "test_000_metal.jpg"))
        self.assertIsNotNone(image, "missing test image fixture")
        masks = segmenter.segment(image)
        self.assertGreater(masks["detections"], 0)
        self.assertFalse(masks.get("fallback"))
        self.assertGreater(int((masks["front"] > 0).sum()), 1000)


@unittest.skipUnless(HAS_WEIGHTS, "no YOLO weights available")
class TestAutomaticMeasurementsAreFeasible(unittest.TestCase):
    """Regression: automatic extraction must satisfy the deformation constraint.

    The extractor used to give the two lens widths and the bridge the *entire*
    frame width, leaving no room for the rim thickness that
    ``validate_combination`` requires. Because the image endpoints always pass
    manual measurements, nothing exercised the automatic path until orbit video
    fed it into the deformer, where every run was rejected as infeasible.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from backend.classifier.shape_classifier import ShapeClassifier
        from backend.measurement.extractor import MeasurementExtractor

        cls.segmenter = GlassesSegmenter()
        cls.measurer = MeasurementExtractor()
        cls.classifier = ShapeClassifier()

    def test_real_photos_produce_a_feasible_combination(self) -> None:
        from backend.template_library.compatibility import validate_combination

        images = sorted(Path("test_images").glob("*.jpg"))[:6]
        self.assertTrue(images, "missing test image fixtures")
        for path in images:
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path.name)
            mask = self.segmenter.segment(image)["front"]
            style = self.classifier.classify_style(image, mask)
            inferred, _ = self.measurer.extract_from_images(
                image, None, mask, style.shape, style.material, style.nose_pads, "#d9a7a2"
            )
            # Must not raise MeasurementCompatibilityError.
            validate_combination(inferred)
            required = 2 * inferred.lens_width + inferred.bridge_width + 2 * inferred.rim_thickness
            self.assertLessEqual(
                required, inferred.frame_width + 1e-8,
                f"{path.name}: 2*lens + bridge + 2*rim = {required} > {inferred.frame_width}",
            )


# ── end to end ──────────────────────────────────────────────────────────────
@unittest.skipUnless(HAS_TEMPLATES, "templates are not available")
@unittest.skipUnless(HAS_WEIGHTS, "no YOLO weights available")
class TestOrbitVideoEndToEnd(unittest.TestCase):
    """Video in, validated GLB out, through the real model and deformers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.work = scratch_dir("_test_orbit_e2e")
        cls.video = orbit_video(cls.work / "orbit.mp4", seconds=10.0, fps=20.0)

    def test_full_pipeline_produces_a_validated_glb(self) -> None:
        from backend.pipeline.video_pipeline import VideoDeformationPipeline

        output = self.work / "orbit_result.glb"
        previews = self.work / "previews"
        result = VideoDeformationPipeline().run_from_video(
            self.video, output, preview_dir=previews
        )

        # GLB exists and passed independent serialized validation.
        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 1000)
        self.assertTrue(Path(result["manifest_json"]).is_file())
        self.assertIn(result["acceptance"]["status"], {"PASS", "REVIEW"})

        video = result["video"]

        # Decode -> gate -> bounded pool -> selection kept 5-10 views.
        self.assertGreaterEqual(video["decode"]["decoded_frames"], 12)
        selected = video["selection"]["selected_count"]
        self.assertGreaterEqual(selected, FrameSelector.MIN_VIEWS)
        self.assertLessEqual(selected, FrameSelector.MAX_VIEWS)
        self.assertEqual(video["selection"]["target_views"], FrameSelector.PREFERRED_VIEWS)
        # The pool is bounded, so segmentation never runs on the whole clip.
        self.assertLessEqual(video["pool"]["selected"], video["pool"]["requested"])
        self.assertLess(video["pool"]["selected"], video["decode"]["decoded_frames"])

        # Views were classified and measured, and the fusion audited itself.
        self.assertGreaterEqual(sum(video["view_distribution"].values()), FrameSelector.MIN_VIEWS)
        self.assertGreaterEqual(len(video["per_view"]), FrameSelector.MIN_VIEWS)
        self.assertIn("frame_width", video["fusion"]["dimensions"])
        self.assertEqual(video["fusion"]["method"], "weighted_median_mad")
        self.assertEqual(video["measurement_source"], "fused")

        # Every view record carries the evidence the manifest promises.
        for entry in video["per_view"]:
            self.assertIn(entry["view"], ORBIT_LABELS)
            self.assertIn("quality_score", entry)
            self.assertIn("sharpness", entry)
            self.assertIn("glare_ratio", entry)
            self.assertIn("timestamp", entry)

        # Rear is refused rather than guessed at.
        self.assertFalse(video["rear_available"])
        self.assertNotIn("rear", video["view_distribution"])

        # Confidence reflects evidence and states its scale assumption.
        confidence = video["confidence"]
        self.assertEqual(confidence["selected_views"], selected)
        self.assertIn(confidence["level"], {"low", "medium", "high"})
        self.assertEqual(confidence["scale_assumption"], "reference_width")
        self.assertLessEqual(confidence["measurement_consistency"], 1.0)
        # 8 views is exactly the preferred count, so a high grade is allowed --
        # but only because the views genuinely agreed.
        if confidence["level"] == "high":
            self.assertGreaterEqual(selected, FrameSelector.PREFERRED_VIEWS)

        # The scale assumption is exposed, not implied.
        self.assertEqual(video["scale"]["mode"], "reference_width")
        self.assertEqual(video["scale"]["reference_width_mm"], 135.0)
        self.assertFalse(video["scale"]["calibrated"])

        # Timings are reported for the expensive stages.
        timings = video["timings"]
        for key in ("decode_seconds", "quality_gate_seconds", "segmentation_seconds",
                    "fusion_seconds", "total_seconds"):
            self.assertIsNotNone(timings[key], key)
        self.assertGreater(timings["total_seconds"], 0.0)

        # Several views must genuinely have contributed to a fused dimension.
        # frame_width is not the right dimension to check: the extractor anchors
        # absolute scale to a fixed 135mm reference (see PRODUCTION_READINESS.md),
        # so frame_width is identical in every view by construction. Lens
        # proportions and rim thickness do vary with the true viewing geometry.
        lens = video["fusion"]["dimensions"]["lens_width"]
        self.assertEqual(lens["source"], "fused")
        self.assertGreaterEqual(lens["contributors"], 2)
        self.assertGreaterEqual(lens["kept"], 2)

        # Fused sizes are physically plausible, not defaults.
        fused = result["measurements"]
        self.assertTrue(100.0 <= fused["frame_width"] <= 160.0, fused["frame_width"])
        self.assertTrue(120.0 <= fused["temple_length"] <= 160.0, fused["temple_length"])

        # Selected-frame previews were written for the viewer.
        self.assertTrue(previews.is_dir())
        self.assertGreaterEqual(len(video["previews"]["frames"]), FrameSelector.MIN_VIEWS)
        self.assertTrue((previews / video["previews"]["grid"]).is_file())

        # The orbit manifest is persisted separately from the export contract.
        video_manifest = Path(result["video_manifest_json"])
        self.assertTrue(video_manifest.is_file())
        manifest = json.loads(video_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection"]["selected"], selected)
        self.assertEqual(len(manifest["views"]), selected)
        self.assertEqual(manifest["template"], result["template"])
        self.assertFalse(manifest["rear_available"])
        self.assertFalse(manifest["reconstruction"])
        self.assertEqual(manifest["scale"]["reference_width_mm"], 135.0)
        self.assertEqual(manifest["output"]["glb"], str(output))

        # The report documents every stage in order.
        names = [stage["name"] for stage in result["pipeline"]]
        self.assertEqual(names[0], "Video Decode")
        self.assertIn("Frame Quality Gate", names)
        self.assertIn("Candidate Pre-selection", names)
        self.assertIn("Segmentation and Eyewear Visibility", names)
        self.assertIn("View Selection", names)
        self.assertIn("Weighted Median + MAD Fusion", names)
        self.assertIn("Template Deformation", names)
        self.assertEqual(names[-1], "GLB Export")
        self.assertTrue(all(stage["status"] == "done" for stage in result["pipeline"]))

    def test_reference_width_calibrates_the_scale_assumption(self) -> None:
        from backend.pipeline.video_pipeline import VideoDeformationPipeline

        output = self.work / "calibrated_result.glb"
        result = VideoDeformationPipeline().run_from_video(
            self.video, output, reference_width_mm=142.0
        )
        scale = result["video"]["scale"]
        self.assertEqual(scale["mode"], "reference_width")
        self.assertEqual(scale["reference_width_mm"], 142.0)
        self.assertTrue(scale["calibrated"])
        # The fused frame width is re-expressed against the supplied reference.
        self.assertAlmostEqual(result["measurements"]["frame_width"], 142.0, delta=0.5)

    def test_reference_width_outside_plausible_range_is_rejected(self) -> None:
        from backend.pipeline.video_pipeline import VideoDeformationPipeline

        with self.assertRaises(ValueError) as raised:
            VideoDeformationPipeline().run_from_video(
                self.video, self.work / "bad_scale.glb", reference_width_mm=5.0
            )
        self.assertIn("reference_width_mm", str(raised.exception))

    def test_manual_measurements_override_the_fused_estimate(self) -> None:
        from backend.pipeline.video_pipeline import VideoDeformationPipeline

        output = self.work / "manual_result.glb"
        manual = measurements(frame_width=132.0, lens_width=50.0, lens_height=44.0,
                              bridge_width=17.0, temple_length=138.0)
        result = VideoDeformationPipeline().run_from_video(
            self.video, output, manual_measurements=manual
        )
        self.assertEqual(result["video"]["measurement_source"], "manual")
        self.assertAlmostEqual(result["measurements"]["frame_width"], 132.0, delta=0.01)
        # The fusion audit is still present so the estimate stays inspectable.
        self.assertIn("frame_width", result["video"]["fusion"]["dimensions"])

    def test_degraded_clip_still_reports_which_frames_were_dropped(self) -> None:
        from backend.video.selection import FrameSelector

        images = orbit_frames(60)
        degraded = []
        for index, image in enumerate(images):
            if index % 5 == 0:
                degraded.append(blur_frame(image, 3.5))
            elif index % 7 == 0:
                degraded.append(glare_frame(image, 0.12))
            else:
                degraded.append(image)
        video = scratch_dir("_test_orbit_degraded") / "degraded.mp4"
        from tests.video_fixture import write_video

        # 6 seconds at 10 fps: long enough to sample well over 18 frames.
        write_video(video, degraded, fps=10.0)

        from backend.pipeline.video_pipeline import VideoDeformationPipeline

        result = VideoDeformationPipeline().run_from_video(
            video, scratch_dir("_test_orbit_degraded") / "degraded.glb"
        )
        gate_stage = next(
            stage for stage in result["pipeline"] if stage["name"] == "Frame Quality Gate"
        )
        self.assertGreater(gate_stage["detail"]["rejected"], 0)
        self.assertTrue(gate_stage["detail"]["rejection_reasons"])
        selected = result["video"]["selection"]["selected_count"]
        self.assertGreaterEqual(selected, FrameSelector.MIN_VIEWS)
        self.assertLessEqual(selected, FrameSelector.MAX_VIEWS)


# ── production contract: counts, duplicates, determinism, visibility ────────
class TestProductionSelectionContract(unittest.TestCase):
    """The 5-10 view contract, duplicate handling and determinism."""

    def setUp(self) -> None:
        self.gate = FrameQualityGate()
        self.selector = FrameSelector()

    def _select(self, count: int, **kwargs):
        frames = as_frames(orbit_frames(count))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        return self.selector.select(frames, qualities, **kwargs)

    def test_never_selects_more_than_the_maximum(self) -> None:
        result = self._select(90)
        self.assertLessEqual(len(result.selected), FrameSelector.MAX_VIEWS)

    def test_targets_the_preferred_count(self) -> None:
        result = self._select(90)
        self.assertEqual(len(result.selected), FrameSelector.PREFERRED_VIEWS)

    def test_never_selects_more_views_than_are_available(self) -> None:
        """With only 6 usable views the pipeline processes 6, not an error."""
        result = self._select(6)
        self.assertEqual(len(result.selected), 6)
        self.assertGreaterEqual(len(result.selected), FrameSelector.MIN_VIEWS)

    def test_fewer_than_the_minimum_is_a_clear_error(self) -> None:
        # Four visually distinct frames, which is below the 5-view floor.
        frames = as_frames(orbit_frames(4))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        with self.assertRaises(ValueError) as raised:
            self.selector.select(frames, qualities)
        message = str(raised.exception)
        self.assertIn(str(FrameSelector.MIN_VIEWS), message)
        self.assertIn("required", message)

    def test_selection_is_deterministic_for_the_same_input(self) -> None:
        frames = as_frames(orbit_frames(60))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        first = self.selector.select(frames, qualities)
        second = self.selector.select(frames, qualities)
        self.assertEqual(
            [item.frame.index for item in first.selected],
            [item.frame.index for item in second.selected],
        )

    def test_duplicate_run_keeps_the_best_representative(self) -> None:
        """A run of near-identical frames contributes one view, the sharpest."""
        base = orbit_frames(9)
        # Frames 1-4 are one pose; make index 3 the sharpest of that run and the
        # others progressively softer, which is what a real pause looks like.
        frames_images = list(base)
        frames_images[1] = blur_frame(base[0], 1.6)
        frames_images[2] = blur_frame(base[0], 1.2)
        frames_images[3] = base[0]
        frames_images[4] = blur_frame(base[0], 1.0)
        frames = as_frames(frames_images)
        qualities = self.gate.evaluate_all([frame.image for frame in frames])

        deduped, dropped = FrameSelector()._deduplicate(
            [
                type("Item", (), {
                    "quality": qualities[index],
                    "frame": frames[index],
                    "descriptor": np.array([1.0, 0.0]),
                    "duplicate_of": None,
                })()
                for index in (1, 2, 3, 4)
            ]
        )
        # All four describe the same pose, so exactly one survives...
        self.assertEqual(dropped, 3)
        self.assertEqual(len(deduped), 1)
        # ...and it is the sharpest one, not simply the first in time.
        self.assertEqual(deduped[0].frame.index, 3)

    def test_a_run_of_consecutive_frames_cannot_fill_the_selection(self) -> None:
        """Ten consecutive samples must not become ten views.

        The descriptor alone cannot guarantee this -- it is noisy where the
        subject is small -- so the minimum temporal gap is what enforces it. The
        descriptor is computed clip-wide because that is how the pipeline
        computes it; its temporal-median background depends on its inputs.
        """
        video = orbit_video(scratch_dir("_test_dedupe") / "dense.mp4", seconds=10.0, fps=20.0)
        extraction = FrameExtractor(target_fps=6.0).extract(video)
        frames = extraction.frames
        qualities = self.gate.evaluate_all([frame.image for frame in frames])

        # Ten consecutive samples: one short moment of the orbit.
        window = list(zip(frames[20:30], qualities[20:30]))
        result = self.selector.select(
            [frame for frame, _ in window],
            [quality for _, quality in window],
            min_views=1,
            max_views=4,
            target_views=4,
            index_space=len(frames),
            enforce_spread=False,
        )
        selected = sorted(item.frame.index for item in result.selected)
        self.assertLess(len(selected), len(window), selected)
        # Whatever survives is separated by at least the enforced gap.
        gaps = [b - a for a, b in zip(selected, selected[1:])]
        self.assertTrue(all(gap >= 1 for gap in gaps), gaps)
        self.assertGreater(sum(gaps), len(gaps), f"selection clustered: {selected}")

    def test_selected_views_are_separated_in_time_across_a_whole_clip(self) -> None:
        frames = as_frames(orbit_frames(90))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        result = self.selector.select(frames, qualities)
        selected = sorted(item.frame.index for item in result.selected)
        gaps = [b - a for a, b in zip(selected, selected[1:])]
        # span/(2*target) = 90/16 = 5 sampled frames minimum separation.
        self.assertTrue(all(gap >= 5 for gap in gaps), f"{selected} gaps={gaps}")

    def test_visually_distinct_frames_are_all_retained(self) -> None:
        images = orbit_frames(8)
        frames = as_frames(images)
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        result = self.selector.select(frames, qualities)
        self.assertEqual(result.dropped_redundant, 0)
        self.assertEqual(len(result.selected), 8)

    def test_preselect_bounds_the_segmented_pool(self) -> None:
        frames = as_frames(orbit_frames(90))
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        pool = self.selector.preselect(frames, qualities, 20)
        self.assertLessEqual(len(pool.selected), 20)
        # The pool must not collapse the orbit onto one sector.
        self.assertGreaterEqual(pool.occupied_sectors, FrameSelector.MIN_OCCUPIED_SECTORS)

    def test_selection_from_one_sector_of_the_clip_is_rejected(self) -> None:
        """Frames drawn from one moment cannot describe a full orbit."""
        images = orbit_frames(40)
        # Only the first quarter of the clip is usable.
        degraded = [
            image if index < 10 else blur_frame(image, 3.5)
            for index, image in enumerate(images)
        ]
        frames = as_frames(degraded)
        qualities = self.gate.evaluate_all([frame.image for frame in frames])
        with self.assertRaises(ValueError) as raised:
            self.selector.select(frames, qualities)
        self.assertIn("angular sectors", str(raised.exception))


class TestEyewearVisibilityGate(unittest.TestCase):
    """The mask-dependent half of the quality gate."""

    def setUp(self) -> None:
        self.gate = FrameQualityGate()

    @staticmethod
    def _mask(shapes, size=(240, 400)) -> np.ndarray:
        mask = np.zeros(size, dtype=np.uint8)
        for (x0, y0), (x1, y1) in shapes:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        return mask

    def test_clean_centred_mask_is_accepted(self) -> None:
        mask = self._mask([((40, 80), (180, 160)), ((220, 80), (360, 160))])
        result = self.gate.evaluate_visibility(mask)
        self.assertTrue(result.accepted, result.reasons)
        self.assertGreater(result.metrics["subject_coverage"], 0.0)

    def test_empty_mask_is_rejected_as_no_eyewear(self) -> None:
        result = self.gate.evaluate_visibility(np.zeros((240, 400), dtype=np.uint8))
        self.assertFalse(result.accepted)
        self.assertTrue(any("no eyewear" in reason for reason in result.reasons), result.reasons)

    def test_mask_touching_the_frame_edge_is_rejected(self) -> None:
        """Partially outside the picture means the silhouette is truncated."""
        mask = self._mask([((0, 80), (140, 160)), ((180, 80), (320, 160))])
        result = self.gate.evaluate_visibility(mask)
        self.assertFalse(result.accepted)
        self.assertTrue(any("frame edge" in reason for reason in result.reasons), result.reasons)

    def test_fragmented_mask_is_rejected(self) -> None:
        """Many disconnected pieces stand in for heavy occlusion."""
        shapes = [((10 + 30 * i, 60), (30 + 30 * i, 90)) for i in range(14)]
        mask = self._mask(shapes, size=(400, 500))
        result = self.gate.evaluate_visibility(mask)
        self.assertFalse(result.accepted)
        self.assertTrue(
            any("fragmented" in reason or "largest" in reason for reason in result.reasons),
            result.reasons,
        )

    def test_comparable_fragments_are_rejected(self) -> None:
        """A silhouette split into comparable pieces is not a trustworthy mask."""
        mask = self._mask([
            ((20, 40), (160, 200)),
            ((180, 40), (310, 200)),
            ((330, 40), (470, 200)),
        ])
        result = self.gate.evaluate_visibility(mask)
        self.assertFalse(result.accepted)
        self.assertTrue(
            any("largest" in reason for reason in result.reasons), result.reasons
        )
        self.assertLess(result.metrics["largest_component_share"], 0.45)

    def test_negligible_specks_do_not_reject_a_solid_mask(self) -> None:
        """A few noise pixels around a solid frame must not fail the frame.

        Real one-class masks carry stray specks; rejecting on raw component count
        would throw away good views. The rule keys on the largest piece's share,
        so specks that are a rounding error are tolerated.
        """
        shapes = [((20, 40), (300, 200))]
        shapes += [((320 + 14 * i, 210), (330 + 14 * i, 220)) for i in range(10)]
        mask = self._mask(shapes, size=(400, 500))
        result = self.gate.evaluate_visibility(mask)
        self.assertTrue(result.accepted, result.reasons)
        self.assertGreater(result.metrics["largest_component_share"], 0.9)


class TestConfidenceIsEvidenceBased(unittest.TestCase):
    """Confidence must follow the evidence, not the fact that a run finished."""

    def setUp(self) -> None:
        self.fuser = RobustMeasurementFuser()

    @staticmethod
    def _views(count: int, view: str = "front", width: float = 140.0, weight: float = 0.95):
        return [
            ViewObservation(view, measurements(frame_width=width), weight=weight, label=f"v{i}")
            for i in range(count)
        ]

    def test_high_grade_needs_the_preferred_view_count(self) -> None:
        five = self.fuser.fuse(self._views(5)).confidence(preferred_views=8)
        eight = self.fuser.fuse(self._views(8)).confidence(preferred_views=8)
        self.assertEqual(five["level"], "medium")
        self.assertEqual(eight["level"], "high")
        self.assertEqual(eight["selected_views"], 8)

    def test_a_strong_score_is_still_capped_below_the_preferred_count(self) -> None:
        """Perfect agreement between 7 views is not the same claim as 8."""
        # Front, side and top together give every dimension direct evidence, so
        # the raw score clears the high band while the view count does not.
        observations = (
            self._views(5)
            + [ViewObservation("side", measurements(), weight=0.95)]
            + [ViewObservation("top", measurements(), weight=0.95)]
        )
        confidence = self.fuser.fuse(observations).confidence(preferred_views=8)
        self.assertEqual(confidence["selected_views"], 7)
        self.assertEqual(confidence["level"], "medium")
        self.assertIn("capped at medium", " ".join(confidence["notes"]))

    def test_disagreeing_views_lower_measurement_consistency(self) -> None:
        agreed = [
            ViewObservation("front", measurements(frame_width=140.0), weight=0.95)
            for _ in range(8)
        ]
        scattered = [
            ViewObservation("front", measurements(frame_width=width), weight=0.95)
            for width in (120.0, 128.0, 136.0, 140.0, 144.0, 150.0, 158.0, 168.0)
        ]
        agree_confidence = self.fuser.fuse(agreed).confidence()
        scatter_confidence = self.fuser.fuse(scattered).confidence()
        self.assertGreater(
            agree_confidence["measurement_consistency"],
            scatter_confidence["measurement_consistency"],
        )

    def test_estimated_dimension_lowers_consistency_and_is_named(self) -> None:
        # Side views can measure temple length but not frame width, so frontal
        # dimensions fall back to estimates and must be reported as such.
        side_only = [
            ViewObservation("side", measurements(frame_width=50.0), weight=0.95)
            for _ in range(8)
        ]
        result = self.fuser.fuse(side_only)
        confidence = result.confidence()
        self.assertEqual(result.dimensions["frame_width"].source, "derived")
        self.assertIn("frame_width", " ".join(confidence["notes"]))
        self.assertLess(confidence["measurement_consistency"], 1.0)

    def test_low_quality_views_lower_quality_consistency(self) -> None:
        sharp = self.fuser.fuse(self._views(8, weight=0.95)).confidence()
        poor = self.fuser.fuse(self._views(8, weight=0.25)).confidence()
        self.assertGreater(sharp["quality_consistency"], poor["quality_consistency"])
        self.assertNotEqual(sharp["level"], poor["level"])

    def test_confidence_reports_the_scale_assumption(self) -> None:
        confidence = self.fuser.fuse(self._views(8)).confidence(scale_assumption="reference_width")
        self.assertEqual(confidence["scale_assumption"], "reference_width")


# ── API contract: input validation at the HTTP boundary ─────────────────────
class TestVideoApiInputValidation(unittest.TestCase):
    """The rejection paths of POST /api/deform/video.

    None of these reach segmentation, so they are cheap: the endpoint must refuse
    the upload before spending anything on the model.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        from backend.api.main import app

        cls.client = TestClient(app)
        cls.work = scratch_dir("_test_video_api")
        cls.video = orbit_video(cls.work / "api_orbit.mp4", seconds=6.0, fps=15.0)
        cls.short = write_video(cls.work / "too_short.mp4", orbit_frames(6), fps=15.0)
        cls.fake = cls.work / "photo.mp4"
        cls.fake.write_bytes((TEST_IMAGES / "test_000_metal.jpg").read_bytes())

    def _post(self, name: str, payload: bytes, content_type: str = "video/mp4", **data):
        return self.client.post(
            "/api/deform/video",
            files={"video": (name, payload, content_type)},
            data=data,
        )

    def test_unsupported_extension_is_rejected_before_decoding(self) -> None:
        response = self._post("glasses.jpg", b"not a video at all", "image/jpeg")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video format", response.json()["detail"])

    def test_missing_video_field_is_rejected(self) -> None:
        response = self.client.post("/api/deform/video", data={"color": "#000000"})
        self.assertEqual(response.status_code, 422)

    def test_empty_upload_is_rejected(self) -> None:
        response = self._post("empty.mp4", b"")
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_corrupt_bytes_named_mp4_is_rejected(self) -> None:
        response = self._post("broken.mp4", b"this is definitely not a video")
        self.assertEqual(response.status_code, 422)
        self.assertIn("video", response.json()["detail"].lower())

    def test_image_named_mp4_is_rejected(self) -> None:
        """A JPEG decodes as a one-frame stream, so it must be caught explicitly."""
        response = self._post("photo.mp4", self.fake.read_bytes())
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertIn("does not", detail)
        self.assertIn("MP4", detail)

    def test_too_short_video_is_rejected_with_advice(self) -> None:
        response = self._post("too_short.mp4", self.short.read_bytes())
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertIn("record at least", detail)
        self.assertIn("8-12s recommended", detail)

    def test_inverted_view_bounds_are_rejected(self) -> None:
        response = self._post(
            "api_orbit.mp4", self.video.read_bytes(), min_views=9, target_views=4, max_views=5
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("min_views", response.json()["detail"])

    def test_reference_width_outside_range_is_rejected(self) -> None:
        response = self._post("api_orbit.mp4", self.video.read_bytes(), reference_width_mm=900)
        self.assertEqual(response.status_code, 400)
        self.assertIn("reference_width_mm", response.json()["detail"])

    def test_unknown_template_is_rejected(self) -> None:
        response = self._post("api_orbit.mp4", self.video.read_bytes(), template="no_such_template")
        self.assertEqual(response.status_code, 400)

    def test_preview_route_blocks_traversal_and_foreign_files(self) -> None:
        for job, name in (
            ("..", "frame_000.jpg"),
            ("1bbec67c99484d42a802d3603e68c177", "..%2f..%2frun.py"),
            ("1bbec67c99484d42a802d3603e68c177", "not-a-jpeg.txt"),
            ("bad.job", "frame_000.jpg"),
        ):
            response = self.client.get(f"/api/output/frames/{job}/{name}")
            self.assertEqual(response.status_code, 404, f"{job}/{name}")

    def test_output_route_rejects_arbitrary_filenames(self) -> None:
        for name in ("run.py", "../../run.py", "job.txt", "job.video.txt"):
            response = self.client.get(f"/api/output/{name}")
            self.assertEqual(response.status_code, 404, name)


if __name__ == "__main__":
    unittest.main()

"""Regression coverage for orbit mask integrity and uncertain orientations."""

import cv2
import json
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from backend.fusion.view_classifier import ViewClassifier
from backend.pipeline.video_pipeline import VideoDeformationPipeline
from backend.video.geometry import OrbitGeometry, OrbitViewGeometry, analyse_mask, analyse_orbit
from backend.video.measurement import DIMENSIONS, INSUFFICIENT, measure_orbit


def _two_rim_mask(left: tuple[int, int], right: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((220, 420), dtype=np.uint8)
    cv2.rectangle(mask, (left[0], 55), (left[1], 155), 255, thickness=-1)
    cv2.rectangle(mask, (right[0], 55), (right[1], 155), 255, thickness=-1)
    return mask


def test_consistent_views_are_usable_and_serialize_mask_state() -> None:
    mask = _two_rim_mask((30, 110), (190, 250))
    orbit = analyse_orbit([mask.copy(), mask.copy(), mask.copy()])

    assert all(view.usable for view in orbit.views)
    payload = orbit.to_dict()
    assert len(payload["views"]) == 3
    assert all(view["mask_inconsistent"] is False for view in payload["views"])


def test_overwide_mask_is_flagged_without_poisoning_frontal_reference(tmp_path: Path) -> None:
    normal = _two_rim_mask((30, 110), (190, 250))
    overwide = _two_rim_mask((20, 200), (270, 390))
    orbit = analyse_orbit([normal.copy(), normal.copy(), normal.copy(), overwide])

    assert orbit.reference_frame in {0, 1, 2}
    assert orbit.frontal_reference_width == analyse_mask(normal).lens_cluster_width
    assert not any(orbit.views[index].mask_inconsistent for index in range(3))
    assert orbit.views[3].mask_inconsistent
    assert not orbit.views[3].usable
    assert orbit.to_dict()["views"][3]["mask_inconsistent"] is True

    output = tmp_path / "orbit-result.glb"
    video_section = {"orbit_geometry": orbit.to_dict()}
    manifest_path = VideoDeformationPipeline._write_video_manifest(
        output, video_section, {}, SimpleNamespace(to_dict=lambda: {}), "safe-template"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["orbit_geometry"]["views"][3]["mask_inconsistent"] is True


def test_inconsistent_mask_is_uncertain_and_cannot_supply_measurements() -> None:
    normal = _two_rim_mask((30, 110), (190, 250))
    overwide = _two_rim_mask((20, 200), (270, 390))
    orbit = analyse_orbit([normal.copy(), normal.copy(), normal.copy(), overwide])
    bad_view = orbit.views[3]
    classified = ViewClassifier()._classify_one(bad_view, orbit)

    assert classified.label == "uncertain"
    assert classified.confidence == 0.0
    assert classified.metrics["mask_inconsistent"] == 1.0

    measurement = measure_orbit(
        OrbitGeometry(
            views=[bad_view],
            frontal_reference_width=orbit.frontal_reference_width,
        ),
        labels=[classified.label],
        confidences=[classified.confidence],
        quality_scores=[1.0],
        reference_width_mm=135.0,
    )
    assert all(view.evidence[name] == INSUFFICIENT for name in DIMENSIONS for view in measurement.views)
    assert not measurement.views[0].observations


def test_low_skew_asymmetric_single_mask_stays_uncertain() -> None:
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.rectangle(mask, (40, 50), (170, 145), 255, thickness=-1)
    cv2.rectangle(mask, (230, 95), (360, 155), 255, thickness=-1)
    # Small symmetric noise must not manufacture a left/right pose cue.
    mask[20, 20] = 255
    mask[180, 380] = 255

    result = ViewClassifier().classify_orbit_view(np.zeros((200, 400, 3), dtype=np.uint8), mask)

    assert result.label == "uncertain"
    assert result.confidence == 0.0


def test_empty_mask_is_explicitly_uncertain() -> None:
    result = ViewClassifier().classify_orbit_view(
        np.zeros((50, 50, 3), dtype=np.uint8), np.zeros((50, 50), dtype=np.uint8)
    )

    assert result.label == "uncertain"
    assert result.confidence == 0.0


def test_api_video_payload_preserves_mask_integrity_evidence() -> None:
    normal = _two_rim_mask((30, 110), (190, 250))
    overwide = _two_rim_mask((20, 200), (270, 390))
    orbit = analyse_orbit([normal.copy(), normal.copy(), normal.copy(), overwide])

    class DummyPipeline:
        segmenter = SimpleNamespace(model_type="test", class_count=1)
        view_classifier = SimpleNamespace(REAR_AVAILABLE=False)

        @staticmethod
        def _timings(*_args):
            return {}

    class DummyExtraction:
        sampled_fps = 6.0

        @staticmethod
        def to_dict():
            return {}

    estimate = {
        "extraction": DummyExtraction(),
        "anchor": {"index": 0, "frame": SimpleNamespace(timestamp=0.0),
                   "view": "front", "view_confidence": 0.8},
        "gate_detail": {},
        "pool_size": 8,
        "pool": SimpleNamespace(selected=[], dropped_redundant=0),
        "selection_summary": {},
        "unusable": [],
        "view_distribution": {"front": 3},
        "model_guard": "",
        "measurement_source": "fused",
        "fusion": SimpleNamespace(to_dict=lambda: {}),
        "per_view": [],
        "orbit_measurement": SimpleNamespace(provenance=lambda: {}),
        "orbit": orbit,
        "scale": {},
        "confidence": {},
        "scale_notes": [],
        "report": object(),
    }

    payload = VideoDeformationPipeline.video_payload(DummyPipeline(), estimate, 0.0)

    assert payload["orbit_geometry"]["views"][3]["mask_inconsistent"] is True

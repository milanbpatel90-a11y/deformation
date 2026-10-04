from __future__ import annotations

import inspect

import numpy as np

from backend.deformer.symmetry import SymmetrySolver
from backend.video_pipeline import VideoTo3DPipeline


def test_symmetry_nearest_distances_matches_bruteforce():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(128, 3))
    target = rng.normal(size=(97, 3))
    expected = np.min(
        np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2),
        axis=1,
    )
    actual = SymmetrySolver._nearest_distances(source, target)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_video_selection_preserves_angular_coverage():
    pipeline = VideoTo3DPipeline(pipeline=object(), coverage_bins=8, sample_count=16)
    observations = [
        {
            "frame_index": i,
            "angular_bin": i % 8,
            "quality_score": 0.9 if i < 8 else 0.8,
        }
        for i in range(16)
    ]
    selected = pipeline._select_angular_coverage(observations)
    report = pipeline._coverage_report(selected)
    assert report["occupied_bins"] == 8
    assert report["coverage_ratio"] == 1.0


def test_video_pipeline_is_not_front_width_ranked():
    source = inspect.getsource(VideoTo3DPipeline._select_angular_coverage)
    assert "width_ratio" not in source
    assert "angular_bin" in source


def test_production_api_contains_top_and_video_endpoints():
    from backend.api.main import app, deform_from_images, deform_from_video

    assert "top" in inspect.signature(deform_from_images).parameters
    assert any(route.path == "/api/deform/video" for route in app.routes)
    assert "reference_width_mm" in inspect.signature(deform_from_video).parameters

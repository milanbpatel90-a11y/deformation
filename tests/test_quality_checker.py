"""Quality regression using the actual production geometric GLB."""

from pathlib import Path

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


def test_quality_checker_high_score_on_production_glb(tmp_path):
    measurements = Measurements(
        frame_width=135.0,
        lens_width=50.0,
        lens_height=46.0,
        bridge_width=16.0,
        temple_length=135.0,
        rim_thickness=1.0,
        material=FrameMaterial.METAL,
        shape=FrameShape.GEOMETRIC,
        nose_pads=True,
    )

    pipeline = DeformationPipeline(Path("templates"))
    result = pipeline.run_from_measurements(
        measurements,
        tmp_path / "quality_identity.glb",
        "geometric_metal",
    )

    quality = result["quality"]
    assert quality["passed"] is True, quality
    assert quality["score"] >= 90.0, quality
    assert quality["warnings"] == [], quality

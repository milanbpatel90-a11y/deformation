from pathlib import Path

import trimesh

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


def test_measurements_pipeline_exports_deformed_glb_with_physical_scale(tmp_path):
    measurements = Measurements(
        frame_width=135.0,
        lens_width=50.0,
        lens_height=46.0,
        bridge_width=16.0,
        temple_length=135.0,
        rim_thickness=1.2,
        material=FrameMaterial.METAL,
        shape=FrameShape.GEOMETRIC,
        nose_pads=True,
        color="#363533",
    )
    output_path = tmp_path / "geometric_metal_smoke.glb"
    result = DeformationPipeline().run_from_measurements(
        measurements,
        output_path,
        template_name="geometric_metal",
    )

    assert Path(result["output_glb"]).is_file()
    assert Path(result["metadata_json"]).is_file()
    assert result["quality"]["passed"] is True

    scene = trimesh.load(str(output_path), force="scene")
    assert isinstance(scene, trimesh.Scene)
    assert {"Frame", "Bridge", "LeftLens", "RightLens", "LeftTemple", "RightTemple"}.issubset(
        set(scene.geometry)
    )
    bounds = scene.bounds
    width_m = float(bounds[1, 0] - bounds[0, 0])
    assert 0.08 <= width_m <= 0.30
    assert result["calibration"]["absolute_size_calibrated"] is True

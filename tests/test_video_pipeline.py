from backend.video_pipeline import VideoTo3DPipeline


def test_fuse_measurements_uses_median_for_numeric_values():
    observations = []
    for width in (136.0, 140.0, 142.0, 300.0):
        observations.append({
            "view_score": 0.8,
            "measurements": {
                "frame_width": width,
                "lens_width": 54.0,
                "lens_height": 49.0,
                "bridge_width": 18.0,
                "temple_length": 140.0,
                "rim_thickness": 1.2,
                "temple_curve_angle": 28.0,
                "material": "metal",
                "shape": "geometric",
                "nose_pads": True,
            },
        })

    fused = VideoTo3DPipeline._fuse_measurements(observations)

    assert fused["frame_width"] == 141.0
    assert fused["lens_width"] == 54.0


def test_confidence_penalizes_inconsistent_measurements():
    observations = [
        {"view_score": 0.9, "measurements": {"frame_width": 140.0}},
        {"view_score": 0.9, "measurements": {"frame_width": 141.0}},
        {"view_score": 0.9, "measurements": {"frame_width": 142.0}},
    ]

    confidence = VideoTo3DPipeline._confidence(observations, observations)

    assert confidence["level"] in {"medium", "high"}
    assert confidence["measurement_cv"] < 0.02

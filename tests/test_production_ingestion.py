import numpy as np
from backend.multiview.fuse_measurements import Measurement, fuse_measurements
from backend.video.quality_gate import frame_score, is_acceptable

def test_quality_gate_returns_bounded_score():
    frame=np.zeros((100,100,3),dtype=np.uint8)
    score=frame_score(frame)
    assert 0.0 <= score["score"] <= 1.0

def test_robust_multiview_fusion_rejects_large_outlier():
    ms=[Measurement(140,52,48,18,140,28,1,0),Measurement(141,53,49,18,142,28,1,0),Measurement(139,51,47,17,139,28,1,0),Measurement(300,90,90,80,300,28,0.2,0)]
    out=fuse_measurements(ms)
    assert 138 < out["frame_width"] < 143
    assert out["overall_confidence"] > 0.8


def test_quality_gate_rejects_uniform_blur():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 80
    assert not is_acceptable(frame)


def test_quality_gate_rejects_glare():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert not is_acceptable(frame)

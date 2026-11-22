from __future__ import annotations
import cv2
import numpy as np
import pytest
from backend.video_extraction.quality_analyzer import analyze_quality
from backend.video_extraction.duplicate_filter import perceptual_hash, hash_distance
from backend.video_extraction.measurement_preprocessor import apply_calibration, to_deformation_measurements
from backend.video_extraction.schemas import VideoLimits
from backend.video_extraction.video_processor import inspect_video
from backend.video_extraction.exceptions import VideoValidationError

def test_blurry_frame_scores_below_sharp_frame():
    rng=np.random.default_rng(12); sharp=(rng.random((360,640,3))*255).astype(np.uint8)
    blurry=cv2.GaussianBlur(sharp,(31,31),0)
    assert analyze_quality(sharp)["sharpness"] > analyze_quality(blurry)["sharpness"]

def test_duplicate_hash_detects_identical_frames():
    image=np.zeros((128,128,3),dtype=np.uint8); image[20:100,30:90]=255
    assert hash_distance(perceptual_hash(image),perceptual_hash(image.copy())) == 0

def test_pixel_measurements_never_become_mm_without_calibration():
    report={"measurements":{"frame_width":{"value":300,"unit":"px","source_view":"front","confidence":.8}}, "warnings":[]}
    calibrated=apply_calibration(report)
    assert calibrated["measurements"]["frame_width"]["unit"] == "px"
    assert calibrated["absolute_dimensions_available"] is False
    with pytest.raises(ValueError,match="Millimetre calibration"):
        to_deformation_measurements(calibrated)

def test_merchant_physical_dimensions_are_explicitly_marked():
    keys=("frame_width","lens_width","lens_height","bridge_width","temple_length","rim_thickness")
    report={"measurements":{k:{"value":100,"unit":"px","source_view":"front","confidence":.5} for k in keys},"warnings":[]}
    supplied={k+"_mm":v for k,v in zip(keys,(140,52,42,18,145,1.4))}
    calibrated=apply_calibration(report,physical_measurements=supplied)
    assert calibrated["absolute_dimensions_available"] is True
    assert to_deformation_measurements(calibrated)["frame_width"] == 140

def test_invalid_video_extension_is_rejected(tmp_path):
    path=tmp_path/"bad.avi"; path.write_bytes(b"not a video")
    with pytest.raises(VideoValidationError,match="Only MP4 and MOV"):
        inspect_video(path,"bad.avi",VideoLimits())

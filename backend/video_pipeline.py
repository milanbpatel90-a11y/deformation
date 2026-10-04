"""Video-assisted eyewear 3D generation.

The capture is treated as a controlled multi-view source, not as unrestricted
photogrammetry. Frames are sampled, segmented, scored for front-facing evidence,
and their measurements are robustly fused before the existing deformation
pipeline creates the production GLB.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.pipeline import DeformationPipeline


class VideoTo3DPipeline:
    def __init__(
        self,
        pipeline: DeformationPipeline | None = None,
        sample_count: int = 36,
        max_dimension: int = 1600,
    ):
        self.pipeline = pipeline or DeformationPipeline()
        self.sample_count = max(8, sample_count)
        self.max_dimension = max_dimension

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        reference_width_mm: float | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = total / fps if fps > 0 else 0.0
        if total < 8:
            cap.release()
            raise ValueError("Video must contain at least 8 readable frames.")

        indices = np.linspace(0, total - 1, self.sample_count, dtype=int)
        observations: list[dict[str, Any]] = []

        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                frame = self._resize(frame)
                observation = self._measure_frame(frame, int(index), fps)
                if observation is not None:
                    observations.append(observation)
        finally:
            cap.release()

        if not observations:
            raise ValueError("No usable eyewear views were detected in the video.")

        # The widest, strongest masks are the best evidence for front geometry.
        ranked = sorted(
            observations,
            key=lambda x: x["view_score"],
            reverse=True,
        )
        selected = ranked[: min(8, len(ranked))]

        measurements = self._fuse_measurements(selected)
        if reference_width_mm is not None:
            measured_width = measurements["frame_width"]
            if measured_width > 0:
                scale = float(reference_width_mm) / measured_width
                for key in (
                    "frame_width",
                    "lens_width",
                    "lens_height",
                    "bridge_width",
                    "temple_length",
                    "rim_thickness",
                ):
                    measurements[key] *= scale

        # Use the normal deformation pipeline as the final source of truth.
        from backend.models import Measurements, FrameMaterial, FrameShape

        model = Measurements(
            frame_width=float(measurements["frame_width"]),
            lens_width=float(measurements["lens_width"]),
            lens_height=float(measurements["lens_height"]),
            bridge_width=float(measurements["bridge_width"]),
            temple_length=float(measurements["temple_length"]),
            rim_thickness=float(measurements["rim_thickness"]),
            material=FrameMaterial(measurements["material"]),
            shape=FrameShape(measurements["shape"]),
            nose_pads=bool(measurements["nose_pads"]),
            temple_curve_angle=float(measurements["temple_curve_angle"]),
            color=color,
        )

        result = self.pipeline.run_from_measurements(
            model,
            output_path,
            template_override,
        )

        metadata_path = Path(output_path).with_suffix(".video.json")
        metadata = {
            "input_video": str(video_path),
            "fps": fps,
            "frame_count": total,
            "duration_seconds": duration,
            "sampled_frames": len(observations),
            "selected_frames": [x["frame_index"] for x in selected],
            "selected_timestamps_seconds": [x["timestamp_seconds"] for x in selected],
            "view_coverage_degrees": round(360.0 * len(observations) / max(total, 1), 2),
            "capture_mode": "video_assisted_template_deformation",
            "confidence": self._confidence(selected, observations),
            "measurements": model.model_dump(),
            "result": result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        result.update(
            {
                "input_video": str(video_path),
                "video_metadata": str(metadata_path),
                "video_confidence": metadata["confidence"],
                "sampled_frames": len(observations),
                "selected_frames": metadata["selected_frames"],
            }
        )
        return result

    def _measure_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
    ) -> dict[str, Any] | None:
        masks = self.pipeline.segmenter.segment(frame)
        mask = masks.get("front")
        if mask is None or int(mask.sum()) == 0:
            mask = masks.get("full")
        if mask is None or int(mask.sum()) == 0:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) < 100:
            return None

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        image_h, image_w = mask.shape[:2]

        area_ratio = float(mask.sum()) / float(image_h * image_w)
        width_ratio = width / float(image_w)
        center_offset = abs(((x0 + x1) / 2.0) - image_w / 2.0) / image_w

        # Front-facing views tend to maximize projected width while remaining
        # centered. Sharpness rejects blurred video frames.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(2000.0))
        view_score = (
            0.55 * min(1.0, width_ratio * 1.8)
            + 0.20 * min(1.0, area_ratio * 12.0)
            + 0.15 * (1.0 - min(1.0, center_offset * 4.0))
            + 0.10 * sharpness_score
        )

        shape, material, nose_pads = self.pipeline.classifier.classify(frame, mask)
        measurements, _ = self.pipeline.measurer.extract_from_images(
            frame,
            None,
            mask,
            shape,
            material,
            nose_pads,
            "#000000",
        )

        return {
            "frame_index": frame_index,
            "timestamp_seconds": frame_index / fps if fps > 0 else 0.0,
            "view_score": float(view_score),
            "width_ratio": width_ratio,
            "sharpness": sharpness,
            "measurements": measurements.model_dump(),
        }

    @staticmethod
    def _fuse_measurements(observations: list[dict[str, Any]]) -> dict[str, Any]:
        if not observations:
            raise ValueError("No observations available for fusion.")

        numeric = (
            "frame_width",
            "lens_width",
            "lens_height",
            "bridge_width",
            "temple_length",
            "rim_thickness",
            "temple_curve_angle",
        )
        fused: dict[str, Any] = {}
        for key in numeric:
            values = [
                float(o["measurements"][key])
                for o in observations
                if key in o["measurements"]
            ]
            fused[key] = float(np.median(values))

        categorical = ["material", "shape", "nose_pads"]
        for key in categorical:
            values = [o["measurements"][key] for o in observations]
            fused[key] = max(set(values), key=values.count)

        return fused

    @staticmethod
    def _confidence(
        selected: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scores = [x["view_score"] for x in selected]
        widths = [x["measurements"]["frame_width"] for x in selected]
        spread = float(np.std(widths) / max(np.mean(widths), 1e-6))
        score = float(np.mean(scores)) if scores else 0.0
        confidence = max(0.0, min(1.0, score * (1.0 - min(1.0, spread * 5.0))))
        return {
            "score": round(confidence, 4),
            "level": (
                "high" if confidence >= 0.80
                else "medium" if confidence >= 0.60
                else "low"
            ),
            "measurement_cv": round(spread, 4),
            "usable_views": len(observations),
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dimension:
            return frame
        scale = self.max_dimension / longest
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

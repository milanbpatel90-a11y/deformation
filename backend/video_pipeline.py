"""Controlled 360-degree video ingestion for production template deformation.

This module is intentionally honest about scope: it is video-assisted
multi-view measurement, not unrestricted photogrammetric reconstruction.
Frames are selected by angular coverage plus image quality, then measurements
are fused and passed to the existing calibrated deformation pipeline.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.models import FrameMaterial, FrameShape, Measurements
from backend.pipeline import DeformationPipeline


class VideoTo3DPipeline:
    def __init__(
        self,
        pipeline: DeformationPipeline | None = None,
        sample_count: int = 72,
        max_dimension: int = 1600,
        coverage_bins: int = 16,
    ) -> None:
        self.pipeline = pipeline or DeformationPipeline()
        self.sample_count = max(16, sample_count)
        self.max_dimension = max_dimension
        self.coverage_bins = max(8, coverage_bins)

    @property
    def production_mode(self) -> bool:
        return os.getenv("DEFIRM_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        reference_width_mm: float | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if self.production_mode and reference_width_mm is None:
            raise ValueError(
                "reference_width_mm is required in production mode. "
                "Uncalibrated video measurements are review-only."
            )
        if reference_width_mm is not None and not 20.0 <= reference_width_mm <= 250.0:
            raise ValueError("reference_width_mm must be between 20 and 250 mm.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = total / fps if fps > 0 else 0.0
        if total < 16:
            cap.release()
            raise ValueError("Video must contain at least 16 readable frames.")

        indices = np.linspace(0, total - 1, min(self.sample_count, total), dtype=int)
        observations: list[dict[str, Any]] = []
        try:
            for index in np.unique(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame = self._resize(frame)
                obs = self._measure_frame(frame, int(index), fps, total)
                if obs is not None:
                    observations.append(obs)
        finally:
            cap.release()

        if not observations:
            raise ValueError("No usable eyewear views were detected in the video.")

        selected = self._select_angular_coverage(observations)
        coverage = self._coverage_report(selected)

        minimum_bins = max(6, self.coverage_bins // 2)
        if coverage["occupied_bins"] < minimum_bins:
            raise ValueError(
                f"Insufficient 360-degree coverage: {coverage['occupied_bins']}/"
                f"{self.coverage_bins} angular sectors usable."
            )

        measurement_views = [
            item for item in selected
            if item["view_type"] in {"front", "perspective"}
        ]
        if not measurement_views:
            measurement_views = selected

        measurements = self._fuse_measurements(measurement_views)
        scale_source = "image_estimate"
        calibrated = False
        if reference_width_mm is not None:
            measured_width = float(measurements["frame_width"])
            if measured_width <= 0:
                raise ValueError("Video produced an invalid frame-width estimate.")
            scale = float(reference_width_mm) / measured_width
            for key in (
                "frame_width", "lens_width", "lens_height", "bridge_width",
                "temple_length", "rim_thickness"
            ):
                measurements[key] *= scale
            scale_source = "image_reference"
            calibrated = True

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
            measurement_scale_source=scale_source,
            measurement_reference_width_mm=reference_width_mm,
            measurement_scale_calibrated=calibrated,
        )

        result = self.pipeline.run_from_measurements(
            model, output_path, template_override
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
            "sampling_coverage": round(len(observations) / max(len(indices), 1), 3),
            "capture_mode": "video_assisted_template_deformation",
            "reconstruction": False,
            "angular_coverage": coverage,
            "confidence": self._confidence(selected, observations),
            "measurement_views": len(measurement_views),
            "measurements": model.model_dump(),
            "result": result,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        result.update({
            "input_video": str(video_path),
            "video_metadata": str(metadata_path),
            "video_confidence": metadata["confidence"],
            "angular_coverage": coverage,
            "sampled_frames": len(observations),
            "selected_frames": metadata["selected_frames"],
        })
        return result

    def _measure_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        total_frames: int,
    ) -> dict[str, Any] | None:
        masks = self.pipeline.segmenter.segment(frame)
        mask = masks.get("front") or masks.get("full")
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(2000.0))
        quality_score = (
            0.40 * min(1.0, area_ratio * 12.0)
            + 0.30 * (1.0 - min(1.0, center_offset * 4.0))
            + 0.30 * sharpness_score
        )

        shape, material, nose_pads = self.pipeline.classifier.classify(frame, mask)
        measurements, _ = self.pipeline.measurer.extract_from_images(
            frame, None, mask, shape, material, nose_pads, "#000000"
        )

        normalized_position = frame_index / max(total_frames - 1, 1)
        angle_deg = normalized_position * 360.0
        view_type = self._classify_view(width_ratio, height / max(width, 1))

        return {
            "frame_index": frame_index,
            "timestamp_seconds": frame_index / fps if fps > 0 else 0.0,
            "angle_deg": angle_deg,
            "angular_bin": int((angle_deg / 360.0) * self.coverage_bins) % self.coverage_bins,
            "quality_score": float(quality_score),
            "width_ratio": width_ratio,
            "sharpness": sharpness,
            "view_type": view_type,
            "measurements": measurements.model_dump(),
        }

    @staticmethod
    def _classify_view(width_ratio: float, height_ratio: float) -> str:
        aspect = 1.0 / max(height_ratio, 1e-6)
        if aspect >= 4.0:
            return "side"
        if aspect < 1.0:
            return "top"
        if aspect < 1.45:
            return "perspective"
        return "front"

    def _select_angular_coverage(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_bin: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(self.coverage_bins)
        }
        for obs in observations:
            by_bin[obs["angular_bin"]].append(obs)

        selected: list[dict[str, Any]] = []
        for bin_index, items in by_bin.items():
            if not items:
                continue
            best = max(items, key=lambda x: x["quality_score"])
            selected.append(best)

        # Keep the strongest additional observations, but never sacrifice
        # angular coverage to chase front-facing width.
        remaining = [
            x for x in observations
            if x["frame_index"] not in {s["frame_index"] for s in selected}
        ]
        remaining.sort(key=lambda x: x["quality_score"], reverse=True)
        selected.extend(remaining[: max(0, 24 - len(selected))])
        return sorted(selected, key=lambda x: x["frame_index"])

    def _coverage_report(self, selected: list[dict[str, Any]]) -> dict[str, Any]:
        bins = sorted({x["angular_bin"] for x in selected})
        return {
            "bins": self.coverage_bins,
            "occupied_bins": len(bins),
            "coverage_ratio": round(len(bins) / self.coverage_bins, 4),
            "occupied": bins,
        }

    @staticmethod
    def _fuse_measurements(observations: list[dict[str, Any]]) -> dict[str, Any]:
        if not observations:
            raise ValueError("No observations available for fusion.")
        numeric = (
            "frame_width", "lens_width", "lens_height", "bridge_width",
            "temple_length", "rim_thickness", "temple_curve_angle"
        )
        fused: dict[str, Any] = {}
        for key in numeric:
            values = [
                float(o["measurements"][key])
                for o in observations if key in o["measurements"]
            ]
            if not values:
                raise ValueError(f"No valid measurement values for {key}.")
            fused[key] = float(np.median(values))

        for key in ("material", "shape", "nose_pads"):
            values = [o["measurements"][key] for o in observations]
            fused[key] = max(set(values), key=values.count)
        return fused

    @staticmethod
    def _confidence(
        selected: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scores = [x["quality_score"] for x in selected]
        widths = [
            x["measurements"]["frame_width"]
            for x in selected
            if x["view_type"] in {"front", "perspective"}
        ]
        spread = float(np.std(widths) / max(np.mean(widths), 1e-6)) if widths else 1.0
        score = float(np.mean(scores)) if scores else 0.0
        coverage = len({x["angular_bin"] for x in selected}) / max(1, max(x["angular_bin"] for x in selected) + 1)
        confidence = max(0.0, min(1.0, score * (1.0 - min(1.0, spread * 5.0))))
        return {
            "score": round(confidence, 4),
            "level": "high" if confidence >= 0.80 else "medium" if confidence >= 0.60 else "low",
            "measurement_cv": round(spread, 4),
            "usable_views": len(observations),
            "angular_bins_selected": len({x["angular_bin"] for x in selected}),
            "coverage_note": "coverage is based on capture order; camera pose is not reconstructed",
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dimension:
            return frame
        scale = self.max_dimension / longest
        return cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

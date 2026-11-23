"""Video-to-reference extraction pipeline orchestration.

IMPLEMENTATION PLAN (written before the bulk of this package was coded,
kept here verbatim as the architecture record the task brief asked for):

    1. video_processor: validate the upload (extension, magic bytes, size,
       then open with cv2.VideoCapture to validate duration/resolution/fps),
       then stream frames sequentially (grab/retrieve, bounded candidate
       count) -- never load the whole video into memory.
    2. frame_extractor: adaptively sample at a configured target rate and
       keep only frames that differ meaningfully from the last KEPT frame
       (motion-aware, not fixed every-Nth-frame), bounded by a max
       candidate count.
    3. For each candidate frame:
       a. glasses_detector (wraps the existing, singly-loaded
          GlassesSegmenter/models/best.pt) -> bbox, combined mask,
          per-class masks/confidences, area-fraction gate.
       b. quality_analyzer -> resolution-normalized sharpness, histogram
          exposure, mask-localized glare.
       c. view_classifier -> heuristic view bucket + confidence, using
          mask aspect ratio / left-right symmetry (NOT a pose estimator).
       d. duplicate_filter -> perceptual hash, for later de-duplication.
       Frames failing detection/quality gates are marked rejected with
       reasons, not silently dropped (they still appear in the report).
    4. duplicate_filter + frame_ranker: drop near-duplicates by phash
       within each view bucket, then keep the top-N highest composite-score
       candidates per view bucket (diversity-aware: every bucket gets a
       chance, not just globally-top-scoring frames).
    5. segmentation_processor: for the selected frames only, persist
       original/overlay/cutout/per-class-mask images via the storage
       abstraction.
    6. measurement_preprocessor: reuse ShapeClassifier + MeasurementExtractor
       + MeasurementFuser (unchanged) to turn the selected images into a
       provenance-tagged measurement bundle, honoring optional manual
       calibration.
    7. Reports (extraction_report.json / quality_report.json /
       measurements.json) are written as the job's result; nothing is fed
       to the real deformation engine until a human approves via the
       dashboard/API (see api/video_extraction_routes.py).
    8. On approval, api/video_extraction_routes.py hands the approved
       frames' saved image paths to the UNMODIFIED
       `DeformationPipeline.run_from_multiple_images`, then validates the
       result with the existing `QualityChecker` before calling it done.

Everything above runs inside one `backend.video_extraction.jobs.JobRunner`
thread-pool task per job so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.video_extraction.config import VideoExtractionSettings, get_settings
from backend.video_extraction.duplicate_filter import compute_phash, filter_duplicates
from backend.video_extraction.exceptions import ProcessingError
from backend.video_extraction.frame_extractor import RawFrame, extract_candidate_frames
from backend.video_extraction.frame_ranker import select_best_per_view
from backend.video_extraction.glasses_detector import GlassesFrameDetector
from backend.video_extraction.measurement_preprocessor import MeasurementPreprocessor
from backend.video_extraction.quality_analyzer import analyze_frame
from backend.video_extraction.schemas import (
    CalibrationInput,
    ExtractionReport,
    FrameCandidate,
    SelectedFrame,
    ViewLabel,
)
from backend.video_extraction.segmentation_processor import persist_selected_frame
from backend.video_extraction.storage import VideoExtractionStorage
from backend.video_extraction.video_processor import VideoMetadata, probe_video

logger = logging.getLogger("video_extraction.pipeline")

ALL_VIEWS = tuple(ViewLabel)


@dataclass
class ExtractionResult:
    report: ExtractionReport
    selected_frames: list[SelectedFrame]
    measurements_bundle: dict


class VideoExtractionPipeline:
    """Stateless orchestrator; holds only the (expensive, singleton) detector."""

    def __init__(self, detector: GlassesFrameDetector | None = None):
        self.detector = detector or GlassesFrameDetector()
        self.measurement_preprocessor = MeasurementPreprocessor()

    def run(
        self,
        job_id: str,
        video_path: Path,
        storage: VideoExtractionStorage,
        settings: VideoExtractionSettings | None = None,
        calibration: CalibrationInput | None = None,
        progress_cb=None,
    ) -> ExtractionResult:
        settings = settings or get_settings()

        def progress(message: str) -> None:
            logger.info("job %s: %s", job_id, message)
            if progress_cb:
                progress_cb(message)

        progress("Probing video metadata")
        metadata = probe_video(video_path, settings)

        progress("Sampling candidate frames")
        raw_frames = extract_candidate_frames(video_path, metadata, settings)
        if not raw_frames:
            raise ProcessingError("No usable frames could be sampled from this video.")

        progress(f"Analyzing {len(raw_frames)} candidate frames")
        candidates, frame_lookup = self._analyze_candidates(raw_frames, settings)

        accepted = [c for c in candidates if not c.rejected]
        if not accepted:
            report = ExtractionReport(
                job_id=job_id,
                video_metadata=metadata,
                candidates_sampled=len(candidates),
                candidates_rejected=len(candidates),
                selected_views=[],
                missing_views=[v.value for v in ALL_VIEWS if v != ViewLabel.UNKNOWN],
                warnings=["No candidate frame passed detection/quality checks. Try a brighter, steadier recording."],
                measurement_warnings=[],
                calibration_status=self._bundle_default_calibration_status(calibration),
            )
            self._write_reports(storage, job_id, report, [], {})
            return ExtractionResult(report=report, selected_frames=[], measurements_bundle={})

        progress("Removing near-duplicate frames per view")
        accepted = self._deduplicate_within_views(accepted)

        progress("Ranking and selecting diverse reference frames")
        selected_by_view = select_best_per_view(accepted, settings)

        progress("Saving selected frame assets")
        selected_frames: list[SelectedFrame] = []
        frames_with_masks: list[tuple[SelectedFrame, np.ndarray, np.ndarray]] = []
        for view, items in selected_by_view.items():
            for idx, candidate in enumerate(items):
                frame, combined_mask, per_class_masks = frame_lookup[candidate.frame_index]
                selected = persist_selected_frame(
                    storage=storage,
                    job_id=job_id,
                    view=view,
                    index_in_view=idx,
                    frame=frame,
                    detection=candidate.detection,
                    combined_mask=combined_mask,
                    per_class_masks=per_class_masks,
                    frame_index=candidate.frame_index,
                    timestamp_sec=candidate.timestamp_sec,
                    view_confidence=candidate.view_confidence,
                    composite_score=candidate.composite_score,
                )
                selected_frames.append(selected)
                frames_with_masks.append((selected, frame, combined_mask))

        progress("Deriving measurements")
        bundle = self.measurement_preprocessor.build(frames_with_masks, calibration=calibration)

        present_views = {f.view.value for f in selected_frames}
        meaningful_views = [v.value for v in ALL_VIEWS if v != ViewLabel.UNKNOWN]
        missing_views = [v for v in meaningful_views if v not in present_views]

        report = ExtractionReport(
            job_id=job_id,
            video_metadata=metadata,
            candidates_sampled=len(candidates),
            candidates_rejected=len([c for c in candidates if c.rejected]),
            selected_views=sorted(present_views),
            missing_views=missing_views,
            warnings=self._collect_candidate_warnings(candidates),
            measurement_warnings=bundle.warnings,
            calibration_status=bundle.calibration_status,
        )

        progress("Writing extraction report")
        self._write_reports(storage, job_id, report, selected_frames, bundle.model_dump())

        return ExtractionResult(report=report, selected_frames=selected_frames, measurements_bundle=bundle.model_dump())

    # ------------------------------------------------------------------

    def _analyze_candidates(
        self, raw_frames: list[RawFrame], settings: VideoExtractionSettings
    ) -> tuple[list[FrameCandidate], dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]]:
        candidates: list[FrameCandidate] = []
        frame_lookup: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}

        for raw in raw_frames:
            detection, combined_mask, per_class_masks = self.detector.detect(raw.image, settings)
            quality = analyze_frame(raw.image, combined_mask if detection.detected else None, settings)

            from backend.video_extraction.view_classifier import classify_frame as classify_video_frame

            view, view_confidence = (
                classify_video_frame(raw.image, combined_mask) if detection.detected else (ViewLabel.UNKNOWN, 0.0)
            )
            phash = compute_phash(raw.image)

            candidate = FrameCandidate(
                frame_index=raw.frame_index,
                timestamp_sec=raw.timestamp_sec,
                view=view,
                view_confidence=view_confidence,
                quality=quality,
                detection=detection,
                phash=phash,
            )

            reasons: list[str] = []
            if not detection.detected:
                reasons.append("No glasses detected (or detection area outside the configured bounds).")
            if quality.is_blurry:
                reasons.append("Frame rejected: too blurry.")
            if not quality.exposure_ok:
                reasons.append("Frame rejected: exposure out of range.")
            if not quality.glare_ok:
                reasons.append("Frame rejected: excessive glare/reflection.")
            if reasons:
                candidate.rejected = True
                candidate.rejection_reasons = reasons

            candidates.append(candidate)
            frame_lookup[raw.frame_index] = (raw.image, combined_mask, per_class_masks)

        return candidates, frame_lookup

    @staticmethod
    def _deduplicate_within_views(candidates: list[FrameCandidate]) -> list[FrameCandidate]:
        by_view: dict[ViewLabel, list[FrameCandidate]] = {}
        for candidate in candidates:
            by_view.setdefault(candidate.view, []).append(candidate)

        kept: list[FrameCandidate] = []
        for view, items in by_view.items():
            scored = [(c.phash or "", c.quality.overall_quality) for c in items]
            keep_indices = set(filter_duplicates(scored))
            kept.extend(items[i] for i in keep_indices)
        return kept

    @staticmethod
    def _collect_candidate_warnings(candidates: list[FrameCandidate]) -> list[str]:
        warnings: list[str] = []
        rejected = [c for c in candidates if c.rejected]
        if rejected:
            warnings.append(f"{len(rejected)} of {len(candidates)} sampled frames were rejected (see per-frame reasons).")
        low_conf = [c for c in candidates if not c.rejected and c.view == ViewLabel.UNKNOWN]
        if low_conf:
            warnings.append(
                f"{len(low_conf)} accepted frame(s) could not be confidently classified by view "
                "(view classification is heuristic; please confirm manually)."
            )
        return warnings

    @staticmethod
    def _bundle_default_calibration_status(calibration: CalibrationInput | None):
        from backend.video_extraction.schemas import CalibrationStatus

        return CalibrationStatus.CALIBRATED if (calibration and calibration.known_frame_width_mm) else CalibrationStatus.ESTIMATED

    @staticmethod
    def _write_reports(
        storage: VideoExtractionStorage,
        job_id: str,
        report: ExtractionReport,
        selected_frames: list[SelectedFrame],
        measurements_bundle: dict,
    ) -> None:
        storage.save(job_id, "extraction_report.json", json.dumps(report.model_dump(mode="json"), indent=2).encode("utf-8"))
        storage.save(
            job_id,
            "quality_report.json",
            json.dumps(
                {
                    "selected_views": report.selected_views,
                    "missing_views": report.missing_views,
                    "warnings": report.warnings,
                    "frames": [f.model_dump(mode="json") for f in selected_frames],
                },
                indent=2,
            ).encode("utf-8"),
        )
        storage.save(job_id, "measurements.json", json.dumps(measurements_bundle, indent=2, default=str).encode("utf-8"))


def decode_image(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ProcessingError(f"Could not decode stored image: {path}")
    return image


def decode_mask(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ProcessingError(f"Could not decode stored mask: {path}")
    return mask

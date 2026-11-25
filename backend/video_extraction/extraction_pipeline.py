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
from backend.video_extraction.duplicate_filter import compute_phash, filter_duplicates, hamming_distance
from backend.video_extraction.exceptions import ProcessingError
from backend.video_extraction.frame_extractor import RawFrame, extract_candidate_frames
from backend.video_extraction.frame_ranker import select_best_per_view
from backend.video_extraction.glasses_detector import GlassesFrameDetector
from backend.video_extraction.measurement_preprocessor import MeasurementPreprocessor
from backend.video_extraction.quality_analyzer import analyze_frame
from backend.video_extraction.schemas import (
    MIN_USEFUL_IMAGE_COUNT,
    TARGET_IMAGE_COUNT,
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
_NAMED_VIEWS = tuple(v for v in ViewLabel if v != ViewLabel.UNKNOWN)
METADATA_FILES = (
    "extraction_report.json",
    "view_classification.json",
    "measurements.json",
    "quality_report.json",
    "segmentation_report.json",
)


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
        cancel_check=None,
    ) -> ExtractionResult:
        """``cancel_check``, when given, is a zero-arg callable returning True
        once this job should stop (e.g. a wall-clock deadline from
        `jobs.py`). It is polled cooperatively between stages and inside the
        per-frame analysis loop -- Python threads cannot be force-killed, so
        this is how a timeout actually interrupts long-running work instead
        of only marking the job failed while the stray thread keeps running."""
        settings = settings or get_settings()

        def progress(message: str) -> None:
            logger.info("job %s: %s", job_id, message)
            if cancel_check and cancel_check():
                raise ProcessingError("Processing timed out and was cancelled.")
            if progress_cb:
                progress_cb(message)

        progress("Probing video metadata")
        metadata = probe_video(video_path, settings)

        progress("Sampling candidate frames")
        raw_frames = extract_candidate_frames(video_path, metadata, settings)
        if not raw_frames:
            raise ProcessingError("No usable frames could be sampled from this video.")

        progress(f"Analyzing {len(raw_frames)} candidate frames")
        candidates, frame_lookup = self._analyze_candidates(raw_frames, settings, cancel_check=cancel_check)

        accepted = [c for c in candidates if not c.rejected]
        if not accepted:
            report = ExtractionReport(
                job_id=job_id,
                video_metadata=metadata,
                video_duration_seconds=metadata.duration_sec,
                selected_images=0,
                candidates_sampled=len(candidates),
                candidates_rejected=len(candidates),
                views={},
                selected_views=[],
                missing_views=[v.value for v in _NAMED_VIEWS],
                quality_passed=False,
                warnings=["No candidate frame passed detection/quality checks. Try a brighter, steadier recording."],
                measurement_warnings=[],
                calibration_status=self._bundle_default_calibration_status(calibration),
            )
            self._write_reports(storage, job_id, report, [], {}, candidates)
            return ExtractionResult(report=report, selected_frames=[], measurements_bundle={})

        progress("Persisting reviewable candidate images")
        self._persist_candidates(storage, job_id, accepted, frame_lookup)

        progress("Removing near-duplicate frames per view")
        accepted = self._deduplicate_within_views(accepted)

        progress("Ranking and selecting diverse reference frames")
        selected_by_view, additional_indices = select_best_per_view(accepted, settings)

        progress("Saving selected frame assets")
        selected_frames: list[SelectedFrame] = []
        frames_with_masks: list[tuple[SelectedFrame, np.ndarray, np.ndarray]] = []
        saved_phashes: list[str] = []
        for view, items in selected_by_view.items():
            for idx, candidate in enumerate(items):
                frame, combined_mask, per_class_masks = frame_lookup[candidate.frame_index]
                dup_score = min(
                    (hamming_distance(candidate.phash, other) for other in saved_phashes if candidate.phash and other),
                    default=0,
                )
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
                    duplicate_score=dup_score,
                    is_additional=candidate.frame_index in additional_indices,
                )
                if candidate.phash:
                    saved_phashes.append(candidate.phash)
                selected_frames.append(selected)
                frames_with_masks.append((selected, frame, combined_mask))

        progress("Deriving measurements")
        bundle = self.measurement_preprocessor.build(frames_with_masks, calibration=calibration)

        report = self._build_report(
            job_id, metadata, candidates, selected_frames, bundle, calibration
        )

        progress("Writing extraction report")
        self._write_reports(storage, job_id, report, selected_frames, bundle.model_dump(), candidates)

        return ExtractionResult(report=report, selected_frames=selected_frames, measurements_bundle=bundle.model_dump())

    # ------------------------------------------------------------------

    def _analyze_candidates(
        self, raw_frames: list[RawFrame], settings: VideoExtractionSettings, cancel_check=None
    ) -> tuple[list[FrameCandidate], dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]]:
        candidates: list[FrameCandidate] = []
        frame_lookup: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}

        for i, raw in enumerate(raw_frames):
            if cancel_check and i % 10 == 0 and cancel_check():
                raise ProcessingError("Processing timed out and was cancelled.")
            detection, combined_mask, per_class_masks = self.detector.detect(raw.image, settings)
            quality = analyze_frame(raw.image, combined_mask if detection.detected else None, settings)

            from backend.video_extraction.view_classifier import classify_frame as classify_video_frame

            view, view_confidence = (
                classify_video_frame(raw.image, combined_mask, per_class_masks)
                if detection.detected
                else (ViewLabel.UNKNOWN, 0.0)
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
    def _collect_candidate_warnings(candidates: list[FrameCandidate], selected_images: int = -1) -> list[str]:
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
        if 0 <= selected_images < MIN_USEFUL_IMAGE_COUNT:
            warnings.append(
                f"Only {selected_images} usable reference image(s) were selected, below the "
                f"{MIN_USEFUL_IMAGE_COUNT}-image target. This is reported honestly rather than padded "
                "with duplicate or low-quality frames -- try a longer, steadier recording covering more angles."
            )
        return warnings

    @staticmethod
    def _bundle_default_calibration_status(calibration: CalibrationInput | None):
        from backend.video_extraction.schemas import CalibrationStatus

        return CalibrationStatus.CALIBRATED if (calibration and calibration.known_frame_width_mm) else CalibrationStatus.ESTIMATED

    @staticmethod
    def _persist_candidates(
        storage: VideoExtractionStorage,
        job_id: str,
        accepted: list[FrameCandidate],
        frame_lookup: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    ) -> None:
        """Save every accepted (non-rejected) candidate as a full-resolution
        JPEG under ``candidates/`` so the merchant can later replace a
        selected image with a different candidate (see
        `/{job_id}/frames/{frame_index}/replace` in the API routes) without
        this pipeline needing to re-seek the source video -- seeking a
        consumer MP4/MOV by frame index is unreliable on many encodings
        (see `video_processor.iter_sampled_frames`'s docstring), so every
        candidate that might later be chosen is captured once, here, during
        the single sequential decode pass. Rejected candidates are not
        persisted -- there is no review value in a frame that already
        failed detection/quality checks, and it keeps job storage bounded.
        """
        for candidate in accepted:
            frame, _combined_mask, _per_class = frame_lookup[candidate.frame_index]
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            storage.save(job_id, f"candidates/candidate_{candidate.frame_index:04d}.jpg", buf.tobytes())

    def _build_report(
        self,
        job_id: str,
        metadata: VideoMetadata,
        candidates: list[FrameCandidate],
        selected_frames: list[SelectedFrame],
        bundle,
        calibration: CalibrationInput | None,
    ) -> ExtractionReport:
        views: dict[str, int] = {}
        for frame in selected_frames:
            views[frame.view.value] = views.get(frame.view.value, 0) + 1
        present_views = set(views.keys())
        missing_views = [v.value for v in _NAMED_VIEWS if v.value not in present_views]

        selected_images = len(selected_frames)
        quality_passed = selected_images >= MIN_USEFUL_IMAGE_COUNT and bundle.sufficient_for_deformation

        return ExtractionReport(
            job_id=job_id,
            video_metadata=metadata,
            video_duration_seconds=metadata.duration_sec,
            target_images=TARGET_IMAGE_COUNT,
            selected_images=selected_images,
            candidates_sampled=len(candidates),
            candidates_rejected=len([c for c in candidates if c.rejected]),
            views=views,
            selected_views=sorted(present_views),
            missing_views=missing_views,
            quality_passed=quality_passed,
            requires_merchant_review=True,
            warnings=self._collect_candidate_warnings(candidates, selected_images),
            measurement_warnings=bundle.warnings,
            calibration_status=bundle.calibration_status,
        )

    @staticmethod
    def _write_reports(
        storage: VideoExtractionStorage,
        job_id: str,
        report: ExtractionReport,
        selected_frames: list[SelectedFrame],
        measurements_bundle: dict,
        candidates: list[FrameCandidate] | None = None,
    ) -> None:
        """Write the five metadata/*.json reports the task brief asks for."""
        candidates = candidates or []

        def save_json(relpath: str, payload) -> None:
            storage.save(job_id, f"metadata/{relpath}", json.dumps(payload, indent=2, default=str).encode("utf-8"))

        save_json("extraction_report.json", report.model_dump(mode="json"))

        save_json(
            "view_classification.json",
            {
                "candidates": [
                    {
                        "frame_index": c.frame_index,
                        "timestamp_sec": c.timestamp_sec,
                        "view": c.view.value,
                        "view_confidence": c.view_confidence,
                        "classification_method": "mask_geometry_heuristic",
                        "detection_confidence": c.detection.confidence,
                        "segmentation_confidence": (
                            round(sum(c.detection.class_confidences.values()) / len(c.detection.class_confidences), 4)
                            if c.detection.class_confidences
                            else 0.0
                        ),
                        "rejected": c.rejected,
                        "rejection_reasons": c.rejection_reasons,
                    }
                    for c in candidates
                ],
                "selected": [
                    {
                        "frame_index": f.frame_index,
                        "view": f.view.value,
                        "view_confidence": f.view_confidence,
                        "manually_replaced": f.manually_replaced,
                        "view_corrected": f.view_corrected,
                    }
                    for f in selected_frames
                ],
            },
        )

        save_json("measurements.json", measurements_bundle)

        save_json(
            "quality_report.json",
            {
                "selected_views": report.selected_views,
                "missing_views": report.missing_views,
                "quality_passed": report.quality_passed,
                "warnings": report.warnings,
                "frames": [
                    {
                        "frame_index": f.frame_index,
                        "view": f.view.value,
                        "composite_score": f.composite_score,
                        "duplicate_score": f.duplicate_score,
                        "is_additional": f.is_additional,
                        "detection_confidence": f.detection.confidence,
                        "approval_status": f.approval_status,
                    }
                    for f in selected_frames
                ],
            },
        )

        save_json(
            "segmentation_report.json",
            {
                "frames": [
                    {
                        "frame_index": f.frame_index,
                        "view": f.view.value,
                        "model_type": f.detection.model_type,
                        "detection_confidence": f.detection.confidence,
                        "component_confidences": f.detection.class_confidences,
                        "component_visibility": f.component_visibility,
                        "missing_components": f.detection.missing_components,
                        "mask_paths": f.mask_paths,
                    }
                    for f in selected_frames
                ],
            },
        )


def rebuild_after_edit(
    job_id: str,
    storage: VideoExtractionStorage,
    metadata: VideoMetadata,
    candidates_sampled: int,
    candidates_rejected: int,
    selected_frames: list[SelectedFrame],
    calibration: CalibrationInput | None,
) -> tuple[ExtractionReport, dict]:
    """Recompute measurements and the extraction report after a merchant
    manually replaces a frame or corrects a view label (see
    `/{job_id}/frames/{frame_index}/replace` and `.../correct-view` in the
    API routes). Reads each selected frame's already-saved image/mask back
    from storage (same `decode_image`/`decode_mask` helpers the rest of this
    module uses) rather than re-running detection/extraction from scratch,
    since only the view-to-image mapping changed, not the pixels."""
    preprocessor = MeasurementPreprocessor()
    frames_with_masks: list[tuple[SelectedFrame, np.ndarray, np.ndarray]] = []
    for frame in selected_frames:
        image = decode_image(storage.open_path(job_id, frame.image_path))
        mask_rel = frame.mask_paths.get("combined") or next(iter(frame.mask_paths.values()), None)
        mask = decode_mask(storage.open_path(job_id, mask_rel)) if mask_rel else np.zeros(image.shape[:2], dtype=np.uint8)
        frames_with_masks.append((frame, image, mask))

    bundle = preprocessor.build(frames_with_masks, calibration=calibration)

    views: dict[str, int] = {}
    for frame in selected_frames:
        views[frame.view.value] = views.get(frame.view.value, 0) + 1
    present_views = set(views.keys())
    missing_views = [v.value for v in _NAMED_VIEWS if v.value not in present_views]
    selected_images = len(selected_frames)
    quality_passed = selected_images >= MIN_USEFUL_IMAGE_COUNT and bundle.sufficient_for_deformation

    warnings: list[str] = []
    if 0 <= selected_images < MIN_USEFUL_IMAGE_COUNT:
        warnings.append(
            f"Only {selected_images} usable reference image(s) are selected, below the "
            f"{MIN_USEFUL_IMAGE_COUNT}-image target."
        )

    report = ExtractionReport(
        job_id=job_id,
        video_metadata=metadata,
        video_duration_seconds=metadata.duration_sec,
        target_images=TARGET_IMAGE_COUNT,
        selected_images=selected_images,
        candidates_sampled=candidates_sampled,
        candidates_rejected=candidates_rejected,
        views=views,
        selected_views=sorted(present_views),
        missing_views=missing_views,
        quality_passed=quality_passed,
        requires_merchant_review=True,
        warnings=warnings,
        measurement_warnings=bundle.warnings,
        calibration_status=bundle.calibration_status,
    )

    VideoExtractionPipeline._write_reports(storage, job_id, report, selected_frames, bundle.model_dump(), candidates=[])
    return report, bundle.model_dump()


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

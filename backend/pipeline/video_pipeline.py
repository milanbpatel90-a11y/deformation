"""Orbit-video deformation pipeline: video in, validated GLB out.

This subclass reuses the still-image deformation machinery unchanged -- the
matcher, the deformers, the material pass and the serialized-GLB validation are
all inherited -- and replaces only the front half of the pipeline, where a video
supplies many automatically chosen views instead of one hand-picked photo:

    decode -> quality gate -> view selection -> 1-class YOLO segmentation
    -> per-view measurement -> weighted median + MAD fusion
    -> template scoring -> deformation -> materials -> GLB -> validation

Two orderings in that chain are deliberate. The quality gate runs *before*
segmentation because it is cheap and mask-free, so a 60-frame clip costs a few
OpenCV reductions rather than 60 YOLO passes. And the coverage check runs
*after* segmentation, because only the real mask can prove a frame actually
contained eyewear -- the temporal-median foreground used for selection is only
an approximation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backend.fusion.opacity_detector import OpacityDetector
from backend.fusion.robust_fuser import RobustMeasurementFuser, ViewObservation
from backend.fusion.view_classifier import ViewClassifier
from backend.materials.appearance import apply_image_appearance
from backend.materials.pbr import apply_materials
from backend.models import Measurements, PipelineReport
from backend.pipeline.deformation_pipeline import DeformationPipeline
from backend.video.frame_extractor import DEFAULT_TARGET_FPS, FrameExtractor
from backend.video.geometry import (
    OrbitGeometry,
    OrbitViewGeometry,
    analyse_mask,
    analyse_orbit,
)
from backend.video.measurement import OrbitMeasurement, measure_orbit
from backend.video.quality_gate import FrameQuality, FrameQualityGate
from backend.video.selection import FrameSelector, SelectionResult
from backend.video.validation import build_validation_report

LOGGER = logging.getLogger(__name__)

#: Long edge of the per-frame previews written for the viewer.
PREVIEW_LONG_EDGE = 320
PREVIEW_COLUMNS = 6
#: Views whose silhouette is square-on enough to supply lens and bridge sizes.
FRONTAL_LABELS = frozenset({"front", "three_quarter", "rear"})


class VideoDeformationPipeline(DeformationPipeline):
    """Run the full deformation pipeline from a 360-degree orbit video."""

    def __init__(self, templates_dir: Path | None = None):
        super().__init__(templates_dir)
        self._install_video_components()

    @classmethod
    def sharing(cls, base: DeformationPipeline) -> "VideoDeformationPipeline":
        """Build an orbit pipeline that reuses an existing pipeline's models.

        Loading the YOLO weights, the template library and the matcher a second
        time would double resident memory for no benefit, so the components are
        shared by reference instead of re-created.
        """
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        instance._install_video_components()
        return instance

    def _install_video_components(self) -> None:
        self.view_classifier = ViewClassifier()
        self.quality_gate = FrameQualityGate()
        self.frame_selector = FrameSelector()
        self.robust_fuser = RobustMeasurementFuser()
        self.opacity_detector = OpacityDetector()

    # ── public entry point ──────────────────────────────────────────────────
    def run_from_video(
        self,
        video_path: Path | str,
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        manual_measurements: Measurements | None = None,
        automatic_appearance: bool = True,
        min_views: int = FrameSelector.MIN_VIEWS,
        max_views: int = FrameSelector.MAX_VIEWS,
        target_views: int = FrameSelector.PREFERRED_VIEWS,
        target_fps: float = DEFAULT_TARGET_FPS,
        preview_dir: Path | str | None = None,
        reference_width_mm: float | None = None,
    ) -> dict:
        """Measure from a video, then deform with the existing engine.

        The measurement half lives in :meth:`estimate_from_video` so that the
        viewer can show the auto-measured sizes before committing to a
        deformation; this method only adds template selection, deformation,
        materials, export and validation.
        """
        total_started = time.perf_counter()
        estimate = self.estimate_from_video(
            video_path,
            color=color,
            manual_measurements=manual_measurements,
            automatic_appearance=automatic_appearance,
            min_views=min_views,
            max_views=max_views,
            target_views=target_views,
            target_fps=target_fps,
            reference_width_mm=reference_width_mm,
        )
        # Timings are written here for the estimate endpoint's benefit and
        # overwritten below with the complete picture once export has run.
        video_section = self.video_payload(estimate, total_started)
        return self._finish_from_measurements(
            fused_measurements=estimate["measurements"],
            anchor=estimate["anchor"],
            style=estimate["style"],
            template_override=template_override,
            output_path=output_path,
            report=estimate["report"],
            video_section=video_section,
            preview_dir=preview_dir,
            selected_views=estimate["selected_views"],
            total_started=total_started,
        )

    def video_payload(self, estimate: dict, total_started: float) -> dict:
        """Assemble the public `video` section from an estimate."""
        extraction = estimate["extraction"]
        anchor = estimate["anchor"]
        video_section = {
            "decode": extraction.to_dict(),
            "gate": estimate["gate_detail"],
            "pool": {
                "requested": estimate["pool_size"],
                "selected": estimate["pool"].selected,
                "dropped_redundant": estimate["pool"].dropped_redundant,
            },
            "selection": estimate["selection_summary"],
            "unusable_frames": estimate["unusable"],
            "view_distribution": estimate["view_distribution"],
            "model": {
                "type": self.segmenter.model_type,
                "classes": self.segmenter.class_count,
                "guard": estimate["model_guard"],
            },
            "anchor": {
                "frame_index": anchor["index"],
                "timestamp": round(anchor["frame"].timestamp, 3),
                "view": anchor["view"],
                "view_confidence": round(anchor["view_confidence"], 4),
            },
            "measurement_source": estimate["measurement_source"],
            "fusion": estimate["fusion"].to_dict(),
            "per_view": estimate["per_view"],
            "per_view_classification": [
                {
                    "frame_index": entry["frame_index"],
                    "view": entry["view"],
                    "confidence": entry["view_confidence"],
                    "yaw_deg": entry["yaw_deg"],
                    "quality_score": entry["quality_score"],
                    "evidence": entry["evidence"],
                }
                for entry in estimate["per_view"]
            ],
            "per_dimension_provenance": estimate["orbit_measurement"].provenance(),
            "orbit_geometry": estimate["orbit"].to_dict(),
            "sampled_fps": round(extraction.sampled_fps, 3),
            "scale": estimate["scale"],
            "confidence": estimate["confidence"],
            "rear_available": self.view_classifier.REAR_AVAILABLE,
            "warnings": list(estimate["scale_notes"]),
        }
        video_section["timings"] = self._timings(
            estimate["report"], round(time.perf_counter() - total_started, 3)
        )
        return video_section

    # ── front half, reusable as a pure estimate ─────────────────────────────
    def estimate_from_video(
        self,
        video_path: Path | str,
        color: str = "#d9a7a2",
        manual_measurements: Measurements | None = None,
        automatic_appearance: bool = True,
        min_views: int = FrameSelector.MIN_VIEWS,
        max_views: int = FrameSelector.MAX_VIEWS,
        target_views: int = FrameSelector.PREFERRED_VIEWS,
        target_fps: float = DEFAULT_TARGET_FPS,
        reference_width_mm: float | None = None,
        report: PipelineReport | None = None,
    ) -> dict:
        """Decode, gate, select, segment, measure and fuse -- without deforming.

        This is the whole measurement half of the video path. Exposing it
        separately lets the viewer populate the measurement fields *before* the
        user commits to a deformation, and means there is exactly one
        implementation of decode/gate/select/measure/fuse rather than a second
        copy for the estimate endpoint.
        """
        report = report if report is not None else PipelineReport()

        # ── 1. decode ───────────────────────────────────────────────────────
        stage = report.add("Video Decode")
        started = stage.start()
        extractor = FrameExtractor(target_fps=target_fps)
        extraction = extractor.extract(video_path)
        stage.finish(started, **extraction.to_dict())

        # ── 2. quality gate (cheap, mask-free, whole clip) ──────────────────
        stage = report.add("Frame Quality Gate")
        started = stage.start()
        qualities = self.quality_gate.evaluate_all([frame.image for frame in extraction.frames])
        rejection_counts = Counter(
            reason.split(" (")[0]
            for quality in qualities
            if not quality.accepted
            for reason in quality.reasons
        )
        gate_detail = {
            "sampled": len(qualities),
            "accepted": sum(1 for quality in qualities if quality.accepted),
            "rejected": sum(1 for quality in qualities if not quality.accepted),
            "rejection_reasons": dict(rejection_counts.most_common()),
        }
        stage.finish(started, **gate_detail)

        # ── 3. bounded candidate pool ───────────────────────────────────────
        # Segmentation is ~200ms a frame, so only a bounded, diverse pool is
        # segmented -- never the whole clip. The pool is twice the maximum final
        # count so the visibility gate still has frames to substitute in.
        stage = report.add("Candidate Pre-selection")
        started = stage.start()
        pool_size = max(max_views * 2, min_views)
        pool = self.frame_selector.preselect(extraction.frames, qualities, pool_size)
        stage.finish(
            started,
            pool_size=pool_size,
            pool_selected=len(pool.selected),
            dropped_redundant=pool.dropped_redundant,
            dropped_soft=pool.dropped_soft,
            occupied_sectors=pool.occupied_sectors,
        )

        # ── 4. segmentation, visibility gate and view classification ────────
        stage = report.add("Segmentation and Eyewear Visibility")
        started = stage.start()
        analysed, unusable = self._analyse_frames(pool)
        if not analysed:
            raise ValueError(
                "Segmentation found no usable eyewear in any candidate frame. "
                "Keep the glasses centred, unoccluded, and inside the picture."
            )
        visibility_reasons = Counter(
            reason.split(" (")[0] for entry in unusable for reason in entry["reasons"]
        )
        model_guard = self._model_guard_note()
        stage.finish(
            started,
            frames_segmented=len(pool.selected),
            frames_visible=len(analysed),
            frames_rejected=len(unusable),
            rejection_reasons=dict(visibility_reasons.most_common()),
            model_type=self.segmenter.model_type,
            model_classes=self.segmenter.class_count,
            model_guard=model_guard,
        )
        if len(analysed) < min_views:
            raise ValueError(
                f"Only {len(analysed)} frame(s) contained usable eyewear; at least {min_views} "
                "are required. Film a slower, steadier 360 degree orbit with the whole frame "
                "inside the picture and no occluding hands."
            )

        # ── 5. classify the orbit as a sequence ─────────────────────────────
        # Pose is only measurable relative to the clip: the frame head is widest
        # square-on and compresses by cos(yaw), so the orbit itself supplies the
        # frontal reference. Classifying frame by frame is what previously left
        # every view labelled "front".
        stage = report.add("View Classification")
        started = stage.start()
        orbit = analyse_orbit([entry["mask"] for entry in analysed])
        orbit_views = self.view_classifier.classify_sequence([entry["mask"] for entry in analysed])
        for entry, view in zip(analysed, orbit_views):
            entry["view"] = view.label
            entry["view_confidence"] = view.confidence
            entry["view_metrics"] = view.metrics
            entry["yaw_deg"] = view.metrics.get("yaw_deg", 0.0)
        classification_distribution = Counter(entry["view"] for entry in analysed)
        stage.finish(
            started,
            frames_classified=len(analysed),
            view_distribution=dict(classification_distribution),
            frontal_reference_width=orbit.frontal_reference_width,
            reference_frame=orbit.reference_frame,
            yaw_source="foreshortening" if orbit.frontal_reference_width else "unavailable",
            notes=orbit.notes,
        )

        # ── 6. geometric-diversity selection (5-10 views) ───────────────────
        stage = report.add("View Selection")
        started = stage.start()
        selection = self._select_diverse(analysed, target_views, max_views, min_views)
        if not selection:
            raise ValueError(
                "The orbit did not yield a usable set of distinct views; at least "
                f"{min_views} are required."
            )
        analysed = selection
        selection_summary = self._selection_summary(analysed, pool, gate_detail, orbit)
        view_distribution = Counter(entry["view"] for entry in analysed)
        stage.finish(
            started,
            **{key: value for key, value in selection_summary.items() if key not in {"frames", "per_view"}},
        )

        # ── 7. perspective-aware per-view measurement ───────────────────────
        stage = report.add("Per-View Measurement")
        started = stage.start()
        anchor = self._pick_anchor(analysed)
        style = self.classifier.classify_style(anchor["image"], anchor["mask"])
        orbit_measurement, observations, per_view = self._measure_orbit_views(
            analysed, orbit, reference_width_mm
        )
        # The anchor's lens contour drives both material lookup and the
        # deformation's contour hint, so it must never be left as None.
        if anchor["lens_contour"] is None:
            anchor["lens_contour"] = self.measurer.extract_lens_contour(anchor["image"], anchor["mask"])
        stage.finish(
            started,
            views_measured=len(observations),
            anchor_frame=anchor["index"],
            anchor_view=anchor["view"],
            mm_per_px=orbit_measurement.scale["mm_per_px"],
            measured_dimensions=sorted(
                {
                    item.name
                    for view in orbit_measurement.views
                    for item in view.observations
                }
            ),
        )

        # ── 8. weighted median + MAD fusion, gated by provenance ────────────
        stage = report.add("Weighted Median + MAD Fusion")
        started = stage.start()
        fusion = self.robust_fuser.fuse(observations)
        fused = fusion.measurements
        scale_report = orbit_measurement.scale
        scale_notes = orbit_measurement.notes

        if automatic_appearance:
            # Colour, material and shape come from the clearest frontal frame,
            # never from a median across the orbit. This mirrors the still-image
            # pipeline, which also lets the image override the classifier.
            fused, style, appearance_mask = apply_image_appearance(
                self, fused, anchor["image"], anchor["mask"]
            )
            anchor["lens_contour"] = self.measurer.extract_lens_contour(
                anchor["image"], appearance_mask
            )

        lens_opacity, lens_color = self.opacity_detector.detect(
            anchor["image"], anchor["lens_contour"]
        )
        fused = fused.model_copy(
            update={
                "lens_color": fused.lens_color or lens_color,
                "lens_opacity": fused.lens_opacity if fused.lens_opacity is not None else lens_opacity,
            }
        )
        stage.finish(
            started,
            dimensions={
                name: {
                    "value": entry.value,
                    "kept": entry.kept,
                    "contributors": entry.contributors,
                    "spread": round(entry.spread, 4),
                    "source": entry.source,
                }
                for name, entry in fusion.dimensions.items()
            },
            view_counts=fusion.view_counts,
        )

        measurement_source = "fused"
        if manual_measurements is not None:
            # Manual sizes win, exactly as in the still-image endpoints; the
            # fusion report is still returned so the estimate stays auditable.
            fused = manual_measurements.model_copy(deep=True)
            measurement_source = "manual"
        elif not automatic_appearance:
            fused = fused.model_copy(update={"color": color})

        # The scale was applied once, in the measurement layer, from the orbit's
        # frontal reference width; confidence reports whether it was calibrated.
        confidence = fusion.confidence(
            scale_assumption=scale_report["mode"],
            preferred_views=target_views,
            scale_calibrated=bool(scale_report["calibrated"]),
        )
        for note in scale_notes:
            if note not in confidence["notes"]:
                confidence["notes"].append(note)

        return {
            "measurements": fused,
            "measurement_source": measurement_source,
            "confidence": confidence,
            "scale": scale_report,
            "scale_notes": scale_notes,
            "selection_summary": selection_summary,
            "orbit_measurement": orbit_measurement,
            "per_view": per_view,
            "view_distribution": dict(view_distribution),
            "classification_distribution": dict(classification_distribution),
            "gate_detail": gate_detail,
            "pool_size": pool_size,
            "pool": pool,
            "selected_views": analysed,
            "unusable": unusable,
            "model_guard": model_guard,
            "extraction": extraction,
            "orbit": orbit,
            "anchor": anchor,
            "style": style,
            "fusion": fusion,
            "report": report,
        }

    # ── geometric-diversity selection ───────────────────────────────────────
    #: Preference order when filling the selection. Profiles come first because
    #: they are the only source of temple evidence and are usually the rarest
    #: usable pose in a clip that spends most of its time facing the camera.
    DIVERSITY_ORDER = (
        "side",
        "top",
        "left_front_perspective",
        "right_front_perspective",
        "front",
    )

    def _select_diverse(
        self,
        analysed: list[dict],
        target_views: int,
        max_views: int,
        min_views: int,
    ) -> list[dict]:
        """Choose 5-10 views that cover as many distinct poses as possible.

        Ranking purely by image quality would pick the eight sharpest frames of
        whatever pose dominated the clip -- which is how an orbit produced eight
        frontal views and no temple evidence. This walks the view classes in
        preference order, taking the best frame still available from each, so a
        single profile frame beats a ninth near-duplicate front frame.
        """
        if not analysed:
            return []

        by_view: dict[str, list[dict]] = {}
        for entry in analysed:
            by_view.setdefault(entry["view"], []).append(entry)
        for candidates in by_view.values():
            candidates.sort(key=lambda item: (-item["quality"].score, item["index"]))

        target = int(min(max(target_views, min_views), max_views))
        chosen: list[dict] = []
        chosen_indices: set[int] = set()

        # Round-robin over the preference order, then any remaining class.
        order = [label for label in self.DIVERSITY_ORDER if label in by_view]
        order += [label for label in sorted(by_view) if label not in self.DIVERSITY_ORDER]
        while len(chosen) < target:
            progressed = False
            for label in order:
                if len(chosen) >= target:
                    break
                candidates = by_view[label]
                while candidates and candidates[0]["index"] in chosen_indices:
                    candidates.pop(0)
                if not candidates:
                    continue
                entry = candidates.pop(0)
                chosen_indices.add(entry["index"])
                chosen.append(entry)
                progressed = True
            if not progressed:
                break

        # If an unusual clip cannot fill the target, fall back to the best of
        # whatever is left rather than returning a short selection.
        if len(chosen) < target:
            remaining = [item for item in analysed if item["index"] not in chosen_indices]
            remaining.sort(key=lambda item: (-item["quality"].score, item["index"]))
            chosen.extend(remaining[: target - len(chosen)])

        chosen.sort(key=lambda item: item["index"])
        return chosen

    def _selection_summary(
        self,
        chosen: list[dict],
        pool: SelectionResult,
        gate_detail: dict,
        orbit,
    ) -> dict:
        indices = [entry["index"] for entry in chosen]
        sectors = FrameSelector._sectors(len(pool.selected) + max(indices or [0]))
        return {
            "selected_count": len(chosen),
            "indices": indices,
            "candidates": gate_detail["accepted"],
            "pool_candidates": len(pool.selected),
            "gated_out": gate_detail["rejected"],
            "dropped_redundant": pool.dropped_redundant,
            "target_views": None,
            "occupied_sectors": len({FrameSelector._sector_for(i, 8, 8) for i in indices}),
            "total_sectors": 8,
            "span_ratio": (
                (max(indices) - min(indices)) / max(max(indices), 1) if len(indices) > 1 else 0.0
            ),
            "vorbit": orbit.to_dict(),
            "per_view": [
                {
                    "frame_index": entry["index"],
                    "timestamp": round(entry["frame"].timestamp, 3),
                    "view": entry["view"],
                    "view_confidence": round(entry["view_confidence"], 4),
                    "yaw_deg": round(entry.get("yaw_deg", 0.0), 1),
                    "quality_score": round(entry["quality"].score, 4),
                    "sharpness": round(entry["quality"].metrics.get("laplacian_variance", 0.0), 2),
                    "glare_ratio": round(entry["quality"].metrics.get("glare_ratio", 0.0), 4),
                    "coverage": round(entry.get("coverage", 0.0), 4),
                }
                for entry in chosen
            ],
        }

    # ── perspective-aware measurement ───────────────────────────────────────
    def _measure_orbit_views(
        self,
        analysed: list[dict],
        orbit,
        reference_width_mm: float | None,
    ) -> tuple[OrbitMeasurement, list[ViewObservation], list[dict]]:
        """Measure each selected view in pixels, then convert once.

        This replaces calling the still-image extractor per frame, which treated
        every view as square-on and produced lens heights that disagreed by 25 mm
        across a single orbit.
        """
        default_width = float(self.measurer.DEFAULT_FRAME_WIDTH_MM)
        if reference_width_mm is None:
            reference = default_width
            source = "extractor_default"
            calibrated = False
        else:
            reference = float(reference_width_mm)
            if not 20.0 <= reference <= 250.0:
                raise ValueError("reference_width_mm must be between 20 and 250 mm.")
            source = "caller_supplied"
            calibrated = True

        # Restrict the orbit context to the selected frames, keeping their masks.
        points = [
            OrbitViewGeometry(
                index=entry["index"],
                mask_geometry=analyse_mask(entry["mask"]),
                yaw_deg=entry.get("yaw_deg", 0.0),
                cos_yaw=float(orbit.views[0].cos_yaw) if orbit.views else 1.0,
            )
            for entry in analysed
        ]
        # Rebuild yaw per selected frame from the orbit's frontal reference.
        for point, entry in zip(points, analysed):
            if point.mask_geometry is not None and orbit.frontal_reference_width > 0:
                ratio = point.mask_geometry.lens_cluster_width / float(
                    orbit.frontal_reference_width
                )
                point.cos_yaw = max(0.0, min(1.0, ratio))
                point.yaw_deg = math.degrees(math.acos(point.cos_yaw))
                point.yaw_source = "foreshortening"
        selected_orbit = OrbitGeometry(
            views=points,
            frontal_reference_width=orbit.frontal_reference_width,
            median_cluster_height=orbit.median_cluster_height,
            fill_front=orbit.fill_front,
            fill_side=orbit.fill_side,
            reference_frame=orbit.reference_frame,
            notes=list(orbit.notes),
        )

        measurement = measure_orbit(
            selected_orbit,
            labels=[entry["view"] for entry in analysed],
            confidences=[entry["view_confidence"] for entry in analysed],
            quality_scores=[entry["quality"].score for entry in analysed],
            reference_width_mm=reference,
            default_width_mm=default_width,
            scale_source=source,
            calibrated=calibrated,
        )

        observations: list[ViewObservation] = []
        per_view: list[dict] = []
        for entry, view_measurement in zip(analysed, measurement.views):
            observations.append(
                ViewObservation(
                    view=entry["view"],
                    measurements=view_measurement.measurements,
                    weight=entry["weight"],
                    label=f"frame {entry['index']}@{entry['frame'].timestamp:.2f}s",
                    evidence=dict(view_measurement.evidence),
                )
            )
            per_view.append(
                {
                    "frame_index": entry["index"],
                    "timestamp": round(entry["frame"].timestamp, 3),
                    "view": entry["view"],
                    "view_confidence": round(entry["view_confidence"], 4),
                    "yaw_deg": round(view_measurement.yaw_deg, 1),
                    "quality_score": round(entry["quality"].score, 4),
                    "sharpness": round(entry["quality"].metrics.get("laplacian_variance", 0.0), 2),
                    "glare_ratio": round(entry["quality"].metrics.get("glare_ratio", 0.0), 4),
                    "coverage": round(entry.get("coverage", 0.0), 4),
                    "weight": round(entry["weight"], 4),
                    "evidence": dict(view_measurement.evidence),
                    "measured": {
                        name: getattr(view_measurement.measurements, name)
                        for name in (
                            "frame_width", "lens_width", "lens_height",
                            "bridge_width", "temple_length", "rim_thickness",
                        )
                    },
                }
            )
        return measurement, observations, per_view

    # ── segmentation model guard ────────────────────────────────────────────
    def _model_guard_note(self) -> str:
        """Describe whether the loaded segmentation model is the intended one."""
        if self.segmenter.class_count == 1:
            return "one-class eyewear model loaded"
        if self.segmenter.class_count == 0:
            return "no YOLO weights loaded; classical fallback in use"
        return (
            f"WARNING: {self.segmenter.class_count}-class model loaded, not the one-class "
            "eyewear model; all class masks are merged into one foreground region"
        )

    # ── analysis helpers ────────────────────────────────────────────────────
    def _analyse_frames(self, pool: SelectionResult) -> tuple[list[dict], list[dict]]:
        """Segment each pooled frame, then apply the eyewear visibility gate.

        A frame whose segmentation finds nothing, or finds a subject that is cut
        off by the frame edge or broken into fragments, is rejected *with its
        reason* rather than being allowed to contribute a fabricated measurement.
        The visibility score is folded into the frame's quality score so that
        both selection and fusion weight reflect how usable the frame really is.
        """
        analysed: list[dict] = []
        unusable: list[dict] = []
        for item in pool.selected:
            image = item.frame.image
            masks = self.segmenter.segment(image)
            mask = masks["front"]
            visibility = self.quality_gate.evaluate_visibility(mask)
            if not visibility.accepted:
                unusable.append(
                    {
                        "index": item.frame.index,
                        "timestamp": round(item.frame.timestamp, 3),
                        "segmentation_model": masks["model_type"],
                        "segmentation_fallback": bool(masks.get("fallback", False)),
                        "reasons": list(visibility.reasons),
                        "metrics": {
                            key: round(value, 4) for key, value in visibility.metrics.items()
                        },
                    }
                )
                continue

            combined = self._combine_quality(item.quality, visibility)
            coverage = visibility.metrics["subject_coverage"]
            analysed.append(
                {
                    "index": item.frame.index,
                    "frame": item.frame,
                    "image": image,
                    "mask": mask,
                    "coverage": coverage,
                    "visibility": visibility,
                    "segmentation_model": masks["model_type"],
                    "segmentation_fallback": bool(masks.get("fallback", False)),
                    # The view label is assigned later, for the whole sequence:
                    # pose is only measurable relative to the orbit.
                    "view": "unknown",
                    "view_confidence": 0.0,
                    "view_metrics": {},
                    "quality": combined,
                    "weight": self._view_weight(combined.score, coverage),
                    "lens_contour": None,
                }
            )
        return analysed, unusable

    @staticmethod
    def _combine_quality(image_quality, visibility) -> FrameQuality:
        """Fold visibility into the frame quality used for ranking and weighting."""
        metrics = dict(image_quality.metrics)
        metrics.update({f"visibility_{k}": v for k, v in visibility.metrics.items()})
        return FrameQuality(
            accepted=True,
            score=float(image_quality.score * visibility.score),
            metrics=metrics,
            reasons=list(visibility.reasons),
        )

    #: Coverage at which a frame is trusted fully. Mask area is a proxy for how
    #: many pixels landed on the rim, and therefore for how precise that view's
    #: measurement is, so a distant frame is down-weighted rather than dropped.
    FULL_TRUST_COVERAGE = 0.02

    def _view_weight(self, quality_score: float, coverage: float) -> float:
        """Combine frame quality with subject coverage into a fusion weight."""
        weight = float(max(0.0, quality_score)) * min(1.0, coverage / self.FULL_TRUST_COVERAGE)
        if coverage > self.quality_gate.MAX_SUBJECT_COVERAGE:
            # The glasses fill nearly the whole frame; part of the subject is
            # likely cropped, so trust the measurements less.
            weight *= 0.5
        return max(0.05, weight)

    @staticmethod
    def _pick_anchor(analysed: list[dict]) -> dict:
        """Choose the frame that best represents the product for appearance/lenses."""
        frontal = [item for item in analysed if item["view"] in FRONTAL_LABELS]
        pool = frontal or analysed
        return max(pool, key=lambda item: item["weight"] * (1.0 + item["coverage"]))

    # ── shared tail: template selection -> deformation -> export ────────────
    def _finish_from_measurements(
        self,
        fused_measurements: Measurements,
        anchor: dict,
        style,
        template_override: str | None,
        output_path: Path | str | None,
        report: PipelineReport,
        video_section: dict,
        preview_dir: Path | str | None = None,
        selected_views: list[dict] | None = None,
        total_started: float | None = None,
    ) -> dict:
        """Template scoring, deformation, materials, export and validation."""
        stage = report.add("Template Scoring")
        started = stage.start()
        features = self.feature_extractor.from_measurements(fused_measurements, style)
        match = self.matcher.match(features, template_override, measurements=fused_measurements)
        template_info = match.best.template
        template_name = template_info.name
        stage.finish(
            started,
            template=template_name,
            score=match.best.score,
            reason=match.best.reason,
            breakdown=match.best.breakdown,
            top_candidates=[
                {"template": candidate.template.name, "score": candidate.score, "reason": candidate.reason}
                for candidate in match.candidates[:3]
            ],
        )

        stage = report.add("Template Deformation")
        started = stage.start()
        rim_pull = self.library.rim_pull_strength(template_name)
        deformed, quality = self._deform_template(
            template_info, fused_measurements, anchor["lens_contour"]
        )
        stage.finish(
            started,
            rim_pull_strength=rim_pull,
            deformation_mode=template_info.deformation_mode,
            scale_factors=self._scale_summary(fused_measurements, template_info.dimensions),
        )

        stage = report.add("Texture Mapping")
        started = stage.start()
        deformed = apply_materials(deformed, fused_measurements)
        if fused_measurements.shape != template_info.shape:
            quality.warnings.append(
                f"Detected/requested shape: {fused_measurements.shape.value}; available template "
                f"geometry: {template_info.shape.value}. Exact silhouette requires a matching template."
            )
        stage.finish(
            started,
            material=fused_measurements.material.value,
            color=fused_measurements.color,
            lens_color=fused_measurements.lens_color,
            lens_opacity=fused_measurements.lens_opacity,
        )

        stage = report.add("Mesh Quality Optimization")
        started = stage.start()
        deformed = self._optimize_scene(deformed)
        stage.finish(started, preserved_topology=True, preserved_materials=True)

        stage = report.add("GLB Export")
        started = stage.start()
        if output_path is None:
            output_path = Path("output/development") / f"{template_name}_deformed.glb"
        out = Path(output_path)
        acceptance, manifest_path = self._export_validated(
            deformed, out, fused_measurements, template_name, template_info, quality
        )
        meta_path = out.with_suffix(".metadata.json")
        stage.finish(started, output=str(out), size_kb=round(out.stat().st_size / 1024, 1))

        if preview_dir is not None and selected_views:
            video_section["previews"] = self._write_previews(selected_views, Path(preview_dir))

        # ── validation: completing is not the same as being trustworthy ─────
        stage = report.add("Geometry Validation")
        started = stage.start()
        production_mode = os.getenv("DEFIRM_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        validation = build_validation_report(
            acceptance, quality.to_dict(), production_mode=production_mode
        )
        video_section["validation"] = validation
        stage.finish(
            started,
            overall=validation["overall"],
            self_intersections=validation["self_intersections"],
            production_mode=production_mode,
        )

        total_seconds = None
        if total_started is not None:
            total_seconds = round(time.perf_counter() - total_started, 3)
        video_section["timings"] = self._timings(report, total_seconds)

        video_manifest_path = self._write_video_manifest(
            out=out,
            video_section=video_section,
            acceptance=acceptance,
            quality=quality,
            template_name=template_name,
        )

        return {
            "quality": quality.to_dict(),
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "manifest_json": str(manifest_path),
            "video_manifest_json": str(video_manifest_path),
            "acceptance": acceptance,
            "measurements": fused_measurements.model_dump(),
            "template": template_name,
            "features": features.model_dump(mode="json"),
            "template_selection": {
                "reason": match.best.reason,
                "score": match.best.score,
                "breakdown": match.best.breakdown,
                "top_candidates": [
                    {"template": candidate.template.name, "score": candidate.score, "reason": candidate.reason}
                    for candidate in match.candidates[:3]
                ],
            },
            "scale_factors": self._scale_summary(fused_measurements, template_info.dimensions),
            "lens_contour": anchor["lens_contour"].model_dump(),
            "pipeline": report.to_dict(),
            "video": video_section,
        }

    # ── reporting ───────────────────────────────────────────────────────────
    @staticmethod
    def _timings(report: PipelineReport, total_seconds: float | None) -> dict:
        """Per-stage wall-clock timings, in the order the stages ran."""
        stages = {stage.name: round(stage.duration_ms / 1000.0, 3) for stage in report.stages}
        measured_total = round(sum(stages.values()), 3)
        return {
            "decode_seconds": stages.get("Video Decode"),
            "quality_gate_seconds": stages.get("Frame Quality Gate"),
            "preselection_seconds": stages.get("Candidate Pre-selection"),
            "segmentation_seconds": stages.get("Segmentation and Eyewear Visibility"),
            "selection_seconds": stages.get("View Selection"),
            "measurement_seconds": stages.get("Per-View Measurement"),
            "fusion_seconds": stages.get("Weighted Median + MAD Fusion"),
            "deformation_seconds": stages.get("Template Deformation"),
            "export_seconds": stages.get("GLB Export"),
            "stages_total_seconds": measured_total,
            "total_seconds": total_seconds if total_seconds is not None else measured_total,
            "by_stage": stages,
        }

    @staticmethod
    def _write_video_manifest(
        out: Path,
        video_section: dict,
        acceptance: dict,
        quality,
        template_name: str,
    ) -> Path:
        """Persist the video-specific manifest next to the GLB.

        Deliberately a separate file from the deformation pipeline's own
        ``.manifest.json``: that one is the release contract for the exported
        mesh and must not be rewritten by a capture-mode concern.
        """
        selection = video_section.get("selection", {})
        decode = video_section.get("decode", {})
        manifest = {
            "job_id": out.stem,
            "capture_mode": "video_assisted_template_deformation",
            "reconstruction": False,
            "video": {
                "duration_seconds": decode.get("video", {}).get("duration_seconds"),
                "fps": decode.get("video", {}).get("fps"),
                "decoded_frames": decode.get("decoded_frames"),
                "sampled_fps": decode.get("sampled_fps"),
            },
            "gate": video_section.get("gate", {}),
            "selection": {
                "candidates": selection.get("candidates"),
                "pool_candidates": selection.get("pool_candidates"),
                "selected": selection.get("selected_count"),
                "min_selected": selection.get("min_selected"),
                "max_selected": selection.get("max_selected"),
                "target_selected": selection.get("target_views"),
                "dropped_redundant": selection.get("dropped_redundant"),
                "occupied_sectors": selection.get("occupied_sectors"),
                "total_sectors": selection.get("total_sectors"),
                "span_ratio": selection.get("span_ratio"),
                "warnings": selection.get("warnings", []),
            },
            "views": video_section.get("per_view", []),
            "per_view_classification": video_section.get("per_view_classification", []),
            "per_dimension_provenance": video_section.get("per_dimension_provenance", {}),
            "orbit_geometry": video_section.get("orbit_geometry", {}),
            "rejected_frames": video_section.get("unusable_frames", []),
            "view_distribution": video_section.get("view_distribution", {}),
            "rear_available": video_section.get("rear_available"),
            "model": video_section.get("model", {}),
            "fusion": video_section.get("fusion", {}),
            "confidence": video_section.get("confidence", {}),
            "scale": video_section.get("scale", {}),
            "validation": video_section.get("validation", {}),
            "warnings": video_section.get("warnings", []),
            "measurement_source": video_section.get("measurement_source"),
            "template": template_name,
            "quality": quality.to_dict(),
            "acceptance": acceptance,
            "previews": video_section.get("previews", {}),
            "timings": video_section.get("timings", {}),
            "output": {
                "glb": str(out),
                "selected_frames": selection.get("selected_count"),
            },
        }
        path = out.with_suffix(".video.json")
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return path

    # ── previews ────────────────────────────────────────────────────────────
    @classmethod
    def _write_previews(cls, selected_views: list[dict], preview_dir: Path) -> dict:
        """Write the chosen frames plus a contact sheet for the viewer.

        Encoding goes through ``imencode`` rather than ``imwrite`` on purpose:
        this project has already been bitten by OpenCV's path-based file APIs on
        Windows 8.3 short paths.
        """
        preview_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        tiles: list[np.ndarray] = []

        for entry in selected_views:
            frame = entry["frame"]
            name = f"frame_{frame.index:03d}.jpg"
            thumb = cls._thumbnail(frame.image)
            if cls._write_jpeg(preview_dir / name, thumb):
                written.append(name)
            tiles.append(
                cls._labelled(thumb, f"{frame.index} {frame.timestamp:.1f}s {entry['view'][:6]}")
            )

        grid_ok = False
        if tiles:
            grid_ok = cls._write_jpeg(preview_dir / "selected_frames.jpg", cls._contact_sheet(tiles))

        return {
            "frames": written,
            "grid": "selected_frames.jpg" if grid_ok else None,
            "directory": str(preview_dir),
        }

    @staticmethod
    def _thumbnail(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= PREVIEW_LONG_EDGE:
            return image
        scale = PREVIEW_LONG_EDGE / float(longest)
        return cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _labelled(tile: np.ndarray, text: str) -> np.ndarray:
        """Draw a legible caption strip under a tile."""
        strip = np.zeros((22, tile.shape[1], 3), dtype=np.uint8)
        cv2.putText(strip, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([tile, strip])

    @staticmethod
    def _contact_sheet(tiles: list[np.ndarray], columns: int = PREVIEW_COLUMNS) -> np.ndarray:
        """Compose thumbnails into a padded grid of uniform width."""
        rows: list[np.ndarray] = []
        for start in range(0, len(tiles), columns):
            row_tiles = tiles[start : start + columns]
            width = max(tile.shape[1] for tile in row_tiles)
            rows.append(
                np.hstack(
                    [
                        cv2.copyMakeBorder(
                            tile, 0, 0, 0, width - tile.shape[1],
                            cv2.BORDER_CONSTANT, value=(18, 18, 30),
                        )
                        for tile in row_tiles
                    ]
                )
            )
        width = max(row.shape[1] for row in rows)
        return np.vstack(
            [
                cv2.copyMakeBorder(
                    row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 30)
                )
                for row in rows
            ]
        )

    @staticmethod
    def _write_jpeg(path: Path, image: np.ndarray) -> bool:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return False
        path.write_bytes(buffer.tobytes())
        return True

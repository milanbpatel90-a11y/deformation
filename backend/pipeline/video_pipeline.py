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
from backend.video.quality_gate import FrameQuality, FrameQualityGate
from backend.video.selection import FrameSelector, SelectionResult

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
        total_started = time.perf_counter()
        report = PipelineReport()

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

        # ── 5. final coverage-aware selection (5-10 views) ──────────────────
        stage = report.add("View Selection")
        started = stage.start()
        selection = self.frame_selector.select(
            [entry["frame"] for entry in analysed],
            [entry["quality"] for entry in analysed],
            min_views=min_views,
            max_views=max_views,
            target_views=target_views,
            index_space=len(extraction.frames),
        )
        selected_indices = {entry.frame.index for entry in selection.selected}
        analysed = [entry for entry in analysed if entry["frame"].index in selected_indices]
        selection_summary = selection.to_dict()
        # The final pass only sees the pool, so its own "candidates" count is the
        # pool size. Report the whole-clip picture instead: how many frames the
        # gate accepted, how many of those were segmented, and how many survived.
        selection_summary["candidates"] = gate_detail["accepted"]
        selection_summary["pool_candidates"] = len(pool.selected)
        view_distribution = Counter(entry["view"] for entry in analysed)
        stage.finish(
            started,
            **{key: value for key, value in selection_summary.items() if key != "frames"},
            view_distribution=dict(view_distribution),
        )

        # ── 6. per-view measurement ─────────────────────────────────────────
        stage = report.add("Per-View Measurement")
        started = stage.start()
        anchor = self._pick_anchor(analysed)
        style = self.classifier.classify_style(anchor["image"], anchor["mask"])
        observations, per_view = self._measure_views(analysed, style, color)
        # The anchor's lens contour drives both material lookup and the
        # deformation's contour hint, so it must never be left as None.
        if anchor["lens_contour"] is None:
            anchor["lens_contour"] = self.measurer.extract_lens_contour(anchor["image"], anchor["mask"])
        stage.finish(
            started,
            views_measured=len(observations),
            anchor_frame=anchor["index"],
            anchor_view=anchor["view"],
        )

        # ── 7. weighted median + MAD fusion ─────────────────────────────────
        stage = report.add("Weighted Median + MAD Fusion")
        started = stage.start()
        fusion = self.robust_fuser.fuse(observations)
        fused = fusion.measurements

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

        # ── 8. absolute scale ───────────────────────────────────────────────
        # The video provides image-space geometry, not millimetres. The extractor
        # anchors frame width to a fixed reference, so that assumption -- or a
        # caller-supplied reference -- is what the numbers mean, and it is
        # recorded in the manifest rather than left implicit.
        scale = self._apply_scale(fused, reference_width_mm, manual_measurements)
        if scale["applied"]:
            fused = scale["measurements"]

        confidence = fusion.confidence(
            scale_assumption=scale["mode"], preferred_views=target_views
        )

        # ── 9-13. template selection through validated export ───────────────
        return self._finish_from_measurements(
            fused_measurements=fused,
            anchor=anchor,
            style=style,
            template_override=template_override,
            output_path=output_path,
            report=report,
            video_section={
                "decode": extraction.to_dict(),
                "gate": gate_detail,
                "pool": {
                    "requested": pool_size,
                    "selected": len(pool.selected),
                    "dropped_redundant": pool.dropped_redundant,
                },
                "selection": selection_summary,
                "unusable_frames": unusable,
                "view_distribution": dict(view_distribution),
                "model": {
                    "type": self.segmenter.model_type,
                    "classes": self.segmenter.class_count,
                    "guard": model_guard,
                },
                "anchor": {
                    "frame_index": anchor["index"],
                    "timestamp": round(anchor["frame"].timestamp, 3),
                    "view": anchor["view"],
                    "view_confidence": round(anchor["view_confidence"], 4),
                },
                "measurement_source": measurement_source,
                "fusion": fusion.to_dict(),
                "per_view": per_view,
                "sampled_fps": round(extraction.sampled_fps, 3),
                "scale": scale["report"],
                "confidence": confidence,
                "rear_available": self.view_classifier.REAR_AVAILABLE,
            },
            preview_dir=preview_dir,
            selection=selection,
            total_started=total_started,
        )

    # ── scale ───────────────────────────────────────────────────────────────
    def _apply_scale(
        self,
        measurements: Measurements,
        reference_width_mm: float | None,
        manual_measurements: Measurements | None,
    ) -> dict:
        """Rescale fused measurements to a caller-supplied reference width.

        ``frame_width`` is anchored to :data:`MeasurementExtractor.DEFAULT_FRAME_WIDTH_MM`
        by construction, so an orbit video alone cannot establish absolute
        millimetres. Passing ``reference_width_mm`` re-expresses every dimension
        against a real known width; without it the estimate stays on the
        extractor's documented assumption. Neither case is a measurement claim.
        """
        default_width = float(self.measurer.DEFAULT_FRAME_WIDTH_MM)
        report = {
            "mode": "reference_width",
            "reference_width_mm": default_width,
            "source": "extractor_default",
            "calibrated": False,
        }
        if reference_width_mm is None or manual_measurements is not None:
            return {"applied": False, "mode": report["mode"], "measurements": measurements,
                    "report": report}

        width = float(reference_width_mm)
        if not 20.0 <= width <= 250.0:
            raise ValueError("reference_width_mm must be between 20 and 250 mm.")
        measured = float(measurements.frame_width)
        if measured <= 0:
            raise ValueError("Video produced an invalid frame-width estimate for calibration.")

        factor = width / measured
        scaled = measurements.model_copy(
            update={
                name: round(float(getattr(measurements, name)) * factor, 2)
                for name in (
                    "frame_width", "lens_width", "lens_height",
                    "bridge_width", "temple_length", "rim_thickness",
                )
            }
        )
        report.update(
            {
                "reference_width_mm": width,
                "source": "caller_supplied",
                "calibrated": True,
                "scale_factor": round(factor, 6),
            }
        )
        return {"applied": True, "mode": report["mode"], "measurements": scaled, "report": report}

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
            view = self.view_classifier.classify_orbit_view(image, mask)
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
                    "view": view.label,
                    "view_confidence": view.confidence,
                    "view_metrics": view.metrics,
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

    def _measure_views(
        self, analysed: list[dict], style, color: str
    ) -> tuple[list[ViewObservation], list[dict]]:
        """Measure each view, routing dimensions to the view that can see them.

        The weighted-median fuser already zeroes out the combinations a view
        cannot supply, so a profile's meaningless frame width is ignored there.
        What this method must get right is feeding the extractor the argument
        that makes it measure the right thing: a profile measures its temple
        through the extractor's ``side`` input, and a plan view measures rim
        thickness through its ``top`` input.
        """
        observations: list[ViewObservation] = []
        per_view: list[dict] = []

        for item in analysed:
            image, mask, view = item["image"], item["mask"], item["view"]
            if view == "side":
                measurements, contour = self.measurer.extract_from_images(
                    image, side=image, mask=mask, shape=style.shape,
                    material=style.material, nose_pads=style.nose_pads, color=color,
                )
            elif view == "top":
                measurements, contour = self.measurer.extract_from_images(
                    image, side=None, mask=mask, shape=style.shape,
                    material=style.material, nose_pads=style.nose_pads, color=color,
                    top=image,
                )
            else:
                measurements, contour = self.measurer.extract_from_images(
                    image, side=None, mask=mask, shape=style.shape,
                    material=style.material, nose_pads=style.nose_pads, color=color,
                )

            if item["view"] in FRONTAL_LABELS and item["lens_contour"] is None:
                item["lens_contour"] = contour

            item["measurements"] = measurements
            observations.append(
                ViewObservation(
                    view=view,
                    measurements=measurements,
                    weight=item["weight"],
                    label=f"frame {item['index']}@{item['frame'].timestamp:.2f}s",
                )
            )
            per_view.append(
                {
                    "frame_index": item["index"],
                    "timestamp": round(item["frame"].timestamp, 3),
                    "view": view,
                    "view_confidence": round(item["view_confidence"], 4),
                    "quality_score": round(item["quality"].score, 4),
                    "sharpness": round(item["quality"].metrics.get("laplacian_variance", 0.0), 2),
                    "glare_ratio": round(item["quality"].metrics.get("glare_ratio", 0.0), 4),
                    "weight": round(item["weight"], 4),
                    "coverage": round(item["coverage"], 4),
                    "measured": {
                        name: getattr(measurements, name)
                        for name in (
                            "frame_width", "lens_width", "lens_height",
                            "bridge_width", "temple_length", "rim_thickness",
                        )
                    },
                }
            )
        return observations, per_view

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
        selection: SelectionResult | None = None,
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

        if preview_dir is not None and selection is not None:
            video_section["previews"] = self._write_previews(selection, Path(preview_dir))

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
            "rejected_frames": video_section.get("unusable_frames", []),
            "view_distribution": video_section.get("view_distribution", {}),
            "rear_available": video_section.get("rear_available"),
            "model": video_section.get("model", {}),
            "fusion": video_section.get("fusion", {}),
            "confidence": video_section.get("confidence", {}),
            "scale": video_section.get("scale", {}),
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
    def _write_previews(cls, selection: SelectionResult, preview_dir: Path) -> dict:
        """Write the chosen frames plus a contact sheet for the viewer.

        Encoding goes through ``imencode`` rather than ``imwrite`` on purpose:
        this project has already been bitten by OpenCV's path-based file APIs on
        Windows 8.3 short paths.
        """
        preview_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        tiles: list[np.ndarray] = []

        for item in selection.selected:
            name = f"frame_{item.frame.index:03d}.jpg"
            thumb = cls._thumbnail(item.frame.image)
            if cls._write_jpeg(preview_dir / name, thumb):
                written.append(name)
            tiles.append(cls._labelled(thumb, f"{item.frame.index} {item.frame.timestamp:.1f}s"))

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

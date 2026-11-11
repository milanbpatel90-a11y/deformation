"""End-to-end template deformation pipeline."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import trimesh
from PIL import Image

from backend.classifier.shape_classifier import ShapeClassifier
from backend.deformer.engine import MeshDeformer
from backend.exporter.glb_exporter import GLBExporter
from backend.materials.pbr import apply_materials
from backend.measurement.extractor import MeasurementExtractor
from backend.models import (
    ExportMetadata,
    FrameMaterial,
    FrameShape,
    Measurements,
    PipelineReport,
    StyleClassification,
)
from backend.segmentation.segmenter import GlassesSegmenter
from backend.template_library.loader import TemplateLibrary
from backend.template_matching import FeatureExtractor, TemplateMatcher
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.deformation_context import DeformationContext
from backend.fusion.view_classifier import ViewClassifier
from backend.fusion.measurement_fuser import MeasurementFuser
from backend.fusion.opacity_detector import OpacityDetector


class DeformationPipeline:
    """
    Full pipeline:
      Images → Segment → Classify → Measure → Features → Score Template → Deform → Material → GLB
    """

    def __init__(self, templates_dir: Path | None = None):
        self.segmenter = GlassesSegmenter()
        self.classifier = ShapeClassifier()
        self.measurer = MeasurementExtractor()
        self.library = TemplateLibrary(templates_dir)
        self.feature_extractor = FeatureExtractor()
        self.matcher = TemplateMatcher(self.library)
        self.exporter = GLBExporter()
        self.descriptor_loader = DescriptorLoader(self.library.templates_dir)

    @staticmethod
    def _require_calibrated_scale(measurements: Measurements) -> None:
        app_env = os.getenv("APP_ENV", "development").strip().lower()
        allow_uncalibrated = os.getenv("ALLOW_UNCALIBRATED_GENERATION", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if app_env in {"production", "prod"} and not measurements.scale_calibrated and not allow_uncalibrated:
            raise ValueError(
                "Production GLB generation requires calibrated physical measurements. "
                "Provide user-measured or manufacturer-specified frame dimensions; "
                "image-only millimetre estimates are not a reliable scale reference."
            )

    @staticmethod
    def _require_quality(quality) -> None:
        if not quality.passed:
            details = "; ".join(quality.warnings) or "Critical geometry checks failed."
            raise ValueError(
                f"Deformation quality gate failed ({quality.score:.1f}/100): {details}"
            )

    def _style_from_measurements(self, measurements: Measurements) -> StyleClassification:
        aspect_ratio = measurements.lens_width / max(measurements.lens_height, 1e-6)
        return StyleClassification(
            shape=measurements.shape,
            material=measurements.material,
            nose_pads=measurements.nose_pads,
            frame_family=self.classifier._infer_family(measurements.shape, aspect_ratio),
            rim_type=self.classifier._infer_rim_type(measurements.shape, measurements.material),
            bridge_type=self.classifier._infer_bridge_type(
                measurements.shape,
                measurements.material,
                measurements.nose_pads,
                aspect_ratio,
            ),
            lens_aspect_ratio=aspect_ratio,
            confidence=1.0,
            metrics={"source": "measurements"},
        )

    @staticmethod
    def _inject_aliases_into_scene(scene: trimesh.Scene, descriptor) -> None:
        """Create independent world-space logical copies for descriptor aliases."""
        from copy import deepcopy

        raw_aliases = descriptor.raw.get("mesh_aliases", {})
        if not isinstance(raw_aliases, dict):
            raw_aliases = {}

        node_geometry_names: set[str] = set()
        geometry_transforms: dict[str, np.ndarray] = {}
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            node_geometry_names.add(geometry_name)
            if geometry_name in geometry_transforms and not np.allclose(
                geometry_transforms[geometry_name], transform
            ):
                raise ValueError(
                    f"GLB geometry '{geometry_name}' has multiple node transforms; "
                    "split the instances before deformation."
                )
            geometry_transforms[geometry_name] = transform

        alias_map: dict[str, str] = {}
        for logical_name, actual_name in raw_aliases.items():
            if not isinstance(logical_name, str) or not isinstance(actual_name, str):
                raise ValueError("Descriptor mesh_aliases must map string logical names to string GLB mesh names.")
            if logical_name == actual_name:
                continue
            if actual_name not in scene.geometry:
                raise ValueError(
                    f"Descriptor alias '{logical_name}' references missing GLB mesh '{actual_name}'."
                )
            if logical_name in node_geometry_names:
                raise ValueError(
                    f"Logical mesh '{logical_name}' already has a GLB node but is aliased to "
                    f"'{actual_name}'. Resolve the naming conflict in the descriptor."
                )
            alias_map[logical_name] = actual_name

        for group_parts in descriptor.vertex_groups.values():
            for logical_name in group_parts:
                if logical_name not in scene.geometry and logical_name not in alias_map:
                    raise ValueError(
                        f"Descriptor requires logical mesh '{logical_name}', but the GLB has no "
                        "matching mesh and no explicit mesh_aliases mapping."
                    )

        for logical_name, actual_name in alias_map.items():
            copy = deepcopy(scene.geometry[actual_name])
            transform = geometry_transforms.get(actual_name)
            if transform is not None and not np.allclose(transform, np.eye(4)):
                copy.apply_transform(transform)
            # Replace unreferenced stale aliases; never overwrite a graph node.
            scene.geometry[logical_name] = copy

    @staticmethod
    def _optimize_scene(scene: trimesh.Scene) -> trimesh.Scene:
        """Run non-destructive cleanup after deformation and material assignment."""
        for geom in scene.geometry.values():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            if hasattr(geom, "fix_normals"):
                geom.fix_normals()
            geom._cache.clear()
        return scene

    @staticmethod
    def _imread(path: Path | str) -> np.ndarray | None:
        """Read common product-image formats from Windows-safe paths."""
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is not None:
            return image

        try:
            with Image.open(BytesIO(buf.tobytes())) as pil_image:
                rgb_image = pil_image.convert("RGB")
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def run_from_images(
        self,
        front_path: Path | str,
        side_path: Path | str | None = None,
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        top_path: Path | str | None = None,
    ) -> dict:
        front = self._imread(front_path)
        if front is None:
            raise ValueError(f"Cannot read front image: {front_path}")

        side = self._imread(side_path) if side_path else None
        top = self._imread(top_path) if top_path else None

        return self.run_from_arrays(front, side, output_path, color, template_override, top=top)

    def run_from_measurements(
        self,
        measurements: Measurements,
        output_path: Path | str,
        template_name: str | None = None,
    ) -> dict:
        self._require_calibrated_scale(measurements)
        style = self._style_from_measurements(measurements)
        features = self.feature_extractor.from_measurements(measurements, style)
        match = self.matcher.match(features, template_name)
        template_info = match.best.template
        template_name = template_info.name

        scene = trimesh.load(template_info.glb_path, force="scene")
        
        descriptor = self.descriptor_loader.load(
            template_name, measurements=measurements, template_info=template_info
        )
        self._inject_aliases_into_scene(scene, descriptor)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=measurements,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx)
        self._require_quality(quality)
        deformed = deformed_ctx.template_scene
        deformed = apply_materials(deformed, measurements)

        out = Path(output_path)
        self.exporter.export(deformed, out, measurements, template_name)

        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, measurements)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
            scale_source=measurements.scale_source,
            scale_calibrated=measurements.scale_calibrated,
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "measurements": measurements.model_dump(),
            "measurement_source": measurements.scale_source,
            "scale_calibrated": measurements.scale_calibrated,
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
            "scale_factors": self._scale_summary(measurements, template_info.dimensions),
            "quality_report": quality.to_dict(),
        }

    def run_from_arrays(
        self,
        front: np.ndarray,
        side: np.ndarray | None = None,
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        top: np.ndarray | None = None,
    ) -> dict:
        report = PipelineReport()

        s1 = report.add("YOLO Segmentation")
        t = s1.start()
        masks = self.segmenter.segment(front)
        s1.finish(t, model=masks.get("model_type", "unknown"), detections=masks.get("detections", 0))

        s2 = report.add("Shape Classification")
        t = s2.start()
        style = self.classifier.classify_style(front, masks["front"])
        s2.finish(
            t,
            shape=style.shape.value,
            material=style.material.value,
            frame_family=style.frame_family.value,
            rim_type=style.rim_type.value,
            bridge_type=style.bridge_type.value,
            lens_aspect_ratio=style.lens_aspect_ratio,
            confidence=style.confidence,
            metrics=style.metrics,
        )

        s3 = report.add("Measurement Extraction")
        t = s3.start()
        measurements, lens_contour = self.measurer.extract_from_images(
            front,
            side,
            masks["front"],
            style.shape,
            style.material,
            style.nose_pads,
            color,
            top=top,
        )
        s3.finish(
            t,
            frame_width=measurements.frame_width,
            lens=f"{measurements.lens_width}×{measurements.lens_height}mm",
            bridge=measurements.bridge_width,
            temple=measurements.temple_length,
            rim_thickness=measurements.rim_thickness,
        )
        self._require_calibrated_scale(measurements)

        s4 = report.add("Feature Extraction")
        t = s4.start()
        features = self.feature_extractor.extract(masks["front"], measurements, style, top=top)
        s4.finish(t, **features.model_dump(mode="json"))

        s5 = report.add("Template Scoring")
        t = s5.start()
        match = self.matcher.match(features, template_override)
        template_info = match.best.template
        template_name = template_info.name
        s5.finish(
            t,
            template=template_name,
            score=match.best.score,
            reason=match.best.reason,
            breakdown=match.best.breakdown,
            top_candidates=[
                {"template": candidate.template.name, "score": candidate.score, "reason": candidate.reason}
                for candidate in match.candidates[:3]
            ],
        )

        s6 = report.add("Template Deformation")
        t = s6.start()
        scene = trimesh.load(template_info.glb_path, force="scene")
        descriptor = self.descriptor_loader.load(template_name, measurements=measurements, template_info=template_info)
        self._inject_aliases_into_scene(scene, descriptor)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=measurements,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx, lens_contour)
        self._require_quality(quality)
        deformed = deformed_ctx.template_scene
        s6.finish(
            t,
            rim_pull_strength=rim_pull,
            deformation_mode="template_vertex_groups",
            contour_used=False,
            scale_factors=self._scale_summary(measurements, template_info.dimensions),
        )

        s7 = report.add("Texture Mapping")
        t = s7.start()
        deformed = apply_materials(deformed, measurements)
        frame_parts = [name for name in deformed.geometry if name not in {"LeftLens", "RightLens"}]
        s7.finish(
            t,
            material=measurements.material.value,
            color=measurements.color,
            frame_parts=frame_parts,
            lens_parts=["LeftLens", "RightLens"],
        )

        s8 = report.add("Mesh Quality Optimization")
        t = s8.start()
        deformed = self._optimize_scene(deformed)
        s8.finish(t, preserved_topology=True, preserved_materials=True)

        s9 = report.add("GLB Export")
        t = s9.start()
        if output_path is None:
            output_path = Path("output") / f"{template_name}_deformed.glb"
        out = Path(output_path)

        self.exporter.export(deformed, out, measurements, template_name)
        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, measurements)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
            scale_source=measurements.scale_source,
            scale_calibrated=measurements.scale_calibrated,
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)
        s9.finish(t, output=str(out), size_kb=round(out.stat().st_size / 1024, 1))

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "measurements": measurements.model_dump(),
            "measurement_source": measurements.scale_source,
            "scale_calibrated": measurements.scale_calibrated,
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
            "scale_factors": self._scale_summary(measurements, template_info.dimensions),
            "lens_contour": lens_contour.model_dump(),
            "quality_report": quality.to_dict(),
            "pipeline": report.to_dict(),
        }

    def run_from_multiple_images(
        self,
        image_paths: list[Path | str],
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
        manual_measurements: Measurements | None = None,
        shape_override: FrameShape | None = None,
        material_override: FrameMaterial | None = None,
    ) -> dict:
        """
        Process 4–5 product views. Photos provide segmentation, contour, style
        and lens appearance; user/manufacturer dimensions provide physical scale.
        """
        if not 4 <= len(image_paths) <= 5:
            raise ValueError("Multi-view generation requires 4–5 product images.")

        report = PipelineReport()
        s1 = report.add("Multi-View Classification and Extraction")
        started = s1.start()

        view_classifier = ViewClassifier()
        fuser = MeasurementFuser()
        opacity_detector = OpacityDetector()
        extracted_views: list[tuple[str, Measurements]] = []
        processed: list[dict] = []

        for image_path in image_paths:
            image = self._imread(image_path)
            if image is None:
                continue

            masks = self.segmenter.segment(image)
            front_mask = masks["front"]
            view_type = view_classifier.classify_view(image, front_mask)
            style = self.classifier.classify_style(image, front_mask)
            estimated, lens_contour = self.measurer.extract_from_images(
                image,
                side=None,
                mask=front_mask,
                shape=style.shape,
                material=style.material,
                nose_pads=style.nose_pads,
                color=color,
            )
            extracted_views.append((view_type, estimated))
            processed.append(
                {
                    "view_type": view_type,
                    "image": image,
                    "mask": front_mask,
                    "style": style,
                    "estimated": estimated,
                    "lens_contour": lens_contour,
                }
            )

        if len(processed) < 4:
            raise ValueError(
                f"Only {len(processed)} valid images could be processed; 4–5 valid views are required."
            )

        front_record = next(
            (record for record in processed if record["view_type"] == "front"),
            processed[0],
        )
        front_image = front_record["image"]
        front_mask = front_record["mask"]
        front_style = front_record["style"].model_copy(deep=True)
        front_lens_contour = front_record["lens_contour"]

        if shape_override is not None:
            front_style.shape = shape_override
        if material_override is not None:
            front_style.material = material_override

        if shape_override is not None or material_override is not None:
            aspect_ratio = (
                manual_measurements.lens_width / max(manual_measurements.lens_height, 1e-6)
                if manual_measurements is not None
                else front_style.lens_aspect_ratio
            )
            front_style.nose_pads = (
                front_style.material in {FrameMaterial.METAL, FrameMaterial.TITANIUM}
                and front_style.shape != FrameShape.RIMLESS
            )
            front_style.frame_family = self.classifier._infer_family(front_style.shape, aspect_ratio)
            front_style.rim_type = self.classifier._infer_rim_type(front_style.shape, front_style.material)
            front_style.bridge_type = self.classifier._infer_bridge_type(
                front_style.shape,
                front_style.material,
                front_style.nose_pads,
                aspect_ratio,
            )
            front_style.lens_aspect_ratio = aspect_ratio
            front_style.metrics = dict(front_style.metrics)
            front_style.metrics["manual_style_override"] = True

        if manual_measurements is not None:
            fused_measurements = manual_measurements.model_copy(deep=True)
            fused_measurements.shape = front_style.shape
            fused_measurements.material = front_style.material
            fused_measurements.nose_pads = front_style.nose_pads
            fused_measurements.color = color
            measurement_source = fused_measurements.scale_source
        else:
            fused_measurements = fuser.fuse(extracted_views)
            measurement_source = fused_measurements.scale_source

        self._require_calibrated_scale(fused_measurements)
        lens_opacity, lens_color = opacity_detector.detect(front_image, front_lens_contour)
        fused_measurements.lens_color = lens_color
        fused_measurements.lens_opacity = lens_opacity

        s1.finish(
            started,
            views=[record["view_type"] for record in processed],
            measurement_source=measurement_source,
        )

        s2 = report.add("Template Scoring")
        started = s2.start()
        features = self.feature_extractor.extract(front_mask, fused_measurements, front_style)
        match = self.matcher.match(features, template_override)
        template_info = match.best.template
        template_name = template_info.name
        s2.finish(
            started,
            template=template_name,
            score=match.best.score,
            reason=match.best.reason,
        )

        s3 = report.add("Template Deformation")
        started = s3.start()
        scene = trimesh.load(template_info.glb_path, force="scene")
        descriptor = self.descriptor_loader.load(
            template_name,
            measurements=fused_measurements,
            template_info=template_info,
        )
        self._inject_aliases_into_scene(scene, descriptor)
        context = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=fused_measurements,
            feature_set=features,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_context, quality = deformer.deform(context, front_lens_contour)
        self._require_quality(quality)
        deformed = self._optimize_scene(
            apply_materials(deformed_context.template_scene, fused_measurements)
        )
        s3.finish(
            started,
            rim_pull_strength=rim_pull,
            deformation_mode="template_vertex_groups",
            scale_factors=self._scale_summary(fused_measurements, template_info.dimensions),
            quality=quality.to_dict(),
        )

        s4 = report.add("Texture and Material Mapping")
        started = s4.start()
        s4.finish(
            started,
            material=fused_measurements.material.value,
            lens_opacity=fused_measurements.lens_opacity,
        )

        s5 = report.add("GLB Export and Validation")
        started = s5.start()
        out_path = Path(output_path) if output_path is not None else Path("output") / f"{uuid4().hex[:12]}.glb"
        self.exporter.export(deformed, out_path, fused_measurements, template_name)
        metadata = ExportMetadata(
            shape=fused_measurements.shape.value,
            material=fused_measurements.material.value,
            frame_width=fused_measurements.frame_width,
            bridge_width=fused_measurements.bridge_width,
            temple_length=fused_measurements.temple_length,
            template_used=template_name,
            color=fused_measurements.color,
            lens_color=fused_measurements.lens_color,
            lens_opacity=fused_measurements.lens_opacity,
            scale_source=fused_measurements.scale_source,
            scale_calibrated=fused_measurements.scale_calibrated,
        )
        metadata_path = out_path.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, fused_measurements)
        self.exporter.export_metadata_json(metadata, anchors, metadata_path)
        s5.finish(
            started,
            output=str(out_path),
            bytes=out_path.stat().st_size,
            scale_source=measurement_source,
            quality=quality.to_dict(),
        )

        return {
            "output_glb": str(out_path),
            "metadata_json": str(metadata_path),
            "measurements": fused_measurements.model_dump(),
            "measurement_source": measurement_source,
            "image_estimates": [
                {
                    "view_type": view_type,
                    "measurements": estimate.model_dump(),
                }
                for view_type, estimate in extracted_views
            ],
            "template": template_name,
            "template_selection": {
                "reason": match.best.reason,
                "score": match.best.score,
                "breakdown": match.best.breakdown,
                "top_candidates": [
                    {
                        "template": candidate.template.name,
                        "score": candidate.score,
                        "reason": candidate.reason,
                    }
                    for candidate in match.candidates[:3]
                ],
            },
            "features": features.model_dump(mode="json"),
            "scale_factors": self._scale_summary(fused_measurements, template_info.dimensions),
            "lens_contour": front_lens_contour.model_dump(),
            "quality_report": quality.to_dict(),
            "pipeline": report.to_dict(),
        }

    def _scale_summary(self, m: Measurements, t) -> dict:
        return {
            "frame_x": round(m.frame_width / t.frame_width, 4),
            "lens_x": round(m.lens_width / t.lens_width, 4),
            "lens_y": round(m.lens_height / t.lens_height, 4),
            "bridge_x": round(m.bridge_width / t.bridge_width, 4),
            "temple_length": round(m.temple_length / t.temple_length, 4),
        }

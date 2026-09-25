"""End-to-end template deformation pipeline."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from backend.classifier.shape_classifier import ShapeClassifier
from backend.deformer.engine import MeshDeformer
from backend.exporter.glb_exporter import GLBExporter
from backend.exporter.glb_validator import validate_glb
from backend.materials.pbr import apply_materials
from backend.measurement.extractor import MeasurementExtractor
from backend.models import ExportMetadata, FrameMaterial, FrameShape, Measurements, PipelineReport, StyleClassification
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
        """Add logical mesh names into scene.geometry so DeformationContext can find them.

        The descriptor_loader resolves aliases (e.g. 'LeftRim' -> 'Plane_glasses_mat_0')
        but the scene itself only has the original GLB names. The DeformationContext builds
        its meshes dict straight from scene.geometry, so logical names must be present there.
        """
        # Collect all unique actual→logical mappings from vertex_groups + descriptor parts
        alias_map: dict[str, str] = {}  # logical_name -> actual_glb_name

        # Pull from descriptor raw mesh_aliases if present
        raw_aliases = descriptor.raw.get("mesh_aliases", {})
        if isinstance(raw_aliases, dict):
            for logical, actual in raw_aliases.items():
                if actual in scene.geometry and logical not in scene.geometry:
                    alias_map[logical] = actual

        # Also cover any logical names derived from vertex_groups that aren't in scene yet
        for group_parts in descriptor.vertex_groups.values():
            for logical_name in group_parts:
                if logical_name not in scene.geometry:
                    # Find the actual mesh this logical name maps to via raw_aliases
                    actual = raw_aliases.get(logical_name)
                    if actual and actual in scene.geometry:
                        alias_map[logical_name] = actual

        # Inject: add logical-named references into scene.geometry
        for logical_name, actual_name in alias_map.items():
            if logical_name not in scene.geometry:
                scene.geometry[logical_name] = scene.geometry[actual_name]

    @staticmethod
    def _template_safety_warnings(descriptor) -> list[str]:
        """Return structural warnings that make a template unsafe for automatic approval."""
        raw_aliases = descriptor.raw.get("mesh_aliases", {})
        if not isinstance(raw_aliases, dict):
            return []

        reverse: dict[str, list[str]] = {}
        for logical, actual in raw_aliases.items():
            if isinstance(logical, str) and isinstance(actual, str):
                reverse.setdefault(actual, []).append(logical)

        critical = {
            "Frame", "Bridge", "LeftRim", "RightRim",
            "LeftLens", "RightLens", "LeftTemple", "RightTemple",
        }
        warnings: list[str] = []
        for actual, logical_names in reverse.items():
            overlapping = sorted(critical.intersection(logical_names))
            if len(overlapping) > 1:
                warnings.append(
                    "Template maps multiple deformable logical parts "
                    f"({', '.join(overlapping)}) to the same mesh '{actual}'. "
                    "Use a template with separated production geometry."
                )
        return warnings

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
        style = self._style_from_measurements(measurements)
        features = self.feature_extractor.from_measurements(measurements, style)
        match = self.matcher.match(features, template_name)
        template_info = match.best.template
        template_name = template_info.name

        descriptor = self.descriptor_loader.load(
            template_name,
            measurements=measurements,
            template_info=template_info,
            require_independent_parts=True,
        )
        scene = self.descriptor_loader.build_deformation_scene(descriptor)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=measurements,
            feature_set=features,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx)
        deformed = deformed_ctx.template_scene

        template_warnings = self._template_safety_warnings(descriptor)
        if template_warnings:
            quality.passed = False
            quality.warnings.extend(template_warnings)

        deformed = apply_materials(deformed, measurements)

        out = Path(output_path)
        self.exporter.export(deformed, out, measurements, template_name)
        glb_validation = validate_glb(out, measurements)
        if not glb_validation.passed:
            raise RuntimeError(
                "Exported GLB failed production validation: "
                + "; ".join(glb_validation.errors)
            )

        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter.compute_anchors_meters(deformed, measurements)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "glb_validation": glb_validation.to_dict(),
            "measurements": measurements.model_dump(),
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
            "quality": quality.to_dict(),
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
        descriptor = self.descriptor_loader.load(
            template_name,
            measurements=measurements,
            template_info=template_info,
            require_independent_parts=True,
        )
        scene = self.descriptor_loader.build_deformation_scene(descriptor)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=measurements,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx, lens_contour)
        deformed = deformed_ctx.template_scene
        s6.finish(
            t,
            rim_pull_strength=rim_pull,
            deformation_mode="template_vertex_groups",
            contour_used=bool(lens_contour.left or lens_contour.right),
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
        anchors = self.exporter.compute_anchors_meters(deformed, measurements)

        metadata = ExportMetadata(
            shape=measurements.shape.value,
            material=measurements.material.value,
            frame_width=measurements.frame_width,
            bridge_width=measurements.bridge_width,
            temple_length=measurements.temple_length,
            template_used=template_name,
            color=measurements.color,
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)
        s9.finish(t, output=str(out), size_kb=round(out.stat().st_size / 1024, 1))

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "glb_validation": glb_validation.to_dict(),
            "measurements": measurements.model_dump(),
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
        Run the full pipeline on 4-5 images.

        Photos are the source of truth for segmentation, style, lens contour,
        lens appearance, and template selection. When manual_measurements is
        provided, its physical dimensions are the source of truth for mesh
        deformation; image-derived millimetre estimates are retained only as
        diagnostics.
        """
        report = PipelineReport()
        s1 = report.add("Multi-View Classification and Extraction")
        t = s1.start()

        view_classifier = ViewClassifier()
        fuser = MeasurementFuser()
        opacity_detector = OpacityDetector()

        extracted_views: list[tuple[str, Measurements]] = []
        processed: list[dict] = []

        for path in image_paths:
            img = self._imread(path)
            if img is None:
                continue

            masks = self.segmenter.segment(img)
            mask = masks["front"]
            view_type = view_classifier.classify_view(img, mask)
            style = self.classifier.classify_style(img, mask)
            estimated, lens_contour = self.measurer.extract_from_images(
                img,
                side=None,
                mask=mask,
                shape=style.shape,
                material=style.material,
                nose_pads=style.nose_pads,
                color=color,
            )

            extracted_views.append((view_type, estimated))
            processed.append(
                {
                    "view_type": view_type,
                    "image": img,
                    "mask": mask,
                    "style": style,
                    "estimated": estimated,
                    "lens_contour": lens_contour,
                }
            )

        if not processed:
            raise ValueError("No valid images could be processed")

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
            # Physical dimensions come from the user; semantic appearance comes
            # from the product photos.
            fused_measurements.shape = front_style.shape
            fused_measurements.material = front_style.material
            fused_measurements.nose_pads = front_style.nose_pads
            fused_measurements.color = color
            measurement_source = "manual"
        else:
            fused_measurements = fuser.fuse(extracted_views)
            measurement_source = "image_estimate"

        lens_opacity, lens_color = opacity_detector.detect(front_image, front_lens_contour)
        fused_measurements.lens_color = lens_color
        fused_measurements.lens_opacity = lens_opacity

        s1.finish(
            t,
            views=[record["view_type"] for record in processed],
            measurement_source=measurement_source,
        )

        s2 = report.add("Template Scoring")
        t = s2.start()

        # Keep visual style from the photos while using authoritative manual
        # dimensions for geometric template matching.
        features = self.feature_extractor.extract(
            front_mask,
            fused_measurements,
            front_style,
        )
        match = self.matcher.match(features, template_override)
        template_info = match.best.template
        template_name = template_info.name
        s2.finish(
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

        s3 = report.add("Template Deformation")
        t = s3.start()
        descriptor = self.descriptor_loader.load(
            template_name,
            measurements=fused_measurements,
            template_info=template_info,
            require_independent_parts=True,
        )
        scene = self.descriptor_loader.build_deformation_scene(descriptor)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=fused_measurements,
            feature_set=features,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx, front_lens_contour)
        deformed = deformed_ctx.template_scene

        template_warnings = self._template_safety_warnings(descriptor)
        if template_warnings:
            quality.passed = False
            quality.warnings.extend(template_warnings)

        s3.finish(
            t,
            rim_pull_strength=rim_pull,
            deformation_mode="template_vertex_groups",
            scale_factors=self._scale_summary(fused_measurements, template_info.dimensions),
            quality=quality.to_dict(),
        )

        s4 = report.add("Texture Mapping")
        t = s4.start()
        deformed = apply_materials(deformed, fused_measurements)
        frame_parts = [name for name in deformed.geometry if name not in {"LeftLens", "RightLens"}]
        s4.finish(
            t,
            material=fused_measurements.material.value,
            color=fused_measurements.color,
            lens_color=fused_measurements.lens_color,
            lens_opacity=fused_measurements.lens_opacity,
            frame_parts=frame_parts,
            lens_parts=["LeftLens", "RightLens"],
        )

        s5 = report.add("Mesh Quality Optimization")
        t = s5.start()
        deformed = self._optimize_scene(deformed)
        s5.finish(t, preserved_topology=True, preserved_materials=True)

        s6 = report.add("GLB Export")
        t = s6.start()
        if output_path is None:
            output_path = Path("output") / f"{template_name}_deformed.glb"
        out = Path(output_path)

        self.exporter.export(deformed, out, fused_measurements, template_name)
        glb_validation = validate_glb(out, fused_measurements)
        if not glb_validation.passed:
            raise RuntimeError(
                "Exported GLB failed production validation: "
                + "; ".join(glb_validation.errors)
            )
        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter.compute_anchors_meters(deformed, fused_measurements)

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
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)
        s6.finish(t, output=str(out), size_kb=round(out.stat().st_size / 1024, 1))

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
            "glb_validation": glb_validation.to_dict(),
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
                    {"template": candidate.template.name, "score": candidate.score, "reason": candidate.reason}
                    for candidate in match.candidates[:3]
                ],
            },
            "features": features.model_dump(mode="json"),
            "scale_factors": self._scale_summary(fused_measurements, template_info.dimensions),
            "lens_contour": front_lens_contour.model_dump(),
            "quality": quality.to_dict(),
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

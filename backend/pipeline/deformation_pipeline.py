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
from backend.materials.pbr import apply_materials
from backend.measurement.extractor import MeasurementExtractor
from backend.models import ExportMetadata, Measurements, PipelineReport, StyleClassification
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

        scene = trimesh.load(template_info.glb_path, force="scene")
        
        # Try to load descriptor, fall back to simple deformation if template not properly prepared
        try:
            descriptor = self.descriptor_loader.load(template_name, measurements=measurements, template_info=template_info)
            ctx = DeformationContext(
                template_info=template_info,
                template_scene=scene,
                descriptor=descriptor,
                measurements=measurements,
            )
            rim_pull = self.library.rim_pull_strength(template_name)
            deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
            deformed_ctx, quality = deformer.deform(ctx)
            deformed = deformed_ctx.template_scene
        except (ValueError, KeyError) as e:
            # Fallback to old simple deformation without descriptors
            import warnings
            warnings.warn(f"Template {template_name} not properly prepared (descriptor error: {e}). Using simple deformation fallback.")
            from backend.deformer.engine import MeshDeformer as OldDeformer
            rim_pull = self.library.rim_pull_strength(template_name)
            deformer = OldDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
            deformed = deformer.deform(measurements)
        
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
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
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
        scene = trimesh.load(template_info.glb_path, force="scene")
        descriptor = self.descriptor_loader.load(template_name, measurements=measurements, template_info=template_info)
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
        )
        self.exporter.export_metadata_json(metadata, anchors, meta_path)
        s9.finish(t, output=str(out), size_kb=round(out.stat().st_size / 1024, 1))

        return {
            "output_glb": str(out),
            "metadata_json": str(meta_path),
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
    ) -> dict:
        """
        Run the full pipeline on 4-6 images:
        1. Read and segment all images.
        2. Classify each view (front, side, top, perspective).
        3. Extract measurements and contours from each view.
        4. Fuse the measurements.
        5. Detect lens opacity & dominant color from the front view's lens region.
        6. Apply template deformation and material mapping using the fused measurements.
        """
        report = PipelineReport()
        s1 = report.add("Multi-View Classification and Extraction")
        t = s1.start()
        
        view_classifier = ViewClassifier()
        fuser = MeasurementFuser()
        opacity_detector = OpacityDetector()
        
        extracted_views = []
        front_image = None
        front_lens_contour = None
        
        for path in image_paths:
            img = self._imread(path)
            if img is None:
                continue
            
            masks = self.segmenter.segment(img)
            view_type = view_classifier.classify_view(img, masks["front"])
            
            style = self.classifier.classify_style(img, masks["front"])
            measurements, lens_contour = self.measurer.extract_from_images(
                img,
                side=None,
                mask=masks["front"],
                shape=style.shape,
                material=style.material,
                nose_pads=style.nose_pads,
                color=color,
            )
            
            extracted_views.append((view_type, measurements))
            
            if view_type == "front" and front_image is None:
                front_image = img
                front_lens_contour = lens_contour
                
        if not extracted_views:
            raise ValueError("No valid images could be processed")
            
        # If no front view was classified, pick the first view as front fallback
        if front_image is None:
            first_view_type, first_ms = extracted_views[0]
            front_image = self._imread(image_paths[0])
            extracted_views[0] = ("front", first_ms)
            masks = self.segmenter.segment(front_image)
            front_lens_contour = self.measurer.extract_lens_contour(front_image, masks["front"])
            
        # Fuse measurements
        fused_measurements = fuser.fuse(extracted_views)
        
        # Detect lens opacity and dominant color
        lens_opacity, lens_color = opacity_detector.detect(front_image, front_lens_contour)
        fused_measurements.lens_color = lens_color
        fused_measurements.lens_opacity = lens_opacity
        
        s1.finish(t, views=[v[0] for v in extracted_views])
        
        # Run template selection, deformation, material mapping, and export
        s2 = report.add("Template Scoring")
        t = s2.start()
        
        style = self._style_from_measurements(fused_measurements)
        features = self.feature_extractor.from_measurements(fused_measurements, style)
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
        scene = trimesh.load(template_info.glb_path, force="scene")
        descriptor = self.descriptor_loader.load(template_name, measurements=fused_measurements, template_info=template_info)
        ctx = DeformationContext(
            template_info=template_info,
            template_scene=scene,
            descriptor=descriptor,
            measurements=fused_measurements,
        )
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed_ctx, quality = deformer.deform(ctx, front_lens_contour)
        deformed = deformed_ctx.template_scene
        s3.finish(
            t,
            rim_pull_strength=rim_pull,
            deformation_mode="template_vertex_groups",
            scale_factors=self._scale_summary(fused_measurements, template_info.dimensions),
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
        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, fused_measurements)

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
            "measurements": fused_measurements.model_dump(),
            "template": template_name,
            "features": features.model_dump(mode="json"),
            "scale_factors": self._scale_summary(fused_measurements, template_info.dimensions),
            "lens_contour": front_lens_contour.model_dump(),
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

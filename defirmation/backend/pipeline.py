"""End-to-end template deformation pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from backend.classifier.shape_classifier import ShapeClassifier
from backend.deformer.engine import MeshDeformer
from backend.exporter.glb_exporter import GLBExporter
from backend.materials.pbr import apply_materials
from backend.measurement.extractor import MeasurementExtractor
from backend.models import Measurements
from backend.segmentation.segmenter import GlassesSegmenter
from backend.template_library.loader import TemplateLibrary


class DeformationPipeline:
    """
    Full pipeline:
      Images → Segment → Classify → Measure → Template → Deform → Material → GLB
    """

    def __init__(self, templates_dir: Path | None = None):
        self.segmenter = GlassesSegmenter()
        self.classifier = ShapeClassifier()
        self.measurer = MeasurementExtractor()
        self.library = TemplateLibrary(templates_dir)
        self.exporter = GLBExporter()

    def run_from_images(
        self,
        front_path: Path | str,
        side_path: Path | str | None = None,
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
    ) -> dict:
        front = cv2.imread(str(front_path))
        if front is None:
            raise ValueError(f"Cannot read front image: {front_path}")

        side = None
        if side_path:
            side = cv2.imread(str(side_path))

        return self.run_from_arrays(front, side, output_path, color, template_override)

    def run_from_measurements(
        self,
        measurements: Measurements,
        output_path: Path | str,
        template_name: str | None = None,
    ) -> dict:
        template_name = template_name or self.library.closest_template(
            measurements.shape, measurements.material, measurements.frame_width
        ).name
        template_info = self.library.load(template_name)

        scene = trimesh.load(template_info.glb_path, force="scene")
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed = deformer.deform(measurements)
        deformed = apply_materials(deformed, measurements)

        out = Path(output_path)
        self.exporter.export(deformed, out, measurements, template_name)

        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, measurements)
        from backend.models import ExportMetadata

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
            "scale_factors": self._scale_summary(measurements, template_info.dimensions),
        }

    def run_from_arrays(
        self,
        front: np.ndarray,
        side: np.ndarray | None = None,
        output_path: Path | str | None = None,
        color: str = "#d9a7a2",
        template_override: str | None = None,
    ) -> dict:
        masks = self.segmenter.segment(front)
        shape, material, nose_pads = self.classifier.classify(front, masks["front"])
        measurements, lens_contour = self.measurer.extract_from_images(
            front, side, masks["front"], shape, material, nose_pads, color
        )

        template_name = template_override or self.classifier.template_name_for(shape, material)
        try:
            template_info = self.library.load(template_name)
        except FileNotFoundError:
            template_info = self.library.load("geometric_metal")
            template_name = "geometric_metal"

        scene = trimesh.load(template_info.glb_path, force="scene")
        rim_pull = self.library.rim_pull_strength(template_name)
        deformer = MeshDeformer(scene, template_info.dimensions, rim_pull_strength=rim_pull)
        deformed = deformer.deform(measurements, lens_contour)
        deformed = apply_materials(deformed, measurements)

        if output_path is None:
            output_path = Path("output") / f"{template_name}_deformed.glb"
        out = Path(output_path)

        self.exporter.export(deformed, out, measurements, template_name)
        meta_path = out.with_suffix(".metadata.json")
        anchors = self.exporter._compute_anchors(deformed, measurements)
        from backend.models import ExportMetadata

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
            "scale_factors": self._scale_summary(measurements, template_info.dimensions),
            "lens_contour": lens_contour.model_dump(),
        }

    def _scale_summary(self, m: Measurements, t) -> dict:
        return {
            "frame_x": round(m.frame_width / t.frame_width, 4),
            "lens_x": round(m.lens_width / t.lens_width, 4),
            "lens_y": round(m.lens_height / t.lens_height, 4),
            "bridge_x": round(m.bridge_width / t.bridge_width, 4),
            "temple_length": round(m.temple_length / t.temple_length, 4),
        }

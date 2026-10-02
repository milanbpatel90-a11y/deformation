"""Deform registered basis-template bundles using their supplied linear basis.

The frame contains coupled rim/bridge regions, so this route intentionally does
not apply the separate-part deformer or require fictitious rim/bridge meshes.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from backend.deformer.calibrated_regions import CalibratedRegions
from backend.template_library.compatibility import (
    validate_combination, MeasurementCompatibilityError, dependent_ranges)

from backend.deformer.quality_checker import QualityReport


class BasisDeformer:
    def __init__(self, root: Path):
        self.root = Path(root)
        descriptor_path = self.root / "metadata/template.json"
        self.template_id = self.root.name
        if descriptor_path.is_file():
            self.template_id = json.loads(descriptor_path.read_text(encoding="utf-8")).get("template_id", self.template_id)
        for relative, magic in (("geometry/template.glb", b"glTF"), ("deformation/basis.npz", b"PK")):
            path = self.root / relative
            if not path.is_file():
                expected = ("real GT_001 delivery asset" if self.template_id == "GT_001"
                            else f"real GT_001 delivery asset (or matching {self.template_id} bundle asset)")
                raise FileNotFoundError(f"Install the {expected}: {path}; binary assets are not supplied by a metadata-only checkout")
            with path.open("rb") as source:
                if not source.read(4).startswith(magic):
                    raise ValueError(f"{path} is not a real binary asset (check Git LFS/delivery files)")
        self.basis_metadata = json.loads(
            (self.root / "deformation/basis_metadata.json").read_text(encoding="utf-8"))
        self.parameters = self.basis_metadata["parameters"]
        with np.load(self.root / "deformation/basis.npz", allow_pickle=False) as data:
            self.basis = data["B"]
            self.rest = data["V0"]
            self.faces = data["faces"]
            self.part_order = data["part_order"].tolist()
        self.scene = trimesh.load(self.root / "geometry/template.glb", force="scene", process=False)
        if len(set(self.part_order)) != len(self.part_order) or set(self.part_order) != set(self.scene.geometry):
            raise ValueError(f"{self.template_id} basis part order does not match GLB geometry")
        vertices, faces, offset = [], [], 0
        for name in self.part_order:
            mesh = self.scene.geometry[name]
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
        vertices = np.vstack(vertices)
        if vertices.shape != self.rest.shape or not np.allclose(vertices, self.rest, atol=1e-5, rtol=0):
            raise ValueError(f"{self.template_id} basis rest vertices do not match GLB geometry")
        if not np.array_equal(np.vstack(faces), self.faces):
            raise ValueError(f"{self.template_id} basis faces do not match GLB topology")
        if self.basis.shape != (*self.rest.shape, len(self.parameters)) or not np.isfinite(self.basis).all():
            raise ValueError(f"Invalid {self.template_id} deformation basis")
        if not np.isfinite(self.rest).all():
            raise ValueError(f"Invalid {self.template_id} rest vertices")
        self.slices = {}
        start = 0
        for name in self.part_order:
            end = start + len(self.scene.geometry[name].vertices)
            self.slices[name] = slice(start, end)
            start = end
        self.landmarks = json.loads((self.root / "metadata/landmarks.json").read_text(encoding="utf-8"))["landmarks"]
        self.regions = CalibratedRegions(self.rest, self.basis, self.slices, self.parameters, self.landmarks)
        self.calibration_matrix = self.regions.calibration
        self.reference_measurements = dict(zip(self.regions.names, self.regions.reference))
        self.reference_measurements["temple_length"] = self.regions.temple_reference

    def deform(self, measurements):
        ranges = {p["name"]: {"min": p["min"], "max": p["max"]} for p in self.parameters}
        validate_combination(measurements, ranges)
        for parameter in self.parameters:
            value = getattr(measurements, parameter["name"])
            if not parameter["min"] <= value <= parameter["max"]:
                raise MeasurementCompatibilityError(f'{parameter["name"]} must be between {parameter["min"]} and {parameter["max"]} mm for {self.template_id}', ranges)
        field = self.regions.configure(measurements)
        vertices = field.warp(self.rest)
        if not np.isfinite(vertices).all() or field.min_jacobian <= 0:
            raise ValueError(f"{self.template_id} deformation produced invalid or folded coordinates")
        quality_limits = self.basis_metadata.get("geometry_quality", {})
        minimum_area = float(quality_limits.get("minimum_triangle_area_mm2", 1e-10))
        maximum_stretch = float(quality_limits.get("maximum_edge_stretch_ratio", 3.0))
        faces = np.asarray(self.faces, dtype=np.int64)
        target_triangles = vertices[faces]
        target_normals = np.cross(target_triangles[:, 1] - target_triangles[:, 0],
                                  target_triangles[:, 2] - target_triangles[:, 0])
        areas = np.linalg.norm(target_normals, axis=1) * 0.5
        if np.any(areas <= minimum_area):
            raise ValueError(f"{self.template_id} deformation created degenerate triangles")
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        source_lengths = np.linalg.norm(self.rest[edges[:, 0]] - self.rest[edges[:, 1]], axis=1)
        target_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        nonzero = source_lengths > 1e-12
        stretch = target_lengths[nonzero] / source_lengths[nonzero]
        max_stretch = float(stretch.max(initial=1.0))
        min_stretch = float(stretch.min(initial=1.0))
        if not np.isfinite(stretch).all() or max_stretch > maximum_stretch:
            worst = int(np.argmax(stretch))
            source_edge = edges[nonzero][worst]
            source_part = next((name for name, sl in self.slices.items()
                                if sl.start <= source_edge[0] < sl.stop), "unknown")
            base_ranges = {p["name"]: {"min": p["min"], "max": p["max"]} for p in self.parameters}
            ranges = dependent_ranges(measurements, base_ranges)
            ranges["geometry_quality"] = {
                "maximum_edge_stretch_ratio": max_stretch,
                "maximum_allowed_edge_stretch_ratio": maximum_stretch,
                "limiting_part": source_part,
            }
            raise MeasurementCompatibilityError(
                f"{self.template_id} maximum edge stretch {max_stretch:.4g} exceeds {maximum_stretch:g} "
                f"in {source_part} (edge vertices {int(source_edge[0])}, {int(source_edge[1])}; "
                f"FW/LW/LH/BW/TL={measurements.frame_width:g}/{measurements.lens_width:g}/"
                f"{measurements.lens_height:g}/{measurements.bridge_width:g}/{measurements.temple_length:g} mm). "
                "Reduce the dimensions until the selected template remains within its geometry limits.",
                ranges,
            )
        scene = self.scene.copy()
        for name, sl in self.slices.items():
            scene.geometry[name].vertices = vertices[sl].copy()
        measured = field.measure(vertices)
        expected = np.array([getattr(measurements,n) for n in field.names])
        errors = np.abs(measured-expected)
        endpoints = field.warp(np.vstack([field.temple_root,field.temple_tip]))
        temple_length = float(np.linalg.norm(endpoints[1]-endpoints[0]))
        maximum_error = max(float(errors.max()),abs(temple_length-measurements.temple_length))
        if maximum_error > 0.5:
            raise ValueError(f"{self.template_id} failed physical dimension validation: {maximum_error:g} mm")
        # Native Z-up -> Y-up. All internal deformation and anchor inputs stay mm.
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        from backend.exporter.geometry import world_vertices
        def to_viewer(point):
            x, y, z = point
            return [float(x), float(z), float(-y)]

        bridge = field.warp(np.asarray([self.landmarks.get("LM_BridgeCenter", [0, 0, 0])], dtype=float))[0]
        anchors = {"NoseBridge": to_viewer(bridge)}
        for name in ("LeftHinge", "RightHinge"):
            if name in scene.geometry:
                points = world_vertices(scene, name)
                anchors[name] = ((points.min(0) + points.max(0)) / 2).tolist()
            else:
                landmark = self.landmarks.get(f"LM_{name}")
                if landmark is not None:
                    anchors[name] = to_viewer(field.warp(np.asarray([landmark], dtype=float))[0])
        scene.metadata["vto_anchors"] = anchors
        scene.metadata["coordinate_units"] = "mm"
        scene.metadata["deformation_validation"] = {
            "measured_mm": dict(zip(field.names, measured.tolist())) | {"temple_length": temple_length},
            "maximum_error_mm": maximum_error, "minimum_jacobian": field.min_jacobian,
            "basis_calibration_matrix": self.calibration_matrix.tolist(),
            "basis_coefficients": field.coefficients.tolist(),
            "bridge_definition": "horizontal gap between lens bounding boxes",
            "maximum_edge_stretch_ratio": max_stretch,
            "minimum_edge_stretch_ratio": min_stretch,
            "minimum_triangle_area_mm2": float(areas.min()),
            "continuous_map_orientation_preserved": True,
        }
        return scene, QualityReport(passed=True, warnings=[
            f"{self.template_id} retains its authored rim cross-section and temple curve; rim thickness reserves outer clearance in coupled validation."],
            breakdown={"finite_geometry": 100.0, "dimension_tolerance_0_5mm": 100.0,
                       "positive_deformation_jacobian": 100.0})

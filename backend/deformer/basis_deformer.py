"""Deform registered Gold Template bundles using their supplied linear basis.

The frame contains coupled rim/bridge regions, so this route intentionally does
not apply the separate-part deformer or require fictitious rim/bridge meshes.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from backend.deformer.calibrated_regions import CalibratedRegions
from backend.template_library.compatibility import validate_combination, MeasurementCompatibilityError

from backend.deformer.quality_checker import QualityReport


class BasisDeformer:
    def __init__(self, root: Path):
        self.root = Path(root)
        for relative, magic in (("geometry/template.glb", b"glTF"), ("deformation/basis.npz", b"PK")):
            path = self.root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Install real GT_001 delivery asset: {path}; binary assets are not supplied by a metadata-only checkout")
            with path.open("rb") as source:
                if not source.read(4).startswith(magic):
                    raise ValueError(f"{path} is not a real binary asset (check Git LFS/delivery files)")
        self.parameters = json.loads((self.root / "deformation/basis_metadata.json").read_text(encoding="utf-8"))["parameters"]
        with np.load(self.root / "deformation/basis.npz", allow_pickle=False) as data:
            self.basis = data["B"]
            self.rest = data["V0"]
            self.faces = data["faces"]
            self.part_order = data["part_order"].tolist()
        self.scene = trimesh.load(self.root / "geometry/template.glb", force="scene", process=False)
        if len(set(self.part_order)) != len(self.part_order) or set(self.part_order) != set(self.scene.geometry):
            raise ValueError("Gold Template basis part order does not match GLB geometry")
        vertices, faces, offset = [], [], 0
        for name in self.part_order:
            mesh = self.scene.geometry[name]
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
        vertices = np.vstack(vertices)
        if vertices.shape != self.rest.shape or not np.allclose(vertices, self.rest, atol=1e-5, rtol=0):
            raise ValueError("Gold Template basis rest vertices do not match GLB geometry")
        if not np.array_equal(np.vstack(faces), self.faces):
            raise ValueError("Gold Template basis faces do not match GLB topology")
        if self.basis.shape != (*self.rest.shape, len(self.parameters)) or not np.isfinite(self.basis).all():
            raise ValueError("Invalid Gold Template deformation basis")
        if not np.isfinite(self.rest).all():
            raise ValueError("Invalid Gold Template rest vertices")
        self.slices = {}
        start = 0
        for name in self.part_order:
            end = start + len(self.scene.geometry[name].vertices)
            self.slices[name] = slice(start, end)
            start = end
        landmarks = json.loads((self.root / "metadata/landmarks.json").read_text(encoding="utf-8"))["landmarks"]
        self.regions = CalibratedRegions(self.rest, self.basis, self.slices, self.parameters, landmarks)
        self.calibration_matrix = self.regions.calibration
        self.reference_measurements = dict(zip(self.regions.names, self.regions.reference))
        self.reference_measurements["temple_length"] = self.regions.temple_reference

    def deform(self, measurements):
        ranges = {p["name"]: {"min": p["min"], "max": p["max"]} for p in self.parameters}
        validate_combination(measurements, ranges)
        for parameter in self.parameters:
            value = getattr(measurements, parameter["name"])
            if not parameter["min"] <= value <= parameter["max"]:
                raise MeasurementCompatibilityError(f'{parameter["name"]} must be between {parameter["min"]} and {parameter["max"]} mm for Gold Template', ranges)
        field = self.regions.configure(measurements)
        vertices = field.warp(self.rest)
        if not np.isfinite(vertices).all() or field.min_jacobian <= 0:
            raise ValueError("Gold Template deformation produced invalid or folded coordinates")
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
            raise ValueError(f"Gold Template failed physical dimension validation: {maximum_error:g} mm")
        # Native Z-up -> Y-up. All internal deformation and anchor inputs stay mm.
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        from backend.exporter.geometry import world_vertices
        anchors = {"NoseBridge": [0.0, 0.0, 0.0]}
        for name in ("LeftHinge", "RightHinge"):
            points = world_vertices(scene, name)
            anchors[name] = ((points.min(0) + points.max(0)) / 2).tolist()
        scene.metadata["vto_anchors"] = anchors
        scene.metadata["coordinate_units"] = "mm"
        scene.metadata["deformation_validation"] = {
            "measured_mm": dict(zip(field.names, measured.tolist())) | {"temple_length": temple_length},
            "maximum_error_mm": maximum_error, "minimum_jacobian": field.min_jacobian,
            "basis_calibration_matrix": self.calibration_matrix.tolist(),
            "basis_coefficients": field.coefficients.tolist(),
            "bridge_definition": "horizontal gap between lens bounding boxes",
        }
        return scene, QualityReport(passed=True, warnings=[
            "Gold Template retains its authored rim cross-section and temple curve; rim thickness reserves outer clearance in coupled validation."],
            breakdown={"finite_geometry": 100.0, "dimension_tolerance_0_5mm": 100.0,
                       "positive_deformation_jacobian": 100.0})

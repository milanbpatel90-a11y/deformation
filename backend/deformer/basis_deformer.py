"""Deform registered Gold Template bundles using their supplied linear basis.

The frame contains coupled rim/bridge regions, so this route intentionally does
not apply the separate-part deformer or require fictitious rim/bridge meshes.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from backend.deformer.quality_checker import QualityReport


class BasisDeformer:
    def __init__(self, root: Path):
        self.root = Path(root)
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
        # The supplied basis shortens the outer temples but leaves their inner
        # inserts at the original length. Transfer the temple displacement to
        # each insert so its metal core cannot protrude from a shortened arm.
        offsets = {}
        start = 0
        for name in self.part_order:
            end = start + len(self.scene.geometry[name].vertices)
            offsets[name] = slice(start, end)
            start = end
        for side in ("Left", "Right"):
            temple = offsets[side + "Temple"]
            insert = offsets[side + "FrameInsert"]
            distance, nearest = cKDTree(self.rest[temple]).query(self.rest[insert], k=4)
            weights = 1 / np.maximum(distance, 1e-5)
            weights /= weights.sum(axis=1, keepdims=True)
            for index, parameter in enumerate(self.parameters):
                if parameter["name"] in {"temple_length", "frame_width"}:
                    self.basis[insert, :, index] = np.sum(
                        self.basis[temple, :, index][nearest] * weights[:, :, None], axis=1)

    def deform(self, measurements):
        deltas = []
        for parameter in self.parameters:
            value = getattr(measurements, parameter["name"])
            if not parameter["min"] <= value <= parameter["max"]:
                raise ValueError(f'{parameter["name"]} must be between {parameter["min"]} and {parameter["max"]} mm for Gold Template')
            deltas.append(value - parameter["default"])
        vertices = self.rest + self.basis @ np.asarray(deltas)
        if not np.isfinite(vertices).all():
            raise ValueError("Gold Template deformation produced invalid coordinates")
        scene = self.scene.copy()
        offset = 0
        for name in self.part_order:
            mesh = scene.geometry[name]
            end = offset + len(mesh.vertices)
            mesh.vertices = vertices[offset:end].copy()
            offset = end
        # The delivery calls its default width 140 mm, but the actual outer
        # frame is 145.332 mm. Normalize the complete assembly together so
        # relative component alignment is retained and requested width is real.
        frame_width = np.ptp(scene.geometry["Frame"].vertices[:, 0])
        width_scale = measurements.frame_width / frame_width
        for mesh in scene.geometry.values():
            mesh.vertices[:, 0] *= width_scale
        # Source: X across, Y depth, Z up. Viewer/export convention: Y up.
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        from backend.exporter.geometry import world_vertices
        anchors = {"NoseBridge": [0.0, 0.0, 0.0]}
        for name in ("LeftHinge", "RightHinge"):
            points = world_vertices(scene, name)
            anchors[name] = ((points.min(0) + points.max(0)) / 2).tolist()
        scene.metadata["vto_anchors"] = anchors
        scene.metadata["coordinate_units"] = "mm"
        warnings = ["Gold Template preserves the supplied rim thickness and temple curve; its basis adjusts the five primary dimensions."]
        dimension_errors = []
        # Nominal basis controls are not certified geometric measurements.
        # Make residual errors explicit instead of reporting requested values
        # as measured dimensions or silently claiming a production-quality fit.
        for name in ("LeftLens", "RightLens"):
            extent = np.ptp(world_vertices(scene, name), axis=0)
            dimension_errors.extend([abs(extent[0] - measurements.lens_width), abs(extent[1] - measurements.lens_height)])
            if abs(extent[0] - measurements.lens_width) > 0.5 or abs(extent[1] - measurements.lens_height) > 0.5:
                warnings.append(f"{name} measured {extent[0]:.2f} x {extent[1]:.2f} mm; requested {measurements.lens_width:.2f} x {measurements.lens_height:.2f} mm. Template calibration is required for an exact fit.")
        dimensions_passed = bool(max(dimension_errors) <= 0.5)
        return scene, QualityReport(score=100.0 if dimensions_passed else 50.0,
            passed=dimensions_passed, warnings=warnings,
            breakdown={"finite_geometry": 100.0, "dimension_tolerance_0_5mm": 100.0 if dimensions_passed else 0.0})

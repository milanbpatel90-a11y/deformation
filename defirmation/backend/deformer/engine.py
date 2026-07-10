"""Mesh deformation engine — per-part vertex scaling and contour fitting."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from backend.models import LensContour, Measurements, ScaleFactors, TemplateDimensions


PART_NAMES = [
    "Frame",
    "LeftLens",
    "RightLens",
    "Bridge",
    "LeftRim",
    "RightRim",
    "LeftTemple",
    "RightTemple",
    "NosePads",
    "TempleTips",
]


class MeshDeformer:
    """
    Deform template vertices to match target measurements.

    Each named part is transformed independently:
    - Front view: frame width, lens size, bridge width
    - Side view: temple length, curve, thickness
    - Nose pads: distance, angle, height
    - Lens contour: rim vertices pulled toward detected polygon
    """

    def __init__(
        self,
        template_scene: trimesh.Scene,
        template_dims: TemplateDimensions,
        rim_pull_strength: float = 0.65,
    ):
        self.template_dims = template_dims
        self.rim_pull_strength = rim_pull_strength
        self.scene = deepcopy(template_scene)
        self._part_centers = self._compute_part_centers()

    def deform(
        self,
        measurements: Measurements,
        lens_contour: LensContour | None = None,
    ) -> trimesh.Scene:
        scales = ScaleFactors.from_measurements(measurements, self.template_dims)

        self._deform_front(scales, measurements)
        self._deform_temples(scales, measurements)
        if measurements.nose_pads:
            self._deform_nose_pads(measurements)
        if lens_contour and (lens_contour.left or lens_contour.right):
            self._deform_lens_contour(lens_contour, measurements)

        return self.scene

    def _compute_part_centers(self) -> dict[str, np.ndarray]:
        centers = {}
        for name, geom in self.scene.geometry.items():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                centers[name] = geom.vertices.mean(axis=0)
            else:
                centers[name] = np.zeros(3)
        return centers

    def _get_geom(self, name: str) -> trimesh.Trimesh | None:
        if name not in self.scene.geometry:
            return None
        return self.scene.geometry[name]

    def _deform_front(self, scales: ScaleFactors, measurements: Measurements) -> None:
        """Scale frame, lenses, bridge, and rims from front-view measurements."""
        frame_center_x = 0.0

        # Global frame width stretch along X
        for part in ["Frame", "LeftRim", "RightRim", "Bridge", "LeftLens", "RightLens"]:
            geom = self._get_geom(part)
            if geom is None:
                continue
            center = self._part_centers.get(part, np.zeros(3))
            self._scale_vertices(geom, center, scales.frame_x, 1.0, scales.lens_y)

        # Per-lens horizontal scaling
        for part, sign in [("LeftLens", -1), ("RightLens", 1), ("LeftRim", -1), ("RightRim", 1)]:
            geom = self._get_geom(part)
            if geom is None:
                continue
            center = self._part_centers.get(part, np.zeros(3))
            extra_x = scales.lens_x / max(scales.frame_x, 0.01)
            self._scale_vertices_about(geom, center, extra_x, 1.0, scales.lens_y)

        # Bridge width
        bridge = self._get_geom("Bridge")
        if bridge is not None:
            center = self._part_centers.get("Bridge", np.zeros(3))
            bridge_scale = scales.bridge_x / max(scales.frame_x, 0.01)
            self._scale_vertices_about(bridge, center, bridge_scale, 1.0, 1.0)

        # Rim thickness (Y scale on rim parts only)
        for part in ["LeftRim", "RightRim"]:
            geom = self._get_geom(part)
            if geom is None:
                continue
            center = self._part_centers.get(part, np.zeros(3))
            self._scale_vertices_about(geom, center, 1.0, scales.rim_thickness, 1.0)

    def _deform_temples(self, scales: ScaleFactors, measurements: Measurements) -> None:
        """Extend temples along X and apply curve angle from side view."""
        for part, sign in [("LeftTemple", -1), ("RightTemple", 1)]:
            geom = self._get_geom(part)
            if geom is None:
                continue

            center = self._part_centers.get(part, np.zeros(3))
            verts = geom.vertices.copy()

            # Scale length along outward X
            hinge = center.copy()
            hinge[0] = 0.0
            for i, v in enumerate(verts):
                delta = v - hinge
                if sign * delta[0] > 0:
                    verts[i, 0] = hinge[0] + delta[0] * scales.temple_length
                    verts[i, 1] = hinge[1] + delta[1] * (1.0 + (scales.temple_length - 1) * 0.3)
                    verts[i, 2] = hinge[2] + delta[2] * scales.temple_thickness

            # Apply temple curve rotation
            angle_delta = measurements.temple_curve_angle - self.template_dims.temple_curve_angle
            if abs(angle_delta) > 0.5:
                rot = Rotation.from_euler("y", sign * angle_delta, degrees=True)
                tip_mask = sign * (verts[:, 0] - hinge[0]) > 0
                for i in np.where(tip_mask)[0]:
                    rel = verts[i] - hinge
                    verts[i] = hinge + rot.apply(rel)

            geom.vertices = verts

        # Temple tips follow temple ends
        for part, temple_part in [("TempleTips", None)]:
            tips = self._get_geom(part)
            if tips is None:
                continue
            # Tips are children of temple deformation in combined mesh; scale similarly
            for t_name, sign in [("LeftTemple", -1), ("RightTemple", 1)]:
                pass

    def _deform_nose_pads(self, measurements: Measurements) -> None:
        pads = self._get_geom("NosePads")
        if pads is None:
            return

        template = self.template_dims
        target_dist = measurements.nose_pad_distance or template.nose_pad_distance
        target_angle = measurements.nose_pad_angle or template.nose_pad_angle
        target_height = measurements.nose_pad_height or template.nose_pad_height

        dist_scale = target_dist / template.nose_pad_distance
        height_scale = target_height / template.nose_pad_height
        angle_delta = target_angle - template.nose_pad_angle

        center = self._part_centers.get("NosePads", np.zeros(3))
        verts = pads.vertices.copy()

        for i, v in enumerate(verts):
            delta = v - center
            verts[i, 0] = center[0] + delta[0] * dist_scale
            verts[i, 1] = center[1] + delta[1] * height_scale
            if abs(angle_delta) > 0.5:
                rot = Rotation.from_euler("z", angle_delta, degrees=True)
                rel = verts[i] - center
                rel[2] = 0
                rotated = rot.apply(rel)
                verts[i, 0] = center[0] + rotated[0]
                verts[i, 1] = center[1] + rotated[1]

        pads.vertices = verts

    def _deform_lens_contour(self, contour: LensContour, measurements: Measurements) -> None:
        """Move rim vertices toward detected 8-vertex lens polygon."""
        frame_w = measurements.frame_width
        frame_h = measurements.lens_height * 1.15

        for part, poly, x_sign in [
            ("LeftRim", contour.left, -1),
            ("RightRim", contour.right, 1),
        ]:
            if not poly:
                continue
            geom = self._get_geom(part)
            if geom is None:
                continue

            verts = geom.vertices.copy()
            center = verts.mean(axis=0)

            target_points = self._polygon_to_3d(poly, frame_w, frame_h, x_sign, center[1])

            for i, v in enumerate(verts):
                # Only move outer rim vertices (farthest from center in XY)
                dist = np.linalg.norm(v[:2] - center[:2])
                if dist < np.linalg.norm(verts[:, :2] - center[:2], axis=1).max() * 0.6:
                    continue

                nearest = self._nearest_point_on_polygon(v[:2], target_points[:, :2])
                blend = self.rim_pull_strength
                verts[i, 0] = v[0] * (1 - blend) + nearest[0] * blend
                verts[i, 1] = v[1] * (1 - blend) + nearest[1] * blend

            geom.vertices = verts

    def _polygon_to_3d(
        self,
        poly: list[list[float]],
        frame_w: float,
        frame_h: float,
        x_sign: int,
        center_y: float,
    ) -> np.ndarray:
        points = []
        half_w = frame_w / 2
        for u, v in poly:
            x = (u - 0.5) * frame_w
            if x_sign < 0:
                x = min(x, -self.template_dims.bridge_width / 2)
            else:
                x = max(x, self.template_dims.bridge_width / 2)
            y = center_y + (0.5 - v) * frame_h
            points.append([x, y, 0.0])
        return np.array(points)

    def _nearest_point_on_polygon(self, point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        if len(polygon) == 0:
            return point
        dists = np.linalg.norm(polygon - point, axis=1)
        return polygon[np.argmin(dists)]

    def _scale_vertices(
        self,
        geom: trimesh.Trimesh,
        center: np.ndarray,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        self._scale_vertices_about(geom, center, sx, sy, sz)

    def _scale_vertices_about(
        self,
        geom: trimesh.Trimesh,
        center: np.ndarray,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        verts = geom.vertices.copy()
        for i, v in enumerate(verts):
            delta = v - center
            verts[i] = center + np.array([delta[0] * sx, delta[1] * sy, delta[2] * sz])
        geom.vertices = verts

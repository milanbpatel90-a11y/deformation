"""Topology-aware rim deformation driven by measurements and (optional) lens contours.

Design
------
Each rim is reshaped with a smooth, per-vertex *warp* of the template's own vertices,
not by replacing vertices with a resampled outline:

* The warp is polar about the lens centre in the lens plane (X, Y). For every azimuth
  ``theta`` it scales radially by ``k(theta) = r_target(theta) / r_template(theta)``,
  so the rim follows the target lens outline while every rim vertex keeps its own
  position relative to the lens: lugs, brow bars, end pieces and fused scan fragments
  move with it instead of being collapsed onto one ring.
* The template outline ``r_template`` is measured from the template's own lens mesh,
  and the baseline target is that same outline scaled to the requested lens width and
  height. Requesting the template's own measurements therefore reproduces the template
  exactly (identity), instead of snapping it to a generic outline.
* An image-derived lens contour (``LensContour``) only changes the *shape*: it is
  blended with the scaled template outline by ``contour_strength``. Size always comes
  from the measurements.
* Depth (Z) is preserved. It is only scaled, about the rim's own mid-plane, by the
  ratio of the requested to the template's declared ``rim_thickness``.

The same displacement is applied to the aggregate ``Frame`` vertices that coincide with
rim vertices (assigned from the deformed rim vertex, never accumulated, so duplicated
seam vertices cannot be moved twice), and falls off smoothly onto nearby frame vertices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.models import LensContour

# Number of azimuth samples used to describe an outline in polar form.
_OUTLINE_SAMPLES = 360
# A single azimuth may not stretch/shrink the rim by more than this factor.
_MIN_RADIAL_SCALE = 0.4
_MAX_RADIAL_SCALE = 2.5
# Soft propagation of rim motion onto neighbouring (non-coincident) frame vertices.
_FRAME_FALLOFF_RADIUS_MM = 12.0
_FRAME_FALLOFF_GAIN = 0.2


@dataclass(frozen=True)
class RimSideResult:
    """Debug summary for one rim deformation pass."""

    side: str
    width_before: float
    width_after: float
    height_before: float
    height_after: float
    contour_points: int


class RimDeformer(BaseDeformer):
    """Reshape left and right rims while preserving topology and depth."""

    stage_name = "rim_deformation"

    def __init__(self, contour_strength: float = 0.7):
        self.contour_strength = float(np.clip(contour_strength, 0.0, 1.0))

    # ------------------------------------------------------------------ apply
    def apply(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> DeformationContext:
        contour = lens_contour or LensContour()
        results: list[RimSideResult] = []

        for side in ("left", "right"):
            rim_parts = context.descriptor.vertex_groups.get(f"{side}_rim", [])
            if not rim_parts:
                continue
            frame_meshes = context.vertex_group_meshes("frame")
            if not frame_meshes:
                continue

            rim_mesh = context.mesh(rim_parts[0])
            frame_mesh = frame_meshes[0]
            before = rim_mesh.vertices.copy()

            template_outline = self._template_outline(context, side, before)
            target_outline, n_points = self._target_outline(context, side, contour, template_outline)
            rim_mesh.vertices = self._warp_rim(
                before,
                template_outline,
                target_outline,
                depth_scale=self._depth_scale(context),
            )
            self._propagate_to_frame(frame_mesh, before, rim_mesh.vertices)
            context.rim_warps[side] = (
                lambda verts, _t=template_outline, _g=target_outline: self._warp_rim(verts, _t, _g, 1.0)
            )

            span_before = np.ptp(before[:, :2], axis=0)
            span_after = np.ptp(rim_mesh.vertices[:, :2], axis=0)
            results.append(
                RimSideResult(
                    side=side,
                    width_before=round(float(span_before[0]), 3),
                    width_after=round(float(span_after[0]), 3),
                    height_before=round(float(span_before[1]), 3),
                    height_after=round(float(span_after[1]), 3),
                    contour_points=n_points,
                )
            )

        return self.update_context(
            context,
            applied=bool(results),
            sides=[result.__dict__ for result in results],
            contour_strength=self.contour_strength,
        )

    # ------------------------------------------------------- outline handling
    @dataclass(frozen=True)
    class _Outline:
        """Star-shaped outline about ``center`` as radius per azimuth sample."""

        center: np.ndarray  # (2,)
        radius: np.ndarray  # (_OUTLINE_SAMPLES,)

    @staticmethod
    def _azimuths() -> np.ndarray:
        return np.linspace(-np.pi, np.pi, _OUTLINE_SAMPLES, endpoint=False)

    @classmethod
    def _polygon_radius_profile(cls, polygon: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Distance from ``center`` to the polygon boundary along each azimuth (outermost hit)."""
        angles = cls._azimuths()
        directions = np.column_stack([np.cos(angles), np.sin(angles)])  # (M, 2)
        starts = polygon - center
        ends = np.roll(polygon, -1, axis=0) - center
        edge = ends - starts  # (E, 2)

        # Solve center + t*d = start + u*edge for t >= 0, 0 <= u <= 1 (2x2 cross products).
        denom = directions[:, None, 0] * edge[None, :, 1] - directions[:, None, 1] * edge[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (starts[None, :, 0] * edge[None, :, 1] - starts[None, :, 1] * edge[None, :, 0]) / denom
            u = (starts[None, :, 0] * directions[:, None, 1] - starts[None, :, 1] * directions[:, None, 0]) / denom
        valid = np.isfinite(t) & (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
        t = np.where(valid, t, -np.inf)
        radius = t.max(axis=1)

        # An azimuth that grazes a vertex exactly can miss every edge: fill from neighbours.
        missing = ~np.isfinite(radius)
        if missing.any():
            good = ~missing
            if not good.any():
                return np.full(_OUTLINE_SAMPLES, max(float(np.linalg.norm(starts, axis=1).mean()), 1e-6))
            radius[missing] = np.interp(
                angles[missing], angles[good], radius[good], period=2.0 * np.pi
            )
        return radius

    @classmethod
    def _outline_from_points(cls, points_xy: np.ndarray) -> "RimDeformer._Outline":
        """Convex-hull outline of a point cloud (lens meshes are convex or near-convex)."""
        pts = np.unique(np.round(points_xy, 6), axis=0)
        center = (pts.min(axis=0) + pts.max(axis=0)) * 0.5
        if len(pts) >= 3:
            try:
                polygon = pts[ConvexHull(pts).vertices]
            except Exception:  # degenerate (collinear) cloud
                polygon = None
        else:
            polygon = None
        if polygon is None or len(polygon) < 3:
            half = np.maximum((pts.max(axis=0) - pts.min(axis=0)) * 0.5, 1e-3)
            polygon = np.array(
                [
                    center + half * [-1, -1],
                    center + half * [1, -1],
                    center + half * [1, 1],
                    center + half * [-1, 1],
                ]
            )
        return cls._Outline(center=center, radius=cls._polygon_radius_profile(polygon, center))

    def _template_outline(
        self, context: DeformationContext, side: str, rim_vertices: np.ndarray
    ) -> "RimDeformer._Outline":
        """Outline of the template's lens (falls back to the rim itself if there is no lens)."""
        lens_meshes = context.vertex_group_meshes(f"{side}_lens")
        if lens_meshes and len(lens_meshes[0].vertices) >= 3:
            return self._outline_from_points(lens_meshes[0].vertices[:, :2])
        return self._outline_from_points(rim_vertices[:, :2])

    def _target_outline(
        self,
        context: DeformationContext,
        side: str,
        contour: LensContour,
        template: "RimDeformer._Outline",
    ) -> tuple["RimDeformer._Outline", int]:
        measured = context.measurements
        dims = context.template_info.dimensions if context.template_info is not None else None
        sign = -1.0 if side == "left" else 1.0

        # Width / height of the template lens from its own outline extent.
        angles = self._azimuths()
        xs = template.radius * np.cos(angles)
        ys = template.radius * np.sin(angles)
        template_width = float(xs.max() - xs.min())
        template_height = float(ys.max() - ys.min())

        sx = float(measured.lens_width) / max(template_width, 1e-6)
        sy = float(measured.lens_height) / max(template_height, 1e-6)

        # Polar radius of the template outline scaled anisotropically by (sx, sy):
        # a point at azimuth theta lands at (sx*x, sy*y), whose azimuth/radius change.
        scaled_x = xs * sx
        scaled_y = ys * sy
        scaled_polygon = np.column_stack([scaled_x, scaled_y])
        scaled_radius = self._polygon_radius_profile(scaled_polygon, np.zeros(2))

        # Lens centre shift from the requested bridge width, relative to the template's own
        # declared bridge width, so identical measurements give zero shift. Each lens moves
        # by half the change; the shift is mirrored for the two sides.
        shift_x = 0.0
        if dims is not None:
            shift_x = sign * 0.5 * (float(measured.bridge_width) - float(dims.bridge_width))
        # Lens centre also follows its own width change (outer edge pinned to the hinge side
        # by the temple stage; here the lens grows symmetrically about its centre).
        target_center = template.center + np.array([shift_x, 0.0])

        radius = scaled_radius
        n_points = _OUTLINE_SAMPLES

        raw_points = contour.left if side == "left" else contour.right
        contour_mm = self._contour_points_to_mm(raw_points, side, context)
        if contour_mm is not None:
            c_center = (contour_mm.min(axis=0) + contour_mm.max(axis=0)) * 0.5
            contour_radius = self._polygon_radius_profile(contour_mm, c_center)
            s = self.contour_strength
            radius = (1.0 - s) * scaled_radius + s * contour_radius
            n_points = len(contour_mm)

        return self._Outline(center=target_center, radius=radius), n_points

    def _contour_points_to_mm(
        self,
        points: list[list[float]],
        side: str,
        context: DeformationContext,
    ) -> np.ndarray | None:
        """Convert a normalised image contour (0..1 over the whole frame) to lens-sized mm points."""
        if len(points) < 4:
            return None
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None

        local_x = arr[:, 0] / 0.5 if side == "left" else (arr[:, 0] - 0.5) / 0.5
        local_y = arr[:, 1]
        local_x = np.clip(local_x, 0.0, 1.0)
        local_y = np.clip(local_y, 0.0, 1.0)

        measured = context.measurements
        x_mm = (local_x - 0.5) * measured.lens_width
        y_mm = (0.5 - local_y) * measured.lens_height
        return np.column_stack([x_mm, y_mm])

    # ----------------------------------------------------------------- warping
    def _depth_scale(self, context: DeformationContext) -> float:
        dims = context.template_info.dimensions if context.template_info is not None else None
        requested = getattr(context.measurements, "rim_thickness", None)
        if dims is None or requested is None or not np.isfinite(requested):
            return 1.0
        reference = float(getattr(dims, "rim_thickness", 0.0) or 0.0)
        if reference <= 1e-6:
            return 1.0
        return float(np.clip(float(requested) / reference, 0.6, 1.6))

    def _warp_rim(
        self,
        vertices: np.ndarray,
        template: "RimDeformer._Outline",
        target: "RimDeformer._Outline",
        depth_scale: float,
    ) -> np.ndarray:
        angles = self._azimuths()
        ratio = np.clip(
            target.radius / np.maximum(template.radius, 1e-6),
            _MIN_RADIAL_SCALE,
            _MAX_RADIAL_SCALE,
        )

        offsets = vertices[:, :2] - template.center
        theta = np.arctan2(offsets[:, 1], offsets[:, 0])
        k = np.interp(theta, angles, ratio, period=2.0 * np.pi)

        warped = vertices.copy()
        warped[:, :2] = target.center + offsets * k[:, None]

        if abs(depth_scale - 1.0) > 1e-9:
            mid_z = float((vertices[:, 2].min() + vertices[:, 2].max()) * 0.5)
            warped[:, 2] = mid_z + (vertices[:, 2] - mid_z) * depth_scale
        return warped

    # ------------------------------------------------------------- propagation
    @staticmethod
    def _propagate_to_frame(
        frame_mesh,
        original_rim_vertices: np.ndarray,
        deformed_rim_vertices: np.ndarray,
    ) -> None:
        """Carry the rim motion into the aggregate Frame mesh.

        Frame vertices that coincide with a rim vertex take that vertex's deformed position
        (assignment, so coincident duplicates are never moved twice). Other frame vertices
        within the falloff radius receive a smooth, weight-averaged share of the nearby
        rim displacements.
        """
        frame_vertices = frame_mesh.vertices.copy()
        deltas = deformed_rim_vertices - original_rim_vertices
        if not np.any(np.abs(deltas) > 0.0):
            return

        rim_tree = cKDTree(original_rim_vertices)
        distance, nearest = rim_tree.query(frame_vertices)
        exact = distance < 1e-5
        frame_vertices[exact] = deformed_rim_vertices[nearest[exact]]

        soft_idx = np.where(~exact)[0]
        if len(soft_idx) > 0:
            neighbours = rim_tree.query_ball_point(frame_vertices[soft_idx], r=_FRAME_FALLOFF_RADIUS_MM)
            for local, rim_ids in enumerate(neighbours):
                if not rim_ids:
                    continue
                rim_ids = np.asarray(rim_ids, dtype=np.int64)
                dist = np.linalg.norm(
                    original_rim_vertices[rim_ids] - frame_vertices[soft_idx[local]], axis=1
                )
                weight = BaseDeformer.apply_falloff(
                    np.clip(1.0 - dist / _FRAME_FALLOFF_RADIUS_MM, 0.0, 1.0)
                )
                total = float(weight.sum())
                if total <= 1e-9:
                    continue
                frame_vertices[soft_idx[local]] += (
                    (deltas[rim_ids] * weight[:, None]).sum(axis=0) / total * _FRAME_FALLOFF_GAIN
                )

        frame_mesh.vertices = frame_vertices

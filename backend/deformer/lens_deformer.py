"""Fit lens meshes to the actual inner aperture of their corresponding rims.

The input GLBs contain curved front/back lens surfaces and curved rim walls. Selecting
vertices at exactly the global maximum/minimum Z is therefore not a reliable way to
identify those surfaces: it can select only one vertex. This implementation instead
uses the projected triangle footprint to recover the rim's inner opening and performs
a radial 2-D map on *all* lens vertices. X/Y positions change; Z coordinates, faces,
UVs, and vertex order remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely import area as shapely_area, linearrings, polygons, union_all
from shapely.errors import GEOSException

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext


@dataclass(frozen=True)
class LensSelection:
    """Lens-local meshes and source geometry for one side."""

    side: str
    lens_part: str
    rim_part: str
    lens_before: np.ndarray
    rim_before: np.ndarray


class LensDeformer(BaseDeformer):
    """Fit each lens into its rim's measured inner aperture.

    ``rim_inset`` is the physical clearance between the lens and the rim's *inner*
    aperture, in mesh coordinate units (the prepared templates use millimetres). A
    small clearance is important because the target boundary is now the real aperture,
    rather than the outside of the rim as in the previous implementation.
    """

    stage_name = "lens_deformation"

    def __init__(self, rim_inset: float = 0.35):
        self.rim_inset = float(max(0.15, rim_inset))

    def apply(self, context: DeformationContext) -> DeformationContext:
        results: list[dict[str, Any]] = []
        for side in ("left", "right"):
            selection = self._collect_lenses(context, side)
            rim_boundary = self._extract_rim_boundary(context, selection)
            target_boundary = self._generate_target_boundary(context, selection, rim_boundary)
            self._fit_lens_to_rim(context, selection, target_boundary)
            results.append(self._validate(context, selection, target_boundary))
        return self.update_context(context, applied=True, sides=results, rim_inset_mm=self.rim_inset)

    def _collect_lenses(self, context: DeformationContext, side: str) -> LensSelection:
        lens_parts = context.descriptor.vertex_groups.get(f"{side}_lens", [])
        rim_parts = context.descriptor.vertex_groups.get(f"{side}_rim", [])
        if not lens_parts or not rim_parts:
            raise ValueError(
                f"Cannot fit {side} lens: the template descriptor must map both "
                f"'{side}_lens' and '{side}_rim' to separate mesh parts."
            )
        lens_part, rim_part = lens_parts[0], rim_parts[0]
        return LensSelection(
            side=side,
            lens_part=lens_part,
            rim_part=rim_part,
            lens_before=context.mesh(lens_part).vertices.copy(),
            rim_before=context.mesh(rim_part).vertices.copy(),
        )

    def _extract_rim_boundary(self, context: DeformationContext, selection: LensSelection) -> np.ndarray:
        """Return the largest plausible hole in the rim's 2-D projected footprint."""
        rim_mesh = context.mesh(selection.rim_part)
        lens_mesh = context.mesh(selection.lens_part)
        rim_footprint = self._projected_footprint(rim_mesh, selection.rim_part)
        lens_footprint = self._projected_footprint(lens_mesh, selection.lens_part)
        lens_center = np.asarray(lens_footprint.centroid.coords[0], dtype=np.float64)
        rim_span = np.ptp(np.asarray(rim_mesh.vertices)[:, :2], axis=0)
        max_center_distance = max(5.0, 0.35 * float(np.linalg.norm(rim_span)))
        min_hole_area = max(1.0, 0.15 * float(lens_footprint.area))

        candidates: list[Polygon] = []
        for component in self._polygon_components(rim_footprint):
            for ring in component.interiors:
                aperture = Polygon(ring)
                if aperture.is_empty or not np.isfinite(aperture.area) or aperture.area < min_hole_area:
                    continue
                min_x, min_y, max_x, max_y = aperture.bounds
                hole_w, hole_h = max_x - min_x, max_y - min_y
                # Ignore small rivet/triangulation holes; a lens aperture should occupy
                # a substantial proportion of this rim, while remain smaller than it.
                if rim_span[0] <= 0 or rim_span[1] <= 0:
                    continue
                if not (0.25 * rim_span[0] <= hole_w < 0.995 * rim_span[0]):
                    continue
                if not (0.25 * rim_span[1] <= hole_h < 0.995 * rim_span[1]):
                    continue
                hole_center = np.asarray(aperture.centroid.coords[0], dtype=np.float64)
                if float(np.linalg.norm(hole_center - lens_center)) > max_center_distance:
                    continue
                candidates.append(aperture)

        if not candidates:
            raise ValueError(
                f"Cannot identify the inner aperture of '{selection.rim_part}'. "
                "The projected rim must contain a distinct, measurable opening; "
                "check the template GLB and its component descriptor."
            )
        aperture = max(candidates, key=lambda p: p.area)
        return np.asarray(aperture.exterior.coords[:-1], dtype=np.float64)

    def _generate_target_boundary(
        self,
        context: DeformationContext,
        selection: LensSelection,
        rim_boundary: np.ndarray,
    ) -> np.ndarray:
        """Inset the actual opening contour by the configured physical clearance."""
        aperture = Polygon(np.asarray(rim_boundary, dtype=np.float64)).buffer(0)
        if aperture.is_empty or not aperture.is_valid or aperture.area <= 1e-6:
            raise ValueError(f"The inner aperture for '{selection.side}' is degenerate or invalid.")
        inset = aperture.buffer(-self.rim_inset, join_style="round")
        pieces = [p for p in self._polygon_components(inset) if not p.is_empty and p.area > 1e-6]
        if not pieces:
            raise ValueError(
                f"The {self.rim_inset:.2f} mm lens clearance collapses the aperture for "
                f"'{selection.side}'. Reduce the clearance or repair the rim geometry."
            )
        target = max(pieces, key=lambda p: p.area)
        points = np.asarray(target.exterior.coords[:-1], dtype=np.float64)
        if len(points) < 8:
            raise ValueError(f"The target aperture for '{selection.side}' has too few contour points.")
        return points

    def _fit_lens_to_rim(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> None:
        """Radially map every lens vertex from its original outline to the aperture.

        Applying one angle-dependent scale to all vertices makes the curved front,
        curved back and side wall follow the same 2-D map. Vertex/face topology, UVs,
        and every Z value are left intact.
        """
        lens_mesh = context.mesh(selection.lens_part)
        source_footprint = self._projected_footprint(lens_mesh, selection.lens_part)
        source_boundary = np.asarray(source_footprint.exterior.coords[:-1], dtype=np.float64)
        target_polygon = Polygon(np.asarray(target_boundary, dtype=np.float64)).buffer(0)
        if target_polygon.is_empty or target_polygon.area <= 1e-6:
            raise ValueError(f"The target aperture for '{selection.side}' is invalid.")
        target_points = np.asarray(target_polygon.exterior.coords[:-1], dtype=np.float64)

        source_center = np.asarray(source_footprint.centroid.coords[0], dtype=np.float64)
        if not source_footprint.buffer(1e-8).covers(Point(*source_center)):
            source_center = np.asarray(source_footprint.representative_point().coords[0], dtype=np.float64)
        target_center = np.asarray(target_polygon.centroid.coords[0], dtype=np.float64)
        vertices = np.asarray(lens_mesh.vertices, dtype=np.float64).copy()
        relative = vertices[:, :2] - source_center
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        source_radius = np.linalg.norm(relative, axis=1)
        source_boundary_radius = self._boundary_radii(source_boundary, source_center, angles)
        target_boundary_radius = self._boundary_radii(target_points, target_center, angles)

        scale = target_boundary_radius / np.maximum(source_boundary_radius, 1e-6)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"Could not derive a finite radial mapping for '{selection.side}' lens.")
        # Avoid catastrophic transformations if a malformed outline produces a nearly
        # zero source radius. Do not silently clamp; fail so the bad asset can be fixed.
        if np.any(scale < 0.15) or np.any(scale > 3.0):
            raise ValueError(
                f"The {selection.side} lens-to-aperture scale is implausible "
                f"(range {float(scale.min()):.3f}-{float(scale.max()):.3f}); "
                "check the lens and rim component geometry."
            )

        vertices[:, :2] = target_center + relative * scale[:, None]
        # Z is deliberately untouched: thickness and curvature remain exactly as-authored.
        lens_mesh.vertices = vertices

    def _validate(
        self,
        context: DeformationContext,
        selection: LensSelection,
        target_boundary: np.ndarray,
    ) -> dict[str, Any]:
        lens_mesh = context.mesh(selection.lens_part)
        lens_footprint = self._projected_footprint(lens_mesh, selection.lens_part)
        target_polygon = Polygon(np.asarray(target_boundary, dtype=np.float64)).buffer(0)
        if target_polygon.is_empty or target_polygon.area <= 1e-6:
            raise ValueError(f"Cannot validate {selection.side} lens against an invalid rim aperture.")

        tolerance = 0.05
        outside = lens_footprint.difference(target_polygon.buffer(tolerance))
        outside_area = float(sum(p.area for p in self._polygon_components(outside)))
        inside_rim = outside_area <= max(0.02, target_polygon.area * 1e-5)

        lens_boundary = np.asarray(lens_footprint.exterior.coords[:-1], dtype=np.float64)
        target_center = np.asarray(target_polygon.centroid.coords[0], dtype=np.float64)
        radial = lens_boundary - target_center
        lens_angles = np.arctan2(radial[:, 1], radial[:, 0])
        lens_radii = np.linalg.norm(radial, axis=1)
        target_radii = self._boundary_radii(
            np.asarray(target_polygon.exterior.coords[:-1], dtype=np.float64), target_center, lens_angles
        )
        fit_error = float(np.mean(np.abs(lens_radii - target_radii)))
        vertices_before = np.asarray(selection.lens_before)
        vertices_after = np.asarray(lens_mesh.vertices)
        thickness_before = float(np.ptp(vertices_before[:, 2]))
        thickness_after = float(np.ptp(vertices_after[:, 2]))
        uv = getattr(lens_mesh.visual, "uv", None)
        uv_count = int(len(uv)) if uv is not None else 0
        actual_width = float(np.ptp(vertices_after[:, 0]))
        actual_height = float(np.ptp(vertices_after[:, 1]))
        requested_width = float(getattr(context.measurements, "lens_width", 0.0) or 0.0)
        requested_height = float(getattr(context.measurements, "lens_height", 0.0) or 0.0)
        width_error_ratio = (
            abs(actual_width - requested_width) / requested_width if requested_width > 0 else 0.0
        )
        height_error_ratio = (
            abs(actual_height - requested_height) / requested_height if requested_height > 0 else 0.0
        )
        dimensions_match = width_error_ratio <= 0.12 and height_error_ratio <= 0.12
        dimension_warning = None
        if not dimensions_match:
            dimension_warning = (
                f"{selection.side.title()}Lens fits inside its rim aperture, but its resulting size "
                f"is {actual_width:.1f}×{actual_height:.1f} mm versus requested "
                f"{requested_width:.1f}×{requested_height:.1f} mm. The template aperture or input "
                "measurements need review; fitting must not silently override physical dimensions."
            )
        valid = bool(
            inside_rim
            and fit_error <= 0.75
            and thickness_after > 0.0
            and np.isfinite(vertices_after).all()
            and len(vertices_after) == len(vertices_before)
        )
        result = {
            "side": selection.side,
            "vertex_count": int(len(vertices_after)),
            "uv_count": uv_count,
            "fit_error_mm": round(fit_error, 6),
            "thickness_mm": round(thickness_after, 6),
            "thickness_delta_mm": round(thickness_after - thickness_before, 9),
            "curvature_delta": round(
                float(np.std(vertices_after[:, 2]) - np.std(vertices_before[:, 2])), 9
            ),
            "inside_rim": inside_rim,
            "outside_area_mm2": round(outside_area, 8),
            "rim_clearance_mm": self.rim_inset,
            "actual_width_mm": round(actual_width, 4),
            "actual_height_mm": round(actual_height, 4),
            "requested_width_mm": round(requested_width, 4) if requested_width else None,
            "requested_height_mm": round(requested_height, 4) if requested_height else None,
            "width_error_ratio": round(width_error_ratio, 6),
            "height_error_ratio": round(height_error_ratio, 6),
            "matches_requested_dimensions": dimensions_match,
            "dimension_warning": dimension_warning,
            "valid": valid,
        }
        return self.validate_constraints(result)

    @staticmethod
    def _boundary_radii(
        boundary: np.ndarray, center: np.ndarray, query_angles: np.ndarray
    ) -> np.ndarray:
        """Interpolate radial boundary distance by angle, including the +/-pi seam."""
        points = np.asarray(boundary, dtype=np.float64)[:, :2]
        relative = points - np.asarray(center, dtype=np.float64)
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        radii = np.linalg.norm(relative, axis=1)
        finite = np.isfinite(angles) & np.isfinite(radii)
        angles, radii = angles[finite], radii[finite]
        if len(angles) < 3:
            raise ValueError("A projected boundary needs at least three finite points.")
        order = np.argsort(angles)
        angles, radii = angles[order], radii[order]
        unique_angles, inverse = np.unique(angles, return_inverse=True)
        unique_radii = np.zeros(len(unique_angles), dtype=np.float64)
        np.maximum.at(unique_radii, inverse, radii)
        if len(unique_angles) < 3:
            raise ValueError("A projected boundary does not span enough directions.")
        extended_angles = np.concatenate(([unique_angles[-1] - 2.0 * np.pi], unique_angles,
                                          [unique_angles[0] + 2.0 * np.pi]))
        extended_radii = np.concatenate(([unique_radii[-1]], unique_radii, [unique_radii[0]]))
        return np.interp(np.asarray(query_angles, dtype=np.float64), extended_angles, extended_radii)

    @staticmethod
    def _projected_footprint(mesh: trimesh.Trimesh, part_name: str) -> Polygon | MultiPolygon:
        """Union valid XY projections of mesh triangles into a measurable footprint."""
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"Cannot project '{part_name}': its vertices are malformed or non-finite.")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError(f"Cannot project '{part_name}': it has no valid triangle faces.")
        triangles = vertices[faces][:, :, :2]
        cross = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1]) - (
            triangles[:, 1, 1] - triangles[:, 0, 1]
        ) * (triangles[:, 2, 0] - triangles[:, 0, 0])
        usable = np.isfinite(cross) & (np.abs(cross) > 2e-7)
        if not np.any(usable):
            raise ValueError(f"Cannot project '{part_name}': its triangles have no usable XY area.")
        projected = triangles[usable]
        # Shapely 2.x vectorized constructors keep this practical for dense real assets.
        rings = linearrings(projected)
        triangles_2d = polygons(rings)
        projected_area = shapely_area(triangles_2d)
        triangles_2d = triangles_2d[projected_area > 1e-7]
        footprint = LensDeformer._union_triangles(triangles_2d, part_name)
        if footprint.is_empty:
            raise ValueError(f"Cannot project '{part_name}': its projected footprint is empty.")
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        components = LensDeformer._polygon_components(footprint)
        if not components:
            raise ValueError(f"Cannot project '{part_name}': no valid polygonal footprint was found.")
        return max(components, key=lambda p: p.area)

    # Exact union first. Real meshes that have been reshaped can contain near-coincident edges
    # that make GEOS throw "TopologyException: Ring edge missing"; snapping every coordinate to a
    # tiny grid fixes that. 1e-6 mm gave an identical area on the failing real mesh, so it is tried
    # before the coarser grids.
    _UNION_GRID_SIZES_MM = (None, 1e-6, 1e-4, 1e-3)

    @staticmethod
    def _union_triangles(triangles_2d: Any, part_name: str) -> Any:
        last_error: Exception | None = None
        for grid in LensDeformer._UNION_GRID_SIZES_MM:
            try:
                return union_all(triangles_2d) if grid is None else union_all(triangles_2d, grid_size=grid)
            except GEOSException as exc:
                last_error = exc
        raise ValueError(
            f"Cannot project '{part_name}': its triangles could not be merged into a footprint "
            f"even after snapping to a {LensDeformer._UNION_GRID_SIZES_MM[-1]} mm grid ({last_error})."
        ) from last_error

    @staticmethod
    def _polygon_components(geometry: Any) -> list[Polygon]:
        if geometry is None or geometry.is_empty:
            return []
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, (MultiPolygon, GeometryCollection)):
            result: list[Polygon] = []
            for part in geometry.geoms:
                result.extend(LensDeformer._polygon_components(part))
            return result
        return []

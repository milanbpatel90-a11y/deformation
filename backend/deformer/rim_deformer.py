"""Rim opening deformation: resize each rim's inner aperture to the requested lens.

Real rim meshes are curved. The previous implementation picked "front" and "back" rim
vertices as those at *exactly* the global max/min Z, which selects a single vertex on the
prepared templates (1 of 15,554 on geometric_metal), so the requested lens size never reached
the rims. This stage works from the projected triangle footprint instead:

1. Recover the rim's inner aperture (the opening the lens sits in) and its outer outline.
2. Build a target aperture: the requested lens width x height (plus the lens clearance the
   LensDeformer will leave), whose *shape* is the template aperture blended toward the photo
   contour by ``contour_strength``. With no usable contour the template shape is kept.
3. Keep the rim's OUTER outline fixed, so the bridge, frame and hinges stay attached, and
   reshape only the opening. The nasal edge of the opening and its vertical centre are
   pinned; growth goes temporally/vertically and is limited by a minimum wall thickness.
   When the request does not fit, the opening is clamped and the stage says so.
4. Move every rim vertex with one radial 2-D map (XY only): the opening boundary goes to the
   target boundary, the outer outline stays put, and the wall in between is blended.

Z coordinates, faces, UVs and vertex order are never changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.affinity import scale as affine_scale
from shapely.affinity import translate
from shapely.geometry import Polygon

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.deformer.lens_deformer import LensDeformer
from backend.models import LensContour

_RAY_COUNT = 720
_OUTLINE_MARGIN_MM = 0.05          # a mapped opening never reaches the outer outline itself
_MIN_UNIFORM_SCALE = 0.2           # below this the request is hopeless for the template
_FILL_STEP = 0.01                  # width/height are raised in 1 % steps after the uniform fit ...
_FILL_STEPS = 120                  # ... at most this many times (covers 0.2 -> 1.0 with headroom)
_STRENGTH_STEP = 0.1               # photo-contour blend is reduced in these steps until the size fits
_OUTLINE_PIN_MM = 0.5              # vertices this close to the outer outline are held (ramped) in place


class RimDeformer(BaseDeformer):
    """Resize each rim's inner opening to the requested lens, keeping its outer outline."""

    stage_name = "rim_deformation"

    def __init__(
        self,
        contour_strength: float = 0.7,
        lens_clearance_mm: float = 0.35,
        min_wall_mm: float = 1.0,
    ):
        # contour_strength: 0 keeps the template opening shape, 1 follows the photo contour.
        self.contour_strength = float(np.clip(contour_strength, 0.0, 1.0))
        # Must match LensDeformer.rim_inset: the lens ends up this far inside the opening.
        self.lens_clearance_mm = float(max(0.0, lens_clearance_mm))
        self.min_wall_mm = float(max(0.2, min_wall_mm))

    # ------------------------------------------------------------------ public API
    def apply(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> DeformationContext:
        contour = lens_contour or LensContour()
        results = [self._deform_side(context, side, contour) for side in ("left", "right")]
        return self.update_context(
            context,
            applied=bool(results),
            sides=results,
            contour_strength=self.contour_strength,
            lens_clearance_mm=self.lens_clearance_mm,
            min_wall_mm=self.min_wall_mm,
        )

    # ------------------------------------------------------------------ one rim
    def _deform_side(self, context: DeformationContext, side: str, contour: LensContour) -> dict[str, Any]:
        helper = LensDeformer()
        selection = helper._collect_lenses(context, side)   # raises a clear error if parts are unmapped
        rim_mesh = context.mesh(selection.rim_part)
        vertices = np.asarray(rim_mesh.vertices, dtype=np.float64).copy()

        footprint = LensDeformer._projected_footprint(rim_mesh, selection.rim_part)
        outer = Polygon(footprint.exterior).buffer(0)
        aperture = self._largest(Polygon(helper._extract_rim_boundary(context, selection)).buffer(0))
        outer = self._largest(outer)

        nasal_sign = -1.0 if side == "left" else 1.0
        if float(np.sign(aperture.centroid.x)) != nasal_sign:
            raise ValueError(
                f"The {side} rim opening is centred at x={aperture.centroid.x:.1f}; the {side} side is expected "
                f"at {'-' if side == 'left' else '+'}X. Check the template's axis convention (Y up, left = -X)."
            )

        requested_lens = (float(context.measurements.lens_width), float(context.measurements.lens_height))
        if not all(np.isfinite(requested_lens)) or min(requested_lens) <= 0:
            raise ValueError(f"Cannot size the {side} rim: lens size {requested_lens} is not positive.")
        want_w = requested_lens[0] + 2.0 * self.lens_clearance_mm
        want_h = requested_lens[1] + 2.0 * self.lens_clearance_mm

        minx, miny, maxx, maxy = aperture.bounds
        pin_x = maxx if side == "left" else minx            # nasal edge stays where the bridge expects it
        centre_y = 0.5 * (miny + maxy)
        allowed = outer.buffer(-self.min_wall_mm).union(aperture)   # room that already exists + wall-safe room
        tolerance = max(0.25, 0.001 * want_w * want_h)              # mm^2 of allowed overshoot (discretisation)

        def place(shape: Polygon) -> Polygon:
            sminx, sminy, smaxx, smaxy = shape.bounds
            return translate(
                shape,
                xoff=pin_x - (smaxx if side == "left" else sminx),
                yoff=centre_y - 0.5 * (sminy + smaxy),
            )

        # Physical size beats photo shape. A photo outline that is "fuller" in the corners than the
        # template opening cannot be placed at the requested size, and clamping it would cost lens
        # height. So use the photo shape only as strongly as the requested size still fits, falling
        # back to the template's own opening shape; only if even that does not fit is the size clamped.
        effective_strength = self.contour_strength
        while True:
            shape_polygon, shape_source = self._target_shape(
                aperture, contour, side, context, want_w, want_h, effective_strength
            )
            placed = place(shape_polygon)
            if placed.difference(allowed).area <= tolerance or effective_strength <= 0.0:
                break
            effective_strength = max(0.0, round(effective_strength - _STRENGTH_STEP, 6))
        target, scale_x, scale_y = self._clamp_to_room(placed, allowed, (pin_x, centre_y))

        moved, max_shift = self._map_vertices(vertices, rim_mesh, aperture, target, outer)

        t_w = target.bounds[2] - target.bounds[0]
        t_h = target.bounds[3] - target.bounds[1]
        clamped = scale_x < 0.999 or scale_y < 0.999
        warning = None
        if clamped:
            warning = (
                f"{side.title()}Rim opening limited to {t_w:.1f}×{t_h:.1f} mm (wanted {want_w:.1f}×{want_h:.1f} mm): "
                f"the template's outer outline has no more room at the {self.min_wall_mm:.1f} mm minimum wall. "
                "Use a larger template or scale the front."
            )
        return {
            "side": side,
            "template_aperture_mm": [round(aperture.bounds[2] - aperture.bounds[0], 3), round(aperture.bounds[3] - aperture.bounds[1], 3)],
            "requested_lens_mm": [round(requested_lens[0], 3), round(requested_lens[1], 3)],
            "wanted_aperture_mm": [round(want_w, 3), round(want_h, 3)],
            "achieved_aperture_mm": [round(t_w, 3), round(t_h, 3)],
            "scale_x": round(scale_x, 4),
            "scale_y": round(scale_y, 4),
            "clamped_by_wall": clamped,
            "min_wall_mm": self.min_wall_mm,
            "pinned_nasal_x": round(pin_x, 4),
            "vertical_centre": round(centre_y, 4),
            "shape_source": shape_source,
            "contour_strength_requested": round(self.contour_strength, 3),
            "contour_strength_used": round(effective_strength, 3),
            "moved_vertices": moved,
            "max_displacement_mm": round(max_shift, 4),
            "warning": warning,
            "valid": bool(np.isfinite(np.asarray(rim_mesh.vertices)).all()),
        }

    # ------------------------------------------------------------------ target shape
    def _target_shape(
        self,
        aperture: Polygon,
        contour: LensContour,
        side: str,
        context: DeformationContext,
        want_w: float,
        want_h: float,
        strength: float | None = None,
    ) -> tuple[Polygon, str]:
        """Template opening shape blended toward the photo contour, scaled to exactly want_w x want_h."""
        strength = self.contour_strength if strength is None else float(np.clip(strength, 0.0, 1.0))
        angles = self._angles()
        template_unit = self._to_unit(aperture)
        radii = self._ray_radii(template_unit, (0.0, 0.0), angles)
        source = "template"

        contour_unit = self._contour_unit(contour, side, context)
        if contour_unit is not None and strength > 0.0:
            contour_radii = self._ray_radii(contour_unit, (0.0, 0.0), angles)
            radii = (1.0 - strength) * radii + strength * contour_radii
            source = "template+photo contour"

        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = (points - 0.5 * (points.min(axis=0) + points.max(axis=0))) / np.ptp(points, axis=0)
        polygon = Polygon(points * np.array([want_w, want_h])).buffer(0)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Could not build a valid target opening for the rim.")
        return self._largest(polygon), source

    def _contour_unit(self, contour: LensContour, side: str, context: DeformationContext) -> Polygon | None:
        raw = contour.left if side == "left" else contour.right
        mm = self._contour_points_to_mm(raw, side, context)
        if mm is None:
            return None
        polygon = Polygon(mm).buffer(0)
        if polygon.is_empty or not polygon.is_valid or polygon.area < 1e-6:
            return None
        unit = self._to_unit(self._largest(polygon))
        # Rays are cast from the bounding-box centre, so it must lie inside the contour.
        return unit if unit.contains(shapely.Point(0.0, 0.0)) else None

    @staticmethod
    def _to_unit(polygon: Polygon) -> Polygon:
        """Centre the polygon's bounding box on the origin and scale it to unit width/height."""
        minx, miny, maxx, maxy = polygon.bounds
        coords = (np.asarray(polygon.exterior.coords) - [0.5 * (minx + maxx), 0.5 * (miny + maxy)]) / [maxx - minx, maxy - miny]
        return Polygon(coords)

    # ------------------------------------------------------------------ fitting to the available room
    def _clamp_to_room(
        self,
        placed: Polygon,
        allowed: Any,
        anchor: tuple[float, float],
    ) -> tuple[Polygon, float, float]:
        """Largest opening <= the wanted one that fits the allowed room, pinned at the anchor point.

        Width and height are limited separately so a template that is only too short does not also
        lose width. Start from the largest uniform scale that fits, then repeatedly raise whichever
        dimension is further from the request while the opening still fits (max-min fairness), so
        neither dimension uses up the slack the other one needs.
        """
        tolerance = max(0.25, 0.001 * placed.area)   # mm^2 of allowed overshoot (polygon discretisation)

        def shaped(fx: float, fy: float) -> Polygon:
            return affine_scale(placed, xfact=fx, yfact=fy, origin=anchor)

        def fits(fx: float, fy: float) -> bool:
            return shaped(fx, fy).difference(allowed).area <= tolerance

        if fits(1.0, 1.0):
            return shaped(1.0, 1.0), 1.0, 1.0

        lo, hi = _MIN_UNIFORM_SCALE, 1.0
        for _ in range(20):                      # largest uniform scale that fits
            mid = 0.5 * (lo + hi)
            if fits(mid, mid):
                lo = mid
            else:
                hi = mid
        fx = fy = lo
        for _ in range(_FILL_STEPS):
            progressed = False
            for dim in sorted(("x", "y"), key=lambda d: fx if d == "x" else fy):
                nx = min(1.0, fx + _FILL_STEP) if dim == "x" else fx
                ny = min(1.0, fy + _FILL_STEP) if dim == "y" else fy
                if (nx, ny) != (fx, fy) and fits(nx, ny):
                    fx, fy = nx, ny
                    progressed = True
                    break
            if not progressed:
                break
        return shaped(fx, fy), fx, fy

    # ------------------------------------------------------------------ the vertex map
    def _map_vertices(
        self,
        vertices: np.ndarray,
        rim_mesh,
        aperture: Polygon,
        target: Polygon,
        outer: Polygon,
    ) -> tuple[int, float]:
        """Radially map rim XY so the template opening becomes the target and the outline stays."""
        common = aperture.intersection(target)
        centre_point = common.centroid if not common.is_empty else aperture.centroid
        centre = np.array([centre_point.x, centre_point.y], dtype=np.float64)

        angles = self._angles()
        r_open = self._ray_radii(aperture, centre, angles)
        r_target = self._ray_radii(target, centre, angles)
        r_outer = np.maximum(self._ray_radii(outer, centre, angles), r_open + 1e-6)
        r_target = np.minimum(r_target, r_outer - _OUTLINE_MARGIN_MM)

        rel = vertices[:, :2] - centre
        radius = np.hypot(rel[:, 0], rel[:, 1])
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        open_v = np.interp(theta, angles, r_open, period=2.0 * np.pi)
        target_v = np.interp(theta, angles, r_target, period=2.0 * np.pi)
        outer_v = np.interp(theta, angles, r_outer, period=2.0 * np.pi)

        span = np.maximum(outer_v - open_v, 1e-3)
        inside = radius <= open_v
        wall = target_v + (radius - open_v) * (outer_v - target_v) / span
        new_radius = np.where(inside, radius * target_v / np.maximum(open_v, 1e-6), np.where(radius < outer_v, wall, radius))

        # The ray model samples the outline at _RAY_COUNT angles, which is only approximate across
        # concave notches (e.g. a hinge lug). Hold vertices that actually lie on / near the outer
        # outline in place, so the bridge, frame and hinges stay attached exactly.
        to_outline = shapely.distance(shapely.points(vertices[:, :2]), outer.exterior)
        keep = np.clip(to_outline / _OUTLINE_PIN_MM, 0.0, 1.0)
        new_radius = radius + (new_radius - radius) * keep

        unit = np.divide(rel, radius[:, None], out=np.zeros_like(rel), where=radius[:, None] > 1e-12)
        updated = vertices.copy()
        updated[:, :2] = centre + unit * new_radius[:, None]
        if not np.isfinite(updated).all():
            raise ValueError("Rim opening map produced non-finite vertices; check the rim geometry.")
        shift = np.linalg.norm(updated[:, :2] - vertices[:, :2], axis=1)
        rim_mesh.vertices = updated          # Z, faces and UVs are untouched
        return int(np.count_nonzero(shift > 1e-6)), float(shift.max())

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _angles() -> np.ndarray:
        return np.linspace(-np.pi, np.pi, _RAY_COUNT, endpoint=False)

    @staticmethod
    def _largest(geometry: Any) -> Polygon:
        parts = LensDeformer._polygon_components(geometry)
        if not parts:
            raise ValueError("Expected a polygon but the geometry is empty.")
        return max(parts, key=lambda p: p.area)

    @staticmethod
    def _ray_radii(polygon: Polygon, centre: Any, angles: np.ndarray) -> np.ndarray:
        """Distance from ``centre`` to the farthest boundary crossing along each ray."""
        centre = np.asarray(centre, dtype=np.float64)
        minx, miny, maxx, maxy = polygon.bounds
        reach = 2.0 * ((max(maxx, centre[0]) - min(minx, centre[0])) + (max(maxy, centre[1]) - min(miny, centre[1]))) + 1.0
        ends = centre + np.column_stack([np.cos(angles), np.sin(angles)]) * reach
        starts = np.broadcast_to(centre, ends.shape)
        rays = shapely.linestrings(np.stack([starts, ends], axis=1))
        hits = shapely.intersection(rays, polygon.exterior)
        coords, index = shapely.get_coordinates(hits, return_index=True)
        radii = np.full(len(angles), -np.inf)
        if len(coords):
            np.maximum.at(radii, index, np.hypot(coords[:, 0] - centre[0], coords[:, 1] - centre[1]))
        missing = ~np.isfinite(radii)
        if missing.all():
            raise ValueError("A shape boundary could not be measured from its centre.")
        if missing.any():
            radii[missing] = np.interp(angles[missing], angles[~missing], radii[~missing], period=2.0 * np.pi)
        return radii

    # ------------------------------------------------------------------ photo contour -> mm
    def _contour_points_to_mm(
        self,
        points: list[list[float]],
        side: str,
        context: DeformationContext,
    ) -> np.ndarray | None:
        if len(points) < 4:
            return None

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            return None

        if side == "left":
            local_x = arr[:, 0] / 0.5
        else:
            local_x = (arr[:, 0] - 0.5) / 0.5
        local_y = arr[:, 1]

        local_x = np.clip(local_x, 0.0, 1.0)
        local_y = np.clip(local_y, 0.0, 1.0)

        measured = context.measurements
        center_x = (-1.0 if side == "left" else 1.0) * (measured.bridge_width * 0.5 + measured.lens_width * 0.5)
        x_mm = center_x + (local_x - 0.5) * measured.lens_width
        y_mm = (0.5 - local_y) * measured.lens_height

        contour = np.column_stack([x_mm, y_mm])
        return self._normalize_contour_orientation(contour, side)

    @staticmethod
    def _normalize_contour_orientation(contour: np.ndarray, side: str) -> np.ndarray:
        center = contour.mean(axis=0)
        angles = np.arctan2(contour[:, 1] - center[1], contour[:, 0] - center[0])
        order = np.argsort(-angles)
        ordered = contour[order]
        start = np.argmin(ordered[:, 0] if side == "left" else -ordered[:, 0])
        return np.roll(ordered, -start, axis=0)

"""Topology-aware rim deformation driven by lens contours and measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.models import LensContour


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
    """Reshape left and right rim loops while preserving topology and thickness."""

    stage_name = "rim_deformation"

    def __init__(self, contour_strength: float = 0.7):
        self.contour_strength = float(np.clip(contour_strength, 0.0, 1.0))

    def apply(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> DeformationContext:
        contour = lens_contour or LensContour()
        results: list[RimSideResult] = []

        for side in ("left", "right"):
            rim_group = f"{side}_rim"
            rim_parts = context.descriptor.vertex_groups.get(rim_group, [])
            if not rim_parts:
                continue

            rim_mesh = context.mesh(rim_parts[0])
            frame_meshes = context.vertex_group_meshes("frame")
            if not frame_meshes:
                continue
            frame_mesh = frame_meshes[0]

            before = rim_mesh.vertices.copy()
            target_contour = self._target_contour(context, side, contour)
            self._deform_rim_mesh(rim_mesh, side, target_contour, context.measurements.rim_thickness)
            self._propagate_to_frame(frame_mesh, before, rim_mesh.vertices)

            bounds_before = np.ptp(before[:, :2], axis=0)
            bounds_after = np.ptp(rim_mesh.vertices[:, :2], axis=0)
            results.append(
                RimSideResult(
                    side=side,
                    width_before=round(float(bounds_before[0]), 3),
                    width_after=round(float(bounds_after[0]), 3),
                    height_before=round(float(bounds_before[1]), 3),
                    height_after=round(float(bounds_after[1]), 3),
                    contour_points=len(target_contour),
                )
            )

        return self.update_context(
            context,
            applied=bool(results),
            sides=[result.__dict__ for result in results],
            contour_strength=self.contour_strength,
        )

    def _deform_rim_mesh(
        self,
        rim_mesh,
        side: str,
        target_contour: np.ndarray,
        target_thickness: float,
    ) -> None:
        vertices = rim_mesh.vertices.copy()
        front_indices, back_indices = self._split_surface_indices(vertices)
        if len(front_indices) == 0 or len(back_indices) == 0:
            return

        front_vertices = vertices[front_indices]
        back_vertices = vertices[back_indices]

        front_order = self._loop_order(front_vertices[:, :2])
        back_order = self._loop_order(back_vertices[:, :2])
        ordered_front_idx = front_indices[front_order]
        ordered_back_idx = back_indices[back_order]

        template_front = vertices[ordered_front_idx][:, :2]
        resampled_target = self._resample_contour(target_contour, len(ordered_front_idx))
        resampled_target = self._align_contour_to_template(template_front, resampled_target)
        blended_target = self._blend_template_and_target(template_front, resampled_target)

        front_centroid = template_front.mean(axis=0)
        target_centroid = blended_target.mean(axis=0)
        front_radial = np.linalg.norm(template_front - front_centroid, axis=1)
        target_radial = np.linalg.norm(blended_target - target_centroid, axis=1)
        radial_ratio = np.divide(target_radial, np.maximum(front_radial, 1e-6))
        # Clamp: a front-ring vertex that happens to sit near its own centroid (a few mm
        # out -- common on a rim built from concatenated/fused scan fragments, where loop
        # vertices aren't evenly spaced around the centroid) divides into a normal-sized
        # target radius and produces a large outlier ratio (confirmed on cat_eye_gold: a
        # 2.2mm front_radial against an 18-40mm target_radial gave an 11.7x ratio). That
        # ratio is then applied to the BACK ring's own, independently-scaled radii, so one
        # outlier blows the entire back ring up to hundreds of mm. No legitimate single-point
        # rim reshape needs more than a few times local radial growth/shrink, so clip it.
        radial_ratio = np.clip(radial_ratio, 0.3, 3.0)

        vertices[ordered_front_idx, 0:2] = blended_target
        thickness = self._target_depth(vertices, ordered_front_idx, ordered_back_idx, target_thickness)

        back_template = vertices[ordered_back_idx][:, :2]
        back_centroid = back_template.mean(axis=0)
        centered_back = back_template - back_centroid

        # radial_ratio is parameterized over the front ring's own vertex count, which does
        # not generally equal the back ring's count (true for any non-box-extruded rim).
        # It must be resampled onto the back ring some other way. Loop-fraction (index/N)
        # interpolation assumes index 0 on both rings sits at the same physical azimuth and
        # that vertex density is otherwise comparable between them -- true for a clean
        # symmetric box extrusion, false for a rim assembled from concatenated/fused scan
        # fragments where the two loops can have very different vertex distributions.
        # Confirmed on cat_eye_gold: loop-fraction interpolation put ratios on the wrong
        # back-ring vertices, so a back vertex 48.9mm from its own centroid picked up a
        # ratio meant for an unrelated front vertex and the back ring blew out to ~170mm.
        # Matching by actual ANGLE around each ring's own centroid is density- and
        # rotation-invariant and keeps a ratio on the azimuth it was actually computed for.
        if len(radial_ratio) == len(centered_back):
            radial_ratio_back = radial_ratio
        else:
            front_centered = template_front - front_centroid
            front_angles = np.arctan2(front_centered[:, 1], front_centered[:, 0])
            back_angles = np.arctan2(centered_back[:, 1], centered_back[:, 0])
            order = np.argsort(front_angles)
            sorted_angles = front_angles[order]
            sorted_ratio = radial_ratio[order]
            # Extend periodically by one wrap on each side so np.interp handles wraparound.
            ext_angles = np.concatenate([sorted_angles[-1:] - 2 * np.pi, sorted_angles, sorted_angles[:1] + 2 * np.pi])
            ext_ratio = np.concatenate([sorted_ratio[-1:], sorted_ratio, sorted_ratio[:1]])
            radial_ratio_back = np.interp(back_angles, ext_angles, ext_ratio)

        scaled_back = centered_back * radial_ratio_back[:, None]
        # Recenter on (back's own native centroid + the net 2D shift the front ring
        # underwent), not on target_centroid directly. Forcing the back ring onto the
        # front ring's target centroid silently assumes the two rings were already
        # coaxial (front_centroid == back_centroid) -- true for a clean box-extruded
        # rim, but false for any rim assembled from concatenated/fused source fragments
        # (confirmed on cat_eye_gold, built from a real scan: front/back centroids sit
        # ~tens of mm apart because the front and back loops don't correspond to a
        # simple vertical extrusion). The old code discarded that native offset entirely,
        # snapping the back ring on top of the front ring's position and producing a
        # multi-hundred-mm frame_width blow-up on deformation.
        net_shift = target_centroid - front_centroid
        vertices[ordered_back_idx, 0:2] = (back_centroid + net_shift) + scaled_back

        front_z = float(np.max(vertices[ordered_front_idx, 2]))
        vertices[ordered_front_idx, 2] = front_z
        vertices[ordered_back_idx, 2] = front_z - thickness

        rim_mesh.vertices = vertices

    def _target_contour(
        self,
        context: DeformationContext,
        side: str,
        lens_contour: LensContour,
    ) -> np.ndarray:
        raw_points = lens_contour.left if side == "left" else lens_contour.right
        desired = self._contour_points_to_mm(raw_points, side, context)
        if desired is None:
            desired = self._fallback_contour(context, side)
        return desired

    def _contour_points_to_mm(
        self,
        points: list[list[float]],
        side: str,
        context: DeformationContext,
    ) -> np.ndarray | None:
        if len(points) < 4:
            return None

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
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
        center_y = 0.0
        width = measured.lens_width
        height = measured.lens_height

        x_mm = center_x + (local_x - 0.5) * width
        y_mm = center_y + (0.5 - local_y) * height

        contour = np.column_stack([x_mm, y_mm])
        return self._normalize_contour_orientation(contour, side)

    def _fallback_contour(self, context: DeformationContext, side: str) -> np.ndarray:
        plane = context.descriptor.lens_planes[side]
        width = context.measurements.lens_width
        height = context.measurements.lens_height
        cx, cy = plane.origin[0], plane.origin[1]
        base = np.array(
            [
                [-0.5, 0.5],
                [-0.72, 0.22],
                [-0.72, -0.22],
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.72, -0.22],
                [0.72, 0.22],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        contour = np.column_stack([cx + base[:, 0] * width, cy + base[:, 1] * height])
        return self._normalize_contour_orientation(contour, side)

    @staticmethod
    def _normalize_contour_orientation(contour: np.ndarray, side: str) -> np.ndarray:
        center = contour.mean(axis=0)
        angles = np.arctan2(contour[:, 1] - center[1], contour[:, 0] - center[0])
        order = np.argsort(-angles)
        ordered = contour[order]
        start = np.argmin(ordered[:, 0] if side == "left" else -ordered[:, 0])
        return np.roll(ordered, -start, axis=0)

    @staticmethod
    def _split_surface_indices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_values = vertices[:, 2]
        z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
        if np.isclose(z_min, z_max):
            # Degenerate: the whole rim sits at a single depth. Treat it as one ring used
            # for both front and back so downstream code still has something to work with.
            idx = np.arange(len(z_values))
            return idx, idx

        # Box-extruded rims (many vertices coplanar at the exact front/back faces) and
        # organic/curved rims (e.g. a thin round wire frame, where only a couple of
        # vertices sit at the exact Z extreme and the rest vary continuously) both need a
        # front/back split. Using an exact isclose-to-extreme match only works for the
        # former and silently starves the split down to 1-2 vertices for the latter,
        # which then breaks the radial-ratio broadcast in _deform_rim_mesh. A midpoint
        # threshold instead always yields two full rings, regardless of topology.
        midpoint = (z_min + z_max) / 2.0
        front_indices = np.where(z_values >= midpoint)[0]
        back_indices = np.where(z_values < midpoint)[0]
        return front_indices, back_indices

    @staticmethod
    def _loop_order(points_2d: np.ndarray) -> np.ndarray:
        center = points_2d.mean(axis=0)
        angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
        return np.argsort(-angles)

    def _resample_contour(self, contour: np.ndarray, count: int) -> np.ndarray:
        if len(contour) == count:
            return contour.copy()

        closed = np.vstack([contour, contour[0]])
        distances = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        if cumulative[-1] <= 1e-6:
            return np.repeat(contour[:1], count, axis=0)

        t = cumulative / cumulative[-1]
        k = min(3, len(contour) - 1)
        spline_x = interpolate.make_interp_spline(t, closed[:, 0], k=k)
        spline_y = interpolate.make_interp_spline(t, closed[:, 1], k=k)
        sample_t = np.linspace(0.0, 1.0, count + 1)[:-1]
        return np.column_stack([spline_x(sample_t), spline_y(sample_t)])

    def _blend_template_and_target(
        self,
        template_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        template_center = template_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        template_shape = template_points - template_center
        target_shape = target_points - target_center
        blended_shape = template_shape * (1.0 - self.contour_strength) + target_shape * self.contour_strength
        return target_center + blended_shape

    @staticmethod
    def _align_contour_to_template(template_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Find the best cyclic shift + optional reversal using FFT cross-correlation (O(n log n))."""
        n = len(template_points)
        if n == 0:
            return target_points

        best = target_points
        best_score = float("inf")

        for candidate in (target_points, target_points[::-1]):
            # Use x-coordinate cross-correlation to find best cyclic shift cheaply
            t_x = template_points[:, 0] - template_points[:, 0].mean()
            c_x = candidate[:, 0] - candidate[:, 0].mean()
            if len(t_x) != len(c_x):
                # Fallback to direct if sizes mismatch
                score = float(np.sum(np.linalg.norm(template_points - candidate, axis=1)))
                if score < best_score:
                    best_score = score
                    best = candidate
                continue

            # FFT cross-correlation on x-coords to find best shift cheaply
            corr = np.real(np.fft.ifft(np.fft.fft(t_x) * np.conj(np.fft.fft(c_x))))
            best_shift = int(np.argmax(corr))

            # Check best_shift and its neighbours (±1) for robustness
            for shift in (best_shift - 1, best_shift, best_shift + 1):
                rolled = np.roll(candidate, shift % n, axis=0)
                score = float(np.sum(np.linalg.norm(template_points - rolled, axis=1)))
                if score < best_score:
                    best_score = score
                    best = rolled

        return best

    @staticmethod
    def _target_depth(
        vertices: np.ndarray,
        front_indices: np.ndarray,
        back_indices: np.ndarray,
        target_thickness: float,
    ) -> float:
        if np.isfinite(target_thickness):
            return float(np.clip(target_thickness, 0.8, 6.0))
        # Fallback only: estimate existing thickness from the two rings' mean Z,
        # not an elementwise difference -- front/back rings are not guaranteed to
        # have equal vertex counts (true for any non-box-extruded rim mesh), so
        # vertices[front_indices, 2] - vertices[back_indices, 2] can raise a
        # broadcast ValueError.
        return float(np.mean(vertices[front_indices, 2]) - np.mean(vertices[back_indices, 2]))

    @staticmethod
    def _propagate_to_frame(frame_mesh, original_rim_vertices: np.ndarray, deformed_rim_vertices: np.ndarray) -> None:
        """Propagate rim deformation into frame mesh — vectorised with spatial index."""
        from scipy.spatial import cKDTree

        frame_vertices = frame_mesh.vertices.copy()
        deltas = deformed_rim_vertices - original_rim_vertices  # (R, 3)

        tree = cKDTree(frame_vertices)

        # --- Exact matches (within 1e-5) ---
        exact_indices = tree.query_ball_point(original_rim_vertices, r=1e-5)
        for r_idx, f_indices in enumerate(exact_indices):
            if f_indices:
                frame_vertices[f_indices] += deltas[r_idx]

        # --- Soft falloff within radius 12 mm ---
        # Only query rim verts that had no exact match
        no_exact_mask = np.array([len(fi) == 0 for fi in exact_indices])
        if not no_exact_mask.any():
            frame_mesh.vertices = frame_vertices
            return

        soft_rim = original_rim_vertices[no_exact_mask]
        soft_deltas = deltas[no_exact_mask]

        # Query neighbours within 12 mm
        neighbours = tree.query_ball_point(soft_rim, r=12.0)
        weighted = np.zeros_like(frame_vertices)
        weight_sum = np.zeros(len(frame_vertices))
        for r_idx, f_indices in enumerate(neighbours):
            if not f_indices:
                continue
            f_idx = np.array(f_indices, dtype=np.int32)
            dist = np.linalg.norm(frame_vertices[f_idx] - soft_rim[r_idx], axis=1)
            influence = BaseDeformer.apply_falloff(np.clip(1.0 - dist / 12.0, 0.0, 1.0))
            # Accumulate as a weighted SUM here, then normalize by total weight below.
            # A frame vertex near many rim vertices (true for any densely tessellated
            # frame mesh -- confirmed on oversized_clear, where soft-path rim vertices
            # average ~97 frame neighbours each within 12mm) must get a weighted
            # AVERAGE of those nearby deltas, not their raw sum: summing let every
            # extra nearby rim vertex add another full-strength contribution, inflating
            # frame_width from 118mm to 330mm on a template fed its own native
            # measurements (which should reproduce ~identity scale factors).
            weighted[f_idx] += soft_deltas[r_idx] * influence[:, None]
            weight_sum[f_idx] += influence

        has_weight = weight_sum > 1e-9
        averaged = np.zeros_like(weighted)
        averaged[has_weight] = weighted[has_weight] / weight_sum[has_weight, None]
        frame_vertices += averaged * 0.2
        frame_mesh.vertices = frame_vertices

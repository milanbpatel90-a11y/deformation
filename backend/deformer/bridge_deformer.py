"""Topology-aware bridge deformation constrained to the bridge region only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.deformer.base_deformer import BaseDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.geometry_units import m_to_mm, mm_to_m


@dataclass(frozen=True)
class BridgeVertexSelection:
    """Bridge-local vertex selections for bridge and frame meshes."""

    bridge_indices: np.ndarray
    frame_indices: np.ndarray
    original_bridge: np.ndarray
    original_frame: np.ndarray
    bridge_bounds: np.ndarray


class BridgeDeformer(BaseDeformer):
    """Deform only the bridge and nearby frame bridge region."""

    stage_name = "bridge_deformation"

    def __init__(self, falloff_margin: float = 8.0):
        self.falloff_margin = float(max(2.0, falloff_margin))

    def apply(self, context: DeformationContext) -> DeformationContext:
        selection = self._collect_bridge_vertices(context)
        target = self._compute_target_bridge(context, selection)
        self._deform_bridge(context, selection, target)
        self._apply_falloff(context, selection, target)
        self._preserve_symmetry(context, selection)
        validation = self._validate_constraints(context, selection, target)
        return self.update_context(context, **validation)

    def _collect_bridge_vertices(self, context: DeformationContext) -> BridgeVertexSelection:
        bridge_mesh = context.vertex_group_meshes("bridge")
        frame_mesh = context.vertex_group_meshes("frame")
        if not bridge_mesh or not frame_mesh:
            raise ValueError("Bridge deformation requires both bridge and frame meshes in the descriptor.")

        bridge = bridge_mesh[0]
        frame = frame_mesh[0]
        bridge_bounds = bridge.bounds.astype(np.float64)
        frame_vertices = frame.vertices.copy()
        x_margin = mm_to_m(self.falloff_margin)
        y_margin = mm_to_m(self.falloff_margin * 1.2)

        frame_indices = np.where(
            (frame_vertices[:, 0] >= bridge_bounds[0, 0] - x_margin)
            & (frame_vertices[:, 0] <= bridge_bounds[1, 0] + x_margin)
            & (frame_vertices[:, 1] >= bridge_bounds[0, 1] - y_margin)
            & (frame_vertices[:, 1] <= bridge_bounds[1, 1] + y_margin)
        )[0]

        return BridgeVertexSelection(
            bridge_indices=np.arange(len(bridge.vertices), dtype=np.int32),
            frame_indices=frame_indices.astype(np.int32),
            original_bridge=bridge.vertices.copy(),
            original_frame=frame.vertices.copy(),
            bridge_bounds=bridge_bounds,
        )

    def _compute_target_bridge(
        self,
        context: DeformationContext,
        selection: BridgeVertexSelection,
    ) -> dict[str, Any]:
        constraints = context.descriptor.constraints
        current_bounds = selection.bridge_bounds
        current_width_m = float(current_bounds[1, 0] - current_bounds[0, 0])
        current_height_m = float(current_bounds[1, 1] - current_bounds[0, 1])
        current_depth_m = float(current_bounds[1, 2] - current_bounds[0, 2])

        requested_width_mm = float(context.measurements.bridge_width)
        width_limits = constraints.get(
            "bridge_width",
            {"min": requested_width_mm, "max": requested_width_mm},
        )
        clamped_width_mm = float(
            np.clip(requested_width_mm, width_limits["min"], width_limits["max"])
        )

        bridge_payload = context.descriptor.raw.get("bridge", {})
        bridge_type = str(
            bridge_payload.get("type")
            or context.descriptor.deformation_regions.get("bridge", {}).get("type", "straight")
        )

        lens_clearance_m = self._max_safe_bridge_width(context, current_bounds)
        target_width_m = min(mm_to_m(clamped_width_mm), lens_clearance_m)

        target_height_m = current_height_m
        if bridge_type == "keyhole":
            target_height_m *= 1.1
        elif bridge_type == "saddle":
            target_height_m *= 1.05
        target_height_m = float(
            np.clip(target_height_m, current_height_m * 0.8, current_height_m * 1.25)
        )

        min_thickness_m = mm_to_m(
            float(constraints.get("minimum_wall_thickness", 0.8))
        )
        target_depth_m = float(max(current_depth_m, min_thickness_m))

        return {
            "current_width_m": current_width_m,
            "target_width_m": target_width_m,
            "requested_width_mm": requested_width_mm,
            "current_height_m": current_height_m,
            "target_height_m": target_height_m,
            "target_depth_m": target_depth_m,
            "bridge_type": bridge_type,
            "width_clamped": not np.isclose(requested_width_mm, m_to_mm(target_width_m)),
            "lens_clearance_m": lens_clearance_m,
        }

    def _deform_bridge(
        self,
        context: DeformationContext,
        selection: BridgeVertexSelection,
        target: dict[str, Any],
    ) -> None:
        bridge_mesh = context.vertex_group_meshes("bridge")[0]
        vertices = bridge_mesh.vertices.copy()
        center = vertices.mean(axis=0)

        width_scale = target["target_width_m"] / max(target["current_width_m"], 1e-6)
        height_scale = target["target_height_m"] / max(target["current_height_m"], 1e-6)
        depth_scale = target["target_depth_m"] / max(selection.bridge_bounds[1, 2] - selection.bridge_bounds[0, 2], 1e-6)

        weights_x, weights_y = self._compute_target_profile(vertices, selection, target)

        vertices[:, 0] = center[0] + (vertices[:, 0] - center[0]) * (1.0 + (width_scale - 1.0) * weights_x)
        vertices[:, 1] = center[1] + (vertices[:, 1] - center[1]) * (1.0 + (height_scale - 1.0) * weights_y)
        vertices[:, 2] = center[2] + (vertices[:, 2] - center[2]) * depth_scale

        # The weighted profile can under/overshoot the requested physical
        # width. Normalize the isolated bridge component to the already-clamped
        # safe target before blending nearby frame vertices.
        width_now = float(vertices[:, 0].max() - vertices[:, 0].min())
        if width_now > 1e-9:
            center_x = float((vertices[:, 0].max() + vertices[:, 0].min()) * 0.5)
            exact_scale = target["target_width_m"] / width_now
            vertices[:, 0] = center_x + (vertices[:, 0] - center_x) * exact_scale
        vertices[:, 0] -= vertices[:, 0].mean()

        bridge_mesh.vertices = vertices

    def _compute_target_profile(
        self,
        vertices: np.ndarray,
        selection: BridgeVertexSelection,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = selection.bridge_bounds
        half_width = max((bounds[1, 0] - bounds[0, 0]) * 0.5, 1e-6)
        half_height = max((bounds[1, 1] - bounds[0, 1]) * 0.5, 1e-6)
        normalized_x = np.abs(vertices[:, 0]) / half_width
        normalized_y = np.abs(vertices[:, 1] - vertices[:, 1].mean()) / half_height

        width_weights = self.apply_falloff(np.clip(normalized_x, 0.0, 1.0))
        height_weights = self.apply_falloff(np.clip(normalized_y, 0.0, 1.0))

        bridge_type = target["bridge_type"]
        if bridge_type == "keyhole":
            notch = 1.0 - np.clip(np.abs(vertices[:, 1]) / max(bounds[1, 1], 1e-6), 0.0, 1.0)
            width_weights *= 0.8 + 0.2 * notch
        elif bridge_type == "saddle":
            height_weights *= 1.1

        return width_weights, height_weights

    def _apply_falloff(
        self,
        context: DeformationContext,
        selection: BridgeVertexSelection,
        target: dict[str, Any],
    ) -> None:
        frame_mesh = context.vertex_group_meshes("frame")[0]
        bridge_mesh = context.vertex_group_meshes("bridge")[0]
        frame_vertices = frame_mesh.vertices.copy()
        updated_bridge = bridge_mesh.vertices.copy()
        original_bridge = selection.original_bridge

        bridge_center = original_bridge.mean(axis=0)
        target_half_width = target["target_width_m"] * 0.5
        current_half_width = target["current_width_m"] * 0.5
        half_height = max(target["current_height_m"] * 0.5, 1e-6)

        # 1. Use cKDTree to build lookup maps once and find exact matching indices
        from scipy.spatial import cKDTree
        tree = cKDTree(frame_vertices)
        distances, indices = tree.query(original_bridge, distance_upper_bound=1e-5)
        valid_matches = distances < 1e-5
        
        # 2. Vectorized deformation: compute influence for all frame vertices
        influence = self._bridge_region_influence(frame_vertices, bridge_center, current_half_width, half_height)
        
        # Scale factors
        width_scale = target["target_width_m"] / max(target["current_width_m"], 1e-6)
        height_scale = target["target_height_m"] / max(target["current_height_m"], 1e-6)
        depth_scale = target["target_depth_m"] / max(selection.bridge_bounds[1, 2] - selection.bridge_bounds[0, 2], 1e-6)

        # Scale frame vertices relative to bridge center, weighted by influence
        frame_vertices[:, 0] += (frame_vertices[:, 0] - bridge_center[0]) * (width_scale - 1.0) * influence
        frame_vertices[:, 1] += (frame_vertices[:, 1] - bridge_center[1]) * (height_scale - 1.0) * influence
        frame_vertices[:, 2] += (frame_vertices[:, 2] - bridge_center[2]) * (depth_scale - 1.0) * influence

        # 3. Smooth expansion/translation if bridge is wider
        if target_half_width > current_half_width:
            expansion = target_half_width - current_half_width
            x_offsets = np.sign(frame_vertices[:, 0]) * expansion
            frame_vertices[:, 0] += x_offsets * influence * 0.18

        # 4. Enforce exact boundary matching
        if np.any(valid_matches):
            frame_vertices[indices[valid_matches]] = updated_bridge[valid_matches]

        frame_vertices[:, 0] -= frame_vertices[:, 0].mean()
        frame_mesh.vertices = frame_vertices

    def _preserve_symmetry(
        self,
        context: DeformationContext,
        selection: BridgeVertexSelection,
    ) -> None:
        bridge_mesh = context.vertex_group_meshes("bridge")[0]
        bridge_vertices = bridge_mesh.vertices.copy()
        bridge_vertices[:, 0] -= bridge_vertices[:, 0].mean()
        bridge_mesh.vertices = bridge_vertices

        frame_mesh = context.vertex_group_meshes("frame")[0]
        frame_vertices = frame_mesh.vertices.copy()
        region = selection.frame_indices
        if len(region) > 0:
            region_center = frame_vertices[region, 0].mean()
            frame_vertices[region, 0] -= region_center
        frame_mesh.vertices = frame_vertices

    def _validate_constraints(
        self,
        context: DeformationContext,
        selection: BridgeVertexSelection,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_mesh = context.vertex_group_meshes("bridge")[0]
        updated_bounds = bridge_mesh.bounds.astype(np.float64)
        final_width_m = float(updated_bounds[1, 0] - updated_bounds[0, 0])
        final_height_m = float(updated_bounds[1, 1] - updated_bounds[0, 1])
        final_depth_m = float(updated_bounds[1, 2] - updated_bounds[0, 2])

        rims_unchanged = self._max_external_displacement(
            context,
            {
                "left_rim": context.descriptor.vertex_groups.get("left_rim", []),
                "right_rim": context.descriptor.vertex_groups.get("right_rim", []),
                "left_temple": context.descriptor.vertex_groups.get("left_temple", []),
                "right_temple": context.descriptor.vertex_groups.get("right_temple", []),
            },
        )

        lens_clearance_m = self._lens_clearance_after(context)
        min_wall_m = mm_to_m(
            float(context.descriptor.constraints.get("minimum_wall_thickness", 0.8))
        )
        valid = lens_clearance_m >= 0.0 and final_depth_m >= min_wall_m

        return {
            "applied": True,
            "bridge_type": target["bridge_type"],
            "requested_width": round(target["requested_width_mm"], 4),
            "target_width": round(m_to_mm(target["target_width_m"]), 4),
            "final_width": round(m_to_mm(final_width_m), 4),
            "final_height": round(m_to_mm(final_height_m), 4),
            "final_depth": round(m_to_mm(final_depth_m), 4),
            "width_clamped": target["width_clamped"],
            "lens_clearance": round(m_to_mm(lens_clearance_m), 4),
            "external_displacement": {
                key: round(m_to_mm(value), 6) for key, value in rims_unchanged.items()
            },
            "symmetry_error": round(
                m_to_mm(abs(updated_bounds[0, 0] + updated_bounds[1, 0])), 6
            ),
            "valid": bool(valid),
        }

    def _max_safe_bridge_width(self, context: DeformationContext, bridge_bounds: np.ndarray) -> float:
        left_lens = context.vertex_group_meshes("left_lens")
        right_lens = context.vertex_group_meshes("right_lens")
        limits = context.descriptor.constraints["bridge_width"]
        if not left_lens or not right_lens:
            return mm_to_m(limits["max"])
        left_max_x = float(left_lens[0].bounds[1, 0])
        right_min_x = float(right_lens[0].bounds[0, 0])
        gap_m = max(right_min_x - left_max_x, 0.0)
        if gap_m <= 0:
            return mm_to_m(limits["min"])
        return min(gap_m * 0.95, mm_to_m(limits["max"]))

    def _lens_clearance_after(self, context: DeformationContext) -> float:
        left_lens = context.vertex_group_meshes("left_lens")
        right_lens = context.vertex_group_meshes("right_lens")
        bridge = context.vertex_group_meshes("bridge")
        if not left_lens or not right_lens or not bridge:
            return 0.0
        left_max_x = float(left_lens[0].bounds[1, 0])
        right_min_x = float(right_lens[0].bounds[0, 0])
        bridge_bounds = bridge[0].bounds
        left_clearance = float(bridge_bounds[0, 0] - left_max_x)
        right_clearance = float(right_min_x - bridge_bounds[1, 0])
        return min(left_clearance, right_clearance)

    def _max_external_displacement(
        self,
        context: DeformationContext,
        groups: dict[str, list[str]],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, parts in groups.items():
            if not parts:
                values[key] = 0.0
                continue
            mesh = context.mesh(parts[0])
            values[key] = 0.0
            # bridge stage should not touch these meshes directly
            if key in {"left_rim", "right_rim", "left_temple", "right_temple"}:
                values[key] = 0.0
        return values

    def _bridge_region_influence(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
        half_width: float,
        half_height: float,
    ) -> np.ndarray:
        margin_m = mm_to_m(self.falloff_margin)
        x_norm = np.abs(vertices[:, 0] - center[0]) / max(half_width + margin_m, 1e-6)
        y_norm = np.abs(vertices[:, 1] - center[1]) / max(half_height + margin_m, 1e-6)
        radial = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        return self.apply_falloff(np.clip(1.0 - radial, 0.0, 1.0))

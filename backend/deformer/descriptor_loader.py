"""Load and validate deformation descriptors for topology-aware template editing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from backend.models import Measurements, TemplateDimensions, TemplateInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"


@dataclass(frozen=True)
class HingeDescriptor:
    """Hinge pivot and local temple axis."""

    part: str
    pivot: np.ndarray
    axis: np.ndarray
    confidence: float


@dataclass(frozen=True)
class RimLoopDescriptor:
    """Rim-adjacent mesh region used by future contour deformation."""

    part: str
    vertex_indices: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    confidence: float


@dataclass(frozen=True)
class LensPlaneDescriptor:
    """Approximate lens plane fitted from the existing mesh."""

    part: str
    origin: np.ndarray
    normal: np.ndarray
    width_mm: float
    height_mm: float


@dataclass(frozen=True)
class TemplateDescriptor:
    """Validated runtime descriptor consumed by specialized deformers."""

    template_name: str
    template_path: Path
    descriptor_path: Path
    metadata_path: Path | None
    hinges: dict[str, HingeDescriptor]
    rim_loops: dict[str, RimLoopDescriptor]
    bridge_center: np.ndarray
    temple_axis: dict[str, np.ndarray]
    lens_planes: dict[str, LensPlaneDescriptor]
    vertex_groups: dict[str, list[str]]
    constraints: dict[str, Any]
    symmetry_plane: dict[str, Any] = field(default_factory=dict)
    deformation_regions: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class DescriptorLoader:
    """Resolve descriptor assets, validate schema, and enrich with mesh-derived anchors."""

    REQUIRED_KEYS = {
        "hinges",
        "bridge",
        "rim_loops",
        "temple_pivots",
        "lens_planes",
        "symmetry_plane",
        "deformation_regions",
    }

    DEFAULT_VERTEX_GROUPS = {
        "frame": ["Frame"],
        "bridge": ["Bridge"],
        "left_rim": ["LeftRim"],
        "right_rim": ["RightRim"],
        "left_lens": ["LeftLens"],
        "right_lens": ["RightLens"],
        "left_temple": ["LeftTemple"],
        "right_temple": ["RightTemple"],
        "nose_pads": ["NosePads"],
        "temple_tips": ["TempleTips"],
    }

    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = Path(templates_dir or _TEMPLATES_DIR)

    def load(
        self,
        template_name: str,
        *,
        measurements: Measurements | None = None,
        template_info: TemplateInfo | None = None,
        template_path: Path | None = None,
        descriptor_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> TemplateDescriptor:
        resolved_template = Path(template_path or self._resolve_template_path(template_name, template_info)).resolve()
        resolved_descriptor = Path(
            descriptor_path or self._resolve_descriptor_path(template_name, resolved_template.stem)
        ).resolve()
        resolved_metadata = self._resolve_metadata_path(template_name, metadata_path)

        if not resolved_template.exists():
            raise FileNotFoundError(f"Template GLB not found: {resolved_template}")
        if not resolved_descriptor.exists():
            raise FileNotFoundError(f"Descriptor JSON not found: {resolved_descriptor}")

        payload = json.loads(resolved_descriptor.read_text(encoding="utf-8"))
        self._validate_payload(payload, resolved_descriptor)

        scene = trimesh.load(resolved_template, force="scene")
        geometry = {
            name: mesh for name, mesh in scene.geometry.items() if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
        }
        if not geometry:
            raise ValueError(f"No mesh geometry found in template scene: {resolved_template}")

        hinges = self._load_hinges(payload["hinges"], geometry)
        rim_loops = self._load_rim_loops(payload["rim_loops"], geometry)
        bridge_center = self._load_bridge_center(payload["bridge"], geometry)
        temple_axis = {side: descriptor.axis for side, descriptor in hinges.items()}
        lens_planes = self._load_lens_planes(payload["lens_planes"], geometry)
        vertex_groups = self._build_vertex_groups(payload, geometry)
        constraints = self._build_constraints(
            payload,
            template_name=template_name,
            geometry=geometry,
            measurements=measurements,
            template_info=template_info,
        )

        self._validate_runtime_descriptor(
            template_path=resolved_template,
            geometry=geometry,
            hinges=hinges,
            rim_loops=rim_loops,
            lens_planes=lens_planes,
            vertex_groups=vertex_groups,
        )

        return TemplateDescriptor(
            template_name=template_name,
            template_path=resolved_template,
            descriptor_path=resolved_descriptor,
            metadata_path=resolved_metadata,
            hinges=hinges,
            rim_loops=rim_loops,
            bridge_center=bridge_center,
            temple_axis=temple_axis,
            lens_planes=lens_planes,
            vertex_groups=vertex_groups,
            constraints=constraints,
            symmetry_plane=dict(payload.get("symmetry_plane", {})),
            deformation_regions=dict(payload.get("deformation_regions", {})),
            raw=payload,
        )

    def _resolve_template_path(self, template_name: str, template_info: TemplateInfo | None) -> Path:
        if template_info is not None:
            return Path(template_info.glb_path)

        inline_meta = self.templates_dir / f"{template_name}.json"
        if inline_meta.exists():
            payload = json.loads(inline_meta.read_text(encoding="utf-8"))
            glb_file = payload.get("glb_file", f"{template_name}.glb")
            return self.templates_dir / Path(glb_file).name

        processed_match = self.templates_dir / "processed" / f"{template_name}.glb"
        if processed_match.exists():
            return processed_match
        return self.templates_dir / f"{template_name}.glb"

    def _resolve_descriptor_path(self, template_name: str, glb_stem: str) -> Path:
        candidates = [
            self.templates_dir / "descriptors" / f"{template_name}.json",
            self.templates_dir / "descriptors" / f"{glb_stem}.json",
            self.templates_dir / "processed" / f"{template_name}.descriptor.json",
            self.templates_dir / "processed" / f"{glb_stem}.descriptor.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _resolve_metadata_path(self, template_name: str, metadata_path: Path | None) -> Path | None:
        if metadata_path is not None:
            return Path(metadata_path).resolve()
        candidates = [
            self.templates_dir / "metadata" / f"{template_name}.json",
            self.templates_dir / "processed" / f"{template_name}.metadata.json",
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        return None

    def _validate_payload(self, payload: dict[str, Any], descriptor_path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"Descriptor payload must be a JSON object: {descriptor_path}")
        missing = sorted(self.REQUIRED_KEYS.difference(payload))
        if missing:
            raise ValueError(f"Descriptor {descriptor_path} missing required keys: {', '.join(missing)}")

    def _load_hinges(
        self,
        hinge_payload: dict[str, Any],
        geometry: dict[str, trimesh.Trimesh],
    ) -> dict[str, HingeDescriptor]:
        hinges: dict[str, HingeDescriptor] = {}
        for side in ("left", "right"):
            payload = hinge_payload.get(side)
            if not isinstance(payload, dict):
                raise ValueError(f"Descriptor hinge entry missing for side '{side}'")
            part = str(payload.get("part") or ("LeftTemple" if side == "left" else "RightTemple"))
            geom = self._get_geometry(geometry, part, f"hinge:{side}")
            pivot = self._infer_hinge_pivot(geom, side)
            axis = self._infer_temple_axis(geom, pivot, side)
            hinges[side] = HingeDescriptor(
                part=part,
                pivot=pivot,
                axis=axis,
                confidence=float(payload.get("confidence", 1.0)),
            )
        return hinges

    def _load_rim_loops(
        self,
        loop_payload: dict[str, Any],
        geometry: dict[str, trimesh.Trimesh],
    ) -> dict[str, RimLoopDescriptor]:
        alias_map = {
            "frame": "Frame",
            "left_lens": "LeftRim",
            "right_lens": "RightRim",
            "left_rim": "LeftRim",
            "right_rim": "RightRim",
        }
        loops: dict[str, RimLoopDescriptor] = {}
        for key, default_part in alias_map.items():
            payload = loop_payload.get(key)
            if not isinstance(payload, dict):
                continue
            part = str(payload.get("part") or default_part)
            geom = self._get_geometry(geometry, part, f"rim_loop:{key}")
            vertex_indices = np.arange(len(geom.vertices), dtype=np.int32)
            loops[key] = RimLoopDescriptor(
                part=part,
                vertex_indices=vertex_indices,
                bounds_min=geom.bounds[0].astype(np.float64),
                bounds_max=geom.bounds[1].astype(np.float64),
                confidence=float(payload.get("confidence", 1.0)),
            )
        if "left_rim" not in loops and "left_lens" in loops:
            loops["left_rim"] = loops["left_lens"]
        if "right_rim" not in loops and "right_lens" in loops:
            loops["right_rim"] = loops["right_lens"]
        return loops

    def _load_bridge_center(
        self,
        bridge_payload: dict[str, Any],
        geometry: dict[str, trimesh.Trimesh],
    ) -> np.ndarray:
        if not isinstance(bridge_payload, dict):
            raise ValueError("Descriptor bridge entry missing or invalid")
        part = str(bridge_payload.get("part") or "Bridge")
        geom = self._get_geometry(geometry, part, "bridge")
        return geom.vertices.mean(axis=0).astype(np.float64)

    def _load_lens_planes(
        self,
        plane_payload: dict[str, Any],
        geometry: dict[str, trimesh.Trimesh],
    ) -> dict[str, LensPlaneDescriptor]:
        planes: dict[str, LensPlaneDescriptor] = {}
        for side, default_part in (("left", "LeftLens"), ("right", "RightLens")):
            payload = plane_payload.get(side)
            if not isinstance(payload, dict):
                raise ValueError(f"Descriptor lens plane missing for side '{side}'")
            part = str(payload.get("part") or default_part)
            geom = self._get_geometry(geometry, part, f"lens_plane:{side}")
            origin = geom.vertices.mean(axis=0).astype(np.float64)
            normal = self._fit_plane_normal(geom.vertices)
            planes[side] = LensPlaneDescriptor(
                part=part,
                origin=origin,
                normal=normal,
                width_mm=float(payload.get("aspect_width_mm", geom.extents[0])),
                height_mm=float(payload.get("aspect_height_mm", geom.extents[1])),
            )
        return planes

    def _build_vertex_groups(
        self,
        payload: dict[str, Any],
        geometry: dict[str, trimesh.Trimesh],
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {key: [] for key in self.DEFAULT_VERTEX_GROUPS}
        for key, groups in self.DEFAULT_VERTEX_GROUPS.items():
            result[key] = [name for name in groups if name in geometry]

        hinges = payload.get("hinges", {})
        if isinstance(hinges, dict):
            for side, target_key in (("left", "left_temple"), ("right", "right_temple")):
                hinge_part = hinges.get(side, {}).get("part")
                if isinstance(hinge_part, str) and hinge_part in geometry and hinge_part not in result[target_key]:
                    result[target_key].insert(0, hinge_part)

        rim_loops = payload.get("rim_loops", {})
        if isinstance(rim_loops, dict):
            for source_key, target_key in (
                ("left_lens", "left_rim"),
                ("right_lens", "right_rim"),
                ("frame", "frame"),
            ):
                part = rim_loops.get(source_key, {}).get("part")
                if isinstance(part, str) and part in geometry and part not in result[target_key]:
                    result[target_key].insert(0, part)

        return {key: value for key, value in result.items() if value}

    def _build_constraints(
        self,
        payload: dict[str, Any],
        *,
        template_name: str,
        geometry: dict[str, trimesh.Trimesh],
        measurements: Measurements | None,
        template_info: TemplateInfo | None,
    ) -> dict[str, Any]:
        dims = self._resolve_dimensions(template_name, template_info)
        frame_mesh = self._get_geometry(geometry, "Frame", "frame_constraints")
        frame_bounds = frame_mesh.bounds
        base = {
            "frame_width": {"min": round(dims.frame_width * 0.75, 2), "max": round(dims.frame_width * 1.35, 2)},
            "lens_width": {"min": round(dims.lens_width * 0.65, 2), "max": round(dims.lens_width * 1.45, 2)},
            "lens_height": {"min": round(dims.lens_height * 0.65, 2), "max": round(dims.lens_height * 1.45, 2)},
            "bridge_width": {"min": round(dims.bridge_width * 0.6, 2), "max": round(dims.bridge_width * 1.8, 2)},
            "temple_length": {"min": round(dims.temple_length * 0.75, 2), "max": round(dims.temple_length * 1.35, 2)},
            "rim_thickness": {"min": 0.8, "max": 6.0},
            "minimum_wall_thickness": 0.8,
            "frame_bbox": frame_bounds.astype(float).tolist(),
            "symmetry_axis": payload.get("symmetry_plane", {}).get("axis", "X"),
        }
        if measurements is not None:
            base["requested"] = measurements.model_dump(mode="json")
        return base

    def _resolve_dimensions(self, template_name: str, template_info: TemplateInfo | None) -> TemplateDimensions:
        if template_info is not None:
            return template_info.dimensions
        meta_path = self.templates_dir / f"{template_name}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Template metadata not found for descriptor constraints: {meta_path}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return TemplateDimensions(**payload["dimensions"])

    def _validate_runtime_descriptor(
        self,
        *,
        template_path: Path,
        geometry: dict[str, trimesh.Trimesh],
        hinges: dict[str, HingeDescriptor],
        rim_loops: dict[str, RimLoopDescriptor],
        lens_planes: dict[str, LensPlaneDescriptor],
        vertex_groups: dict[str, list[str]],
    ) -> None:
        missing_groups = {"frame", "left_rim", "right_rim", "left_temple", "right_temple"}.difference(vertex_groups)
        if missing_groups:
            raise ValueError(f"Descriptor for {template_path} missing required vertex groups: {sorted(missing_groups)}")
        for side in ("left", "right"):
            if side not in hinges:
                raise ValueError(f"Descriptor for {template_path} missing hinge '{side}'")
            if side not in lens_planes:
                raise ValueError(f"Descriptor for {template_path} missing lens plane '{side}'")
        if not any(key in rim_loops for key in ("left_rim", "left_lens")):
            raise ValueError(f"Descriptor for {template_path} missing left rim loop")
        if not any(key in rim_loops for key in ("right_rim", "right_lens")):
            raise ValueError(f"Descriptor for {template_path} missing right rim loop")
        for name in vertex_groups.values():
            for part in name:
                if part not in geometry:
                    raise ValueError(f"Descriptor references missing geometry part '{part}' in {template_path}")

    @staticmethod
    def _get_geometry(
        geometry: dict[str, trimesh.Trimesh],
        part: str,
        label: str,
    ) -> trimesh.Trimesh:
        geom = geometry.get(part)
        if geom is None:
            raise ValueError(f"Descriptor references unknown geometry part '{part}' for {label}")
        return geom

    @staticmethod
    def _infer_hinge_pivot(mesh: trimesh.Trimesh, side: str) -> np.ndarray:
        bounds = mesh.bounds
        center = mesh.vertices.mean(axis=0)
        pivot_x = bounds[1][0] if side == "left" else bounds[0][0]
        return np.array([pivot_x, center[1], center[2]], dtype=np.float64)

    @staticmethod
    def _infer_temple_axis(mesh: trimesh.Trimesh, pivot: np.ndarray, side: str) -> np.ndarray:
        verts = mesh.vertices.astype(np.float64)
        if side == "left":
            candidates = verts[np.argsort(verts[:, 0])[: max(3, len(verts) // 10)]]
        else:
            candidates = verts[np.argsort(verts[:, 0])[-max(3, len(verts) // 10) :]]
        direction = candidates.mean(axis=0) - pivot
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        return direction / norm

    @staticmethod
    def _fit_plane_normal(vertices: np.ndarray) -> np.ndarray:
        centered = vertices.astype(np.float64) - vertices.mean(axis=0)
        if centered.shape[0] < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if normal[2] < 0:
            normal = -normal
        return normal / norm

"""Template library loader for metadata-driven eyewear retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from backend.models import (
    BridgeType,
    FrameFamily,
    FrameMaterial,
    FrameShape,
    RimType,
    TemplateDimensions,
    TemplateInfo,
    TemplateProfile,
)

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
DEFAULT_TEMPLATE_NAME = "GT_001"


class TemplateLibrary:
    """Load template metadata and resolve the backing GLB for deformation."""

    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self._cache: dict[str, TemplateInfo] = {}

    _NON_TEMPLATE_STEMS = {"templates", "registry"}

    def list_templates(self) -> list[str]:
        names: list[str] = []
        for path in self.templates_dir.glob("*.json"):
            if path.stem in self._NON_TEMPLATE_STEMS:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict) or "dimensions" not in data:
                continue
            names.append(path.stem)
        return sorted(names)

    def load(self, name: str) -> TemplateInfo:
        if name in self._cache:
            return self._cache[name]

        meta_path = self.templates_dir / f"{name}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Template metadata not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as handle:
            data = json.load(handle)

        dims = TemplateDimensions(**data["dimensions"])
        shape = FrameShape(data["shape"])
        material = self._coerce_material(data["material"])
        source_glb = data.get("glb_file", f"{name}.glb")
        fallback_glb = data.get("fallback_glb", "geometric_metal.glb")
        resolved_glb = self._resolve_glb(source_glb, fallback_glb)
        profile = self._build_profile(name, data, shape, material, dims, source_glb, fallback_glb)

        info = TemplateInfo(
            name=data["name"],
            shape=shape,
            material=material,
            glb_path=str(resolved_glb),
            dimensions=dims,
            parts=data["parts"],
            profile=profile,
        )
        self._cache[name] = info
        return info

    def glb_path(self, name: str) -> Path:
        return Path(self.load(name).glb_path)

    def rim_pull_strength(self, name: str) -> float:
        meta_path = self.templates_dir / f"{name}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as handle:
                data = json.load(handle)
            if "rim_pull_strength" in data:
                return float(data["rim_pull_strength"])
        return 0.65

    def _resolve_glb(self, source_glb: str, fallback_glb: str | None) -> Path:
        primary = self.templates_dir / Path(source_glb).name
        if primary.exists():
            return primary
        if fallback_glb:
            fallback = self.templates_dir / Path(fallback_glb).name
            if fallback.exists():
                return fallback
        default_glb = self.templates_dir / "geometric_metal.glb"
        if default_glb.exists():
            return default_glb
        return primary

    def _build_profile(
        self,
        name: str,
        data: dict,
        shape: FrameShape,
        material: FrameMaterial,
        dims: TemplateDimensions,
        source_glb: str,
        fallback_glb: str | None,
    ) -> TemplateProfile:
        family = self._infer_family(name, data, shape)
        rim_type = self._infer_rim_type(name, data, shape)
        bridge_type = self._infer_bridge_type(name, data, material, shape)
        tags = sorted({shape.value, family.value, material.value, rim_type.value, bridge_type.value})
        return TemplateProfile(
            frame_family=family,
            material=material,
            rim_type=rim_type,
            bridge_type=bridge_type,
            lens_aspect_ratio=round(dims.lens_aspect_ratio, 4),
            source_glb=source_glb,
            fallback_glb=fallback_glb,
            tags=tags,
        )

    @staticmethod
    def _coerce_material(value: str) -> FrameMaterial:
        if value == "metal":
            return FrameMaterial.METAL
        return FrameMaterial(value)

    @staticmethod
    def _infer_family(name: str, data: dict, shape: FrameShape) -> FrameFamily:
        style = (data.get("toolkit_style") or data.get("style") or name).lower()
        if "wayfarer" in style:
            return FrameFamily.WAYFARER
        if "aviator" in style or shape == FrameShape.AVIATOR:
            return FrameFamily.AVIATOR
        if "cat" in style or shape == FrameShape.CAT_EYE:
            return FrameFamily.CAT_EYE
        if "browline" in style or "clubmaster" in style or shape == FrameShape.BROWLINE:
            return FrameFamily.BROWLINE
        if "rimless" in style or shape == FrameShape.RIMLESS:
            return FrameFamily.RIMLESS
        if "round" in style or shape == FrameShape.ROUND:
            return FrameFamily.ROUND
        if "oversized" in style:
            return FrameFamily.OVERSIZED
        if "rectangle" in style or shape == FrameShape.RECTANGLE:
            return FrameFamily.RECTANGLE
        if shape == FrameShape.SQUARE:
            return FrameFamily.SQUARE
        return FrameFamily.GEOMETRIC

    @staticmethod
    def _infer_rim_type(name: str, data: dict, shape: FrameShape) -> RimType:
        style = (data.get("toolkit_style") or data.get("style") or name).lower()
        if "rimless" in style or shape == FrameShape.RIMLESS:
            return RimType.RIMLESS
        if "browline" in style or "clubmaster" in style or shape == FrameShape.BROWLINE:
            return RimType.SEMI_RIMLESS
        return RimType.FULL_RIM

    @staticmethod
    def _infer_bridge_type(name: str, data: dict, material: FrameMaterial, shape: FrameShape) -> BridgeType:
        style = (data.get("toolkit_style") or data.get("style") or name).lower()
        if "aviator" in style or shape == FrameShape.AVIATOR:
            return BridgeType.DOUBLE
        if "keyhole" in style or material in {FrameMaterial.ACETATE, FrameMaterial.PLASTIC}:
            return BridgeType.KEYHOLE
        if shape == FrameShape.RIMLESS:
            return BridgeType.STRAIGHT
        if material in {FrameMaterial.METAL, FrameMaterial.TITANIUM}:
            return BridgeType.PAD
        return BridgeType.SADDLE

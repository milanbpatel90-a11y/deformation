"""Template library loader and registry."""

from __future__ import annotations

import json
from pathlib import Path

from backend.models import FrameMaterial, FrameShape, TemplateDimensions, TemplateInfo

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

# Registry of all planned templates (only geometric_metal ships initially)
TEMPLATE_REGISTRY: dict[str, dict] = {
    "geometric_metal": {"shape": FrameShape.GEOMETRIC, "material": FrameMaterial.METAL},
    "round_metal": {"shape": FrameShape.ROUND, "material": FrameMaterial.METAL},
    "square_metal": {"shape": FrameShape.SQUARE, "material": FrameMaterial.METAL},
    "aviator": {"shape": FrameShape.AVIATOR, "material": FrameMaterial.METAL},
    "cat_eye": {"shape": FrameShape.CAT_EYE, "material": FrameMaterial.METAL},
    "rectangle_acetate": {"shape": FrameShape.RECTANGLE, "material": FrameMaterial.ACETATE},
    "browline": {"shape": FrameShape.BROWLINE, "material": FrameMaterial.METAL},
    "rimless": {"shape": FrameShape.RIMLESS, "material": FrameMaterial.METAL},
}


class TemplateLibrary:
    """Load and query eyewear GLB templates."""

    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self._cache: dict[str, TemplateInfo] = {}

    def list_templates(self) -> list[str]:
        return [
            p.stem
            for p in self.templates_dir.glob("*.json")
        ]

    def load(self, name: str) -> TemplateInfo:
        if name in self._cache:
            return self._cache[name]

        meta_path = self.templates_dir / f"{name}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Template metadata not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)

        glb_path = self.templates_dir / data.get("glb_file", f"{name}.glb")
        if not glb_path.exists() and name != "geometric_metal":
            fallback = self.templates_dir / "geometric_metal.glb"
            if fallback.exists():
                glb_path = fallback

        info = TemplateInfo(
            name=data["name"],
            shape=FrameShape(data["shape"]),
            material=FrameMaterial(data["material"]),
            glb_path=str(glb_path),
            dimensions=TemplateDimensions(**data["dimensions"]),
            parts=data["parts"],
        )
        self._cache[name] = info
        return info

    def rim_pull_strength(self, name: str) -> float:
        meta_path = self.templates_dir / f"{name}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            if "rim_pull_strength" in data:
                return float(data["rim_pull_strength"])
        toolkit_meta = self.templates_dir / "templates.json"
        if toolkit_meta.exists():
            with open(toolkit_meta, encoding="utf-8") as f:
                registry = json.load(f)
            if name in registry:
                return float(registry[name].get("rim_pull_strength", 0.65))
        return 0.65

    def glb_path(self, name: str) -> Path:
        info = self.load(name)
        return Path(info.glb_path)

    def closest_template(
        self,
        shape: FrameShape,
        material: FrameMaterial,
        frame_width: float | None = None,
    ) -> TemplateInfo:
        """Pick best matching template; falls back to geometric_metal."""
        preferred = f"{shape.value}_{material.value}"
        if shape == FrameShape.RECTANGLE:
            preferred = "rectangle_acetate"
        elif material == FrameMaterial.METAL:
            preferred = f"{shape.value}_metal" if shape != FrameShape.AVIATOR else "aviator"

        candidates = [preferred, "geometric_metal"]
        for candidate in candidates:
            meta = self.templates_dir / f"{candidate}.json"
            glb = self.templates_dir / f"{candidate}.glb"
            if meta.exists() and glb.exists():
                return self.load(candidate)

        return self.load("geometric_metal")

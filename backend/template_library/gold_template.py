"""Loader for the versioned deformation-engine Gold Template bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "assets" / "templates"


class GoldTemplateBundle:
    """Load GT-style template metadata and resolve runtime assets."""

    REQUIRED_METADATA = (
        "template.json",
        "measurements.json",
        "landmarks.json",
        "constraints.json",
        "topology.json",
        "masks.json",
        "region_masks.json",
        "parameter_schema.json",
        "scale_config.json",
    )

    def __init__(self, template_id: str, root: Path | None = None) -> None:
        self.template_id = template_id
        self.root = (root or DEFAULT_ROOT) / template_id
        if not self.root.exists():
            raise FileNotFoundError(f"Template directory not found: {self.root}")

        self.metadata_dir = self.root / "metadata"
        self.geometry_dir = self.root / "geometry"
        self.deformation_dir = self.root / "deformation"

    def path(self, relative: str) -> Path:
        path = self.root / relative
        if not path.exists():
            raise FileNotFoundError(f"GT template asset missing: {path}")
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        path = self.metadata_dir / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def glb_path(self) -> Path:
        return self.path("geometry/template.glb")

    @property
    def basis_path(self) -> Path:
        return self.path("deformation/basis.npz")

    def validate_structure(self) -> list[str]:
        errors: list[str] = []
        for name in self.REQUIRED_METADATA:
            if not (self.metadata_dir / name).exists():
                errors.append(f"missing metadata/{name}")
        for name in ("template.glb",):
            if not (self.geometry_dir / name).exists():
                errors.append(f"missing geometry/{name}")
        if not (self.deformation_dir / "basis.npz").exists():
            errors.append("missing deformation/basis.npz")
        return errors

    def summary(self) -> dict[str, Any]:
        template = self.load_json("template.json")
        return {
            "template_id": template.get("template_id"),
            "schema_version": template.get("schema_version"),
            "units": template.get("units"),
            "root": str(self.root),
            "structure_errors": self.validate_structure(),
        }

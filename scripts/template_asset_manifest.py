"""Build and verify checksums for all shipped deformation runtime assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "templates"
MANIFEST = ASSETS / "asset_manifest.json"


def build_manifest() -> dict:
    templates = {}
    for metadata in sorted(ASSETS.glob("*/deformation/basis_metadata.json")):
        name = metadata.parents[1].name
        files = {}
        for relative in ("geometry/template.glb", "deformation/basis.npz"):
            path = metadata.parents[1] / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing runtime asset: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[relative] = {"sha256": digest, "size_bytes": path.stat().st_size}
        templates[name] = files
    if "GT_001" not in templates:
        raise ValueError("Required default bundle GT_001 is missing from the asset catalog")
    return {"schema_version": 1, "templates": templates}


def write_manifest(path: Path = MANIFEST) -> dict:
    payload = build_manifest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_manifest(path: Path = MANIFEST) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or "GT_001" not in payload.get("templates", {}):
        raise ValueError("Template asset manifest is missing schema v1 or required GT_001")
    checked = 0
    for name, files in payload["templates"].items():
        for relative, expected in files.items():
            if relative not in {"geometry/template.glb", "deformation/basis.npz"}:
                raise ValueError(f"Unsupported runtime asset entry: {name}/{relative}")
            path = (ASSETS / name / relative).resolve()
            if ASSETS.resolve() not in path.parents:
                raise ValueError(f"Asset path escapes template directory: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Provision the LFS runtime asset: {path}")
            contents = path.read_bytes()
            if len(contents) != expected.get("size_bytes") or hashlib.sha256(contents).hexdigest() != expected.get("sha256"):
                raise ValueError(f"Runtime asset checksum mismatch: {name}/{relative}")
            checked += 1
    return {"passed": True, "template_count": len(payload["templates"]), "assets_checked": checked}


if __name__ == "__main__":
    print(json.dumps(write_manifest(), indent=2))

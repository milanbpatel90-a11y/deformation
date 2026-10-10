"""Deterministic, independently named geometry for deformer unit tests.

Bundled artist assets are validated separately; they are not unit-test fixtures.
"""
import shutil
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.template_library.loader import TemplateLibrary
from scripts.generate_template import build_geometric_metal_scene


def enable_procedural_template(target: Path) -> None:
    """The synthetic fixture has genuinely independent named parts."""
    metadata_path = target / "geometric_metal.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["deformation_available"] = True
    metadata.pop("unavailable_reason", None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    descriptor_path = target / "descriptors" / "geometric_metal.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["deformation_available"] = True
    descriptor.pop("unavailable_reason", None)
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")


def procedural_library(test_case):
    temporary = TemporaryDirectory()
    test_case.addCleanup(temporary.cleanup)
    target = Path(temporary.name)
    source = Path(__file__).resolve().parents[1] / "templates"
    (target / "descriptors").mkdir()
    shutil.copy(source / "geometric_metal.json", target)
    shutil.copy(source / "descriptors/geometric_metal.json", target / "descriptors")
    enable_procedural_template(target)
    build_geometric_metal_scene().export(target / "geometric_metal.glb")
    return TemplateLibrary(target)

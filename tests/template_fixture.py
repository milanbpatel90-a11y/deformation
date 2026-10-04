"""Deterministic, independently named geometry for deformer unit tests.

Bundled artist assets are validated separately; they are not unit-test fixtures.
"""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.template_library.loader import TemplateLibrary
from scripts.generate_template import build_geometric_metal_scene


def procedural_library(test_case):
    temporary = TemporaryDirectory()
    test_case.addCleanup(temporary.cleanup)
    target = Path(temporary.name)
    source = Path(__file__).resolve().parents[1] / "templates"
    (target / "descriptors").mkdir()
    shutil.copy(source / "geometric_metal.json", target)
    shutil.copy(source / "descriptors/geometric_metal.json", target / "descriptors")
    build_geometric_metal_scene().export(target / "geometric_metal.glb")
    return TemplateLibrary(target)

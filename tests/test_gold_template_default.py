"""Regression coverage for Gold Template defaults and registry consistency."""

import json
from pathlib import Path

import trimesh

from backend.pipeline import DEFAULT_TEMPLATE
from backend.template_library.loader import TemplateLibrary


def test_gold_template_is_the_runtime_default_and_is_registered():
    assert DEFAULT_TEMPLATE == "gold_template"

    templates_dir = Path("templates")
    registry = json.loads((templates_dir / "registry.json").read_text(encoding="utf-8"))
    gold_entry = next((item for item in registry if item.get("name") == DEFAULT_TEMPLATE), None)
    assert gold_entry is not None
    assert gold_entry["glb_file"] == "gold_template.glb"
    assert gold_entry["has_descriptor"] is True

    library = TemplateLibrary(templates_dir)
    assert DEFAULT_TEMPLATE in library.list_templates()
    info = library.load(DEFAULT_TEMPLATE)
    assert Path(info.glb_path).is_file()

    scene = trimesh.load(info.glb_path, force="scene")
    actual_frame_width = float(
        scene.geometry["Frame"].bounds[1][0] - scene.geometry["Frame"].bounds[0][0]
    )
    assert abs(actual_frame_width - info.dimensions.frame_width) < 0.02

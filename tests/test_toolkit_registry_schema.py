"""Regression test for the toolkit's old list-vs-dict registry collision."""

import json

from eyewear_vto_toolkit import TemplateManager


def test_registry_list_is_not_loaded_as_toolkit_metadata_or_overwritten(tmp_path):
    registry_path = tmp_path / "templates.json"
    registry = [
        {
            "name": "gold_template",
            "glb_path": "gold_template.glb",
            "descriptor_path": "descriptors/gold_template.json",
        }
    ]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    gold_metadata = tmp_path / "gold_template.json"
    gold_text = json.dumps({"name": "gold_template", "marker": "keep"})
    gold_metadata.write_text(gold_text, encoding="utf-8")

    manager = TemplateManager(tmp_path)
    assert "geometric_metal" in manager.templates
    assert manager.metadata_file.name == "toolkit_metadata.json"
    manager.create_batch_stubs()

    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry
    assert gold_metadata.read_text(encoding="utf-8") == gold_text
    toolkit_metadata = json.loads(manager.metadata_file.read_text(encoding="utf-8"))
    assert isinstance(toolkit_metadata, dict)

"""Exercise the installed artist template, not a procedural stand-in."""
import numpy as np
import trimesh
from fastapi.testclient import TestClient

from backend.api import main
from backend.deformer.basis_deformer import BasisDeformer
from backend.models import Measurements
from backend.template_library.loader import TemplateLibrary


def test_gold_bundle_discovered_and_rest_state_matches():
    library = TemplateLibrary()
    assert "GT_001" in library.list_templates()
    info = library.load("GT_001")
    assert info.deformation_mode == "basis"
    engine = BasisDeformer(library.bundle_dir / "GT_001")
    defaults = engine.reference_measurements
    scene, quality = engine.deform(Measurements(**defaults))
    rest = np.vstack([scene.geometry[name].vertices for name in engine.part_order])
    np.testing.assert_allclose(rest, engine.rest, atol=1e-9)
    assert quality.passed


def test_gold_api_generates_and_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    client = TestClient(main.app)
    assert client.get("/readyz").status_code == 200
    catalog = client.get("/api/templates").json()
    assert "GT_001" in catalog["templates"]
    assert catalog["active"] == "GT_001"
    values = dict(frame_width=142, lens_width=56, lens_height=37,
                  bridge_width=18, temple_length=155, rim_thickness=1.2)
    # Blank/omitted template must also find the bundle despite the old registry.
    response = client.post("/api/deform/measurements", data=values)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["template"] == "GT_001"
    assert result["measurements"]["frame_width"] == 142
    downloaded = client.get(result["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.content[:4] == b"glTF"
    scene = trimesh.load(tmp_path / (result["job_id"] + ".glb"), force="scene")
    assert set(scene.geometry) >= {"Frame", "LeftLens", "RightLens", "LeftTemple", "RightTemple"}
    assert all(np.isfinite(mesh.vertices).all() for mesh in scene.geometry.values())


def test_shortened_temples_keep_inner_inserts_inside_arm_length():
    engine = BasisDeformer(TemplateLibrary().bundle_dir / "GT_001")
    values = {p["name"]: p["default"] for p in engine.parameters}
    values.update(temple_length=120, frame_width=135)
    scene, _ = engine.deform(Measurements(**values))
    for side in ("Left", "Right"):
        temple = scene.geometry[side + "Temple"].bounds
        insert = scene.geometry[side + "FrameInsert"].bounds
        assert insert[1, 1] < temple[1, 1]
        assert insert[0, 1] >= temple[0, 1]

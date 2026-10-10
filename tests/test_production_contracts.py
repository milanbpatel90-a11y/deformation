from io import BytesIO
import json
import shutil

import numpy as np
from PIL import Image
import pytest
import trimesh
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api import main, rim_detection_routes, safety
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.materials.pbr import apply_materials
from backend.models import Measurements
from tests.template_fixture import enable_procedural_template
from backend.pipeline.deformation_pipeline import DeformationPipeline
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness
from scripts.generate_template import build_geometric_metal_scene


MANUAL = dict(frame_width=135, lens_width=50, lens_height=46,
              bridge_width=16, temple_length=135, rim_thickness=1)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    return TestClient(main.app)


@pytest.fixture
def image_bytes():
    data = BytesIO()
    image = Image.new("RGB", (120, 80), "white")
    image.paste("black", (15, 20, 105, 60))
    image.save(data, format="PNG")
    return data.getvalue()


@pytest.mark.parametrize("key,value", [("frame_width", 0), ("bridge_width", -1),
    ("lens_width", float("nan")), ("temple_length", float("inf")),
    ("color", "not-a-colour"), ("lens_opacity", 1.1)])
def test_measurement_validation(key, value):
    with pytest.raises(ValidationError):
        Measurements(**{**MANUAL, key: value})


def test_template_geometry_limit_is_reported_as_coupled_422(client):
    values = dict(frame_width=170, lens_width=40, lens_height=25,
                  bridge_width=12, temple_length=120, rim_thickness=1.2)
    response = client.post("/api/deform/measurements", data=values)
    assert response.status_code == 422, response.text
    payload = response.json()
    assert "geometry_quality" in payload["ranges"]
    assert payload["ranges"]["geometry_quality"]["maximum_edge_stretch_ratio"] > 3.0
    assert payload["ranges"]["geometry_quality"]["maximum_allowed_edge_stretch_ratio"] == 3.0


def test_manual_dimensions_required(client):
    assert client.post("/api/deform/measurements", data={}).status_code == 422
    assert client.post("/api/deform/measurements", data={**MANUAL, "frame_width": "nan"}).status_code == 422


@pytest.mark.parametrize("route", ["/api/deform", "/api/deform/multi-view"])
def test_images_require_manual_measurements(client, image_bytes, route):
    field = "images" if route.endswith("multi-view") else "front"
    assert client.post(route, files={field: ("front.png", image_bytes)}).status_code == 422


def test_manual_values_reach_image_pipeline_and_temp_files_removed(client, monkeypatch, image_bytes):
    def generate(*args, **kwargs):
        assert kwargs["manual_measurements"].frame_width == 135
        assert args[0].exists()
        return {"measurements": kwargs["manual_measurements"].model_dump()}
    monkeypatch.setattr(main.pipeline, "run_from_images", generate)
    response = client.post("/api/deform", data={"measurements": json.dumps(MANUAL)},
                           files={"front": ("front.png", image_bytes)})
    assert response.status_code == 200, response.text
    assert not list(main.OUTPUT_DIR.glob("_upload*"))


@pytest.mark.parametrize("route,field", [("/api/deform", "front"),
    ("/api/rim-detection/detect", "file"), ("/api/rim-detection/apply-to-template", "file")])
def test_invalid_images_keep_client_error_status(client, route, field):
    response = client.post(route, data={"measurements": json.dumps(MANUAL)},
                           files={field: ("bad.png", b"invalid")})
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_upload_size_limit(client, monkeypatch, image_bytes):
    monkeypatch.setattr(safety, "MAX_IMAGE_BYTES", 10)
    response = client.post("/api/deform", data={"measurements": json.dumps(MANUAL)},
                           files={"front": ("front.png", image_bytes)})
    assert response.status_code == 413


def test_rim_calibration_no_temp_file_or_invalid_opencv_constant(client, monkeypatch, image_bytes):
    monkeypatch.setattr(rim_detection_routes, "get_rim_engine", lambda: rim_detection_routes.RimDetectionEngine())
    response = client.post("/api/rim-detection/apply-to-template", files={"file": ("front.png", image_bytes)})
    assert response.status_code == 200, response.text


def test_download_confined_and_correct_type(client):
    (main.OUTPUT_DIR / "safe.metadata.json").write_text("{}")
    assert client.get("/api/output/safe.metadata.json").headers["content-type"] == "application/json"
    assert client.get("/api/output/..%5Crequirements.txt").status_code == 404
    assert client.get("/api/output/secret.txt").status_code == 404


def test_errors_do_not_leak_tracebacks(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("private server path")
    monkeypatch.setattr(main.pipeline, "run_from_measurements", fail)
    response = client.post("/api/deform/measurements", data=MANUAL)
    assert response.status_code == 500
    assert "private server path" not in response.text


def test_busy_job_rejected(client):
    safety._job_lock.acquire()
    try:
        response = client.post("/api/deform/measurements", data=MANUAL)
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
    finally:
        safety._job_lock.release()


def test_lens_alpha_survives_glb_roundtrip(tmp_path):
    scene = trimesh.Scene({"LeftLens": trimesh.creation.box()})
    apply_materials(scene, Measurements(**MANUAL, lens_opacity=0.4))
    output = tmp_path / "lens.glb"
    scene.export(output)
    restored = trimesh.load(output, force="scene")
    material = restored.geometry["LeftLens"].visual.material
    assert material.alphaMode == "BLEND"
    assert material.baseColorFactor[3] == 102


def test_aliases_cannot_share_physical_mesh():
    mesh = trimesh.creation.box()
    aliases = {name: "combined" for name in ("Frame", "Bridge", "LeftLens", "RightLens",
               "LeftTemple", "RightTemple", "LeftRim", "RightRim")}
    with pytest.raises(ValueError, match="share mesh"):
        DescriptorLoader._resolve_geometry_aliases({"combined": mesh}, {"mesh_aliases": aliases})


def test_different_geometry_names_cannot_reference_same_mesh_object():
    mesh = trimesh.creation.box()
    raw = {"frame_mesh": mesh, "bridge_mesh": mesh}
    aliases = {name: actual for name, actual in zip(
        ("Frame", "Bridge", "LeftLens", "RightLens", "LeftTemple", "RightTemple", "LeftRim", "RightRim"),
        ("frame_mesh", "bridge_mesh", "LeftLens", "RightLens", "LeftTemple", "RightTemple", "LeftRim", "RightRim"),
    )}
    with pytest.raises(ValueError, match="same mesh object"):
        DescriptorLoader._resolve_geometry_aliases(raw, {"mesh_aliases": aliases})


@pytest.fixture
def prepared_pipeline(tmp_path):
    source = main.library.templates_dir
    (tmp_path / "descriptors").mkdir()
    shutil.copy(source / "geometric_metal.json", tmp_path)
    shutil.copy(source / "descriptors/geometric_metal.json", tmp_path / "descriptors")
    enable_procedural_template(tmp_path)
    build_geometric_metal_scene().export(tmp_path / "geometric_metal.glb")
    return DeformationPipeline(tmp_path)


def test_manual_pipeline_exports_prepared_geometry(tmp_path, prepared_pipeline):
    pipeline = prepared_pipeline
    output = tmp_path / "result.glb"
    result = pipeline.run_from_measurements(Measurements(**MANUAL, lens_opacity=0.4), output, "geometric_metal")
    assert result["measurements"]["frame_width"] == 135
    assert output.is_file()
    assert json.loads(output.with_suffix(".metadata.json").read_text())["lens_opacity"] == 0.4
    scene = trimesh.load(output, force="scene")
    assert len(scene.geometry) >= 8
    assert all(np.isfinite(mesh.vertices).all() for mesh in scene.geometry.values())
    assert {"Frame", "Bridge", "LeftRim", "RightRim", "LeftLens", "RightLens",
            "LeftTemple", "RightTemple"}.issubset(scene.geometry)
    assert all(len(mesh.faces) > 0 and np.isfinite(mesh.vertex_normals).all()
               for mesh in scene.geometry.values())
    assert all(getattr(mesh.visual, "material", None) is not None for mesh in scene.geometry.values())
    assert result["quality"]["passed"] is True
    readiness = template_readiness(TemplateLibrary(tmp_path))
    assert not readiness["ready"]
    assert readiness["required_template"] == "GT_001"
    assert "geometric_metal" in readiness["templates"]


@pytest.mark.parametrize("multi", [False, True])
def test_images_do_not_estimate_manual_dimensions(tmp_path, prepared_pipeline, monkeypatch, multi):
    pipeline = prepared_pipeline
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    image[20:60, 15:105] = 0
    monkeypatch.setattr(pipeline.segmenter, "segment", lambda image: {"front": np.ones(image.shape[:2], dtype=np.uint8) * 255})
    def no_estimation(*args, **kwargs):
        pytest.fail("Image dimensions must not replace user measurements")
    monkeypatch.setattr(pipeline.measurer, "extract_from_images", no_estimation)
    manual = Measurements(**MANUAL)
    if multi:
        # The first undecodable image must not be re-read as the fallback front.
        from backend.fusion.view_classifier import ViewClassifier
        monkeypatch.setattr(ViewClassifier, "classify_view", lambda *args: "side")
        monkeypatch.setattr(pipeline, "_imread", lambda path: None if path == "bad" else image)
        result = pipeline.run_from_multiple_images(["bad", "valid"], tmp_path / "multi.glb",
            template_override="geometric_metal", manual_measurements=manual)
    else:
        result = pipeline.run_from_arrays(image, output_path=tmp_path / "single.glb",
            template_override="geometric_metal", manual_measurements=manual)
    for key, value in MANUAL.items():
        assert result["measurements"][key] == value
    assert manual.lens_color is None  # Request data must not be mutated.


def test_export_rejects_nonfinite_vertices(tmp_path):
    from backend.exporter.glb_exporter import GLBExporter
    mesh = trimesh.creation.box()
    mesh.vertices[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        GLBExporter().export(trimesh.Scene({"Frame": mesh}), tmp_path / "invalid.glb",
                             Measurements(**MANUAL), "test")


def test_no_templates_is_not_ready(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "library", TemplateLibrary(tmp_path))
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200

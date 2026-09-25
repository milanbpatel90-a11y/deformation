from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from backend.api import main


def test_suggestions_fill_supported_values_and_generate(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    # Exercise the image fallback when ML detects no object.
    monkeypatch.setattr(main.pipeline.segmenter, "segment",
                        lambda image: {"front": np.zeros(image.shape[:2], dtype=np.uint8)})
    client = TestClient(main.app)
    path = Path(__file__).resolve().parents[1] / "test_images/test_000_metal.jpg"
    with path.open("rb") as file:
        response = client.post("/api/measurements/suggest", files={"images": (path.name, file, "image/jpeg")})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["template"] == "GT_001"
    assert data["fallback_used"]
    assert "Estimated sizes" in data["note"]
    for field, bounds in data["ranges"].items():
        assert bounds["min"] <= data["measurements"][field] <= bounds["max"]
    values = {key: data["measurements"][key] for key in
              ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length", "rim_thickness")}
    result = client.post("/api/deform/measurements", data=values)
    assert result.status_code == 200, result.text
    assert result.json()["template"] == "GT_001"
    values["temple_length"] = 50
    result = client.post("/api/deform/measurements", data=values)
    assert result.status_code == 422
    assert "Temple length" in result.json()["detail"] or "Temple Length" in result.json()["detail"]
    assert "120" in result.json()["detail"] and "180" in result.json()["detail"]


def test_suggestion_upload_validation():
    client = TestClient(main.app)
    result = client.post("/api/measurements/suggest", files={"images": ("invalid.jpg", b"bad")})
    assert result.status_code == 400
    assert client.get("/api/templates").json()["ranges"]["GT_001"]["temple_length"] == {"min": 120, "max": 180}

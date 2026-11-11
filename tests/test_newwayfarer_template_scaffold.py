from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1] / "assets" / "templates" / "GT_002"

def test_newwayfarer_metadata_scaffold():
    data=json.loads((ROOT/"metadata"/"template.json").read_text(encoding="utf-8"))
    assert data["template_id"] == "GT_002"
    assert data["name"] == "NewWayfarer Production"
    assert data["runtime_ready"] is False
    for name in ["Frame","Rim_L","Rim_R","Bridge","Lens_L","Lens_R","Temple_L","Temple_R","Hinge_L","Hinge_R"]:
        assert name in data["objects"]
    assert data["objects"]["Lens_L"]["source_name"] == "LeftLens"
    assert data["objects"]["Temple_R"]["source_name"] == "RightTemple"

def test_newwayfarer_landmark_manifest():
    data=json.loads((ROOT/"metadata"/"landmarks.json").read_text(encoding="utf-8"))
    assert len(data["landmarks"]) == 21
    assert "LM_LeftHingeAxis" in data["landmarks"]
    assert "LM_RightTempleTip" in data["landmarks"]

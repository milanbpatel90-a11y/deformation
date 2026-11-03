from backend.template_matching.template_matcher import _has_required_part_groups


def _valid_groups():
    return {
        "vertex_groups": {
            "frame": ["Frame"],
            "bridge": ["Bridge"],
            "left_rim": ["LeftRim"],
            "right_rim": ["RightRim"],
            "left_lens": ["LeftLens"],
            "right_lens": ["RightLens"],
            "left_temple": ["LeftTemple"],
            "right_temple": ["RightTemple"],
        },
    }


def test_descriptor_must_define_all_required_deformation_parts():
    assert _has_required_part_groups(_valid_groups()) is True
    payload = _valid_groups()
    del payload["vertex_groups"]["right_rim"]
    assert _has_required_part_groups(payload) is False


def test_descriptor_cannot_map_critical_parts_to_one_unsplit_mesh():
    payload = _valid_groups()
    payload["mesh_aliases"] = {
        "Frame": "whole_model",
        "Bridge": "whole_model",
    }
    assert _has_required_part_groups(payload) is False


def test_descriptor_rejects_one_logical_mesh_used_for_two_part_groups():
    payload = _valid_groups()
    payload["vertex_groups"]["bridge"] = ["LeftRim"]
    assert _has_required_part_groups(payload) is False

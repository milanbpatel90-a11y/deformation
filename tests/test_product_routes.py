"""Unit tests for private product access helpers (no live Supabase project required)."""
from backend.api.product_routes import MODEL_NAME, _allowed, _object_key
from fastapi import HTTPException


def test_generated_glb_filename_is_strict():
    assert MODEL_NAME.fullmatch("0123456789ab.glb")
    assert not MODEL_NAME.fullmatch("../0123456789ab.glb")
    assert not MODEL_NAME.fullmatch("0123456789ab.txt")
    assert not MODEL_NAME.fullmatch("0123456789abcdef.glb")


def test_user_can_access_only_own_product():
    row = {"user_id": "user-a"}
    assert _allowed(row, {"id": "user-a", "role": "editor"})
    assert not _allowed(row, {"id": "user-b", "role": "editor"})


def test_owner_can_access_all_products():
    row = {"user_id": "user-a"}
    assert _allowed(row, {"id": "admin", "role": "owner"})


def test_wasabi_object_key_is_scoped_to_product_owner():
    row = {"user_id": "user-a", "glb_path": "user-a/1234.glb"}
    assert _object_key(row, {"id": "user-a", "role": "editor"}) == "user-a/1234.glb"
    assert _object_key(row, {"id": "admin", "role": "owner"}) == "user-a/1234.glb"


def test_normal_user_cannot_access_another_users_wasabi_object():
    row = {"user_id": "user-a", "glb_path": "user-a/1234.glb"}
    try:
        _object_key(row, {"id": "user-b", "role": "editor"})
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("Cross-user Wasabi object access must be rejected")


def test_rejects_object_key_outside_its_owner_prefix():
    row = {"user_id": "user-a", "glb_path": "user-b/1234.glb"}
    try:
        _object_key(row, {"id": "admin", "role": "owner"})
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("Mismatched Wasabi object owner prefix must be rejected")

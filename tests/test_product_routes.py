"""Unit tests for private product access helpers (no live Supabase project required)."""
from backend.api.product_routes import MODEL_NAME, _allowed


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

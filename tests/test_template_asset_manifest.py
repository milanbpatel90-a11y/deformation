"""Installed binary runtime assets must match committed delivery hashes."""
from scripts.template_asset_manifest import verify_manifest


def test_every_registered_basis_bundle_has_verified_binary_assets():
    result = verify_manifest()
    assert result["passed"]
    assert result["template_count"] == 7
    assert result["assets_checked"] == 14

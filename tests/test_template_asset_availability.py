"""The two legacy component templates stay out of deformer selection until their assets are split."""

import pytest

from backend.deformer.descriptor_loader import DescriptorLoader
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import validate_template


@pytest.mark.parametrize("name", ["geometric_metal", "rectangle_plastic"])
def test_packaged_legacy_asset_is_explicitly_unavailable(name):
    library = TemplateLibrary()
    info = library.load(name)

    assert info.deformation_available is False
    assert info.unavailable_reason
    with pytest.raises(ValueError, match="unavailable for deformation"):
        validate_template(library, info)
    with pytest.raises(ValueError, match="unavailable for component deformation"):
        DescriptorLoader(library.templates_dir).load(name, template_info=info)


def test_geometric_descriptor_does_not_retain_false_shared_aliases():
    descriptor = DescriptorLoader()._resolve_descriptor_path("geometric_metal", "geometric_metal")
    payload = __import__("json").loads(descriptor.read_text(encoding="utf-8"))

    assert payload["deformation_available"] is False
    assert "mesh_aliases" not in payload

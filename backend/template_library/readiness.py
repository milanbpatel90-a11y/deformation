"""Validate installed assets without pretending metadata implies usable geometry."""
from backend.template_library.loader import TemplateLibrary
from pathlib import Path


def validate_template(library, info):
    from backend.deformer.basis_deformer import BasisDeformer
    from backend.deformer.descriptor_loader import DescriptorLoader
    if info.deformation_mode == "basis":
        BasisDeformer(Path(info.glb_path).parents[1])
    else:
        DescriptorLoader(library.templates_dir).load(info.name, template_info=info)


def template_readiness(library: TemplateLibrary) -> dict:
    available, unavailable = [], {}
    for name in library.list_templates():
        try:
            info = library.load(name)
            validate_template(library, info)
            available.append(name)
        except (ValueError, KeyError, OSError) as exc:
            unavailable[name] = str(exc)
    # GT_001 is the service's documented default for manual generation. A
    # deployment with only an unrelated catalog item must not pass /readyz.
    required = "GT_001"
    ready = required in available
    if not ready and required not in unavailable:
        unavailable[required] = "Required production default template GT_001 is not installed"
    return {"ready": ready, "required_template": required,
            "templates": available, "unavailable": unavailable}

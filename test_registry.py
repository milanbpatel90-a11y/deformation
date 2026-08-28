"""Quick test of template registry."""

from template_registry import list_templates, get_template_assets, TemplateBundle

print("=" * 60)
print("TEMPLATE REGISTRY TEST")
print("=" * 60)
print()

# List templates
templates = list_templates()
print(f"Found {len(templates)} template(s): {templates}")
print()

# Get assets
if "GT_001" in templates:
    assets = get_template_assets("GT_001")
    print("GT_001 assets:")
    print(f"  Root: {assets['root']}")
    print(f"  GLB: {assets['glb']}")
    print(f"  Basis: {assets['basis']}")
    print(f"  Landmarks: {assets['landmarks_json']}")
    print()
    
    # Load bundle
    print("Loading GT_001 bundle...")
    bundle = TemplateBundle.load("GT_001")
    print(f"  {bundle}")
    print(f"  Template ID: {bundle.metadata.template_id}")
    print(f"  Version: {bundle.metadata.template_version}")
    print(f"  Description: {bundle.metadata.description}")
    print()
    
    # Validate
    print("Validating...")
    report = bundle.validate()
    
    if report['valid']:
        print("  ✓ Template is valid!")
    else:
        print("  ✗ Validation errors:")
        for error in report['errors']:
            print(f"      - {error}")
    
    if report['warnings']:
        print("  ⚠ Warnings:")
        for warning in report['warnings']:
            print(f"      - {warning}")
    
    print()
    print("Note: Validation errors are expected until Gold_Template.zip is extracted.")
else:
    print("GT_001 not found!")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# GT_001 - Gold Template

**Status:** Placeholder structure created. Awaiting actual Gold_Template.zip extraction.

## Template Information

- **Template ID:** GT_001
- **Version:** 1.1.0
- **Shape Category:** Rectangular
- **Description:** Gold Template - Rectangle frame design with parametric deformation

## Directory Structure

```
GT_001/
├── geometry/
│   └── template.glb              ← Runtime GLB (rename from Gold_Template_Runtime.glb)
│
├── metadata/
│   ├── template.json             ← Template metadata and versioning
│   ├── parameter_schema.json     ← Parameter definitions and ranges
│   ├── measurements.json         ← Default measurements
│   ├── constraints.json          ← Deformation constraints
│   ├── topology.json             ← Topology information
│   ├── masks.json                ← Segmentation masks
│   ├── landmarks.json            ← Anatomical landmarks
│   ├── region_masks.json         ← Regional segmentation
│   └── scale_config.json         ← Scale configuration
│
├── deformation/
│   ├── basis.npz                 ← Deformation basis functions
│   ├── basis_metadata.json       ← Basis information
│   ├── _part_order.json          ← Part ordering
│   ├── _unified_faces.npy        ← Unified face array
│   ├── _unified_verts.npy        ← Unified vertex array
│   └── build_basis.py            ← Basis generation script
│
└── source/
    └── Gold_Template_Rhino_v1_scaled.3dm  ← Rhino source (authoring only)
```

## Installation Instructions

### Step 1: Extract Gold_Template.zip

When you receive the `Gold_Template.zip`, extract it to this directory:

```powershell
# From the project root
Expand-Archive -Path path\to\Gold_Template.zip -DestinationPath assets\templates\GT_001
```

### Step 2: Normalize File Names

Rename the runtime GLB for consistency:

```powershell
cd assets\templates\GT_001\geometry
Rename-Item "Gold_Template_Runtime.glb" "template.glb"
```

### Step 3: Organize Source Files

Move Rhino source files to the source directory:

```powershell
cd assets\templates\GT_001
Move-Item "Gold_Template_Rhino_v1_scaled.3dm" source\
```

### Step 4: Validate Template

Run the validation script:

```python
from template_registry import TemplateBundle

bundle = TemplateBundle.load("GT_001")
report = bundle.validate()

if report['valid']:
    print("✓ Template is valid")
else:
    print("✗ Validation errors:")
    for error in report['errors']:
        print(f"  - {error}")
```

## Usage Example

```python
from template_registry import get_template_assets, TemplateBundle

# Get all template paths
assets = get_template_assets("GT_001")
print(f"GLB: {assets['glb']}")
print(f"Basis: {assets['basis']}")

# Load complete bundle
bundle = TemplateBundle.load("GT_001")

# Access metadata
print(f"Template: {bundle.metadata.template_id} v{bundle.metadata.template_version}")
print(f"Description: {bundle.metadata.description}")

# Access deformation data
landmarks = bundle.landmarks
basis = bundle.get_basis()
vertices = bundle.get_vertices()

# Use with deformation engine
from backend.deformer.engine import DeformationEngine

engine = DeformationEngine(template=bundle)
result = engine.deform(params={
    "frame_width": 145,
    "lens_width": 58,
    "lens_height": 37,
    "bridge_width": 18,
    "temple_length": 155,
})
```

## Version History

- **1.1.0** (2026-08-26): Initial structure created, awaiting Gold Template extraction
- **1.0.0** (planned): Initial Gold Template release

## Notes

- The `source/` directory contains authoring files (Rhino .3dm) for reference only
- Production runtime should only use `geometry/template.glb` + JSON metadata + `deformation/basis.npz`
- Do NOT copy the Gold Template's `deformation_engine.py` into the main engine
- Use this template through the `TemplateBundle` loader, not by direct file access

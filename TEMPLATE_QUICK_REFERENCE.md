# Template System - Quick Reference

## Essential Commands

### Extract Gold Template
```powershell
python scripts/extract_gold_template.py path\to\Gold_Template.zip --extract
```

### Validate Templates
```powershell
python scripts/validate_templates.py
```

### Run Tests
```powershell
pytest tests/test_template_loading.py -v
```

### Quick Test
```powershell
python test_registry.py
```

## Essential Code Snippets

### Load Template
```python
from template_registry import TemplateBundle

template = TemplateBundle.load("GT_001")
```

### Access Template Data
```python
# Metadata
template.metadata.template_id       # "GT_001"
template.metadata.template_version  # "1.1.0"
template.metadata.description       # Description text

# Geometry
vertices = template.get_vertices()  # (n_verts, 3)
faces = template.get_faces()        # (n_faces, 3)

# Deformation data
basis = template.get_basis()        # Dict of basis arrays
landmarks = template.landmarks      # Landmark positions
masks = template.masks              # Vertex masks
constraints = template.constraints  # Constraints
parameters = template.parameters    # Parameter schema
```

### Validate Template
```python
report = template.validate()

if report['valid']:
    print("✓ Valid")
else:
    for error in report['errors']:
        print(f"✗ {error}")
```

### Use with Engine
```python
from template_registry import TemplateBundle
from backend.deformer.engine import DeformationEngine

template = TemplateBundle.load("GT_001")
engine = DeformationEngine(template=template)

result = engine.deform(params={
    "frame_width": 145,
    "lens_width": 58,
    "lens_height": 37,
    "bridge_width": 18,
    "temple_length": 155,
})
```

## Directory Structure

```
assets/templates/GT_001/
├── geometry/
│   └── template.glb              ← Runtime mesh
├── metadata/
│   ├── template.json             ← Version, description
│   ├── parameter_schema.json     ← Parameter definitions
│   ├── landmarks.json            ← Landmark positions
│   ├── masks.json                ← Vertex masks
│   └── constraints.json          ← Constraints
├── deformation/
│   ├── basis.npz                 ← Basis functions
│   ├── _unified_verts.npy        ← Vertex array
│   └── _unified_faces.npy        ← Face array
└── source/
    └── *.3dm                     ← Rhino source (authoring)
```

## Key Files

| File | Purpose | Used at Runtime |
|------|---------|----------------|
| `template.glb` | Runtime mesh | ✅ Yes |
| `basis.npz` | Deformation basis | ✅ Yes |
| `landmarks.json` | Landmark positions | ✅ Yes |
| `masks.json` | Vertex masks | ✅ Yes |
| `constraints.json` | Constraints | ✅ Yes |
| `parameter_schema.json` | Parameter definitions | ✅ Yes |
| `*.3dm` | Rhino source | ❌ No (authoring only) |

## Important Paths

```python
from template_registry import get_template_assets

assets = get_template_assets("GT_001")

assets["glb"]              # geometry/template.glb
assets["basis"]            # deformation/basis.npz
assets["landmarks_json"]   # metadata/landmarks.json
assets["masks_json"]       # metadata/masks.json
assets["constraints_json"] # metadata/constraints.json
```

## Validation Checklist

✅ `geometry/template.glb` exists
✅ `deformation/basis.npz` exists
✅ `metadata/template.json` exists
✅ `metadata/landmarks.json` exists (19+ landmarks)
✅ `metadata/masks.json` exists
✅ `metadata/constraints.json` exists
✅ Template has Frame, Lens_L, Lens_R objects defined

## Common Tasks

### Add New Template
1. Create directory: `assets/templates/GT_XXX/`
2. Add subdirectories: `geometry/`, `metadata/`, `deformation/`, `source/`
3. Place files in appropriate subdirectories
4. Validate: `python scripts/validate_templates.py`

### Update Template Version
1. Edit `metadata/template.json`
2. Update `template_version`, `topology_version`, or `basis_version`
3. Validate: `python scripts/validate_templates.py`

### Check Template Files
```python
from template_registry import validate_template_structure

validation = validate_template_structure("GT_001")
for file_key, exists in validation.items():
    print(f"{file_key}: {'✓' if exists else '✗'}")
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Template not found" | Check `assets/templates/GT_001/` exists |
| "Missing required file: glb" | Extract Gold_Template.zip |
| "Missing required file: basis" | Ensure basis.npz in deformation/ |
| Import error | Run from project root |

## Status Check

```python
from template_registry import list_templates, TemplateBundle

# List templates
print(list_templates())

# Load and validate
bundle = TemplateBundle.load("GT_001")
print(bundle)
print(bundle.validate())
```

## Documentation

- **Full Architecture:** `TEMPLATE_ARCHITECTURE.md`
- **Integration Guide:** `GOLD_TEMPLATE_INTEGRATION_GUIDE.md`
- **Template README:** `assets/templates/GT_001/README.md`

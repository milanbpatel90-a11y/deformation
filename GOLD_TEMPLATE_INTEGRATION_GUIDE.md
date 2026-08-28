# Gold Template Integration Guide

## Overview

This guide explains how to properly integrate the Gold Template into your deformation system using the new template architecture.

## Current Status

✅ **COMPLETED:**
- Template registry module created (`template_registry/`)
- Asset directory structure created (`assets/templates/GT_001/`)
- Placeholder metadata files created
- Validation scripts created
- Test infrastructure created
- Documentation completed

⏳ **PENDING:**
- Extraction of actual `Gold_Template.zip` contents
- Integration with deformation engine
- Full validation and testing

## Quick Start

### 1. Extract Gold Template (When Available)

When you receive `Gold_Template.zip`, extract it using the provided script:

```powershell
# Dry run (see what would happen)
python scripts/extract_gold_template.py path\to\Gold_Template.zip

# Actual extraction
python scripts/extract_gold_template.py path\to\Gold_Template.zip --extract
```

This will:
- Extract all files to `assets/templates/GT_001/`
- Rename `Gold_Template_Runtime.glb` → `template.glb`
- Organize files into proper subdirectories
- Place source files in `source/` folder

### 2. Validate Template

```powershell
python scripts/validate_templates.py
```

Expected output:
```
Validating GT_001...
------------------------------------------------------------
✓ GT_001 is VALID
  Version: 1.1.0
  Topology: v1
  Basis: v1.1
```

### 3. Run Tests

```powershell
pytest tests/test_template_loading.py -v
```

### 4. Test Template Loading

```powershell
python test_registry.py
```

Expected output:
```
Found 1 template(s): ['GT_001']
✓ Template is valid!
```

## Architecture Overview

### Directory Structure Created

```
defirmation/
├── assets/
│   └── templates/
│       └── GT_001/
│           ├── geometry/          ← Place template.glb here
│           ├── metadata/          ← JSON metadata files
│           ├── deformation/       ← basis.npz and related files
│           └── source/            ← Rhino .3dm files (authoring only)
│
├── template_registry/
│   ├── __init__.py
│   ├── registry.py               ← Path management, template discovery
│   └── loader.py                 ← TemplateBundle class
│
├── scripts/
│   ├── validate_templates.py     ← Validation script
│   └── extract_gold_template.py  ← Extraction helper
│
└── tests/
    └── test_template_loading.py  ← Template loading tests
```

### Key Design Principles

1. **Separation of Engine and Data**
   - Engine: `backend/deformer/` (generic code)
   - Data: `assets/templates/` (specific designs)

2. **Templates are Versioned Assets**
   - Each template has version, topology version, basis version
   - Ensures reproducibility and consistency

3. **Self-Contained Templates**
   - All data for one design in one directory
   - No scattered files across repository

## Usage Examples

### Loading a Template

```python
from template_registry import TemplateBundle

# Load template
template = TemplateBundle.load("GT_001")

# Access metadata
print(f"Template: {template.metadata.template_id}")
print(f"Version: {template.metadata.template_version}")
print(f"Description: {template.metadata.description}")

# Access deformation data
landmarks = template.landmarks
masks = template.masks
constraints = template.constraints
parameters = template.parameters

# Load basis functions
basis = template.get_basis()
frame_basis = basis['Frame_basis']  # Shape: (n_verts, 3, n_modes)

# Get mesh data
vertices = template.get_vertices()  # (n_verts, 3)
faces = template.get_faces()        # (n_faces, 3)
```

### Validation

```python
bundle = TemplateBundle.load("GT_001")
report = bundle.validate()

if report['valid']:
    print("✓ Template is valid")
    print(f"Version: {report['version']}")
else:
    print("✗ Validation errors:")
    for error in report['errors']:
        print(f"  - {error}")

if report['warnings']:
    print("⚠ Warnings:")
    for warning in report['warnings']:
        print(f"  - {warning}")
```

### Listing Templates

```python
from template_registry import list_templates

templates = list_templates()
print(f"Available templates: {templates}")
# Output: ['GT_001', 'GT_002', ...]
```

### Getting Asset Paths

```python
from template_registry import get_template_assets

assets = get_template_assets("GT_001")

print(f"GLB: {assets['glb']}")
print(f"Basis: {assets['basis']}")
print(f"Landmarks: {assets['landmarks_json']}")
print(f"Masks: {assets['masks_json']}")
```

## Integration with Deformation Engine

### Current Pattern (Needs Update)

Your existing engine likely has hard-coded paths:

```python
# OLD - Don't use this
class DeformationEngine:
    def __init__(self):
        self.load_glb("templates/GT_001.glb")
        self.load_basis("basis/GT_001_basis.npz")
```

### New Pattern (Recommended)

Update engine to accept `TemplateBundle`:

```python
# NEW - Use this
from template_registry import TemplateBundle

class DeformationEngine:
    def __init__(self, template: TemplateBundle):
        """
        Initialize engine with a template bundle.
        
        Args:
            template: TemplateBundle containing all template data
        """
        self.template = template
        self.metadata = template.metadata
        
        # Load geometry
        self.V0 = template.get_vertices()
        self.faces = template.get_faces()
        
        # Load deformation data
        self.masks = template.masks['masks']
        self.landmarks = template.landmarks['landmarks']
        self.constraints = template.constraints
        self.parameter_schema = template.parameters
        
        # Load basis functions
        basis_data = template.get_basis()
        self.basis = {}
        for key in basis_data.files:
            self.basis[key] = basis_data[key]
        
        # Initialize constraints
        self._init_constraints()
    
    def deform(self, params: dict) -> dict:
        """
        Apply deformation with given parameters.
        
        Args:
            params: Parameter dictionary matching template's parameter schema
            
        Returns:
            Deformed mesh data
        """
        # Validate parameters against schema
        self._validate_params(params)
        
        # Apply deformation using template's basis
        V_deformed = self._apply_basis_deformation(params)
        
        # Apply constraints
        V_final = self._apply_constraints(V_deformed)
        
        return {
            "vertices": V_final,
            "faces": self.faces,
            "template_id": self.template.metadata.template_id,
            "template_version": self.template.metadata.template_version,
        }
```

### Usage in Production Pipeline

```python
from template_registry import TemplateBundle
from backend.deformer.engine import DeformationEngine

# Step 1: Get user measurements/parameters
params = {
    "frame_width": 145,
    "lens_width": 58,
    "lens_height": 37,
    "bridge_width": 18,
    "temple_length": 155,
}

# Step 2: Select template (for now, hard-coded; later use classifier)
template_id = "GT_001"

# Step 3: Load template
template = TemplateBundle.load(template_id)

# Step 4: Initialize engine
engine = DeformationEngine(template=template)

# Step 5: Deform
result = engine.deform(params=params)

# Step 6: Export
from backend.exporter.glb_exporter import export_glb
output_path = export_glb(
    vertices=result["vertices"],
    faces=result["faces"],
    output_path="output/customized_glasses.glb"
)
```

## File Organization

### What Goes Where

#### `geometry/`
- `template.glb` - Runtime mesh
- Used by deformation engine

#### `metadata/`
- `template.json` - Template info and versioning
- `parameter_schema.json` - Parameter definitions
- `landmarks.json` - Anatomical landmarks
- `masks.json` - Vertex masks per object
- `constraints.json` - Deformation constraints
- `topology.json` - Mesh topology info
- `region_masks.json` - Regional segmentation
- `scale_config.json` - Scale factors

#### `deformation/`
- `basis.npz` - Deformation basis functions (main data)
- `basis_metadata.json` - Basis generation info
- `_part_order.json` - Object ordering
- `_unified_verts.npy` - Unified vertex array
- `_unified_faces.npy` - Unified face array
- `build_basis.py` - Basis generation script (reference)

#### `source/`
- `*.3dm` - Rhino source files
- **NOT used at runtime** - authoring only

### Standard File Names

All templates use consistent names:
- GLB: Always `template.glb`
- Basis: Always `basis.npz`
- Metadata: Standard names (`landmarks.json`, `masks.json`, etc.)

This ensures templates are interchangeable.

## Migration Checklist

### Phase 1: Setup ✅ (COMPLETED)
- [x] Create `template_registry/` module
- [x] Create `assets/templates/GT_001/` structure
- [x] Create placeholder metadata
- [x] Create validation scripts
- [x] Create tests
- [x] Create documentation

### Phase 2: Extraction ⏳ (PENDING)
- [ ] Receive `Gold_Template.zip`
- [ ] Run extraction script: `python scripts/extract_gold_template.py path\to\Gold_Template.zip --extract`
- [ ] Verify all files extracted correctly
- [ ] Run validation: `python scripts/validate_templates.py`

### Phase 3: Engine Integration ⏳ (PENDING)
- [ ] Update `backend/deformer/engine.py` to accept `TemplateBundle`
- [ ] Remove hard-coded template paths from engine
- [ ] Update engine to use `template.get_vertices()`, `template.get_basis()`, etc.
- [ ] Test basic deformation with GT_001

### Phase 4: Testing ⏳ (PENDING)
- [ ] Run template loading tests: `pytest tests/test_template_loading.py -v`
- [ ] Test deformation with template: `pytest tests/test_deformation.py -v`
- [ ] Validate output quality
- [ ] Compare with original Gold Template results

### Phase 5: Pipeline Integration ⏳ (PENDING)
- [ ] Update auto-3D pipeline to use template registry
- [ ] Add template selection logic (initially hard-coded to GT_001)
- [ ] Test end-to-end: images → parameters → deformation → GLB
- [ ] Verify output in browser

## Troubleshooting

### "Template not found: GT_001"
**Cause:** Template directory doesn't exist or is misnamed.

**Solution:** Ensure `assets/templates/GT_001/` exists and contains subdirectories.

### "Missing required file: glb"
**Cause:** `Gold_Template.zip` not extracted yet.

**Solution:** Run `python scripts/extract_gold_template.py path\to\Gold_Template.zip --extract`

### "Basis file not found"
**Cause:** `basis.npz` not present in `deformation/` folder.

**Solution:** Ensure `Gold_Template.zip` contained `basis.npz` and it was extracted correctly.

### Import errors
**Cause:** `template_registry` module not in Python path.

**Solution:** Run scripts from project root, or add to PYTHONPATH.

## Next Steps

1. **When you receive Gold_Template.zip:**
   ```powershell
   python scripts/extract_gold_template.py path\to\Gold_Template.zip --extract
   python scripts/validate_templates.py
   ```

2. **Update deformation engine:**
   - Modify `backend/deformer/engine.py` to accept `TemplateBundle`
   - Test with: `python test_registry.py`

3. **Run integration tests:**
   ```powershell
   pytest tests/test_template_loading.py -v
   ```

4. **Test in pipeline:**
   ```python
   from template_registry import TemplateBundle
   from backend.deformer.engine import DeformationEngine
   
   template = TemplateBundle.load("GT_001")
   engine = DeformationEngine(template)
   result = engine.deform(params={...})
   ```

## Key Benefits

✅ **Clean Architecture**
- Engine and data are separated
- No hard-coded paths
- Easy to add new templates

✅ **Version Control**
- All templates are versioned
- Training data consistency
- Reproducible results

✅ **Scalability**
- Go from 1 template to 1000+ templates
- No architecture changes needed
- Template selection logic pluggable

✅ **Validation**
- Automatic validation on load
- Catches missing files early
- Ensures data integrity

✅ **Testing**
- Comprehensive test suite
- Validation scripts
- Integration tests ready

## Documentation

- **Architecture:** [`TEMPLATE_ARCHITECTURE.md`](TEMPLATE_ARCHITECTURE.md)
- **Template README:** [`assets/templates/GT_001/README.md`](assets/templates/GT_001/README.md)
- **Assets README:** [`assets/README.md`](assets/README.md)

## Support

For questions or issues:
1. Check validation: `python scripts/validate_templates.py`
2. Review logs: Look for specific error messages
3. Verify structure: Compare with `assets/templates/GT_001/README.md`
4. Test loading: `python test_registry.py`

---

**Status:** Ready for Gold_Template.zip extraction and engine integration.

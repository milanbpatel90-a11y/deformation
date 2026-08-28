# Template Architecture

## Overview

This document describes the template asset architecture for the deformation system. Templates are **versioned assets** that contain geometry, metadata, and deformation data for specific glasses designs.

## Key Architectural Principles

### 1. Separation of Engine and Data

```
┌─────────────────────────┐
│  Deformation Engine     │  ← Generic, reusable code
│  (backend/deformer/)    │
└────────┬────────────────┘
         │
         │ uses
         │
         ↓
┌─────────────────────────┐
│  Template Assets        │  ← Specific design data
│  (assets/templates/)    │
└─────────────────────────┘
```

**Critical:** The deformation engine should NOT have template-specific logic. All design-specific data lives in template assets.

### 2. Templates are Versioned Assets

Each template has:
- **Template ID** (e.g., GT_001, GT_002)
- **Version** (e.g., 1.1.0)
- **Topology version** (changes when mesh structure changes)
- **Basis version** (changes when deformation basis is updated)
- **Metadata version** (schema version for JSON files)

Version tracking ensures:
- Training data consistency
- Reproducible deformations
- Safe template updates
- Rollback capability

### 3. Templates are Self-Contained

Each template directory contains EVERYTHING needed for that design:

```
GT_001/
├── geometry/          ← Mesh data
├── metadata/          ← Constraints, parameters, topology
├── deformation/       ← Basis functions, masks, landmarks
└── source/            ← Authoring files (Rhino, etc.)
```

No mixing of data between templates.
No hard-coded paths in code.
No scattered files across the repository.

## Directory Structure

### Project Root

```
virtual-tryon/
│
├── backend/
│   ├── deformer/              ← Generic deformation engine
│   ├── classifier/            ← Shape classification
│   ├── measurement/           ← Measurement extraction
│   └── ...
│
├── assets/
│   └── templates/             ← All template assets
│       ├── GT_001/            ← Gold Template v1 (rectangular)
│       ├── GT_002/            ← Future: round frames
│       ├── GT_003/            ← Future: aviator
│       └── ...
│
├── template_registry/         ← Template loading and management
│   ├── __init__.py
│   ├── registry.py            ← Path management, discovery
│   └── loader.py              ← Template bundle loading
│
├── scripts/
│   ├── validate_templates.py
│   └── extract_gold_template.py
│
└── tests/
    └── test_template_loading.py
```

### Template Structure (GT_001)

```
assets/templates/GT_001/
│
├── README.md                  ← Template-specific documentation
│
├── geometry/
│   └── template.glb           ← Runtime mesh (renamed from Gold_Template_Runtime.glb)
│
├── metadata/
│   ├── template.json          ← Template ID, version, description, objects
│   ├── parameter_schema.json  ← Parameter definitions and ranges
│   ├── measurements.json      ← Default measurements
│   ├── constraints.json       ← Deformation constraints
│   ├── topology.json          ← Vertex/face counts, connectivity
│   ├── masks.json             ← Per-object vertex masks
│   ├── landmarks.json         ← Anatomical landmark positions
│   ├── region_masks.json      ← Regional segmentation
│   └── scale_config.json      ← Scale factors and units
│
├── deformation/
│   ├── basis.npz              ← Deformation basis functions (main data)
│   ├── basis_metadata.json    ← Basis dimensions, generation params
│   ├── _part_order.json       ← Object ordering for basis
│   ├── _unified_verts.npy     ← Unified vertex array
│   ├── _unified_faces.npy     ← Unified face array
│   └── build_basis.py         ← Basis generation script (reference)
│
└── source/
    └── Gold_Template_Rhino_v1_scaled.3dm  ← Rhino source (authoring only)
```

## File Descriptions

### Geometry Files

#### `geometry/template.glb`
- Runtime mesh in GLB format
- Contains all objects (Frame, Lens_L, Lens_R, Rim_L, Rim_R, Bridge, Temple_L, Temple_R)
- Used by deformation engine for initial mesh state
- **Standard name across all templates:** Always `template.glb`

### Metadata Files

#### `metadata/template.json`
Core template information:
```json
{
  "template_id": "GT_001",
  "template_version": "1.1.0",
  "topology_version": "1",
  "basis_version": "1.1",
  "metadata_version": "1",
  "description": "...",
  "shape_category": "rectangular",
  "objects": {
    "Frame": {...},
    "Lens_L": {...}
  }
}
```

#### `metadata/parameter_schema.json`
Defines deformation parameters:
```json
{
  "parameters": {
    "frame_width": {
      "type": "float",
      "unit": "mm",
      "min": 120,
      "max": 160,
      "default": 140
    }
  }
}
```

#### `metadata/landmarks.json`
Anatomical landmark positions:
```json
{
  "landmarks": {
    "bridge_center": {"index": 42, "position": [0, 0, 0]},
    "temple_left_hinge": {"index": 156, "position": [...]},
    ...
  }
}
```

#### `metadata/masks.json`
Vertex masks for each object:
```json
{
  "masks": {
    "Frame": {"indices": [0, 1, 2, ...]},
    "Lens_L": {"indices": [100, 101, ...]},
    ...
  }
}
```

#### `metadata/constraints.json`
Deformation constraints:
```json
{
  "symmetry": {
    "pairs": [
      {"left": "Lens_L", "right": "Lens_R"}
    ]
  },
  "geometry": {
    "min_edge_length": 0.1,
    "smoothness_weight": 0.5
  }
}
```

### Deformation Files

#### `deformation/basis.npz`
Numpy archive containing deformation basis functions:
- `Frame_basis`: (n_frame_verts, 3, n_modes)
- `Lens_L_basis`: (n_lens_verts, 3, n_modes)
- etc.

Generated from PCA or other methods. Enables fast parametric deformation.

#### `deformation/basis_metadata.json`
Information about basis generation:
```json
{
  "generation_method": "PCA",
  "n_modes": 10,
  "variance_explained": 0.95,
  "generated_date": "2026-01-15"
}
```

### Source Files

#### `source/*.3dm`
Rhino source files used for authoring. **NOT used at runtime.**

These are reference files for:
- Geometry editing
- Basis regeneration
- Design iteration

Production code should NEVER load .3dm files.

## Template Registry API

### Basic Usage

```python
from template_registry import TemplateBundle

# Load a template
bundle = TemplateBundle.load("GT_001")

# Access metadata
print(bundle.metadata.template_id)
print(bundle.metadata.template_version)

# Access deformation data
landmarks = bundle.landmarks
masks = bundle.masks
constraints = bundle.constraints
parameters = bundle.parameters

# Load basis functions
basis = bundle.get_basis()
frame_basis = basis['Frame_basis']

# Get mesh data
vertices = bundle.get_vertices()  # (n_verts, 3)
faces = bundle.get_faces()        # (n_faces, 3)
```

### Advanced Usage

```python
from template_registry import (
    list_templates,
    get_template_assets,
    get_template_version,
    validate_template_structure
)

# List all templates
templates = list_templates()  # ['GT_001', 'GT_002', ...]

# Get asset paths
assets = get_template_assets("GT_001")
glb_path = assets["glb"]
basis_path = assets["basis"]

# Check version
version = get_template_version("GT_001")
print(f"Template version: {version['template_version']}")
print(f"Basis version: {version['basis_version']}")

# Validate structure
validation = validate_template_structure("GT_001")
if not all(validation.values()):
    print("Missing files:", [k for k, v in validation.items() if not v])
```

### Validation

```python
bundle = TemplateBundle.load("GT_001")
report = bundle.validate()

if report['valid']:
    print("✓ Template is valid")
else:
    print("Errors:")
    for error in report['errors']:
        print(f"  - {error}")
    
    print("Warnings:")
    for warning in report['warnings']:
        print(f"  - {warning}")
```

## Integration with Deformation Engine

### Current Pattern (Old)

```python
# WRONG: Hard-coded paths
engine = DeformationEngine()
engine.load_template("path/to/GT_001.glb")
```

### New Pattern

```python
# RIGHT: Template bundle
from template_registry import TemplateBundle
from backend.deformer.engine import DeformationEngine

# Load template
template = TemplateBundle.load("GT_001")

# Initialize engine with template
engine = DeformationEngine(template=template)

# Deform with parameters
result = engine.deform(params={
    "frame_width": 145,
    "lens_width": 58,
    "lens_height": 37,
    "bridge_width": 18,
    "temple_length": 155,
})
```

### Engine Should Accept Template Bundle

Modify `backend/deformer/engine.py`:

```python
class DeformationEngine:
    def __init__(self, template: TemplateBundle):
        self.template = template
        self.metadata = template.metadata
        
        # Load geometry
        self.V0 = template.get_vertices()
        self.faces = template.get_faces()
        
        # Load deformation data
        self.masks = template.masks
        self.landmarks = template.landmarks
        self.constraints = template.constraints
        self.basis = template.get_basis()
        
        # Initialize constraints
        self._init_constraints()
    
    def deform(self, params: dict) -> dict:
        # Use self.template data for deformation
        ...
```

## Production Pipeline Flow

```
User uploads images
        ↓
Vision model extracts features
        ↓
Parameter predictor → params
Template retriever → template_id
        ↓
template = TemplateBundle.load(template_id)
        ↓
engine = DeformationEngine(template)
result = engine.deform(params)
        ↓
Constraint validation
Quality checking
        ↓
GLB export
        ↓
Browser rendering
```

## Template Lifecycle

### 1. Creation
- Design in Rhino/Blender
- Export runtime GLB
- Generate basis functions
- Create metadata JSON files
- Place in `assets/templates/GT_XXX/`

### 2. Version Control
- Track version in `template.json`
- Use Git for metadata and small files
- Use Git LFS or S3 for large binaries (GLB, basis.npz)

### 3. Validation
```bash
python scripts/validate_templates.py
```

### 4. Testing
```bash
pytest tests/test_template_loading.py -v
```

### 5. Deployment
- Templates loaded from `assets/templates/` in development
- Templates fetched from S3/MinIO in production
- Version pinning ensures consistency

## Multi-Template Future

When you have multiple templates:

```python
# Shape classification
shape = shape_classifier.predict(image)  # "rectangular"

# Template retrieval
template_id = template_matcher.find_best(shape, features)  # "GT_001"

# Load appropriate template
template = TemplateBundle.load(template_id)

# Deform
engine = DeformationEngine(template)
result = engine.deform(params)
```

Supports scaling from 1 template to 1000+ templates without architecture changes.

## Don't Do This

❌ **Don't put template data in engine code:**
```python
# WRONG
class DeformationEngine:
    def __init__(self):
        self.template_path = "templates/GT_001.glb"  # Hard-coded!
```

❌ **Don't copy Gold Template engine into main engine:**
```python
# WRONG
backend/deformer/
    engine.py
    gold_template_engine.py  # Duplicate logic!
```

❌ **Don't scatter template files:**
```python
# WRONG
templates/GT_001.glb
metadata/GT_001_masks.json
basis/GT_001_basis.npz
```

❌ **Don't use Rhino files at runtime:**
```python
# WRONG
import rhinoscriptsyntax as rs
mesh = rs.LoadFile("GT_001.3dm")  # NO! Use GLB + metadata
```

## Do This

✅ **Use template registry:**
```python
template = TemplateBundle.load("GT_001")
```

✅ **Keep templates self-contained:**
```
GT_001/
    geometry/
    metadata/
    deformation/
```

✅ **Version everything:**
```json
{
  "template_version": "1.1.0",
  "topology_version": "1",
  "basis_version": "1.1"
}
```

✅ **Validate before use:**
```python
report = template.validate()
assert report['valid']
```

## Next Steps

1. **Extract Gold_Template.zip:**
   ```bash
   python scripts/extract_gold_template.py path/to/Gold_Template.zip --extract
   ```

2. **Validate GT_001:**
   ```bash
   python scripts/validate_templates.py
   ```

3. **Run tests:**
   ```bash
   pytest tests/test_template_loading.py -v
   ```

4. **Connect to engine:**
   - Modify `backend/deformer/engine.py` to accept `TemplateBundle`
   - Remove hard-coded template paths
   - Use `template.get_vertices()`, `template.get_basis()`, etc.

5. **Integration test:**
   ```python
   template = TemplateBundle.load("GT_001")
   engine = DeformationEngine(template)
   result = engine.deform(params={...})
   ```

## Summary

The template architecture provides:
- ✅ Clean separation between engine and data
- ✅ Version control for templates
- ✅ Self-contained template assets
- ✅ Standardized loading interface
- ✅ Validation and testing
- ✅ Scalability to many templates
- ✅ Production-ready structure

This architecture ensures that as you add more templates (round, aviator, cat-eye, etc.), the system scales cleanly without modifying core engine code.

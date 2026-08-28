# Template Structure - Visual Overview

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEFORMATION SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌─────────────────────┐             ┌─────────────────────┐
│  DEFORMATION ENGINE │             │  TEMPLATE REGISTRY  │
│  (Generic Code)     │◄────uses────│  (Data Management)  │
│                     │             │                     │
│ backend/deformer/   │             │ template_registry/  │
│   engine.py         │             │   registry.py       │
│   constraints.py    │             │   loader.py         │
│   quality_checker.py│             │                     │
└─────────────────────┘             └──────────┬──────────┘
                                               │
                                               │ loads
                                               │
                                               ▼
                              ┌────────────────────────────┐
                              │    TEMPLATE ASSETS         │
                              │  (Versioned Design Data)   │
                              │                            │
                              │  assets/templates/         │
                              └────────────┬───────────────┘
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 │                         │                         │
                 ▼                         ▼                         ▼
         ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
         │    GT_001     │        │    GT_002     │        │    GT_003     │
         │  (Rectangle)  │        │    (Round)    │        │   (Aviator)   │
         │               │        │   [Future]    │        │   [Future]    │
         └───────────────┘        └───────────────┘        └───────────────┘
```

## Template Bundle Structure

```
GT_001/  ← Single self-contained template
│
├── 📄 README.md
│   └─ Documentation for this template
│
├── 📁 geometry/
│   └── template.glb                    ← Runtime mesh (GLB format)
│
├── 📁 metadata/
│   ├── template.json                   ← ID, version, objects
│   ├── parameter_schema.json           ← Deformation parameters
│   ├── measurements.json               ← Default measurements
│   ├── constraints.json                ← Deformation constraints
│   ├── topology.json                   ← Mesh topology info
│   ├── masks.json                      ← Per-object vertex masks
│   ├── landmarks.json                  ← Anatomical landmarks
│   ├── region_masks.json               ← Regional segmentation
│   └── scale_config.json               ← Scale factors
│
├── 📁 deformation/
│   ├── basis.npz                       ← Deformation basis (main)
│   ├── basis_metadata.json             ← Basis generation info
│   ├── _part_order.json                ← Object ordering
│   ├── _unified_verts.npy              ← Unified vertices
│   ├── _unified_faces.npy              ← Unified faces
│   └── build_basis.py                  ← Basis build script
│
└── 📁 source/
    └── Gold_Template_Rhino_v1_scaled.3dm  ← Rhino source (authoring)
```

## Data Flow

### Loading Flow

```
User Code
    │
    │ TemplateBundle.load("GT_001")
    ▼
Template Registry
    │
    │ 1. Find GT_001 directory
    │ 2. Validate structure
    │ 3. Load JSON metadata
    ▼
Template Bundle
    │
    │ - metadata: template info
    │ - landmarks: positions
    │ - masks: vertex indices
    │ - constraints: rules
    │ - basis: deformation data
    ▼
Deformation Engine
    │
    │ engine.deform(params)
    ▼
Deformed Mesh
```

### Production Pipeline Flow

```
User Uploads Images
        │
        ▼
┌───────────────────┐
│ Vision Model      │  Extracts features from images
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Parameter         │  Predicts measurements
│ Predictor         │  → frame_width, lens_width, etc.
└────────┬──────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌───────────────────┐  ┌───────────────────┐
│ Template          │  │ Parameters        │
│ Retriever         │  │                   │
│ → "GT_001"        │  │ {"frame_width":   │
└────────┬──────────┘  │  145, ...}        │
         │             └───────┬───────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Template Registry    │
         │ template = load(id)  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Deformation Engine   │
         │ engine = Engine(tmpl)│
         │ result = deform(p)   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Quality Validation   │
         │ - Topology check     │
         │ - Constraint check   │
         │ - Measurement check  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ GLB Export           │
         │ customized.glb       │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Browser Rendering    │
         │ 3D Try-On            │
         └──────────────────────┘
```

## Template Versioning

```
GT_001/metadata/template.json
{
  "template_id": "GT_001",
  "template_version": "1.1.0",        ← Overall version
  "topology_version": "1",             ← Mesh structure version
  "basis_version": "1.1",              ← Deformation basis version
  "metadata_version": "1"              ← Schema version
}

Version Changes:
├─ template_version (1.0.0 → 1.1.0)
│  └─ Any update to the template
│
├─ topology_version (1 → 2)
│  └─ Vertex/face count changes
│  └─ Invalidates old basis functions
│
├─ basis_version (1.0 → 1.1)
│  └─ Deformation basis regenerated
│  └─ Same topology, different modes
│
└─ metadata_version (1 → 2)
   └─ JSON schema changes
   └─ New fields added/removed
```

## Key Components

### Template Registry (`template_registry/`)

```python
# registry.py
┌─────────────────────────────────┐
│ get_template_dir(id)            │  → Path to template
│ get_template_assets(id)         │  → Dict of all paths
│ list_templates()                │  → Available template IDs
│ get_template_version(id)        │  → Version info
│ validate_template_structure(id) │  → File existence check
└─────────────────────────────────┘

# loader.py
┌─────────────────────────────────┐
│ TemplateBundle                  │
│   .load(id)                     │  → Load template
│   .get_basis()                  │  → Get basis arrays
│   .get_vertices()               │  → Get vertices
│   .get_faces()                  │  → Get faces
│   .validate()                   │  → Validate bundle
│   .metadata                     │  → Template metadata
│   .landmarks                    │  → Landmark positions
│   .masks                        │  → Vertex masks
│   .constraints                  │  → Constraints
│   .parameters                   │  → Parameter schema
└─────────────────────────────────┘
```

### Deformation Engine (`backend/deformer/`)

```python
# engine.py (Updated to use templates)
┌─────────────────────────────────┐
│ DeformationEngine(template)     │
│                                 │
│ Inputs:                         │
│   - template: TemplateBundle    │
│                                 │
│ Uses:                           │
│   - template.get_vertices()     │
│   - template.get_basis()        │
│   - template.masks              │
│   - template.landmarks          │
│   - template.constraints        │
│                                 │
│ Methods:                        │
│   - deform(params) → result     │
│   - validate_params(params)     │
│   - apply_constraints(mesh)     │
└─────────────────────────────────┘
```

## File Formats

### JSON Metadata Files

```json
// template.json
{
  "template_id": "GT_001",
  "template_version": "1.1.0",
  "description": "...",
  "objects": {
    "Frame": {...},
    "Lens_L": {...}
  }
}

// landmarks.json
{
  "landmarks": {
    "bridge_center": {
      "index": 42,
      "position": [0.0, 0.0, 0.0]
    }
  }
}

// masks.json
{
  "masks": {
    "Frame": {
      "indices": [0, 1, 2, ...]
    }
  }
}

// constraints.json
{
  "symmetry": {
    "pairs": [
      {"left": "Lens_L", "right": "Lens_R"}
    ]
  },
  "geometry": {
    "min_edge_length": 0.1
  }
}
```

### Binary Data Files

```
basis.npz (NumPy archive)
├─ Frame_basis: (n_verts, 3, n_modes)
├─ Lens_L_basis: (n_verts, 3, n_modes)
├─ Lens_R_basis: (n_verts, 3, n_modes)
└─ ...

_unified_verts.npy (NumPy array)
└─ (total_vertices, 3)

_unified_faces.npy (NumPy array)
└─ (total_faces, 3)

template.glb (GL Transmission Format)
└─ Binary 3D mesh with materials
```

## Usage Patterns

### Pattern 1: Load and Inspect

```python
from template_registry import TemplateBundle

# Load
template = TemplateBundle.load("GT_001")

# Inspect
print(template.metadata.template_id)      # GT_001
print(template.metadata.template_version) # 1.1.0
print(len(template.landmarks['landmarks'])) # 19+

# Validate
report = template.validate()
print(report['valid'])  # True/False
```

### Pattern 2: Use with Engine

```python
from template_registry import TemplateBundle
from backend.deformer.engine import DeformationEngine

# Load template
template = TemplateBundle.load("GT_001")

# Create engine
engine = DeformationEngine(template=template)

# Deform
result = engine.deform({
    "frame_width": 145,
    "lens_width": 58,
    # ...
})

# Use result
vertices = result["vertices"]
faces = result["faces"]
```

### Pattern 3: Multiple Templates

```python
from template_registry import list_templates, TemplateBundle

# List all
templates = list_templates()  # ['GT_001', 'GT_002', ...]

# Load multiple
bundles = {}
for tid in templates:
    bundles[tid] = TemplateBundle.load(tid)

# Select best
template_id = classifier.predict(image)  # "GT_001"
template = bundles[template_id]
```

## Validation Workflow

```
Template Loading
       │
       ▼
┌──────────────────┐
│ Structure Check  │  Do required files exist?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Metadata Check   │  Is template.json valid?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Landmark Check   │  Are landmarks defined?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Object Check     │  Are required objects present?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Basis Check      │  Can basis be loaded?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Report           │  valid, errors, warnings
└──────────────────┘
```

## Scaling to Multiple Templates

```
Current State:
assets/templates/
└── GT_001/         Rectangle frame

Future State:
assets/templates/
├── GT_001/         Rectangle
├── GT_002/         Round
├── GT_003/         Aviator
├── GT_004/         Cat-eye
├── GT_005/         Clubmaster
└── ...

Code remains identical:
template = TemplateBundle.load(template_id)
engine = DeformationEngine(template)
```

## Summary

### Core Principles

1. **Separation:** Engine code ≠ Template data
2. **Versioning:** All templates are versioned
3. **Self-Contained:** Each template has all its data
4. **Standard Interface:** Same API for all templates
5. **Validation:** Automatic integrity checking

### Benefits

✅ Clean architecture
✅ Easy to add templates
✅ Version control
✅ Reproducible results
✅ Testable
✅ Scalable

### Next Steps

1. Extract Gold_Template.zip → `assets/templates/GT_001/`
2. Validate with `python scripts/validate_templates.py`
3. Update engine to use `TemplateBundle`
4. Test integration
5. Deploy to pipeline

---

**Current Status:** Structure ready, awaiting Gold_Template.zip extraction

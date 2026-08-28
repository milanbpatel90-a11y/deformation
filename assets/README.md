# Assets Directory

This directory contains versioned assets for the deformation system, organized by type.

## Directory Structure

```
assets/
└── templates/          ← Template assets (geometry, metadata, deformation data)
    ├── GT_001/         ← Gold Template v1 (rectangular)
    ├── GT_002/         ← Future templates...
    └── ...
```

## Templates

Templates are self-contained asset bundles containing:
- **Geometry:** Runtime GLB mesh
- **Metadata:** Parameters, constraints, topology
- **Deformation:** Basis functions, masks, landmarks
- **Source:** Authoring files (Rhino, etc.)

### Current Templates

#### GT_001 - Gold Template
- **Status:** Awaiting Gold_Template.zip extraction
- **Shape:** Rectangular
- **Version:** 1.1.0
- **Description:** Gold Template with parametric deformation

See `templates/GT_001/README.md` for details.

## Usage

### Loading Templates

```python
from template_registry import TemplateBundle

# Load template
template = TemplateBundle.load("GT_001")

# Access data
vertices = template.get_vertices()
basis = template.get_basis()
landmarks = template.landmarks
```

### Listing Templates

```python
from template_registry import list_templates

templates = list_templates()
print(templates)  # ['GT_001', 'GT_002', ...]
```

## Adding New Templates

1. Create template directory:
   ```
   assets/templates/NEW_ID/
   ├── geometry/
   ├── metadata/
   ├── deformation/
   └── source/
   ```

2. Add required files (see `GT_001/` for structure)

3. Validate:
   ```bash
   python scripts/validate_templates.py
   ```

4. Test:
   ```bash
   pytest tests/test_template_loading.py -v
   ```

## Version Control

### Git
- Metadata JSON files: committed to Git
- Documentation: committed to Git

### Git LFS / S3
- Large binary files (GLB, NPZ, NPY): use Git LFS or object storage
- Recommended for files > 1 MB

### Production
- Templates loaded from S3/MinIO in production
- Version pinning ensures consistency
- Registry maintains mapping between IDs and storage locations

## Documentation

See [`TEMPLATE_ARCHITECTURE.md`](../TEMPLATE_ARCHITECTURE.md) for complete architecture documentation.

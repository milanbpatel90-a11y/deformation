# Template Issue Fix - Quick Guide

## Current Problem

**Error:** `Descriptor references unknown geometry part 'LeftTemple' for hinge:left`

**Root Cause:** The template GLB files have generic mesh names (like `Plane_glasses_mat_0`) but the descriptor JSON files expect specific names (like `LeftTemple`, `RightTemple`, etc.).

**Example:**
- **Template GLB has:** `['Plane_glasses_mat_0', 'Plane.002_glasses_mat_0', 'Cube.002_glass_mat_0', 'Plane.001_glasses_mat_0']`
- **Descriptor expects:** `['Frame', 'Bridge', 'LeftTemple', 'RightTemple', 'LeftLens', 'RightLens', 'LeftRim', 'RightRim']`

This mismatch causes the pipeline to fail.

---

## Quick Fix (Use Old Pipeline Temporarily)

The easiest solution for testing is to use the old deformation engine that doesn't require descriptors:

### Option 1: Use Old MeshDeformer Temporarily

Edit `backend/pipeline/deformation_pipeline.py` and look for where it creates the deformation context. Switch to using the old `MeshDeformer` from `backend/deformer/engine.py` instead of the descriptor-based pipeline.

**File:** `backend/deformer/engine.py`
**Class:** `MeshDeformer` 

This class works with **any** template GLB without needing descriptors.

### Option 2: Skip Descriptor Loading

In `backend/deformer/descriptor_loader.py`, add a fallback when geometry parts aren't found:

```python
def _get_geometry(self, geometry: dict, part: str, label: str) -> trimesh.Trimesh:
    """Get a named geometry part or raise ValueError."""
    geom = geometry.get(part)
    if geom is None:
        # FALLBACK: Try to find a mesh that might be this part
        # For now, just skip descriptor loading and use simple deformation
        import warnings
        warnings.warn(f"Mesh '{part}' not found for {label}, skipping descriptor-based deformation")
        return None  # Or return a dummy mesh
    return geom
```

---

## Proper Fix (Template Preparation)

To properly fix this, you need to **prepare templates** with correct mesh names.

### Step 1: Open Template in Blender

1. Download Blender: https://www.blender.org/
2. Open your template GLB (e.g., `templates/geometric_metal.glb`)
3. View the Object Outliner

### Step 2: Rename Meshes

Rename each mesh to match expected names:

| Old Name (Generic) | New Name (Expected) | Purpose |
|--------------------|---------------------|---------|
| Plane_glasses_mat_0 | Frame | Main frame body |
| Plane.001_glasses_mat_0 | Bridge | Bridge between lenses |
| Plane.002_glasses_mat_0 | LeftTemple | Left temple arm |
| Plane.003_glasses_mat_0 | RightTemple | Right temple arm |
| Cube.002_glass_mat_0 | LeftLens | Left lens |
| Cube.003_glass_mat_0 | RightLens | Right lens |
| (if exists) | LeftRim | Left lens rim |
| (if exists) | RightRim | Right lens rim |
| (if exists) | NosePads | Nose pads |

### Step 3: Export from Blender

1. File → Export → glTF 2.0 (.glb)
2. Save to `templates/processed/template_name.glb`
3. Settings:
   - Format: GLB Binary
   - Include: Selected Objects (or Visible Objects)
   - Transform: +Y Up
   - Geometry: Apply Modifiers

### Step 4: Create Descriptor JSON

Create a descriptor file at `templates/descriptors/template_name.json`:

```json
{
  "version": "1.0",
  "template_name": "geometric_metal",
  "vertex_groups": {
    "frame": ["Frame"],
    "bridge": ["Bridge"],
    "left_temple": ["LeftTemple"],
    "right_temple": ["RightTemple"],
    "left_lens": ["LeftLens"],
    "right_lens": ["RightLens"],
    "left_rim": ["LeftRim"],
    "right_rim": ["RightRim"],
    "nose_pads": ["NosePads"]
  },
  "hinges": {
    "left": {
      "anchor_part": "LeftTemple",
      "pivot": [-70.0, 0.0, 0.0],
      "axis": [0.0, 1.0, 0.0]
    },
    "right": {
      "anchor_part": "RightTemple",
      "pivot": [70.0, 0.0, 0.0],
      "axis": [0.0, 1.0, 0.0]
    }
  },
  "lens_planes": {
    "left": {
      "origin": [-35.0, 0.0, 0.0],
      "normal": [0.0, 0.0, 1.0],
      "up": [0.0, 1.0, 0.0]
    },
    "right": {
      "origin": [35.0, 0.0, 0.0],
      "normal": [0.0, 0.0, 1.0],
      "up": [0.0, 1.0, 0.0]
    }
  },
  "template_dims": {
    "frame_width": 145.0,
    "lens_width": 54.0,
    "lens_height": 50.0,
    "bridge_width": 18.0,
    "temple_length": 140.0,
    "rim_thickness": 1.2,
    "nose_pad_distance": 15.0,
    "nose_pad_height": 18.0,
    "nose_pad_angle": 15.0,
    "temple_curve_angle": 28.0
  }
}
```

### Step 5: Update Registry

Add template to `templates/registry.json`:

```json
[
  {
    "name": "geometric_metal",
    "glb_path": "processed/geometric_metal.glb",
    "descriptor_path": "descriptors/geometric_metal.json",
    "metadata_path": "metadata/geometric_metal.json",
    "thumbnail_path": "thumbnails/geometric_metal.png"
  }
]
```

---

## Alternative: Use Designer Tool

There's a `designer/` folder in your project. This might have tools for preparing templates.

Check:
- `designer/template_prep.py`
- `designer/mesh_editor.py`
- Or run: `python designer/prepare_template.py`

---

## Testing After Fix

1. Restart backend: Kill uvicorn, restart
2. Try uploading images again
3. Should now work without descriptor errors

---

## Recommended Short-Term Solution

**For testing right now**, modify the pipeline to use the old `MeshDeformer` temporarily:

**File:** `backend/pipeline/deformation_pipeline.py`

Find where it loads descriptor and replace with:

```python
# Temporarily use old deformer without descriptors
from backend.deformer.engine import MeshDeformer

deformer = MeshDeformer(
    template_scene=template_scene,
    template_dims=template_dims,
)
deformed_scene = deformer.deform(measurements)
```

This bypasses descriptor loading and uses simple scaling deformation.

---

## Long-Term Solution (Week 1 Implementation)

This is exactly why **Week 1 focuses on deformation quality**:

1. **Week 1-2:** Build better deformers (Bridge, Temple, Lens)
2. **Week 3:** Better template preparation workflow
3. **Week 4:** Testing with properly prepared templates

The current error shows that templates need proper structure before the advanced deformation can work.

---

## Summary

**Problem:** Templates not properly prepared
**Quick Fix:** Use old MeshDeformer (no descriptors)
**Proper Fix:** Rename meshes in Blender + create descriptors
**Long-Term:** Week 1-4 implementation plan addresses this

For now, use the quick fix to test the system. Then implement proper template preparation as part of the quality improvement phase.

---

**Next:** Once you can generate models (even with simple deformation), you can start Week 1 improvements to make deformation look better.

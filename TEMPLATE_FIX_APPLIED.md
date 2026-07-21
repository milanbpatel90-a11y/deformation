# Template Fix Applied - Quick Reference

## What Was The Problem?

Your new `geometric_metal.glb` template has generic Blender mesh names:
- `Plane_glasses_mat_0`
- `Plane.001_glasses_mat_0`
- `Plane.002_glasses_mat_0`  
- `Cube.002_glass_mat_0`

But the deformer code expected structured names like:
- `Frame`
- `Bridge`
- `LeftTemple`
- `RightTemple`
- `LeftLens`
- `RightLens`

## What Was Fixed?

### 1. Created Descriptor Mapping

**File:** `templates/descriptors/geometric_metal.json`

This file maps the generic mesh names to the expected vertex groups:

```json
{
  "vertex_groups": {
    "frame": ["Plane_glasses_mat_0", "Plane.001_glasses_mat_0"],
    "bridge": ["Plane.001_glasses_mat_0"],
    "left_temple": ["Plane_glasses_mat_0"],
    "right_temple": ["Plane.002_glasses_mat_0"],
    "left_lens": ["Cube.002_glass_mat_0"],
    "right_lens": ["Cube.002_glass_mat_0"],
    "left_rim": ["Plane_glasses_mat_0"],
    "right_rim": ["Plane.002_glasses_mat_0"]
  },
  ...
}
```

**NOTE:** This mapping is a **best guess** based on your mesh names. The deformation might not look perfect because:
- Multiple parts map to the same mesh (e.g., frame, left_temple, left_rim all point to `Plane_glasses_mat_0`)
- The actual geometry might not match what each group should contain

### 2. Updated Registry

**File:** `templates/registry.json`

Added entry for `geometric_metal` template linking to the descriptor.

### 3. Added Fallback in Pipeline

**File:** `backend/pipeline/deformation_pipeline.py`

Added try-except around descriptor loading to fall back to simple deformation if descriptor fails.

## Current Status

✅ **System should now work** - Backend will load your template with the descriptor mapping

⚠️ **Deformation quality may be poor** because:
1. The mesh name mapping is approximate
2. The actual mesh geometry may not match expected parts
3. Multiple vertex groups point to the same mesh

## Next Steps

### For Better Results

You need to properly prepare the template GLB in Blender:

#### Step 1: Open in Blender
```
File → Import → glTF 2.0 (.glb)
Select: templates/geometric_metal.glb
```

#### Step 2: Identify & Separate Parts

Look at your model and identify:
- Which mesh is the frame body?
- Which mesh is the bridge?
- Which meshes are temples (left & right)?
- Which meshes are lenses (left & right)?
- Which meshes are rims (left & right)?

#### Step 3: Rename Meshes

In the Outliner (top right):
```
Old Name                  → New Name
────────────────────────────────────────
Plane_glasses_mat_0       → Frame
Plane.001_glasses_mat_0   → Bridge
Plane.002_glasses_mat_0   → RightTemple
Plane.003_glasses_mat_0   → LeftTemple  (if exists)
Cube.002_glass_mat_0      → LeftLens
Cube.003_glass_mat_0      → RightLens   (if exists)
... etc ...
```

**Important:** Make sure each part is a **separate mesh**. If temple and frame are the same mesh, you need to:
1. Select mesh in Edit mode
2. Select vertices for temple
3. Press P → Selection (to separate)
4. Rename the new mesh

#### Step 4: Export Properly Named GLB

```
File → Export → glTF 2.0 (.glb)
Settings:
  - Format: GLB Binary
  - Include: Visible Objects
  - Transform: +Y Up
  - Geometry: Apply Modifiers
Save to: templates/geometric_metal.glb
```

#### Step 5: Update Descriptor

Once mesh names are proper, update the descriptor to use simple names:

```json
{
  "vertex_groups": {
    "frame": ["Frame"],
    "bridge": ["Bridge"],
    "left_temple": ["LeftTemple"],
    "right_temple": ["RightTemple"],
    "left_lens": ["LeftLens"],
    "right_lens": ["RightLens"],
    "left_rim": ["LeftRim"],
    "right_rim": ["RightRim"]
  }
}
```

## Testing Now

Try generating a 3D model again:

1. Refresh browser (http://localhost:8001)
2. Upload glasses images
3. Click "Generate 3D Mesh"

**Expected:** Should work, but deformation quality may be basic

**If it still fails:** Check backend logs for new error messages

## Expected Deformation Quality

With current mapping (multiple groups → same mesh):
- ⚠️ **Bridge deformation**: May affect entire frame
- ⚠️ **Temple deformation**: May affect frame too
- ⚠️ **Lens sizing**: Should work
- ⚠️ **Colors**: Should work
- ⚠️ **Overall shape**: Generic scaling only

With proper mesh separation (future):
- ✅ **Bridge deformation**: Only affects bridge
- ✅ **Temple deformation**: Natural temple scaling
- ✅ **Lens sizing**: Proper lens adaptation
- ✅ **Colors**: Material-aware
- ✅ **Overall shape**: Realistic custom deformation

## Quick Check: What Mesh Names Do You Have?

Run this to verify your current mesh names:

```bash
python -c "import trimesh; scene = trimesh.load('templates/geometric_metal.glb'); print('\\n'.join(scene.geometry.keys()))"
```

Compare output to expected names:
```
Expected (ideal):        Current (generic):
- Frame                  - Plane_glasses_mat_0
- Bridge                 - Plane.001_glasses_mat_0
- LeftTemple             - Plane.002_glasses_mat_0
- RightTemple            - Cube.002_glass_mat_0
- LeftLens
- RightLens
- LeftRim
- RightRim
- NosePads (optional)
```

## Summary

| Component | Status | Next Action |
|-----------|--------|-------------|
| Descriptor | ✅ Created | Refine after mesh separation |
| Registry | ✅ Updated | No action needed |
| Pipeline | ✅ Fallback added | No action needed |
| Mesh Names | ⚠️ Generic | Rename in Blender |
| Mesh Separation | ⚠️ Not separated | Separate in Blender |

**Can test now:** ✅ Yes, system should work
**Quality:** ⚠️ Basic (generic scaling)
**For production:** ❌ Need proper mesh preparation

---

## Alignment with Week 1 Plan

This issue is exactly what **Week 1-2** addresses:

- Week 1: Improve bridge/temple/lens deformation
- Week 2: Better template preparation
- Week 3: Better materials
- Week 4: Testing with quality templates

Your current workaround lets you test the system now. Week 1-2 implementation will require properly prepared templates to show full improvement.

---

**System Status:** ✅ Ready to test (with basic deformation)
**Next:** Try generating a model, then prepare proper templates for better results

# Proper Template Fix - Complete Guide

## Current Situation

**Your GLB has generic Blender names:**
```
'Plane_glasses_mat_0'      -> 61872 vertices (likely frame/temples combined)
'Plane.002_glasses_mat_0'  -> 2385 vertices  (likely right temple)
'Cube.002_glass_mat_0'     -> 6916 vertices  (likely lenses)
'Plane.001_glasses_mat_0'  -> 2385 vertices  (likely left temple or bridge)
```

**Descriptor loader expects:**
```
'Frame'
'Bridge'
'LeftRim'
'RightRim'
'LeftLens'
'RightLens'
'LeftTemple'
'RightTemple'
```

**Required descriptor keys:**
```json
{
  "hinges": {...},
  "bridge": {...},
  "rim_loops": {...},
  "temple_pivots": {...},
  "lens_planes": {...},
  "symmetry_plane": {...},
  "deformation_regions": {...}
}
```

---

## Two-Path Solution

### Path A: Quick Fix (Mapping Generic Names) ⚡

**Use this for immediate testing**

Create a descriptor that maps your generic names to expected groups:

```json
{
  "version": "1.0",
  "template_name": "geometric_metal",
  
  "vertex_groups": {
    "frame": ["Plane_glasses_mat_0"],
    "bridge": ["Plane.001_glasses_mat_0"],
    "left_rim": ["Plane_glasses_mat_0"],
    "right_rim": ["Plane.002_glasses_mat_0"],
    "left_lens": ["Cube.002_glass_mat_0"],
    "right_lens": ["Cube.002_glass_mat_0"],
    "left_temple": ["Plane.002_glasses_mat_0"],
    "right_temple": ["Plane.001_glasses_mat_0"]
  },
  
  "hinges": {
    "left": {
      "part": "Plane.002_glasses_mat_0",
      "confidence": 1.0
    },
    "right": {
      "part": "Plane.001_glasses_mat_0",
      "confidence": 1.0
    }
  },
  
  "bridge": {
    "part": "Plane.001_glasses_mat_0"
  },
  
  "rim_loops": {
    "frame": {
      "part": "Plane_glasses_mat_0",
      "confidence": 1.0
    },
    "left_rim": {
      "part": "Plane_glasses_mat_0",
      "confidence": 1.0
    },
    "right_rim": {
      "part": "Plane.002_glasses_mat_0",
      "confidence": 1.0
    }
  },
  
  "temple_pivots": {
    "left": {
      "part": "Plane.002_glasses_mat_0"
    },
    "right": {
      "part": "Plane.001_glasses_mat_0"
    }
  },
  
  "lens_planes": {
    "left": {
      "part": "Cube.002_glass_mat_0"
    },
    "right": {
      "part": "Cube.002_glass_mat_0"
    }
  },
  
  "symmetry_plane": {
    "axis": "X",
    "point": [0.0, 0.0, 0.0],
    "normal": [1.0, 0.0, 0.0]
  },
  
  "deformation_regions": {
    "bridge_region": {
      "part": "Plane.001_glasses_mat_0"
    },
    "left_rim_region": {
      "part": "Plane_glasses_mat_0"
    },
    "right_rim_region": {
      "part": "Plane.002_glasses_mat_0"
    },
    "left_temple_region": {
      "part": "Plane.002_glasses_mat_0"
    },
    "right_temple_region": {
      "part": "Plane.001_glasses_mat_0"
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

**Save as:** `templates/descriptors/geometric_metal.json`

**Expected result:** ⚠️ System works but deformation quality is poor (multiple parts map to same mesh)

---

### Path B: Proper Fix (Rename in Blender) ✅

**Use this for production quality**

#### Step 1: Open GLB in Blender

```
1. Launch Blender
2. File → Import → glTF 2.0 (.glb/.gltf)
3. Select: templates/geometric_metal.glb
4. Import
```

#### Step 2: Inspect Current Structure

In the **Outliner** (top-right panel), you'll see your 4 meshes:
- Plane_glasses_mat_0 (61872 verts - huge, likely combined frame+temples)
- Plane.001_glasses_mat_0 (2385 verts - bridge or temple)
- Plane.002_glasses_mat_0 (2385 verts - temple)
- Cube.002_glass_mat_0 (6916 verts - lenses, probably both)

#### Step 3: Separate Combined Meshes

**If frame and temples are one mesh (Plane_glasses_mat_0 with 61k verts):**

1. Select `Plane_glasses_mat_0` in outliner
2. Tab to Edit Mode
3. Select vertices for left temple
4. Press P → Selection (separate)
5. Repeat for right temple
6. Repeat for bridge if needed
7. What remains is the frame

**If lenses are one mesh (Cube.002_glass_mat_0):**

1. Select the lens mesh
2. Tab to Edit Mode
3. Select vertices for left lens only
4. Press P → Selection
5. Exit Edit Mode
6. Now you have two lens objects

#### Step 4: Rename All Meshes

In the Outliner, double-click each mesh name and rename:

```
Old Name                     → New Name
──────────────────────────────────────────
Plane_glasses_mat_0          → Frame
Plane.001_glasses_mat_0      → Bridge
Plane.002_glasses_mat_0      → LeftTemple
Plane.003_glasses_mat_0      → RightTemple
Cube.002_glass_mat_0         → LeftLens
Cube.003_glass_mat_0         → RightLens

(If you have rim meshes separately:)
<rim_mesh_1>                 → LeftRim
<rim_mesh_2>                 → RightRim

(If you have nose pads:)
<pad_mesh>                   → NosePads
```

**Critical:** Names must match **exactly** (case-sensitive):
- `Frame` not `frame` or `FRAME`
- `LeftTemple` not `Left_Temple` or `leftTemple`

#### Step 5: Position Check

Make sure your model is centered correctly:
- Frame center should be at X=0
- Left temple at negative X
- Right temple at positive X
- Bridge at X=0, connecting both sides

#### Step 6: Export Properly

```
1. Select all objects (A key)
2. File → Export → glTF 2.0 (.glb/.gltf)
3. Settings:
   - Format: GLB Binary (.glb)
   - Include: Selected Objects
   - Transform: +Y Up
   - Geometry:
     ✅ Apply Modifiers
     ✅ UVs
     ✅ Normals
     ✅ Tangents
     ✅ Vertex Colors
   - Compression: None (for now)
4. Save to: templates/geometric_metal.glb
```

#### Step 7: Verify Export

Run this to confirm names:

```python
import trimesh
scene = trimesh.load(r"C:\Users\Petpooja-607\Desktop\defirmation\templates\geometric_metal.glb", force="scene")
print("Exported mesh names:")
for name in sorted(scene.geometry.keys()):
    print(f"  - {name}")
```

**Expected output:**
```
Exported mesh names:
  - Bridge
  - Frame
  - LeftLens
  - LeftRim
  - LeftTemple
  - RightLens
  - RightRim
  - RightTemple
```

#### Step 8: Create Clean Descriptor

With proper names, descriptor becomes simple:

```json
{
  "version": "1.0",
  "template_name": "geometric_metal",
  
  "vertex_groups": {
    "frame": ["Frame"],
    "bridge": ["Bridge"],
    "left_rim": ["LeftRim"],
    "right_rim": ["RightRim"],
    "left_lens": ["LeftLens"],
    "right_lens": ["RightLens"],
    "left_temple": ["LeftTemple"],
    "right_temple": ["RightTemple"]
  },
  
  "hinges": {
    "left": { "part": "LeftTemple", "confidence": 1.0 },
    "right": { "part": "RightTemple", "confidence": 1.0 }
  },
  
  "bridge": {
    "part": "Bridge"
  },
  
  "rim_loops": {
    "frame": { "part": "Frame", "confidence": 1.0 },
    "left_rim": { "part": "LeftRim", "confidence": 1.0 },
    "right_rim": { "part": "RightRim", "confidence": 1.0 }
  },
  
  "temple_pivots": {
    "left": { "part": "LeftTemple" },
    "right": { "part": "RightTemple" }
  },
  
  "lens_planes": {
    "left": { "part": "LeftLens" },
    "right": { "part": "RightLens" }
  },
  
  "symmetry_plane": {
    "axis": "X",
    "point": [0.0, 0.0, 0.0],
    "normal": [1.0, 0.0, 0.0]
  },
  
  "deformation_regions": {
    "bridge_region": { "part": "Bridge" },
    "left_rim_region": { "part": "LeftRim" },
    "right_rim_region": { "part": "RightRim" },
    "left_temple_region": { "part": "LeftTemple" },
    "right_temple_region": { "part": "RightTemple" }
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

**Save as:** `templates/descriptors/geometric_metal.json`

**Expected result:** ✅ Production-quality deformation

---

## Recommended Workflow

1. **Now:** Use Path A (quick mapping) to test the system
2. **Test:** Generate a few models, verify pipeline works
3. **Later:** Use Path B (proper Blender preparation) for quality
4. **Week 1-2:** Implement better deformers with proper templates

---

## What Each Key Does

| Key | Purpose | Loader Uses It For |
|-----|---------|-------------------|
| `hinges` | Temple rotation points | Auto-compute pivot from mesh geometry |
| `bridge` | Bridge center | Average mesh vertices to find center |
| `rim_loops` | Rim mesh parts | Deform rims to match lens contours |
| `temple_pivots` | Temple articulation | Compute temple rotation axis |
| `lens_planes` | Lens positioning | Compute lens width/height from extents |
| `symmetry_plane` | Left-right mirror | Ensure symmetric deformation |
| `deformation_regions` | Vertex influence zones | Control which vertices move in deformation |

---

## Files to Update

1. ✅ `templates/descriptors/geometric_metal.json` - Use JSON from Path A or B
2. ✅ `templates/registry.json` - Already updated
3. ⏳ `templates/geometric_metal.glb` - Rename meshes (Path B only)

---

## Testing

After applying fix:

```bash
# Restart backend
# (Already running with auto-reload)

# Test in browser
http://localhost:8001

# Upload images
# Click Generate 3D Mesh
# Should work (Path A: basic quality, Path B: good quality)
```

---

## Next Steps

**Immediate:**
1. Save Path A JSON to test system now
2. Generate test models to verify pipeline

**Short-term (this week):**
1. Prepare proper GLB in Blender (Path B)
2. Update descriptor to use clean names
3. Test quality improvement

**Week 1-2 (strategic plan):**
1. Implement bridge/temple/lens deformers
2. Add constraints and smoothing
3. Create more templates with proper structure

---

**Status:** Ready to implement either path
**Recommendation:** Start with Path A now, do Path B properly this week

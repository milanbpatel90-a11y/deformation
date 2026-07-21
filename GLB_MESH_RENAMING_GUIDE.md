# GLB Mesh Renaming Guide - Critical Fix Required

## 🔴 Problem Identified

Your `geometric_metal.glb` has **generic Blender export names** but the deformation engine expects **exact, structured names**.

### Current Mesh Names (Wrong)
```
✗ Plane_glasses_mat_0       (61,872 vertices)
✗ Plane.002_glasses_mat_0   (2,385 vertices)  
✗ Cube.002_glass_mat_0      (6,916 vertices)
✗ Plane.001_glasses_mat_0   (2,385 vertices)
```

### Required Mesh Names (Correct)
```
✓ Frame         - Main frame/rim structure
✓ Bridge        - Nose bridge connecting two lenses
✓ LeftTemple    - Left arm/temple
✓ RightTemple   - Right arm/temple
✓ LeftLens      - Left lens geometry
✓ RightLens     - Right lens geometry
✓ LeftRim       - Left lens rim (may overlap with Frame)
✓ RightRim      - Right lens rim (may overlap with Frame)
```

---

## ✅ Two Solutions

### Option A: Quick Test (Temporary Workaround)

**Status**: ✅ Already applied  
**File**: `templates/descriptors/geometric_metal.json` (updated with proper structure)

This descriptor now has all 7 required top-level keys but points to the **wrong mesh names** since your GLB still has generic names. The system will fail when it tries to load geometry.

**Result**: Will get error `"Descriptor references unknown geometry part 'Frame'"` because the GLB doesn't have a mesh named "Frame".

---

### Option B: Proper Fix (Production Quality) ⭐ RECOMMENDED

You **must** open the GLB in Blender and rename each mesh object to match the exact names above.

---

## 📋 Step-by-Step Blender Renaming Instructions

### 1. Open GLB in Blender
```
File → Import → glTF 2.0 (.glb/.gltf)
Navigate to: C:\Users\Petpooja-607\Desktop\defirmation\templates\geometric_metal.glb
```

### 2. Enter Object Mode
- Press `Tab` to ensure you're in Object mode (not Edit mode)

### 3. Identify Each Mesh
Look at the **Outliner** panel (top-right). You'll see:
```
Scene Collection
  ├─ Plane_glasses_mat_0       (61,872 verts - likely Frame + Rims)
  ├─ Plane.002_glasses_mat_0   (2,385 verts - likely RightTemple or Bridge)
  ├─ Cube.002_glass_mat_0      (6,916 verts - likely Lenses combined)
  └─ Plane.001_glasses_mat_0   (2,385 verts - likely LeftTemple or Bridge)
```

### 4. Best Guess Mapping (verify visually)

**Large mesh (61,872 vertices)** → Probably contains Frame + Rims  
→ **Action**: This mesh needs to be **separated** into Frame, LeftRim, RightRim (and possibly Bridge)

**Cube mesh (6,916 vertices)** → Lenses  
→ **Action**: Separate into LeftLens and RightLens

**Two small plane meshes (2,385 vertices each)** → Temples  
→ **Action**: Rename to LeftTemple and RightTemple

### 5. Separate Combined Meshes

#### A. Separate the Frame mesh
1. Select `Plane_glasses_mat_0` in Outliner
2. Press `Tab` to enter Edit mode
3. Press `P` → "By Loose Parts" (if frame/rims are disconnected)
   - OR manually select vertices:
     - Press `Alt+A` to deselect all
     - Press `L` while hovering over the left rim → Press `P` → "Selection" → name it "LeftRim"
     - Repeat for right rim → name it "RightRim"
     - Repeat for bridge → name it "Bridge"
     - Remaining geometry → name it "Frame"

#### B. Separate the Lens mesh
1. Select `Cube.002_glass_mat_0`
2. Enter Edit mode (`Tab`)
3. Select left lens vertices → `P` → "Selection" → name "LeftLens"
4. Select right lens vertices → `P` → "Selection" → name "RightLens"

#### C. Rename the Temples
1. Click `Plane.001_glasses_mat_0` in Outliner → Double-click name → rename to **LeftTemple**
2. Click `Plane.002_glasses_mat_0` in Outliner → Double-click name → rename to **RightTemple**

### 6. Verify All 8 Names Exist

Check the Outliner - you should now see:
```
Scene Collection
  ├─ Frame
  ├─ Bridge
  ├─ LeftTemple
  ├─ RightTemple
  ├─ LeftLens
  ├─ RightLens
  ├─ LeftRim
  └─ RightRim
```

### 7. Export as GLB

```
File → Export → glTF 2.0 (.glb/.gltf)
Format: glTF Binary (.glb)
Include: ✓ Selected Objects (or Visible Objects)
Transform: +Y Up
Geometry: ✓ Apply Modifiers
Save to: C:\Users\Petpooja-607\Desktop\defirmation\templates\geometric_metal.glb
```

**⚠️ IMPORTANT**: Make sure "+Y Up" is selected - this matches the coordinate system your deformers expect.

### 8. Verify the Fix

Run the diagnostic script again:
```bash
python check_glb_mesh_names.py
```

You should see:
```
✅ Perfect! All required mesh names are present.
```

### 9. Test the Pipeline

```bash
# Start backend (if not running)
python -m uvicorn backend.api.main:app --reload --port 8000

# Open viewer in browser
# Upload 3 images and test
```

---

## 🔍 Why This Matters

The deformation engine uses a **name-based lookup system**:

```python
# descriptor_loader.py line ~100
scene = trimesh.load(resolved_template, force="scene")
geometry = {name: mesh for name, mesh in scene.geometry.items()}

# Then later (~228):
def _get_geometry(self, geometry, part, label):
    if part not in geometry:
        raise ValueError(f"Descriptor references unknown geometry part '{part}' for {label}")
    return geometry[part]
```

**If the names don't match exactly, the loader cannot find the mesh and fails immediately.**

Each deformer then accesses meshes by these exact names:
- `bridge_deformer.py` → looks for `"Bridge"`, `"Frame"` vertex groups
- `temple_deformer.py` → looks for `"LeftTemple"`, `"RightTemple"` 
- `rim_deformer.py` → looks for `"LeftRim"`, `"RightRim"`, `"Frame"`

---

## 📊 Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Descriptor JSON | ✅ Fixed | All 7 required keys present |
| GLB Mesh Names | ❌ Wrong | Still has generic Blender names |
| Deformer Code | ✅ Ready | Expects exact names |
| System | 🔴 Blocked | Cannot run until GLB is fixed |

---

## 🚀 Next Steps

1. **Open `geometric_metal.glb` in Blender** (5 min)
2. **Separate and rename all meshes** following guide above (10-15 min)
3. **Export as GLB with +Y Up** (1 min)
4. **Run `python check_glb_mesh_names.py`** to verify (10 sec)
5. **Test full pipeline** with image upload (2 min)

**Total time estimate**: ~20 minutes for production-quality fix

---

## 📝 Alternative: Use a Pre-Made Template

If renaming is too complex, you can also:
1. Use one of the existing templates that already has proper naming
2. Copy its structure as a reference
3. Check if `templates/` folder has other GLB files with correct naming

Run this to check other templates:
```bash
python -c "import trimesh; import glob; [print(f'\n{f}:', list(trimesh.load(f, force='scene').geometry.keys())) for f in glob.glob('templates/*.glb')]"
```

---

## ❓ Need Help?

If you get stuck:
1. Share a screenshot of your Blender Outliner panel
2. Run the diagnostic script and share output
3. Ask about specific separation steps

The descriptor JSON is now correct - the only blocker is the GLB mesh naming.

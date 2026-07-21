# 🚀 Quick Start - Fix & Test in 25 Minutes

**Current blocker**: GLB mesh naming  
**Time to fix**: ~20 minutes  
**Time to test**: ~5 minutes

---

## ⚡ What You Need to Do NOW

### 1. Check the Problem (30 seconds)

```bash
python check_glb_mesh_names.py
```

You'll see:
```
❌ MISSING 8 required mesh names
⚠️  Found 4 extra/generic mesh names
```

This confirms the GLB has wrong names.

---

### 2. Fix in Blender (20 minutes)

**📖 Open this file and follow it step-by-step:**
```
GLB_MESH_RENAMING_GUIDE.md
```

**Quick checklist:**
- [ ] Open `templates/geometric_metal.glb` in Blender
- [ ] Separate the 61k-vertex mesh into Frame, Bridge, LeftRim, RightRim
- [ ] Separate the 6k-vertex lens mesh into LeftLens, RightLens
- [ ] Rename the two 2k-vertex meshes to LeftTemple, RightTemple
- [ ] Export as GLB (Format: Binary, Transform: +Y Up, Geometry: Apply Modifiers)
- [ ] Save to same location: `templates/geometric_metal.glb`

---

### 3. Verify the Fix (10 seconds)

```bash
python check_glb_mesh_names.py
```

Expected output:
```
✅ Perfect! All required mesh names are present.
```

---

### 4. Test the Full System (5 minutes)

```bash
# Start backend (if not already running)
python -m uvicorn backend.api.main:app --reload --port 8000
```

Open browser:
```
http://localhost:8001
```

Upload 3 test images and click "Generate 3D Model"

**Expected result:**
- ✅ No descriptor errors
- ✅ GLB downloads successfully
- ✅ Model appears in viewer

---

## 🎯 Success = You See a 3D Model

Once you see a glasses model rotating in the viewer, **you're unblocked** and can move to quality improvements.

---

## 📋 If You Get Stuck

### Problem: "I don't have Blender"
**Solution**: Download free from https://www.blender.org/download/

### Problem: "I can't separate the meshes"
**Solution**: 
1. In Edit mode, hover over geometry
2. Press `L` to select linked vertices
3. Press `P` → "Selection" to separate
4. Repeat for each part

### Problem: "Export settings are confusing"
**Solution**: Only these settings matter:
- **Format**: glTF Binary (.glb) ← dropdown at top
- **Transform**: +Y Up ← under "Transform" section
- **Geometry**: Apply Modifiers ← under "Geometry" section

### Problem: "Verification still fails after fix"
**Action**: Share the output of `python check_glb_mesh_names.py` - it will show exactly which names are still wrong

---

## 📁 Files You Need

| File | Purpose |
|------|---------|
| `check_glb_mesh_names.py` | Diagnostic tool (run before & after) |
| `GLB_MESH_RENAMING_GUIDE.md` | **Step-by-step Blender instructions** |
| `CURRENT_STATUS_AND_NEXT_STEPS.md` | Full context and status |
| `templates/geometric_metal.glb` | **The file you need to fix** |
| `templates/descriptors/geometric_metal.json` | ✅ Already fixed (don't touch) |

---

## 🔄 The Fix in 3 Steps (Visual)

```
BEFORE (Wrong):
├─ Plane_glasses_mat_0       (generic name ❌)
├─ Plane.002_glasses_mat_0   (generic name ❌)
├─ Cube.002_glass_mat_0      (generic name ❌)
└─ Plane.001_glasses_mat_0   (generic name ❌)

        ↓ [Separate & Rename in Blender]

AFTER (Correct):
├─ Frame          ✅
├─ Bridge         ✅
├─ LeftTemple     ✅
├─ RightTemple    ✅
├─ LeftLens       ✅
├─ RightLens      ✅
├─ LeftRim        ✅
└─ RightRim       ✅
```

---

## ⏱️ Timeline

| Time | Action |
|------|--------|
| 0:00 | Run `python check_glb_mesh_names.py` → See problem |
| 0:01 | Open Blender, import GLB |
| 0:05 | Separate Frame mesh (largest, 61k verts) |
| 0:10 | Separate Lens mesh (6k verts) |
| 0:12 | Rename Temple meshes (2k verts each) |
| 0:15 | Verify all 8 names in Outliner |
| 0:18 | Export as GLB with correct settings |
| 0:20 | Run verification script → See ✅ |
| 0:21 | Start backend server |
| 0:22 | Open viewer, upload images |
| 0:25 | **See working 3D model** 🎉 |

---

## 🎉 What Happens After Fix

Once unblocked, you can focus on **quality improvements**:

1. **Deformation realism** (bridge smoothness, temple curves)
2. **Material quality** (glass transparency, metal reflections)
3. **Measurement accuracy** (multi-view fusion tuning)

But **none of that matters until the GLB is fixed** - the system is 100% blocked on mesh naming.

---

## 💡 Why This Fix Is Critical

The descriptor loader uses **exact name-based lookups**:

```python
# This code expects EXACT mesh names
geometry = scene.geometry  # {"Frame": mesh1, "Bridge": mesh2, ...}
frame_mesh = geometry["Frame"]  # ← Fails if name is "Plane_glasses_mat_0"
```

**No workaround exists** - the names must match what the deformer code expects.

---

**Ready? Open `GLB_MESH_RENAMING_GUIDE.md` and start!** 🚀

---

**After the fix, come back and run:**
```bash
python check_glb_mesh_names.py  # Should see ✅
python -m uvicorn backend.api.main:app --reload --port 8000
# Open http://localhost:8001 and test
```

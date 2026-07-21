"""Quick diagnostic: check what mesh names are in the geometric_metal.glb file."""
import trimesh

glb_path = r"C:\Users\Petpooja-607\Desktop\defirmation\templates\geometric_metal.glb"

print("=" * 70)
print("GLB MESH NAMES DIAGNOSTIC")
print("=" * 70)
print(f"\nLoading: {glb_path}\n")

try:
    scene = trimesh.load(glb_path, force="scene")
    
    print(f"Found {len(scene.geometry)} mesh(es) in the GLB:\n")
    
    for name, mesh in scene.geometry.items():
        n_verts = len(mesh.vertices) if hasattr(mesh, "vertices") else "n/a"
        print(f"  '{name}'  ->  {n_verts} vertices")
    
    print("\n" + "=" * 70)
    print("REQUIRED MESH NAMES (from descriptor_loader.py):")
    print("=" * 70)
    print("\nThe deformer expects these EXACT names:")
    print("  - Frame")
    print("  - Bridge")
    print("  - LeftTemple")
    print("  - RightTemple")
    print("  - LeftLens")
    print("  - RightLens")
    print("  - LeftRim")
    print("  - RightRim")
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)
    
    required_names = {"Frame", "Bridge", "LeftTemple", "RightTemple", 
                     "LeftLens", "RightLens", "LeftRim", "RightRim"}
    found_names = set(scene.geometry.keys())
    
    missing = required_names - found_names
    extra = found_names - required_names
    
    if missing:
        print(f"\n❌ MISSING {len(missing)} required mesh names:")
        for name in sorted(missing):
            print(f"   - {name}")
    
    if extra:
        print(f"\n⚠️  Found {len(extra)} extra/generic mesh names:")
        for name in sorted(extra):
            print(f"   - {name}")
    
    if not missing and not extra:
        print("\n✅ Perfect! All required mesh names are present.")
    elif missing:
        print("\n📋 ACTION REQUIRED:")
        print("   Your GLB needs to be opened in Blender and each mesh renamed to match")
        print("   the required names exactly (case-sensitive). See PROPER_TEMPLATE_FIX.md")
        print("   for detailed instructions.")
    
except Exception as e:
    print(f"❌ Error loading GLB: {e}")

print("\n" + "=" * 70)

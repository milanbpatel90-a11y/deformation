"""Analyze each mesh in the GLB to determine which part it represents."""
import trimesh
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

scene = trimesh.load('templates/geometric_metal.glb', force='scene')

print("=" * 70)
print("MESH ANALYSIS FOR geometric_metal.glb")
print("=" * 70)

for name, mesh in sorted(scene.geometry.items()):
    if not isinstance(mesh, trimesh.Trimesh):
        continue
    verts = mesh.vertices
    centroid = verts.mean(axis=0)
    bbox = verts.max(axis=0) - verts.min(axis=0)
    print(f"\n--- {name} ---")
    print(f"  Vertices:     {len(verts)}")
    print(f"  Faces:        {len(mesh.faces)}")
    print(f"  Centroid:     x={centroid[0]:6.2f}  y={centroid[1]:6.2f}  z={centroid[2]:6.2f}")
    print(f"  Bounding box: x={bbox[0]:6.2f}  y={bbox[1]:6.2f}  z={bbox[2]:6.2f}")
    print(f"  X range:      {verts[:,0].min():.2f} to {verts[:,0].max():.2f}")
    print(f"  Y range:      {verts[:,1].min():.2f} to {verts[:,1].max():.2f}")
    print(f"  Z range:      {verts[:,2].min():.2f} to {verts[:,2].max():.2f}")

print("\n" + "=" * 70)
print("RECOMMENDED MAPPING (based on geometry analysis)")
print("=" * 70)

# Check if the GLB vertices are disconnected components within each mesh
print("\nChecking for disconnected components in large mesh...")
for name, mesh in sorted(scene.geometry.items()):
    if not isinstance(mesh, trimesh.Trimesh):
        continue
    if len(mesh.vertices) > 5000:
        try:
            # Try to find connected components
            shells = mesh.split(only_watertight=False)
            print(f"  {name}: {len(shells)} connected component(s)")
            for i, shell in enumerate(shells):
                bbox = shell.vertices.max(axis=0) - shell.vertices.min(axis=0)
                c = shell.vertices.mean(axis=0)
                print(f"    Component {i}: {len(shell.vertices)} verts, center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f})")
        except Exception as e:
            print(f"  {name}: could not split - {e}")

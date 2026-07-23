"""Verify renamed GLB has correct mesh names."""
import trimesh
import sys
sys.stdout.reconfigure(encoding='utf-8')

required = ['Frame','Bridge','LeftTemple','RightTemple','LeftLens','RightLens','LeftRim','RightRim']
scene = trimesh.load('templates/geometric_metal.glb', force='scene')
names = list(scene.geometry.keys())

print(f"Found {len(names)} mesh(es) in geometric_metal.glb:")
for n in sorted(names):
    v = scene.geometry[n].vertices.shape[0]
    print(f"  \u2713 {n}  ({v} verts)")

missing = [r for r in required if r not in names]
extra = [n for n in names if n not in required]

print()
if not missing:
    print("ALL 8 REQUIRED MESH NAMES PRESENT")
else:
    print(f"MISSING: {missing}")
if extra:
    print(f"EXTRA/GENERIC: {extra}")

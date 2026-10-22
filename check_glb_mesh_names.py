"""Diagnostic: verify GLB mesh names match what the deformation engine expects.

Usage:
  python check_glb_mesh_names.py
  python check_glb_mesh_names.py templates/rectangle_metal.glb
  python check_glb_mesh_names.py templates/*.glb
"""
from __future__ import annotations

import sys
from pathlib import Path

import trimesh

REQUIRED_NAMES = {
    "Frame",
    "Bridge",
    "LeftTemple",
    "RightTemple",
    "LeftLens",
    "RightLens",
    "LeftRim",
    "RightRim",
}

OPTIONAL_NAMES = {
    "LeftNosePad",
    "RightNosePad",
    "NosePads",
    "LeftHinge",
    "RightHinge",
    "TempleTips",
}


def check_one(glb_path: Path) -> bool:
    print("=" * 70)
    print(f"GLB: {glb_path}")
    print("=" * 70)

    if not glb_path.exists():
        print(f"❌ File not found: {glb_path}")
        return False

    try:
        scene = trimesh.load(str(glb_path), force="scene")
    except Exception as exc:
        print(f"❌ Failed to load: {exc}")
        return False

    found = set()
    for name, mesh in scene.geometry.items():
        n_verts = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
        print(f"  '{name}'  →  {n_verts:,} vertices")
        found.add(name)

    missing = REQUIRED_NAMES - found
    optional_present = OPTIONAL_NAMES & found
    extra = found - REQUIRED_NAMES - OPTIONAL_NAMES

    print()
    if missing:
        print(f"❌ MISSING {len(missing)} required mesh names:")
        for name in sorted(missing):
            print(f"   - {name}")
    else:
        print("✅ All required mesh names present")

    if optional_present:
        print(f"ℹ️  Optional parts found: {', '.join(sorted(optional_present))}")

    if extra:
        print(f"⚠️  Extra / generic names (may need aliases in descriptor):")
        for name in sorted(extra):
            print(f"   - {name}")

    ok = len(missing) == 0
    if not ok:
        print("\n📋 ACTION: Open in Blender, separate + rename meshes, re-export +Y Up.")
        print("   See GLB_MESH_RENAMING_GUIDE.md")
    print()
    return ok


def main() -> int:
    root = Path(__file__).resolve().parent
    args = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else []

    if not args:
        # Default: check all known production templates
        candidates = [
            root / "templates" / name
            for name in [
                "rectangle_metal.glb",
                "cat_eye_gold.glb",
                "oversized_clear.glb",
                "slim_rectangle_steel.glb",
                "round_wire_metal.glb",
                "geometric_metal.glb",
            ]
        ]
        args = [p for p in candidates if p.exists()]
        if not args:
            # Fall back to any .glb in templates/
            args = sorted((root / "templates").glob("*.glb"))

    if not args:
        print("No GLB files found. Pass a path: python check_glb_mesh_names.py path/to/file.glb")
        return 1

    results = [check_one(p) for p in args]
    passed = sum(1 for r in results if r)
    print("=" * 70)
    print(f"Result: {passed}/{len(results)} templates have correct mesh names")
    print("=" * 70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Gold Template Extraction Helper

Automates extraction and normalization of Gold_Template.zip into the proper
asset structure at assets/templates/GT_001/.
"""

import sys
import shutil
from pathlib import Path
from zipfile import ZipFile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_gold_template(zip_path: str, dry_run: bool = False) -> bool:
    """
    Extract Gold_Template.zip to assets/templates/GT_001/.
    
    Args:
        zip_path: Path to Gold_Template.zip
        dry_run: If True, only print what would be done
        
    Returns:
        True if successful, False otherwise
    """
    zip_file = Path(zip_path)
    
    if not zip_file.exists():
        print(f"✗ Error: ZIP file not found: {zip_path}")
        return False
    
    # Target directory
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "assets" / "templates" / "GT_001"
    
    print("=" * 60)
    print("GOLD TEMPLATE EXTRACTION")
    print("=" * 60)
    print(f"Source: {zip_file}")
    print(f"Target: {target_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL'}")
    print()
    
    if dry_run:
        print("Dry run mode - no files will be modified\n")
    
    # Extract to temporary location first
    temp_extract = project_root / "temp_gold_template"
    
    print("Step 1: Extracting ZIP...")
    if not dry_run:
        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
    print(f"  {'✓' if not dry_run else '→'} Extracted to temp location\n")
    
    # Find the Gold_Template directory in extracted files
    gold_template_dir = None
    if not dry_run:
        for item in temp_extract.iterdir():
            if "gold" in item.name.lower() and "template" in item.name.lower():
                gold_template_dir = item
                break
        
        if not gold_template_dir:
            # Maybe files are at root
            gold_template_dir = temp_extract
    
    # File mapping: source -> destination
    file_mappings = {
        # Geometry
        "Gold_Template_Runtime.glb": target_dir / "geometry" / "template.glb",
        
        # Source files
        "Gold_Template_Rhino_v1_scaled.3dm": target_dir / "source" / "Gold_Template_Rhino_v1_scaled.3dm",
        
        # Metadata (these should already be in metadata/ subdirectory)
        "metadata/template.json": target_dir / "metadata" / "template.json",
        "metadata/parameter_schema.json": target_dir / "metadata" / "parameter_schema.json",
        "metadata/measurements.json": target_dir / "metadata" / "measurements.json",
        "metadata/constraints.json": target_dir / "metadata" / "constraints.json",
        "metadata/topology.json": target_dir / "metadata" / "topology.json",
        "metadata/masks.json": target_dir / "metadata" / "masks.json",
        "metadata/landmarks.json": target_dir / "metadata" / "landmarks.json",
        "metadata/region_masks.json": target_dir / "metadata" / "region_masks.json",
        "metadata/scale_config.json": target_dir / "metadata" / "scale_config.json",
        
        # Deformation data
        "deformation/basis.npz": target_dir / "deformation" / "basis.npz",
        "deformation/basis_metadata.json": target_dir / "deformation" / "basis_metadata.json",
        "deformation/_part_order.json": target_dir / "deformation" / "_part_order.json",
        "deformation/_unified_faces.npy": target_dir / "deformation" / "_unified_faces.npy",
        "deformation/_unified_verts.npy": target_dir / "deformation" / "_unified_verts.npy",
        "deformation/build_basis.py": target_dir / "deformation" / "build_basis.py",
    }
    
    print("Step 2: Organizing files...")
    
    copied_count = 0
    missing_count = 0
    
    for source_rel, dest_path in file_mappings.items():
        if not dry_run and gold_template_dir:
            source_path = gold_template_dir / source_rel
            
            if source_path.exists():
                # Create parent directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ {source_rel} -> {dest_path.relative_to(project_root)}")
                copied_count += 1
            else:
                print(f"  ⚠ Missing: {source_rel}")
                missing_count += 1
        else:
            print(f"  → Would copy: {source_rel}")
    
    print()
    
    # Clean up temp directory
    if not dry_run:
        print("Step 3: Cleaning up...")
        shutil.rmtree(temp_extract, ignore_errors=True)
        print("  ✓ Removed temporary files\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not dry_run:
        print(f"Files copied: {copied_count}")
        print(f"Files missing: {missing_count}")
        
        if missing_count > 0:
            print("\n⚠ Some files were not found in the ZIP.")
            print("The ZIP structure may be different than expected.")
            print("Check the extracted files manually.")
        else:
            print("\n✓ All expected files extracted successfully!")
            print("\nNext steps:")
            print("1. Validate the template:")
            print("   python scripts/validate_templates.py")
            print("\n2. Test template loading:")
            print("   pytest tests/test_template_loading.py -v")
    else:
        print("Dry run completed. Use --extract to actually copy files.")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Gold Template to asset structure")
    parser.add_argument(
        "zip_path",
        help="Path to Gold_Template.zip"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Actually extract files (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    success = extract_gold_template(args.zip_path, dry_run=not args.extract)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

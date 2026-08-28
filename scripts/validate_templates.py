"""
Template Validation Script

Validates all templates in the assets/templates/ directory.
Reports on missing files, invalid metadata, and structural issues.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from template_registry import list_templates, TemplateBundle, get_template_assets


def validate_all_templates() -> Dict[str, Dict]:
    """
    Validate all available templates.
    
    Returns:
        Dictionary mapping template IDs to validation reports
    """
    templates = list_templates()
    results = {}
    
    print(f"Found {len(templates)} template(s): {', '.join(templates)}\n")
    
    for template_id in templates:
        print(f"Validating {template_id}...")
        print("-" * 60)
        
        try:
            bundle = TemplateBundle.load(template_id)
            report = bundle.validate()
            results[template_id] = report
            
            # Print results
            if report["valid"]:
                print(f"✓ {template_id} is VALID")
            else:
                print(f"✗ {template_id} has ERRORS:")
                for error in report["errors"]:
                    print(f"    ✗ {error}")
            
            if report["warnings"]:
                print(f"  Warnings:")
                for warning in report["warnings"]:
                    print(f"    ⚠ {warning}")
            
            # Print version info
            print(f"  Version: {report['version']}")
            print(f"  Topology: v{bundle.metadata.topology_version}")
            print(f"  Basis: v{bundle.metadata.basis_version}")
            
        except Exception as e:
            print(f"✗ FAILED to load {template_id}: {str(e)}")
            results[template_id] = {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
            }
        
        print()
    
    return results


def check_template_files(template_id: str) -> None:
    """
    Check which files exist for a template.
    
    Args:
        template_id: Template identifier
    """
    print(f"Checking files for {template_id}...")
    print("-" * 60)
    
    assets = get_template_assets(template_id)
    
    file_checks = [
        ("GLB", assets["glb"]),
        ("Template JSON", assets["template_json"]),
        ("Parameter Schema", assets["parameter_schema"]),
        ("Landmarks", assets["landmarks_json"]),
        ("Masks", assets["masks_json"]),
        ("Constraints", assets["constraints_json"]),
        ("Basis", assets["basis"]),
        ("Unified Vertices", assets["unified_verts"]),
        ("Unified Faces", assets["unified_faces"]),
    ]
    
    for name, path in file_checks:
        status = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
        print(f"  {status} {name:20s} {size}")
    
    print()


def print_summary(results: Dict[str, Dict]) -> None:
    """Print validation summary."""
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total = len(results)
    valid = sum(1 for r in results.values() if r["valid"])
    invalid = total - valid
    
    print(f"Total templates: {total}")
    print(f"Valid: {valid}")
    print(f"Invalid: {invalid}")
    
    if invalid > 0:
        print("\nTemplates with errors:")
        for template_id, report in results.items():
            if not report["valid"]:
                print(f"  - {template_id}: {len(report['errors'])} error(s)")


def main():
    """Main validation routine."""
    print("=" * 60)
    print("TEMPLATE VALIDATION")
    print("=" * 60)
    print()
    
    # Validate all templates
    results = validate_all_templates()
    
    # Print detailed file checks
    for template_id in list_templates():
        check_template_files(template_id)
    
    # Print summary
    print_summary(results)
    
    # Exit with error if any template is invalid
    if any(not r["valid"] for r in results.values()):
        print("\n⚠ Some templates have validation errors.")
        print("This is expected until Gold_Template.zip is extracted to GT_001/")
        return 1
    
    print("\n✓ All templates are valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

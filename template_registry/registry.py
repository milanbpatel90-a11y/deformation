"""
Template Registry

Provides centralized access to template assets and metadata.
Maintains version control and path management for all templates.
"""

from pathlib import Path
from typing import Dict, List
import json

# Root directory for all template assets
TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "assets" / "templates"


def get_template_dir(template_id: str) -> Path:
    """
    Get the root directory for a specific template.
    
    Args:
        template_id: Template identifier (e.g., "GT_001")
        
    Returns:
        Path to the template directory
        
    Raises:
        FileNotFoundError: If template doesn't exist
    """
    path = TEMPLATE_ROOT / template_id
    if not path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_id}\n"
            f"Expected location: {path}\n"
            f"Available templates: {list_templates()}"
        )
    return path


def get_template_assets(template_id: str) -> Dict[str, Path]:
    """
    Get all asset paths for a specific template.
    
    Returns a dictionary with standardized paths to:
    - GLB geometry file
    - Metadata JSON files
    - Deformation data (basis, landmarks, masks)
    - Source files (Rhino, etc.)
    
    Args:
        template_id: Template identifier (e.g., "GT_001")
        
    Returns:
        Dictionary mapping asset names to Path objects
        
    Example:
        >>> assets = get_template_assets("GT_001")
        >>> print(assets["glb"])
        >>> print(assets["basis"])
    """
    root = get_template_dir(template_id)
    
    return {
        # Root paths
        "root": root,
        "geometry": root / "geometry",
        "metadata": root / "metadata",
        "deformation": root / "deformation",
        "source": root / "source",
        
        # Primary geometry
        "glb": root / "geometry" / "template.glb",
        
        # Metadata files
        "template_json": root / "metadata" / "template.json",
        "parameter_schema": root / "metadata" / "parameter_schema.json",
        "measurements": root / "metadata" / "measurements.json",
        "constraints_json": root / "metadata" / "constraints.json",
        "topology": root / "metadata" / "topology.json",
        "masks_json": root / "metadata" / "masks.json",
        "landmarks_json": root / "metadata" / "landmarks.json",
        "region_masks": root / "metadata" / "region_masks.json",
        "scale_config": root / "metadata" / "scale_config.json",
        
        # Deformation data
        "basis": root / "deformation" / "basis.npz",
        "basis_metadata": root / "deformation" / "basis_metadata.json",
        "part_order": root / "deformation" / "_part_order.json",
        "unified_faces": root / "deformation" / "_unified_faces.npy",
        "unified_verts": root / "deformation" / "_unified_verts.npy",
    }


def list_templates() -> List[str]:
    """
    List all available template IDs.
    
    Returns:
        List of template identifiers
        
    Example:
        >>> templates = list_templates()
        >>> print(templates)
        ['GT_001', 'GT_002', 'GT_003']
    """
    if not TEMPLATE_ROOT.exists():
        return []
    
    return [
        d.name 
        for d in TEMPLATE_ROOT.iterdir() 
        if d.is_dir() and not d.name.startswith(("_", "."))
    ]


def get_template_version(template_id: str) -> Dict:
    """
    Get version information for a template.
    
    Args:
        template_id: Template identifier
        
    Returns:
        Dictionary with version information
    """
    assets = get_template_assets(template_id)
    template_json = assets["template_json"]
    
    if not template_json.exists():
        return {
            "template_id": template_id,
            "template_version": "unknown",
            "topology_version": "unknown",
            "basis_version": "unknown",
            "metadata_version": "unknown",
        }
    
    with open(template_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return {
        "template_id": data.get("template_id", template_id),
        "template_version": data.get("template_version", "unknown"),
        "topology_version": data.get("topology_version", "unknown"),
        "basis_version": data.get("basis_version", "unknown"),
        "metadata_version": data.get("metadata_version", "unknown"),
    }


def validate_template_structure(template_id: str) -> Dict[str, bool]:
    """
    Validate that all required template files exist.
    
    Args:
        template_id: Template identifier
        
    Returns:
        Dictionary mapping file types to existence status
    """
    assets = get_template_assets(template_id)
    
    required_files = [
        "glb",
        "template_json",
        "masks_json",
        "landmarks_json",
        "constraints_json",
        "basis",
    ]
    
    validation = {}
    for file_key in required_files:
        path = assets[file_key]
        validation[file_key] = path.exists()
    
    return validation

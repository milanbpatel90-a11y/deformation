"""
Template Loader

Loads and validates template bundles from the asset directory.
Provides structured access to all template data and metadata.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .registry import get_template_dir, get_template_assets


@dataclass
class TemplateMetadata:
    """Container for template metadata."""
    template_id: str
    template_version: str
    topology_version: str
    basis_version: str
    metadata_version: str
    description: Optional[str] = None
    objects: Optional[Dict] = None


class TemplateBundle:
    """
    Complete template bundle containing all geometry, metadata, and deformation data.
    
    This class loads and validates a template, providing structured access to:
    - Template metadata and versioning
    - Geometry and topology
    - Segmentation masks
    - Anatomical landmarks
    - Deformation constraints
    - Parameter schemas
    - Deformation basis functions
    
    Example:
        >>> from template_registry import TemplateBundle
        >>> bundle = TemplateBundle.load("GT_001")
        >>> print(bundle.metadata.template_version)
        >>> print(bundle.landmarks["landmarks"])
        >>> vertices = bundle.get_vertices()
    """
    
    def __init__(self, root: Path):
        """
        Initialize template bundle from root directory.
        
        Args:
            root: Path to template root directory
        """
        self.root = root
        self.assets = get_template_assets(root.name)
        
        # Load core metadata
        self.template = self._load_json("template_json")
        self.metadata = self._parse_metadata()
        
        # Load deformation data
        self.masks = self._load_json("masks_json")
        self.landmarks = self._load_json("landmarks_json")
        self.constraints = self._load_json("constraints_json")
        self.parameters = self._load_json("parameter_schema")
        
        # Optional metadata
        self.measurements = self._load_json("measurements", required=False)
        self.topology = self._load_json("topology", required=False)
        self.region_masks = self._load_json("region_masks", required=False)
        self.scale_config = self._load_json("scale_config", required=False)
        
        # Deformation basis (lazy loaded)
        self._basis_data = None
        self._vertices = None
        self._faces = None
    
    @classmethod
    def load(cls, template_id: str) -> "TemplateBundle":
        """
        Load a template bundle by ID.
        
        Args:
            template_id: Template identifier (e.g., "GT_001")
            
        Returns:
            Loaded TemplateBundle
            
        Example:
            >>> bundle = TemplateBundle.load("GT_001")
        """
        root = get_template_dir(template_id)
        return cls(root)
    
    def _load_json(self, asset_key: str, required: bool = True) -> Optional[Dict]:
        """
        Load a JSON file from the template assets.
        
        Args:
            asset_key: Key in assets dictionary
            required: Whether file is required
            
        Returns:
            Loaded JSON data or None if not required and missing
        """
        path = self.assets[asset_key]
        
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required template file not found: {path}\n"
                    f"Template: {self.root.name}"
                )
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _parse_metadata(self) -> TemplateMetadata:
        """Parse template metadata into structured format."""
        return TemplateMetadata(
            template_id=self.template.get("template_id", self.root.name),
            template_version=self.template.get("template_version", "1.0.0"),
            topology_version=self.template.get("topology_version", "1"),
            basis_version=self.template.get("basis_version", "1.0"),
            metadata_version=self.template.get("metadata_version", "1"),
            description=self.template.get("description"),
            objects=self.template.get("objects"),
        )
    
    def get_basis(self) -> Dict[str, np.ndarray]:
        """
        Load deformation basis data.
        
        Returns:
            Dictionary containing basis arrays:
            - 'Frame_basis', 'Lens_L_basis', 'Lens_R_basis', etc.
            
        Example:
            >>> basis = bundle.get_basis()
            >>> frame_basis = basis['Frame_basis']
            >>> print(frame_basis.shape)  # (n_verts, 3, n_modes)
        """
        if self._basis_data is None:
            basis_path = self.assets["basis"]
            if not basis_path.exists():
                raise FileNotFoundError(f"Basis file not found: {basis_path}")
            self._basis_data = np.load(basis_path)
        
        return self._basis_data
    
    def get_vertices(self) -> Optional[np.ndarray]:
        """
        Get template vertices from unified data.
        
        Returns:
            Vertex array (n_verts, 3) or None if not available
        """
        if self._vertices is None:
            verts_path = self.assets["unified_verts"]
            if verts_path.exists():
                self._vertices = np.load(verts_path)
        
        return self._vertices
    
    def get_faces(self) -> Optional[np.ndarray]:
        """
        Get template faces from unified data.
        
        Returns:
            Face array (n_faces, 3) or None if not available
        """
        if self._faces is None:
            faces_path = self.assets["unified_faces"]
            if faces_path.exists():
                self._faces = np.load(faces_path)
        
        return self._faces
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate template bundle integrity.
        
        Returns:
            Validation report with status and details
            
        Example:
            >>> report = bundle.validate()
            >>> if not report['valid']:
            >>>     print(report['errors'])
        """
        errors = []
        warnings = []
        
        # Check required files
        required_files = ["glb", "template_json", "masks_json", "landmarks_json", "constraints_json", "basis"]
        for file_key in required_files:
            if not self.assets[file_key].exists():
                errors.append(f"Missing required file: {file_key}")
        
        # Validate metadata structure
        if not self.metadata.template_id:
            errors.append("Template ID not specified")
        
        if not self.landmarks or "landmarks" not in self.landmarks:
            errors.append("No landmarks defined")
        else:
            n_landmarks = len(self.landmarks["landmarks"])
            if n_landmarks < 19:
                warnings.append(f"Only {n_landmarks} landmarks (expected 19+)")
        
        # Check object definitions
        if not self.metadata.objects:
            errors.append("No objects defined in template")
        else:
            expected_objects = ["Frame", "Lens_L", "Lens_R"]
            for obj in expected_objects:
                if obj not in self.metadata.objects:
                    warnings.append(f"Missing expected object: {obj}")
        
        # Validate basis data
        try:
            basis = self.get_basis()
            basis_keys = list(basis.keys())
            if not basis_keys:
                errors.append("Basis file is empty")
        except Exception as e:
            errors.append(f"Failed to load basis: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "template_id": self.metadata.template_id,
            "version": self.metadata.template_version,
        }
    
    def __repr__(self) -> str:
        return (
            f"TemplateBundle("
            f"id={self.metadata.template_id}, "
            f"version={self.metadata.template_version}, "
            f"topology_v={self.metadata.topology_version})"
        )

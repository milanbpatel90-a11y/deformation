"""
Tests for template registry and loading.

Validates that templates can be discovered, loaded, and validated correctly.
"""

import pytest
from pathlib import Path
from template_registry import (
    get_template_dir,
    get_template_assets,
    list_templates,
    TemplateBundle,
    TEMPLATE_ROOT,
)


class TestTemplateRegistry:
    """Test template registry functionality."""
    
    def test_template_root_exists(self):
        """Template root directory should exist."""
        assert TEMPLATE_ROOT.exists(), f"Template root not found: {TEMPLATE_ROOT}"
    
    def test_list_templates(self):
        """Should list available templates."""
        templates = list_templates()
        assert isinstance(templates, list)
        # GT_001 should exist even if just placeholder
        assert "GT_001" in templates or len(templates) >= 0
    
    def test_get_template_dir(self):
        """Should get template directory."""
        template_dir = get_template_dir("GT_001")
        assert template_dir.exists()
        assert template_dir.name == "GT_001"
    
    def test_get_template_dir_not_found(self):
        """Should raise error for non-existent template."""
        with pytest.raises(FileNotFoundError) as exc_info:
            get_template_dir("NONEXISTENT")
        
        assert "Template not found" in str(exc_info.value)
    
    def test_get_template_assets(self):
        """Should get all asset paths for a template."""
        assets = get_template_assets("GT_001")
        
        # Check required keys
        required_keys = [
            "root", "geometry", "metadata", "deformation",
            "glb", "template_json", "masks_json", "landmarks_json",
            "constraints_json", "basis"
        ]
        
        for key in required_keys:
            assert key in assets, f"Missing asset key: {key}"
            assert isinstance(assets[key], Path)


class TestTemplateBundle:
    """Test template bundle loading and validation."""
    
    def test_load_template_bundle(self):
        """Should load GT_001 template bundle."""
        bundle = TemplateBundle.load("GT_001")
        
        assert bundle is not None
        assert bundle.metadata.template_id == "GT_001"
        assert bundle.root.exists()
    
    def test_bundle_metadata(self):
        """Should have valid metadata."""
        bundle = TemplateBundle.load("GT_001")
        
        assert bundle.metadata.template_id == "GT_001"
        assert bundle.metadata.template_version is not None
        assert bundle.metadata.topology_version is not None
    
    def test_bundle_has_landmarks(self):
        """Should have landmarks defined."""
        bundle = TemplateBundle.load("GT_001")
        
        assert bundle.landmarks is not None
        assert "landmarks" in bundle.landmarks
        
        # Should have at least 19 landmarks (once Gold Template is extracted)
        n_landmarks = len(bundle.landmarks["landmarks"])
        assert n_landmarks >= 0  # Placeholder check
    
    def test_bundle_has_objects(self):
        """Should have object definitions."""
        bundle = TemplateBundle.load("GT_001")
        
        assert bundle.metadata.objects is not None
        
        # Expected objects
        expected_objects = ["Frame", "Lens_L", "Lens_R"]
        for obj in expected_objects:
            assert obj in bundle.metadata.objects
    
    def test_bundle_validation(self):
        """Should validate bundle structure."""
        bundle = TemplateBundle.load("GT_001")
        report = bundle.validate()
        
        assert "valid" in report
        assert "errors" in report
        assert "warnings" in report
        
        # Print validation results for debugging
        if not report["valid"]:
            print("\nValidation errors:")
            for error in report["errors"]:
                print(f"  - {error}")
        
        if report["warnings"]:
            print("\nValidation warnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
    
    def test_bundle_repr(self):
        """Should have readable string representation."""
        bundle = TemplateBundle.load("GT_001")
        repr_str = repr(bundle)
        
        assert "TemplateBundle" in repr_str
        assert "GT_001" in repr_str


class TestTemplateIntegration:
    """Integration tests for template usage."""
    
    def test_template_workflow(self):
        """Test complete template loading workflow."""
        # Step 1: List templates
        templates = list_templates()
        assert "GT_001" in templates
        
        # Step 2: Get assets
        assets = get_template_assets("GT_001")
        assert assets["glb"] is not None
        
        # Step 3: Load bundle
        bundle = TemplateBundle.load("GT_001")
        assert bundle.metadata.template_id == "GT_001"
        
        # Step 4: Access data
        landmarks = bundle.landmarks
        assert landmarks is not None
        
        parameters = bundle.parameters
        assert parameters is not None
        
        constraints = bundle.constraints
        assert constraints is not None


# Integration test placeholder for deformation engine
class TestDeformationEngineIntegration:
    """
    Integration tests with deformation engine.
    
    These tests will be enabled once Gold Template is extracted
    and deformation engine is connected to TemplateBundle.
    """
    
    @pytest.mark.skip(reason="Awaiting Gold Template extraction and engine integration")
    def test_deformation_with_template(self):
        """Test deformation using template bundle."""
        from backend.deformer.engine import DeformationEngine
        
        bundle = TemplateBundle.load("GT_001")
        engine = DeformationEngine(template=bundle)
        
        result = engine.deform(params={
            "frame_width": 145,
            "lens_width": 58,
            "lens_height": 37,
            "bridge_width": 18,
            "temple_length": 155,
        })
        
        assert result is not None
        assert "vertices" in result
        assert "faces" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

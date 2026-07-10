"""
Parametric 3D GLB Generator for Eyewear
Generates 3D glasses models from measurements using trimesh and pygltflib.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple, Optional
import json

from backend.models import Measurements, LensContour, FrameShape


class ParametricGLBGenerator:
    """Generate 3D GLB models from parametric measurements."""
    
    def __init__(self):
        """Initialize the generator."""
        self.scene = None
    
    def generate_from_measurements(
        self,
        measurements: Measurements,
        lens_contour: LensContour,
        output_path: str | Path
    ) -> Path:
        """
        Generate a complete 3D glasses model from measurements.
        
        Args:
            measurements: Parametric measurements
            lens_contour: Lens shape contours
            output_path: Output GLB file path
            
        Returns:
            Path to generated GLB file
        """
        output_path = Path(output_path)
        
        # Create scene
        self.scene = trimesh.Scene()
        
        # Generate components
        left_lens = self._create_lens(measurements, lens_contour.left, "left")
        right_lens = self._create_lens(measurements, lens_contour.right, "right")
        frame_rim = self._create_frame_rim(measurements, lens_contour)
        bridge = self._create_bridge(measurements)
        left_temple = self._create_temple(measurements, "left")
        right_temple = self._create_temple(measurements, "right")
        
        # Add to scene
        self.scene.add_geometry(left_lens, node_name="left_lens")
        self.scene.add_geometry(right_lens, node_name="right_lens")
        self.scene.add_geometry(frame_rim, node_name="frame_rim")
        self.scene.add_geometry(bridge, node_name="bridge")
        self.scene.add_geometry(left_temple, node_name="left_temple")
        self.scene.add_geometry(right_temple, node_name="right_temple")
        
        # Add nose pads if present
        if measurements.nose_pads:
            left_pad = self._create_nose_pad(measurements, "left")
            right_pad = self._create_nose_pad(measurements, "right")
            self.scene.add_geometry(left_pad, node_name="left_nose_pad")
            self.scene.add_geometry(right_pad, node_name="right_nose_pad")
        
        # Export to GLB
        self.scene.export(str(output_path))
        
        # Save metadata
        self._save_metadata(measurements, output_path)
        
        return output_path
    
    def _create_lens(
        self,
        measurements: Measurements,
        contour_points: List[List[float]],
        side: str
    ) -> trimesh.Trimesh:
        """Create a lens mesh from contour points."""
        
        # Convert normalized contour to 3D points
        lens_width = measurements.lens_width
        lens_height = measurements.lens_height
        
        # Scale contour points to actual dimensions
        points_3d = []
        for point in contour_points:
            # Normalize to lens dimensions
            if side == "left":
                x = (point[0] - 0.25) * lens_width * 2  # Center around left lens
            else:
                x = (point[0] - 0.75) * lens_width * 2  # Center around right lens
            y = (point[1] - 0.5) * lens_height * 2
            points_3d.append([x, y])
        
        # Create 2D polygon
        from shapely.geometry import Polygon
        try:
            poly = Polygon(points_3d)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
        except Exception:
            # Fallback to simple rectangle
            hw = lens_width / 2
            hh = lens_height / 2
            poly = Polygon([
                [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
            ])
        
        # Extrude to create 3D lens (thin)
        lens_thickness = 2.0  # mm
        mesh = trimesh.creation.extrude_polygon(poly, height=lens_thickness)
        
        # Position lens
        if side == "left":
            offset_x = -(measurements.bridge_width / 2 + lens_width / 2)
        else:
            offset_x = measurements.bridge_width / 2 + lens_width / 2
        
        mesh.apply_translation([offset_x, 0, 0])
        
        # Set material (transparent for lens)
        mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.9, 0.9, 0.95, 0.3],  # Slightly tinted, transparent
            metallicFactor=0.0,
            roughnessFactor=0.1
        )
        
        return mesh
    
    def _create_frame_rim(
        self,
        measurements: Measurements,
        lens_contour: LensContour
    ) -> trimesh.Trimesh:
        """Create frame rim around lenses."""
        
        rim_thickness = measurements.rim_thickness
        
        # Create rim as offset of lens contours
        meshes = []
        
        for side, contour_points in [("left", lens_contour.left), ("right", lens_contour.right)]:
            # Convert to 3D
            lens_width = measurements.lens_width
            lens_height = measurements.lens_height
            
            points_3d = []
            for point in contour_points:
                if side == "left":
                    x = (point[0] - 0.25) * lens_width * 2
                else:
                    x = (point[0] - 0.75) * lens_width * 2
                y = (point[1] - 0.5) * lens_height * 2
                points_3d.append([x, y])
            
            # Create rim as tube around contour
            from shapely.geometry import Polygon, LineString
            try:
                line = LineString(points_3d + [points_3d[0]])  # Close the loop
                rim_poly = line.buffer(rim_thickness / 2)
                
                # Extrude
                rim_mesh = trimesh.creation.extrude_polygon(rim_poly, height=3.0)
                
                # Position
                if side == "left":
                    offset_x = -(measurements.bridge_width / 2 + lens_width / 2)
                else:
                    offset_x = measurements.bridge_width / 2 + lens_width / 2
                
                rim_mesh.apply_translation([offset_x, 0, -1.5])
                meshes.append(rim_mesh)
            except Exception:
                pass  # Skip if rim creation fails
        
        # Combine rims
        if meshes:
            combined = trimesh.util.concatenate(meshes)
        else:
            # Fallback: simple rectangular rims
            combined = self._create_simple_rim(measurements)
        
        # Set material
        color = self._hex_to_rgb(measurements.color)
        combined.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=color + [1.0],
            metallicFactor=0.7 if measurements.material.value == "metal" else 0.1,
            roughnessFactor=0.3 if measurements.material.value == "metal" else 0.6
        )
        
        return combined
    
    def _create_simple_rim(self, measurements: Measurements) -> trimesh.Trimesh:
        """Fallback: create simple rectangular rim."""
        rim_thickness = measurements.rim_thickness
        lens_width = measurements.lens_width
        lens_height = measurements.lens_height
        
        meshes = []
        for side in ["left", "right"]:
            # Create rectangular rim
            outer_w = lens_width + rim_thickness
            outer_h = lens_height + rim_thickness
            inner_w = lens_width - rim_thickness
            inner_h = lens_height - rim_thickness
            
            # Create as difference of two rectangles
            from shapely.geometry import Polygon
            outer = Polygon([
                [-outer_w/2, -outer_h/2],
                [outer_w/2, -outer_h/2],
                [outer_w/2, outer_h/2],
                [-outer_w/2, outer_h/2]
            ])
            inner = Polygon([
                [-inner_w/2, -inner_h/2],
                [inner_w/2, -inner_h/2],
                [inner_w/2, inner_h/2],
                [-inner_w/2, inner_h/2]
            ])
            rim_poly = outer.difference(inner)
            
            rim_mesh = trimesh.creation.extrude_polygon(rim_poly, height=3.0)
            
            # Position
            if side == "left":
                offset_x = -(measurements.bridge_width / 2 + lens_width / 2)
            else:
                offset_x = measurements.bridge_width / 2 + lens_width / 2
            
            rim_mesh.apply_translation([offset_x, 0, -1.5])
            meshes.append(rim_mesh)
        
        return trimesh.util.concatenate(meshes)
    
    def _create_bridge(self, measurements: Measurements) -> trimesh.Trimesh:
        """Create bridge connecting the two lenses."""
        
        bridge_width = measurements.bridge_width
        bridge_height = 4.0  # mm
        bridge_depth = 3.0  # mm
        
        # Create as cylinder or box
        if measurements.shape == FrameShape.RIMLESS:
            # Thin bridge for rimless
            bridge = trimesh.creation.cylinder(
                radius=1.0,
                height=bridge_width,
                sections=16
            )
            # Rotate to horizontal
            bridge.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [0, 1, 0]
            ))
        else:
            # Box bridge
            bridge = trimesh.creation.box(
                extents=[bridge_width, bridge_height, bridge_depth]
            )
        
        # Position at center, slightly above lens center
        bridge.apply_translation([0, measurements.lens_height * 0.3, 0])
        
        # Set material
        color = self._hex_to_rgb(measurements.color)
        bridge.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=color + [1.0],
            metallicFactor=0.7 if measurements.material.value == "metal" else 0.1,
            roughnessFactor=0.3 if measurements.material.value == "metal" else 0.6
        )
        
        return bridge
    
    def _create_temple(self, measurements: Measurements, side: str) -> trimesh.Trimesh:
        """Create temple arm."""
        
        temple_length = measurements.temple_length
        temple_radius = 1.5  # mm
        curve_angle = measurements.temple_curve_angle or 28.0
        
        # Create straight part
        straight_length = temple_length * 0.7
        straight = trimesh.creation.cylinder(
            radius=temple_radius,
            height=straight_length,
            sections=16
        )
        
        # Rotate to horizontal
        straight.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]
        ))
        
        # Create curved part
        curved_length = temple_length * 0.3
        curve_segments = 10
        curve_points = []
        
        for i in range(curve_segments + 1):
            t = i / curve_segments
            x = straight_length / 2 + t * curved_length
            y = 0
            z = -t * curved_length * np.sin(np.radians(curve_angle))
            curve_points.append([x, y, z])
        
        # Create curved temple as swept cylinder
        curved_meshes = []
        for i in range(len(curve_points) - 1):
            segment = trimesh.creation.cylinder(
                radius=temple_radius,
                height=np.linalg.norm(
                    np.array(curve_points[i+1]) - np.array(curve_points[i])
                ),
                sections=8
            )
            # Position and orient segment
            center = (np.array(curve_points[i]) + np.array(curve_points[i+1])) / 2
            segment.apply_translation(center)
            curved_meshes.append(segment)
        
        # Combine straight and curved parts
        if curved_meshes:
            temple = trimesh.util.concatenate([straight] + curved_meshes)
        else:
            temple = straight
        
        # Position temple
        frame_width = measurements.frame_width
        lens_width = measurements.lens_width
        
        if side == "left":
            x_offset = -frame_width / 2
            temple.apply_translation([x_offset, 0, 0])
        else:
            x_offset = frame_width / 2
            temple.apply_translation([x_offset, 0, 0])
            # Mirror for right side
            temple.apply_transform(trimesh.transformations.reflection_matrix(
                [0, 0, 0], [1, 0, 0]
            ))
        
        # Set material
        color = self._hex_to_rgb(measurements.color)
        temple.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=color + [1.0],
            metallicFactor=0.7 if measurements.material.value == "metal" else 0.1,
            roughnessFactor=0.3 if measurements.material.value == "metal" else 0.6
        )
        
        return temple
    
    def _create_nose_pad(self, measurements: Measurements, side: str) -> trimesh.Trimesh:
        """Create nose pad."""
        
        pad_height = measurements.nose_pad_height or 3.0
        pad_width = 8.0  # mm
        pad_depth = 4.0  # mm
        
        # Create as ellipsoid
        pad = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        pad.apply_scale([pad_width / 2, pad_height / 2, pad_depth / 2])
        
        # Position
        bridge_width = measurements.bridge_width
        pad_distance = measurements.nose_pad_distance or bridge_width * 0.6
        pad_angle = measurements.nose_pad_angle or 15.0
        
        if side == "left":
            x_offset = -pad_distance / 2
        else:
            x_offset = pad_distance / 2
        
        y_offset = -measurements.lens_height * 0.4  # Below lens center
        z_offset = 5.0  # Forward
        
        pad.apply_translation([x_offset, y_offset, z_offset])
        
        # Angle inward
        if side == "left":
            pad.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(pad_angle), [0, 1, 0]
            ))
        else:
            pad.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(-pad_angle), [0, 1, 0]
            ))
        
        # Set material (soft silicone-like)
        pad.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.9, 0.9, 0.9, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8
        )
        
        return pad
    
    def _hex_to_rgb(self, hex_color: str) -> List[float]:
        """Convert hex color to RGB [0-1]."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            return [r, g, b]
        return [0.0, 0.0, 0.0]  # Default black
    
    def _save_metadata(self, measurements: Measurements, output_path: Path) -> None:
        """Save measurements metadata alongside GLB."""
        metadata = {
            "measurements": measurements.model_dump(),
            "generator": "ParametricGLBGenerator",
            "version": "1.0"
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Made with Bob

# Week 1, Day 1-2: Bridge Deformer Implementation

## Objective

Create a dedicated `BridgeDeformer` module that handles sophisticated bridge deformation, replacing the current simple width scaling with realistic bridge shape adaptation.

## Current State

**File:** `backend/deformer/engine.py` - `_deform_bridge()` method

```python
def _deform_bridge(self, scales: ScaleFactors) -> None:
    """Scale bridge width around the centreline and blend rim hinge transitions."""
    bridge_scale = self._safe_scale(scales.bridge_x, 0.6, 1.8)
    self._scale_group(VertexGroup.BRIDGE, np.array([bridge_scale, 1.0, 1.0]))
    
    # ... rim blending code ...
```

**What it does:**
- Simple width scaling (X-axis only)
- Some rim hinge blending
- No profile adaptation

**Problem:**
- Doesn't look like bridge of real glasses
- No nose pad positioning
- No profile variation (flat vs arched)

## New Architecture

### File Structure
```
backend/deformer/
├── bridge_deformer.py          # NEW - dedicated bridge logic
├── engine.py                   # MODIFIED - call BridgeDeformer
└── deformation_context.py      # UNCHANGED
```

### New File: `backend/deformer/bridge_deformer.py`

```python
"""Topology-aware bridge deformation driven by measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from backend.deformer.deformation_context import DeformationContext
    from backend.models import Measurements


@dataclass(frozen=True)
class BridgeSideResult:
    """Debug summary for bridge deformation."""
    
    width_before: float
    width_after: float
    height_before: float
    height_after: float
    profile_adapted: bool


class BridgeDeformer:
    """
    Reshape bridge geometry based on measurements.
    
    Handles:
    - Bridge width adaptation
    - Bridge height/profile adaptation
    - Nose pad repositioning
    - Smooth transitions to rims
    - Asymmetry handling
    """
    
    def __init__(
        self,
        width_influence: float = 0.8,
        height_influence: float = 0.6,
        hinge_smoothing: float = 0.5,
    ):
        self.width_influence = float(np.clip(width_influence, 0.0, 1.0))
        self.height_influence = float(np.clip(height_influence, 0.0, 1.0))
        self.hinge_smoothing = float(np.clip(hinge_smoothing, 0.0, 1.0))
    
    def apply(self, context: DeformationContext) -> DeformationContext:
        """Apply bridge deformation to the scene."""
        result = self._deform_bridge_mesh(context)
        
        context.update_metadata(
            bridge_deformation={
                "applied": result is not None,
                "width_influence": self.width_influence,
                "height_influence": self.height_influence,
                **(result.__dict__ if result else {}),
            }
        )
        return context
    
    def _deform_bridge_mesh(self, context: DeformationContext) -> BridgeSideResult | None:
        """Reshape bridge based on measurements."""
        try:
            bridge_mesh = context.mesh("Bridge")
        except KeyError:
            return None
        
        measurements = context.measurements
        template_dims = context.descriptor.template_dims
        
        # Get bridge geometry
        verts_before = bridge_mesh.vertices.copy()
        bridge_bounds = self._compute_bounds(verts_before)
        
        # Calculate deformation parameters
        width_scale = self._safe_scale(
            measurements.bridge_width / template_dims.bridge_width,
            0.6, 1.8
        )
        height_scale = self._safe_scale(
            measurements.bridge_height / template_dims.bridge_height if hasattr(template_dims, 'bridge_height') else 1.0,
            0.7, 1.5
        )
        
        # Apply width deformation
        verts = self._deform_width(verts_before, bridge_bounds, width_scale)
        
        # Apply height/profile deformation
        verts = self._deform_profile(verts, bridge_bounds, height_scale, measurements)
        
        # Position nose pads if present
        verts = self._position_nose_pads(verts, context, measurements)
        
        # Smooth transitions to rims
        verts = self._smooth_rim_transitions(verts, context, bridge_bounds)
        
        # Apply smoothing to bridge
        verts = self._smooth_surface(verts)
        
        # Assign back to mesh
        bridge_mesh.vertices = verts
        
        # Compute result summary
        verts_after = verts
        result = BridgeSideResult(
            width_before=float(bridge_bounds["extent"][0]),
            width_after=float(np.ptp(verts_after[:, 0])),
            height_before=float(bridge_bounds["extent"][2]),
            height_after=float(np.ptp(verts_after[:, 2])),
            profile_adapted=abs(height_scale - 1.0) > 0.05,
        )
        return result
    
    def _deform_width(
        self,
        vertices: np.ndarray,
        bounds: dict,
        width_scale: float,
    ) -> np.ndarray:
        """Scale bridge width (X-axis) while preserving depth (Z-axis)."""
        verts = vertices.copy()
        center_x = bounds["center"][0]
        
        # Weight vertices by distance from center
        # Center vertices move less, outer vertices move more
        distance_from_center = np.abs(verts[:, 0] - center_x)
        max_distance = max(bounds["extent"][0] * 0.5, 1e-6)
        weight = np.clip(distance_from_center / max_distance, 0.0, 1.0)
        weight = self._smoothstep(weight)
        
        # Apply width scaling with influence control
        delta_scale = (width_scale - 1.0) * self.width_influence
        effective_scale = 1.0 + delta_scale * weight
        
        verts[:, 0] = center_x + (verts[:, 0] - center_x) * effective_scale
        
        return verts
    
    def _deform_profile(
        self,
        vertices: np.ndarray,
        bounds: dict,
        height_scale: float,
        measurements,
    ) -> np.ndarray:
        """Adapt bridge profile (height and curvature)."""
        verts = vertices.copy()
        center_y = bounds["center"][1]
        center_z = bounds["center"][2]
        
        # Scale bridge height
        height_delta = (height_scale - 1.0) * self.height_influence
        effective_height_scale = 1.0 + height_delta
        
        # Apply height scaling
        verts[:, 2] = center_z + (verts[:, 2] - center_z) * effective_height_scale
        
        # TODO: Add curvature adaptation based on frame_family
        # Different frame families have different bridge profiles:
        # - Flat/rectangular: minimal curve
        # - Wayfarer: moderate downward curve
        # - Round: pronounced curve
        # This requires template metadata enhancement
        
        return verts
    
    def _position_nose_pads(
        self,
        vertices: np.ndarray,
        context: DeformationContext,
        measurements,
    ) -> np.ndarray:
        """Position nose pads based on measurement hints."""
        try:
            nose_pads = context.mesh("NosePads")
        except KeyError:
            return vertices  # No nose pads in this template
        
        verts = vertices.copy()
        nose_verts = nose_pads.vertices.copy()
        
        # Get target positions from measurements
        target_distance = getattr(measurements, 'nose_pad_distance', None)
        target_height = getattr(measurements, 'nose_pad_height', None)
        
        if target_distance is None or target_height is None:
            return vertices  # Can't position without measurements
        
        template_dims = context.descriptor.template_dims
        distance_scale = self._safe_scale(
            target_distance / template_dims.nose_pad_distance,
            0.6, 1.8
        )
        height_scale = self._safe_scale(
            target_height / template_dims.nose_pad_height,
            0.6, 1.8
        )
        
        # Get bounds of nose pads
        nose_bounds = self._compute_bounds(nose_verts)
        
        # Scale distance (X-axis separation)
        nose_verts[:, 0] = nose_bounds["center"][0] + \
            (nose_verts[:, 0] - nose_bounds["center"][0]) * distance_scale
        
        # Scale height (Y-axis)
        nose_verts[:, 1] = nose_bounds["center"][1] + \
            (nose_verts[:, 1] - nose_bounds["center"][1]) * height_scale
        
        nose_pads.vertices = nose_verts
        
        return vertices  # Bridge vertices unchanged by nose pad repositioning
    
    def _smooth_rim_transitions(
        self,
        vertices: np.ndarray,
        context: DeformationContext,
        bridge_bounds: dict,
    ) -> np.ndarray:
        """Smooth transitions between bridge and rim hinges."""
        verts = vertices.copy()
        
        # Get rim geometry to understand hinge positions
        try:
            left_rim = context.mesh("LeftRim")
            right_rim = context.mesh("RightRim")
        except KeyError:
            return vertices  # Can't smooth without rims
        
        # Blend bridge with rim hinge positions
        # Find vertices near hinge areas and smooth them
        hinge_radius = 15.0  # mm - typical hinge radius
        
        for rim_mesh, sign in [(left_rim, -1.0), (right_rim, 1.0)]:
            rim_hinge = rim_mesh.vertices.mean(axis=0)
            
            # Find bridge vertices near this hinge
            distance_to_hinge = np.linalg.norm(
                verts - rim_hinge,
                axis=1
            )
            near_hinge = distance_to_hinge < hinge_radius
            
            if not np.any(near_hinge):
                continue
            
            # Smooth these vertices toward the hinge
            hinge_weight = 1.0 - np.clip(
                distance_to_hinge / hinge_radius,
                0.0, 1.0
            )
            hinge_weight = self._smoothstep(hinge_weight)
            
            # Blend toward hinge with influence control
            blend_factor = hinge_weight * self.hinge_smoothing * 0.3
            verts[near_hinge] = (
                verts[near_hinge] * (1.0 - blend_factor[:, None]) +
                rim_hinge * blend_factor[:, None]
            )
        
        return verts
    
    def _smooth_surface(self, vertices: np.ndarray) -> np.ndarray:
        """Apply light Laplacian smoothing to bridge surface."""
        # This is a simplified version
        # A full version would use mesh connectivity
        # For now, just slightly smooth based on local neighbors
        
        # Find local neighborhoods (vertices within 5mm)
        verts_smooth = vertices.copy()
        
        for i, vertex in enumerate(vertices):
            distances = np.linalg.norm(vertices - vertex, axis=1)
            neighbors = distances < 5.0
            neighbors[i] = False  # Exclude self
            
            if np.any(neighbors):
                neighbor_mean = vertices[neighbors].mean(axis=0)
                # Blend slightly toward neighbor mean (light smoothing)
                verts_smooth[i] = vertex * 0.9 + neighbor_mean * 0.1
        
        return verts_smooth
    
    @staticmethod
    def _compute_bounds(vertices: np.ndarray) -> dict:
        """Compute bounding box and properties."""
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        extent = maximum - minimum
        center = (minimum + maximum) / 2.0
        
        return {
            "minimum": minimum,
            "maximum": maximum,
            "extent": extent,
            "center": center,
        }
    
    @staticmethod
    def _smoothstep(value: np.ndarray | float) -> np.ndarray | float:
        """Smoothstep function for smooth transitions."""
        clipped = np.clip(value, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)
    
    @staticmethod
    def _safe_scale(value: float, minimum: float, maximum: float) -> float:
        """Safely clamp scale value."""
        if not np.isfinite(value):
            return 1.0
        return float(np.clip(value, minimum, maximum))
```

## Integration into Engine

**File:** `backend/deformer/engine.py` - Modify to use `BridgeDeformer`

**Current code (lines ~90-120):**
```python
def _deform_bridge(self, scales: ScaleFactors) -> None:
    """Scale bridge width around the centreline..."""
    bridge_scale = self._safe_scale(scales.bridge_x, 0.6, 1.8)
    self._scale_group(VertexGroup.BRIDGE, np.array([bridge_scale, 1.0, 1.0]))
    # ... blending code ...
```

**Replace with:**
```python
def deform(self, measurements, lens_contour=None):
    """Return a deformed scene driven by measurements."""
    scales = ScaleFactors.from_measurements(measurements, self.template_dims)
    
    self._deform_front(scales)
    
    # NEW: Use dedicated bridge deformer
    bridge_deformer = BridgeDeformer(
        width_influence=0.8,
        height_influence=0.6,
    )
    # Need to convert MeshDeformer to use DeformationContext
    # OR: Call deformer logic directly
    
    self._deform_temples(scales, measurements)
    self._deform_nose_pads(measurements)
    self._refresh_normals()
    
    return self.scene
```

## Required Changes to Support DeformationContext

The `MeshDeformer` currently doesn't use `DeformationContext`. To integrate `BridgeDeformer`, we need a path forward:

### Option A: Keep Two Parallel Systems (Simpler)
- Leave `MeshDeformer` as-is for now
- Have `BridgeDeformer` work on raw trimesh
- Later unify them

### Option B: Migrate to DeformationContext (Better Long-term)
- Refactor `MeshDeformer` to use context
- All deformers use context
- Cleaner architecture

**Recommendation:** Option B - do the migration

## Migration Steps (if doing Option B)

1. Create wrapper around existing MeshDeformer logic
2. Convert to context-based pipeline
3. Wire in BridgeDeformer
4. Later migrate temple/frame/etc

## Test Plan

**Test File:** `tests/test_bridge_deformer.py`

```python
def test_bridge_width_scaling():
    """Bridge should scale width proportionally."""
    # Arrange: Load template
    # Act: Create BridgeDeformer, apply 1.5x scale
    # Assert: Bridge width increased ~50%

def test_bridge_profile_adaptation():
    """Bridge height should adapt based on measurements."""
    # Arrange: Load template with high/low height measurements
    # Act: Apply deformer
    # Assert: Height changed appropriately

def test_nose_pad_positioning():
    """Nose pads should reposition based on measurements."""
    # Arrange: Load template with custom nose pad measurements
    # Act: Apply deformer
    # Assert: Nose pads moved to correct position

def test_rim_transition_smoothing():
    """Bridge should smooth smoothly into rim hinges."""
    # Arrange: Load template
    # Act: Apply deformer with large width change
    # Assert: No sharp angles at hinge transition

def test_bridge_symmetry():
    """Bridge deformation should maintain symmetry."""
    # Arrange: Load template with asymmetric input
    # Act: Apply deformer
    # Assert: Bridge is symmetric (averaged)

def test_invalid_measurements_clamp():
    """Invalid measurements should be clamped to safe ranges."""
    # Arrange: Create extreme measurements
    # Act: Apply deformer
    # Assert: No crashes, output is reasonable
```

## Success Criteria

✅ **Functionally:**
- Generated bridge looks intentional, not generic
- Width scaling feels natural
- Profile adapts to measurements
- Nose pads position correctly
- Smooth transitions at hinges
- No visible artifacts

✅ **Technically:**
- Code is well-documented
- Tests pass 100%
- No regressions in other components
- Integrates cleanly with engine

✅ **Visually (User Test):**
- User thinks bridge is "just right"
- Doesn't look like generic scaling
- Looks like it was designed for their glasses

## Time Estimate

- [ ] Write BridgeDeformer class: 1.5 hours
- [ ] Write tests: 1 hour
- [ ] Debug and refine: 1 hour
- [ ] Integration: 0.5 hours
- **Total: ~4 hours**

## Next: Day 3-4 Temple Deformer

Once bridge is working, temple deformer follows the same pattern with additional:
- Curvature adaptation
- Thickness tapering
- Hinge stress distribution

---

**Ready to start Week 1? Begin with BridgeDeformer.** ✅

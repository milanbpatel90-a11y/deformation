"""PBR material definitions for frame and lens."""

from __future__ import annotations

import trimesh
from trimesh.visual.material import PBRMaterial

from backend.models import FrameMaterial, Measurements


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return [r, g, b, int(alpha * 255)]


def frame_material(measurements: Measurements) -> PBRMaterial:
    """Create PBR material for the frame based on material type."""
    rgba = hex_to_rgba(measurements.color)

    if measurements.material == FrameMaterial.METAL:
        return PBRMaterial(
            name="FrameMetal",
            baseColorFactor=rgba,
            metallicFactor=1.0,
            roughnessFactor=0.25,
        )
    if measurements.material == FrameMaterial.ACETATE:
        return PBRMaterial(
            name="FrameAcetate",
            baseColorFactor=rgba,
            metallicFactor=0.0,
            roughnessFactor=0.45,
        )
    if measurements.material == FrameMaterial.PLASTIC:
        return PBRMaterial(
            name="FramePlastic",
            baseColorFactor=rgba,
            metallicFactor=0.0,
            roughnessFactor=0.6,
        )
    return PBRMaterial(
        name="FrameDefault",
        baseColorFactor=rgba,
        metallicFactor=0.8,
        roughnessFactor=0.3,
    )


def lens_material(measurements: Measurements) -> PBRMaterial:
    """Thin transparent/tinted lens material."""
    color_hex = measurements.lens_color or "#ffffff"
    opacity = measurements.lens_opacity if measurements.lens_opacity is not None else 0.31
    rgba = hex_to_rgba(color_hex, alpha=opacity)
    
    # For higher opacity (sunglasses), make it slightly more reflective/metallic
    metallic = 0.15 if opacity > 0.5 else 0.0
    
    return PBRMaterial(
        name="Lens",
        baseColorFactor=rgba,
        metallicFactor=metallic,
        roughnessFactor=0.05,
    )


def apply_materials(scene: trimesh.Scene, measurements: Measurements) -> trimesh.Scene:
    """Assign PBR materials to named parts in the scene."""
    frame_mat = frame_material(measurements)
    lens_mat = lens_material(measurements)

    lens_parts = {"LeftLens", "RightLens"}

    for name, geom in scene.geometry.items():
        if name in lens_parts:
            geom.visual.material = lens_mat
        else:
            geom.visual.material = frame_mat

    return scene

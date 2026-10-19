"""PBR material definitions for frame and lens — production-quality eyewear look."""

from __future__ import annotations

import trimesh
from trimesh.visual.material import PBRMaterial

from backend.models import FrameMaterial, Measurements


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) < 6:
        h = (h + "000000")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return [r, g, b, int(max(0.0, min(1.0, alpha)) * 255)]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def frame_material(measurements: Measurements) -> PBRMaterial:
    """Create PBR material for the frame based on material type and color."""
    rgba = hex_to_rgba(measurements.color)
    mat = measurements.material

    if mat == FrameMaterial.METAL:
        return PBRMaterial(
            name="FrameMetal",
            baseColorFactor=rgba,
            metallicFactor=0.95,
            roughnessFactor=0.22,
            emissiveFactor=[0.0, 0.0, 0.0],
        )

    if mat == FrameMaterial.TITANIUM:
        return PBRMaterial(
            name="FrameTitanium",
            baseColorFactor=rgba,
            metallicFactor=0.88,
            roughnessFactor=0.32,
        )

    if mat == FrameMaterial.ACETATE:
        return PBRMaterial(
            name="FrameAcetate",
            baseColorFactor=rgba,
            metallicFactor=0.0,
            roughnessFactor=0.28,
        )

    if mat == FrameMaterial.PLASTIC:
        return PBRMaterial(
            name="FramePlastic",
            baseColorFactor=rgba,
            metallicFactor=0.0,
            roughnessFactor=0.48,
        )

    if mat == FrameMaterial.MIXED:
        return PBRMaterial(
            name="FrameMixed",
            baseColorFactor=rgba,
            metallicFactor=0.55,
            roughnessFactor=0.35,
        )

    return PBRMaterial(
        name="FrameDefault",
        baseColorFactor=rgba,
        metallicFactor=0.8,
        roughnessFactor=0.3,
    )


def lens_material(measurements: Measurements) -> PBRMaterial:
    """Optical lens: clear, tinted, or sunglass with appropriate transmission."""
    color_hex = measurements.lens_color or "#e8f4ff"
    opacity = measurements.lens_opacity
    if opacity is None:
        opacity = 0.22

    opacity = _clamp01(opacity)
    rgba = hex_to_rgba(color_hex, alpha=opacity)

    if opacity > 0.55:
        metallic = 0.12
        roughness = 0.04
    elif opacity > 0.35:
        metallic = 0.05
        roughness = 0.06
    else:
        metallic = 0.0
        roughness = 0.08

    return PBRMaterial(
        name="Lens",
        baseColorFactor=rgba,
        metallicFactor=metallic,
        roughnessFactor=roughness,
        alphaMode="BLEND",
        doubleSided=True,
    )


def hinge_material(measurements: Measurements) -> PBRMaterial:
    """Small metal hinges — always metallic regardless of frame body."""
    base = measurements.color if measurements.material in {
        FrameMaterial.METAL, FrameMaterial.TITANIUM, FrameMaterial.MIXED
    } else "#c0c0c0"
    rgba = hex_to_rgba(base)
    return PBRMaterial(
        name="HingeMetal",
        baseColorFactor=rgba,
        metallicFactor=1.0,
        roughnessFactor=0.18,
    )


def nose_pad_material() -> PBRMaterial:
    """Soft silicone-like nose pads."""
    return PBRMaterial(
        name="NosePad",
        baseColorFactor=[230, 230, 225, 255],
        metallicFactor=0.0,
        roughnessFactor=0.75,
    )


def apply_materials(scene: trimesh.Scene, measurements: Measurements) -> trimesh.Scene:
    """Assign differentiated PBR materials to named parts in the scene."""
    frame_mat = frame_material(measurements)
    lens_mat = lens_material(measurements)
    hinge_mat = hinge_material(measurements)
    pad_mat = nose_pad_material()

    lens_parts = {"LeftLens", "RightLens", "Lens"}
    hinge_parts = {"LeftHinge", "RightHinge", "Hinge"}
    pad_parts = {"LeftNosePad", "RightNosePad", "NosePads", "NosePad"}

    for name, geom in scene.geometry.items():
        if not hasattr(geom, "visual"):
            continue
        if name in lens_parts:
            geom.visual.material = lens_mat
        elif name in hinge_parts:
            geom.visual.material = hinge_mat
        elif name in pad_parts:
            geom.visual.material = pad_mat
        else:
            geom.visual.material = frame_mat

    return scene

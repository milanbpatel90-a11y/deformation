"""Procedural fallback templates with stable named parts and hinge pivots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _ring(center_x: float, width: float, height: float, z: float, tube: float) -> trimesh.Trimesh:
    outer = trimesh.creation.annulus(r_min=max(0.001, 0.5 - tube / max(width, height)), r_max=0.5, height=0.02, sections=48)
    outer.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    scale = np.eye(4)
    scale[0, 0], scale[1, 1], scale[2, 2] = width, height, 1.0
    outer.apply_transform(scale)
    outer.apply_translation([center_x, 0, z])
    return outer


def load_template(frame_type: str, templates_dir: str | Path | None = None) -> dict[str, Any]:
    del templates_dir
    frame = _ring(-0.035, 0.065, 0.042, 0.0, 0.004).copy()
    right = _ring(0.035, 0.065, 0.042, 0.0, 0.004).copy()
    lens_l = trimesh.creation.cylinder(0.019, 0.001, sections=48); lens_l.apply_translation([-0.035, 0, 0])
    lens_r = trimesh.creation.cylinder(0.019, 0.001, sections=48); lens_r.apply_translation([0.035, 0, 0])
    bridge = trimesh.creation.box([0.012, 0.004, 0.004]); bridge.apply_translation([0, 0, 0])
    temples = {}
    for name, x in (("temple_L", -0.0675), ("temple_R", 0.0675)):
        bar = trimesh.creation.box([0.14, 0.004, 0.006]); bar.apply_translation([x + (-0.07 if x < 0 else 0.07), -0.01, 0])
        temples[name] = bar
    return {"frame_type": frame_type, "parts": {"front": trimesh.util.concatenate([frame, right, bridge]), "lens_L": lens_l, "lens_R": lens_r, **temples}, "pivots": {"hinge_L": [-0.0675, 0, 0], "hinge_R": [0.0675, 0, 0], "bridge": [0, 0, 0]}}

"""Template-preserving mesh deformation for eyewear GLB assets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import typing

import numpy as np
import trimesh

from backend.models import LensContour, Measurements, TemplateDimensions

from backend.deformer.bridge_deformer import BridgeDeformer
from backend.deformer.constraints import ConstraintSolver
from backend.deformer.lens_deformer import LensDeformer
from backend.deformer.rim_deformer import RimDeformer
from backend.deformer.temple_deformer import TempleDeformer
from backend.deformer.symmetry import SymmetrySolver
from backend.deformer.smoother import MeshSmoother
from backend.deformer.quality_checker import QualityChecker, QualityReport
from backend.deformer.deformation_context import DeformationContext


class MeshDeformer:
    """Deform a professional template GLB using a modular stage pipeline."""

    def __init__(
        self,
        template_scene: trimesh.Scene,
        template_dims: TemplateDimensions,
        rim_pull_strength: float = 0.65,
    ):
        self.template_scene = template_scene
        self.template_dims = template_dims
        self.rim_pull_strength = float(np.clip(rim_pull_strength, 0.0, 1.0))

        self.stages = [
            RimDeformer(contour_strength=self.rim_pull_strength),
            BridgeDeformer(),
            TempleDeformer(),
            LensDeformer(),
            ConstraintSolver(),
            SymmetrySolver(),
            MeshSmoother(lam=0.5, mu=-0.53, iterations=12),
        ]
        self.quality_checker = QualityChecker()

    def deform(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> tuple[DeformationContext, QualityReport]:
        for stage in self.stages:
            if hasattr(stage, "apply"):
                if isinstance(stage, RimDeformer):
                    context = stage.apply(context, lens_contour)
                else:
                    context = stage.apply(context)

        self._refresh_normals(context)
        # Deformers mutate context.meshes, not the source scene graph's original
        # geometry. Synchronize before returning so export uses deformed parts.
        context.sync_scene_from_meshes()
        quality = self.quality_checker.evaluate(context)
        return context, quality

    def _refresh_normals(self, context: DeformationContext) -> None:
        for geom in context.meshes.values():
            if hasattr(geom, "fix_normals"):
                geom.fix_normals()
            if hasattr(geom, "_cache"):
                geom._cache.clear()

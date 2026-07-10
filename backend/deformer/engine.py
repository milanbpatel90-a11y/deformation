"""Template-preserving mesh deformation for eyewear GLB assets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import typing

import numpy as np
import trimesh

from backend.models import LensContour, Measurements, TemplateDimensions

# We load deformers individually in the main orchestration class.
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
    """
    Deform a professional template GLB using a modular stage pipeline.

    The engine orchestrates specialized deformers, acting on the DeformationContext.
    """

    def __init__(
        self,
        template_scene: trimesh.Scene,
        template_dims: TemplateDimensions,
        rim_pull_strength: float = 0.65,
    ):
        self.template_scene = template_scene
        self.template_dims = template_dims
        self.rim_pull_strength = float(np.clip(rim_pull_strength, 0.0, 1.0))
        
        # Instantiate stages
        self.stages = [
            RimDeformer(contour_strength=self.rim_pull_strength),
            BridgeDeformer(),
            TempleDeformer(),
            LensDeformer(),
            ConstraintSolver(),
            SymmetrySolver(),
            MeshSmoother(),
        ]
        self.quality_checker = QualityChecker()

    def deform(
        self,
        context: DeformationContext,
        lens_contour: LensContour | None = None,
    ) -> tuple[DeformationContext, QualityReport]:
        """Apply all deformation stages to the context and return quality report."""
        
        for stage in self.stages:
            if hasattr(stage, "apply"):
                # Handle special argument requirements (e.g., lens_contour)
                if isinstance(stage, RimDeformer):
                    context = stage.apply(context, lens_contour)
                else:
                    context = stage.apply(context)

        # Refresh normals at the very end
        self._refresh_normals(context)

        quality = self.quality_checker.evaluate(context)

        return context, quality

    def _refresh_normals(self, context: DeformationContext) -> None:
        """Fix normals after all mesh edits."""
        for geom in context.meshes.values():
            if hasattr(geom, "fix_normals"):
                geom.fix_normals()
            if hasattr(geom, "_cache"):
                geom._cache.clear()

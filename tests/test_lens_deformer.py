import unittest
from pathlib import Path

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.lens_deformer import LensDeformer
from backend.deformer.rim_deformer import RimDeformer
from backend.models import FrameMaterial, FrameShape, LensContour, Measurements
from backend.template_library.loader import TemplateLibrary


class LensDeformerTests(unittest.TestCase):
    def _context(self) -> DeformationContext:
        library = TemplateLibrary(Path("templates"))
        info = library.load("geometric_metal")
        measurements = Measurements(
            frame_width=135.0,
            lens_width=50.0,
            lens_height=46.0,
            bridge_width=16.0,
            temple_length=135.0,
            rim_thickness=1.0,
            material=FrameMaterial.METAL,
            shape=FrameShape.RECTANGLE,
            nose_pads=True,
        )
        descriptor = DescriptorLoader(Path("templates")).load("geometric_metal", measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_lens_deformation_preserves_topology_and_thickness(self) -> None:
        ctx = self._context()
        left_before = ctx.mesh("LeftLens").vertices.copy()
        right_before = ctx.mesh("RightLens").vertices.copy()
        ctx = RimDeformer(contour_strength=0.8).apply(ctx, None)
        deformer = LensDeformer()
        selections = {
            "LeftLens": deformer._collect_lenses(ctx, "left"),
            "RightLens": deformer._collect_lenses(ctx, "right"),
        }
        ctx = deformer.apply(ctx)

        def local_thickness(vertices, selection):
            origin, basis = deformer._lens_basis(selection)
            local = deformer._to_local(vertices, origin, basis)
            front_idx = deformer._surface_indices(local, front=True)
            back_idx = deformer._surface_indices(local, front=False)
            return float(
                np.mean(local[front_idx, 2]) - np.mean(local[back_idx, 2])
            )

        for part, before in (("LeftLens", left_before), ("RightLens", right_before)):
            after = ctx.mesh(part).vertices.copy()
            self.assertEqual(len(after), len(before))
            selection = selections[part]
            thickness_before = local_thickness(before, selection)
            thickness_after = local_thickness(after, selection)
            self.assertAlmostEqual(thickness_after, thickness_before, places=6)
        self.assertTrue(ctx.metadata["lens_deformation"]["applied"])
        for side in ctx.metadata["lens_deformation"]["sides"]:
            self.assertTrue(side["valid"], side)


if __name__ == "__main__":
    unittest.main()

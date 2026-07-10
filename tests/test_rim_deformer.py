import unittest
from pathlib import Path

import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.rim_deformer import RimDeformer
from backend.models import FrameMaterial, FrameShape, LensContour, Measurements
from backend.template_library.loader import TemplateLibrary


class RimDeformerTests(unittest.TestCase):
    def _context(self) -> DeformationContext:
        library = TemplateLibrary(Path("templates"))
        info = library.load("geometric_metal")
        measurements = Measurements(
            frame_width=148.0,
            lens_width=58.0,
            lens_height=40.0,
            bridge_width=18.0,
            temple_length=140.0,
            rim_thickness=2.2,
            material=FrameMaterial.METAL,
            shape=FrameShape.RECTANGLE,
            nose_pads=True,
        )
        descriptor = DescriptorLoader(Path("templates")).load("geometric_metal", measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_rim_deformation_preserves_symmetry(self) -> None:
        ctx = self._context()
        contour = LensContour(
            left=[[0.04, 0.18], [0.10, 0.06], [0.38, 0.04], [0.46, 0.20], [0.46, 0.76], [0.36, 0.92], [0.10, 0.90], [0.03, 0.70]],
            right=[[0.54, 0.20], [0.62, 0.04], [0.90, 0.06], [0.96, 0.18], [0.97, 0.70], [0.90, 0.90], [0.64, 0.92], [0.54, 0.76]],
        )
        ctx = RimDeformer(contour_strength=0.8).apply(ctx, contour)
        left = ctx.mesh("LeftRim").bounds
        right = ctx.mesh("RightRim").bounds
        self.assertAlmostEqual(abs(left[0, 0]), abs(right[1, 0]), places=3)
        self.assertAlmostEqual(abs(left[1, 0]), abs(right[0, 0]), places=3)
        self.assertTrue(ctx.metadata["rim_deformation"]["applied"])


if __name__ == "__main__":
    unittest.main()

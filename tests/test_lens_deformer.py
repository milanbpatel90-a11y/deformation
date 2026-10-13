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
    def _context(self, template_name: str = "geometric_metal") -> DeformationContext:
        library = TemplateLibrary(Path("templates"))
        info = library.load(template_name)
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
        descriptor = DescriptorLoader(Path("templates")).load(template_name, measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_lens_deformation_preserves_topology_and_fits_the_real_aperture(self) -> None:
        # Run against both released split assets. Real lens surfaces are curved, so selecting
        # only vertices at exact max/min Z misses most of the boundary.
        contour = LensContour(
            left=[[0.04, 0.18], [0.10, 0.06], [0.38, 0.04], [0.46, 0.20], [0.46, 0.76], [0.36, 0.92], [0.10, 0.90], [0.03, 0.70]],
            right=[[0.54, 0.20], [0.62, 0.04], [0.90, 0.06], [0.96, 0.18], [0.97, 0.70], [0.90, 0.90], [0.64, 0.92], [0.54, 0.76]],
        )
        for template_name in ("geometric_metal", "rectangle_plastic"):
            with self.subTest(template=template_name):
                ctx = self._context(template_name)
                ctx = RimDeformer(contour_strength=0.8).apply(ctx, contour)
                lens_before = {name: ctx.mesh(name).vertices.copy() for name in ("LeftLens", "RightLens")}
                lens_faces = {name: ctx.mesh(name).faces.copy() for name in ("LeftLens", "RightLens")}
                lens_uv = {
                    name: ctx.mesh(name).visual.uv.copy()
                    for name in ("LeftLens", "RightLens")
                    if getattr(ctx.mesh(name).visual, "uv", None) is not None
                }
                ctx = LensDeformer().apply(ctx)
                for part, before in lens_before.items():
                    after = ctx.mesh(part).vertices.copy()
                    self.assertEqual(len(after), len(before))
                    self.assertAlmostEqual(float(np.ptp(after[:, 2])), float(np.ptp(before[:, 2])), places=5)
                    # The algorithm fits in XY and deliberately preserves the authored curved Z profile.
                    np.testing.assert_array_equal(after[:, 2], before[:, 2])
                    np.testing.assert_array_equal(ctx.mesh(part).faces, lens_faces[part])
                    if part in lens_uv:
                        np.testing.assert_array_equal(ctx.mesh(part).visual.uv, lens_uv[part])
                self.assertTrue(ctx.metadata["lens_deformation"]["applied"])
                by_side = {row["side"]: row for row in ctx.metadata["lens_deformation"]["sides"]}
                self.assertEqual(set(by_side), {"left", "right"})
                for side in by_side.values():
                    self.assertTrue(side["valid"], side)
                    self.assertTrue(side["inside_rim"], side)
                    self.assertLessEqual(side["fit_error_mm"], 0.75, side)
                    self.assertAlmostEqual(side["thickness_delta_mm"], 0.0, places=5)
                    self.assertIn("matches_requested_dimensions", side)
                    # The flag and the warning must always agree: a size miss is never silent.
                    self.assertEqual(side["matches_requested_dimensions"], not side["dimension_warning"], side)
                    if template_name == "geometric_metal":
                        # This template's whole rim is only ~58 mm wide, so a 58 mm lens cannot fit.
                        # Fitting it inside must be reported as a size mismatch, not hidden.
                        self.assertFalse(side["matches_requested_dimensions"], side)
                    else:
                        # The rim stage now opens the aperture toward the request, so the lens gets
                        # close to the requested size (it used to stay at the template's native size).
                        self.assertLessEqual(side["width_error_ratio"], 0.15, side)
                        self.assertLessEqual(side["height_error_ratio"], 0.15, side)



if __name__ == "__main__":
    unittest.main()

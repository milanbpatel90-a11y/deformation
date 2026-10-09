from tests.template_fixture import procedural_library
import unittest
from pathlib import Path

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.rim_deformer import RimDeformer
from backend.models import FrameMaterial, FrameShape, LensContour, Measurements
from backend.template_library.loader import TemplateLibrary


class RimDeformerTests(unittest.TestCase):
    def _context(self) -> DeformationContext:
        library = procedural_library(self)
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
        descriptor = DescriptorLoader(library.templates_dir).load("geometric_metal", measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_rim_deformation_preserves_symmetry(self) -> None:
        ctx = self._context()
        contour = LensContour(
            left=[[0.04, 0.18], [0.10, 0.06], [0.38, 0.04], [0.46, 0.20], [0.46, 0.76], [0.36, 0.92], [0.10, 0.90], [0.03, 0.70]],
            right=[[0.54, 0.20], [0.62, 0.04], [0.90, 0.06], [0.96, 0.18], [0.97, 0.70], [0.90, 0.90], [0.64, 0.92], [0.54, 0.76]],
        )
        left_before = ctx.mesh("LeftRim").vertices.copy()
        right_before = ctx.mesh("RightRim").vertices.copy()
        bridge_before = ctx.mesh("Bridge").vertices.copy()
        lenses_before = {name: ctx.mesh(name).vertices.copy() for name in ("LeftLens", "RightLens")}
        temples_before = {name: ctx.mesh(name).vertices.copy() for name in ("LeftTemple", "RightTemple")}
        ctx = RimDeformer(contour_strength=0.8).apply(ctx, contour)
        for name, before in (("LeftRim", left_before), ("RightRim", right_before)):
            self.assertGreater(float(np.linalg.norm(ctx.mesh(name).vertices - before, axis=1).max()), 0.0)
        np.testing.assert_array_equal(ctx.mesh("Bridge").vertices, bridge_before)
        for name, before in {**lenses_before, **temples_before}.items():
            np.testing.assert_array_equal(ctx.mesh(name).vertices, before)
        left = ctx.mesh("LeftRim").bounds
        right = ctx.mesh("RightRim").bounds
        self.assertAlmostEqual(abs(left[0, 0]), abs(right[1, 0]), places=3)
        self.assertAlmostEqual(abs(left[1, 0]), abs(right[0, 0]), places=3)
        self.assertTrue(ctx.metadata["rim_deformation"]["applied"])

    def test_propagation_is_exact_at_shared_vertices_and_bounded_in_falloff(self) -> None:
        frame = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0], [20.0, 0.0, 0.0]]),
            faces=np.empty((0, 3), dtype=np.int64), process=False,
        )
        rim = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        deformed = rim + np.array([0.0, 0.0, 1.0])

        RimDeformer._propagate_to_frame(frame, rim, deformed)

        self.assertAlmostEqual(frame.vertices[0, 2], 1.0, places=8)
        self.assertGreater(frame.vertices[1, 2], 0.0)
        self.assertLess(frame.vertices[1, 2], 0.2)
        self.assertAlmostEqual(frame.vertices[2, 2], 0.0, places=8)

    def test_propagation_rejects_mismatched_basis_vertex_arrays(self) -> None:
        frame = trimesh.Trimesh(vertices=np.zeros((1, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
        with self.assertRaises(ValueError, msg="input correspondence must be validated"):
            RimDeformer._propagate_to_frame(frame, np.zeros((2, 3)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()

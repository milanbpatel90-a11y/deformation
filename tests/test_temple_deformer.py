import unittest
from pathlib import Path

import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.deformer.temple_deformer import TempleDeformer
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


def _degenerate_triangle_count(mesh: trimesh.Trimesh) -> int:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces):
        return 0
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    return int(np.count_nonzero(double_area <= 1e-12))


class TempleDeformerTests(unittest.TestCase):
    def _context(self) -> DeformationContext:
        library = TemplateLibrary(Path("templates"))
        info = library.load("geometric_metal")
        measurements = Measurements(
            frame_width=135.0,
            lens_width=50.0,
            lens_height=46.0,
            bridge_width=16.0,
            temple_length=150.0,
            rim_thickness=1.4,
            material=FrameMaterial.METAL,
            shape=FrameShape.GEOMETRIC,
            nose_pads=True,
            temple_curve_angle=36.0,
        )
        descriptor = DescriptorLoader(Path("templates")).load("geometric_metal", measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_temple_deformation_keeps_frame_fixed_and_extends_temples(self) -> None:
        ctx = self._context()
        frame_before = ctx.mesh("Frame").vertices.copy()
        left_before = ctx.mesh("LeftTemple").bounds.copy()
        right_before = ctx.mesh("RightTemple").bounds.copy()
        left_degenerate_before = _degenerate_triangle_count(ctx.mesh("LeftTemple"))
        right_degenerate_before = _degenerate_triangle_count(ctx.mesh("RightTemple"))
        ctx = TempleDeformer().apply(ctx)
        frame_after = ctx.mesh("Frame").vertices.copy()
        left_after = ctx.mesh("LeftTemple").bounds.copy()
        right_after = ctx.mesh("RightTemple").bounds.copy()
        self.assertAlmostEqual(float(np.linalg.norm(frame_after - frame_before, axis=1).max()), 0.0, places=6)
        self.assertGreater(abs(float(left_after[0, 0])), abs(float(left_before[0, 0])))
        self.assertGreater(abs(float(right_after[1, 0])), abs(float(right_before[1, 0])))
        self.assertTrue(ctx.metadata["temple_deformation"]["applied"])
        self.assertEqual(_degenerate_triangle_count(ctx.mesh("LeftTemple")), left_degenerate_before)
        self.assertEqual(_degenerate_triangle_count(ctx.mesh("RightTemple")), right_degenerate_before)
        self.assertEqual(left_degenerate_before, 0)
        self.assertEqual(right_degenerate_before, 0)
        for name in ("LeftTemple", "RightTemple"):
            normals = np.asarray(ctx.mesh(name).vertex_normals, dtype=np.float64)
            self.assertTrue(np.isfinite(normals).all())
            self.assertTrue((np.linalg.norm(normals, axis=1) > 1e-12).all())


if __name__ == "__main__":
    unittest.main()

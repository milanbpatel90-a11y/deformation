import unittest
from pathlib import Path

import numpy as np
import trimesh

from backend.deformer.bridge_deformer import BridgeDeformer
from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


class BridgeDeformerTests(unittest.TestCase):
    def _context(self) -> DeformationContext:
        library = TemplateLibrary(Path("templates"))
        info = library.load("geometric_metal")
        measurements = Measurements(
            frame_width=135.0,
            lens_width=50.0,
            lens_height=46.0,
            bridge_width=22.0,
            temple_length=135.0,
            rim_thickness=1.2,
            material=FrameMaterial.METAL,
            shape=FrameShape.GEOMETRIC,
            nose_pads=True,
        )
        descriptor = DescriptorLoader(Path("templates")).load("geometric_metal", measurements=measurements, template_info=info)
        scene = trimesh.load(info.glb_path, force="scene")
        return DeformationContext(template_info=info, template_scene=scene, descriptor=descriptor, measurements=measurements)

    def test_bridge_deformation_changes_bridge_only(self) -> None:
        ctx = self._context()
        left_rim_before = ctx.mesh("LeftRim").vertices.copy()
        right_rim_before = ctx.mesh("RightRim").vertices.copy()
        left_temple_before = ctx.mesh("LeftTemple").vertices.copy()
        bridge_before = ctx.mesh("Bridge").bounds.copy()
        ctx = BridgeDeformer().apply(ctx)
        bridge_after = ctx.mesh("Bridge").bounds.copy()
        self.assertGreater(float(bridge_after[1, 0] - bridge_after[0, 0]), float(bridge_before[1, 0] - bridge_before[0, 0]))
        self.assertAlmostEqual(float(np.linalg.norm(ctx.mesh("LeftRim").vertices - left_rim_before, axis=1).max()), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(ctx.mesh("RightRim").vertices - right_rim_before, axis=1).max()), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(ctx.mesh("LeftTemple").vertices - left_temple_before, axis=1).max()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

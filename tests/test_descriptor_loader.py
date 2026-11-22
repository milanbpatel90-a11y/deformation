import unittest
from pathlib import Path

from backend.deformer.descriptor_loader import DescriptorLoader
from backend.models import FrameMaterial, FrameShape, Measurements
from backend.template_library.loader import TemplateLibrary


class DescriptorLoaderEmptyAnchorTests(unittest.TestCase):
    """Regression test for optional temple-tip empties.

    rectangle_plastic's descriptor declares 'left_temple_tip'/'right_temple_tip'
    empties (Left_Temple_End / Right_Temple_End) that are not present as scene
    nodes in rectangle_plastic.glb. The deformation engine derives temple
    length/axis from actual temple geometry and never reads these empties, so
    their absence must not raise -- it previously did, which made
    DescriptorLoader.load() fail for this template before the deformer ever ran.
    """

    def test_rectangle_plastic_descriptor_loads_despite_missing_temple_tip_empties(self) -> None:
        library = TemplateLibrary(Path("templates"))
        info = library.load("rectangle_plastic")
        measurements = Measurements(
            frame_width=140.0,
            lens_width=53.0,
            lens_height=42.0,
            bridge_width=17.2,
            temple_length=145.0,
            rim_thickness=1.0,
            material=FrameMaterial.PLASTIC if hasattr(FrameMaterial, "PLASTIC") else FrameMaterial.ACETATE,
            shape=FrameShape.RECTANGLE,
            nose_pads=False,
        )
        descriptor = DescriptorLoader(Path("templates")).load(
            "rectangle_plastic", measurements=measurements, template_info=info
        )
        # The required (non-optional) empties must still resolve normally.
        self.assertIn("Bridge_Center", descriptor.empty_anchors)
        self.assertIn("Left_Hinge", descriptor.empty_anchors)
        self.assertIn("Right_Hinge", descriptor.empty_anchors)
        # The optional temple-tip empties are absent from the GLB and must be
        # silently skipped rather than populated or raising.
        self.assertNotIn("Left_Temple_End", descriptor.empty_anchors)
        self.assertNotIn("Right_Temple_End", descriptor.empty_anchors)

    def test_missing_required_empty_still_raises(self) -> None:
        """A missing empty for a role that is NOT in the optional allow-list
        must still raise, so this fix cannot silently hide a real authoring
        error in template data."""
        payload = {
            "empties": {
                "bridge": "Does_Not_Exist_Node",
            }
        }
        import trimesh

        scene = trimesh.Scene()
        with self.assertRaises(ValueError):
            DescriptorLoader._load_empty_anchors(payload, scene)


if __name__ == "__main__":
    unittest.main()

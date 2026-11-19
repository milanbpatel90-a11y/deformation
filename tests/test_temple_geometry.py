import unittest

import numpy as np
import trimesh

from backend.deformer.temple_deformer import TempleDeformer, TempleSelection
from backend.deformer.temple_geometry import infer_temple_hinge


class _MeshContext:
    def __init__(self, mesh):
        self._mesh = mesh

    def mesh(self, name):
        if name != "LeftTemple":
            raise KeyError(name)
        return self._mesh


class TempleGeometryTests(unittest.TestCase):
    def _z_axis_temple(self):
        # Mirrors the uploaded asset's layout: frame on XY, temple length on Z.
        temple = trimesh.creation.cylinder(radius=1.3, height=171.5, sections=16)
        temple.apply_translation([-57.0, -7.0, 71.57])
        frame = trimesh.creation.box(extents=[123.7, 54.6, 5.2])
        return temple, frame

    def test_hinge_inference_uses_long_axis_and_frame_proximity(self):
        temple, frame = self._z_axis_temple()
        pivot, axis = infer_temple_hinge(temple, frame, "left")

        self.assertGreater(abs(float(axis[2])), 0.99)
        self.assertGreater(float(axis[2]), 0.0)
        self.assertAlmostEqual(float(pivot[0]), -57.0, delta=0.2)
        self.assertAlmostEqual(float(pivot[1]), -7.0, delta=0.2)
        self.assertAlmostEqual(float(pivot[2]), -14.18, delta=1.0)

    def test_length_deformation_targets_actual_geometry_length(self):
        temple, frame = self._z_axis_temple()
        pivot, axis = infer_temple_hinge(temple, frame, "left")
        before = temple.vertices.copy()
        current_length = TempleDeformer._temple_length(before, pivot, axis)
        self.assertGreater(current_length, 170.0)

        selection = TempleSelection(
            side="left",
            temple_part="LeftTemple",
            tip_part=None,
            hinge_pivot=pivot,
            axis=axis,
            wrap_axis=np.array([0.0, 1.0, 0.0]),
            temple_before=before,
            tip_before=None,
        )
        context = _MeshContext(temple)
        TempleDeformer()._extend_length(
            context,
            selection,
            {"current_length": current_length, "target_length": 150.0},
        )

        final_length = TempleDeformer._temple_length(temple.vertices, pivot, axis)
        self.assertAlmostEqual(final_length, 150.0, delta=0.05)
        self.assertLess(final_length, current_length)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import trimesh

from backend.deformer.temple_deformer import TempleDeformer, TempleSelection
from backend.deformer.temple_geometry import infer_temple_hinge


class _MeshContext:
    def __init__(self, meshes):
        self._meshes = meshes

    def mesh(self, name):
        return self._meshes[name]


class TempleGeometryTests(unittest.TestCase):
    def _temple_with_hinge_assembly(self, side):
        """Make a temple with a multi-shell hinge, main arm, and ear tip."""
        x0 = -57.8 if side == "left" else 57.8
        vertices = []

        # Hinge connector: the pivot should sit near the centre of this short
        # assembly, not on its back-most end-cap.
        for z in np.linspace(-14.187, -7.436, 10):
            y = -1.3 - 0.02 * (z + 11.0)
            for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                vertices.append([x0 + 1.1 * np.cos(angle), y + 1.25 * np.sin(angle), z])

        # Main temple arm, extending away from the hinge.
        for z in np.linspace(-7.484, 98.87, 12):
            y = -1.3 - 0.02 * (z + 7.4)
            for x_offset, y_offset in (
                (-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8),
                (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5),
            ):
                vertices.append([x0 + x_offset, y + y_offset, z])

        # Ear tip: the final section bends down progressively.
        for z in np.linspace(98.379, 157.323, 8):
            progress = (z - 98.379) / (157.323 - 98.379)
            y = -3.0 - 10.0 * progress
            for x_offset, y_offset in ((-0.8, -1.0), (0.8, -1.0), (0.8, 1.0), (-0.8, 1.0)):
                vertices.append([x0 + x_offset, y + y_offset, z])

        temple = trimesh.Trimesh(vertices=np.asarray(vertices), faces=[], process=False)
        frame = trimesh.creation.box(extents=[123.7, 54.6, 5.2])
        return temple, frame

    def test_hinge_inference_uses_long_axis_and_connector_centre_for_both_sides(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                temple, frame = self._temple_with_hinge_assembly(side)
                pivot, axis = infer_temple_hinge(temple, frame, side)

                self.assertGreater(abs(float(axis[2])), 0.99)
                self.assertGreater(float(axis[2]), 0.0)
                expected_x = -57.8 if side == "left" else 57.8
                self.assertAlmostEqual(float(pivot[0]), expected_x, delta=0.2)
                self.assertAlmostEqual(float(pivot[1]), -1.3, delta=0.3)
                self.assertAlmostEqual(float(pivot[2]), -10.4, delta=0.8)

    def test_wrap_and_length_change_preserve_hinge_and_target_150_units(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                temple, frame = self._temple_with_hinge_assembly(side)
                pivot, axis = infer_temple_hinge(temple, frame, side)
                before = temple.vertices.copy()
                initial_length = TempleDeformer._temple_length(before, pivot, axis)
                self.assertGreater(initial_length, 165.0)

                selection = TempleSelection(
                    side=side,
                    temple_part="Temple",
                    tip_part=None,
                    hinge_pivot=pivot,
                    axis=axis,
                    wrap_axis=np.array([0.0, 1.0, 0.0]),
                    temple_before=before,
                    tip_before=None,
                )
                context = _MeshContext({"Temple": temple, "Frame": frame})
                deformer = TempleDeformer()
                target = {"wrap_angle": 8.0, "target_length": 150.0}

                deformer._apply_wrap(context, selection, target)
                deformer._extend_length(context, selection, target)

                final_axis = deformer._axis_from_vertices(temple.vertices, pivot, axis)
                final_length = deformer._temple_length(temple.vertices, pivot, final_axis)
                nearest_hinge_vertices = np.argsort(np.linalg.norm(before - pivot, axis=1))[:24]
                hinge_drift = np.linalg.norm(
                    temple.vertices[nearest_hinge_vertices] - before[nearest_hinge_vertices], axis=1
                ).max()

                self.assertAlmostEqual(final_length, 150.0, delta=0.15)
                self.assertLess(final_length, initial_length)
                self.assertLess(hinge_drift, 0.05)


if __name__ == "__main__":
    unittest.main()

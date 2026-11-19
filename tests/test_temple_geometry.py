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
    def _z_axis_temple(self, side):
        """Build a multi-row temple similar to the uploaded GLB's geometry."""
        x = -57.45 if side == "left" else 57.45
        z_start = -14.187
        z_end = 157.323

        # Dense rows at the hinge end emulate a real tessellated temple. This
        # catches pivot estimators that average a large 6%-length band.
        offsets = [0.0, 0.15, 0.35, 0.60, 0.90, 1.20]
        offsets.extend(np.linspace(3.0, z_end - z_start, 54).tolist())
        offsets = sorted(set(float(v) for v in offsets if v <= z_end - z_start))

        vertices = []
        for offset in offsets:
            z = z_start + offset
            y = -1.50 - 0.071 * offset
            for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                vertices.append([
                    x + 1.30 * np.cos(angle),
                    y + 7.37 * np.sin(angle),
                    z,
                ])

        temple = trimesh.Trimesh(vertices=np.asarray(vertices), faces=[], process=False)
        frame = trimesh.creation.box(extents=[123.7, 54.6, 5.2])
        return temple, frame

    def test_hinge_inference_uses_long_axis_and_frame_proximity_for_both_sides(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                temple, frame = self._z_axis_temple(side)
                pivot, axis = infer_temple_hinge(temple, frame, side)

                self.assertGreater(abs(float(axis[2])), 0.99)
                self.assertGreater(float(axis[2]), 0.0)
                expected_x = -57.45 if side == "left" else 57.45
                self.assertAlmostEqual(float(pivot[0]), expected_x, delta=0.2)
                self.assertAlmostEqual(float(pivot[1]), -1.50, delta=0.35)
                self.assertAlmostEqual(float(pivot[2]), -14.187, delta=0.45)

    def test_wrap_and_length_change_preserve_hinge_and_target_150_units(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                temple, frame = self._z_axis_temple(side)
                pivot, axis = infer_temple_hinge(temple, frame, side)
                before = temple.vertices.copy()
                initial_length = TempleDeformer._temple_length(before, pivot, axis)
                self.assertGreater(initial_length, 170.0)

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

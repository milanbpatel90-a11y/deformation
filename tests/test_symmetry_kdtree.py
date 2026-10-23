"""Correctness regression for KD-tree nearest-distance queries."""

import numpy as np

from backend.deformer.symmetry import SymmetrySolver


def test_kdtree_nearest_distances_matches_brute_force_reference():
    rng = np.random.default_rng(20261023)
    source = rng.normal(size=(137, 3))
    target = rng.normal(size=(211, 3))

    expected = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1).min(axis=1)
    actual = SymmetrySolver._nearest_distances(source, target)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

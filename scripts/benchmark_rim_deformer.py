"""Measure RimDeformer's spatial propagation on deterministic synthetic meshes.

Run with ``python -m scripts.benchmark_rim_deformer --vertices 61872``. This is
an algorithmic stress test, not a substitute for timing a labeled customer job.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import trimesh

from backend.deformer.rim_deformer import RimDeformer


def benchmark(vertices: int) -> dict:
    if vertices < 2:
        raise ValueError("vertices must be at least 2")
    x = np.linspace(-70.0, 70.0, vertices)
    rim = np.column_stack((x, 8.0 * np.sin(x / 9.0), np.zeros_like(x)))
    frame = np.column_stack((x, 8.0 * np.sin(x / 9.0) + 2.0, np.zeros_like(x)))
    deformed = rim.copy()
    deformed[:, 2] += 0.6 * np.cos(x / 13.0)
    mesh = trimesh.Trimesh(vertices=frame, faces=np.empty((0, 3), dtype=np.int64), process=False)
    start = time.perf_counter()
    RimDeformer._propagate_to_frame(mesh, rim, deformed)
    elapsed = time.perf_counter() - start
    return {
        "algorithm": "frame-to-rim cKDTree nearest query",
        "frame_vertices": vertices,
        "rim_vertices": vertices,
        "potential_all_pairs_comparisons": vertices * vertices,
        "elapsed_seconds": round(elapsed, 4),
        "maximum_frame_displacement_mm": float(np.linalg.norm(mesh.vertices - frame, axis=1).max()),
        "finite_output": bool(np.isfinite(mesh.vertices).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=61872)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.vertices), indent=2))


if __name__ == "__main__":
    main()

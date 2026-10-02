"""Export-only cleanup: never mutate basis topology or the input scene."""
from copy import deepcopy
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def world_vertices(scene, geometry_name):
    instances = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        if name == geometry_name:
            instances.append(trimesh.transform_points(scene.geometry[name].vertices, matrix))
    if not instances:
        raise ValueError(f"No scene instance for {geometry_name}")
    return np.vstack(instances)


def dimensions(scene):
    """Axis-aligned dimensions in scene units; temple extent is not arm arc length."""
    result = {}
    for name in ("Frame", "LeftLens", "RightLens", "LeftTemple", "RightTemple"):
        if name in scene.geometry:
            vertices = world_vertices(scene, name)
            result[name] = {"bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                            "extent": np.ptp(vertices, axis=0).tolist()}
    return result


def prepare_geometry(scene):
    """Weld coincident seams, fix closed shells, split sharp edges, export normals.

    No decimation: the surface and component names are preserved. Textured meshes
    retain their UV seams. Positional welding tolerance is <=1e-8 input units.
    """
    stats = {}
    for name, original in list(scene.geometry.items()):
        mesh = original.copy()
        before = len(mesh.vertices)
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
            raise ValueError(f"Invalid export mesh: {name}")
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-10))
        mesh.remove_unreferenced_vertices()
        textured = isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None
        # An artist's UV seams must remain independently addressable.
        if not textured:
            mesh.merge_vertices(merge_norm=True, digits_vertex=8)
            if mesh.is_watertight:
                mesh.fix_normals(multibody=True)
            elif name.startswith("Right") and "Left" + name[5:] in scene.geometry:
                # Confirm mirror correspondence and opposing normals, rather than
                # inferring outward winding from an open shell's signed volume.
                left = scene.geometry["Left" + name[5:]].copy()
                left.merge_vertices(merge_norm=True, digits_vertex=8)
                reflected = left.vertices * [-1, 1, 1]
                distances, index = cKDTree(reflected).query(mesh.vertices)
                dot = np.sum(mesh.vertex_normals * (left.vertex_normals[index] * [-1, 1, 1]), axis=1)
                if np.max(distances) < 0.001 and np.median(dot) < -0.99:
                    mesh.invert()
            mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(35), facet_minarea=None)
            mesh.metadata = deepcopy(original.metadata)
            if isinstance(original.visual, trimesh.visual.TextureVisuals):
                mesh.visual = trimesh.visual.TextureVisuals(material=deepcopy(original.visual.material))
            elif original.visual.kind == 'vertex':
                nearest = cKDTree(original.vertices).query(mesh.vertices)[1]
                mesh.visual.vertex_colors = original.visual.vertex_colors[nearest]
        # Explicitly populate fresh normals; export_glb(include_normals=True)
        # ensures they are written even when the cache was cleared upstream.
        normals = mesh.vertex_normals
        if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) < 0.5):
            raise ValueError(f"Invalid surface normals: {name}")
        scene.geometry[name] = mesh
        stats[name] = {"vertices_before": before, "vertices_after": len(mesh.vertices),
                       "triangles_before": len(original.faces), "triangles_after": len(mesh.faces)}
    return stats

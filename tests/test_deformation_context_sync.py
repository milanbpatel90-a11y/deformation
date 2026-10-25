import numpy as np
import trimesh

from backend.deformer.deformation_context import DeformationContext
from backend.deformer.descriptor_loader import TemplateDescriptor


def _descriptor():
    return TemplateDescriptor(
        template_name="sync_test",
        template_path=None,
        descriptor_path=None,
        metadata_path=None,
        hinges={},
        rim_loops={},
        bridge_center=np.zeros(3),
        empty_anchors={},
        temple_axis={},
        lens_planes={},
        vertex_groups={"frame": ["Frame"], "bridge": ["Bridge"]},
        constraints={},
    )


def test_sync_scene_exports_deformed_mesh_not_original_scene_graph_geometry():
    source = trimesh.Scene()
    source.add_geometry(
        trimesh.creation.box(extents=(10.0, 4.0, 2.0)),
        geom_name="Frame",
        node_name="Frame",
    )
    source.add_geometry(
        trimesh.creation.box(extents=(2.0, 2.0, 2.0)),
        geom_name="Bridge",
        node_name="Bridge",
        transform=trimesh.transformations.translation_matrix((8.0, 0.0, 0.0)),
    )
    original_source_bounds = source.bounds.copy()

    context = DeformationContext(
        template_info=None,
        template_scene=source,
        descriptor=_descriptor(),
        measurements=None,
    )
    frame = context.mesh("Frame")
    frame.vertices[:, 0] *= 2.0

    synced = context.sync_scene_from_meshes()

    assert set(synced.geometry) == {"Frame", "Bridge"}
    frame_width = float(np.ptp(synced.geometry["Frame"].vertices[:, 0]))
    assert frame_width == 20.0
    assert np.allclose(source.bounds, original_source_bounds)
    assert not np.allclose(synced.bounds, original_source_bounds)

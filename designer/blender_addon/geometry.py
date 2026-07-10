from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .config import DEFAULT_CONFIG
from .utils import BlenderOperationError, _require_blender, get_logger

try:
    import bmesh
    from bmesh.types import BMesh, BMFace
    from bpy.types import Mesh, Object
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree
except ImportError:  # pragma: no cover - Blender runtime only
    bmesh = None
    BMesh = object  # type: ignore[assignment]
    BMFace = object  # type: ignore[assignment]
    Mesh = object  # type: ignore[assignment]
    Object = object  # type: ignore[assignment]
    Matrix = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]
    BVHTree = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)
MeshLike = Union[Object, Mesh, BMesh]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding-box data."""

    min_corner: Vector
    max_corner: Vector
    center: Vector
    size: Vector


@dataclass(frozen=True, slots=True)
class NonManifoldReport:
    """Summary of non-manifold topology issues."""

    is_non_manifold: bool
    vertex_count: int
    edge_count: int
    face_count: int
    non_manifold_vertices: int
    non_manifold_edges: int
    non_manifold_ratio: float


@dataclass(frozen=True, slots=True)
class SelfIntersectionReport:
    """Summary of self-intersection detection."""

    has_self_intersections: bool
    intersecting_pairs: int
    intersecting_faces: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MeshRepairReport:
    """Summarizes mesh repair operations."""

    removed_duplicate_vertices: int
    removed_loose_elements: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    removed_zero_area_faces: int
    removed_internal_faces: int
    dissolved_collinear_edges: int
    merged_coplanar_faces: int
    triangulated: bool
    cleaned_custom_normals: bool
    non_manifold_after_repair: bool
    self_intersections_after_repair: bool


@dataclass(frozen=True, slots=True)
class _MeshContext:
    """Internal mesh resolution context."""

    mesh: Optional[Mesh]
    matrix_world: Matrix
    name: str


def _resolve_context(mesh_like: MeshLike) -> _MeshContext:
    """Resolves mesh data and transform context."""
    _require_blender()
    if isinstance(mesh_like, BMesh):
        return _MeshContext(mesh=None, matrix_world=Matrix.Identity(4), name="BMesh")
    if isinstance(mesh_like, Object):
        if mesh_like.type != "MESH":
            raise TypeError(f"Expected a mesh object, got '{mesh_like.type}'.")
        return _MeshContext(
            mesh=mesh_like.data,
            matrix_world=mesh_like.matrix_world.copy(),
            name=mesh_like.name,
        )
    if isinstance(mesh_like, Mesh):
        return _MeshContext(
            mesh=mesh_like,
            matrix_world=Matrix.Identity(4),
            name=mesh_like.name,
        )
    raise TypeError("Expected a Blender mesh object, Mesh, or BMesh instance.")


def _create_bmesh(
    mesh_like: MeshLike,
    world_space: bool = False,
) -> tuple[BMesh, bool, _MeshContext]:
    """Builds a BMesh view of mesh-like data."""
    context = _resolve_context(mesh_like)
    if isinstance(mesh_like, BMesh):
        bm = mesh_like
        owned = False
    else:
        bm = bmesh.new()
        bm.from_mesh(context.mesh)
        owned = True

    if world_space and context.matrix_world != Matrix.Identity(4):
        bmesh.ops.transform(bm, matrix=context.matrix_world, verts=bm.verts)
    return bm, owned, context


def _write_bmesh(bm: BMesh, context: _MeshContext) -> None:
    """Writes BMesh changes back to mesh data."""
    if context.mesh is None:
        bm.normal_update()
        return
    bm.normal_update()
    bm.to_mesh(context.mesh)
    context.mesh.update()


def _free_bmesh(bm: BMesh, owned: bool) -> None:
    """Frees temporary BMesh instances."""
    if owned:
        bm.free()


def _quantized_vector(
    vector: Vector,
    digits: int = 6,
) -> tuple[float, float, float]:
    """Returns a rounded vector tuple suitable for hashing."""
    return (
        round(float(vector.x), digits),
        round(float(vector.y), digits),
        round(float(vector.z), digits),
    )


def _face_signature(
    face: BMFace,
    digits: int = 6,
) -> tuple[tuple[float, float, float], ...]:
    """Builds a deterministic face signature from vertex coordinates."""
    return tuple(sorted(_quantized_vector(vert.co, digits) for vert in face.verts))


def _degenerate_faces(
    bm: BMesh,
    area_epsilon_m2: float,
) -> list[BMFace]:
    """Collects degenerate faces in a BMesh."""
    return [
        face
        for face in bm.faces
        if len(face.verts) < 3 or face.calc_area() <= area_epsilon_m2
    ]


def _triangulate_bmesh(bm: BMesh) -> bool:
    """Triangulates n-gons in a BMesh when needed."""
    faces = [face for face in bm.faces if len(face.verts) > 3]
    if not faces:
        return False
    bmesh.ops.triangulate(
        bm,
        faces=faces,
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    return True


def merge_duplicate_vertices(
    mesh_like: MeshLike,
    distance_mm: Optional[float] = None,
) -> int:
    """Merges vertices closer than the configured threshold."""
    threshold_mm = distance_mm or DEFAULT_CONFIG.validation.merge_distance_mm
    threshold_m = threshold_mm / 1000.0
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        before_count = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=threshold_m)
        removed = before_count - len(bm.verts)
        _write_bmesh(bm, context)
        return removed
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to merge duplicate vertices for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def recalculate_normals(mesh_like: MeshLike, inside: bool = False) -> None:
    """Recalculates mesh normals."""
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        faces = list(bm.faces)
        if faces:
            bmesh.ops.recalc_face_normals(bm, faces=faces)
            if inside:
                bmesh.ops.reverse_faces(bm, faces=faces)
        _write_bmesh(bm, context)
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to recalculate normals for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def remove_loose_geometry(mesh_like: MeshLike) -> int:
    """Deletes isolated edges and vertices."""
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        loose_verts = [vert for vert in bm.verts if not vert.link_edges]
        removed = len(loose_edges) + len(loose_verts)
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
        _write_bmesh(bm, context)
        return removed
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove loose geometry for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def triangulate_if_needed(mesh_like: MeshLike) -> bool:
    """Triangulates non-triangular faces."""
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        changed = _triangulate_bmesh(bm)
        if changed:
            _write_bmesh(bm, context)
        return changed
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to triangulate mesh '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def remove_degenerate_faces(mesh_like: MeshLike) -> int:
    """Removes faces with insufficient vertices or near-zero area."""
    epsilon_m2 = DEFAULT_CONFIG.validation.degenerate_face_area_mm2 / 1_000_000.0
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        faces = _degenerate_faces(bm, epsilon_m2)
        if not faces:
            return 0
        count = len(faces)
        bmesh.ops.delete(bm, geom=faces, context="FACES")
        _write_bmesh(bm, context)
        return count
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove degenerate faces for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def remove_duplicate_faces(mesh_like: MeshLike) -> int:
    """Removes faces occupying the same geometric footprint."""
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        duplicate_faces: list[BMFace] = []
        seen: dict[tuple[tuple[float, float, float], ...], BMFace] = {}
        for face in bm.faces:
            signature = _face_signature(face)
            existing = seen.get(signature)
            if existing is None:
                seen[signature] = face
            else:
                duplicate_faces.append(face)

        if not duplicate_faces:
            return 0
        count = len(duplicate_faces)
        bmesh.ops.delete(bm, geom=duplicate_faces, context="FACES")
        _write_bmesh(bm, context)
        return count
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove duplicate faces for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def remove_zero_area_faces(mesh_like: MeshLike) -> int:
    """Removes zero-area faces."""
    epsilon_m2 = DEFAULT_CONFIG.validation.degenerate_face_area_mm2 / 1_000_000.0
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        faces = [face for face in bm.faces if face.calc_area() <= epsilon_m2]
        if not faces:
            return 0
        count = len(faces)
        bmesh.ops.delete(bm, geom=faces, context="FACES")
        _write_bmesh(bm, context)
        return count
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove zero-area faces for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def remove_internal_faces(
    mesh_like: MeshLike,
    distance_epsilon_mm: float = 0.05,
) -> int:
    """Removes faces likely buried inside the shell."""
    epsilon_m = max(distance_epsilon_mm / 1000.0, 1e-6)
    bm, owned, context = _create_bmesh(mesh_like)
    try:
        bm.faces.ensure_lookup_table()
        if len(bm.faces) < 2:
            return 0

        bvh = BVHTree.FromBMesh(bm, epsilon=epsilon_m)
        faces_to_delete: set[BMFace] = set()
        for face in bm.faces:
            center = face.calc_center_median()
            normal = face.normal.normalized()
            forward = bvh.ray_cast(center + (normal * epsilon_m * 2.0), normal)
            backward = bvh.ray_cast(center - (normal * epsilon_m * 2.0), -normal)
            if forward[2] is None or backward[2] is None:
                continue
            if forward[2] == face.index or backward[2] == face.index:
                continue
            if forward[3] > epsilon_m * 20.0 or backward[3] > epsilon_m * 20.0:
                continue
            faces_to_delete.add(face)

        if not faces_to_delete:
            return 0
        count = len(faces_to_delete)
        bmesh.ops.delete(bm, geom=list(faces_to_delete), context="FACES")
        _write_bmesh(bm, context)
        return count
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to remove internal faces for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def dissolve_collinear_edges(
    mesh_like: MeshLike,
    angle_limit_degrees: float = 0.5,
) -> int:
    """Dissolves nearly collinear edges."""
    from math import radians

    bm, owned, context = _create_bmesh(mesh_like)
    try:
        before_edges = len(bm.edges)
        bmesh.ops.dissolve_limit(
            bm,
            angle_limit=radians(angle_limit_degrees),
            use_dissolve_boundaries=False,
            verts=list(bm.verts),
            edges=list(bm.edges),
        )
        dissolved = max(before_edges - len(bm.edges), 0)
        _write_bmesh(bm, context)
        return dissolved
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to dissolve collinear edges for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def merge_coplanar_faces(
    mesh_like: MeshLike,
    angle_limit_degrees: float = 1.0,
) -> int:
    """Merges adjacent coplanar faces by dissolving shared edges."""
    from math import radians

    bm, owned, context = _create_bmesh(mesh_like)
    try:
        before_faces = len(bm.faces)
        bmesh.ops.dissolve_limit(
            bm,
            angle_limit=radians(angle_limit_degrees),
            use_dissolve_boundaries=False,
            verts=list(bm.verts),
            edges=list(bm.edges),
        )
        merged = max(before_faces - len(bm.faces), 0)
        _write_bmesh(bm, context)
        return merged
    except Exception as exc:  # pragma: no cover - Blender runtime only
        raise BlenderOperationError(
            f"Failed to merge coplanar faces for '{context.name}'."
        ) from exc
    finally:
        _free_bmesh(bm, owned)


def calculate_surface_area(
    mesh_like: MeshLike,
    world_space: bool = True,
) -> float:
    """Computes total mesh surface area in square meters."""
    bm, owned, _ = _create_bmesh(mesh_like, world_space=world_space)
    try:
        return float(sum(face.calc_area() for face in bm.faces))
    finally:
        _free_bmesh(bm, owned)


def calculate_volume(
    mesh_like: MeshLike,
    world_space: bool = True,
) -> float:
    """Computes closed mesh volume in cubic meters."""
    bm, owned, _ = _create_bmesh(mesh_like, world_space=world_space)
    try:
        _triangulate_bmesh(bm)
        return float(abs(bm.calc_volume(signed=True)))
    finally:
        _free_bmesh(bm, owned)


def calculate_dimensions_mm(
    mesh_like: MeshLike,
    world_space: bool = True,
) -> Tuple[float, float, float]:
    """Calculates bounding-box dimensions in millimeters."""
    bbox = calculate_bounding_box(mesh_like, world_space=world_space)
    return (
        float(bbox.size.x * 1000.0),
        float(bbox.size.y * 1000.0),
        float(bbox.size.z * 1000.0),
    )


def calculate_center_of_mass(
    mesh_like: MeshLike,
    world_space: bool = True,
) -> Vector:
    """Calculates mesh center of mass."""
    bm, owned, _ = _create_bmesh(mesh_like, world_space=world_space)
    try:
        if not bm.faces:
            if not bm.verts:
                return Vector((0.0, 0.0, 0.0))
            total = Vector((0.0, 0.0, 0.0))
            for vert in bm.verts:
                total += vert.co
            return total / len(bm.verts)

        _triangulate_bmesh(bm)
        total_volume = 0.0
        weighted_center = Vector((0.0, 0.0, 0.0))
        origin = Vector((0.0, 0.0, 0.0))
        for face in bm.faces:
            verts = [vert.co.copy() for vert in face.verts]
            if len(verts) != 3:
                continue
            tetra_volume = verts[0].dot((verts[1] - origin).cross(verts[2] - origin)) / 6.0
            centroid = (origin + verts[0] + verts[1] + verts[2]) / 4.0
            weighted_center += centroid * tetra_volume
            total_volume += tetra_volume

        if abs(total_volume) > 1e-12:
            return weighted_center / total_volume

        total = Vector((0.0, 0.0, 0.0))
        for vert in bm.verts:
            total += vert.co
        return total / max(len(bm.verts), 1)
    finally:
        _free_bmesh(bm, owned)


def calculate_bounding_box(
    mesh_like: MeshLike,
    world_space: bool = True,
) -> BoundingBox:
    """Calculates an axis-aligned bounding box."""
    bm, owned, _ = _create_bmesh(mesh_like, world_space=world_space)
    try:
        if not bm.verts:
            zero = Vector((0.0, 0.0, 0.0))
            return BoundingBox(min_corner=zero, max_corner=zero, center=zero, size=zero)

        min_corner = Vector(
            (
                min(vert.co.x for vert in bm.verts),
                min(vert.co.y for vert in bm.verts),
                min(vert.co.z for vert in bm.verts),
            )
        )
        max_corner = Vector(
            (
                max(vert.co.x for vert in bm.verts),
                max(vert.co.y for vert in bm.verts),
                max(vert.co.z for vert in bm.verts),
            )
        )
        size = max_corner - min_corner
        center = (min_corner + max_corner) * 0.5
        return BoundingBox(
            min_corner=min_corner,
            max_corner=max_corner,
            center=center,
            size=size,
        )
    finally:
        _free_bmesh(bm, owned)


def check_non_manifold(mesh_like: MeshLike) -> NonManifoldReport:
    """Checks for non-manifold edges and vertices."""
    bm, owned, _ = _create_bmesh(mesh_like)
    try:
        non_manifold_edges = [edge for edge in bm.edges if not edge.is_manifold]
        non_manifold_vertices = [vert for vert in bm.verts if not vert.is_manifold]
        total_elements = max(len(bm.edges) + len(bm.verts), 1)
        ratio = (len(non_manifold_edges) + len(non_manifold_vertices)) / total_elements
        return NonManifoldReport(
            is_non_manifold=bool(non_manifold_edges or non_manifold_vertices),
            vertex_count=len(bm.verts),
            edge_count=len(bm.edges),
            face_count=len(bm.faces),
            non_manifold_vertices=len(non_manifold_vertices),
            non_manifold_edges=len(non_manifold_edges),
            non_manifold_ratio=float(ratio),
        )
    finally:
        _free_bmesh(bm, owned)


def check_self_intersections(
    mesh_like: MeshLike,
    epsilon_mm: float = 0.01,
) -> SelfIntersectionReport:
    """Detects self-intersecting faces using BVH overlap."""
    epsilon_m = epsilon_mm / 1000.0
    bm, owned, _ = _create_bmesh(mesh_like)
    try:
        bm.faces.ensure_lookup_table()
        if not bm.faces:
            return SelfIntersectionReport(False, 0, ())

        bvh = BVHTree.FromBMesh(bm, epsilon=epsilon_m)
        overlaps = bvh.overlap(bvh)
        filtered_pairs: set[tuple[int, int]] = set()
        intersecting_faces: set[int] = set()
        for face_index_a, face_index_b in overlaps:
            if face_index_a == face_index_b:
                continue
            face_a = bm.faces[face_index_a]
            face_b = bm.faces[face_index_b]
            shared_verts = {vert.index for vert in face_a.verts}.intersection(
                vert.index for vert in face_b.verts
            )
            if len(shared_verts) >= 2:
                continue
            pair = tuple(sorted((face_index_a, face_index_b)))
            filtered_pairs.add(pair)
            intersecting_faces.update(pair)

        return SelfIntersectionReport(
            has_self_intersections=bool(filtered_pairs),
            intersecting_pairs=len(filtered_pairs),
            intersecting_faces=tuple(sorted(intersecting_faces)),
        )
    finally:
        _free_bmesh(bm, owned)


def repair_mesh(mesh_like: MeshLike) -> MeshRepairReport:
    """Runs the standard repair pipeline on a mesh."""
    cleaned_custom_normals = clean_custom_normals(mesh_like)
    removed_duplicate_vertices = merge_duplicate_vertices(mesh_like)
    removed_degenerate_faces = remove_degenerate_faces(mesh_like)
    removed_duplicate_faces = remove_duplicate_faces(mesh_like)
    removed_zero_area_faces = remove_zero_area_faces(mesh_like)
    removed_internal_faces = remove_internal_faces(mesh_like)
    dissolved_collinear_edges = dissolve_collinear_edges(mesh_like)
    merged_coplanar_faces = merge_coplanar_faces(mesh_like)
    removed_loose_elements = remove_loose_geometry(mesh_like)
    triangulated = triangulate_if_needed(mesh_like)
    recalculate_normals(mesh_like)
    non_manifold = check_non_manifold(mesh_like)
    intersections = check_self_intersections(mesh_like)
    return MeshRepairReport(
        removed_duplicate_vertices=removed_duplicate_vertices,
        removed_loose_elements=removed_loose_elements,
        removed_degenerate_faces=removed_degenerate_faces,
        removed_duplicate_faces=removed_duplicate_faces,
        removed_zero_area_faces=removed_zero_area_faces,
        removed_internal_faces=removed_internal_faces,
        dissolved_collinear_edges=dissolved_collinear_edges,
        merged_coplanar_faces=merged_coplanar_faces,
        triangulated=triangulated,
        cleaned_custom_normals=cleaned_custom_normals,
        non_manifold_after_repair=non_manifold.is_non_manifold,
        self_intersections_after_repair=intersections.has_self_intersections,
    )


def clean_custom_normals(mesh_like: MeshLike) -> bool:
    """Clears problematic custom normals and validates mesh custom data."""
    context = _resolve_context(mesh_like)
    if context.mesh is None:
        return False

    mesh = context.mesh
    changed = False
    try:
        if getattr(mesh, "has_custom_normals", False):
            split_normals = [(0.0, 0.0, 0.0)] * len(mesh.loops)
            mesh.normals_split_custom_set(split_normals)
            changed = True
    except Exception:
        LOGGER.debug("Unable to reset custom normals for '%s'.", context.name)

    validated = mesh.validate(verbose=False, clean_customdata=True)
    mesh.update()
    return bool(changed or validated)


__all__ = [
    "BoundingBox",
    "MeshRepairReport",
    "NonManifoldReport",
    "SelfIntersectionReport",
    "calculate_bounding_box",
    "calculate_center_of_mass",
    "calculate_dimensions_mm",
    "calculate_surface_area",
    "calculate_volume",
    "check_non_manifold",
    "check_self_intersections",
    "clean_custom_normals",
    "dissolve_collinear_edges",
    "merge_coplanar_faces",
    "merge_duplicate_vertices",
    "recalculate_normals",
    "remove_degenerate_faces",
    "remove_duplicate_faces",
    "remove_internal_faces",
    "remove_loose_geometry",
    "remove_zero_area_faces",
    "repair_mesh",
    "triangulate_if_needed",
]

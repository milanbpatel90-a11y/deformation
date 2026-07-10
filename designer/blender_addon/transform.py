from __future__ import annotations

from typing import Optional, Sequence

from .config import DEFAULT_CONFIG
from .geometry import calculate_dimensions_mm
from .utils import _require_blender, center_origin as _center_origin, get_logger

try:
    from bpy.types import Object
    from mathutils import Euler, Matrix, Quaternion, Vector
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]
    Euler = object  # type: ignore[assignment]
    Matrix = object  # type: ignore[assignment]
    Quaternion = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


def _require_mesh_object(object_: Object) -> None:
    """Validates mesh object input."""
    _require_blender()
    if object_.type != "MESH":
        raise TypeError(f"Expected a mesh object, got '{object_.type}'.")


def _axis_vector(axis: str) -> Vector:
    """Returns a normalized vector for a symbolic axis name."""
    axis_map = {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
        "-X": Vector((-1.0, 0.0, 0.0)),
        "-Y": Vector((0.0, -1.0, 0.0)),
        "-Z": Vector((0.0, 0.0, -1.0)),
    }
    key = axis.strip().upper()
    if key not in axis_map:
        raise ValueError(f"Unsupported axis '{axis}'. Expected one of {tuple(axis_map)}.")
    return axis_map[key].copy()


def _rotation_matrix_from_value(rotation) -> Matrix:
    """Normalizes supported rotation inputs to a 4x4 matrix."""
    if rotation is None:
        return Matrix.Identity(4)
    if isinstance(rotation, Matrix):
        return rotation.to_4x4()
    if isinstance(rotation, Euler):
        return rotation.to_matrix().to_4x4()
    if isinstance(rotation, Quaternion):
        return rotation.to_matrix().to_4x4()
    if isinstance(rotation, Sequence):
        if len(rotation) == 3:
            euler = Euler(tuple(float(value) for value in rotation), "XYZ")
            return euler.to_matrix().to_4x4()
        if len(rotation) == 4:
            quaternion = Quaternion(tuple(float(value) for value in rotation))
            return quaternion.to_matrix().to_4x4()
    raise TypeError("Unsupported rotation input.")


def center_origin(object_: Object) -> None:
    """Moves the object origin to the mesh bounds center."""
    _require_mesh_object(object_)
    _center_origin(object_)


def move_to_world_origin(object_: Object) -> None:
    """Moves an object to the world origin without altering geometry."""
    _require_mesh_object(object_)
    object_.matrix_world.translation = Vector((0.0, 0.0, 0.0))


def align_forward_axis(
    object_: Object,
    source_axis: str = "Y",
    target_axis: str = "-Y",
) -> None:
    """Rotates an object so its source axis points to a target world axis."""
    _require_mesh_object(object_)
    local_axis = _axis_vector(source_axis)
    target_vector = _axis_vector(target_axis)
    current_vector = (object_.matrix_world.to_3x3() @ local_axis).normalized()
    rotation = current_vector.rotation_difference(target_vector).to_matrix().to_4x4()
    object_.matrix_world = rotation @ object_.matrix_world


def align_up_axis(
    object_: Object,
    source_axis: str = "Z",
    target_axis: str = "Z",
) -> None:
    """Rotates an object so its source up axis matches a target world axis."""
    _require_mesh_object(object_)
    local_axis = _axis_vector(source_axis)
    target_vector = _axis_vector(target_axis)
    current_vector = (object_.matrix_world.to_3x3() @ local_axis).normalized()
    rotation = current_vector.rotation_difference(target_vector).to_matrix().to_4x4()
    object_.matrix_world = rotation @ object_.matrix_world


def apply_location(object_: Object) -> None:
    """Applies object location to mesh data."""
    _require_mesh_object(object_)
    translation = Matrix.Translation(object_.location.copy())
    object_.data.transform(translation)
    object_.location = Vector((0.0, 0.0, 0.0))
    object_.data.update()


def apply_rotation(object_: Object) -> None:
    """Applies object rotation to mesh data."""
    _require_mesh_object(object_)
    rotation_matrix = object_.matrix_basis.to_3x3().normalized().to_4x4()
    object_.data.transform(rotation_matrix)
    if object_.rotation_mode == "QUATERNION":
        object_.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    elif object_.rotation_mode == "AXIS_ANGLE":
        object_.rotation_axis_angle = (0.0, 0.0, 0.0, 1.0)
    else:
        object_.rotation_euler = Euler((0.0, 0.0, 0.0), object_.rotation_mode)
    object_.data.update()


def apply_scale(object_: Object) -> None:
    """Applies object scale to mesh data."""
    _require_mesh_object(object_)
    scale_matrix = Matrix.Diagonal((object_.scale.x, object_.scale.y, object_.scale.z, 1.0))
    object_.data.transform(scale_matrix)
    object_.scale = Vector((1.0, 1.0, 1.0))
    object_.data.update()


def normalize_scale_to_width(
    object_: Object,
    target_width_mm: float = DEFAULT_CONFIG.validation.default_target_width_mm,
) -> float:
    """Uniformly scales an object to match a target width."""
    _require_mesh_object(object_)
    dimensions = calculate_dimensions_mm(object_, world_space=True)
    current_width = max(dimensions[0], 1e-9)
    factor = target_width_mm / current_width
    object_.scale *= factor
    return float(factor)


def freeze_transform(object_: Object) -> None:
    """Applies location, rotation, and scale to mesh data."""
    _require_mesh_object(object_)
    matrix = object_.matrix_basis.copy()
    object_.data.transform(matrix)
    object_.location = Vector((0.0, 0.0, 0.0))
    object_.scale = Vector((1.0, 1.0, 1.0))
    if object_.rotation_mode == "QUATERNION":
        object_.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    elif object_.rotation_mode == "AXIS_ANGLE":
        object_.rotation_axis_angle = (0.0, 0.0, 0.0, 1.0)
    else:
        object_.rotation_euler = Euler((0.0, 0.0, 0.0), object_.rotation_mode)
    object_.data.update()


def rotate_to_front_view(object_: Object) -> None:
    """Aligns an imported object to the add-on front-view convention."""
    _require_mesh_object(object_)
    align_forward_axis(object_, source_axis="Y", target_axis="-Y")
    align_up_axis(object_, source_axis="Z", target_axis="Z")


def reset_transform(object_: Object) -> None:
    """Resets location, rotation, and scale without editing geometry."""
    _require_mesh_object(object_)
    object_.location = Vector((0.0, 0.0, 0.0))
    object_.scale = Vector((1.0, 1.0, 1.0))
    if object_.rotation_mode == "QUATERNION":
        object_.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    elif object_.rotation_mode == "AXIS_ANGLE":
        object_.rotation_axis_angle = (0.0, 0.0, 0.0, 1.0)
    else:
        object_.rotation_euler = Euler((0.0, 0.0, 0.0), object_.rotation_mode)


def compute_transform_matrix(
    location: Optional[Sequence[float]] = None,
    rotation: Optional[object] = None,
    scale: Optional[Sequence[float]] = None,
) -> Matrix:
    """Builds a 4x4 transform matrix from components."""
    _require_blender()
    matrix = Matrix.Identity(4)
    if location is not None:
        if len(location) != 3:
            raise ValueError("Location must contain exactly three values.")
        matrix @= Matrix.Translation(Vector(tuple(float(value) for value in location)))
    matrix @= _rotation_matrix_from_value(rotation)
    if scale is not None:
        if len(scale) != 3:
            raise ValueError("Scale must contain exactly three values.")
        matrix @= Matrix.Diagonal(tuple(float(value) for value in scale) + (1.0,))
    return matrix


__all__ = [
    "align_forward_axis",
    "align_up_axis",
    "apply_location",
    "apply_rotation",
    "apply_scale",
    "center_origin",
    "compute_transform_matrix",
    "freeze_transform",
    "move_to_world_origin",
    "normalize_scale_to_width",
    "reset_transform",
    "rotate_to_front_view",
]

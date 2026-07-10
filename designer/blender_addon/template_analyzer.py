from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .geometry import (
    BoundingBox,
    calculate_bounding_box,
    calculate_dimensions_mm,
    check_non_manifold,
)
from .materials import validate_materials
from .mesh_splitter import (
    BRIDGE,
    FRAME,
    LEFT_LENS,
    LEFT_PAD,
    LEFT_TEMPLE,
    RIGHT_LENS,
    RIGHT_PAD,
    RIGHT_TEMPLE,
    MeshPart,
    split_eyewear_mesh,
)
from .utils import _require_blender, count_triangles, count_vertices, get_logger

try:
    from bpy.types import Object
    from mathutils import Vector
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]
    Vector = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TemplateAnalysis:
    """Complete analysis result for an eyewear template."""

    style: str
    frame_family: str
    material: str
    bridge_type: str
    rim_type: str
    dimensions: dict[str, float]
    symmetry: float
    triangle_count: int
    vertex_count: int
    bounding_box: dict[str, tuple[float, float, float]]
    quality_score: int
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)


class TemplateAnalyzer:
    """Analyzes imported eyewear meshes and derives template metadata."""

    def __init__(self, objects: Optional[Iterable[Object]] = None) -> None:
        """Initializes the analyzer."""
        _require_blender()
        if objects is None:
            bpy_module = __import__("bpy")
            self.objects = [object_ for object_ in bpy_module.context.scene.objects if object_.type == "MESH"]
        else:
            self.objects = [object_ for object_ in objects if object_.type == "MESH"]
        self.parts = split_eyewear_mesh(self.objects)

    def analyze(self) -> TemplateAnalysis:
        """Runs the complete template analysis pipeline."""
        bbox = self.calculate_dimensions()
        frame_family = self.detect_frame_family()
        material = self.detect_material()
        bridge_type = self.detect_bridge_type()
        rim_type = self.detect_rim_type()
        style = f"{frame_family}_{material}".lower()
        symmetry = self.calculate_symmetry()
        triangle_count = sum(count_triangles(object_) for object_ in self.objects)
        vertex_count = sum(count_vertices(object_) for object_ in self.objects)
        quality_score = self.estimate_quality_score()
        return TemplateAnalysis(
            style=style,
            frame_family=frame_family,
            material=material,
            bridge_type=bridge_type,
            rim_type=rim_type,
            dimensions={
                "frame_width": self.calculate_frame_width(),
                "frame_height": self.calculate_frame_height(),
                "lens_width": self.calculate_lens_width(),
                "lens_height": self.calculate_lens_height(),
                "bridge_width": self.calculate_bridge_width(),
                "temple_length": self.calculate_temple_length(),
                "frame_thickness": self.estimate_frame_thickness(),
                "vertex_density": self.estimate_vertex_density(),
            },
            symmetry=symmetry,
            triangle_count=triangle_count,
            vertex_count=vertex_count,
            bounding_box={
                "min": tuple(float(value) for value in bbox.min_corner),
                "max": tuple(float(value) for value in bbox.max_corner),
                "size": tuple(float(value) for value in bbox.size),
            },
            quality_score=quality_score,
            parts={
                name: {
                    "object_count": part.object_count,
                    "confidence": part.confidence,
                    "metadata": part.metadata,
                }
                for name, part in self.parts.items()
            },
        )

    def detect_frame(self) -> MeshPart:
        """Returns the detected frame/front part."""
        return self.parts[FRAME]

    def detect_lenses(self) -> tuple[MeshPart, MeshPart]:
        """Returns detected lens parts."""
        return self.parts[LEFT_LENS], self.parts[RIGHT_LENS]

    def detect_bridge(self) -> MeshPart:
        """Returns the detected bridge part."""
        return self.parts[BRIDGE]

    def detect_temples(self) -> tuple[MeshPart, MeshPart]:
        """Returns detected temple parts."""
        return self.parts[LEFT_TEMPLE], self.parts[RIGHT_TEMPLE]

    def detect_nose_pads(self) -> tuple[MeshPart, MeshPart]:
        """Returns detected nose pad parts."""
        return self.parts[LEFT_PAD], self.parts[RIGHT_PAD]

    def detect_frame_family(self) -> str:
        """Infers the eyewear frame family from aspect ratio and silhouette."""
        width = self.calculate_frame_width()
        height = self.calculate_frame_height()
        ratio = width / max(height, 1e-6)
        bridge = self.detect_bridge()
        if ratio >= 1.65:
            return "rectangle"
        if ratio >= 1.35:
            return "wayfarer"
        if ratio >= 1.1:
            return "square"
        if bridge.object_count and ratio < 1.05:
            return "round"
        return "oval"

    def detect_material(self) -> str:
        """Infers the dominant frame material."""
        materials = validate_materials(self.objects)
        if materials.material_count == 0:
            return "unknown"

        keywords = {
            "metal": ("metal", "steel", "gold", "silver", "titanium"),
            "acetate": ("acetate", "plastic", "horn", "tortoise"),
            "rimless": ("rimless", "nylon"),
        }
        scores = {"metal": 0, "acetate": 0, "rimless": 0}
        for object_ in self.objects:
            for slot in object_.material_slots:
                material = slot.material
                if material is None:
                    continue
                material_name = material.name.lower()
                for label, terms in keywords.items():
                    if any(term in material_name for term in terms):
                        scores[label] += 1
                if material.use_nodes and material.node_tree is not None:
                    for node in material.node_tree.nodes:
                        if node.type != "BSDF_PRINCIPLED":
                            continue
                        metallic = float(node.inputs["Metallic"].default_value)
                        transmission = (
                            float(node.inputs["Transmission Weight"].default_value)
                            if "Transmission Weight" in node.inputs
                            else 0.0
                        )
                        if metallic > 0.3:
                            scores["metal"] += 2
                        if transmission > 0.2:
                            scores["rimless"] += 1
                        if metallic < 0.15 and transmission < 0.1:
                            scores["acetate"] += 1
        return max(scores, key=scores.get)

    def detect_bridge_type(self) -> str:
        """Infers the bridge type from bridge geometry."""
        bridge_part = self.detect_bridge()
        if not bridge_part.objects:
            return "standard"
        bbox = calculate_bounding_box(bridge_part.objects[0], world_space=True)
        if bbox.size.z > bbox.size.y * 0.9:
            return "saddle"
        if bbox.center.z < self.calculate_dimensions().center.z - self.calculate_dimensions().size.z * 0.08:
            return "keyhole"
        return "standard"

    def detect_rim_type(self) -> str:
        """Infers rim type from lens and frame relationships."""
        frame = self.detect_frame()
        left_lens, right_lens = self.detect_lenses()
        if not left_lens.objects and not right_lens.objects:
            return "full_rim"
        if self.detect_material() == "rimless":
            return "rimless"
        frame_triangles = sum(count_triangles(object_) for object_ in frame.objects)
        lens_triangles = sum(count_triangles(object_) for object_ in left_lens.objects + right_lens.objects)
        if lens_triangles > 0 and frame_triangles < lens_triangles * 0.6:
            return "semi_rimless"
        return "full_rim"

    def calculate_symmetry(self) -> float:
        """Estimates left/right symmetry on the X axis."""
        bbox = self.calculate_dimensions()
        center_x = bbox.center.x
        left_weight = 0.0
        right_weight = 0.0
        for object_ in self.objects:
            object_bbox = calculate_bounding_box(object_, world_space=True)
            volume_proxy = float(object_bbox.size.x * object_bbox.size.y * object_bbox.size.z)
            if object_bbox.center.x < center_x:
                left_weight += volume_proxy
            else:
                right_weight += volume_proxy
        if left_weight + right_weight == 0.0:
            return 1.0
        return max(0.0, 1.0 - (abs(left_weight - right_weight) / (left_weight + right_weight)))

    def calculate_dimensions(self) -> BoundingBox:
        """Calculates the combined template bounding box."""
        bbox = calculate_bounding_box(self.objects[0], world_space=True)
        for object_ in self.objects[1:]:
            object_bbox = calculate_bounding_box(object_, world_space=True)
            min_corner = Vector((
                min(bbox.min_corner.x, object_bbox.min_corner.x),
                min(bbox.min_corner.y, object_bbox.min_corner.y),
                min(bbox.min_corner.z, object_bbox.min_corner.z),
            ))
            max_corner = Vector((
                max(bbox.max_corner.x, object_bbox.max_corner.x),
                max(bbox.max_corner.y, object_bbox.max_corner.y),
                max(bbox.max_corner.z, object_bbox.max_corner.z),
            ))
            size = max_corner - min_corner
            center = (min_corner + max_corner) * 0.5
            bbox = BoundingBox(min_corner=min_corner, max_corner=max_corner, center=center, size=size)
        return bbox

    def calculate_frame_width(self) -> float:
        """Returns overall frame width in millimeters."""
        return float(self.calculate_dimensions().size.x * 1000.0)

    def calculate_frame_height(self) -> float:
        """Returns overall frame height in millimeters."""
        return float(self.calculate_dimensions().size.z * 1000.0)

    def calculate_lens_width(self) -> float:
        """Returns average lens width in millimeters."""
        widths = [
            calculate_dimensions_mm(object_, world_space=True)[0]
            for part in self.detect_lenses()
            for object_ in part.objects
        ]
        return float(sum(widths) / len(widths)) if widths else 0.0

    def calculate_lens_height(self) -> float:
        """Returns average lens height in millimeters."""
        heights = [
            calculate_dimensions_mm(object_, world_space=True)[2]
            for part in self.detect_lenses()
            for object_ in part.objects
        ]
        return float(sum(heights) / len(heights)) if heights else 0.0

    def calculate_bridge_width(self) -> float:
        """Returns bridge width in millimeters."""
        bridge = self.detect_bridge()
        widths = [
            calculate_dimensions_mm(object_, world_space=True)[0]
            for object_ in bridge.objects
        ]
        if widths:
            return float(sum(widths) / len(widths))
        left_lens, right_lens = self.detect_lenses()
        if left_lens.objects and right_lens.objects:
            left_bbox = calculate_bounding_box(left_lens.objects[0], world_space=True)
            right_bbox = calculate_bounding_box(right_lens.objects[0], world_space=True)
            return float((right_bbox.min_corner.x - left_bbox.max_corner.x) * 1000.0)
        return 0.0

    def calculate_temple_length(self) -> float:
        """Returns average temple length in millimeters."""
        lengths = []
        for part in self.detect_temples():
            for object_ in part.objects:
                dims = calculate_dimensions_mm(object_, world_space=True)
                lengths.append(max(dims[0], dims[1], dims[2]))
        return float(sum(lengths) / len(lengths)) if lengths else 0.0

    def estimate_frame_thickness(self) -> float:
        """Estimates front frame thickness in millimeters."""
        frame = self.detect_frame()
        if not frame.objects:
            return 0.0
        thicknesses = [
            min(calculate_dimensions_mm(object_, world_space=True))
            for object_ in frame.objects
        ]
        return float(sum(thicknesses) / len(thicknesses)) if thicknesses else 0.0

    def estimate_vertex_density(self) -> float:
        """Estimates vertex density per cubic millimeter."""
        bbox = self.calculate_dimensions()
        volume_mm3 = max(float(bbox.size.x * bbox.size.y * bbox.size.z) * 1_000_000_000.0, 1.0)
        vertices = sum(count_vertices(object_) for object_ in self.objects)
        return float(vertices / volume_mm3)

    def estimate_quality_score(self) -> int:
        """Estimates a quality score from topology and material metrics."""
        score = 100
        non_manifold_penalty = 0
        for object_ in self.objects:
            report = check_non_manifold(object_)
            if report.is_non_manifold:
                non_manifold_penalty += min(int(report.non_manifold_ratio * 100) + 5, 25)
        score -= non_manifold_penalty
        symmetry = self.calculate_symmetry()
        score -= int((1.0 - symmetry) * 20.0)
        vertex_density = self.estimate_vertex_density()
        if vertex_density < 1e-6:
            score -= 10
        if self.detect_rim_type() == "rimless":
            score -= 2
        return max(min(score, 100), 0)


__all__ = ["TemplateAnalysis", "TemplateAnalyzer"]

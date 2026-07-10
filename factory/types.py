"""Shared dataclasses and enums for the Blender Template Factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Geometry / topology enums
# ---------------------------------------------------------------------------

class PartKind(str, Enum):
    """Semantic classification of an eyewear mesh component."""

    FRAME = "frame"
    LEFT_LENS = "left_lens"
    RIGHT_LENS = "right_lens"
    LEFT_RIM = "left_rim"
    RIGHT_RIM = "right_rim"
    BRIDGE = "bridge"
    LEFT_TEMPLE = "left_temple"
    RIGHT_TEMPLE = "right_temple"
    TEMPLE_TIPS = "temple_tips"
    NOSE_PADS = "nose_pads"
    UNKNOWN = "unknown"


class Shape(str, Enum):
    """High-level frame shape used for metadata."""

    RECTANGLE = "rectangle"
    ROUND = "round"
    OVAL = "oval"
    SQUARE = "square"
    CAT_EYE = "cat_eye"
    AVIATOR = "aviator"
    BROWLINE = "browline"
    CLUBMASTER = "clubmaster"
    GEOMETRIC = "geometric"
    RIMLESS = "rimless"
    OVERSIZED = "oversized"
    WAYFARER = "wayfarer"
    OTHER = "other"


class Material(str, Enum):
    """Frame material family."""

    ACETATE = "acetate"
    METAL = "metal"
    TITANIUM = "titanium"
    PLASTIC = "plastic"
    TR90 = "tr90"
    WOOD = "wood"
    CARBON = "carbon"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RimType(str, Enum):
    FULL_RIM = "full_rim"
    SEMI_RIMLESS = "semi_rimless"
    RIMLESS = "rimless"


class BridgeType(str, Enum):
    PAD = "pad"
    KEYHOLE = "keyhole"
    SINGLE = "single"
    DOUBLE = "double"
    UNIVERSAL = "universal"


class FrameFamily(str, Enum):
    ACETATE = "acetate"
    METAL = "metal"
    RIMLESS = "rimless"
    MIXED = "mixed"
    SPORT = "sport"


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box in millimetres (post-normalisation)."""

    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.min[0] + self.max[0]) / 2,
            (self.min[1] + self.max[1]) / 2,
            (self.min[2] + self.max[2]) / 2,
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"min": list(self.min), "max": list(self.max)}


@dataclass(slots=True)
class AnalysisResult:
    """Geometric/topological analysis of the imported scene."""

    bbox_mm: BoundingBox
    frame_width_mm: float
    frame_height_mm: float
    frame_depth_mm: float
    triangle_count: int
    vertex_count: int
    material_count: int
    object_count: int
    mesh_count: int
    has_normals: bool
    has_uvs: bool
    connected_components: int
    symmetry_score: float  # 0..1
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClassifiedPart:
    """A single classified mesh part inside the template."""

    name: str                       # Blender object name (e.g. "LeftTemple")
    kind: PartKind
    vertex_count: int
    triangle_count: int
    bbox: BoundingBox
    center: tuple[float, float, float]
    confidence: float
    source_object: str              # Original object name before rename
    is_separated_mesh: bool
    vertex_group: str | None = None
    material: str | None = None


@dataclass(slots=True)
class ComponentClassification:
    """Full classification of the template."""

    parts: list[ClassifiedPart]
    unknown_objects: list[str]
    frame_part: str | None = None
    left_lens_part: str | None = None
    right_lens_part: str | None = None
    bridge_part: str | None = None
    left_temple_part: str | None = None
    right_temple_part: str | None = None

    def by_kind(self, kind: PartKind) -> list[ClassifiedPart]:
        return [p for p in self.parts if p.kind == kind]


# ---------------------------------------------------------------------------
# Vertex groups
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VertexGroupSpec:
    """Mapping from a logical region name to vertex indices in the frame mesh."""

    name: str
    indices: list[int]

    @property
    def size(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QualityReport:
    """Quality report for one processed template."""

    score: int                      # 0..100
    checks: dict[str, bool]
    issues: list[str]
    metrics: dict[str, float]
    passed: bool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegistryEntry:
    """A single registry entry."""

    id: str
    shape: str
    material: str
    frame_family: str
    bridge_type: str
    rim_type: str
    descriptor: str
    metadata: str
    thumbnail: str
    quality: int
    processed_glb: str
    passed: bool = True
    created_at: str = ""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FactoryPaths:
    """Filesystem layout for the factory."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    descriptors_dir: Path
    metadata_dir: Path
    thumbnails_dir: Path
    registry_path: Path
    log_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "FactoryPaths":
        root = Path(root).resolve()
        return cls(
            root=root,
            raw_dir=root / "raw",
            processed_dir=root / "processed",
            descriptors_dir=root / "descriptors",
            metadata_dir=root / "metadata",
            thumbnails_dir=root / "thumbnails",
            registry_path=root / "registry.json",
            log_path=root / "factory.log",
        )

    def ensure(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.descriptors_dir,
            self.metadata_dir,
            self.thumbnails_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

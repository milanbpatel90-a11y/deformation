from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .feature_detector import TemplateAnalyzer
from .splitter import BRIDGE, FRAME, LEFT_LENS, LEFT_TEMPLE, RIGHT_LENS, RIGHT_TEMPLE
from .utils import ensure_directory, get_logger

try:
    from bpy.types import Object
except ImportError:  # pragma: no cover - Blender runtime only
    Object = object  # type: ignore[assignment]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TemplateDescriptor:
    """Serializable deformation descriptor for a processed template."""

    hinges: dict[str, object]
    bridge: dict[str, object]
    rim_loops: dict[str, object]
    temple_pivots: dict[str, object]
    lens_planes: dict[str, object]
    symmetry_plane: dict[str, object]
    deformation_regions: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        ensure_directory(target.parent)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def generate_descriptor(objects: Iterable[Object]) -> TemplateDescriptor:
    """Builds a first-pass deformation descriptor from analyzed mesh parts."""
    analyzer = TemplateAnalyzer(objects)
    analysis = analyzer.analyze()
    parts = analysis.parts
    frame_width = analysis.dimensions["frame_width"]
    bridge_width = analysis.dimensions["bridge_width"]
    temple_length = analysis.dimensions["temple_length"]

    return TemplateDescriptor(
        hinges={
            "left": {"part": LEFT_TEMPLE, "confidence": parts.get(LEFT_TEMPLE, {}).get("confidence", 0.0)},
            "right": {"part": RIGHT_TEMPLE, "confidence": parts.get(RIGHT_TEMPLE, {}).get("confidence", 0.0)},
        },
        bridge={
            "part": BRIDGE,
            "type": analysis.bridge_type,
            "width_mm": bridge_width,
            "confidence": parts.get(BRIDGE, {}).get("confidence", 0.0),
        },
        rim_loops={
            "frame": {"part": FRAME, "confidence": parts.get(FRAME, {}).get("confidence", 0.0)},
            "left_lens": {"part": LEFT_LENS, "confidence": parts.get(LEFT_LENS, {}).get("confidence", 0.0)},
            "right_lens": {"part": RIGHT_LENS, "confidence": parts.get(RIGHT_LENS, {}).get("confidence", 0.0)},
        },
        temple_pivots={
            "left": {"part": LEFT_TEMPLE, "estimated_length_mm": temple_length},
            "right": {"part": RIGHT_TEMPLE, "estimated_length_mm": temple_length},
        },
        lens_planes={
            "left": {"aspect_width_mm": analysis.dimensions["lens_width"], "aspect_height_mm": analysis.dimensions["lens_height"]},
            "right": {"aspect_width_mm": analysis.dimensions["lens_width"], "aspect_height_mm": analysis.dimensions["lens_height"]},
        },
        symmetry_plane={"axis": "X", "score": analysis.symmetry, "frame_width_mm": frame_width},
        deformation_regions={
            "frame": {"family": analysis.frame_family},
            "bridge": {"type": analysis.bridge_type},
            "temples": {"average_length_mm": temple_length},
        },
    )


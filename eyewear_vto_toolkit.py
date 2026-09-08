"""
Eyewear VTO Toolkit
- Template/asset management with JSON stub generation
- Segmentation dataset pipeline
- Lens contour deformation calibration
- YOLOv8-Seg training prep
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_DIR = PROJECT_ROOT / "templates"

STUB_DEFINITIONS = [
    ("plastic_tortoise", "plastic", "geometric"),
    ("acetate_black", "acetate", "rectangle"),
    ("cat_eye_gold", "cat_eye", "cat_eye"),
    ("clubmaster_brown", "browline", "browline"),
    ("oversized_clear", "geometric", "geometric"),
    ("browline_metal", "browline", "browline"),
    ("rimless_titanium", "rimless", "rimless"),
]


def _get_cv2():
    """Lazy import cv2 to avoid heavy initialization on module load."""
    import cv2
    return cv2


def _get_np():
    """Lazy import numpy for deferred loading."""
    import numpy as np
    return np


@dataclass
class FrameMetadata:
    """Toolkit calibration metadata for a frame template."""

    name: str
    style: str
    glb_path: str
    bridge_width_mm: float
    temple_length_mm: float
    nose_bridge_landmark: int = 6
    left_hinge_landmark: int = 234
    right_hinge_landmark: int = 454
    ffd_grid_dims: tuple[int, int, int] = (4, 4, 4)
    lens_edge_vertices: list[int] = field(default_factory=list)
    rim_pull_strength: float = 0.65
    fallback_glb: str = "geometric_metal.glb"
    created_at: str | None = None
    last_calibrated: str | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameMetadata:
        ffd = data.get("ffd_grid_dims", [4, 4, 4])
        return cls(
            name=data["name"],
            style=data["style"],
            glb_path=str(data["glb_path"]),
            bridge_width_mm=float(data["bridge_width_mm"]),
            temple_length_mm=float(data["temple_length_mm"]),
            nose_bridge_landmark=int(data.get("nose_bridge_landmark", 6)),
            left_hinge_landmark=int(data.get("left_hinge_landmark", 234)),
            right_hinge_landmark=int(data.get("right_hinge_landmark", 454)),
            ffd_grid_dims=tuple(ffd),
            lens_edge_vertices=list(data.get("lens_edge_vertices", [])),
            rim_pull_strength=float(data.get("rim_pull_strength", 0.65)),
            fallback_glb=str(data.get("fallback_glb", "geometric_metal.glb")),
            created_at=data.get("created_at"),
            last_calibrated=data.get("last_calibrated"),
        )


class TemplateManager:
    """Manages frame templates, stubs, and calibration metadata."""

    def __init__(self, templates_dir: str | Path | None = None):
        self.templates_dir = Path(templates_dir or DEFAULT_TEMPLATES_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.templates_dir / "templates.json"
        self.templates: dict[str, FrameMetadata] = self._load_metadata()
        self._ensure_base_template()

    def _ensure_base_template(self) -> None:
        if "geometric_metal" not in self.templates:
            base_glb = self.templates_dir / "geometric_metal.glb"
            self.templates["geometric_metal"] = FrameMetadata(
                name="geometric_metal",
                style="metal",
                glb_path=str(base_glb),
                bridge_width_mm=16.0,
                temple_length_mm=135.0,
                lens_edge_vertices=[8, 9, 10, 11, 12, 13, 14, 15],
                rim_pull_strength=0.65,
            )

    def _load_metadata(self) -> dict[str, FrameMetadata]:
        if not self.metadata_file.exists():
            return {}
        with open(self.metadata_file, encoding="utf-8") as f:
            data = json.load(f)
        return {k: FrameMetadata.from_dict(v) for k, v in data.items()}

    def save_metadata(self) -> None:
        data = {k: asdict(v) for k, v in self.templates.items()}
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved metadata for %d templates", len(self.templates))

    def resolved_glb_path(self, template_name: str) -> Path:
        """Return GLB path, falling back to geometric_metal.glb if missing."""
        meta = self.get_or_fallback(template_name)
        glb = Path(meta.glb_path)
        if glb.exists():
            return glb
        fallback = self.templates_dir / meta.fallback_glb
        if fallback.exists():
            logger.warning("GLB missing for %s, using %s", template_name, fallback.name)
            return fallback
        raise FileNotFoundError(f"No GLB found for {template_name} or fallback")

    def create_stub(
        self,
        name: str,
        style: str,
        shape: str = "geometric",
        material: str = "metal",
        base_template: str | None = "geometric_metal",
    ) -> FrameMetadata:
        if base_template and base_template in self.templates:
            base = self.templates[base_template]
            name_hash = int(hashlib.md5(name.encode()).hexdigest(), 16)
            bridge_var = 0.95 + (name_hash % 100) / 1000
            metadata = FrameMetadata(
                name=name,
                style=style,
                glb_path=str(self.templates_dir / f"{name}.glb"),
                bridge_width_mm=round(base.bridge_width_mm * bridge_var, 2),
                temple_length_mm=base.temple_length_mm,
                nose_bridge_landmark=base.nose_bridge_landmark,
                left_hinge_landmark=base.left_hinge_landmark,
                right_hinge_landmark=base.right_hinge_landmark,
                ffd_grid_dims=base.ffd_grid_dims,
                lens_edge_vertices=base.lens_edge_vertices.copy(),
                rim_pull_strength=base.rim_pull_strength,
                fallback_glb="geometric_metal.glb",
            )
        else:
            metadata = FrameMetadata(
                name=name,
                style=style,
                glb_path=str(self.templates_dir / f"{name}.glb"),
                bridge_width_mm=16.0,
                temple_length_mm=140.0,
                lens_edge_vertices=[],
                rim_pull_strength=0.65,
            )

        self.templates[name] = metadata
        self._write_backend_json(name, shape, material, metadata)
        logger.info("Created stub: %s (style: %s)", name, style)
        return metadata

    def _write_backend_json(
        self,
        name: str,
        shape: str,
        material: str,
        metadata: FrameMetadata,
    ) -> None:
        """Write per-template JSON compatible with backend TemplateLibrary."""
        base_path = self.templates_dir / "geometric_metal.json"
        if base_path.exists():
            with open(base_path, encoding="utf-8") as f:
                base_data = json.load(f)
        else:
            base_data = {
                "dimensions": {
                    "frame_width": 135,
                    "lens_width": 50,
                    "lens_height": 46,
                    "bridge_width": 16,
                    "temple_length": 135,
                    "rim_thickness": 1.0,
                    "temple_curve_angle": 28,
                    "nose_pad_distance": 16,
                    "nose_pad_angle": 15,
                    "nose_pad_height": 3,
                },
                "parts": [
                    "Frame", "LeftLens", "RightLens", "Bridge",
                    "LeftRim", "RightRim", "LeftTemple", "RightTemple",
                    "NosePads", "TempleTips",
                ],
            }

        glb_file = f"{name}.glb" if (self.templates_dir / f"{name}.glb").exists() else "geometric_metal.glb"
        dims = dict(base_data["dimensions"])
        dims["bridge_width"] = metadata.bridge_width_mm
        dims["temple_length"] = metadata.temple_length_mm

        payload = {
            "name": name,
            "shape": shape,
            "material": material,
            "glb_file": glb_file,
            "dimensions": dims,
            "parts": base_data["parts"],
            "rim_pull_strength": metadata.rim_pull_strength,
            "toolkit_style": metadata.style,
        }
        out = self.templates_dir / f"{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def create_batch_stubs(self, base_template: str = "geometric_metal") -> int:
        created = 0
        for name, style, shape in STUB_DEFINITIONS:
            if name not in self.templates:
                material = "acetate" if "acetate" in name else "metal"
                if "rimless" in name:
                    material = "metal"
                self.create_stub(name, style, shape, material, base_template)
                created += 1
        self.save_metadata()
        logger.info("Created %d template stubs", created)
        return created

    def get_or_fallback(self, template_name: str) -> FrameMetadata:
        if template_name in self.templates:
            return self.templates[template_name]
        logger.warning("Template %s not found, using geometric_metal fallback", template_name)
        return self.templates["geometric_metal"]

    def get_rim_pull_strength(self, template_name: str) -> float:
        return self.get_or_fallback(template_name).rim_pull_strength

    def update_calibration(
        self,
        template_name: str,
        lens_edge_vertices: list[int],
        rim_pull_strength: float,
    ) -> None:
        if template_name not in self.templates:
            raise KeyError(f"Unknown template: {template_name}")

        self.templates[template_name].lens_edge_vertices = lens_edge_vertices
        self.templates[template_name].rim_pull_strength = rim_pull_strength
        self.templates[template_name].last_calibrated = datetime.now().isoformat()
        self.save_metadata()

        per_template = self.templates_dir / f"{template_name}.json"
        if per_template.exists():
            with open(per_template, encoding="utf-8") as f:
                data = json.load(f)
            data["rim_pull_strength"] = rim_pull_strength
            with open(per_template, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


@dataclass
class SegmentationDatasetEntry:
    image_id: str
    image_path: str
    mask_path: str
    bounding_box: tuple[float, float, float, float]
    style: str
    face_json_path: str | None = None


class SegmentationDatasetPipeline:
    """Prepare and manage segmentation training data."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.dataset_root / "images"
        self.masks_dir = self.dataset_root / "masks"
        self.metadata_dir = self.dataset_root / "metadata"
        self.splits_dir = self.dataset_root / "splits"
        self.labels_dir = self.dataset_root / "labels"

        for d in [self.images_dir, self.masks_dir, self.metadata_dir, self.splits_dir, self.labels_dir]:
            d.mkdir(exist_ok=True)

        self.entries: list[SegmentationDatasetEntry] = []
        self.load_manifest()

    def load_manifest(self) -> None:
        manifest_path = self.metadata_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [SegmentationDatasetEntry(**e) for e in data]
            logger.info("Loaded %d dataset entries", len(self.entries))

    def save_manifest(self) -> None:
        manifest_path = self.metadata_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)
        logger.info("Saved manifest with %d entries", len(self.entries))

    def add_entry(
        self,
        image_path: str,
        mask_path: str,
        style: str,
        bbox: tuple[float, float, float, float],
        face_json_path: str | None = None,
    ) -> None:
        image_id = hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:8]
        self.entries.append(
            SegmentationDatasetEntry(
                image_id=image_id,
                image_path=str(image_path),
                mask_path=str(mask_path),
                bounding_box=bbox,
                style=style,
                face_json_path=face_json_path,
            )
        )
        logger.info("Added entry: %s (%s)", image_id, style)

    def _mask_to_yolo_seg_line(self, mask_path: Path, img_w: int, img_h: int) -> str | None:
        cv2 = _get_cv2()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            return None
        coords = []
        for pt in approx.reshape(-1, 2):
            coords.extend([pt[0] / img_w, pt[1] / img_h])
        return "0 " + " ".join(f"{c:.6f}" for c in coords)

    def export_yolo_labels(self, split_name: str) -> int:
        """Write YOLO segmentation label files for a split."""
        split_manifest = self.splits_dir / split_name / "manifest.json"
        if not split_manifest.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

        with open(split_manifest, encoding="utf-8") as f:
            split_data = json.load(f)

        label_out = self.labels_dir / split_name
        label_out.mkdir(parents=True, exist_ok=True)
        image_out = self.dataset_root / "images" / split_name
        image_out.mkdir(parents=True, exist_ok=True)

        written = 0
        cv2 = _get_cv2()
        shutil = __import__("shutil")
        for entry_dict in split_data["entries"]:
            entry = SegmentationDatasetEntry(**entry_dict)
            img_path = Path(entry.image_path)
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            stem = img_path.stem
            dest_img = image_out / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            line = self._mask_to_yolo_seg_line(Path(entry.mask_path), w, h)
            if line:
                (label_out / f"{stem}.txt").write_text(line, encoding="utf-8")
                written += 1

        logger.info("Exported %d YOLO labels for split '%s'", written, split_name)
        return written

    def create_yolo_structure(
        self,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> dict[str, list[SegmentationDatasetEntry]]:
        entries = self.entries.copy()
        random.shuffle(entries)
        n = len(entries)
        train_idx = int(n * train_split)
        val_idx = train_idx + int(n * val_split)

        splits = {
            "train": entries[:train_idx],
            "val": entries[train_idx:val_idx],
            "test": entries[val_idx:],
        }

        for split_name, split_entries in splits.items():
            split_dir = self.splits_dir / split_name
            split_dir.mkdir(exist_ok=True)
            with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"split": split_name, "count": len(split_entries), "entries": [asdict(e) for e in split_entries]},
                    f,
                    indent=2,
                )
            self.export_yolo_labels(split_name)
            logger.info("Created %s split: %d samples", split_name, len(split_entries))

        return splits

    def stats(self) -> dict[str, Any]:
        styles: dict[str, int] = {}
        for e in self.entries:
            styles[e.style] = styles.get(e.style, 0) + 1
        return {"total_samples": len(self.entries), "style_distribution": styles}


class DeformationCalibrationWorkflow:
    """Rim contour detection and rim_pull_strength estimation."""

    def __init__(self, template_manager: TemplateManager, yolo_model_path: str | Path | None = None):
        self.template_manager = template_manager
        self.yolo_model_path = Path(yolo_model_path) if yolo_model_path else None
        self.calibration_log: list[dict[str, Any]] = []
        self._yolo = None

    def _load_yolo(self):
        if self._yolo is None and self.yolo_model_path and self.yolo_model_path.exists():
            from ultralytics import YOLO
            self._yolo = YOLO(str(self.yolo_model_path))
        return self._yolo

    def detect_rim_from_image(self, image_path: str | Path) -> dict[str, Any]:
        """
        Detect glasses rim from image using YOLO (if available) + OpenCV contour fitting.
        """
        from backend.measurement.extractor import MeasurementExtractor
        from backend.segmentation.segmenter import GlassesSegmenter

        cv2 = _get_cv2()
        np = _get_np()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        yolo_confidence = 0.0

        model = self._load_yolo()
        if model is not None:
            results = model(image, verbose=False)
            for result in results:
                if result.masks is None:
                    continue
                for m in result.masks.data:
                    resized = cv2.resize(m.cpu().numpy().astype(np.float32), (w, h))
                    mask = np.maximum(mask, (resized > 0.5).astype(np.uint8) * 255)
                if result.boxes is not None and len(result.boxes.conf):
                    yolo_confidence = float(result.boxes.conf.max())

        if mask.sum() == 0:
            segmenter = GlassesSegmenter()
            mask = segmenter.segment(image)["front"]

        measurer = MeasurementExtractor()
        lens_contour = measurer.extract_lens_contour(image, mask)

        all_vertices: list[int] = []
        vertex_count = len(lens_contour.left) + len(lens_contour.right)
        for i in range(vertex_count):
            all_vertices.append(8 + i)

        edge_confidence = max(yolo_confidence, self._edge_confidence_from_mask(mask))
        fit_error = self._contour_fit_error(mask, lens_contour)
        rim_pull = self.estimate_rim_pull_strength(edge_confidence, fit_error)

        return {
            "detected_vertices": all_vertices,
            "edge_detection_confidence": round(edge_confidence, 4),
            "contour_fit_error": round(fit_error, 4),
            "proposed_rim_pull_strength": round(rim_pull, 4),
            "lens_contour": lens_contour.model_dump(),
        }

    def _edge_confidence_from_mask(self, mask: np.ndarray) -> float:
        cv2 = _get_cv2()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        h, w = mask.shape[:2]
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        x, y, contour_w, contour_h = cv2.boundingRect(contour)
        area_ratio = area / max(w * h, 1)
        center = (x + contour_w / 2.0, y + contour_h / 2.0)
        center_distance = (((center[0] - w / 2.0) / max(w, 1)) ** 2 + ((center[1] - h / 2.0) / max(h, 1)) ** 2) ** 0.5
        aspect = contour_w / max(contour_h, 1)

        area_score = 1.0 if 0.025 <= area_ratio <= 0.28 else max(0.0, 1.0 - abs(area_ratio - 0.12) / 0.20)
        center_score = max(0.0, 1.0 - center_distance * 6.0)
        aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.2)

        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = min(1.0, area / hull_area)

        confidence = 0.50 * area_score + 0.25 * center_score + 0.15 * aspect_score + 0.10 * solidity
        return float(max(0.0, min(1.0, confidence)))

    def _contour_fit_error(self, mask: np.ndarray, lens_contour) -> float:
        if not lens_contour.left and not lens_contour.right:
            return 0.5
        h, w = mask.shape[:2]
        total_pts = len(lens_contour.left) + len(lens_contour.right)
        if total_pts == 0:
            return 0.5
        outside = 0
        for poly in [lens_contour.left, lens_contour.right]:
            for u, v in poly:
                x = int(u * w)
                y = int(v * h)
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                if mask[y, x] == 0:
                    outside += 1
        return min(1.0, outside / total_pts)

    def log_calibration_attempt(
        self,
        template_name: str,
        test_image: str,
        detected_vertices: list[int],
        error_metrics: dict[str, Any],
    ) -> None:
        self.calibration_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "template": template_name,
                "test_image": test_image,
                "detected_rim_vertices": detected_vertices,
                "error_metrics": error_metrics,
            }
        )

    def estimate_rim_pull_strength(
        self,
        edge_detection_confidence: float,
        contour_fit_error: float,
    ) -> float:
        strength = min(1.0, edge_detection_confidence * (1.0 - contour_fit_error))
        return max(0.1, round(strength, 4))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Eyewear VTO Toolkit")
    subparsers = parser.add_subparsers(dest="command")

    template_parser = subparsers.add_parser("templates", help="Manage frame templates")
    template_parser.add_argument("--templates-dir", default=str(DEFAULT_TEMPLATES_DIR))
    template_parser.add_argument("--create-stubs", action="store_true")
    template_parser.add_argument("--list", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Manage segmentation dataset")
    dataset_parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))
    dataset_parser.add_argument("--stats", action="store_true")
    dataset_parser.add_argument("--create-splits", action="store_true")

    args = parser.parse_args()

    if args.command == "templates":
        tm = TemplateManager(args.templates_dir)
        if args.create_stubs:
            count = tm.create_batch_stubs()
            print(f"Created {count} template stubs in {args.templates_dir}")
        if args.list:
            print(f"\n{len(tm.templates)} templates registered:")
            for name, meta in tm.templates.items():
                glb = Path(meta.glb_path)
                resolved = tm.resolved_glb_path(name) if glb.exists() or (tm.templates_dir / meta.fallback_glb).exists() else None
                status = "OK" if resolved else "MISSING GLB"
                fallback = "" if glb.exists() else " (fallback)"
                print(f"  [{status}] {name:25} {meta.style:15} rim_pull={meta.rim_pull_strength:.2f}{fallback}")

    elif args.command == "dataset":
        pipeline = SegmentationDatasetPipeline(args.dataset_root)
        if args.stats:
            stats = pipeline.stats()
            print(f"\nDataset Stats:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Style distribution: {stats['style_distribution']}")
        if args.create_splits:
            pipeline.create_yolo_structure()
            print(f"\nYOLO splits + labels written under {args.dataset_root}")


if __name__ == "__main__":
    main()

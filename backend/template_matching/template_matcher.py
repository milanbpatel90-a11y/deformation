"""Compare extracted features against the template library and rank all candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.models import TemplateInfo
from backend.template_library.loader import TemplateLibrary
from backend.template_matching.feature_extractor import EyewearFeatureSet
from backend.template_matching.scoring import ScoredTemplate, WeightedTemplateScorer


@dataclass(frozen=True)
class TemplateMatchResult:
    """Best template plus ranked alternatives."""

    best: ScoredTemplate
    candidates: list[ScoredTemplate]


class TemplateMatcher:
    """Rank templates from `registry.json` with metadata-file fallback."""

    def __init__(self, library: TemplateLibrary, scorer: WeightedTemplateScorer | None = None):
        self.library = library
        self.scorer = scorer or WeightedTemplateScorer()

    def match(
        self,
        features: EyewearFeatureSet,
        override: str | None = None,
    ) -> TemplateMatchResult:
        if override:
            try:
                template = self.library.load(override)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Requested template '{override}' is unavailable or has no GLB metadata."
                ) from exc
            scored = self.scorer.score(features, template)
            return TemplateMatchResult(best=scored, candidates=[scored])

        candidates = []
        for template in self._load_candidates():
            candidates.append(self.scorer.score(features, template))

        if not candidates:
            raise RuntimeError(
                "No usable eyewear template is registered. Each selectable template must "
                "have a GLB, measurements JSON, and valid descriptor JSON."
            )

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return TemplateMatchResult(best=ranked[0], candidates=ranked)

    def _load_candidates(self) -> list[TemplateInfo]:
        names = self._registry_template_names()
        templates: list[TemplateInfo] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            try:
                info = self.library.load(name)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if not Path(info.glb_path).is_file():
                continue

            descriptor_candidates = [
                self.library.templates_dir / "descriptors" / f"{name}.json",
                self.library.templates_dir / "descriptors" / f"{Path(info.glb_path).stem}.json",
                self.library.templates_dir / "processed" / f"{name}.descriptor.json",
                self.library.templates_dir / "processed" / f"{Path(info.glb_path).stem}.descriptor.json",
            ]
            descriptor_path = next((p for p in descriptor_candidates if p.is_file()), None)
            if descriptor_path is None:
                continue
            try:
                descriptor_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            required_descriptor_keys = {
                "hinges", "bridge", "rim_loops", "temple_pivots",
                "lens_planes", "symmetry_plane", "deformation_regions",
            }
            if not isinstance(descriptor_payload, dict) or not required_descriptor_keys.issubset(descriptor_payload):
                continue

            templates.append(info)
        return templates

    def _registry_template_names(self) -> list[str]:
        registry_path = self.library.templates_dir / "registry.json"
        names: list[str] = []
        if registry_path.exists():
            try:
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("template_name") or item.get("template_id")
                        if isinstance(name, str) and name:
                            names.append(name)
        if names:
            return names
        return self.library.list_templates()

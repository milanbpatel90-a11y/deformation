"""Compare extracted features against the template library and rank all candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.models import TemplateInfo
from backend.deformer.descriptor_loader import DescriptorLoader
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import validate_template
from backend.template_library.compatibility import incompatibilities, MeasurementCompatibilityError
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
        measurements=None,
    ) -> TemplateMatchResult:
        if override:
            template = self.library.load(override)
            validate_template(self.library, template)
            issues = incompatibilities(template, measurements) if measurements is not None else []
            if issues:
                raise MeasurementCompatibilityError(f"{template.name}: " + "; ".join(issues))
            scored = self.scorer.score(features, template)
            return TemplateMatchResult(best=scored, candidates=[scored])

        candidates = []
        rejected = []
        for template in self._load_candidates():
            issues = incompatibilities(template, measurements) if measurements is not None else []
            if issues:
                rejected.append(f"{template.name}: " + "; ".join(issues))
                continue
            candidates.append(self.scorer.score(features, template))

        if not candidates:
            if rejected:
                raise MeasurementCompatibilityError("No available template supports these measurements. " + " | ".join(rejected))
            raise ValueError("No prepared templates are available for deformation")

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return TemplateMatchResult(best=ranked[0], candidates=ranked)

    def _load_candidates(self) -> list[TemplateInfo]:
        names = self._registry_template_names()
        templates: list[TemplateInfo] = []
        seen: set[str] = set()
        descriptors = DescriptorLoader(self.library.templates_dir)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            try:
                info = self.library.load(name)
                validate_template(self.library, info)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if not Path(info.glb_path).exists():
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
            return list(dict.fromkeys(names + self.library.list_templates()))
        return self.library.list_templates()

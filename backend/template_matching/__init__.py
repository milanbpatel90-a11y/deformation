"""Template matching pipeline for feature extraction and weighted scoring."""

from backend.template_matching.feature_extractor import EyewearFeatureSet, FeatureExtractor
from backend.template_matching.scoring import ScoredTemplate, WeightedTemplateScorer
from backend.template_matching.template_matcher import TemplateMatchResult, TemplateMatcher

__all__ = [
    "EyewearFeatureSet",
    "FeatureExtractor",
    "ScoredTemplate",
    "TemplateMatchResult",
    "TemplateMatcher",
    "WeightedTemplateScorer",
]

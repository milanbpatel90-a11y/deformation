import pytest

from backend.template_matching.template_matcher import TemplateMatcher


class MissingTemplateLibrary:
    templates_dir = None

    def load(self, name):
        raise FileNotFoundError(name)

    def list_templates(self):
        return []


def test_explicit_missing_template_does_not_silently_select_another():
    matcher = TemplateMatcher(MissingTemplateLibrary())
    with pytest.raises(FileNotFoundError, match="Requested template"):
        matcher.match(None, "not-a-real-template")

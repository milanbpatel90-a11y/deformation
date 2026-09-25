"""Run before deployment: python -m scripts.check_readiness."""
import json
from backend.template_library.loader import TemplateLibrary
from backend.template_library.readiness import template_readiness


def main():
    report = template_readiness(TemplateLibrary())
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

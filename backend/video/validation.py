"""Explicit mesh-validation status: "the pipeline finished" is not "the mesh is trustworthy".

The export path already produces two independent evidence records -- the
*acceptance* report from :func:`backend.exporter.production_validation.validate_production_glb`
and the deformation *quality* report -- but nothing outside that path turns them
into a single, explicit, per-check verdict. A serialized GLB that parses, has
the right units and carries triangles proves only that the *container* is well
formed. It does not prove that the positions are finite, that normals are unit
length, that indices are in range, that the exported millimetres match the
requested measurements, that the deformation map did not fold, or that the
surface does not intersect itself. Those are separate facts and each needs its
own recorded evidence.

This module maps the evidence that genuinely exists in those two dicts onto one
status per check, using :data:`VALIDATION_STATUSES`:

``PASS``
    The source report actually contains a positive result for that check.
``WARN``
    The result exists but is degraded, or the run is explicitly unverified.
``FAIL``
    A mapped source check reported a failure or invalid result.
``NOT_CHECKED``
    No evidence exists in the supplied dicts. This is never silently upgraded
    to ``PASS``: absent evidence is reported as absent. An unchecked
    self-intersection pass therefore can never produce ``overall == "PASS"``.

Every status is derived from a named field read in the sources, and each detail
string cites the file and line that produced it, so a verdict can be argued
with rather than trusted. ``overall`` is deterministic: ``FAIL`` if any check
fails, else ``WARN`` if any check warns *or* was not checked, else ``PASS``.
``production_mode`` is echoed for the caller and annotates details; it does not
by itself change a status.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "VALIDATION_STATUSES",
    "ValidationCheck",
    "CHECK_NAMES",
    "build_validation_report",
]

#: Every status a check -- or the report as a whole -- may carry.
VALIDATION_STATUSES = ("PASS", "WARN", "FAIL", "NOT_CHECKED")

#: The per-check keys of the report, in report order.
CHECK_NAMES = (
    "finite_geometry",
    "expected_dimensions",
    "constraints",
    "jacobian",
    "normals",
    "indices",
    "gltf_structure",
    "self_intersections",
)

#: ``quality_checker.QualityChecker.evaluate`` fails the report below this score
#: (backend/deformer/quality_checker.py:64).
_CONSTRAINT_FAIL_SCORE = 70.0

#: A breakdown score of 100 means the axis passed outright; lower scores are
#: partial credit and are reported as WARN unless the axis collapsed.
_FULL_SCORE = 100.0

#: Error strings produced by ``validate_production_glb`` are attributed to the
#: check they falsify. First match wins; an error matching no pattern is
#: attributed to ``gltf_structure`` (the catch-all "serialized GLB contract"
#: check) so an unrecognised failure is fail-closed rather than ignored.
_ERROR_ROUTES: tuple[tuple[str, str], ...] = (
    (r"must be finite VEC3|finite VEC2 per vertex|non-finite triangles|zero-volume bounds",
     "finite_geometry"),
    (r"triangle indices|degenerate or non-finite triangles", "indices"),
    (r"NORMAL|normals", "normals"),
    (r"dimensional error", "expected_dimensions"),
    (r"jacobian|folded", "jacobian"),
    (r"constraint", "constraints"),
    (r"self[- ]?intersect|Open3D|collision audit|could not resolve triangles",
     "self_intersections"),
)


@dataclass
class ValidationCheck:
    """One named mesh-validation verdict and the evidence behind it."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in VALIDATION_STATUSES:
            raise ValueError(
                f"invalid validation status {self.status!r}; expected one of {VALIDATION_STATUSES}"
            )
        if not self.name:
            raise ValueError("validation check requires a name")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _as_mapping(value: Any) -> Mapping:
    """Return a read-only mapping view, coercing objects that expose ``to_dict``."""
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        produced = to_dict()
        if isinstance(produced, Mapping):
            return produced
    return {}


def _source_checks(acceptance: Mapping) -> Mapping:
    checks = acceptance.get("checks")
    return checks if isinstance(checks, Mapping) else {}


def _breakdown(quality: Mapping) -> Mapping:
    breakdown = quality.get("breakdown")
    return breakdown if isinstance(breakdown, Mapping) else {}


def _score(breakdown: Mapping, key: str) -> float | None:
    """Read a numeric breakdown score, or ``None`` when the evidence is absent."""
    value = breakdown.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _score_status(score: float) -> str:
    """Translate a 0-100 quality axis into a check status."""
    if score >= _FULL_SCORE:
        return "PASS"
    if score <= 0.0:
        return "FAIL"
    return "WARN"


def _texts(*collections: Any) -> list[str]:
    out: list[str] = []
    for collection in collections:
        if isinstance(collection, str):
            out.append(collection)
        elif isinstance(collection, (list, tuple)):
            out.extend(str(item) for item in collection)
    return out


def _route_errors(acceptance: Mapping) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name in CHECK_NAMES}
    for message in _texts(acceptance.get("errors")):
        for pattern, name in _ERROR_ROUTES:
            if re.search(pattern, message):
                buckets[name].append(message)
                break
        else:
            buckets["gltf_structure"].append(message)
    return buckets


def _primitives_read(acceptance: Mapping, checks: Mapping) -> int:
    count = checks.get("primitive_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return 0
    return max(0, count)


def _join(messages: list[str]) -> str:
    return "; ".join(messages)


def _finite_geometry(errors, score, primitives) -> ValidationCheck:
    if errors:
        return ValidationCheck("finite_geometry", "FAIL", _join(errors))
    if score is not None:
        return ValidationCheck(
            "finite_geometry",
            _score_status(score),
            "quality.breakdown['finite_geometry']"
            f"={score:g} (backend/deformer/basis_deformer.py:163)",
        )
    if primitives:
        return ValidationCheck(
            "finite_geometry",
            "PASS",
            f"acceptance traversal validated {primitives} primitive(s): POSITION/NORMAL/TEXCOORD"
            " finite and triangle areas non-degenerate, with no error recorded"
            " (backend/exporter/production_validation.py:84,97,107,122)",
        )
    return ValidationCheck(
        "finite_geometry",
        "NOT_CHECKED",
        "no finiteness evidence: acceptance reported no primitives and no quality breakdown"
        " (backend/exporter/production_validation.py:113)",
    )


def _expected_dimensions(errors, score, source_checks) -> ValidationCheck:
    if errors:
        return ValidationCheck("expected_dimensions", "FAIL", _join(errors))
    maximum_error = source_checks.get("maximum_dimension_error_mm")
    if isinstance(maximum_error, (int, float)) and not isinstance(maximum_error, bool):
        if score is not None and score < _FULL_SCORE:
            return ValidationCheck(
                "expected_dimensions",
                "WARN",
                f"serialized maximum_dimension_error_mm={maximum_error:g} mm passed, but"
                f" quality.breakdown['dimension_tolerance_0_5mm']={score:g}"
                " (backend/deformer/basis_deformer.py:163)",
            )
        return ValidationCheck(
            "expected_dimensions",
            "PASS",
            f"serialized maximum_dimension_error_mm={maximum_error:g} mm with no dimensional"
            " error recorded (backend/exporter/production_validation.py:149)",
        )
    if source_checks.get("dimensional_validation") == "not_applicable_uncalibrated_template":
        return ValidationCheck(
            "expected_dimensions",
            "NOT_CHECKED",
            "acceptance checks['dimensional_validation']=="
            "'not_applicable_uncalibrated_template': the template has no calibrated deformation"
            " certificate, so the exported millimetres were never verified"
            " (backend/exporter/production_validation.py:159)"
            + (f"; unexported in-memory score {score:g} is not export evidence" if score is not None else ""),
        )
    if score is not None:
        return ValidationCheck(
            "expected_dimensions",
            _score_status(score),
            "quality.breakdown['dimension_tolerance_0_5mm']"
            f"={score:g} (in-memory fit, backend/deformer/basis_deformer.py:123-130,163)",
        )
    return ValidationCheck(
        "expected_dimensions",
        "NOT_CHECKED",
        "acceptance.checks carries neither measured_mm/dimension_errors_mm nor a quality"
        " dimension score (backend/exporter/production_validation.py:147-149)",
    )


def _constraints(errors, score, quality, quality_warnings) -> ValidationCheck:
    if errors:
        return ValidationCheck("constraints", "FAIL", _join(errors))
    if any("Critical constraint failures" in warning for warning in quality_warnings):
        return ValidationCheck(
            "constraints",
            "FAIL",
            "quality warning 'Critical constraint failures detected.'"
            " (backend/deformer/quality_checker.py:66)",
        )
    if quality.get("passed") is False:
        # The pipeline deletes any export whose quality report failed
        # (backend/pipeline/deformation_pipeline.py:108), so a supplied
        # passed==False is a hard failure even when the constraint axis itself
        # scored full marks; it is reported against this check because the
        # constraint score is the quality gate's only per-axis veto.
        return ValidationCheck(
            "constraints",
            "FAIL",
            "quality.passed==False: the deformation quality gate rejected this run"
            + (f" (quality.breakdown['constraints']={score:g})" if score is not None else "")
            + " (backend/deformer/quality_checker.py:60-66, enforced by"
            " backend/pipeline/deformation_pipeline.py:108)",
        )
    if score is not None:
        status = "FAIL" if score < _CONSTRAINT_FAIL_SCORE else ("PASS" if score >= _FULL_SCORE else "WARN")
        return ValidationCheck(
            "constraints",
            status,
            f"quality.breakdown['constraints']={score:g} against the"
            f" {_CONSTRAINT_FAIL_SCORE:g} failure threshold"
            " (backend/deformer/quality_checker.py:45,64)",
        )
    return ValidationCheck(
        "constraints",
        "NOT_CHECKED",
        "no constraint-solver evidence: the basis path enforces measurement compatibility"
        " by raising before export, so no constraint record reaches these dicts"
        " (backend/deformer/basis_deformer.py:74)",
    )


def _jacobian(errors, score) -> ValidationCheck:
    if errors:
        return ValidationCheck("jacobian", "FAIL", _join(errors))
    if score is not None:
        return ValidationCheck(
            "jacobian",
            _score_status(score),
            "quality.breakdown['positive_deformation_jacobian']"
            f"={score:g}; the basis deform refuses a field with min_jacobian<=0"
            " (backend/deformer/basis_deformer.py:81,164)",
        )
    return ValidationCheck(
        "jacobian",
        "NOT_CHECKED",
        "the deformation map's Jacobian sign is not reported in this acceptance/quality pair"
        " (only backend/deformer/basis_deformer.py:152,164 records minimum_jacobian)",
    )


def _normals(errors, source_checks, primitives) -> ValidationCheck:
    if errors:
        return ValidationCheck("normals", "FAIL", _join(errors))
    if source_checks.get("normals_required") is True and primitives:
        return ValidationCheck(
            "normals",
            "PASS",
            f"vertex NORMAL data required and accepted for {primitives} primitive(s),"
            " unit length within 1e-3 (backend/exporter/production_validation.py:87-90,114)",
        )
    return ValidationCheck(
        "normals",
        "NOT_CHECKED",
        "acceptance reported no primitives whose NORMAL data was verified"
        " (backend/exporter/production_validation.py:114)",
    )


def _indices(errors, source_checks) -> ValidationCheck:
    if errors:
        return ValidationCheck("indices", "FAIL", _join(errors))
    triangles = source_checks.get("triangle_count")
    if isinstance(triangles, int) and not isinstance(triangles, bool) and triangles > 0:
        return ValidationCheck(
            "indices",
            "PASS",
            f"{triangles} triangle(s) with in-range, complete, non-degenerate indices"
            " (backend/exporter/production_validation.py:100-108,113)",
        )
    return ValidationCheck(
        "indices",
        "NOT_CHECKED",
        "acceptance reported no triangle_count, so index validity was never established"
        " (backend/exporter/production_validation.py:113)",
    )


def _gltf_structure(errors, source_checks) -> ValidationCheck:
    if errors:
        return ValidationCheck("gltf_structure", "FAIL", _join(errors))
    version = source_checks.get("gltf_version")
    units_ok = (
        source_checks.get("coordinate_units") == "m"
        and source_checks.get("measurement_units") == "mm"
    )
    if version == "2.0" and units_ok:
        return ValidationCheck(
            "gltf_structure",
            "PASS",
            "asset version 2.0 with coordinate_units='m' and measurement_units='mm', single"
            " embedded buffer and all required parts present, with no error recorded"
            " (backend/exporter/production_validation.py:40-60,127-130)",
        )
    if version is not None:
        return ValidationCheck(
            "gltf_structure",
            "WARN",
            f"acceptance reported gltf_version={version!r}, coordinate_units="
            f"{source_checks.get('coordinate_units')!r}, measurement_units="
            f"{source_checks.get('measurement_units')!r} without a recorded error"
            " (backend/exporter/production_validation.py:42,59-60)",
        )
    return ValidationCheck(
        "gltf_structure",
        "NOT_CHECKED",
        "acceptance carries no gltf_version/structure evidence"
        " (backend/exporter/production_validation.py:42)",
    )


def _self_intersections(errors, source_checks, production_mode) -> ValidationCheck:
    if errors:
        return ValidationCheck("self_intersections", "FAIL", _join(errors))
    entry = source_checks.get("self_intersections")
    entry = entry if isinstance(entry, Mapping) else {}
    state = str(entry.get("status", "")).lower()
    counts = entry.get("triangles_by_part")
    if state == "checked" and isinstance(counts, Mapping):
        findings = {str(part): int(count) for part, count in counts.items()}
        if any(findings.values()):
            return ValidationCheck(
                "self_intersections",
                "FAIL",
                f"exact per-part intersection counts report triangles: {findings}"
                " (backend/exporter/production_validation.py:201-205)",
            )
        return ValidationCheck(
            "self_intersections",
            "PASS",
            f"exact per-part intersection check ran and found none: {findings or 'no parts'}"
            " (backend/exporter/production_validation.py:203)",
        )
    if state == "not_checked":
        return ValidationCheck(
            "self_intersections",
            "NOT_CHECKED",
            "acceptance.checks['self_intersections'].status=='not_checked': the exact BVH check"
            " was not run, which is why development output is marked REVIEW"
            " (backend/exporter/production_validation.py:206-208,210-211)"
            + ("; production mode requires this check" if production_mode else ""),
        )
    if state == "unavailable":
        return ValidationCheck(
            "self_intersections",
            "NOT_CHECKED",
            "acceptance.checks['self_intersections'].status=='unavailable': Open3D was missing,"
            " so no exact result exists (backend/exporter/production_validation.py:170-171)",
        )
    return ValidationCheck(
        "self_intersections",
        "NOT_CHECKED",
        "acceptance.checks has no exact self-intersection result"
        " (backend/exporter/production_validation.py:207)",
    )


def build_validation_report(
    acceptance: dict,
    quality: dict | None = None,
    production_mode: bool = False,
) -> dict:
    """Derive one explicit status per mesh check from real acceptance/quality evidence.

    ``acceptance`` is the dict returned by
    :func:`backend.exporter.production_validation.validate_production_glb`
    (``status``/``checks``/``errors``/``warnings``); ``quality`` is the
    ``QualityReport.to_dict()`` payload (``score``/``passed``/``warnings``/
    ``breakdown``). Nothing else is consulted: a check with no evidence is
    ``NOT_CHECKED``, never ``PASS``.

    The result carries the eight per-check keys, ``overall``, ``production_mode``
    and ``checks`` (one ``{"name", "status", "detail"}`` dict per check, in
    :data:`CHECK_NAMES` order).
    """
    acceptance = _as_mapping(acceptance)
    quality = _as_mapping(quality)
    source_checks = _source_checks(acceptance)
    breakdown = _breakdown(quality)
    quality_warnings = _texts(quality.get("warnings"))
    errors = _route_errors(acceptance)
    primitives = _primitives_read(acceptance, source_checks)

    checks = [
        _finite_geometry(errors["finite_geometry"], _score(breakdown, "finite_geometry"), primitives),
        _expected_dimensions(
            errors["expected_dimensions"], _score(breakdown, "dimension_tolerance_0_5mm"), source_checks
        ),
        _constraints(
            errors["constraints"],
            _score(breakdown, "constraints"),
            quality,
            quality_warnings,
        ),
        _jacobian(errors["jacobian"], _score(breakdown, "positive_deformation_jacobian")),
        _normals(errors["normals"], source_checks, primitives),
        _indices(errors["indices"], source_checks),
        _gltf_structure(errors["gltf_structure"], source_checks),
        _self_intersections(errors["self_intersections"], source_checks, bool(production_mode)),
    ]

    statuses = [check.status for check in checks]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses or "NOT_CHECKED" in statuses:
        # An unchecked axis is never reported as overall PASS: "the pipeline
        # completed" is not "the geometry is trustworthy".
        overall = "WARN"
    else:
        overall = "PASS"

    report: dict[str, Any] = {check.name: check.status for check in checks}
    report["overall"] = overall
    report["production_mode"] = bool(production_mode)
    report["checks"] = [check.to_dict() for check in checks]
    return report

"""Image-derived size estimates, bounded to the selected template's support."""
import numpy as np
from backend.fusion.view_classifier import ViewClassifier
from backend.template_library.compatibility import measurement_ranges, dependent_ranges, validate_combination
from backend.materials.appearance import detect_appearance


def suggest_measurements(pipeline, images, side=None, top=None):
    classifier = ViewClassifier()
    views = []
    fallback_used = False
    for image in images:
        mask = pipeline.segmenter.segment(image)["front"]
        if not np.count_nonzero(mask):
            mask = pipeline.measurer._auto_mask(image)
            fallback_used = True
        if np.count_nonzero(mask) > 0:
            views.append((image, mask, classifier.classify_view(image, mask)))
    if not views:
        raise ValueError("No glasses detected. Use a clear front-facing photo with a plain background.")
    front, mask, _ = next((v for v in views if v[2] == "front"), views[0])
    if side is None:
        side = next((v[0] for v in views if v[2] == "side" and v[0] is not front), None)
    style, color, mask = detect_appearance(pipeline, front, mask)
    estimates, _ = pipeline.measurer.extract_from_images(front, side, mask, style.shape,
        style.material, style.nose_pads, color=color, top=top)
    features = pipeline.feature_extractor.from_measurements(estimates, style)
    match = pipeline.matcher.match(features)
    template = match.best.template
    ranges = measurement_ranges(template)
    suggested = estimates.model_copy(deep=True)
    adjustments = []
    for name, bounds in ranges.items():
        value = getattr(estimates, name)
        supported = max(bounds["min"], min(bounds["max"], value))
        if supported != value:
            adjustments.append(f'{name.replace("_", " ").title()} adjusted from {value:g} to {supported:g} mm to fit {template.name}.')
            setattr(suggested, name, supported)
    minimum_width = 2*ranges.get("lens_width", {"min": 1})["min"] + suggested.bridge_width + 2*suggested.rim_thickness
    if suggested.frame_width < minimum_width:
        maximum_width = ranges.get("frame_width", {"max": 300})["max"]
        if minimum_width > maximum_width:
            raise ValueError("No feasible suggestion within template limits")
        adjustments.append(f"Frame width increased to {minimum_width:g} mm to reserve the bridge and rim margins.")
        suggested.frame_width = minimum_width
    maximum_lens = (suggested.frame_width-suggested.bridge_width-2*suggested.rim_thickness)/2
    if suggested.lens_width > maximum_lens:
        adjustments.append(f"Lens width adjusted from {suggested.lens_width:g} to {maximum_lens:g} mm to reserve the bridge and rim margins.")
        suggested.lens_width = maximum_lens
    validate_combination(suggested, ranges)
    return {"measurements": suggested.model_dump(mode="json"), "template": template.name,
            "ranges": ranges, "dependent_ranges": dependent_ranges(suggested, ranges), "adjustments": adjustments,
            "note": "Estimated sizes, not measured millimetres: photos have no physical scale. Review and edit before generating.",
            "source": "image_estimate", "fallback_used": fallback_used,
            "assumed_frame_width_mm": pipeline.measurer.DEFAULT_FRAME_WIDTH_MM}

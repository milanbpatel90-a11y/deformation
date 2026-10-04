"""Estimate frame appearance from foreground pixels, without changing dimensions."""
import cv2
import numpy as np


def detect_appearance(pipeline, image, mask=None):
    if mask is None or not np.count_nonzero(mask):
        mask = pipeline.measurer._auto_mask(image)
    style = pipeline.classifier.classify_style(image, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No frame foreground available for appearance detection")
    boxes = [cv2.boundingRect(contour) for contour in contours]
    x0 = min(box[0] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    width = x1 - x0
    # Sample the rim/bridge/temple pixels across all connected components,
    # rather than measuring the filled convex hull or a single eye only.
    band = np.zeros_like(mask)
    cv2.drawContours(band, contours, -1, 255, max(2, round(width * 0.02)))
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    background = np.median(border.astype(float), axis=0)
    contrast = np.linalg.norm(image.astype(float) - background, axis=2)
    selected = (mask > 0) & (band > 0) & (contrast > 25)
    pixels = image[selected]
    if len(pixels) < 10:
        pixels = image[(mask > 0) & (contrast > 25)]
    if len(pixels) < 10:
        raise ValueError("Frame colour is unclear; use a contrasting plain background")
    style.material = pipeline.classifier.classify_material_pixels(pixels)
    style.nose_pads = style.material.value in {"metal", "titanium"} and style.shape.value != "rimless"
    style.rim_type = pipeline.classifier._infer_rim_type(style.shape, style.material)
    style.bridge_type = pipeline.classifier._infer_bridge_type(
        style.shape, style.material, style.nose_pads, style.lens_aspect_ratio
    )
    style.metrics["material_source"] = "frame_rim_pixels"
    # Dominant colour bucket avoids bright reflections biasing the average.
    quantized = pixels.astype(int) // 32
    codes = quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
    dominant = np.bincount(codes, minlength=512).argmax()
    b, g, r = np.median(pixels[codes == dominant], axis=0).astype(int)
    return style, f"#{r:02x}{g:02x}{b:02x}", mask


def apply_image_appearance(pipeline, measurements, image, mask=None):
    style, color, mask = detect_appearance(pipeline, image, mask)
    result = measurements.model_copy(deep=True)
    result.color = color
    result.shape = style.shape
    result.material = style.material
    result.nose_pads = style.nose_pads
    return result, style, mask

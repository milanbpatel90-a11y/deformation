"""A genuine multi-angle orbit fixture: a 3D eyewear model rendered through a
pinhole camera.

The existing fixture foreshortens a single front photograph, which can only ever
produce frontal-hemisphere silhouettes -- it has no side or top geometry to
find, so a test built on it cannot fail when the classifier calls every frame
"front". That is precisely the defect this fixture exists to expose.

Here the glasses are a real 3D model in millimetres (two lens rims, a bridge and
two temple arms swept backwards in -Z) and each frame is an orthographic-ish
perspective projection at a known yaw, with optional pitch for plan views. At
yaw 0 the temples project almost to a point behind the frame, exactly as a front
view should; at yaw 90 the frame collapses and the temples sweep out a long
profile silhouette. The yaw of every frame is therefore known ground truth.

Both the rendered image and the ground-truth silhouette are returned, so mask
level tests never depend on the segmentation model, and image level tests can
check the model against the silhouette it should have produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ── eyewear model, in millimetres ───────────────────────────────────────────
LENS_CENTRE_X = 32.0
LENS_HALF_W = 26.0
LENS_HALF_H = 22.0
RIM_BAND = 4.0
BRIDGE_Y = 7.0
BRIDGE_HALF_W = 10.0
BRIDGE_HALF_H = 3.5
HINGE_X = 56.0
TEMPLE_HALF_H = 2.6
TEMPLE_HALF_W = 2.2
TEMPLE_LENGTH = 145.0
TEMPLE_DROP = 2.0

# ── camera ──────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1280, 720
CAMERA_DISTANCE = 900.0
FOCAL_PX = 2600.0
CENTRE = (CANVAS_W / 2.0, CANVAS_H * 0.52)

BACKDROP = (238, 238, 238)
#: Dark acetate. Deliberately dark enough that the classical threshold fallback
#: can segment the steeply rotated frames the trained model misses -- which is
#: what that fallback exists for, and the only way this fixture can carry
#: genuine side/top evidence into the pipeline end to end.
FRAME_COLOUR = (34, 35, 39)
LENS_COLOUR = (196, 202, 210)    # pale tinted lens
TEMPLE_COLOUR = (30, 31, 35)
NOISE_SIGMA = 3.0

#: Yaws used by the default orbit: a full sweep with real side geometry.
DEFAULT_YAWS = tuple(range(-90, 91, 10))


def project(points: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Rotate about Y (orbit yaw) then X (camera pitch), then project."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x1 = x * math.cos(yaw) + z * math.sin(yaw)
    z1 = -x * math.sin(yaw) + z * math.cos(yaw)

    y2 = y * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y * math.sin(pitch) + z1 * math.cos(pitch)

    depth = np.maximum(CAMERA_DISTANCE - z2, 1.0)
    u = CENTRE[0] + FOCAL_PX * x1 / depth
    v = CENTRE[1] - FOCAL_PX * y2 / depth
    return np.stack([u, v, z2], axis=1)


def _ellipse_points(cx: float, cy: float, half_w: float, half_h: float, count: int = 64) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.stack(
        [cx + half_w * np.cos(angle), cy + half_h * np.sin(angle), np.zeros_like(angle)],
        axis=1,
    )


def _box_points(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> np.ndarray:
    return np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )


@dataclass
class Primitive:
    """One filled shape, drawn back to front."""

    points: np.ndarray          # (N, 3) model-space polygon
    colour: tuple[int, int, int]
    #: Whether this shape is part of the eyewear silhouette.
    in_silhouette: bool = True
    name: str = ""
    order: float = field(default=0.0)


def eyewear_primitives() -> list[Primitive]:
    """The model as back-to-front-agnostic filled polygons."""
    shapes: list[Primitive] = []

    # Temple arms sweep backwards in -Z, so they hide behind the frame head-on
    # and project into a long profile at yaw +/-90.
    for sign in (-1, 1):
        hinge = sign * HINGE_X
        tip = sign * (HINGE_X - 2.0)
        shapes.append(
            Primitive(
                points=_box_points(
                    min(hinge, tip), max(hinge, tip),
                    BRIDGE_Y - TEMPLE_HALF_H, BRIDGE_Y + TEMPLE_HALF_H,
                    -TEMPLE_LENGTH, 0.0,
                ),
                colour=TEMPLE_COLOUR,
                name=f"temple_{'l' if sign < 0 else 'r'}",
            )
        )
        # The drop at the tip, so the arm is not a flat bar.
        shapes.append(
            Primitive(
                points=_box_points(
                    tip - 6.0, tip + 6.0,
                    BRIDGE_Y - TEMPLE_HALF_H - TEMPLE_DROP * 3.0, BRIDGE_Y - TEMPLE_HALF_H,
                    -TEMPLE_LENGTH - 4.0, -TEMPLE_LENGTH + 26.0,
                ),
                colour=TEMPLE_COLOUR,
                name=f"temple_tip_{'l' if sign < 0 else 'r'}",
            )
        )

    # Bridge.
    shapes.append(
        Primitive(
            points=_box_points(
                -BRIDGE_HALF_W, BRIDGE_HALF_W,
                BRIDGE_Y - BRIDGE_HALF_H, BRIDGE_Y + BRIDGE_HALF_H,
                -3.0, 3.0,
            ),
            colour=FRAME_COLOUR,
            name="bridge",
        )
    )

    # Rims: outer ring, then the lens aperture drawn on top of it.
    for sign in (-1, 1):
        cx = sign * LENS_CENTRE_X
        shapes.append(
            Primitive(
                points=_ellipse_points(cx, 0.0, LENS_HALF_W, LENS_HALF_H),
                colour=FRAME_COLOUR,
                name=f"rim_{'l' if sign < 0 else 'r'}",
            )
        )
    for sign in (-1, 1):
        cx = sign * LENS_CENTRE_X
        shapes.append(
            Primitive(
                points=_ellipse_points(
                    cx, 0.0, LENS_HALF_W - RIM_BAND, LENS_HALF_H - RIM_BAND
                ),
                colour=LENS_COLOUR,
                in_silhouette=True,  # a lens is part of "eyewear"
                name=f"lens_{'l' if sign < 0 else 'r'}",
            )
        )
    return shapes


def render_view(
    yaw_deg: float,
    pitch_deg: float = 0.0,
    noise: float = NOISE_SIGMA,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one orbit frame. Returns (bgr image, ground-truth silhouette)."""
    image = np.full((CANVAS_H, CANVAS_W, 3), BACKDROP, dtype=np.uint8)
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    projected = []
    for index, shape in enumerate(eyewear_primitives()):
        pts = project(shape.points, yaw_deg, pitch_deg)
        projected.append((float(np.mean(pts[:, 2])), index, shape, pts))
    # Painter's algorithm: most negative Z is furthest from the camera.
    projected.sort(key=lambda item: item[0])

    for depth, _index, shape, pts in projected:
        polygon = np.round(pts[:, :2]).astype(np.int32)
        # Back-facing thin boxes collapse to a sliver; convexHull keeps them solid.
        hull = cv2.convexHull(polygon)
        if cv2.contourArea(hull) < 1.0:
            continue
        cv2.fillConvexPoly(image, hull, shape.colour, lineType=cv2.LINE_AA)
        if shape.in_silhouette:
            cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        image = np.clip(
            image.astype(np.float32) + rng.normal(0.0, noise, image.shape), 0, 255
        ).astype(np.uint8)
    return image, mask


def orbit_views(
    yaws: tuple[float, ...] | list[float] | None = None,
    pitch_deg: float = 0.0,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """(yaw, image, silhouette) for each sample of an orbit."""
    yaws = list(DEFAULT_YAWS if yaws is None else yaws)
    out = []
    for index, yaw in enumerate(yaws):
        image, mask = render_view(yaw, pitch_deg, seed=index)
        out.append((float(yaw), image, mask))
    return out


def orbit_frames(
    yaws: tuple[float, ...] | list[float] | None = None,
    pitch_deg: float = 0.0,
) -> list[np.ndarray]:
    return [image for _yaw, image, _mask in orbit_views(yaws, pitch_deg)]


def orbit_masks(
    yaws: tuple[float, ...] | list[float] | None = None,
    pitch_deg: float = 0.0,
) -> list[np.ndarray]:
    return [mask for _yaw, _image, mask in orbit_views(yaws, pitch_deg)]


def multi_angle_video(
    path: Path,
    yaws: tuple[float, ...] | list[float] | None = None,
    fps: float = 20.0,
    repeats: int = 3,
) -> Path:
    """Write the orbit as a video, each pose repeated to imitate real sampling."""
    from tests.video_fixture import write_video

    frames: list[np.ndarray] = []
    for image in orbit_frames(yaws):
        frames.extend([image] * max(1, repeats))
    return write_video(path, frames, fps)

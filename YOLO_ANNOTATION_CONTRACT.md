# YOLO Component Annotation Contract

## Purpose

The current repository contains a single-class trained model (eyewear). Its high
segmentation score validates only the whole-object silhouette. It does not provide
trustworthy per-component masks.

Do not invent component labels by splitting the existing binary polygon.
Component labels must come from visible image evidence or deterministic rendering
from a correctly separated 3D template.

## Canonical classes

| ID | Class | Geometry represented | Mandatory |
|---:|---|---|---|
| 0 | left_rim | Visible left front frame/rim material around the lens | yes when visible |
| 1 | right_rim | Visible right front frame/rim material around the lens | yes when visible |
| 2 | bridge | Physical bridge connecting the two fronts | yes when visible |
| 3 | left_temple | Left temple/arm behind the front | only when visible |
| 4 | right_temple | Right temple/arm behind the front | only when visible |
| 5 | left_lens | Left lens surface or clearly visible lens-opening boundary | yes when visible |
| 6 | right_lens | Right lens surface or clearly visible lens-opening boundary | yes when visible |
| 7 | nose_pad | Physical nose pad and visible support geometry | only when visible |
| 8 | temple_tip | Distal temple/end-tip geometry | only when visible |

The frame class is intentionally not a training class. The complete front frame can
be derived as left_rim union right_rim union bridge when all are available.

## Annotation rules

1. Trace only geometry supported by visible pixels. Never guess hidden temple or pad geometry.
2. Keep rim, bridge, lens, temple and tip boundaries separate.
3. For transparent lenses, label the visible lens boundary/surface; ignore reflections as classes.
4. Do not mirror an annotation to an invisible opposite component.
5. Nose pads and tips may be absent when they are too small or fully occluded.
6. The production task assumes one eyewear product per image; multiple visible products are separate instances.

## View guidance

Front / 3-quarter: prioritize both rims, bridge and both lenses; add visible temples/pads.
Side: prioritize the visible temple, tip and any visible rim/lens boundary.
Top: annotate only boundaries that are actually measurable in that view.

## Split rule

Keep all views of the same physical product in the same split. Do not put one view in
train and another view of the same product in val/test.

## Quality gate

- Class IDs must be 0 through 8.
- Each polygon must use class x1 y1 x2 y2 ... xn yn.
- Coordinates must be normalized to [0,1].
- Minimum three points per polygon.
- No zero-area or obviously self-intersecting polygons.
- Review samples from every component and every view type before training.

## Relationship to the 3D contract

2D labels are measurement evidence. They are not the mesh naming contract.

2D left_rim maps to 3D LeftRim.
2D right_rim maps to 3D RightRim.
2D bridge maps to 3D Bridge.
2D left_temple maps to 3D LeftTemple.
2D right_temple maps to 3D RightTemple.
2D left_lens/right_lens map to 3D LeftLens/RightLens.
2D nose_pad maps to 3D NosePads.
2D temple_tip maps to 3D TempleTips.

## Critical current-state limitation

The existing dataset/masks PNG files and YOLO labels are whole-eyewear silhouettes.
There is no ground-truth per-component mask set in the repository today.

A precise component dataset must therefore come from:
1. correctly separated and role-labeled GLB templates rendered into semantic masks; or
2. manually reviewed component annotations of the product images.

The current binary polygons must not be algorithmically split and treated as ground truth.
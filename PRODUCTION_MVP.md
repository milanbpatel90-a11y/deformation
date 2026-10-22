# Production MVP — Glasses 3D Model from Product Images

**Goal:** User uploads 4–5 product photos of a pair of glasses → system returns a usable GLB for virtual try-on.

**Business reality:** Accurate absolute size from photos alone is hard. For production quality you need either known manufacturer dimensions or a scale reference (ruler / credit card in one photo).

---

## What already works in this repo

| Capability | Status |
|------------|--------|
| Multi-view upload API (`POST /api/deform/multi-view`) | ✅ Implemented |
| Viewer multi-file upload (4–6 images) | ✅ Implemented |
| Segmentation (YOLO) | ✅ |
| Shape / material classification | ✅ |
| Measurement extraction + fusion | ✅ (relative scale strong; absolute scale weak without reference) |
| Template matching | ✅ |
| Specialized deformers (bridge, temple, rim, lens, constraints, symmetry, smoother) | ✅ Code present |
| Materials + GLB export | ✅ Basic PBR |
| Cleaned templates (Oct 2026) | ✅ `rectangle_metal`, `cat_eye_gold`, `oversized_clear`, `slim_rectangle_steel`, `round_wire_metal` |

---

## Critical path to a sellable MVP (2–4 weeks)

### Week 0 — Stabilize (do this first)

1. **Verify templates**
   ```bash
   python check_glb_mesh_names.py
   ```
   Every production template must list: `Frame`, `Bridge`, `LeftTemple`, `RightTemple`, `LeftLens`, `RightLens`, `LeftRim`, `RightRim`.

2. **Use only cleaned templates** for production. Prefer:
   - `rectangle_metal`
   - `cat_eye_gold`
   - `round_wire_metal`
   - `oversized_clear`
   - `slim_rectangle_steel`

   Avoid `geometric_metal` until mesh names are fixed in Blender.

3. **Smoke test**
   ```bash
   pip install -r requirements.txt
   python -m uvicorn backend.api.main:app --reload --port 8000
   # Open http://localhost:8000/viewer/
   # Upload 4–5 product images via Multi-View card → Generate
   ```

### Week 1 — Measurement quality (biggest business risk)

Problem: product photos have unknown scale. Wrong mm → wrong fit on face.

**Solutions (pick one or combine):**

A. **Known product dimensions (recommended for B2B)**  
   Ask merchant for: frame width, lens width, bridge, temple length.  
   Use `/api/deform/measurements` or pass override measurements into the pipeline.

B. **Scale reference in one photo**  
   Ask user to place a credit card (85.60 × 53.98 mm) or ruler next to the glasses. Detect card → calibrate pixels → mm.

C. **Relative-only + template prior**  
   Keep relative proportions from photos; lock absolute size to closest template family average (acceptable for fashion try-on, weaker for prescription fit).

Implement scale-reference detection in `backend/measurement/` before promising “real size”.

### Week 2 — Deformation visual quality

- Run 20 real product SKUs through multi-view.
- Visually score: bridge, temples, rim contour, lens fit, symmetry, materials.
- Tune `rim_pull_strength` per template in `templates/*.json`.
- Fix any template that still fails descriptor load (rename meshes in Blender).

### Week 3 — Product dashboard (minimal SaaS shell)

You already have API + viewer. For a business you need:

1. Auth (merchant login)
2. Job queue (upload → process async → notify when GLB ready)
3. SKU storage (images + measurements + GLB URL)
4. Simple catalog UI: upload 4–5 images → “Generate 3D” → preview → download / publish

Keep the current FastAPI backend; put a thin Next.js / React dashboard in front if you want a polished product.

### Week 4 — Try-on on face

Current viewer has a basic webcam try-on hook (MediaPipe). For production:

- Align glasses using face landmarks (pupils, nose bridge, temple points)
- Scale using interpupillary distance or known frame width
- Occlusion (glasses behind hair is hard — start without it)

This is a separate product phase after GLB generation is reliable.

---

## How to use the system for 4–5 glasses images today

**API**
```bash
curl -X POST http://localhost:8000/api/deform/multi-view \
  -F "images=@front.jpg" \
  -F "images=@side.jpg" \
  -F "images=@angle1.jpg" \
  -F "images=@angle2.jpg" \
  -F "color=#c0c0c0" \
  -F "template=rectangle_metal"
```

**Viewer**  
Open `/viewer/` → use **Multi-View (4-6 images)** card → Generate 3D Mesh.

**Preferred templates** for production: the five cleaned ones listed above.

---

## What NOT to do for MVP

- Do not promise millimetre-perfect fit from photos alone without scale reference or manufacturer data.
- Do not expand infrastructure (queues, microservices) until 20 real SKUs look good.
- Do not use un-named / generic Blender meshes as production templates.
- Do not treat documentation status files (`CURRENT_STATUS_*`, `LIVE_STATUS`) as source of truth — code + `check_glb_mesh_names.py` are the source of truth.

---

## Success criteria for “production ready” deformation

1. ≥90% of uploads complete without 500 errors.
2. GLB loads in Three.js and looks like the product (shape family correct).
3. Bridge / temples / rims do not self-intersect.
4. Left–right symmetry error < 1 mm.
5. Absolute size within ±3 mm of manufacturer specs when scale is known.
6. Processing time < 15 s on a mid GPU / modern CPU for 4 images.

---

## Immediate commands

```bash
# 1. Check templates
python check_glb_mesh_names.py

# 2. Install & run
pip install -r requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# 3. Open product UI
# http://localhost:8000/viewer/

# 4. API docs
# http://localhost:8000/docs
```

---

## Recommended product positioning

**Phase 1 (now):** “Upload product photos → get 3D model for web try-on” (fashion / e-commerce).  
**Phase 2:** Manufacturer dimensions + scale card → prescription-grade size.  
**Phase 3:** Live face try-on with good alignment.

Do not skip Phase 1 quality for Phase 3 features.

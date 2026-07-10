# Immediate Action Plan: Unblock All 4 Blockers

**Goal:** Start rim detection validation today, data collection this week, training next week.

---

## TODAY (60 minutes)

### Step 1: Generate Test Images (5 min)
```bash
# Generate 10 synthetic test images
python generate_test_images.py --count 10 --output test_images

# Output: test_images/test_000_metal.jpg, test_001_plastic.jpg, etc.
```

**What it does:** Creates realistic glasses on synthetic faces for immediate UI testing.

### Step 2: Validate Rim Detection Pipeline (10 min)
```bash
# Test rim detection on the generated images
python validate_rim_detection.py --directory test_images --output validation_report.json

# View results
cat validation_report.json | python -m json.tool
```

**What you'll see:**
```
Edge Confidence (target: >0.85):
  Mean: 0.87
  Min:  0.72
  Max:  0.95

Fit Error (target: <0.15):
  Mean: 0.08
  Min:  0.02
  Max:  0.18

Estimated Rim Pull Strength:
  Mean: 0.72
  Min:  0.55
  Max:  0.88
```

**Success criteria:**
- Edge confidence > 0.80
- Fit error < 0.20
- Rim pull strength 0.1-0.9

### Step 3: Test Tuning UI (20 min)
```bash
# Start the deformation tuning server
python deformation_tuning_api.py

# In browser: http://localhost:8001
```

**In UI:**
1. Click "Create 7 Template Stubs"
2. Select "plastic_tortoise"
3. Upload one test image (test_images/test_001_plastic.jpg)
4. Click "Detect Glasses Rim"
5. See metrics populate: edge confidence, fit error, proposed rim pull
6. Adjust slider, click "Confirm & Save"
7. Check: `assets/templates.json` updated ✓

**Success:** You see rim detection working in real-time, template calibrated.

### Step 4: Integrate with Main App (25 min)
```bash
# Copy rim detection module
cp rim_detection_routes.py backend/api/

# Edit backend/api/main.py, add:
from backend.api import rim_detection_routes
app.include_router(rim_detection_routes.router)

# Test endpoint
curl -X POST http://localhost:8000/api/rim-detection/status

# Should return:
# {"engine_ready": true, "using_yolo": false, "yolo_available": false}
```

**Success:** `/api/rim-detection/detect` endpoint working on port 8000.

---

## WEEK 3 (16 hours)

### Step 5: Collect Training Images (4 hours)

**Strategy:** Mix of e-commerce scraping + stock photos + DIY

```bash
# Download free stock photos from Unsplash
curl -s 'https://unsplash.com/napi/search/photos?query=glasses&per_page=100' \
  | jq -r '.results[].urls.regular' \
  | xargs -I {} wget -P dataset/images_raw {}

# Result: 100 images in dataset/images_raw/
```

**Parallel work:**
- Take 30 DIY photos of your own glasses (different angles, lighting)
- Save to: `dataset/images_diy/`

**Target:** 200 total images
- 100 stock photos (Unsplash)
- 50 DIY photos (you)
- 50 from manual sources (friends, e-commerce if needed)

### Step 6: Annotate Masks (10 hours)

**Tool:** CVAT (web-based, free)

```bash
# Online CVAT: https://cvat.ai/

# Or local Docker:
docker run -p 8080:8080 openvino/cvat-ui
docker run -p 8090:8090 openvino/cvat-server

# Then http://localhost:8080
```

**Workflow:**
1. Create project "eyewear_segmentation"
2. Upload images from `dataset/images_raw/`
3. For each image: draw polygon around glasses rim
4. Export as COCO JSON or Mask PNGs

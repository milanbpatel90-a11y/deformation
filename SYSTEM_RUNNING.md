# System Running - Complete Pipeline Active

## ✅ Services Started

### Backend API Server
- **Status:** Running
- **URL:** http://localhost:8000
- **Port:** 8000
- **Framework:** FastAPI with Uvicorn
- **Endpoints:**
  - `POST /api/deform` - Generate 3D model from images
  - `GET /api/output/{filename}` - Download generated GLB
  - `GET /docs` - API documentation
  - `GET /redoc` - Alternative API docs

### Frontend Web Server
- **Status:** Running
- **URL:** http://localhost:8001
- **Port:** 8001
- **Framework:** Python HTTP Server
- **Serves:** `viewer/index.html` and Three.js viewer

---

## 🚀 How to Use

### Option 1: Browser (Recommended)

1. **Open your browser:**
   ```
   http://localhost:8001/index.html
   ```

2. **Or directly:**
   ```
   http://localhost:8001
   ```

3. **What you'll see:**
   - Upload form (left sidebar)
   - 3D viewer (right canvas)
   - Color picker
   - Template selector

4. **To generate a 3D model:**
   - Click "Front view" and select a glasses image
   - (Optional) Select side and top view images
   - (Optional) Choose frame color
   - (Optional) Enter template name
   - Click "Generate 3D Mesh"
   - Wait for processing
   - View the 3D result

### Option 2: API Direct Testing

**Using curl or Postman:**

```bash
curl -X POST http://localhost:8000/api/deform \
  -F "front=@/path/to/front_image.jpg" \
  -F "color=#d9a7a2"
```

**Response:**
```json
{
  "download_url": "/api/output/abc123.glb",
  "measurements": {
    "frame_width": 145,
    "lens_width": 54,
    "lens_height": 50,
    "bridge_width": 18,
    "temple_length": 140,
    "rim_thickness": 1.2
  },
  "template": "geometric_metal",
  "pipeline": [
    {
      "name": "YOLO Segmentation",
      "duration_ms": 234,
      "status": "done",
      "detail": {}
    },
    ...
  ]
}
```

### Option 3: Test Images

Sample test images are available at:
- `test_images/` directory
- Or download from the app's file picker

---

## 📊 Complete System Architecture

```
Browser (port 8001)
   ↓
Frontend: viewer/index.html
   ↓ FormData POST
Backend API (port 8000)
   ↓
├─ Segmentation (YOLO)
├─ Measurement Extraction
├─ Template Matching
├─ Deformation Engine
├─ Material Transfer
└─ GLB Export
   ↓ download_url
GLTFLoader (Three.js)
   ↓
3D Viewer Display
```

---

## 🧪 Test Scenarios

### Scenario 1: Quick Test (2 min)
1. Open http://localhost:8001
2. Click front view
3. Select any glasses image
4. Click "Generate 3D Mesh"
5. Wait for processing
6. See 3D result in viewer

### Scenario 2: Full Pipeline (5 min)
1. Upload front + side + top images
2. Select frame color
3. Enter template name (or leave for auto-detect)
4. Click Generate
5. Watch pipeline stages complete
6. View measurements
7. Download GLB file
8. Inspect in viewer

### Scenario 3: Manual GLB Loading (2 min)
1. Paste URL or select file in "Or load existing GLB" section
2. Click "Load Model"
3. Interact with 3D model

### Scenario 4: URL Query Parameters (1 min)
1. Generate a GLB and copy download URL
2. Open: `http://localhost:8001?glb=/api/output/abc123.glb`
3. Model auto-loads

---

## 🔍 API Documentation

### Endpoint: POST /api/deform

**Purpose:** Upload images and generate 3D GLB

**Request:**
```
Content-Type: multipart/form-data

Parameters:
- front (File, required): Front view image
- side (File, optional): Side view image
- top (File, optional): Top view image
- color (string, optional): Hex color code (default: #d9a7a2)
- template (string, optional): Template name (default: auto-detect)
```

**Response (200):**
```json
{
  "download_url": "/api/output/job_id.glb",
  "measurements": {
    "frame_width": number,
    "lens_width": number,
    "lens_height": number,
    "bridge_width": number,
    "temple_length": number,
    "rim_thickness": number,
    "material": string,
    "color": string
  },
  "template": "template_name",
  "pipeline": [
    {
      "name": "stage_name",
      "duration_ms": number,
      "status": "done|error",
      "detail": {}
    }
  ]
}
```

**Response (400/500):**
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Endpoint: GET /api/output/{filename}

**Purpose:** Download generated GLB file

**Parameters:**
- filename: Job ID with .glb extension (e.g., abc123.glb)

**Response:** Binary GLB file

### Endpoint: GET /docs

**Purpose:** Interactive API documentation (Swagger UI)

**Access:** http://localhost:8000/docs

---

## 🛠️ Troubleshooting

### Issue: API Returns 500 Error

**Check:**
1. Backend still running? Check terminal 2
2. GPU memory? Run `nvidia-smi` if using GPU
3. YOLO model loaded? Check console for errors
4. Valid image format? Try JPEG or PNG

### Issue: Viewer Blank/Black

**Check:**
1. WebGL supported? Open browser console (F12)
2. JavaScript errors? Check console
3. GLB loaded? Check Network tab in DevTools
4. Try manual URL: http://localhost:8001?glb=/api/output/sample.glb

### Issue: 404 on /api/deform

**Check:**
1. Backend running on port 8000?
2. Correct URL? Should be http://localhost:8000
3. API module imported? Check terminal 2 for errors

### Issue: CORS Error

**Check:**
1. Verify CORS enabled in `backend/api/main.py`
2. Should have: `allow_origins=["*"]` or specific origin
3. Restart backend if config changed

### Issue: Slow Generation (>30 seconds)

**Check:**
1. CPU/GPU usage? Monitor system
2. First run? YOLO model takes time to load
3. Large image? Try smaller resolution
4. Network? Check connectivity

---

## 📈 Performance Monitoring

### Expected Timings
- Frontend load: <500ms
- API response: 2-10 seconds (first run slower)
- GLB download: <1 second
- Model render: <100ms
- Total user experience: 3-15 seconds

### If Slower:
1. Check system resources
2. Monitor GPU/CPU usage
3. Check network latency
4. Verify no background processes

---

## 🎯 Next Steps

### To Test Quality of Deformation:

1. Generate a few models with different glasses
2. Inspect GLB output quality
3. Check bridge, temple, lens deformation
4. Compare with original image

### To Test UI/UX:

1. Try all interactive features
2. Test error cases (no image, invalid file)
3. Check responsive layout
4. Verify status messages

### To Move to Week 1 Implementation:

1. Create `backend/deformer/bridge_deformer.py`
2. Follow `WEEK_1_BRIDGE_DEFORMER.md`
3. Integrate and test
4. Observe quality improvement

---

## 📝 Logs & Debugging

### Backend Logs
Terminal 2 shows:
- Request received
- Processing stages
- Errors/warnings
- Response sent

### Frontend Logs
Browser Console (F12) shows:
- Network requests
- Three.js initialization
- GLB loading
- User interactions

### API Documentation
Visit: http://localhost:8000/docs
- See all endpoints
- Try requests directly
- View schema

---

## 🚦 System Status

| Component | Status | URL | Port |
|-----------|--------|-----|------|
| Frontend | ✅ Running | http://localhost:8001 | 8001 |
| Backend API | ✅ Running | http://localhost:8000 | 8000 |
| Segmentation | ✅ Ready | (via API) | - |
| Measurement | ✅ Ready | (via API) | - |
| Templates | ✅ Ready | (via API) | - |
| Deformation | ✅ Ready | (via API) | - |
| Export | ✅ Ready | (via API) | - |
| Viewer | ✅ Ready | (Browser) | - |

---

## 🎬 Ready to Use!

**Everything is running. You can now:**

1. ✅ Upload images
2. ✅ Generate 3D models
3. ✅ View in browser
4. ✅ Download GLB files
5. ✅ Test the complete pipeline

**Open:** http://localhost:8001

Then follow strategic documents to improve quality:
- `README_START_HERE.md`
- `STRATEGIC_ROADMAP.md`
- `WEEK_1_BRIDGE_DEFORMER.md`

---

## 💡 Pro Tips

1. **First run slower?** YOLO model is loading. Subsequent runs faster.
2. **Want to test API?** Visit http://localhost:8000/docs
3. **Debug 3D?** Right-click → Inspect → Console in browser
4. **Save results?** Use Download GLB button
5. **Share results?** Copy URL with ?glb= parameter

---

## 🔗 Important URLs

| Purpose | URL |
|---------|-----|
| Main App | http://localhost:8001 |
| API Docs | http://localhost:8000/docs |
| API Root | http://localhost:8000 |
| Sample Model | http://localhost:8001?glb=/api/output/sample.glb |

---

## ✨ System Complete

**All services running. All components active. Ready for production testing.**

Next: Test the pipeline, then execute Week 1 improvements. ✅

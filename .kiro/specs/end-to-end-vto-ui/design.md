# Technical Design Document: End-to-End VTO UI

## Overview

The End-to-End VTO UI is a browser-based single-page application (SPA) that provides a seamless interface for uploading eyewear product images, submitting them to a FastAPI backend for processing, and displaying the resulting 3D GLB model in an interactive Three.js viewer.

**Scope:** Frontend HTML/CSS/JavaScript application that integrates with existing FastAPI backend at `http://localhost:8000`

**Key Interfaces:**
- Upload form with three image fields (front required, side/top optional)
- Color picker and optional template selector
- Real-time pipeline progress display
- Three.js viewer with OrbitControls for 3D interaction
- Manual GLB loader for testing and sharing

**Existing Resources Leveraged:**
- `viewer/index.html` - existing Three.js template structure and styling
- FastAPI backend with endpoints: `POST /api/deform`, `GET /api/output/{filename}`, `GET /api/templates`
- Three.js v0.162.0 from CDN (jsDelivr)

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser Client                         │
│  ┌─────────────────┐                   ┌──────────────────┐ │
│  │  Upload Panel   │                   │   3D Viewer      │ │
│  │  - Front image  │                   │  (Three.js)      │ │
│  │  - Side image   │◄──────────────────├──────────────────┤ │
│  │  - Top image    │                   │  - OrbitControls │ │
│  │  - Color pick   │                   │  - Auto camera   │ │
│  │  - Template     │                   │  - Multi-light   │ │
│  └─────────────────┘                   │  - Ambient       │ │
│          │                             │  - Key + Fill    │ │
│          │ FormData POST               └──────────────────┘ │
│          │ /api/deform                                      │
│          ▼                                                   │
├─────────────────────────────────────────────────────────────┤
│                    FastAPI Backend                          │
│              (http://localhost:8000)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  POST /api/deform                                    │  │
│  │  - Segmentation (YOLOv8)                             │  │
│  │  - Measurement Extraction                            │  │
│  │  - Template Selection                                │  │
│  │  - Mesh Deformation                                  │  │
│  │  - Material Application                              │  │
│  │  - GLB Export                                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ▲
         │ JSON Response + download_url
         │ GLB GET /api/output/{job_id}.glb
         │
         └─ GLTFLoader fetches GLB
```

### Data Flow

```
User Input (Images + Color + Template)
    ↓
Validation (Front image required)
    ↓
FormData Construction
    ↓
POST /api/deform
    ↓
Status: "Running pipeline…"
    ↓
Response: { download_url, measurements, template, pipeline[] }
    ↓
Parse & Validate Response
    ↓
Render Pipeline Stages Table
    ↓
Display Measurements Summary
    ↓
Load GLB from download_url
    ↓
Scene Setup: Add Model → Fit Camera → Render
```

---

## Components and Interfaces

### 1. HTML Structure

```html
<!-- Two-pane layout -->
<div id="sidebar">
  <!-- Upload Section -->
  <div class="section">
    <h3>Upload Images</h3>
    <div class="img-card">
      <input type="file" id="input-front" />
      <img class="preview" id="prev-front" />
      <span class="badge required">required</span>
    </div>
    <div class="img-card">
      <input type="file" id="input-side" />
      <img class="preview" id="prev-side" />
      <span class="badge">optional</span>
    </div>
    <div class="img-card">
      <input type="file" id="input-top" />
      <img class="preview" id="prev-top" />
      <span class="badge">optional</span>
    </div>
  </div>

  <!-- Options Section -->
  <div class="section">
    <h3>Options</h3>
    <label>Frame colour</label>
    <input type="color" id="color-pick" value="#d9a7a2" />
    <label>Template</label>
    <input type="text" id="template-input" />
  </div>

  <!-- Generate & Status -->
  <button id="generate-btn" disabled>Generate 3D Mesh</button>
  <div id="status-box">Upload a front image to begin.</div>
  <button id="download-btn" style="display:none">⬇ Download GLB</button>

  <!-- Pipeline Progress Table -->
  <div id="pipeline-stages" style="display:none">
    <table>
      <thead>
        <tr><th>#</th><th>Stage</th><th>ms</th><th>Detail</th></tr>
      </thead>
      <tbody id="stages-body"></tbody>
    </table>
  </div>

  <!-- Manual Load -->
  <div class="section">
    <h3>Or load existing GLB</h3>
    <input type="text" id="glb-url" />
    <input type="file" id="glb-file" accept=".glb" />
    <button id="load-btn">Load Model</button>
  </div>
</div>

<div id="canvas-container">
  <canvas></canvas>
  <div id="overlay-status">Drag to rotate · Scroll to zoom</div>
</div>
```

### 2. CSS Styling Strategy

**Layout:**
- Two-pane: sidebar (300px fixed) + canvas (flex: 1)
- Dark theme (#1a1a2e background, #12122a sidebar)
- Professional colors: #e94560 (generate button), #50fa7b (success green), #ff5555 (error red)

**Responsive Principles:**
- Sidebar scrollable if content exceeds viewport height
- Canvas expands to fill remaining space
- Button states: disabled (gray), enabled (red), hover (bright red)

**Typography:**
- Body: system-ui sans-serif, 13px base
- Headings: 15px bold, section titles 11px uppercase
- Monospace for values: measurements, milliseconds

**Color Palette:**
- Dark backgrounds: #1a1a2e, #12122a, #1e1e3a
- Accents: #8be9fd (cyan), #e94560 (red), #50fa7b (green), #ff5555 (error)
- Text: #eee (primary), #aaa (secondary), #555 (tertiary)

### 3. Component: Upload Panel

**Purpose:** Allow users to select front, side, and top images

**Behavior:**
- Click card → open file picker
- File selected → generate thumbnail preview using FileReader API
- Show "required" badge on front, "optional" on side/top
- Preview image displays in-place (70px minimum height)

**State Management:**
- Store File objects in `uploadedFiles = { front, side, top }`
- Track preview URL state (update DOM directly)

**Example:**
```javascript
const fileInput = document.getElementById('input-front');
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    const preview = document.getElementById('prev-front');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    uploadedFiles.front = file;
    updateGenerateBtn();
  }
});
```

### 4. Component: Options Panel

**Purpose:** Color picker and template selector

**Behavior:**
- Color picker: HTML5 `<input type="color">` with default #d9a7a2
- Template input: text field with placeholder "auto-detect"
- Values persisted in DOM until user changes them

**State Management:**
- Color value read from picker when form submitted
- Template value read from input (empty = backend auto-detect)

### 5. Component: Status Box

**Purpose:** Display feedback during processing and results

**Display States:**
1. **Initial:** "Upload a front image to begin."
2. **Processing:** "Running pipeline…"
3. **Success:** Multi-line measurements summary
4. **Error:** Red text with error message

**Content:**
```
✓ Done · Template: geometric_metal
Frame 145mm · Lens 54×50mm
Bridge 18mm · Temple 140mm · Rim 1.2mm
```

### 6. Component: Pipeline Stages Table

**Purpose:** Show real-time progress of backend processing

**Columns:**
- # (stage number)
- Stage (name with emoji icon)
- ms (duration in milliseconds)
- Detail (key-value pairs or "—")

**Stage Icons:**
- 🔍 YOLO Segmentation
- 🔷 Shape Classification
- 📐 Measurement Extraction
- 📋 Template Selection
- 🔧 Template Deformation
- 🎨 Texture Mapping
- 📦 GLB Export

**Color Coding:**
- Green (#50fa7b) for status="done"
- Red (#ff5555) for status="error"

### 7. Component: Three.js Viewer

**Scene Setup:**
```javascript
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(45, width/height, 0.1, 1000);
camera.position.set(0, 0, 200);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
```

**Lighting Setup:**
```javascript
// Ambient: base illumination
const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);

// Key light: main directional light
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(50, 80, 100);
scene.add(key);

// Fill light: secondary accent light
const fill = new THREE.DirectionalLight(0x8be9fd, 0.4);
fill.position.set(-60, -20, 50);
scene.add(fill);
```

**Camera Auto-Fit Algorithm:**
```javascript
function fitCamera(model) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  
  camera.position.set(
    center.x,
    center.y,
    center.z + maxDim * 2.5  // 2.5× multiplier for optimal view
  );
  controls.target.copy(center);
  controls.update();
}
```

---

## Data Models

### API Request: POST /api/deform

```javascript
const formData = new FormData();
formData.append('front', frontFile);           // Required: File
formData.append('side', sideFile);             // Optional: File
formData.append('top', topFile);               // Optional: File
formData.append('color', '#d9a7a2');           // String: hex color
formData.append('template', 'geometric_metal'); // Optional: string

const response = await fetch('/api/deform', {
  method: 'POST',
  body: formData
});
```

### API Response: /api/deform

```json
{
  "job_id": "a1b2c3d4e5f6",
  "download_url": "/api/output/a1b2c3d4e5f6.glb",
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
      "status": "done",
      "duration_ms": 234,
      "detail": {
        "model": "best.pt",
        "input_size": "640x640",
        "confidence": 0.95
      }
    },
    {
      "name": "Shape Classification",
      "status": "done",
      "duration_ms": 45,
      "detail": {
        "predicted_shape": "geometric",
        "confidence": 0.87
      }
    },
    {
      "name": "Measurement Extraction",
      "status": "done",
      "duration_ms": 128,
      "detail": {
        "frames_detected": 1,
        "measurement_method": "contour_fitting"
      }
    },
    {
      "name": "Template Selection",
      "status": "done",
      "duration_ms": 67,
      "detail": {
        "selected_template": "geometric_metal",
        "match_score": 0.92
      }
    },
    {
      "name": "Template Deformation",
      "status": "done",
      "duration_ms": 312,
      "detail": {
        "vertices_modified": 2048,
        "scaling_factors": "[1.2, 0.95, 1.1]"
      }
    },
    {
      "name": "Texture Mapping",
      "status": "done",
      "duration_ms": 156,
      "detail": {
        "material": "pbr_metal",
        "color": "#d9a7a2"
      }
    },
    {
      "name": "GLB Export",
      "status": "done",
      "duration_ms": 89,
      "detail": {
        "file_size_mb": 2.3,
        "compression": "gltf2"
      }
    }
  ]
}
```

### Internal State: ApplicationState

```javascript
const appState = {
  // Upload state
  uploadedFiles: {
    front: null,   // File object
    side: null,
    top: null
  },
  
  // Options state
  selectedColor: '#d9a7a2',
  selectedTemplate: '',
  
  // Processing state
  isProcessing: false,
  currentJobId: null,
  lastResponse: null,
  
  // Viewer state
  currentModel: null,  // THREE.Group
  cameraFitZoom: 2.5
};
```

---

## Correctness Properties

Since this feature involves UI interaction, form handling, API integration, and 3D rendering (not pure data transformation), property-based testing is **not applicable** to this feature. Instead, the design relies on:

1. **Example-based unit tests** for discrete functions (URL construction, form validation, response parsing)
2. **Integration tests** for API communication and response handling
3. **Visual regression tests** for renderer output
4. **Manual QA** for UI interactions (form submission, button state transitions, canvas interactions)

The UI logic is primarily imperative (handling user clicks, API responses) rather than declarative properties that could be tested universally.

---

## Error Handling

### Error Categories

**1. Network Errors**
- Backend server unreachable
- Network timeout
- CORS policy rejection

**Response:** Display "Network error: {error.message}" in Status_Box with red styling

**2. API Errors (4xx, 5xx)**
- Invalid image format
- Missing required fields
- Backend processing failure

**Response:** Display error detail from response in Status_Box with red styling

**3. File Loading Errors**
- Invalid GLB file
- Corrupted file
- GLTFLoader parsing failure

**Response:** Display "Error loading model: {error.message}" in overlay status

**4. Validation Errors**
- Missing front image when Generate clicked
- Missing GLB URL/file when Load clicked

**Response:** Silently skip (no error shown); button remains disabled or action ignored

### Error Handling Patterns

```javascript
// Pattern 1: Fetch with error handling
try {
  const response = await fetch('/api/deform', { method: 'POST', body: form });
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.detail || `Server error: ${response.status}`);
  }
  
  // Process success
  handleSuccess(data);
} catch (error) {
  statusBox.classList.add('error');
  statusBox.textContent = `Network error: ${error.message}`;
  generateBtn.disabled = false;
  generateBtn.textContent = 'Generate 3D Mesh';
}

// Pattern 2: Response validation
function validateResponse(data) {
  const required = ['download_url', 'measurements', 'template'];
  for (const field of required) {
    if (!(field in data)) {
      console.warn(`Missing field: ${field}`);
      return false;
    }
  }
  return true;
}

// Pattern 3: GLB loading error handling
loader.load(url, onSuccess, undefined, (err) => {
  overlayStatus.textContent = `Error loading model: ${err.message}`;
});
```

### Graceful Degradation

- **Missing measurements fields:** Display available fields, skip missing ones
- **Missing pipeline array:** Skip pipeline stages table, show summary only
- **Invalid thumbnail preview:** Display placeholder instead
- **WebGL unavailable:** Display clear message (browser doesn't support 3D)



---

## Testing Strategy

### Unit Tests (Example-Based)

#### Category 1: Form Validation

1. **Test: Generate button state with no front image**
   - Input: No front image selected
   - Expected: Button disabled, cursor: not-allowed
   - Example: Initial page load

2. **Test: Generate button state with front image**
   - Input: Front image selected
   - Expected: Button enabled (red background, clickable)
   - Example: User selects front.jpg

3. **Test: Image preview display**
   - Input: User selects image file
   - Expected: Preview thumbnail displayed, placeholder hidden
   - Example: Display 70px preview of front.jpg

#### Category 2: API Response Parsing

1. **Test: Parse valid API response**
   - Input: Valid JSON response with all required fields
   - Expected: Extract measurements, template, pipeline, download_url correctly
   - Example: Response with frame_width=145, template="geometric_metal"

2. **Test: Parse API response with missing optional fields**
   - Input: Response missing pipeline array
   - Expected: Display measurements summary, skip pipeline table
   - Example: Graceful degradation when pipeline is null

3. **Test: Handle HTTP error response**
   - Input: HTTP 500 with error detail
   - Expected: Display error message in red text
   - Example: "Error: Failed to process image"

#### Category 3: Status Display

1. **Test: Display processing status**
   - Input: User clicks Generate button
   - Expected: Status shows "Running pipeline…", button text changes to "Generating…"
   - Example: Before response received

2. **Test: Display success status**
   - Input: Successful API response received
   - Expected: Status displays template name, frame dimensions, bridge width, etc.
   - Example: "✓ Done · Template: geometric_metal"

3. **Test: Display error status**
   - Input: API error response
   - Expected: Status displays error in red text
   - Example: "Error: Front image not readable"

#### Category 4: Three.js Scene

1. **Test: Scene initialization**
   - Input: Page loads
   - Expected: Scene created with dark background, camera at (0, 0, 200), three lights present
   - Example: Verify scene.children.length >= 3 (ambient + key + fill)

2. **Test: Camera auto-fit**
   - Input: GLB model loaded (bounding box size 100×100×100)
   - Expected: Camera positioned to frame model, target at model center
   - Example: Verify camera.position.z > model.center.z

3. **Test: OrbitControls damping**
   - Input: User drags canvas
   - Expected: Camera moves smoothly, continues to decelerate when drag ends
   - Example: Verify enableDamping = true and dampingFactor = 0.05

#### Category 5: Manual GLB Loading

1. **Test: Load from URL input**
   - Input: User enters "/api/output/job.glb" and clicks Load
   - Expected: GLTFLoader fetches and renders model
   - Example: Manual loading of previous job

2. **Test: Load from file input**
   - Input: User selects local .glb file
   - Expected: GLTFLoader renders local file
   - Example: Testing external 3D files

3. **Test: URL query parameter loading**
   - Input: Page loads with "?glb=/api/output/test.glb"
   - Expected: Model auto-loads on page load
   - Example: Sharing link to pre-loaded model

### Integration Tests

1. **End-to-End: Upload → Process → Render**
   - Flow: Select front image → Click Generate → Wait for response → Model displays
   - Verification: Status shows measurements, model appears in viewport, camera auto-fit works
   - Duration: ~5-10 seconds (depends on backend processing)

2. **Pipeline Progress Display**
   - Flow: Trigger generation → Monitor response with pipeline array
   - Verification: Each stage renders in table with name, duration, status
   - Duration: Entire pipeline display

3. **Error Recovery**
   - Flow: Generate with invalid image → See error → Select new image → Generate again
   - Verification: Error clears, new generation succeeds
   - Duration: Two generation attempts

4. **Download Functionality**
   - Flow: Successful generation → Click Download button
   - Verification: File downloads with job_id as filename
   - Duration: File download

### Browser Compatibility Testing

| Browser | Version | Test | Status |
|---------|---------|------|--------|
| Chrome | 90+ | Upload, generate, render | ✓ |
| Firefox | 88+ | Upload, generate, render | ✓ |
| Safari | 14+ | Upload, generate, render | ✓ |
| Edge | 90+ | Upload, generate, render | ✓ |

### Performance Benchmarks

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Page load (scene init) | <500ms | Time to first render |
| Thumbnail preview | <100ms | File → preview display |
| GLB load and render | <2s | fetch + GLTFLoader + scene add |
| Render framerate | 60fps | OrbitControls + animation loop |
| Memory usage | <200MB | Browser DevTools heap |

### Visual Regression Testing

1. **Upload panel layout:** Sidebar 300px width, cards stacked vertically
2. **Color picker:** Input displays current color swatch
3. **Status box:** Multi-line text wraps correctly, colors apply (green/red)
4. **Pipeline table:** Columns aligned, icons display, colors correct
5. **3D viewer:** Dark background, lighting realistic, model in center, controls responsive

### Manual QA Checklist

- [ ] Upload front image, see preview and button enabled
- [ ] Upload front + side, both preview correctly
- [ ] Change color picker, see color update in form
- [ ] Enter template name, verify it's sent in request
- [ ] Click Generate with valid image, see "Generating…" status
- [ ] Wait for response, see pipeline stages display
- [ ] Click Download, file downloads correctly
- [ ] Rotate model by dragging mouse
- [ ] Zoom model with mouse wheel
- [ ] Load model from URL manually
- [ ] Use ?glb= query parameter to load model
- [ ] Test error case: network down, see error message
- [ ] Test error case: invalid image, see backend error displayed
- [ ] Resize browser window, canvas resizes correctly
- [ ] Test on mobile (tablet): layout responsive or graceful
- [ ] Test on different browsers: Chrome, Firefox, Safari

---

## Implementation Details

### 1. Form Submission Logic

```javascript
async function generateMesh() {
  const frontFile = document.getElementById('input-front').files[0];
  if (!frontFile) return; // Validation
  
  const form = new FormData();
  form.append('front', frontFile);
  
  const sideFile = document.getElementById('input-side').files[0];
  if (sideFile) form.append('side', sideFile);
  
  const topFile = document.getElementById('input-top').files[0];
  if (topFile) form.append('top', topFile);
  
  form.append('color', document.getElementById('color-pick').value);
  
  const template = document.getElementById('template-input').value.trim();
  if (template) form.append('template', template);
  
  // Update UI state
  const btn = document.getElementById('generate-btn');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  
  const statusBox = document.getElementById('status-box');
  statusBox.className = '';
  statusBox.textContent = 'Running pipeline…';
  
  try {
    const resp = await fetch('/api/deform', { method: 'POST', body: form });
    const data = await resp.json();
    
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    
    handleSuccess(data);
  } catch (error) {
    statusBox.classList.add('error');
    statusBox.textContent = `Error: ${error.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate 3D Mesh';
  }
}
```

### 2. Response Handling

```javascript
function handleSuccess(response) {
  // Parse measurements
  const m = response.measurements || {};
  const statusBox = document.getElementById('status-box');
  
  statusBox.textContent =
    `✓ Done · Template: ${response.template}\n` +
    `Frame ${m.frame_width}mm · Lens ${m.lens_width}×${m.lens_height}mm\n` +
    `Bridge ${m.bridge_width}mm · Temple ${m.temple_length}mm · Rim ${m.rim_thickness}mm`;
  
  // Render pipeline stages
  if (response.pipeline && Array.isArray(response.pipeline)) {
    renderPipelineStages(response.pipeline);
  }
  
  // Show download button
  const downloadBtn = document.getElementById('download-btn');
  downloadBtn.style.display = 'block';
  downloadBtn.onclick = () => {
    window.location.href = response.download_url;
  };
  
  // Load GLB model
  loadGLB(response.download_url);
}
```

### 3. Pipeline Stages Rendering

```javascript
function renderPipelineStages(stages) {
  const tbody = document.getElementById('stages-body');
  tbody.innerHTML = '';
  
  const ICONS = {
    'YOLO Segmentation': '🔍',
    'Shape Classification': '🔷',
    'Measurement Extraction': '📐',
    'Template Selection': '📋',
    'Template Deformation': '🔧',
    'Texture Mapping': '🎨',
    'GLB Export': '📦'
  };
  
  stages.forEach((stage, i) => {
    const icon = ICONS[stage.name] || '▸';
    const statusClass = `status-${stage.status}`; // 'status-done' or 'status-error'
    
    const detailStr = stage.detail
      ? Object.entries(stage.detail)
          .filter(([, v]) => !Array.isArray(v) || v.length <= 4)
          .map(([k, v]) => `${k}: ${v}`)
          .join(' · ')
      : '—';
    
    const row = `
      <tr>
        <td>${i + 1}</td>
        <td class="name">${icon} ${stage.name}</td>
        <td class="ms">${stage.duration_ms}</td>
        <td class="detail ${statusClass}">${detailStr}</td>
      </tr>
    `;
    
    tbody.innerHTML += row;
  });
  
  document.getElementById('pipeline-stages').style.display = 'block';
}
```

### 4. GLB Loading

```javascript
function loadGLB(url) {
  const loader = new GLTFLoader();
  const overlayStatus = document.getElementById('overlay-status');
  
  overlayStatus.textContent = 'Loading model…';
  
  loader.load(
    url,
    (gltf) => {
      // Success
      if (currentModel) scene.remove(currentModel);
      
      currentModel = gltf.scene;
      scene.add(currentModel);
      
      fitCamera(currentModel);
      
      overlayStatus.textContent = 'Drag to rotate · Scroll to zoom';
    },
    undefined,
    (err) => {
      // Error
      overlayStatus.textContent = `Error loading model: ${err.message}`;
    }
  );
}

function fitCamera(model) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  
  camera.position.set(
    center.x,
    center.y,
    center.z + maxDim * 2.5
  );
  
  controls.target.copy(center);
  controls.update();
}
```

### 5. Image Preview Handler

```javascript
function setupImagePreview(inputId, previewId, placeholderId) {
  document.getElementById(inputId).addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const url = URL.createObjectURL(file);
    const preview = document.getElementById(previewId);
    const placeholder = document.getElementById(placeholderId);
    
    preview.src = url;
    preview.style.display = 'block';
    placeholder.style.display = 'none';
    
    updateGenerateBtn();
  });
}

setupImagePreview('input-front', 'prev-front', 'ph-front');
setupImagePreview('input-side', 'prev-side', 'ph-side');
setupImagePreview('input-top', 'prev-top', 'ph-top');
```

### 6. Generate Button State Management

```javascript
function updateGenerateBtn() {
  const hasFront = document.getElementById('input-front').files.length > 0;
  document.getElementById('generate-btn').disabled = !hasFront;
}

// Listen to image input changes
document.getElementById('input-front').addEventListener('change', updateGenerateBtn);
document.getElementById('input-side').addEventListener('change', updateGenerateBtn);
document.getElementById('input-top').addEventListener('change', updateGenerateBtn);
```

### 7. Manual GLB Loading

```javascript
document.getElementById('load-btn').addEventListener('click', () => {
  const fileInput = document.getElementById('glb-file');
  const urlInput = document.getElementById('glb-url');
  
  if (fileInput.files.length > 0) {
    // Load from file
    const url = URL.createObjectURL(fileInput.files[0]);
    loadGLB(url);
  } else if (urlInput.value.trim()) {
    // Load from URL
    loadGLB(urlInput.value.trim());
  }
});

// Auto-load from query parameter
const params = new URLSearchParams(window.location.search);
if (params.has('glb')) {
  loadGLB(params.get('glb'));
}
```

### 8. Render Animation Loop

```javascript
function animate() {
  requestAnimationFrame(animate);
  
  controls.update();
  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener('resize', () => {
  const width = container.clientWidth;
  const height = container.clientHeight;
  
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  
  renderer.setSize(width, height);
});
```

---

## Performance Optimization Strategies

### 1. Scene Initialization (<500ms target)

- Lazy-load Three.js and GLTFLoader from CDN (cached after first load)
- Create scene, camera, renderer synchronously (fast)
- Add lights immediately (minimal overhead)
- Initialize OrbitControls on canvas element (deferred until needed)

### 2. File Upload Preview (<100ms)

- Use `URL.createObjectURL(file)` for instant preview (no processing)
- Avoid FileReader or canvas manipulation
- Preview image is rendered by browser (hardware accelerated)

### 3. GLB Loading (<2s typical)

- GLTFLoader handles streaming (progressive loading if network allows)
- Gzip compression on server reduces file size
- Browser caching for repeated loads

### 4. Render Performance (60fps)

- Keep animation loop simple: `controls.update() + renderer.render()`
- Avoid DOM manipulation in render loop
- Use requestAnimationFrame for smooth motion

### 5. Memory Management

- Clean up old models with `scene.remove(oldModel)`
- Dispose geometries and textures if models are replaced frequently
- Use `URL.revokeObjectURL()` after preview displayed (optional but good practice)

---

## Browser Compatibility

### Supported Browsers

- **Chrome/Edge 90+**: Full support (WebGL 2, ES2020)
- **Firefox 88+**: Full support
- **Safari 14+**: Full support (limited WebGL 2 features, but sufficient for GLB viewer)

### Feature Detection

```javascript
// Check WebGL support
function isWebGLSupported() {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('webgl2'))
    );
  } catch (e) {
    return false;
  }
}

if (!isWebGLSupported()) {
  document.body.innerHTML = '<div style="color:red;padding:20px">' +
    'Your browser does not support WebGL. ' +
    'Please use Chrome, Firefox, Safari, or Edge.' +
    '</div>';
}
```

### Fallback Strategies

- **No WebGL:** Display message, disable 3D viewer, allow download-only workflow
- **Fetch API not available:** Show error (unlikely in modern browsers)
- **File API not available:** Disable upload (show message)
- **FormData not available:** Construct multipart/form-data manually (edge case)

---

## Deployment and Configuration

### Environment Variables

- `API_BASE_URL`: Backend base URL (default: `http://localhost:8000`)
- `API_ENDPOINT_DEFORM`: POST endpoint (default: `/api/deform`)
- `API_ENDPOINT_TEMPLATES`: GET endpoint (default: `/api/templates`)
- `API_ENDPOINT_OUTPUT`: GET endpoint for files (default: `/api/output`)

### HTML File Location

**Path:** `viewer/index.html`

This is the main single-page application file. It can be served by:
1. FastAPI static mount: `app.mount("/viewer", StaticFiles(directory="viewer", html=True))`
2. Separate web server: nginx, Apache, etc.
3. Development: `python -m http.server 8000` in `viewer/` directory

### CDN Dependencies

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"
  }
}
</script>
```

**Why v0.162.0?** Stable, widely tested, includes OrbitControls and GLTFLoader.

### CORS Configuration

**Backend (FastAPI):**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This is already configured in `backend/api/main.py`.

---

## Security Considerations

### Input Validation

1. **File Type:** Browser file picker limits to image/* (enforcement)
2. **File Size:** Backend should enforce max size (e.g., 10MB per image)
3. **Image Content:** Backend validates images are valid and contain eyewear
4. **Color Hex:** Validate color is valid hex format before sending
5. **Template Name:** Backend rejects unknown template names

### API Security

1. **CORS:** Production should limit `allow_origins` to known domains
2. **Rate Limiting:** Backend should implement rate limiting (e.g., 10 requests/minute per IP)
3. **Authentication:** Frontend should support API keys or OAuth if needed
4. **HTTPS:** Production deployment must use HTTPS (TLS/SSL)

### Client-Side Security

1. **XSS Prevention:** No eval(), innerHTML used only with trusted data
2. **Output Encoding:** Error messages displayed as text, not HTML
3. **URL Validation:** Validate GLB URLs are relative or from same origin (prevent SSRF)

---

## Future Enhancements

1. **Progressive Upload:** Show segmentation mask preview before final generation
2. **Template Browser:** Visual gallery of available templates with thumbnails
3. **Multi-model Comparison:** Load and compare multiple generated models side-by-side
4. **Augmented Reality:** Mobile AR viewer using Three.js AR capabilities
5. **Export Formats:** Support USDZ, FBX, OBJ in addition to GLB
6. **Collaborative Features:** Share job IDs with team, comment on measurements
7. **Accessibility:** Add keyboard controls (arrow keys for orbit, +/- for zoom)
8. **Analytics:** Track which templates are popular, common measurements
9. **Local Processing:** WebWorker-based processing for measurement extraction (offline)
10. **Custom Materials:** Advanced PBR controls (roughness, metallic, IOR) in UI


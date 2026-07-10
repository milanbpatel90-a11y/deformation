# Implementation Plan: End-to-End VTO UI

## Overview

The implementation follows a bottom-up approach: establish the HTML/CSS foundation and Three.js scene setup, then wire form inputs to API calls, parse responses, render results, and finally test the complete flow. The existing `viewer/index.html` template is already partially complete and will be extended with form state management and backend integration logic.

## Tasks

### Phase 1: HTML & CSS Foundation

- [x] 1. Verify and complete sidebar/canvas layout structure
  - Review `viewer/index.html` two-pane layout (sidebar 300px fixed, canvas flex: 1)
  - Ensure all section titles, dividers, and spacing match design spec
  - Verify responsive CSS classes are in place
  - _Requirements: 12.1, 12.4, 12.5_

- [ ]* 1.1 Write unit tests for layout CSS
  - Test sidebar width is 300px minimum
  - Test canvas container has flex: 1
  - Test button states (enabled/disabled) apply correct colors
  - _Requirements: 12.1, 15.1, 15.2_

### Phase 2: Form Components - Image Upload

- [x] 2. Wire image upload handlers for preview display
  - Implement file input change listeners for front, side, top inputs
  - Use FileReader API or URL.createObjectURL for thumbnail previews
  - Display 70px preview images in upload cards
  - Hide placeholder when image selected, show when cleared
  - _Requirements: 1.2, 1.3, 1.7_

- [x] 2.1 Implement preview cleanup and memory management
  - Revoke object URLs when image is replaced or cleared
  - Clean up old blob URLs to prevent memory leaks
  - _Requirements: 1.7_

- [ ]* 2.2 Write unit tests for image preview handlers
  - Test file selection triggers preview display
  - Test multiple selections replace previous preview
  - Test preview image dimensions are correct
  - _Requirements: 1.3, 1.7_

### Phase 3: Form Components - Generate Button State

- [x] 3. Implement generate button enable/disable logic
  - Create `updateGenerateBtn()` function that checks for front image
  - Attach event listeners to all three image inputs
  - Set button disabled state based on front file presence
  - Update button text on processing (see Phase 5)
  - _Requirements: 1.5, 1.6, 15.1, 15.2, 15.3_

- [ ]* 3.1 Write unit tests for button state management
  - Test button disabled when no front image
  - Test button enabled when front image selected
  - Test button disabled when front image cleared
  - _Requirements: 1.5, 1.6_

### Phase 4: Form Components - Color & Template Options

- [x] 4. Implement color picker and template input handling
  - Ensure color picker input has default value `#d9a7a2`
  - Wire template text input field with placeholder "auto-detect"
  - Store selected color and template values for form submission
  - Persist values until user changes them
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 14.1, 14.4_

- [ ]* 4.1 Write unit tests for form value persistence
  - Test color picker default value is set
  - Test template input accepts text input
  - Test values persist after page interactions
  - _Requirements: 14.1, 14.4_

### Phase 5: Three.js Scene Setup

- [ ] 5. Initialize Three.js scene, camera, renderer
  - Create scene with dark background `#1a1a2e`
  - Initialize perspective camera with 45° FOV
  - Create WebGL renderer with antialiasing enabled
  - Set ACES Filmic tone mapping
  - Append renderer canvas to container
  - _Requirements: 7.1, 7.2, 7.6, 7.7_

- [ ] 5.1 Implement Three.js lighting setup (3-light system)
  - Add ambient light (intensity 0.6+)
  - Add key directional light at position (50, 80, 100) with intensity 1.2+
  - Add fill directional light at position (-60, -20, 50) with intensity 0.4 or less
  - Verify lights are in world space and do not rotate with camera
  - _Requirements: 7.3, 7.4, 7.5, 9.6, 9.7_

- [ ] 5.2 Implement window resize handler for responsive rendering
  - Listen for `resize` event
  - Update camera aspect ratio to match container
  - Update renderer size to match container dimensions
  - Call `camera.updateProjectionMatrix()`
  - _Requirements: 7.8, 12.3, 17.4_

- [ ]* 5.3 Write unit tests for scene initialization
  - Verify scene background color is correct
  - Verify camera FOV is 45 degrees
  - Verify renderer antialiasing is enabled
  - Verify three lights are in scene
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

### Phase 6: Render Loop and Animation

- [ ] 6. Implement animation loop with OrbitControls update
  - Create `animate()` function that calls `requestAnimationFrame`
  - Update OrbitControls each frame: `controls.update()`
  - Render scene: `renderer.render(scene, camera)`
  - Start animation loop on page load
  - _Requirements: 9.2, 9.4, 9.5, 17.4_

- [ ] 6.1 Configure OrbitControls damping for smooth camera motion
  - Enable damping: `controls.enableDamping = true`
  - Set damping factor to 0.05: `controls.dampingFactor = 0.05`
  - Ensure camera decelerates smoothly when drag ends
  - _Requirements: 9.4, 9.5_

- [ ]* 6.1 Write tests for render loop performance
  - Verify requestAnimationFrame is called continuously
  - Verify scene renders without errors
  - _Requirements: 17.4_

### Phase 7: Form Submission & API Integration

- [x] 7. Implement form submission to POST /api/deform
  - Create `generateMesh()` function called on Generate button click
  - Validate front image is selected (skip submission if not)
  - Build FormData with front, side (optional), top (optional)
  - Append color picker value as hex string
  - Append template input value if non-empty
  - Send POST request to `/api/deform`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 19.2_

- [ ] 7.1 Implement request state management during API call
  - Disable generate button during request: `btn.disabled = true`
  - Change button text to "Generating…"
  - Show status "Running pipeline…" in status box
  - Hide download button and pipeline table
  - _Requirements: 4.1, 4.2_

- [ ] 7.2 Implement error handling for network failures
  - Wrap fetch in try/catch block
  - Catch network errors and display "Network error: {message}" in red
  - Re-enable button and restore text on error
  - Apply error styling to status box
  - _Requirements: 3.10, 13.3, 19.3_

- [ ]* 7.3 Write unit tests for form submission logic
  - Test FormData construction with all fields
  - Test missing optional files are handled correctly
  - Test error message is displayed on network error
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.10_

### Phase 8: API Response Parsing & Validation

- [ ] 8. Parse and validate API response structure
  - Check HTTP response status is 200 before processing JSON
  - Validate response contains required fields: `download_url`, `measurements`, `template`
  - Handle missing optional fields gracefully (display what's available)
  - Extract measurements object with expected fields: frame_width, lens_width, lens_height, bridge_width, temple_length, rim_thickness
  - _Requirements: 3.5, 3.6, 3.7, 16.1, 16.2, 16.3_

- [ ] 8.1 Implement response validation with graceful degradation
  - Log warnings for missing fields instead of throwing errors
  - Continue processing even if optional pipeline array is missing
  - Display partial measurements if some fields are absent
  - _Requirements: 16.4, 16.5_

- [ ]* 8.1 Write unit tests for response parsing
  - Test valid response is parsed correctly
  - Test missing optional fields are handled
  - Test HTTP error responses are detected
  - _Requirements: 3.5, 3.6, 3.7, 16.1, 16.3_

### Phase 9: Status Display with Measurements

- [ ] 9. Implement status display with measurement summary
  - Create status text in format: "✓ Done · Template: {template_name}"
  - Append line 2: "Frame {frame_width}mm · Lens {lens_width}×{lens_height}mm"
  - Append line 3: "Bridge {bridge_width}mm · Temple {temple_length}mm · Rim {rim_thickness}mm"
  - Use multiline display with proper line breaks
  - Set status box background and text color to match design
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 9.1 Write unit tests for status display formatting
  - Test multiline text format matches spec
  - Test all measurements are displayed
  - Test template name is shown
  - _Requirements: 5.1, 5.2, 5.3_

### Phase 10: Pipeline Stages Table Rendering

- [ ] 10. Implement pipeline stages table rendering
  - Check response contains `pipeline` array before rendering
  - Create table with columns: #, Stage, ms, Detail
  - Map stage emoji icons to stage names (🔍 YOLO, 🔷 Shape, 📐 Measurement, 📋 Template, 🔧 Deformation, 🎨 Texture, 📦 Export)
  - Display stage number, name with icon, duration_ms, and detail
  - Extract detail key-value pairs from detail object, skip large arrays
  - _Requirements: 4.3, 4.4, 4.5_

- [ ] 10.1 Implement color coding for pipeline stage status
  - Apply green color (#50fa7b) to rows with status="done"
  - Apply red color (#ff5555) to rows with status="error"
  - Show table only after pipeline array is present
  - _Requirements: 4.6, 4.7_

- [ ]* 10.2 Write unit tests for pipeline table rendering
  - Test table renders with correct column headers
  - Test emoji icons map correctly to stage names
  - Test color coding applies based on status
  - _Requirements: 4.4, 4.5, 4.6, 4.7_

### Phase 11: Download Button & File Download

- [ ] 11. Implement download button visibility and file download
  - Hide download button initially (display: none)
  - Show download button after successful generation
  - Wire button click to trigger file download via `window.location.href`
  - Download file using `download_url` from API response
  - Name downloaded file as `{job_id}.glb`
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 11.1 Hide download button on generation start and errors
  - Hide button when Generate button is clicked
  - Hide button if API returns error
  - Hide button on network errors
  - _Requirements: 6.2, 6.5_

- [ ]* 11.1 Write unit tests for download button behavior
  - Test button is hidden initially
  - Test button shows after successful response
  - Test button triggers download with correct URL
  - _Requirements: 6.1, 6.2, 6.3_

### Phase 12: GLB Loading with GLTFLoader

- [ ] 12. Implement GLB model loading using GLTFLoader
  - Import GLTFLoader from Three.js addons
  - Create `loadGLB(url)` function that accepts GLB file URL
  - Use GLTFLoader to load GLB from URL
  - Add progress/loading feedback: "Loading model…" in overlay status
  - Add success callback to add model to scene
  - Add error callback to display error message
  - _Requirements: 8.2, 8.3, 8.4, 8.7, 8.8_

- [ ] 12.1 Implement model cleanup and replacement
  - Remove previous model from scene before loading new model
  - Store reference to currentModel in global variable
  - Use `scene.remove(oldModel)` to remove old model
  - _Requirements: 8.3_

- [ ]* 12.2 Write unit tests for GLB loading
  - Test GLTFLoader is called with correct URL
  - Test model is added to scene on success
  - Test error message displayed on load failure
  - _Requirements: 8.2, 8.3, 8.7, 8.8_

### Phase 13: Camera Auto-Fit Algorithm

- [ ] 13. Implement camera auto-fit for loaded models
  - Create `fitCamera(model)` function
  - Calculate bounding box of model using `THREE.Box3().setFromObject(model)`
  - Get bounding box size and center
  - Calculate max dimension from size.x, size.y, size.z
  - Position camera at center + (maxDim × 2.5) on Z axis
  - Set OrbitControls target to bounding box center
  - Call `controls.update()` to apply changes
  - _Requirements: 8.5, 8.6, 9.8_

- [ ]* 13.1 Write unit tests for camera auto-fit
  - Test bounding box is calculated correctly
  - Test camera position is set appropriately
  - Test controls target is set to model center
  - _Requirements: 8.5, 8.6_

### Phase 14: Manual GLB Loading (URL & File Input)

- [ ] 14. Implement manual GLB loading from URL input
  - Create event listener on Load button click
  - Read URL from glb-url text input
  - If URL is non-empty, call `loadGLB(url)`
  - Otherwise, do nothing (no error)
  - _Requirements: 10.3_

- [ ] 14.1 Implement manual GLB loading from file input
  - Create event listener on Load button click
  - Read file from glb-file file input
  - If file is selected, create object URL using `URL.createObjectURL(file)`
  - Call `loadGLB(objectUrl)`
  - Prioritize file input over URL input if both have values
  - _Requirements: 10.4, 10.5_

- [ ] 14.2 Implement file input clearing and validation
  - Validate that URL or file input has value before loading
  - Skip load action if both inputs are empty (no error)
  - _Requirements: 10.6_

- [ ]* 14.3 Write unit tests for manual GLB loading
  - Test URL input loading works correctly
  - Test file input loading works correctly
  - Test file input prioritized over URL
  - _Requirements: 10.3, 10.4, 10.5_

### Phase 15: URL Query Parameter Support

- [ ] 15. Implement auto-load GLB from ?glb= query parameter
  - Parse URL search parameters on page load
  - Check for `?glb={url}` parameter
  - If present, automatically call `loadGLB(url)` on page load
  - Show "Loading model…" status while loading
  - If load fails, display error but keep page functional
  - _Requirements: 11.1, 11.2_

- [ ]* 15.1 Write unit tests for query parameter loading
  - Test page loads and auto-loads GLB from query param
  - Test default status displays if no query param
  - _Requirements: 11.1, 11.3_

### Phase 16: Generate Button Re-enable & Form Persistence

- [ ] 16. Implement button re-enable after processing completes
  - After API response (success or error), re-enable generate button
  - Restore button text to "Generate 3D Mesh"
  - Call `updateGenerateBtn()` to verify button state is correct
  - _Requirements: 4.8, 14.2, 14.3_

- [ ] 16.1 Verify form state persists between submissions
  - Confirm image files persist in input elements after submission
  - Confirm color picker value persists
  - Confirm template input value persists
  - Do not clear form on successful generation
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ]* 16.1 Write unit tests for button state transitions
  - Test button is disabled during generation
  - Test button is re-enabled after success
  - Test button is re-enabled after error
  - _Requirements: 4.8, 15.3_

### Phase 17: Error Handling & User Feedback

- [ ] 17. Implement HTTP error response handling
  - Check `response.ok` after fetch
  - If HTTP error (4xx, 5xx), extract error detail from response JSON
  - Display "Error: {detail}" in status box with red styling
  - Apply error class to status box
  - Do not attempt to load model on error
  - _Requirements: 3.8, 3.9, 13.2_

- [ ] 17.1 Implement graceful degradation for partial responses
  - If measurements object is missing fields, display available fields
  - If pipeline array is missing, skip pipeline table rendering
  - Do not crash or show generic error for partial data
  - _Requirements: 16.4, 16.5_

- [ ] 17.2 Implement overlay status updates for model loading
  - Show "Loading model…" while GLTFLoader is fetching
  - Show "Drag to rotate · Scroll to zoom" after successful load
  - Show error message if load fails
  - _Requirements: 8.7, 8.8, 9.3_

- [ ]* 17.3 Write unit tests for error handling
  - Test HTTP errors display correct message
  - Test network errors display user-friendly message
  - Test error styling is applied
  - _Requirements: 3.8, 3.9, 13.2, 13.3_

### Phase 18: Initial Page Status

- [ ] 18. Set initial status box message on page load
  - Display "Upload a front image to begin." on page load
  - Ensure status box is visible and ready for user interaction
  - _Requirements: 11.3_

### Phase 19: Three.js & Browser Compatibility

- [ ] 19. Verify Three.js version and compatibility
  - Confirm Three.js v0.162.0 is loaded from CDN
  - Verify GLTFLoader is correctly imported
  - Verify OrbitControls is correctly imported
  - Test in Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [ ] 19.1 Test WebGL support detection and graceful degradation
  - Verify WebGL renderer initializes without errors
  - If WebGL is unavailable, display clear message (basic check)
  - _Requirements: 18.6_

- [ ]* 19.2 Write browser compatibility tests
  - Test page loads in Chrome, Firefox, Safari, Edge
  - Test form submission works in all browsers
  - Test model rendering works in all browsers
  - _Requirements: 18.1, 18.2, 18.3_

### Phase 20: CORS & API Integration

- [ ] 20. Verify CORS configuration and API connectivity
  - Confirm backend at http://localhost:8000 accepts requests
  - Verify /api/deform endpoint accepts POST with FormData
  - Verify /api/output/{filename} returns GLB files
  - Test fetch requests work without CORS errors in browser
  - _Requirements: 19.1, 19.2, 19.3, 19.4_

### Phase 21: Performance Validation

- [ ] 21. Validate scene initialization performance (<500ms)
  - Measure time from page load to first render
  - Verify scene setup (scene, camera, renderer, lights) is fast
  - Verify no performance issues on initial page load
  - _Requirements: 17.2_

- [ ] 21.1 Validate thumbnail preview performance (<100ms)
  - Measure time from file selection to preview display
  - Verify FileReader or URL.createObjectURL is efficient
  - No jank or visual delay when selecting images
  - _Requirements: 17.3_

- [ ] 21.2 Validate GLB loading performance (<2s typical)
  - Measure time from GLTFLoader fetch to scene display
  - Typical GLB files load in <2 seconds over localhost
  - No page freezing during GLB loading
  - _Requirements: 17.4_

- [ ]* 21.3 Write performance benchmarks
  - Measure page load to first render
  - Measure file selection to preview display
  - Measure API request to model display
  - _Requirements: 17.2, 17.3, 17.4, 17.5_

### Phase 22: Integration Testing & QA

- [ ] 22. End-to-end integration test: upload → generate → render
  - Select front image file
  - Verify button becomes enabled
  - Click Generate button
  - Wait for "Running pipeline…" status
  - Wait for response and measurement display
  - Verify status shows "✓ Done · Template: {template}"
  - Verify pipeline stages table renders with all stages
  - Verify model loads and displays in viewer
  - Verify camera is auto-fitted to model
  - _Requirements: 1.1, 3.1, 4.1, 5.1, 8.1, 9.1, 10.1_

- [ ] 22.1 Integration test: interactive model controls
  - Drag mouse on canvas to rotate model
  - Verify model rotates smoothly with damping
  - Scroll mouse wheel to zoom
  - Verify zoom works and camera distance changes
  - Verify overlay text shows "Drag to rotate · Scroll to zoom"
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 22.2 Integration test: download functionality
  - Generate model successfully
  - Verify Download button appears
  - Click Download button
  - Verify file downloads with correct name (job_id.glb)
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 22.3 Integration test: manual GLB loading
  - Enter URL in glb-url input (/api/output/test.glb)
  - Click Load Model button
  - Verify model loads if URL is valid
  - Select .glb file from file input
  - Click Load Model button
  - Verify model loads from file
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 22.4 Integration test: error handling
  - Generate with no front image selected
  - Verify button remains disabled, no submission
  - Generate with invalid image (if backend rejects)
  - Verify error message displays in red
  - Upload new valid image
  - Verify form is ready for next submission
  - _Requirements: 1.8, 3.8, 13.2, 13.3_

- [ ] 22.5 Integration test: query parameter auto-load
  - Navigate to page with ?glb=/api/output/test.glb
  - Verify model auto-loads on page load
  - Verify "Loading model…" displays during load
  - Navigate to page without query parameter
  - Verify default status "Upload a front image to begin."
  - _Requirements: 11.1, 11.2, 11.3_

- [ ] 22.6 Responsive layout verification
  - Test two-pane layout on desktop (1920×1080)
  - Verify sidebar is 300px wide
  - Verify canvas fills remaining space
  - Resize browser window
  - Verify canvas resizes correctly
  - Verify controls remain accessible
  - _Requirements: 12.1, 12.2, 12.3_

- [ ] 22.7 Manual QA checklist validation
  - [ ] Upload front image, see preview and button enabled
  - [ ] Upload front + side, both preview correctly
  - [ ] Change color picker, verify color is sent in request
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
  - [ ] Test on different browsers: Chrome, Firefox, Safari
  - _Requirements: 1.1, 2.1, 3.1, 6.1, 9.1, 10.1, 14.1, 18.1, 18.2, 18.3_

### Phase 23: Final Checkpoint

- [ ] 23. Ensure all tests pass and feature is complete
  - Run all unit tests and verify they pass
  - Run integration tests and verify end-to-end flow works
  - Verify all 20 requirements are implemented
  - Verify no console errors or warnings
  - Verify performance benchmarks are met
  - Ask the user if questions arise.
  - _Requirements: 1.1 through 20.4_

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP completion. However, including them ensures code quality and proper error handling.
- Most tests should be example-based unit tests (specific inputs → expected outputs) rather than property-based tests, since this is a UI system with imperative logic, not declarative transformations.
- The existing `viewer/index.html` template is already partially complete with Three.js scene setup and some event handlers; most implementation work is extending and refining existing code.
- The backend at http://localhost:8000 must be running for form submission and GLB loading to work. Error handling gracefully handles connection failures.
- All form state is persisted in DOM elements (no localStorage) during the browser session.
- ImageFile previews use `URL.createObjectURL()` for instant display and must be revoked when replaced to avoid memory leaks.

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": [
        "1.1",
        "5.1",
        "5.2",
        "6.1"
      ],
      "description": "Foundation: HTML/CSS layout, Three.js scene setup, animation loop"
    },
    {
      "id": 1,
      "tasks": [
        "2.1",
        "3.1",
        "4.1",
        "18.1",
        "19.1"
      ],
      "description": "Form components: image preview, button state, color/template inputs, initial status"
    },
    {
      "id": 2,
      "tasks": [
        "5.1",
        "6.1"
      ],
      "description": "Lighting and damping configuration for scene"
    },
    {
      "id": 3,
      "tasks": [
        "2.2",
        "3.2",
        "4.2",
        "5.3",
        "6.2"
      ],
      "description": "Unit tests for form and scene components"
    },
    {
      "id": 4,
      "tasks": [
        "7.1",
        "7.2"
      ],
      "description": "Form submission: build FormData and send to API"
    },
    {
      "id": 5,
      "tasks": [
        "8.1",
        "8.2"
      ],
      "description": "API response parsing and validation"
    },
    {
      "id": 6,
      "tasks": [
        "9.1",
        "10.1",
        "10.2",
        "11.1"
      ],
      "description": "Status and results display: measurements, pipeline table, download button"
    },
    {
      "id": 7,
      "tasks": [
        "7.3",
        "8.3",
        "9.2",
        "10.3",
        "11.2"
      ],
      "description": "Unit tests for API integration and display logic"
    },
    {
      "id": 8,
      "tasks": [
        "12.1",
        "12.2",
        "13.1"
      ],
      "description": "GLB loading and model setup"
    },
    {
      "id": 9,
      "tasks": [
        "14.1",
        "14.2",
        "14.3"
      ],
      "description": "Manual GLB loading from URL or file"
    },
    {
      "id": 10,
      "tasks": [
        "15.1",
        "16.1",
        "16.2"
      ],
      "description": "Query parameter loading and button state management"
    },
    {
      "id": 11,
      "tasks": [
        "12.3",
        "14.4",
        "15.2"
      ],
      "description": "Unit tests for GLB loading and manual loading"
    },
    {
      "id": 12,
      "tasks": [
        "17.1",
        "17.2",
        "17.3"
      ],
      "description": "Error handling and graceful degradation"
    },
    {
      "id": 13,
      "tasks": [
        "17.4",
        "19.2",
        "20.1"
      ],
      "description": "Unit tests for error handling and browser compatibility"
    },
    {
      "id": 14,
      "tasks": [
        "19.3",
        "19.4",
        "20.2"
      ],
      "description": "Browser and CORS compatibility verification"
    },
    {
      "id": 15,
      "tasks": [
        "21.1",
        "21.2",
        "21.3"
      ],
      "description": "Performance validation for scene, preview, and GLB loading"
    },
    {
      "id": 16,
      "tasks": [
        "21.4"
      ],
      "description": "Performance benchmarking"
    },
    {
      "id": 17,
      "tasks": [
        "22.1",
        "22.2",
        "22.3",
        "22.4",
        "22.5"
      ],
      "description": "Integration tests: end-to-end flows"
    },
    {
      "id": 18,
      "tasks": [
        "22.6",
        "22.7"
      ],
      "description": "Responsive layout and manual QA checklist"
    },
    {
      "id": 19,
      "tasks": [
        "23.1"
      ],
      "description": "Final checkpoint: all tests pass and feature complete"
    }
  ]
}
```

## Execution Notes

**Wave 0-3:** Establish foundation - HTML/CSS layout, Three.js scene, form components, lighting, animation loop

**Wave 4-7:** Implement form submission flow - API integration, response parsing, display logic

**Wave 8-11:** Implement viewer capabilities - GLB loading, manual loading, query parameters, button management

**Wave 12-16:** Polish and validation - Error handling, browser/CORS compatibility, performance testing

**Wave 17-19:** Testing and QA - Integration tests, responsive layout verification, final checkpoint

All leaf tasks (tasks with decimal notation like 1.1, 2.2, etc.) are independent within their wave and can be executed in parallel. Tasks within the same wave do not depend on each other, but tasks in wave N depend on completion of waves 0..N-1.


# Requirements Document

## Feature: End-to-End VTO UI

## Introduction

The Defirmation Virtual Try-On (VTO) system aims to provide users with a seamless, interactive experience for visualizing deformed eyewear frames in 3D. Users will upload three product images (front, side, top views) from their inventory, and the system will generate a realistic 3D GLB model that can be viewed in an interactive browser-based viewer with rotation, zoom, and lighting controls.

This requirements document specifies the complete end-to-end user experience, from image upload through 3D visualization, ensuring the browser UI connects the existing backend API (FastAPI on port 8000) with a professional, responsive Three.js viewer.

## Glossary

- **System**: The Defirmation VTO UI, a web application that accepts user inputs and displays 3D eyewear models
- **User**: A developer, product manager, or customer who uploads eyewear product images
- **Frontend**: Browser-based web UI built with HTML, CSS, and JavaScript (Three.js)
- **Backend**: FastAPI server running on port 8000 with endpoints `/api/deform` and `/api/deform/measurements`
- **Pipeline**: The backend processing sequence (segmentation, measurement extraction, mesh deformation, GLB export)
- **Job**: A single user request to generate a 3D model, identified by a unique job_id
- **GLB File**: A binary 3D model file in glTF 2.0 format (`.glb` extension)
- **Deformation**: The process of scaling and modifying mesh vertices to match detected eyewear measurements
- **Measurement**: Quantified dimensions extracted from images (frame width, lens width, bridge width, temple length, rim thickness)
- **Template**: A pre-built base 3D mesh (e.g., `geometric_metal`) that is deformed to match measurements
- **Upload Form**: The UI component allowing users to select and preview product images
- **Progress Indicator**: UI element displaying real-time feedback during backend processing
- **Viewer**: Three.js-based interactive 3D scene for displaying and manipulating the deformed model
- **Overlay Controls**: Text or visual indicators showing camera manipulation instructions (rotation, zoom)

## Requirements

### Requirement 1: User Upload and Image Selection

**User Story:** As a user, I want to upload three product images (front, side, top views) of eyewear frames, so that the system can measure and generate a 3D model.

#### Acceptance Criteria

1. THE Upload_Form SHALL display exactly three image input fields labeled "Front view", "Side view", and "Top view"
2. WHEN a user clicks an image input field, THE System SHALL open a browser file picker dialog accepting only image file types (JPEG, PNG, WebP, GIF)
3. WHEN a user selects a valid image file, THE System SHALL display a thumbnail preview (minimum 70px height) of the selected image within the input field
4. THE Upload_Form SHALL mark the Front view field as required with a visual indicator; Side view and Top view fields SHALL be marked as optional
5. WHEN the user has selected at least a Front view image, THE Generate_Button SHALL become enabled (change from disabled to clickable state); WHEN the user removes the Front view image, THE Generate_Button SHALL be disabled regardless of previous state
6. WHEN the user has not selected a Front view image OR other validation conditions prevent submission, THE Generate_Button SHALL remain disabled
7. WHEN a user removes a selected image and reselects a different image, THE System SHALL replace the previous preview with the new preview without page reload
8. WHEN a user clicks the Generate_Button without a Front view image, THE System SHALL not submit the form to the backend

### Requirement 2: Frame Color Selection and Template Configuration

**User Story:** As a user, I want to specify the frame color and optionally select a template style, so that the generated 3D model matches my product specifications.

#### Acceptance Criteria

1. THE Options_Panel SHALL display a color picker input labeled "Frame colour" with a default value of `#d9a7a2` (light rose)
2. WHEN the user interacts with the color picker and selects a new color, THE System SHALL update the displayed color swatch immediately without backend processing
3. THE Options_Panel SHALL display a text input field labeled "Template" with placeholder text "auto-detect"
4. WHEN the user leaves the Template input empty, THE System SHALL instruct the backend to auto-detect the template based on the uploaded images
5. WHEN the user enters a valid template name (e.g., "cat_eye_gold"), THE System SHALL send that template name to the backend in the deformation request
6. WHEN the Template input contains text, THE System SHALL not perform validation; validation SHALL occur on the backend
7. THE color and template values SHALL be persisted in the UI form until the user manually changes them

### Requirement 3: Backend API Integration for Image Upload and Processing

**User Story:** As a user, I want to submit my image selection to the backend and receive a 3D model, so that I can visualize the deformed eyewear frames.

#### Acceptance Criteria

1. WHEN the user clicks the Generate_Button, THE System SHALL collect the Front, Side (if provided), and Top (if provided) image files from the upload form
2. WHEN the user clicks the Generate_Button, THE System SHALL collect the Frame colour value and Template value (if provided) from the Options_Panel
3. WHEN the user clicks the Generate_Button, THE System SHALL construct a multipart/form-data POST request to `/api/deform` endpoint on the backend
4. WHEN the POST request is sent, THE System SHALL include the form parameters: `front` (required File), `side` (optional File), `top` (optional File), `color` (string in hex format, e.g., `#d9a7a2`), `template` (optional string)
5. WHEN the backend returns a successful response (HTTP 200), THE System SHALL extract the `download_url` field from the JSON response; IF the response contains any error indicator, THE System SHALL treat it as a definitive error and display the error instead of processing success indicators
6. WHEN the backend returns a successful response, THE System SHALL extract the `measurements` object containing fields: `frame_width`, `lens_width`, `lens_height`, `bridge_width`, `temple_length`, `rim_thickness`
7. WHEN the backend returns a successful response, THE System SHALL extract the `template` field (the auto-detected or user-specified template name)
8. WHEN the backend returns an error response (HTTP 4xx or 5xx), THE System SHALL display the error message from the response in the Status_Box with error styling (red text)
9. WHEN the backend returns an error response, THE System SHALL NOT attempt to load a 3D model into the Viewer
10. WHEN a network error occurs (no response from server), THE System SHALL catch the exception and display a user-friendly error message in the Status_Box

### Requirement 4: Real-Time Progress Feedback During Processing

**User Story:** As a user, I want to see real-time progress feedback while the backend is processing my images, so that I know the system is working and not frozen.

#### Acceptance Criteria

1. WHEN the user clicks the Generate_Button, THE System SHALL immediately disable the Generate_Button and change its text to "Generating…"
2. WHEN processing begins, THE System SHALL display "Running pipeline…" in the Status_Box before any backend response is received
3. WHEN the backend response is received and includes a `pipeline` array field, THE System SHALL render a Pipeline_Stages_Table displaying each processing stage
4. THE Pipeline_Stages_Table SHALL display columns: Stage number, Stage name (with emoji icon), Duration (in milliseconds), and Detail information
5. THE Pipeline_Stages_Table SHALL display rows for each stage in the `pipeline` array (e.g., "YOLO Segmentation", "Shape Classification", "Measurement Extraction", "Template Selection", "Template Deformation", "Texture Mapping", "GLB Export")
6. WHEN a stage has `status: "done"`, THE System SHALL display the stage status text in green (`#50fa7b`)
7. WHEN a stage has `status: "error"`, THE System SHALL display the stage status text in red (`#ff5555`)
8. WHEN the processing completes successfully, THE System SHALL re-enable the Generate_Button and restore its text to "Generate 3D Mesh"; WHEN processing fails, THE button remains disabled until a new generation attempt is made

### Requirement 5: Status Display with Measurement Summary

**User Story:** As a user, I want to see a summary of the extracted measurements and selected template after processing, so that I can verify the dimensions match my product.

#### Acceptance Criteria

1. WHEN the backend returns a successful response, THE System SHALL display a summary in the Status_Box with the format: "✓ Done · Template: {template_name}"
2. WHEN the backend returns a successful response, THE System SHALL append a second line displaying: "Frame {frame_width}mm · Lens {lens_width}×{lens_height}mm"
3. WHEN the backend returns a successful response, THE System SHALL append a third line displaying: "Bridge {bridge_width}mm · Temple {temple_length}mm · Rim {rim_thickness}mm"
4. THE Status_Box SHALL display this multi-line text with appropriate line breaks and no truncation
5. WHEN the user uploads a new set of images and clicks Generate_Button again, THE System SHALL replace the previous status content with new processing feedback

### Requirement 6: GLB Model Download Capability

**User Story:** As a user, I want to download the generated 3D model file so that I can use it in external applications or for archival.

#### Acceptance Criteria

1. WHEN processing completes successfully, THE System SHALL display a Download_Button labeled "⬇ Download GLB"
2. WHEN processing is in progress or not yet started, THE System SHALL hide (display: none) the Download_Button
3. WHEN the user clicks the Download_Button, THE System SHALL trigger a file download of the GLB file using the `download_url` provided by the backend
4. WHEN the file downloads, THE System SHALL name the downloaded file using the `job_id` from the backend response (e.g., `a1b2c3d4e5f6.glb`)
5. WHEN processing encounters an error, THE System SHALL hide the Download_Button and not display it until the next successful generation

### Requirement 7: Three.js 3D Viewer Initialization and Scene Setup

**User Story:** As a developer, I want the Three.js viewer to be properly initialized with a realistic scene, so that the 3D model displays correctly with appropriate lighting and camera positioning.

#### Acceptance Criteria

1. THE Viewer SHALL initialize a Three.js scene with a dark background color (`#1a1a2e` or similar)
2. THE Viewer SHALL initialize a perspective camera with a 45-degree field of view
3. THE Viewer SHALL add an ambient light source with intensity 0.6 or greater to provide base illumination
4. THE Viewer SHALL add a key (directional) light positioned at approximately (50, 80, 100) with intensity 1.2 or greater
5. THE Viewer SHALL add a fill light (secondary directional light) positioned opposite the key light with lower intensity (0.4 or less)
6. WHEN the Viewer initializes, THE WebGL renderer SHALL use antialiasing (antialias: true)
7. WHEN the Viewer initializes, THE WebGL renderer SHALL apply ACES Filmic tone mapping for realistic color rendering
8. WHEN the browser window resizes, THE System SHALL update the camera aspect ratio and renderer size to maintain proper display

### Requirement 8: GLB Model Loading and Display

**User Story:** As a user, I want the generated 3D model to load and display automatically after processing completes, so that I can immediately see the result without additional steps.

#### Acceptance Criteria

1. WHEN the backend returns a successful response with a `download_url`, THE System SHALL construct the full URL by concatenating the base API URL with the `download_url` path
2. WHEN the URL is constructed, THE System SHALL use the GLTFLoader to load the GLB file from that URL
3. WHEN the GLB file loads successfully, THE System SHALL remove any previously loaded model from the scene
4. WHEN the GLB file loads successfully, THE System SHALL add the new model to the Three.js scene
5. WHEN the GLB file loads successfully, THE System SHALL automatically fit the camera to frame the entire model in view (calculate bounding box and position camera accordingly)
6. WHEN the GLB file loads successfully, THE System SHALL update the OrbitControls target to the center of the model
7. WHEN the GLB file fails to load, THE System SHALL catch the error and display "Error loading model: {error_message}" in the overlay status text
8. WHILE the GLB file is loading, THE System SHALL display "Loading model…" in the overlay status text

### Requirement 9: Interactive Viewer Controls (Rotation, Zoom, Lighting)

**User Story:** As a user, I want to rotate, zoom, and adjust lighting to examine the 3D model from all angles, so that I can thoroughly evaluate the eyewear fit and appearance.

#### Acceptance Criteria

1. WHEN the user clicks and drags on the canvas, THE System SHALL rotate the model using OrbitControls (mouse drag = camera orbit around model)
2. WHEN the user scrolls the mouse wheel on the canvas, THE System SHALL zoom in and out by adjusting the camera distance from the model
3. WHEN the user interacts with the model, THE System SHALL display "Drag to rotate · Scroll to zoom" in the overlay status text
4. WHEN OrbitControls is enabled, THE System SHALL apply damping (enableDamping: true) to provide smooth, physics-based camera motion
5. WHEN the user stops dragging, THE System SHALL continue to decelerate smoothly until the camera comes to rest (damping effect)
6. THE Viewer SHALL maintain three light sources (ambient, key, fill) that remain fixed in world space and do not rotate with the camera
7. THE Viewer lighting setup SHALL create a realistic, professional appearance without harsh shadows or flat shading
8. WHEN the model is displayed, THE System SHALL automatically fit the camera to an appropriate distance (approximately 2.5× the maximum model dimension)

### Requirement 10: Manual GLB Loading (URL and File Input)

**User Story:** As a user, I want to manually load existing GLB models by URL or file upload, so that I can view previously generated models or external 3D files.

#### Acceptance Criteria

1. THE Manual_Load_Panel SHALL display a text input field labeled "Or load existing GLB" with placeholder text "/api/output/job.glb"
2. THE Manual_Load_Panel SHALL display a file input field that accepts `.glb` files
3. WHEN the user enters a URL in the text input and clicks "Load Model" button, THE System SHALL use the GLTFLoader to load the GLB from that URL
4. WHEN the user selects a `.glb` file from the file input and clicks "Load Model" button, THE System SHALL use the GLTFLoader to load the GLB from the File object (using URL.createObjectURL)
5. IF both the URL field and file input have values, THE System SHALL prioritize the file input (load from file, not URL)
6. WHEN the user attempts to load a model and no URL or file is selected, THE System SHALL not attempt to load and not display an error

### Requirement 11: URL Query Parameter Support

**User Story:** As a developer, I want to share links to pre-loaded 3D models via URL query parameters, so that users can view specific models without manual entry.

#### Acceptance Criteria

1. WHEN the page loads and the URL contains a query parameter `?glb={url}`, THE System SHALL automatically load the GLB file from that URL
2. WHEN the GLB file is automatically loaded from URL parameter, THE System SHALL not display an error if the file fails to load; instead, the page SHALL remain ready for manual input
3. WHEN the page loads without a `?glb=` query parameter, THE System SHALL display the default status "Upload a front image to begin."

### Requirement 12: Responsive UI Layout

**User Story:** As a user, I want the UI to display clearly on various screen sizes and maintain usability, so that I can use the application on desktop and tablet devices.

#### Acceptance Criteria

1. THE UI SHALL use a two-pane layout: a left sidebar (300px minimum width) containing controls, and a right canvas area (flex: 1) containing the 3D viewer
2. WHEN the browser window width is less than 900px, THE System MAY stack the sidebar and canvas vertically or use a responsive design (not required for initial release, but CSS SHALL be flexible enough to support this)
3. WHEN the browser window is resized, THE System SHALL update the renderer canvas size and camera aspect ratio to maintain correct 3D rendering
4. THE Sidebar controls (buttons, inputs, text areas) SHALL remain visible and scrollable if content exceeds the sidebar height
5. WHEN the user scrolls within the Sidebar, THE System SHALL scroll the control panel content, not the 3D model view

### Requirement 13: Error Handling and User Feedback

**User Story:** As a user, I want clear error messages when something goes wrong, so that I understand what failed and can take corrective action.

#### Acceptance Criteria

1. WHEN the user attempts to upload a file that is not a valid image, THE System SHALL allow the file to be selected (browser file picker does not validate); validation SHALL occur when the user clicks Generate_Button or when the backend processes the file
2. WHEN the backend returns an error (HTTP 4xx or 5xx), THE System SHALL display the error detail in the Status_Box with red text styling (class="error")
3. WHEN a network error occurs (e.g., backend server is unreachable), THE System SHALL display "Network error: {error.message}" in the Status_Box with red text styling
4. WHEN the GLTFLoader fails to load a GLB file, THE System SHALL display "Error loading model: {error.message}" in the overlay status with red or warning styling
5. THE Status_Box SHALL remain visible at all times and be the primary location for user-facing error messages

### Requirement 14: Form State Persistence

**User Story:** As a user, I want my form selections (uploaded images, color choice, template name) to persist until I manually change them, so that I can iterate on designs without re-entering choices.

#### Acceptance Criteria

1. WHEN the user selects an image, changes the color, or enters a template name, THE System SHALL retain these values in the DOM (form elements)
2. WHEN the user clicks Generate_Button and the backend processing completes, THE System SHALL NOT clear the form inputs
3. WHEN the user clicks Generate_Button again (with the same or different images), THE System SHALL submit the current form values without requiring the user to re-enter them
4. WHEN the user manually clears an image field or reselects a different image, THE System SHALL update only that field; other fields SHALL retain their values
5. THE Frontend application SHALL NOT use browser localStorage or sessionStorage for form persistence (form state persists only during the browser session)

### Requirement 15: Accessibility and Visual Indicators

**User Story:** As a user, I want clear visual indicators for button states, required fields, and processing status, so that I can understand the application state at a glance.

#### Acceptance Criteria

1. THE Generate_Button SHALL display a disabled state (grayed out, cursor: not-allowed) when no Front image is selected
2. THE Generate_Button SHALL display an enabled state (red background, clickable) when a Front image is selected
3. THE Generate_Button text SHALL change to "Generating…" during backend processing and revert to "Generate 3D Mesh" when complete
4. THE Front_View_Upload_Field SHALL display a "required" badge in a distinct color (e.g., red or pink) to indicate it is mandatory
5. THE Side_View_Upload_Field and Top_View_Upload_Field SHALL display "optional" badges to indicate they are not required
6. THE Download_Button SHALL initially be hidden (display: none) and only appear after successful model generation
7. THE Status_Box text SHALL display green status (`✓ Done`) for successful completions and red text for errors
8. WHEN OrbitControls is active, THE System SHALL display "Drag to rotate · Scroll to zoom" as an always-visible overlay

### Requirement 16: API Response Validation

**User Story:** As a developer, I want the frontend to validate API responses for consistency and completeness, so that unexpected backend responses do not cause crashes.

#### Acceptance Criteria

1. WHEN the backend returns a response, THE System SHALL check that the response status is 200 (success) before processing the JSON body
2. WHEN the backend response includes `download_url`, THE System SHALL verify it is a non-empty string before using it in GLTFLoader
3. WHEN the backend response includes `measurements` object, THE System SHALL verify it contains all expected fields (frame_width, lens_width, lens_height, bridge_width, temple_length, rim_thickness) before displaying them
4. IF the backend response is missing required fields, THE System SHALL display a partial status message using available fields and skip missing ones (graceful degradation); if graceful handling fails, THE System SHALL show a partial status message or a generic error message
5. WHEN the backend response includes a `pipeline` array field, THE System SHALL iterate through the array and render each stage; if a stage is missing expected properties (name, duration_ms, status), THE System SHALL handle the missing property gracefully and continue rendering

### Requirement 17: Performance and Load Time Optimization

**User Story:** As a user, I want the application to load quickly and respond to my interactions without noticeable lag, so that the experience feels smooth and professional.

#### Acceptance Criteria

1. THE Frontend application SHALL be a single HTML file with inline or linked CSS and JavaScript (no additional HTTP requests for styles, except Three.js and GLTFLoader from CDN)
2. WHEN the page loads, THE Three.js scene initialization and renderer setup SHALL complete in under 500ms
3. WHEN the user selects an image, THE System SHALL display the thumbnail preview in under 100ms (thumbnail generation or display shall be performant)
4. WHEN the user interacts with the 3D model (rotate, zoom), THE System SHALL render frames at 60fps or higher (OrbitControls update and render loop)
5. WHEN the GLB file is loading from the backend, THE System SHALL show loading feedback ("Loading model…") to indicate the page is responsive

### Requirement 18: Browser Compatibility

**User Story:** As a user, I want the application to work on modern web browsers, so that I can access it from my preferred browser.

#### Acceptance Criteria

1. THE Application SHALL support Chrome/Chromium version 90 or later
2. THE Application SHALL support Firefox version 88 or later
3. THE Application SHALL support Safari version 14 or later
4. THE Application SHALL use Three.js from a CDN (jsDelivr or similar) to ensure availability
5. THE Application SHALL use standard HTML5 and CSS3 features; no proprietary browser extensions SHALL be required
6. WHEN the browser does not support WebGL, THE System SHALL display a clear message to the user (graceful degradation or clear error message is sufficient; fallback functionality is not required)

### Requirement 19: CORS and Cross-Origin Requests

**User Story:** As a developer, I want the frontend to handle CORS correctly when communicating with the backend API, so that requests are not blocked by browser security policies.

#### Acceptance Criteria

1. THE Backend API (FastAPI) SHALL allow CORS requests from all origins (allow_origins=["*"]) to support development and flexible deployment
2. WHEN the frontend makes a POST request to `/api/deform`, THE System SHALL use standard fetch API with method: "POST" and body: FormData
3. WHEN the frontend makes a GET request to `/api/output/{filename}`, THE System SHALL use standard fetch or direct URL assignment
4. IF a CORS error occurs, THE System SHALL display a network error message to the user

### Requirement 20: Integration of Existing Viewer HTML

**User Story:** As a developer, I want to leverage the existing Three.js viewer HTML template, so that I do not duplicate code and maintain consistency with the designed UI.

#### Acceptance Criteria

1. THE Frontend application SHALL use the existing `viewer/index.html` template located at the project root
2. THE Existing Three.js setup (scene, lighting, camera, OrbitControls) in `viewer/index.html` SHALL be preserved and reused
3. THE Existing event handlers for image upload, Generate_Button, and status display in `viewer/index.html` SHALL be updated to use the correct backend endpoint and response handling
4. WHEN the backend is integrated, THE System SHALL not require rewriting the entire viewer; only backend integration updates SHALL be needed


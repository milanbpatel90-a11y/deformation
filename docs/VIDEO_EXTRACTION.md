# Eyewear video-to-reference extraction

This subsystem extracts reviewable 2D references; it is not photogrammetry and does not directly reconstruct a complete 3D mesh. The existing descriptor/template deformation pipeline remains the intended GLB generator.

## Recording guidance
Record one pair of glasses on a plain contrasting background. Use diffuse lighting, avoid glare, keep both temples inside the image, and slowly show front, front-left/right three-quarter, both sides and top views. Hold each view steady for about one second. Add a ruler/reference marker in the same plane if physical calibration is needed, or enter verified physical dimensions.

## Input and configuration
MP4 and MOV are supported. Defaults: 150 MiB maximum, 90 seconds, 480x360 minimum, 3840x2160 maximum and 60 FPS maximum. Environment options: VIDEO_MAX_UPLOAD_BYTES, VIDEO_MAX_DURATION_SECONDS, VIDEO_SAMPLE_INTERVAL_SECONDS, VIDEO_MAX_SAMPLE_FRAMES, VIDEO_MAX_SELECTED_FRAMES and VIDEO_MIN_DETECTION_CONFIDENCE. Install FFmpeg and ffprobe in the Railway image for robust codec inspection.

## Processing and limitations
Metadata validation -> bounded timestamp sampling -> quality/object checks -> perceptual duplicate rejection -> conservative view classification -> existing YOLO semantic segmentation -> merchant review -> explicit calibration -> template deformation -> GLB validation. YOLO semantic masks do not determine camera pose. Current bbox heuristic can only cautiously identify front-like views; unknown is safer for other views until a pose model or manual labels are available. Missing component masks remain missing. Pixel measurements are never passed as millimetres. Front calibration cannot calibrate side/top views. Rim thickness and projected temple extent are low-confidence approximations.

## Deployment
Set VIDEO_STORAGE_BACKEND=s3 and WASABI_BUCKET, WASABI_ENDPOINT_URL, WASABI_REGION, WASABI_ACCESS_KEY_ID and WASABI_SECRET_ACCESS_KEY. Keep the bucket private. Put VIDEO_EXTRACTION_DB_PATH on a persistent Railway volume. Configure VIDEO_MAX_CONCURRENT_JOBS. Install FFmpeg/ffprobe. Local storage is for development only.

## Security requirements for API integration
Every job lookup and asset fetch must verify authenticated merchant ownership; admin role access follows the repository owner role. Never return filesystem paths or raw object keys. Use an authenticated asset proxy or short-lived signed URLs after ownership checks. Delete temporary workspaces on all exits. Do not automatically approve low-quality frames.

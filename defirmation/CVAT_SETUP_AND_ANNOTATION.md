# 🎯 CVAT Setup and Annotation Guide

Complete guide for setting up CVAT and annotating eyewear images for instance segmentation.

---

## 📋 Table of Contents

1. [CVAT Setup Options](#cvat-setup-options)
2. [Quick Start (Online - Recommended)](#quick-start-online)
3. [Local Docker Setup](#local-docker-setup)
4. [Creating Your Project](#creating-your-project)
5. [Uploading Images](#uploading-images)
6. [Annotation Workflow](#annotation-workflow)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Exporting Annotations](#exporting-annotations)
9. [Troubleshooting](#troubleshooting)

---

## 🌐 CVAT Setup Options

### Option A: Online (EASIEST - Recommended for Beginners)
- ✅ No installation required
- ✅ Works immediately
- ✅ Free tier available
- ✅ Accessible from anywhere
- ⚠️ Requires internet connection
- ⚠️ Limited storage on free tier

### Option B: Local Docker (Advanced)
- ✅ Full control
- ✅ Unlimited storage
- ✅ Works offline
- ✅ No data privacy concerns
- ⚠️ Requires Docker installation
- ⚠️ Uses local resources

---

## 🚀 Quick Start (Online)

### Step 1: Sign Up (2 minutes)

1. Go to: **https://cvat.ai/**
2. Click **"Sign Up"** (top right)
3. Fill in:
   - Email
   - Username
   - Password
4. Verify email
5. Log in

### Step 2: Create Organization (Optional)

1. Click your profile icon
2. Select **"Create Organization"**
3. Name: `eyewear_project` (or any name)
4. Click **"Submit"**

### Step 3: Create Project

1. Click **"Projects"** in sidebar
2. Click **"+"** button (Create new project)
3. Fill in:
   - **Name:** `eyewear_segmentation`
   - **Labels:** Click "Add label"
     - **Label name:** `eyewear`
     - **Type:** Keep default (any shape)
   - Click **"Submit"**

### Step 4: Create Task

1. Inside your project, click **"+"** (Create new task)
2. Fill in:
   - **Name:** `batch_001_60_images`
   - **Subset:** `train` (or leave empty)
3. Click **"Submit"**

### Step 5: Upload Images

1. Click on your task name
2. Click **"Upload data"**
3. Select upload method:
   - **Local files:** Drag & drop or browse
   - **Remote files:** Paste URLs
4. Select all 60 images from `dataset/images_raw/`
5. Click **"Submit"**
6. Wait for upload (may take 5-10 minutes)

---

## 🐳 Local Docker Setup

### Prerequisites

- Docker Desktop installed
- 8GB+ RAM available
- 10GB+ disk space

### Installation Steps

```bash
# 1. Clone CVAT repository
git clone https://github.com/opencv/cvat.git
cd cvat

# 2. Start CVAT services
docker compose up -d

# 3. Wait for services to start (2-3 minutes)
# Check status:
docker compose ps

# 4. Access CVAT
# Open browser: http://localhost:8080

# 5. Default credentials
# Username: admin
# Password: changeme
```

### First Time Setup

1. Log in with default credentials
2. Change password immediately:
   - Click profile icon → Settings → Change password
3. Follow same project creation steps as online version

### Stopping CVAT

```bash
# Stop services
docker compose stop

# Start again later
docker compose start

# Remove completely
docker compose down
```

---

## 🎨 Creating Your Project

### Project Configuration

**Project Name:** `eyewear_segmentation`

**Task Type:** Instance Segmentation

**Labels:**
- **Label 1:** `eyewear`
  - Color: Choose any (e.g., blue)
  - Attributes: None needed for now

**Advanced Settings (Optional):**
- Bug tracker: Leave empty
- Source storage: Local
- Target storage: Local

---

## 📤 Uploading Images

### Best Practices

1. **Organize before upload:**
   ```
   dataset/images_raw/
   ├── unsplash_*.jpg (50 images)
   └── diy_*.jpg (10 images)
   ```

2. **Upload in batches:**
   - Batch 1: 60 images (today)
   - Batch 2: 30 images (tomorrow)
   - etc.

3. **Naming convention:**
   - Keep original filenames
   - CVAT will preserve them

### Upload Methods

**Method 1: Drag & Drop**
- Select all images in file explorer
- Drag into CVAT upload area
- Wait for upload

**Method 2: Browse Files**
- Click "Select files"
- Multi-select images (Ctrl+A or Cmd+A)
- Click "Open"

**Method 3: Remote URLs** (for Unsplash direct)
- Paste image URLs (one per line)
- CVAT downloads them

---

## ✏️ Annotation Workflow

### Starting Annotation

1. Click on your task
2. Click **"Job #1"** (or any job)
3. You'll see the annotation interface

### Annotation Interface Overview

```
┌─────────────────────────────────────────────────┐
│  [Tools] [Shapes] [Labels]        [Save] [Next] │
├─────────────────────────────────────────────────┤
│                                                  │
│                                                  │
│              IMAGE CANVAS                        │
│                                                  │
│                                                  │
├─────────────────────────────────────────────────┤
│  Frame: 1/60    [<] [>]    Zoom: 100%           │
└─────────────────────────────────────────────────┘
```

### Step-by-Step Annotation

#### For Each Image:

**1. Select Polygon Tool**
   - Click polygon icon (or press `N`)
   - Or press `G` for AI-assisted polygon

**2. Draw Around Eyewear**
   - Click to place points around eyewear outline
   - Follow the contour closely
   - 15-30 points is usually enough
   - Double-click or press `N` to finish

**3. Verify Label**
   - Should auto-assign "eyewear" label
   - If not, select from dropdown

**4. Refine if Needed**
   - Click on polygon to select
   - Drag points to adjust
   - Add points: Click on edge
   - Delete points: Select point + Delete key

**5. Save & Next**
   - Press `Ctrl+S` to save
   - Press `F` to go to next frame

### Annotation Tips

**✅ DO:**
- Follow eyewear edges precisely
- Include temples (arms) if visible
- Annotate both lenses as one object
- Zoom in (scroll wheel) for precision
- Take breaks every 10 images

**❌ DON'T:**
- Rush through annotations
- Skip partially visible eyewear
- Annotate reflections separately
- Forget to save (Ctrl+S)

### Quality Checklist

For each annotation, verify:
- [ ] Polygon follows eyewear outline
- [ ] No gaps or overlaps
- [ ] Temples included if visible
- [ ] Label is "eyewear"
- [ ] Saved (green checkmark)

---

## ⌨️ Keyboard Shortcuts

### Essential Shortcuts

| Key | Action |
|-----|--------|
| `N` | Polygon tool |
| `G` | AI-assisted polygon (magic wand) |
| `Ctrl+S` | Save annotations |
| `F` | Next frame/image |
| `D` | Previous frame/image |
| `Space` | Play/pause (for videos) |
| `+` / `-` | Zoom in/out |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Delete` | Delete selected shape |
| `Esc` | Cancel current action |

### Advanced Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Copy shape |
| `Ctrl+V` | Paste shape |
| `H` | Hide/show labels |
| `Ctrl+H` | Hide/show UI |
| `1-9` | Switch between labels |
| `Shift+Click` | Multi-select shapes |

### Pro Tips

1. **Speed up with AI polygon:**
   - Press `G`
   - Click inside eyewear
   - CVAT auto-traces edges
   - Adjust if needed

2. **Batch operations:**
   - Select multiple shapes (Shift+Click)
   - Change label for all at once

3. **Quick navigation:**
   - `F` → `F` → `F` to skip through
   - Use slider for big jumps

---

## 📦 Exporting Annotations

### Export Formats

**For YOLOv8 Segmentation:**
- Format: **COCO 1.0** or **YOLO 1.1**
- Includes: Images + JSON annotations

### Export Steps

1. Go to your task
2. Click **"Actions"** → **"Export task dataset"**
3. Select format:
   - **COCO 1.0** (recommended)
   - Or **YOLO 1.1** (direct YOLOv8 format)
4. Click **"Export"**
5. Download ZIP file
6. Extract to `dataset/annotations/`

### Post-Export Processing

```bash
# Extract annotations
unzip task_batch_001-coco.zip -d dataset/annotations/

# Verify structure
ls dataset/annotations/
# Should see:
# - annotations/
# - images/
# - instances_default.json
```

### Converting COCO to YOLO Format

If you exported COCO format, convert to YOLO:

```python
# Use this script (create convert_coco_to_yolo.py)
from pycocotools.coco import COCO
import os

coco = COCO('dataset/annotations/instances_default.json')
# ... conversion code ...
```

Or use existing tools:
```bash
pip install coco2yolo
coco2yolo --coco_json dataset/annotations/instances_default.json \
          --output dataset/labels/
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Upload Fails
**Problem:** Images won't upload
**Solutions:**
- Check file size (max 50MB per image)
- Check format (JPG, PNG supported)
- Try smaller batches (10-20 images)
- Check internet connection

#### 2. Polygon Won't Close
**Problem:** Can't finish polygon
**Solutions:**
- Double-click last point
- Press `N` key
- Right-click → "Finish"

#### 3. Can't See Annotations
**Problem:** Annotations invisible
**Solutions:**
- Press `H` to toggle visibility
- Check opacity slider (top right)
- Zoom in (might be too small)

#### 4. Slow Performance
**Problem:** CVAT is laggy
**Solutions:**
- Close other browser tabs
- Reduce image quality (Settings)
- Use local Docker version
- Annotate in smaller batches

#### 5. Lost Progress
**Problem:** Annotations disappeared
**Solutions:**
- Check if saved (Ctrl+S)
- Look in "Actions" → "View history"
- Contact CVAT support if online
- Check Docker logs if local

### Getting Help

**Online CVAT:**
- Forum: https://github.com/opencv/cvat/discussions
- Docs: https://opencv.github.io/cvat/docs/

**Local Docker:**
- Check logs: `docker compose logs cvat`
- Restart: `docker compose restart`

---

## 📊 Progress Tracking

### Daily Goals

**Today (Day 1):**
- [ ] 60 images uploaded
- [ ] 10 images annotated
- [ ] Familiar with interface

**Tomorrow (Day 2):**
- [ ] +30 images uploaded (90 total)
- [ ] +20 images annotated (30 total)

**Day 3:**
- [ ] +30 images uploaded (120 total)
- [ ] +30 images annotated (60 total)

**Day 4:**
- [ ] +40 images uploaded (160 total)
- [ ] +40 images annotated (100 total)

**Day 5:**
- [ ] +40 images uploaded (200 total)
- [ ] +100 images annotated (200 total) ✅

### Time Estimates

- **Setup CVAT:** 5-10 minutes
- **Upload 60 images:** 5-10 minutes
- **Annotate 1 image:** 3-5 minutes
- **Annotate 10 images:** 40-50 minutes
- **Export annotations:** 2-3 minutes

---

## 🎯 Success Criteria

By end of today, you should have:

✅ CVAT account created (online) or Docker running (local)
✅ Project "eyewear_segmentation" created
✅ Task with 60 images uploaded
✅ 10 images fully annotated
✅ Comfortable with annotation interface
✅ Ready to scale up tomorrow

---

## 📚 Additional Resources

**CVAT Documentation:**
- Official Docs: https://opencv.github.io/cvat/docs/
- Video Tutorials: https://www.youtube.com/c/CVAT

**Annotation Best Practices:**
- COCO Dataset Guidelines
- YOLOv8 Segmentation Docs

**Community:**
- CVAT GitHub: https://github.com/opencv/cvat
- Discord/Slack: Check CVAT website

---

## 🚀 Next Steps After Annotation

1. **Export annotations** (COCO or YOLO format)
2. **Verify quality** (spot-check 10 random images)
3. **Convert to YOLOv8 format** (if needed)
4. **Split dataset** (train/val/test)
5. **Start training** YOLOv8 segmentation model

---

**Good luck with your annotations! 🎨**

Remember: Quality > Speed. Take your time on the first 10 images to build good habits.
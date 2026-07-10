# 🚀 TODAY'S EXECUTION PLAN - Data Collection Sprint

**Goal:** Collect 60 images + Annotate 10 images + CVAT setup  
**Time:** 3 hours  
**Date:** Today

---

## ⏱️ Timeline Overview

```
NOW → 0:15    Download 50 Unsplash images (background)
0:15 → 0:45   Take 10 DIY photos (parallel)
0:45 → 1:00   Verify & organize 60 images
1:00 → 1:15   Set up CVAT + Create project
1:15 → 1:25   Upload 60 images to CVAT
1:25 → 2:05   Annotate first 10 images
2:05          ✅ DONE!
```

---

## 📋 Step-by-Step Checklist

### Phase 1: Download Images (15 min)

#### Step 1.1: Get Unsplash API Key (5 min)

**If you don't have an API key yet:**

1. Go to: https://unsplash.com/developers
2. Click **"Register as a Developer"**
3. Accept terms
4. Click **"New Application"**
5. Fill in:
   - Application name: `Eyewear Dataset Collection`
   - Description: `Collecting eyewear images for ML training`
6. Accept API terms
7. Click **"Create Application"**
8. Copy your **Access Key**

**Configure the script:**

```bash
# Open download_unsplash_glasses.py
# Find line 15:
UNSPLASH_ACCESS_KEY = "YOUR_ACCESS_KEY_HERE"

# Replace with your actual key:
UNSPLASH_ACCESS_KEY = "your_actual_key_here"
```

- [ ] Unsplash account created
- [ ] API key obtained
- [ ] Script configured with API key

#### Step 1.2: Run Download Script (10 min)

```bash
# Create output directory
mkdir -p dataset/images_raw

# Run download (this will take ~10 minutes)
python download_unsplash_glasses.py --count 50 --output dataset/images_raw
```

**Expected output:**
```
🔍 Starting download of 50 eyewear images from Unsplash...
📁 Output directory: dataset/images_raw

🔎 Searching: 'eyeglasses'...
  Found 30 images
✓ Downloaded: unsplash_eyeglasses_abc123.jpg
  Progress: 1/50
...
✅ Download complete! 50 images saved to dataset/images_raw
```

**While downloading, move to Phase 2!**

- [ ] Download script started
- [ ] Script running in background

---

### Phase 2: DIY Photos (30 min - PARALLEL)

#### Step 2.1: Gather Equipment (5 min)

**You need:**
- [ ] Eyeglasses (yours or borrowed)
- [ ] Phone camera or webcam
- [ ] White/plain background (wall, paper, cloth)
- [ ] Good lighting (near window or lamp)

**Optional but helpful:**
- [ ] Tripod or phone stand
- [ ] Multiple eyewear styles
- [ ] Different angles/positions

#### Step 2.2: Setup Photo Station (5 min)

1. **Background:**
   - Hang white sheet or use white wall
   - Or use plain colored background
   - Avoid busy patterns

2. **Lighting:**
   - Natural light is best (near window)
   - Or use desk lamp
   - Avoid harsh shadows

3. **Camera:**
   - Position phone/camera
   - Landscape orientation
   - Stable position

- [ ] Background ready
- [ ] Lighting set up
- [ ] Camera positioned

#### Step 2.3: Take 10 Photos (20 min)

**Photo checklist for each shot:**

**Photo 1-3: Front view**
- Glasses centered
- Both lenses visible
- Clear focus
- [ ] diy_001.jpg
- [ ] diy_002.jpg
- [ ] diy_003.jpg

**Photo 4-6: Angled view**
- 45-degree angle
- Temple visible
- Good lighting
- [ ] diy_004.jpg
- [ ] diy_005.jpg
- [ ] diy_006.jpg

**Photo 7-8: Close-up**
- Fill frame with glasses
- Detail visible
- Sharp focus
- [ ] diy_007.jpg
- [ ] diy_008.jpg

**Photo 9-10: Different styles**
- Different eyewear if available
- Or different positions
- Variety is good
- [ ] diy_009.jpg
- [ ] diy_010.jpg

#### Step 2.4: Transfer & Rename (5 min)

```bash
# Transfer photos from phone to computer
# Rename to: diy_001.jpg, diy_002.jpg, etc.
# Move to: dataset/images_raw/

# Example commands:
# Windows:
copy C:\Users\YourName\Pictures\*.jpg dataset\images_raw\
rename dataset\images_raw\IMG_001.jpg diy_001.jpg

# Mac/Linux:
cp ~/Pictures/*.jpg dataset/images_raw/
mv dataset/images_raw/IMG_001.jpg dataset/images_raw/diy_001.jpg
```

- [ ] Photos transferred to computer
- [ ] Renamed to diy_001.jpg through diy_010.jpg
- [ ] Moved to dataset/images_raw/

---

### Phase 3: Verify Collection (15 min)

#### Step 3.1: Check Download Complete

```bash
# Check if download finished
# Look for "Download complete!" message

# Count files
ls dataset/images_raw/ | wc -l
# Should show: 60 (50 Unsplash + 10 DIY)
```

- [ ] Download script completed
- [ ] 50 Unsplash images downloaded
- [ ] No error messages

#### Step 3.2: Verify All Images

```bash
# List all images
ls -lh dataset/images_raw/

# Check file sizes (should be 100KB - 5MB each)
# Check file types (all .jpg or .png)
```

**Quick visual check:**
- [ ] Open 5 random images
- [ ] All show eyewear clearly
- [ ] No corrupted files
- [ ] Good variety of styles

#### Step 3.3: Organize (if needed)

```bash
# Optional: Create subdirectories
mkdir -p dataset/images_raw/unsplash
mkdir -p dataset/images_raw/diy

# Move files
mv dataset/images_raw/unsplash_*.jpg dataset/images_raw/unsplash/
mv dataset/images_raw/diy_*.jpg dataset/images_raw/diy/
```

- [ ] 60 total images confirmed
- [ ] All images valid
- [ ] Ready for upload

---

### Phase 4: CVAT Setup (15 min)

#### Step 4.1: Choose CVAT Option

**Option A: Online (Recommended)**
- Faster setup
- No installation
- Works immediately
- → Go to Step 4.2

**Option B: Local Docker**
- More control
- Offline capable
- → Go to Step 4.3

#### Step 4.2: Online CVAT Setup (10 min)

1. **Sign up:**
   - Go to: https://cvat.ai/
   - Click "Sign Up"
   - Email: _______________
   - Username: _______________
   - Password: _______________
   - [ ] Account created
   - [ ] Email verified
   - [ ] Logged in

2. **Create project:**
   - Click "Projects" → "+"
   - Name: `eyewear_segmentation`
   - [ ] Project created

3. **Add label:**
   - In project settings
   - Click "Add label"
   - Name: `eyewear`
   - Color: Blue (or any)
   - [ ] Label added

4. **Create task:**
   - Click "+" in project
   - Name: `batch_001_60_images`
   - [ ] Task created

**→ Skip to Phase 5**

#### Step 4.3: Local Docker Setup (15 min)

```bash
# 1. Clone CVAT
git clone https://github.com/opencv/cvat.git
cd cvat

# 2. Start services
docker compose up -d

# 3. Wait for startup (2-3 min)
# Watch logs:
docker compose logs -f

# 4. Access CVAT
# Open: http://localhost:8080
# Login: admin / changeme
```

- [ ] Docker installed
- [ ] CVAT cloned
- [ ] Services started
- [ ] Accessible at localhost:8080
- [ ] Logged in

**Create project (same as online):**
- [ ] Project created: `eyewear_segmentation`
- [ ] Label added: `eyewear`
- [ ] Task created: `batch_001_60_images`

---

### Phase 5: Upload Images (10 min)

#### Step 5.1: Prepare Upload

1. **Open your task:**
   - Click on `batch_001_60_images`
   - Click "Upload data"

2. **Select upload method:**
   - Choose "Local files"
   - Or drag & drop

- [ ] Task opened
- [ ] Upload dialog ready

#### Step 5.2: Upload All 60 Images

```bash
# Select all images:
# Navigate to: dataset/images_raw/
# Select all (Ctrl+A or Cmd+A)
# Drag into CVAT upload area
```

**Upload progress:**
- [ ] Upload started
- [ ] Progress bar visible
- [ ] Wait for completion (5-10 min)

**Verify upload:**
- [ ] "Upload complete" message
- [ ] 60 frames shown
- [ ] No errors

---

### Phase 6: Annotate First 10 Images (40 min)

#### Step 6.1: Open Annotation Interface

1. Click on your task
2. Click "Job #1"
3. Annotation interface opens

- [ ] Annotation interface loaded
- [ ] First image visible
- [ ] Tools panel visible

#### Step 6.2: Learn the Interface (5 min)

**Quick orientation:**
- [ ] Found polygon tool (N key)
- [ ] Found save button (Ctrl+S)
- [ ] Found next button (F key)
- [ ] Tested zoom (scroll wheel)

**Practice on first image:**
- [ ] Drew test polygon
- [ ] Deleted it (Delete key)
- [ ] Ready to start for real

#### Step 6.3: Annotate Images 1-10 (35 min)

**For EACH image (repeat 10 times):**

**Image 1:**
1. Press `N` for polygon tool
2. Click around eyewear outline (15-30 points)
3. Double-click to finish
4. Verify label is "eyewear"
5. Press `Ctrl+S` to save
6. Press `F` for next image
- [ ] Image 1 annotated

**Image 2:**
- [ ] Image 2 annotated

**Image 3:**
- [ ] Image 3 annotated

**Image 4:**
- [ ] Image 4 annotated

**Image 5:**
- [ ] Image 5 annotated

**Image 6:**
- [ ] Image 6 annotated

**Image 7:**
- [ ] Image 7 annotated

**Image 8:**
- [ ] Image 8 annotated

**Image 9:**
- [ ] Image 9 annotated

**Image 10:**
- [ ] Image 10 annotated

#### Step 6.4: Quality Check (5 min)

**Review your annotations:**
- [ ] Go back to image 1 (press `D` repeatedly)
- [ ] Check 3 random annotations
- [ ] Verify polygons follow edges
- [ ] All labeled correctly
- [ ] All saved (green checkmarks)

---

## ✅ Final Verification

### Completion Checklist

**Data Collection:**
- [ ] 50 Unsplash images downloaded
- [ ] 10 DIY photos taken
- [ ] 60 total images in dataset/images_raw/
- [ ] All images valid and clear

**CVAT Setup:**
- [ ] CVAT account/instance running
- [ ] Project created: eyewear_segmentation
- [ ] Label created: eyewear
- [ ] Task created with 60 images

**Annotations:**
- [ ] 10 images fully annotated
- [ ] Quality verified
- [ ] All saved in CVAT

**Time Spent:**
- Download: _____ min
- DIY photos: _____ min
- CVAT setup: _____ min
- Annotation: _____ min
- **Total: _____ min** (target: 180 min)

---

## 📊 Today's Achievements

```
✅ 60 images collected (30% of 200 target)
✅ 10 images annotated (5% of 200 target)
✅ CVAT infrastructure ready
✅ Workflow established
```

---

## 🎯 Tomorrow's Plan

**Day 2 Goals:**
- [ ] Download +30 Unsplash images (90 total)
- [ ] Take +0 DIY photos (or more if you want)
- [ ] Annotate +20 images (30 total)
- [ ] Time: 2 hours

**Commands for tomorrow:**
```bash
# Download more images
python download_unsplash_glasses.py --count 30 --output dataset/images_raw

# Upload to CVAT (create new task: batch_002_30_images)
# Annotate 20 more images
```

---

## 🔧 Troubleshooting

### If Download Fails:
```bash
# Check API key is set
grep "UNSPLASH_ACCESS_KEY" download_unsplash_glasses.py

# Check internet connection
ping unsplash.com

# Try smaller batch
python download_unsplash_glasses.py --count 10 --output dataset/images_raw
```

### If CVAT Upload Fails:
- Try smaller batches (10-20 images)
- Check file sizes (max 50MB each)
- Check internet connection
- Try different browser

### If Annotation is Slow:
- Close other browser tabs
- Reduce image quality in CVAT settings
- Take breaks every 10 images
- Use keyboard shortcuts

---

## 📞 Need Help?

**CVAT Issues:**
- Docs: https://opencv.github.io/cvat/docs/
- Forum: https://github.com/opencv/cvat/discussions

**Script Issues:**
- Check Python version: `python --version` (need 3.7+)
- Install requirements: `pip install requests pillow`

**General Questions:**
- Review CVAT_SETUP_AND_ANNOTATION.md
- Check DATA_SOURCING_STRATEGY.md

---

## 🎉 Celebrate Your Progress!

You're building a real ML dataset from scratch. This is the foundation of your eyewear segmentation model. Every image you annotate brings you closer to a working system.

**Keep going! 💪**

---

**Start NOW:** Run the download script and begin your sprint! 🚀
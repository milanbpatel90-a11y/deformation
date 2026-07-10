# Training Data Strategy: 200+ Eyewear Product Images

## Goal
Collect diverse eyewear product photos with clear segmentation masks for YOLOv8-Seg training.

**Target:** 200-300 images  
**Breakdown:** 50% metal, 30% plastic, 15% acetate, 5% other styles  
**Quality:** Clear, well-lit, 640x480+ resolution, glasses clearly visible  

---

## Option A: E-Commerce Scraping (50 images/hour)

### Best Sources
1. **Warby Parker** (warbyparker.com)
   - High-quality product photos
   - Multiple angles per frame
   - Terms: Personal use, educational only
   - Extraction: Browser dev tools → image URLs

2. **GlassesUSA** (glassesusa.com)
   - Large inventory (10k+ frames)
   - Consistent lighting, clean backgrounds
   - Format: JPG, 640x480+

3. **EyeBuyDirect** (eyebuydirect.com)
   - Diverse styles
   - Product photos + lifestyle shots

4. **Firmoo** (firmoo.com)
   - Large catalog
   - International brands

### Scraping Script
```python
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

def scrape_frames_from_warby_parker():
    """Scrape product images from WP catalog"""
    
    url = "https://www.warbyparker.com/eyeglasses"
    
    # Use browser automation (Selenium) or direct requests
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract image URLs
    images = soup.find_all('img', class_='product-image')
    
    out_dir = Path("dataset/images_raw")
    out_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(images[:50]):  # Limit to 50
        if img.get('src'):
            img_url = img['src']
            if not img_url.startswith('http'):
                img_url = "https://www.warbyparker.com" + img_url
            
            try:
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    out_path = out_dir / f"warby_parker_{i:03d}.jpg"
                    out_path.write_bytes(response.content)
                    print(f"✓ {out_path.name}")
            except Exception as e:
                print(f"✗ Failed: {img_url} ({e})")
            
            time.sleep(0.5)  # Polite delay

scrape_frames_from_warby_parker()
```

**Time Investment:** 2-3 hours for 100 images  
**Quality:** ⭐⭐⭐⭐ (high)  
**Legality:** ⚠️ Check ToS (personal/educational use typically allowed)

---

## Option B: Stock Photo Sites (Free/Paid)

### Free Sources
1. **Unsplash** (unsplash.com) — Search "glasses"
   - License: CC0 (public domain)
   - ~200+ relevant images
   - Download script:
     ```bash
     curl -s 'https://unsplash.com/napi/search/photos?query=glasses&per_page=50' \
       | jq -r '.results[].urls.regular' \
       | xargs -I {} wget {}
     ```

2. **Pexels** (pexels.com) — Search "sunglasses", "eyeglasses"
   - License: Free for commercial use
   - ~500+ relevant images
   - API available

3. **Pixabay** (pixabay.com)
   - License: Free for commercial use
   - Large eyewear collection

### Paid Sources (Budget-Friendly)
1. **Shutterstock** (~$0.50/image with subscription)
2. **Getty Images** (expensive, good for premium)
3. **Adobe Stock** (included with Creative Cloud)

**Time Investment:** 1-2 hours for 100 images  
**Quality:** ⭐⭐⭐ (good, some lifestyle shots)  
**Legality:** ✅ Clear (licensed)

---

## Option C: Synthetic Image Generation (Instant)

### Use Existing Script
Already provided: `generate_test_images.py`

```bash
python generate_test_images.py --count 100 --output dataset/images_synthetic
```

Generates:
- Metal frames: thin, rectangular
- Plastic frames: thicker, rounder
- Oversized frames: large elliptical
- Cat-eye frames: angled winged

**Pros:**
- Instant, unlimited quantity
- Perfect masks (automatic)
- Diverse angles/lighting

**Cons:**
- Unrealistic (not real product photos)
- Limited for real-world validation

**Use Case:** Training only, not for real VTO testing

**Time Investment:** 0 minutes (script does it)

---

## Option D: User-Generated + Your Own Photos

### DIY Approach
1. Take photos of your own glasses (or borrow from friends)
   - Different styles: metal, plastic, cat-eye
   - Varied lighting: indoor, outdoor, bright, dim
   - Varied angles: head-on, tilted, 3/4 view
   - Different faces/skin tones
   
2. Quick photoshoot: 30 minutes → 50 high-quality images
   - Use phone camera (good enough)
   - White/neutral background
   - Even lighting (avoid harsh shadows)

3. Get friends/colleagues to contribute
   - Crowd-source 50-100 images
   - Diverse styles naturally

**Time Investment:** 1-2 hours for 50 images  
**Quality:** ⭐⭐⭐⭐⭐ (if done well)  
**Legality:** ✅ Own photos

---

## Recommended Strategy (Fastest)

### Week 3: Collect 200 Images
```
Option A (E-commerce scraping):  80 images  [2 hours]
Option B (Stock photos):         80 images  [1 hour]
Option C (Synthetic):            30 images  [1 minute]
Option D (DIY/friends):          20 images  [1 hour]
─────────────────────────────────────────────
Total:                          210 images  [~4 hours work]
```

**Breakdown by Style:**
- Metal: 100 images (50%)
- Plastic: 60 images (30%)
- Acetate: 30 images (15%)
- Other (cat-eye, oversized): 10 images (5%)

---

## Annotation Workflow

### Tool: CVAT (Recommended)
**Free, web-based segmentation annotation**

1. **Install locally or use online:**
   ```bash
   # Local installation
   docker run -p 8080:8080 openvino/cvat-ui
   docker run -p 8090:8090 openvino/cvat-core
   ```

2. **Create project:**
   - Name: "Eyewear Segmentation"
   - Task type: "Instance Segmentation"
   - Labels: 1 label "eyewear"

3. **Upload images:**
   - dataset/images_raw/*.jpg → CVAT project

4. **Annotate:**
   - For each image: trace glasses outline (polygon or free-hand)
   - ~2-5 minutes per image
   - **200 images × 3 min = 600 minutes = 10 hours**

5. **Export:**
   - Format: COCO JSON or Mask PNG
   - Output: dataset/masks/*.png

### Alternative Tools
- **Labelbox** (commercial, $$ but fast UI)
- **RoboFlow** (online, pre-trained assistants)
- **Manual** with GIMP (free but tedious)

---

## Post-Collection: Organization

### Directory Structure
```
dataset/
├── images/
│   ├── warby_parker_001.jpg
│   ├── glasses_usA_002.jpg
│   ├── stock_unsplash_003.jpg
│   ├── my_photo_001.jpg
│   └── ...200+ total
│
├── masks/
│   ├── warby_parker_001.png
│   ├── glasses_usa_002.png
│   ├── stock_unsplash_003.png
│   ├── my_photo_001.png
│   └── ...matching count
│
└── metadata/
    └── manifest.json  (auto-generated by pipeline)
```

### Validation
```bash
# Verify images/masks match
python -c "
from pathlib import Path
images = set(p.stem for p in Path('dataset/images').glob('*.jpg'))
masks = set(p.stem for p in Path('dataset/masks').glob('*.png'))
missing_masks = images - masks
if missing_masks:
    print(f'Missing masks: {missing_masks}')
else:
    print(f'✓ All {len(images)} images have masks')
"
```

---

## Timeline

| Week | Task | Time | Output |
|------|------|------|--------|
| 3 | Collect images (scrape, stock, DIY) | 4 hours | 200+ JPG files |
| 3 | Annotate masks in CVAT | 10 hours | 200+ PNG masks |
| 3 | Register to pipeline | 1 hour | dataset/images/, dataset/masks/ |
| 3 | Create YOLO splits | 30 min | dataset/splits/{train,val,test}/manifest.json |
| **Week 4** | **Train YOLOv8-Seg** | **1 hour** | **runs/segment/eyewear_seg/weights/best.pt** |

---

## Quick Start: Scraping Template

```bash
# Install dependencies
pip install requests beautifulsoup4 lxml selenium

# Save this as scrape_warby.py
python scrape_warby.py

# Result: dataset/images_raw/warby_parker_*.jpg
```

---

## Budget Estimate

| Source | Cost | Images | Quality |
|--------|------|--------|---------|
| E-commerce scraping | $0 | 100+ | ⭐⭐⭐⭐ |
| Free stock photos | $0 | 100+ | ⭐⭐⭐ |
| Paid stock photos | $50 | 50+ | ⭐⭐⭐⭐ |
| DIY photography | $0 | 50+ | ⭐⭐⭐⭐⭐ |
| Synthetic generation | $0 | 100+ | ⭐⭐ |
| **Total (free path)** | **$0** | **200+** | **Good** |

---

## Next Step

1. Start with **Option A + B** (scrape + stock photos) — 4 hours → 160 images
2. Fill gap with **Option D** (DIY) — 1 hour → 40 images
3. Use **CVAT** to annotate masks — 10 hours
4. Run **pipeline to register** → ready for YOLOv8 training

**Total time investment:** ~16 hours  
**Result:** Production-ready training dataset

Go! 🚀

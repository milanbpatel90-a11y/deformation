#!/usr/bin/env python3
"""
Download eyewear images from Unsplash API
Requires: pip install requests pillow
"""

import os
import sys
import argparse
import requests
import time
from pathlib import Path
from typing import List, Dict

# Unsplash API Configuration
UNSPLASH_ACCESS_KEY = "YOUR_ACCESS_KEY_HERE"  # Get from https://unsplash.com/developers
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"

# Search queries for diverse eyewear images
SEARCH_QUERIES = [
    "eyeglasses",
    "sunglasses",
    "reading glasses",
    "prescription glasses",
    "fashion glasses",
    "eyewear",
    "spectacles",
    "optical glasses",
    "designer glasses",
    "vintage glasses"
]


def download_image(url: str, save_path: Path) -> bool:
    """Download image from URL and save to path"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {save_path.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {save_path.name}: {e}")
        return False


def search_unsplash(query: str, per_page: int = 30, page: int = 1) -> List[Dict]:
    """Search Unsplash for images"""
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }
    
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": "landscape"  # Better for eyewear detection
    }
    
    try:
        response = requests.get(UNSPLASH_API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"✗ API Error for query '{query}': {e}")
        return []


def download_unsplash_images(count: int, output_dir: Path) -> int:
    """Download specified number of eyewear images from Unsplash"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Starting download of {count} eyewear images from Unsplash...")
    print(f"📁 Output directory: {output_dir}")
    
    if UNSPLASH_ACCESS_KEY == "YOUR_ACCESS_KEY_HERE":
        print("\n⚠️  WARNING: No Unsplash API key configured!")
        print("Please get your free API key from: https://unsplash.com/developers")
        print("Then edit this script and replace YOUR_ACCESS_KEY_HERE with your key.\n")
        return 0
    
    downloaded = 0
    images_per_query = max(1, count // len(SEARCH_QUERIES))
    
    for query in SEARCH_QUERIES:
        if downloaded >= count:
            break
        
        print(f"\n🔎 Searching: '{query}'...")
        results = search_unsplash(query, per_page=images_per_query)
        
        if not results:
            print(f"  No results for '{query}'")
            continue
        
        print(f"  Found {len(results)} images")
        
        for idx, photo in enumerate(results):
            if downloaded >= count:
                break
            
            # Get regular size image URL (good balance of quality/size)
            image_url = photo["urls"]["regular"]
            photo_id = photo["id"]
            
            # Save with descriptive filename
            filename = f"unsplash_{query.replace(' ', '_')}_{photo_id}.jpg"
            save_path = output_dir / filename
            
            # Skip if already exists
            if save_path.exists():
                print(f"  ⊙ Skipped (exists): {filename}")
                continue
            
            # Download with rate limiting
            if download_image(image_url, save_path):
                downloaded += 1
                print(f"  Progress: {downloaded}/{count}")
            
            # Rate limiting (Unsplash allows 50 requests/hour for free tier)
            time.sleep(0.5)
    
    print(f"\n✅ Download complete! {downloaded} images saved to {output_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download eyewear images from Unsplash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_unsplash_glasses.py --count 50 --output dataset/images_raw
  python download_unsplash_glasses.py --count 100 --output data/raw_images

Note: You need a free Unsplash API key from https://unsplash.com/developers
      Edit this script and replace YOUR_ACCESS_KEY_HERE with your key.
        """
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images to download (default: 50)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/images_raw",
        help="Output directory for downloaded images (default: dataset/images_raw)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("🕶️  UNSPLASH EYEWEAR IMAGE DOWNLOADER")
    print("=" * 60)
    
    downloaded = download_unsplash_images(args.count, output_dir)
    
    if downloaded > 0:
        print(f"\n📊 Summary:")
        print(f"   Total downloaded: {downloaded}")
        print(f"   Location: {output_dir.absolute()}")
        print(f"\n💡 Next steps:")
        print(f"   1. Add 10 DIY photos to {output_dir}")
        print(f"   2. Set up CVAT for annotation")
        print(f"   3. Upload all images to CVAT")
        print(f"   4. Start annotating!")
    else:
        print("\n❌ No images downloaded. Please check:")
        print("   1. Your Unsplash API key is configured")
        print("   2. You have internet connection")
        print("   3. Unsplash API is accessible")
    
    return 0 if downloaded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob

#!/usr/bin/env python3
"""
Convert COCO Dataset to PerSAM Folder Structure

PerSAM expects:
data/
  Images/
    obj_name/
      00.jpg
      01.jpg
  Annotations/
    obj_name/
      00.png

This script converts our filtered COCO JSON into this format.
"""

import os
import json
import shutil
import cv2
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path

# Configuration - Using Absolute Paths to avoid CWD ambiguity
# Assuming script is in <root>/scripts/
ROOT_DIR = Path(__file__).parent.parent.absolute()
COCO_JSON = ROOT_DIR / "data/datasets/PerSam-F 26/instances_filtered.json"
IMAGES_DIR = ROOT_DIR / "data/datasets/PerSam-F 26/images"
OUTPUT_DIR = ROOT_DIR / "data/datasets/PerSam-F_26_Formatted"

def convert():
    print(f"[Convert] Starting conversion...")
    print(f"[Convert] Root Dir: {ROOT_DIR}")
    print(f"[Convert] JSON Path: {COCO_JSON}")
    
    if not COCO_JSON.exists():
        print(f"[Convert] Error: {COCO_JSON} not found!")
        return

    print(f"[Convert] Loading {COCO_JSON}...")
    try:
        with open(COCO_JSON, "r") as f:
            coco = json.load(f)
    except Exception as e:
        print(f"[Convert] Error loading JSON: {e}")
        return

    # Dictionary to map image_id to file_name
    img_map = {img['id']: img['file_name'] for img in coco['images']}
    
    # We treat "stone" as the single object category
    obj_name = "stone"
    
    out_img_dir = OUTPUT_DIR / "Images" / obj_name
    out_mask_dir = OUTPUT_DIR / "Annotations" / obj_name
    
    print(f"[Convert] Creating directories: {out_img_dir}")
    if OUTPUT_DIR.exists():
        print(f"[Convert] Cleaning up old output: {OUTPUT_DIR}")
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    print("[Convert] Processing annotations...")
    
    count = 0
    
    for i, ann in enumerate(tqdm(coco['annotations'])):
        img_id = ann['image_id']
        file_name = img_map.get(img_id)
        
        if not file_name:
            print(f"[Convert] No filename for img_id {img_id}")
            continue
            
        src_img_path = IMAGES_DIR / os.path.basename(file_name)
        if not src_img_path.exists():
            print(f"[Convert] Warning: Image {src_img_path} not found. Skipping.")
            continue
            
        # Read Image
        image = cv2.imread(str(src_img_path))
        if image is None:
            print(f"[Convert] Error: Failed to read image using cv2: {src_img_path}")
            continue
            
        # Create Mask
        h, w = image.shape[:2]
        gray_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Handle segmentation (Polygon or RLE)
        seg = ann['segmentation']
        if isinstance(seg, list):
            # Polygon
            for poly_list in seg:
                poly = np.array(poly_list).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(gray_mask, [poly], 255)
        elif isinstance(seg, dict):
             print(f"[Convert] Skipping RLE for now")
             pass
            
        # Validate Mask
        mask_area = np.count_nonzero(gray_mask)
        if mask_area < 100: # Threshold for tiny objects that might vanish after resizing
             print(f"[Convert] Warning: Small/Empty mask ({mask_area} px) for annotation {i} (Img: {file_name}). Skipping.")
             continue

        # Write files
        idx_str = f"{count:02}"
        dst_img_path = out_img_dir / f"{idx_str}.jpg"
        dst_mask_path = out_mask_dir / f"{idx_str}.png"
        
        cv2.imwrite(str(dst_img_path), image)
        cv2.imwrite(str(dst_mask_path), gray_mask)
        
        count += 1
        
    print(f"[Convert] Done! Converted {count} items to {OUTPUT_DIR}")

if __name__ == "__main__":
    convert()

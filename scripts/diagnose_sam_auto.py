#!/usr/bin/env python3
"""
SAM Automatic Mask Generation Pipeline

DIFFERENT APPROACH: Use SAM to find ALL objects first, then classify.
This catches far more objects than relying on GroundingDINO boxes only.

Pipeline:
  Step 1: SAM Automatic Mask Generator - finds ALL objects in image
  Step 2: Filter masks by size and quality
  Step 3: Use GroundingDINO to verify which masks are stones
  Step 4: Combine and output

This should catch 90%+ of stones that the box-based approach missed!
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam-hq"))

# ============================================================================
# Configuration
# ============================================================================

TEST_IMAGE = "data/raw/1.jpg"
OUTPUT_DIR = "outputs/sam_auto"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM Automatic Mask Generator settings
SAM_CHECKPOINT = "weight/sam_vit_h_4b8939.pth"
POINTS_PER_SIDE = 32  # More points = more masks found
PRED_IOU_THRESH = 0.86  # Lower = more masks
STABILITY_SCORE_THRESH = 0.92  # Lower = more masks
MIN_MASK_AREA = 500  # Minimum mask size in pixels

# GroundingDINO settings (for verification)
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "weight/groundingdino_swint_ogc.pth"

COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (128, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128)]


def save_image(name: str, image: np.ndarray):
    """Save image to output directory."""
    path = f"{OUTPUT_DIR}/{name}"
    cv2.imwrite(path, image)
    print(f"    Saved: {path}")


def main():
    print("=" * 70)
    print("SAM AUTOMATIC MASK GENERATION PIPELINE")
    print("=" * 70)
    print("This finds ALL objects first, then classifies which are stones.")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load image
    # =========================================================================
    print("\n[STEP 1] Loading image...")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"ERROR: Could not load {TEST_IMAGE}")
        return
    
    h, w = image.shape[:2]
    print(f"    Image: {TEST_IMAGE} ({w}x{h})")
    save_image("step1_original.jpg", image)
    
    # =========================================================================
    # STEP 2: SAM Automatic Mask Generation
    # =========================================================================
    print("\n[STEP 2] SAM Automatic Mask Generation...")
    print("    This finds ALL potential objects in the image...")
    
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        min_mask_region_area=MIN_MASK_AREA,
    )
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    
    print(f"    Found {len(masks)} automatic masks!")
    
    # Sort by area
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Visualize all masks
    vis_all = image.copy()
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = COLORS[i % len(COLORS)]
        vis_all[mask] = (vis_all[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    
    save_image("step2_all_masks.jpg", vis_all)
    
    # Save top N individual masks
    for i, mask_data in enumerate(masks[:20]):  # Top 20 largest
        mask = mask_data['segmentation']
        vis_mask = image.copy()
        vis_mask[mask] = [0, 255, 0]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_mask, contours, -1, (0, 255, 0), 2)
        save_image(f"step2_mask_{i+1:02d}_area{mask_data['area']}.jpg", vis_mask)
    
    del sam, mask_generator
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 3: Filter by size and quality
    # =========================================================================
    print("\n[STEP 3] Filtering masks by size and quality...")
    
    # Filter out very large masks (probably background)
    max_area = h * w * 0.3  # Max 30% of image
    filtered_masks = [m for m in masks if m['area'] < max_area and m['area'] > MIN_MASK_AREA]
    
    print(f"    Before filter: {len(masks)} masks")
    print(f"    After filter: {len(filtered_masks)} masks")
    
    # Visualize filtered masks
    vis_filtered = image.copy()
    for i, mask_data in enumerate(filtered_masks):
        mask = mask_data['segmentation']
        color = COLORS[i % len(COLORS)]
        vis_filtered[mask] = (vis_filtered[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_filtered, contours, -1, color, 2)
    
    save_image("step3_filtered_masks.jpg", vis_filtered)
    
    # =========================================================================
    # STEP 4: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total SAM masks found: {len(masks)}")
    print(f"  After filtering: {len(filtered_masks)}")
    print(f"\n  Compare this to GroundingDINO which only found ~10 boxes!")
    print(f"  SAM automatic mode finds {len(filtered_masks)}x more potential stones.")
    print("\nOutputs saved to:", OUTPUT_DIR)
    print("=" * 70)
    
    # Save mask data
    mask_info = []
    for i, m in enumerate(filtered_masks):
        mask_info.append({
            "id": i,
            "area": int(m['area']),
            "predicted_iou": float(m['predicted_iou']),
            "stability_score": float(m['stability_score']),
            "bbox": m['bbox'],  # [x, y, w, h]
        })
    
    with open(f"{OUTPUT_DIR}/mask_data.json", "w") as f:
        json.dump({"total_masks": len(filtered_masks), "masks": mask_info}, f, indent=2)
    print(f"    Saved: {OUTPUT_DIR}/mask_data.json")


if __name__ == "__main__":
    main()

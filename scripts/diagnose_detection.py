#!/usr/bin/env python3
"""
Single Image Detection Diagnostic Tool

Tests detection on ONE image to diagnose and improve accuracy.
Shows detailed results at each pipeline step.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam-hq"))

# ============================================================================
# Configuration - ADJUST THESE FOR TESTING
# ============================================================================

# Test image
TEST_IMAGE = "data/raw/1.jpg"

# Output directory for diagnostic results
OUTPUT_DIR = "outputs/diagnostic"

# Prompts to test - ADD/REMOVE to improve detection
PROMPTS = [
    "stone",
    "rock",
    "pebble",
]

# Detection thresholds - Raised to reduce false positives
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.30

# Maximum box size (as ratio of image) - filters out full-image detections
MAX_BOX_RATIO = 0.4  # Box can't be more than 40% of image width/height

# Minimum box size (as ratio of image) - filters out tiny noise
MIN_BOX_RATIO = 0.02  # Box must be at least 2% of image

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "weight/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weight/sam_vit_h_4b8939.pth"

# ============================================================================
# Main Diagnostic
# ============================================================================

def main():
    print("=" * 60)
    print("SINGLE IMAGE DETECTION DIAGNOSTIC")
    print("=" * 60)
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Prompts: {PROMPTS}")
    print(f"Box Threshold: {BOX_THRESHOLD}")
    print(f"Text Threshold: {TEXT_THRESHOLD}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load image
    print("\n[1/5] Loading image...")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"ERROR: Could not load image {TEST_IMAGE}")
        return
    
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")
    
    # Load GroundingDINO
    print("\n[2/5] Loading GroundingDINO...")
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    
    grounding_dino = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
    grounding_dino = grounding_dino.float().to(DEVICE)
    print("  GroundingDINO loaded successfully")
    
    # Load SAM
    print("\n[3/5] Loading SAM...")
    from segment_anything import sam_model_registry, SamPredictor
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    print("  SAM loaded successfully")
    
    # Transform image for GroundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pil, None)
    
    # Run detection with each prompt
    print("\n[4/5] Running detection with each prompt...")
    all_boxes = []
    all_scores = []
    all_phrases = []
    
    for prompt in PROMPTS:
        print(f"\n  Prompt: '{prompt}'")
        
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=grounding_dino,
                image=image_transformed,
                caption=prompt + ".",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )
        
        print(f"    Detected: {len(boxes)} boxes")
        for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
            print(f"      [{i+1}] {phrase}: {score:.3f} - box: {box.numpy()}")
        
        if len(boxes) > 0:
            all_boxes.append(boxes.cpu().numpy())
            all_scores.append(logits.cpu().numpy())
            all_phrases.extend(phrases)
    
    if not all_boxes:
        print("\n  NO DETECTIONS! Try:")
        print("    - Lower BOX_THRESHOLD (e.g., 0.15)")
        print("    - Different prompts")
        return
    
    # Combine and NMS
    combined_boxes = np.vstack(all_boxes)
    combined_scores = np.concatenate(all_scores)
    
    print(f"\n  Total detections before NMS: {len(combined_boxes)}")
    
    # Convert to xyxy for NMS
    boxes_xyxy = np.zeros((len(combined_boxes), 4))
    for i, box in enumerate(combined_boxes):
        cx, cy, bw, bh = box
        boxes_xyxy[i] = [
            (cx - bw/2) * w,
            (cy - bh/2) * h,
            (cx + bw/2) * w,
            (cy + bh/2) * h
        ]
    
    keep_indices = torchvision.ops.nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(combined_scores, dtype=torch.float32),
        0.5
    ).numpy()
    
    final_boxes = combined_boxes[keep_indices]
    final_scores = combined_scores[keep_indices]
    final_boxes_xyxy = boxes_xyxy[keep_indices]
    
    print(f"  After NMS: {len(final_boxes)} boxes")
    
    # Filter by box size - remove full-image and tiny detections
    size_mask = []
    for box in final_boxes:
        cx, cy, bw, bh = box
        # Check if box is within acceptable size range
        if bw <= MAX_BOX_RATIO and bh <= MAX_BOX_RATIO and bw >= MIN_BOX_RATIO and bh >= MIN_BOX_RATIO:
            size_mask.append(True)
        else:
            print(f"    FILTERED: box size {bw:.2f}x{bh:.2f} outside range [{MIN_BOX_RATIO}, {MAX_BOX_RATIO}]")
            size_mask.append(False)
    
    size_mask = np.array(size_mask)
    final_boxes = final_boxes[size_mask]
    final_scores = final_scores[size_mask]
    final_boxes_xyxy = final_boxes_xyxy[size_mask]
    
    print(f"  After size filter: {len(final_boxes)} boxes")
    
    # SAM segmentation
    print("\n[5/5] Running SAM segmentation...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    masks = []
    for i, box_xyxy in enumerate(final_boxes_xyxy):
        mask_predictions, scores, _ = sam_predictor.predict(
            box=box_xyxy,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        masks.append(mask_predictions[best_idx])
        print(f"  Mask {i+1}: area={mask_predictions[best_idx].sum()} pixels, score={scores[best_idx]:.3f}")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Detection boxes only
    vis_boxes = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (box_xyxy, score) in enumerate(zip(final_boxes_xyxy, final_scores)):
        x1, y1, x2, y2 = map(int, box_xyxy)
        color = colors[i % len(colors)]
        cv2.rectangle(vis_boxes, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_boxes, f"stone {score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    boxes_path = f"{OUTPUT_DIR}/1_detection_boxes.jpg"
    cv2.imwrite(boxes_path, vis_boxes)
    print(f"  Saved: {boxes_path}")
    
    # 2. Masks overlay
    vis_masks = image.copy()
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        overlay[mask] = color
    vis_masks = cv2.addWeighted(overlay, 0.4, vis_masks, 0.6, 0)
    
    # Add contours
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(vis_masks, contours, -1, color, 2)
    
    masks_path = f"{OUTPUT_DIR}/2_segmentation_masks.jpg"
    cv2.imwrite(masks_path, vis_masks)
    print(f"  Saved: {masks_path}")
    
    # 3. Combined view
    vis_combined = image.copy()
    for i, (mask, box_xyxy, score) in enumerate(zip(masks, final_boxes_xyxy, final_scores)):
        color = colors[i % len(colors)]
        # Mask
        vis_combined[mask] = (vis_combined[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        # Contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_combined, contours, -1, color, 2)
        # Box + label
        x1, y1, x2, y2 = map(int, box_xyxy)
        cv2.rectangle(vis_combined, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis_combined, f"{i+1}: {score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    combined_path = f"{OUTPUT_DIR}/3_combined_result.jpg"
    cv2.imwrite(combined_path, vis_combined)
    print(f"  Saved: {combined_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Image: {TEST_IMAGE}")
    print(f"  Prompts used: {PROMPTS}")
    print(f"  Thresholds: box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"  Total detections: {len(masks)}")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("  - 1_detection_boxes.jpg  (GroundingDINO boxes)")
    print("  - 2_segmentation_masks.jpg  (SAM masks)")
    print("  - 3_combined_result.jpg  (Final result)")
    print("\n" + "=" * 60)
    print("TO IMPROVE DETECTION:")
    print("  - Edit PROMPTS list to add domain-specific terms")
    print("  - Lower BOX_THRESHOLD for more detections")
    print("  - Raise BOX_THRESHOLD for fewer false positives")
    print("=" * 60)


if __name__ == "__main__":
    main()

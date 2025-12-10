#!/usr/bin/env python3
"""
Step-by-Step Diagnostic Pipeline

Saves output from EACH step separately so you can diagnose
which step needs improvement:

  Step 1: Original image
  Step 2: RAM tags (text output)
  Step 3: GroundingDINO boxes per prompt
  Step 4: NMS merged boxes
  Step 5: Size-filtered boxes
  Step 6: SAM masks per box
  Step 7: Final combined result

All outputs saved to outputs/step_by_step/ for review.
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

import torch
import torchvision

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam-hq"))
sys.path.insert(0, str(Path(__file__).parent.parent / "recognize-anything"))

# ============================================================================
# Configuration
# ============================================================================

TEST_IMAGE = "data/raw/1.jpg"
OUTPUT_DIR = "outputs/step_by_step"

# Base prompts
PROMPTS = ["stone", "rock", "pebble"]

# Thresholds
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MAX_BOX_RATIO = 0.4
MIN_BOX_RATIO = 0.02

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "weight/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weight/sam_vit_h_4b8939.pth"
RAM_CHECKPOINT = "weight/ram_swin_large_14m.pth"

# Colors for visualization
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def save_step(name: str, image: np.ndarray = None, text: str = None, data: dict = None):
    """Save step output with proper naming."""
    base = f"{OUTPUT_DIR}/{name}"
    
    if image is not None:
        cv2.imwrite(f"{base}.jpg", image)
        print(f"    Saved: {base}.jpg")
    
    if text is not None:
        with open(f"{base}.txt", "w") as f:
            f.write(text)
        print(f"    Saved: {base}.txt")
    
    if data is not None:
        with open(f"{base}.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"    Saved: {base}.json")


def draw_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray = None, labels: list = None):
    """Draw boxes on image."""
    vis = image.copy()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        label = f"#{i+1}"
        if scores is not None and i < len(scores):
            label += f" {scores[i]:.2f}"
        if labels is not None and i < len(labels):
            label = f"{labels[i]} {label}"
        
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def main():
    print("=" * 70)
    print("STEP-BY-STEP DIAGNOSTIC PIPELINE")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Original Image
    # =========================================================================
    print("\n[STEP 1] Loading original image...")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"ERROR: Could not load {TEST_IMAGE}")
        return
    
    h, w = image.shape[:2]
    print(f"    Image: {TEST_IMAGE} ({w}x{h})")
    save_step("step1_original", image)
    
    # =========================================================================
    # STEP 2: RAM Auto-Tagging
    # =========================================================================
    print("\n[STEP 2] RAM Auto-Tagging...")
    ram_tags = []
    stone_tags = []
    
    try:
        from ram.models import ram
        from ram import inference_ram
        import torchvision.transforms as T
        
        model = ram(pretrained=RAM_CHECKPOINT, image_size=384, vit='swin_l')
        model.eval().to(DEVICE)
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize((384, 384)), T.ToTensor(), normalize])
        
        image_pil = Image.open(TEST_IMAGE).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            tags, _ = inference_ram(image_tensor, model)
        
        ram_tags = tags.split(" | ") if tags else []
        stone_keywords = ["stone", "rock", "pebble", "mineral", "rocky", "stony", "rubble"]
        stone_tags = [t for t in ram_tags if any(kw in t.lower() for kw in stone_keywords)]
        
        del model
        torch.cuda.empty_cache()
        
        print(f"    All tags: {ram_tags}")
        print(f"    Stone tags: {stone_tags}")
        
    except Exception as e:
        print(f"    WARNING: RAM failed: {e}")
    
    save_step("step2_ram_tags", text=f"All tags: {ram_tags}\nStone tags: {stone_tags}")
    
    # =========================================================================
    # STEP 3: GroundingDINO Detection (per prompt)
    # =========================================================================
    print("\n[STEP 3] GroundingDINO Detection (per prompt)...")
    
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as GT
    
    grounding_dino = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
    grounding_dino = grounding_dino.float().to(DEVICE)
    
    transform = GT.Compose([
        GT.RandomResize([800], max_size=1333),
        GT.ToTensor(),
        GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pil, None)
    
    all_prompts = PROMPTS + [t for t in stone_tags if t not in PROMPTS]
    all_boxes = []
    all_scores = []
    all_phrases = []
    
    for prompt in all_prompts:
        print(f"\n    Prompt: '{prompt}'")
        
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=grounding_dino,
                image=image_transformed,
                caption=prompt + ".",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )
        
        boxes_np = boxes.cpu().numpy()
        scores_np = logits.cpu().numpy()
        
        print(f"      Found {len(boxes)} boxes")
        
        # Convert to xyxy for visualization
        boxes_xyxy = []
        for box in boxes_np:
            cx, cy, bw, bh = box
            boxes_xyxy.append([
                (cx - bw/2) * w,
                (cy - bh/2) * h,
                (cx + bw/2) * w,
                (cy + bh/2) * h
            ])
        
        if len(boxes_xyxy) > 0:
            vis = draw_boxes(image, np.array(boxes_xyxy), scores_np, phrases)
            save_step(f"step3_{prompt}_boxes", vis)
            
            all_boxes.extend(boxes_np)
            all_scores.extend(scores_np)
            all_phrases.extend(phrases)
    
    del grounding_dino
    torch.cuda.empty_cache()
    
    if not all_boxes:
        print("\n    NO DETECTIONS! Lower threshold or try different prompts.")
        return
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    
    print(f"\n    Total raw boxes: {len(all_boxes)}")
    save_step("step3_all_raw", data={
        "total_boxes": len(all_boxes),
        "prompts_used": all_prompts,
        "boxes": all_boxes.tolist(),
        "scores": all_scores.tolist()
    })
    
    # =========================================================================
    # STEP 4: NMS Merge
    # =========================================================================
    print("\n[STEP 4] NMS Merge...")
    
    # Convert all boxes to xyxy
    boxes_xyxy = np.zeros((len(all_boxes), 4))
    for i, box in enumerate(all_boxes):
        cx, cy, bw, bh = box
        boxes_xyxy[i] = [
            (cx - bw/2) * w,
            (cy - bh/2) * h,
            (cx + bw/2) * w,
            (cy + bh/2) * h
        ]
    
    keep_indices = torchvision.ops.nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(all_scores, dtype=torch.float32),
        0.5
    ).numpy()
    
    nms_boxes = all_boxes[keep_indices]
    nms_scores = all_scores[keep_indices]
    nms_boxes_xyxy = boxes_xyxy[keep_indices]
    
    print(f"    Before NMS: {len(all_boxes)} -> After NMS: {len(nms_boxes)}")
    
    vis = draw_boxes(image, nms_boxes_xyxy, nms_scores)
    save_step("step4_after_nms", vis)
    
    # =========================================================================
    # STEP 5: Size Filter
    # =========================================================================
    print("\n[STEP 5] Size Filter...")
    
    size_mask = []
    for i, box in enumerate(nms_boxes):
        cx, cy, bw, bh = box
        keep = bw <= MAX_BOX_RATIO and bh <= MAX_BOX_RATIO and bw >= MIN_BOX_RATIO and bh >= MIN_BOX_RATIO
        if not keep:
            print(f"    FILTERED #{i+1}: size {bw:.2f}x{bh:.2f}")
        size_mask.append(keep)
    
    size_mask = np.array(size_mask)
    final_boxes = nms_boxes[size_mask]
    final_scores = nms_scores[size_mask]
    final_boxes_xyxy = nms_boxes_xyxy[size_mask]
    
    print(f"    Before filter: {len(nms_boxes)} -> After filter: {len(final_boxes)}")
    
    vis = draw_boxes(image, final_boxes_xyxy, final_scores)
    save_step("step5_after_filter", vis)
    
    # =========================================================================
    # STEP 6: SAM Segmentation (per box)
    # =========================================================================
    print("\n[STEP 6] SAM Segmentation (per box)...")
    
    from segment_anything import sam_model_registry, SamPredictor
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    masks = []
    for i, box_xyxy in enumerate(final_boxes_xyxy):
        mask_predictions, scores, _ = sam_predictor.predict(
            box=box_xyxy,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        mask = mask_predictions[best_idx]
        masks.append(mask)
        
        # Save individual mask
        vis_mask = image.copy()
        vis_mask[mask] = [0, 255, 0]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_mask, contours, -1, (0, 255, 0), 2)
        save_step(f"step6_mask_{i+1}", vis_mask)
        
        print(f"    Mask #{i+1}: area={mask.sum():,} pixels, score={scores[best_idx]:.3f}")
    
    del sam, sam_predictor
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 7: Final Combined Result
    # =========================================================================
    print("\n[STEP 7] Final Combined Result...")
    
    vis = image.copy()
    for i, (mask, box_xyxy, score) in enumerate(zip(masks, final_boxes_xyxy, final_scores)):
        color = COLORS[i % len(COLORS)]
        vis[mask] = (vis[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        x1, y1, x2, y2 = map(int, box_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"#{i+1} {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    save_step("step7_final_result", vis)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Step 1: Original image ({w}x{h})")
    print(f"  Step 2: RAM tags - {len(stone_tags)} stone-related tags")
    print(f"  Step 3: GroundingDINO - {len(all_boxes)} raw boxes from {len(all_prompts)} prompts")
    print(f"  Step 4: NMS - {len(nms_boxes)} merged boxes")
    print(f"  Step 5: Size filter - {len(final_boxes)} boxes")
    print(f"  Step 6: SAM - {len(masks)} masks")
    print(f"  Step 7: Final result")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nReview each step to identify which one needs improvement!")
    print("=" * 70)


if __name__ == "__main__":
    main()

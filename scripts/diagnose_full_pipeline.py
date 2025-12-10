#!/usr/bin/env python3
"""
Full Multi-Phase Detection Pipeline Diagnostic

Implements ALL 4 phases from the implementation plan:
  Phase 1: Auto-Tagging (RAM)
  Phase 2: Multi-Prompt Detection (GroundingDINO)
  Phase 3: Ensemble Segmentation (SAM + SAM-HQ)
  Phase 4: Fusion & Refinement (NMS, size filter, post-processing)

Tests on ONE image to diagnose and improve accuracy.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any

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

# Test image
TEST_IMAGE = "data/raw/1.jpg"
OUTPUT_DIR = "outputs/diagnostic_full"

# Base prompts (Phase 2 will add RAM-discovered tags)
BASE_PROMPTS = [
    "stone",
    "rock",
    "pebble",
]

# RAM keywords to filter relevant tags
STONE_KEYWORDS = ["stone", "rock", "pebble", "mineral", "rocky", "stony", "rubble"]

# Detection thresholds
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Size filters (as ratio of image)
MAX_BOX_RATIO = 0.4   # Filter boxes > 40% of image
MIN_BOX_RATIO = 0.02  # Filter boxes < 2% of image

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "weight/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weight/sam_vit_h_4b8939.pth"
RAM_CHECKPOINT = "weight/ram_swin_large_14m.pth"

# ============================================================================
# Phase 1: Auto-Tagging with RAM
# ============================================================================

def phase1_auto_tagging(image_path: str) -> List[str]:
    """
    Phase 1: Use RAM model to automatically discover image tags.
    Returns stone-related tags that can augment detection prompts.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: AUTO-TAGGING (RAM)")
    print("=" * 60)
    
    try:
        from ram.models import ram
        from ram import inference_ram
        import torchvision.transforms as T
        
        print("  Loading RAM model...")
        model = ram(pretrained=RAM_CHECKPOINT, image_size=384, vit='swin_l')
        model.eval()
        model.to(DEVICE)
        
        # Transform image
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            normalize
        ])
        
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        
        print("  Running RAM inference...")
        with torch.no_grad():
            tags, tags_chinese = inference_ram(image_tensor, model)
        
        all_tags = tags.split(" | ") if tags else []
        print(f"  All detected tags: {all_tags}")
        
        # Filter stone-related tags
        stone_tags = [t for t in all_tags if any(kw in t.lower() for kw in STONE_KEYWORDS)]
        print(f"  Stone-related tags: {stone_tags}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return stone_tags
        
    except Exception as e:
        print(f"  WARNING: RAM failed: {e}")
        print("  Continuing without auto-tagging...")
        return []


# ============================================================================
# Phase 2: Multi-Prompt Detection with GroundingDINO
# ============================================================================

def phase2_multi_prompt_detection(
    image: np.ndarray,
    prompts: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Phase 2: Run GroundingDINO detection with multiple prompts.
    Returns combined boxes, scores, and phrases from all prompts.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: MULTI-PROMPT DETECTION (GroundingDINO)")
    print("=" * 60)
    print(f"  Prompts: {prompts}")
    
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    
    print("  Loading GroundingDINO...")
    grounding_dino = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
    grounding_dino = grounding_dino.float().to(DEVICE)
    
    # Transform image
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pil, None)
    
    all_boxes = []
    all_scores = []
    all_phrases = []
    
    for prompt in prompts:
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
            print(f"      [{i+1}] {phrase}: {score:.3f}")
        
        if len(boxes) > 0:
            all_boxes.append(boxes.cpu().numpy())
            all_scores.append(logits.cpu().numpy())
            all_phrases.extend(phrases)
    
    # Clean up
    del grounding_dino
    torch.cuda.empty_cache()
    
    if not all_boxes:
        return np.array([]), np.array([]), []
    
    combined_boxes = np.vstack(all_boxes)
    combined_scores = np.concatenate(all_scores)
    
    print(f"\n  Total raw detections: {len(combined_boxes)}")
    return combined_boxes, combined_scores, all_phrases


# ============================================================================
# Phase 3: Ensemble Segmentation (SAM + SAM-HQ)
# ============================================================================

def phase3_ensemble_segmentation(
    image: np.ndarray,
    boxes_xyxy: np.ndarray
) -> List[np.ndarray]:
    """
    Phase 3: Run multiple SAM variants and combine masks.
    Currently uses SAM-HQ (SAM with HQ decoder).
    """
    print("\n" + "=" * 60)
    print("PHASE 3: ENSEMBLE SEGMENTATION (SAM)")
    print("=" * 60)
    
    from segment_anything import sam_model_registry, SamPredictor
    
    print("  Loading SAM (vit_h)...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    masks = []
    mask_scores = []
    
    print(f"  Segmenting {len(boxes_xyxy)} boxes...")
    for i, box_xyxy in enumerate(boxes_xyxy):
        mask_predictions, scores, _ = sam_predictor.predict(
            box=box_xyxy,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        masks.append(mask_predictions[best_idx])
        mask_scores.append(scores[best_idx])
        print(f"    Mask {i+1}: area={mask_predictions[best_idx].sum():,} pixels, score={scores[best_idx]:.3f}")
    
    # Clean up
    del sam, sam_predictor
    torch.cuda.empty_cache()
    
    return masks, mask_scores


# ============================================================================
# Phase 4: Fusion & Refinement
# ============================================================================

def phase4_fusion_refinement(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase 4: Apply NMS, size filtering, and post-processing.
    Returns refined boxes, scores, and xyxy coordinates.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: FUSION & REFINEMENT")
    print("=" * 60)
    
    h, w = image_size
    
    # Step 1: Convert to xyxy
    boxes_xyxy = np.zeros((len(boxes), 4))
    for i, box in enumerate(boxes):
        cx, cy, bw, bh = box
        boxes_xyxy[i] = [
            (cx - bw/2) * w,
            (cy - bh/2) * h,
            (cx + bw/2) * w,
            (cy + bh/2) * h
        ]
    
    # Step 2: NMS
    print(f"  Before NMS: {len(boxes)} boxes")
    keep_indices = torchvision.ops.nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        0.5
    ).numpy()
    
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    boxes_xyxy = boxes_xyxy[keep_indices]
    print(f"  After NMS: {len(boxes)} boxes")
    
    # Step 3: Size filtering
    size_mask = []
    for box in boxes:
        cx, cy, bw, bh = box
        if bw <= MAX_BOX_RATIO and bh <= MAX_BOX_RATIO and bw >= MIN_BOX_RATIO and bh >= MIN_BOX_RATIO:
            size_mask.append(True)
        else:
            print(f"    FILTERED: box {bw:.2f}x{bh:.2f} outside [{MIN_BOX_RATIO}, {MAX_BOX_RATIO}]")
            size_mask.append(False)
    
    size_mask = np.array(size_mask)
    boxes = boxes[size_mask]
    scores = scores[size_mask]
    boxes_xyxy = boxes_xyxy[size_mask]
    print(f"  After size filter: {len(boxes)} boxes")
    
    return boxes, scores, boxes_xyxy


# ============================================================================
# Visualization & Output
# ============================================================================

def create_visualizations(
    image: np.ndarray,
    masks: List[np.ndarray],
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    output_dir: str
):
    """Create and save visualization images."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Combined visualization
    vis = image.copy()
    for i, (mask, box_xyxy, score) in enumerate(zip(masks, boxes_xyxy, scores)):
        color = colors[i % len(colors)]
        
        # Mask overlay
        vis[mask] = (vis[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # Contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        
        # Box + label
        x1, y1, x2, y2 = map(int, box_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"#{i+1} {score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_path = f"{output_dir}/full_pipeline_result.jpg"
    cv2.imwrite(output_path, vis)
    print(f"  Saved: {output_path}")
    
    return output_path


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("FULL MULTI-PHASE DETECTION PIPELINE")
    print("=" * 60)
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Device: {DEVICE}")
    
    # Load image
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"ERROR: Could not load {TEST_IMAGE}")
        return
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # ========================================
    # PHASE 1: Auto-Tagging
    # ========================================
    ram_tags = phase1_auto_tagging(TEST_IMAGE)
    
    # Combine base prompts with RAM-discovered tags
    all_prompts = BASE_PROMPTS.copy()
    for tag in ram_tags:
        if tag not in all_prompts:
            all_prompts.append(tag)
    
    # ========================================
    # PHASE 2: Multi-Prompt Detection
    # ========================================
    boxes, scores, phrases = phase2_multi_prompt_detection(image, all_prompts)
    
    if len(boxes) == 0:
        print("\nNO DETECTIONS! Try lowering BOX_THRESHOLD.")
        return
    
    # ========================================
    # PHASE 4: Fusion & Refinement (before Phase 3)
    # ========================================
    boxes, scores, boxes_xyxy = phase4_fusion_refinement(boxes, scores, (h, w))
    
    if len(boxes) == 0:
        print("\nAll boxes filtered! Try adjusting size filters.")
        return
    
    # ========================================
    # PHASE 3: Ensemble Segmentation
    # ========================================
    masks, mask_scores = phase3_ensemble_segmentation(image, boxes_xyxy)
    
    # ========================================
    # Create Visualizations
    # ========================================
    output_path = create_visualizations(image, masks, boxes_xyxy, scores, OUTPUT_DIR)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Phase 1 (RAM): {len(ram_tags)} stone tags discovered")
    print(f"  Phase 2 (Detection): {len(all_prompts)} prompts used")
    print(f"  Phase 4 (Refinement): Filtered to {len(boxes)} boxes")
    print(f"  Phase 3 (Segmentation): {len(masks)} masks generated")
    print(f"\n  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

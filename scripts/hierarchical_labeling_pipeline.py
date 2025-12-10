#!/usr/bin/env python3
"""
Hierarchical Multi-Level Auto-Labeling Pipeline

This pipeline detects stones at multiple size levels for maximum recall:
  - Level 1: Large objects (>10% of image area)
  - Level 2: Medium objects (2-10% of image area)
  - Level 3: Small objects (0.5-2% of image area)

Uses SAM Automatic Mask Generator to find ALL objects first,
then filters and validates by hierarchical size levels.

Output: COCO JSON annotations + YOLO format + per-level visualizations
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datetime import datetime

import torch
import torchvision

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam-hq"))
sys.path.insert(0, str(Path(__file__).parent.parent / "recognize-anything"))

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LevelConfig:
    """Configuration for each hierarchical level."""
    name: str
    min_area_ratio: float  # Min area as ratio of image
    max_area_ratio: float  # Max area as ratio of image
    color: Tuple[int, int, int]  # BGR color for visualization


@dataclass
class PipelineConfig:
    """Overall pipeline configuration."""
    # Input/Output
    input_image: str = "data/raw/1.jpg"
    output_dir: str = "outputs/hierarchical"
    
    # SAM settings
    sam_checkpoint: str = "weight/sam_vit_h_4b8939.pth"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    
    # Validation rules
    min_aspect_ratio: float = 0.3  # w/h minimum
    max_aspect_ratio: float = 3.0  # w/h maximum
    max_overlap_iou: float = 0.5   # Max allowed overlap
    
    # Hierarchical levels
    levels: List[LevelConfig] = None
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = [
                LevelConfig("large", 0.10, 0.40, (0, 0, 255)),    # Red - Large
                LevelConfig("medium", 0.02, 0.10, (0, 255, 0)),   # Green - Medium
                LevelConfig("small", 0.005, 0.02, (255, 0, 0)),   # Blue - Small
            ]


# =============================================================================
# Validation Rules
# =============================================================================

def validate_mask(mask: np.ndarray, image_area: int, config: PipelineConfig) -> Dict[str, Any]:
    """
    Validate a mask against all rules.
    Returns dict with validation results and metadata.
    """
    area = mask.sum()
    area_ratio = area / image_area
    
    # Get bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return {"valid": False, "reason": "empty_mask"}
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    width = cmax - cmin
    height = rmax - rmin
    aspect_ratio = width / max(height, 1)
    
    # Validation checks
    valid = True
    reasons = []
    
    # Check aspect ratio
    if aspect_ratio < config.min_aspect_ratio:
        valid = False
        reasons.append(f"aspect_ratio_too_low:{aspect_ratio:.2f}")
    if aspect_ratio > config.max_aspect_ratio:
        valid = False
        reasons.append(f"aspect_ratio_too_high:{aspect_ratio:.2f}")
    
    # Determine level
    level = None
    for lvl in config.levels:
        if lvl.min_area_ratio <= area_ratio < lvl.max_area_ratio:
            level = lvl.name
            break
    
    if level is None:
        valid = False
        reasons.append(f"outside_size_range:{area_ratio:.4f}")
    
    return {
        "valid": valid,
        "reasons": reasons,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "bbox": [int(cmin), int(rmin), int(width), int(height)],  # x, y, w, h
        "aspect_ratio": float(aspect_ratio),
        "level": level
    }


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / max(union, 1)


def remove_overlapping_masks(masks: List[Dict], config: PipelineConfig) -> List[Dict]:
    """Remove masks that overlap too much, keeping higher confidence ones."""
    if len(masks) <= 1:
        return masks
    
    # Sort by predicted_iou (confidence)
    masks = sorted(masks, key=lambda x: x.get('predicted_iou', 0), reverse=True)
    
    keep = []
    for mask_data in masks:
        mask = mask_data['segmentation']
        
        # Check overlap with already kept masks
        overlaps = False
        for kept in keep:
            iou = compute_iou(mask, kept['segmentation'])
            if iou > config.max_overlap_iou:
                overlaps = True
                break
        
        if not overlaps:
            keep.append(mask_data)
    
    return keep


# =============================================================================
# Pipeline Steps
# =============================================================================

def step1_sam_auto_masks(image: np.ndarray, config: PipelineConfig) -> List[Dict]:
    """Step 1: Generate all masks using SAM Automatic Mask Generator."""
    print("\n[STEP 1] SAM Automatic Mask Generator...")
    
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    
    sam = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
    )
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    
    print(f"    Found {len(masks)} raw masks")
    
    del sam, mask_generator
    torch.cuda.empty_cache()
    
    return masks


def step2_validate_and_classify(
    masks: List[Dict], 
    image: np.ndarray, 
    config: PipelineConfig
) -> Dict[str, List[Dict]]:
    """Step 2: Validate masks and classify by hierarchical level."""
    print("\n[STEP 2] Validating and classifying masks...")
    
    h, w = image.shape[:2]
    image_area = h * w
    
    # Classify masks by level
    levels = {lvl.name: [] for lvl in config.levels}
    levels["invalid"] = []
    
    for mask_data in masks:
        mask = mask_data['segmentation']
        validation = validate_mask(mask, image_area, config)
        
        mask_data['validation'] = validation
        
        if validation['valid']:
            levels[validation['level']].append(mask_data)
        else:
            levels['invalid'].append(mask_data)
    
    # Remove overlapping masks within each level
    for lvl_name in [l.name for l in config.levels]:
        original_count = len(levels[lvl_name])
        levels[lvl_name] = remove_overlapping_masks(levels[lvl_name], config)
        print(f"    {lvl_name}: {original_count} -> {len(levels[lvl_name])} after overlap removal")
    
    print(f"    Invalid: {len(levels['invalid'])} masks filtered")
    
    return levels


def step3_visualize_levels(
    image: np.ndarray, 
    levels: Dict[str, List[Dict]], 
    config: PipelineConfig
) -> Dict[str, np.ndarray]:
    """Step 3: Create visualizations for each level."""
    print("\n[STEP 3] Creating visualizations...")
    
    visualizations = {}
    
    for lvl in config.levels:
        vis = image.copy()
        masks_in_level = levels.get(lvl.name, [])
        
        for i, mask_data in enumerate(masks_in_level):
            mask = mask_data['segmentation']
            color = lvl.color
            
            # Mask overlay
            vis[mask] = (vis[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            
            # Contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 2)
            
            # Label
            bbox = mask_data['validation']['bbox']
            cv2.putText(
                vis, f"#{i+1}", 
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        visualizations[lvl.name] = vis
        print(f"    {lvl.name}: {len(masks_in_level)} objects")
    
    # Combined visualization
    vis_combined = image.copy()
    for lvl in config.levels:
        for mask_data in levels.get(lvl.name, []):
            mask = mask_data['segmentation']
            color = lvl.color
            vis_combined[mask] = (vis_combined[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_combined, contours, -1, color, 2)
    
    visualizations['combined'] = vis_combined
    
    return visualizations


def step4_export_coco(
    image: np.ndarray,
    levels: Dict[str, List[Dict]], 
    config: PipelineConfig
) -> Dict:
    """Step 4: Export to COCO JSON format."""
    print("\n[STEP 4] Exporting to COCO JSON...")
    
    h, w = image.shape[:2]
    
    coco = {
        "info": {
            "description": "Stone Detection Dataset",
            "version": "1.0",
            "date_created": datetime.now().isoformat()
        },
        "images": [{
            "id": 1,
            "file_name": Path(config.input_image).name,
            "width": w,
            "height": h
        }],
        "categories": [
            {"id": 1, "name": "stone", "supercategory": "object"}
        ],
        "annotations": []
    }
    
    ann_id = 1
    for lvl in config.levels:
        for mask_data in levels.get(lvl.name, []):
            validation = mask_data['validation']
            bbox = validation['bbox']  # [x, y, w, h]
            
            # Get segmentation polygon
            mask = mask_data['segmentation'].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                if len(contour) >= 3:
                    segmentation.append(contour.flatten().tolist())
            
            if not segmentation:
                continue
            
            coco["annotations"].append({
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": validation['area'],
                "iscrowd": 0,
                "attributes": {
                    "level": lvl.name,
                    "predicted_iou": float(mask_data.get('predicted_iou', 0)),
                    "stability_score": float(mask_data.get('stability_score', 0))
                }
            })
            ann_id += 1
    
    print(f"    Exported {len(coco['annotations'])} annotations")
    return coco


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(config: PipelineConfig = None):
    """Run the complete hierarchical labeling pipeline."""
    if config is None:
        config = PipelineConfig()
    
    print("=" * 70)
    print("HIERARCHICAL MULTI-LEVEL AUTO-LABELING PIPELINE")
    print("=" * 70)
    print(f"Input: {config.input_image}")
    print(f"Output: {config.output_dir}")
    print(f"Levels: {[l.name for l in config.levels]}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(config.input_image)
    if image is None:
        print(f"ERROR: Could not load {config.input_image}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Step 1: SAM Auto Mask Generation
    masks = step1_sam_auto_masks(image, config)
    
    # Step 2: Validate and Classify
    levels = step2_validate_and_classify(masks, image, config)
    
    # Step 3: Visualize
    visualizations = step3_visualize_levels(image, levels, config)
    
    # Save visualizations
    for name, vis in visualizations.items():
        path = f"{config.output_dir}/{name}.jpg"
        cv2.imwrite(path, vis)
        print(f"    Saved: {path}")
    
    # Step 4: Export COCO
    coco = step4_export_coco(image, levels, config)
    coco_path = f"{config.output_dir}/annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"    Saved: {coco_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = sum(len(levels.get(l.name, [])) for l in config.levels)
    print(f"  Total valid detections: {total}")
    for lvl in config.levels:
        count = len(levels.get(lvl.name, []))
        print(f"    - {lvl.name}: {count}")
    print(f"  Invalid (filtered): {len(levels.get('invalid', []))}")
    print(f"\nOutputs saved to: {config.output_dir}/")
    print("=" * 70)
    
    return levels, coco


if __name__ == "__main__":
    run_pipeline()

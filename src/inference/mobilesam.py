#!/usr/bin/env python3
"""
MobileSAM + COCO Annotation Generator

Runs MobileSAM for automatic mask generation with improved segmentation quality.
Uses centralized configuration and utilities from src/utils.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Import centralized utilities
from src.utils import (
    # JSON & metadata
    save_json,
    save_dataset_metadata,
    # COCO utilities
    save_coco_annotations,
    create_coco_annotations_from_masks,
    create_coco_image_entry,
    # Mask & visualization utilities
    post_process_mask,
    create_mask_visualization,
    save_overlay,
    # Bbox utilities
    get_bbox_from_mask,
    filter_overlapping_bboxes,
    # Point selection
    smart_point_selection,
)

# Try loading config, use defaults if not available
try:
    from src.utils.settings import config
    input_dir = Path(config.get('paths.data', 'data')) / 'raw'
    output_dir = Path(config.get('paths.results', 'results')) / 'mobilesam_results'
    sam_checkpoint = Path(config.get('models.mobilesam.path', 'weight/mobile_sam.pt'))
except ImportError:
    input_dir = Path("./data/raw")
    output_dir = Path("./results/mobilesam_results")
    sam_checkpoint = Path("./weight/mobile_sam.pt")

# Default categories
DEFAULT_CATEGORIES = [
    {'id': 1, 'name': 'stone', 'supercategory': 'object'}
]

# Create output directories
images_dir = output_dir / "images"
masks_dir = output_dir / "masks"
samples_dir = output_dir / "samples"
metadata_dir = output_dir / "metadata"
for d in [images_dir, masks_dir, samples_dir, metadata_dir]:
    d.mkdir(parents=True, exist_ok=True)

# MobileSAM initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_t"](checkpoint=str(sam_checkpoint))
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)


# =============================================================================
# Local Helper Functions
# =============================================================================

def get_improved_bbox(mask, min_area=100):
    """
    Get improved bbox with fill ratio quality check.
    Uses centralized get_bbox_from_mask internally.
    """
    bbox = get_bbox_from_mask(mask, min_area)
    if bbox is None:
        return None, 0
    
    x, y, w, h = bbox
    bbox_area = w * h
    mask_area = np.sum(mask)
    
    # Fill ratio check - reject if mask doesn't fill bbox well
    fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
    if fill_ratio < 0.3:
        return None, 0
    
    return bbox, mask_area


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_image(img_path, output_name, img_id):
    """
    Process a single image with MobileSAM.
    
    Returns:
        Tuple of (metadata_dict, object_masks, object_bboxes, coco_image_entry)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return None, [], [], None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # Adaptive parameters based on image size
    total_image_area = height * width
    min_area = max(100, total_image_area * 0.0001)
    max_area = total_image_area * 0.8
    
    # Generate initial masks
    auto_masks = mask_generator.generate(img_rgb)
    
    # Filter masks by area and compactness
    cand_masks = []
    for m in auto_masks:
        area = m["area"]
        if min_area < area < max_area:
            try:
                contours, _ = cv2.findContours(
                    m["segmentation"].astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    if perimeter > 0:
                        compactness = area / (perimeter * perimeter)
                        if compactness > 0.01:
                            cand_masks.append(m)
                    else:
                        cand_masks.append(m)
                else:
                    cand_masks.append(m)
            except Exception:
                cand_masks.append(m)
    
    cand_masks.sort(key=lambda x: x["area"], reverse=True)
    cand_masks = cand_masks[:min(20, len(cand_masks))]

    # Initialize predictor
    predictor.set_image(img_rgb)

    # Combined mask for visualization
    fine_combined = np.zeros(img.shape[:2], bool)
    # Individual object masks for COCO
    object_masks = []
    object_bboxes = []

    for m in cand_masks:
        seg = m["segmentation"]
        
        # Smart point selection using centralized utility
        points = smart_point_selection(seg, num_points=12)
        if len(points) == 0:
            continue
        
        pts = points.astype(np.float32)
        labs = np.ones(len(pts), dtype=int)
        
        is_large = m["area"] > total_image_area * 0.1
        multimask = is_large or len(points) > 8
        
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labs,
            multimask_output=multimask
        )
        
        if multimask and len(masks) > 1:
            best_mask_idx = np.argmax(scores)
            mask_to_add = masks[best_mask_idx]
        else:
            mask_to_add = masks[0]
        
        # Post-process using centralized utility
        processed_mask = post_process_mask(mask_to_add, min_area=int(min_area))
        
        # Get bbox using local wrapper
        bbox, area = get_improved_bbox(processed_mask, min_area=min_area)
        
        if bbox is not None and area > min_area:
            fine_combined |= processed_mask
            object_masks.append(processed_mask.astype(np.uint8))
            object_bboxes.append(bbox)

    # Filter overlapping bboxes using centralized utility
    if len(object_bboxes) > 1:
        object_bboxes, object_masks = filter_overlapping_bboxes(
            object_bboxes, object_masks
        )

    # Create and save visualization mask using centralized utility
    mask = create_mask_visualization(fine_combined, kernel_size=5)
    cv2.imwrite(str(masks_dir / f"{output_name}_mask.png"), mask)
    cv2.imwrite(str(images_dir / f"{output_name}.png"), img)

    # Create overlay with bboxes
    overlay = img_rgb.copy()
    rng = np.random.default_rng(42)
    for m in cand_masks:
        seg = (m["segmentation"]).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(x) for x in rng.integers(0, 255, size=3))
        cv2.drawContours(overlay, cnts, -1, color, 2)
    
    # Draw bboxes on overlay
    for bbox in object_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    
    cv2.imwrite(str(samples_dir / f"{output_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save metadata using centralized utility
    metadata = {
        "image_path": f"images/{output_name}.png",
        "mask_path": f"masks/{output_name}_mask.png",
        "overlay_path": f"samples/{output_name}_overlay.png",
        "image_size": [int(x) for x in img.shape[:2]],
        "mask_area": int(np.sum(mask > 0)),
        "num_segments": len(object_masks),
        "bboxes": object_bboxes
    }
    save_json(metadata, metadata_dir / f"{output_name}.json")

    # Create COCO image entry using centralized utility
    coco_image = create_coco_image_entry(img_id, width, height, f"{output_name}.png")

    return metadata, object_masks, object_bboxes, coco_image


def main():
    """Main entry point."""
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # COCO storage
    coco_images = []
    coco_annotations = []
    ann_id = 1
    
    results = []
    for img_id, img_path in enumerate(image_files, 1):
        print(f"Processing {img_path.name}...")
        output_name = img_path.stem
        
        metadata, object_masks, object_bboxes, coco_image = process_image(
            img_path, output_name, img_id
        )
        
        if metadata:
            results.append(metadata)
            
            # Add COCO image entry
            if coco_image:
                coco_images.append(coco_image)
            
            # Create COCO annotations from masks using centralized utility
            new_annotations, ann_id = create_coco_annotations_from_masks(
                object_masks, ann_id, img_id, 
                category_id=1, 
                bboxes=object_bboxes
            )
            coco_annotations.extend(new_annotations)
    
    # Save dataset metadata using centralized utility
    save_dataset_metadata(results, output_dir / "dataset_metadata.json")
    print(f"\nProcessed {len(results)} images. Results in {output_dir}/")
    
    # Save COCO annotations using centralized utility
    save_coco_annotations(
        coco_images, coco_annotations, DEFAULT_CATEGORIES, 
        output_dir / "annotations_coco.json"
    )
    print(f"COCO annotations saved to {output_dir / 'annotations_coco.json'}")


if __name__ == "__main__":
    main()

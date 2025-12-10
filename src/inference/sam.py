#!/usr/bin/env python3
"""
SAM Dataset Generation Script

Generates segmentation datasets using MobileSAM with COCO annotations.
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
    # COCO utilities
    load_coco_annotations,
    save_coco_annotations,
    load_or_create_coco_dataset,
    save_dataset_metadata,
    create_coco_annotations_from_masks,
    create_coco_image_entry,
    build_mask_from_coco,
    draw_coco_overlay,
    # Mask & visualization utilities
    create_mask_visualization,
    save_overlay,
    save_json,
    calculate_mask_iou,
)

# Try loading config, use defaults if not available
try:
    from src.utils.settings import config
    input_dir = Path(config.get('paths.results', 'results')) / 'mobilesam_results' / 'samples'
    output_dir = Path(config.get('paths.results', 'results')) / 'dataset_results'
    sam_checkpoint = Path(config.get('models.mobilesam.path', 'weight/mobile_sam.pt'))
except ImportError:
    input_dir = Path("./results/mobilesam_results/samples/")
    output_dir = Path("./results/dataset_results")
    sam_checkpoint = Path("./weight/mobile_sam.pt")


# Default categories for stone detection
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
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.6,
    min_mask_region_area=100
)
predictor = SamPredictor(sam)


def process_image(img_path, output_name, img_id, existing_coco=None, min_area=100):
    """
    Process a single image with MobileSAM.
    
    Args:
        img_path: Path to input image
        output_name: Base name for output files
        img_id: Image ID for COCO annotation
        existing_coco: Existing COCO dataset to extend
        min_area: Minimum mask area threshold
        
    Returns:
        Tuple of (metadata_dict, list of COCO annotations, COCO image entry)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return None, [], None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # Area thresholds
    total_image_area = height * width
    min_area = max(min_area, total_image_area * 0.00005)
    max_area = total_image_area * 0.9
    min_hole_area = 300

    # Get existing mask from COCO if available
    existing_mask = None
    if existing_coco is not None:
        existing_mask = build_mask_from_coco(existing_coco, img_id, (height, width))

    # Initialize combined mask and object masks
    fine_combined = existing_mask.astype(bool) if existing_mask is not None else np.zeros((height, width), bool)
    object_masks = []

    if existing_mask is not None:
        # Process only holes in existing mask
        holes_mask = (existing_mask == 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_hole_area:
                continue
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                         stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            roi = (slice(y, y+h), slice(x, x+w))
            img_crop = img_rgb[roi]
            auto_masks = mask_generator.generate(img_crop)
            for m in auto_masks:
                seg = m["segmentation"]
                seg_full = np.zeros((height, width), dtype=np.uint8)
                seg_full[roi] = seg
                fine_combined |= seg_full.astype(bool)
                object_masks.append(seg_full)
    else:
        # First pass - process entire image
        auto_masks = mask_generator.generate(img_rgb)
        cand_masks = [m for m in auto_masks if min_area < m["area"] < max_area]
        cand_masks.sort(key=lambda x: x["area"], reverse=True)
        predictor.set_image(img_rgb)
        
        for m in cand_masks:
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            is_large = m["area"] > total_image_area * 0.1
            num_perimeter_points = 6 if is_large else 3
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            points = [(cx, cy)]
            cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(cnts) > 0:
                contour = cnts[0][:, 0, :]
                idxs = np.linspace(0, len(contour)-1, num_perimeter_points, dtype=int)
                for idx in idxs:
                    x, y = contour[idx]
                    points.append((x, y))
            pts = np.array(points)
            labs = np.ones(len(points), dtype=int)
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labs,
                multimask_output=True if is_large else False
            )
            if is_large and len(masks) > 1:
                best_mask_idx = np.argmax(scores)
                mask_to_add = masks[best_mask_idx]
            else:
                mask_to_add = masks[0]
            fine_combined |= mask_to_add.astype(bool)
            object_masks.append(mask_to_add.astype(np.uint8))

    # Create and save visualization mask using centralized utility
    mask = create_mask_visualization(fine_combined, kernel_size=7, morph_operation="close")
    cv2.imwrite(str(masks_dir / f"{output_name}_mask.png"), mask)
    cv2.imwrite(str(images_dir / f"{output_name}.png"), img)

    # Save overlay using centralized utility
    save_overlay(img_rgb, object_masks, samples_dir / f"{output_name}_overlay.png", is_rgb=True)

    # Prepare and save metadata
    metadata = {
        "image_path": f"images/{output_name}.png",
        "mask_path": f"masks/{output_name}_mask.png",
        "overlay_path": f"samples/{output_name}_overlay.png",
        "image_size": [int(x) for x in img.shape[:2]],
        "mask_area": int(np.sum(mask > 0)),
        "num_segments": len(object_masks)
    }
    save_json(metadata, metadata_dir / f"{output_name}.json")

    # Create COCO image entry using centralized utility
    coco_image = create_coco_image_entry(img_id, width, height, f"{output_name}.png")

    return metadata, object_masks, coco_image


def main():
    """Main entry point."""
    global ann_id
    
    # Load or create COCO dataset using centralized utility
    coco_path = output_dir / "annotations_coco.json"
    coco_images, coco_annotations, categories, ann_id, file2id = load_or_create_coco_dataset(
        coco_path, DEFAULT_CATEGORIES
    )
    
    # Get existing COCO for hole filling
    existing_coco = None
    if coco_path.exists():
        existing_coco = load_coco_annotations(coco_path)

    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    results = []
    
    for img_path in image_files:
        output_name = img_path.stem
        img_id = file2id.get(output_name, len(coco_images) + 1)
        print(f"Processing {img_path.name}...")
        
        metadata, object_masks, coco_image = process_image(
            img_path, output_name, img_id, existing_coco
        )
        
        if metadata:
            results.append(metadata)
            
            # Add COCO image entry
            if coco_image and not any(img['id'] == img_id for img in coco_images):
                coco_images.append(coco_image)
            
            # Create COCO annotations from masks using centralized utility
            new_annotations, ann_id = create_coco_annotations_from_masks(
                object_masks, ann_id, img_id, category_id=1, min_area=100
            )
            coco_annotations.extend(new_annotations)
    
    # Save dataset metadata using centralized utility
    save_dataset_metadata(results, output_dir / "dataset_metadata.json")
    print(f"\nProcessed {len(results)} images. Results in {output_dir}/")
    
    # Save COCO annotations using centralized utility
    save_coco_annotations(coco_images, coco_annotations, categories, coco_path)
    print(f"COCO annotations saved to {coco_path}")

    # Draw merged overlays using centralized utility
    coco_json = load_coco_annotations(coco_path)
    draw_coco_overlay(coco_json, images_dir, output_dir / "merged_overlays")


if __name__ == "__main__":
    main()
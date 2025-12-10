#!/usr/bin/env python3
"""
Combined YOLO + MobileSAM Inference

Two-stage inference combining YOLO detection with MobileSAM segmentation.
Uses the individual yolo.py and mobilesam.py scripts and centralized utilities.

Stage 1: YOLO for object detection
Stage 2: MobileSAM for detailed segmentation within detected regions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import from individual inference scripts
from src.inference.yolo import load_yolo_model, detect_objects

# Import centralized utilities
from src.utils import (
    save_json,
    save_dataset_metadata,
    save_coco_annotations,
    create_coco_annotations_from_masks,
    create_coco_image_entry,
    post_process_mask,
    draw_detections_on_image,
    calculate_bbox_iou,
)

# Try loading config
try:
    from src.utils.settings import config
    input_dir = Path(config.get('paths.data', 'data')) / 'raw'
    output_dir = Path(config.get('paths.results', 'results')) / 'yolo_sam_results'
    yolo_checkpoint = Path(config.get('paths.weights', 'weights')) / 'best.pt'
    sam_checkpoint = Path(config.get('paths.weights', 'weights')) / 'mobile_sam.pt'
except ImportError:
    input_dir = Path("./data/raw")
    output_dir = Path("./results/yolo_sam_results")
    yolo_checkpoint = Path("./weight/best.pt")
    sam_checkpoint = Path("./weight/mobile_sam.pt")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    {'id': 1, 'name': 'stone', 'supercategory': 'object'}
]


# =============================================================================
# Model Loading
# =============================================================================

def load_sam_model(checkpoint_path: str = None, device: str = None):
    """Load MobileSAM model.
    
    Args:
        checkpoint_path: Path to SAM weights
        device: Device to use
        
    Returns:
        Tuple of (sam_model, predictor)
    """
    from mobile_sam import sam_model_registry, SamPredictor
    
    checkpoint = checkpoint_path or str(sam_checkpoint)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
    sam.to(device)
    sam.eval()
    
    predictor = SamPredictor(sam)
    
    logger.info(f"MobileSAM loaded from: {checkpoint}")
    return sam, predictor


# =============================================================================
# Combined Processing Functions
# =============================================================================

def refine_detection_with_sam(
    predictor,
    image: np.ndarray,
    detection: Dict,
    padding: int = 5,
    min_area: int = 100,
) -> Tuple[Optional[np.ndarray], float]:
    """Refine a YOLO detection using MobileSAM.
    
    Args:
        predictor: SAM predictor (already set with image)
        image: Original image
        detection: YOLO detection dictionary
        padding: Padding around bbox for SAM
        min_area: Minimum mask area
        
    Returns:
        Tuple of (refined_mask, confidence_score)
    """
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add padding
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)
    
    input_box = np.array([x1_pad, y1_pad, x2_pad, y2_pad])
    
    try:
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])
        
        # Post-process
        processed = post_process_mask(best_mask, min_area=min_area)
        
        if np.sum(processed) >= min_area:
            return processed, best_score
        else:
            return None, 0.0
            
    except Exception as e:
        logger.warning(f"SAM refinement failed: {e}")
        return None, 0.0


def process_image(
    yolo_model,
    sam_predictor,
    img_path: Path,
    output_name: str,
    img_id: int,
    conf_threshold: float = 0.25,
) -> Tuple[Optional[Dict], List[np.ndarray], Dict]:
    """Process image with YOLO + MobileSAM.
    
    Args:
        yolo_model: Loaded YOLO model
        sam_predictor: SAM predictor
        img_path: Input image path
        output_name: Output name
        img_id: Image ID
        conf_threshold: YOLO confidence threshold
        
    Returns:
        Tuple of (metadata, masks, coco_image_entry)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Could not read: {img_path}")
        return None, [], None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Stage 1: YOLO detection
    logger.info(f"Stage 1: YOLO detection on {img_path.name}")
    detections = detect_objects(yolo_model, img, conf_threshold=conf_threshold)
    logger.info(f"  Found {len(detections)} detections")
    
    if not detections:
        return None, [], None
    
    # Stage 2: SAM refinement
    logger.info(f"Stage 2: SAM refinement")
    sam_predictor.set_image(img_rgb)
    
    refined_detections = []
    masks = []
    
    for i, detection in enumerate(detections):
        mask, score = refine_detection_with_sam(sam_predictor, img, detection)
        
        if mask is not None:
            refined = detection.copy()
            refined['sam_score'] = score
            refined['mask_area'] = int(np.sum(mask))
            refined['stage'] = 'yolo+sam'
            refined_detections.append(refined)
            masks.append(mask)
            logger.debug(f"  Detection {i+1}: refined with SAM score {score:.3f}")
    
    logger.info(f"  Refined {len(refined_detections)} detections")
    
    # Create visualization
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Draw detections and masks
    vis_image = img.copy()
    for mask, det in zip(masks, refined_detections):
        # Draw mask overlay
        color = (0, 255, 255)  # Yellow
        vis_image[mask.astype(bool)] = (
            vis_image[mask.astype(bool)] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)
    
    vis_image = draw_detections_on_image(vis_image, refined_detections)
    cv2.imwrite(str(vis_dir / f"{output_name}_combined.jpg"), vis_image)
    
    # Save image
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{output_name}.jpg"), img)
    
    # Metadata
    metadata = {
        'image_path': f"images/{output_name}.jpg",
        'image_size': [height, width],
        'yolo_detections': len(detections),
        'refined_detections': len(refined_detections),
        'detections': refined_detections,
    }
    
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    save_json(metadata, metadata_dir / f"{output_name}.json")
    
    coco_image = create_coco_image_entry(img_id, width, height, f"{output_name}.jpg")
    
    return metadata, masks, coco_image


def main():
    """Main entry point."""
    # Create output directories
    for subdir in ["images", "visualizations", "metadata", "masks"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    yolo_model = load_yolo_model()
    sam_model, sam_predictor = load_sam_model()
    
    # Find images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        logger.error(f"No images in {input_dir}")
        return
    
    logger.info(f"Processing {len(image_files)} images")
    
    # Storage
    coco_images = []
    coco_annotations = []
    ann_id = 1
    results = []
    
    for img_id, img_path in enumerate(image_files, 1):
        logger.info(f"[{img_id}/{len(image_files)}] {img_path.name}")
        
        metadata, masks, coco_image = process_image(
            yolo_model, sam_predictor,
            img_path, img_path.stem, img_id
        )
        
        if metadata:
            results.append(metadata)
            
            if coco_image:
                coco_images.append(coco_image)
            
            # COCO annotations from masks
            new_anns, ann_id = create_coco_annotations_from_masks(
                masks, ann_id, img_id, category_id=1
            )
            coco_annotations.extend(new_anns)
    
    # Save results
    save_dataset_metadata(results, output_dir / "dataset_metadata.json")
    save_coco_annotations(
        coco_images, coco_annotations, DEFAULT_CATEGORIES,
        output_dir / "annotations_coco.json"
    )
    
    logger.info(f"Done! Results: {output_dir}")


if __name__ == "__main__":
    main()

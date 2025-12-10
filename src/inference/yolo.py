#!/usr/bin/env python3
"""
YOLO Object Detection Inference

Runs YOLO for object detection with COCO-format annotation output.
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
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import centralized utilities
from src.utils import (
    # JSON & metadata
    save_json,
    save_dataset_metadata,
    NumpyEncoder,
    # COCO utilities
    save_coco_annotations,
    create_coco_image_entry,
    # Visualization
    draw_detections_on_image,
)

# Try loading config, use defaults if not available
try:
    from src.utils.settings import config
    input_dir = Path(config.get('paths.data', 'data')) / 'raw'
    output_dir = Path(config.get('paths.results', 'results')) / 'yolo_results'
    yolo_checkpoint = Path(config.get('paths.weights', 'weights')) / 'best.pt'
except ImportError:
    input_dir = Path("./data/raw")
    output_dir = Path("./results/yolo_results")
    yolo_checkpoint = Path("./weight/best.pt")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default categories
DEFAULT_CATEGORIES = [
    {'id': 1, 'name': 'stone', 'supercategory': 'object'}
]


# =============================================================================
# YOLO Model Initialization
# =============================================================================

def load_yolo_model(checkpoint_path: str = None, device: str = None):
    """Load YOLO model.
    
    Args:
        checkpoint_path: Path to YOLO weights
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Loaded YOLO model
    """
    from ultralytics import YOLO
    
    checkpoint = checkpoint_path or str(yolo_checkpoint)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLO(checkpoint)
    model.to(device)
    
    logger.info(f"YOLO model loaded from: {checkpoint}")
    logger.info(f"Using device: {device}")
    
    return model


# =============================================================================
# Detection Functions
# =============================================================================

def detect_objects(
    model,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Run YOLO detection on an image.
    
    Args:
        model: YOLO model
        image: Input image (BGR format)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections
        class_names: Optional mapping of class IDs to names
        
    Returns:
        List of detection dictionaries with bbox, confidence, class info
    """
    results = model(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False
    )
    
    detections = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        # Get class names from model if not provided
        model_names = getattr(results[0], 'names', {})
        
        for i in range(len(boxes)):
            class_id = int(class_ids[i])
            
            # Get class name
            if class_names and class_id in class_names:
                class_name = class_names[class_id]
            elif class_id in model_names:
                class_name = model_names[class_id]
            else:
                class_name = 'stone'
            
            detection = {
                'bbox': [float(boxes[i][0]), float(boxes[i][1]), 
                        float(boxes[i][2]), float(boxes[i][3])],
                'bbox_xywh': [
                    float(boxes[i][0]),
                    float(boxes[i][1]),
                    float(boxes[i][2] - boxes[i][0]),
                    float(boxes[i][3] - boxes[i][1])
                ],
                'confidence': float(confidences[i]),
                'class_id': class_id,
                'class_name': class_name,
                'stage': 'yolo',
            }
            detections.append(detection)
    
    return detections


def create_coco_annotation_from_detection(
    detection: Dict,
    ann_id: int,
    image_id: int,
    category_id: int = 1,
) -> Dict:
    """Create COCO annotation entry from a detection.
    
    Args:
        detection: Detection dictionary with bbox
        ann_id: Annotation ID
        image_id: Image ID
        category_id: Category ID
        
    Returns:
        COCO annotation dictionary
    """
    bbox = detection['bbox_xywh']
    area = bbox[2] * bbox[3]
    
    return {
        'id': ann_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'area': float(area),
        'iscrowd': 0,
        'score': detection['confidence'],
    }


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_image(
    model,
    img_path: Path,
    output_name: str,
    img_id: int,
    conf_threshold: float = 0.25,
    save_visualization: bool = True,
) -> Tuple[Optional[Dict], List[Dict], Dict]:
    """Process a single image with YOLO.
    
    Args:
        model: YOLO model
        img_path: Path to input image
        output_name: Output filename (without extension)
        img_id: Image ID for COCO annotations
        conf_threshold: Detection confidence threshold
        save_visualization: Whether to save visualization
        
    Returns:
        Tuple of (metadata, detections, coco_image_entry)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Could not read image: {img_path}")
        return None, [], None
    
    height, width = img.shape[:2]
    
    # Run detection
    detections = detect_objects(
        model, img, 
        conf_threshold=conf_threshold
    )
    
    logger.info(f"Detected {len(detections)} objects in {img_path.name}")
    
    # Save visualization
    if save_visualization and detections:
        vis_image = draw_detections_on_image(img, detections)
        vis_path = output_dir / "visualizations" / f"{output_name}_detections.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_path), vis_image)
    
    # Save image copy
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{output_name}.jpg"), img)
    
    # Create metadata
    metadata = {
        'image_path': f"images/{output_name}.jpg",
        'image_size': [height, width],
        'num_detections': len(detections),
        'detections': detections,
        'confidence_threshold': conf_threshold,
    }
    
    # Save metadata
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    save_json(metadata, metadata_dir / f"{output_name}.json")
    
    # Create COCO image entry
    coco_image = create_coco_image_entry(img_id, width, height, f"{output_name}.jpg")
    
    return metadata, detections, coco_image


def main():
    """Main entry point."""
    # Create output directories
    for subdir in ["images", "visualizations", "metadata"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_yolo_model()
    
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # COCO storage
    coco_images = []
    coco_annotations = []
    ann_id = 1
    
    results = []
    for img_id, img_path in enumerate(image_files, 1):
        logger.info(f"Processing [{img_id}/{len(image_files)}]: {img_path.name}")
        output_name = img_path.stem
        
        metadata, detections, coco_image = process_image(
            model, img_path, output_name, img_id
        )
        
        if metadata:
            results.append(metadata)
            
            if coco_image:
                coco_images.append(coco_image)
            
            # Create COCO annotations
            for detection in detections:
                annotation = create_coco_annotation_from_detection(
                    detection, ann_id, img_id, category_id=1
                )
                coco_annotations.append(annotation)
                ann_id += 1
    
    # Save dataset metadata
    save_dataset_metadata(results, output_dir / "dataset_metadata.json")
    logger.info(f"Processed {len(results)} images")
    
    # Save COCO annotations
    save_coco_annotations(
        coco_images, coco_annotations, DEFAULT_CATEGORIES,
        output_dir / "annotations_coco.json"
    )
    logger.info(f"COCO annotations saved to {output_dir / 'annotations_coco.json'}")
    
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
